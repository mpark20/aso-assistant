"""
LLM utilities for the ASO workflow:
1. core call_llm
2. tool orchestration
3. specific tool implementations
"""
import asyncio
import json
import re
import uuid
import threading
import time
import litellm
from litellm.types.utils import Function, ChatCompletionMessageToolCall
import httpx
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

from aso_workflow.utils.apis import browse_webpage, search_omim
from aso_workflow.utils.pubmed import fetch_pubmed, fetch_pmc_fulltext, url_to_pmid, url_to_pmcid
from aso_workflow.prompts import SYSTEM_PROMPTS


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite-preview"
HELPER_MODEL =  "gemini/gemini-3.1-flash-lite-preview" # "gemini/gemma-3-27b-it"
MAX_RAW_CONTENT_CHARS = 80_000
MAX_TOOL_CALLS = 6


RATE_LIMITS = {
    "gemini/gemini-2.5-flash": {"tpm": 200_000, "rpm": 5},
    "gemini/gemini-3-flash-preview": {"tpm": 200_000, "rpm": 5},
    "gemini/gemini-2.5-flash-lite": {"tpm": 200_000, "rpm": 10},
    "gemini/gemini-3.1-flash-lite-preview": {"tpm": 200_000, "rpm": 10},
    "gemini/gemma-3-27b-it": {"tpm": 12_000, "rpm": 30},
}

# ── Rate Limit Handling ──────────────────────────────────────────────────────────
_rate_limit_state: dict[str, dict] = {}
_rate_limit_lock = threading.Lock()


def _wait_for_rate_limit(model: str) -> None:
    """
    Block until the next request would be within rpm and tpm limits for the model.
    Uses a sliding 1-minute window. Reserves a request slot before returning.
    """
    limits = RATE_LIMITS.get(model)
    if not limits:
        return

    tpm = limits["tpm"]
    rpm = limits["rpm"]

    while True:
        with _rate_limit_lock:
            if model not in _rate_limit_state:
                _rate_limit_state[model] = {"request_times": [], "token_records": []}
            state = _rate_limit_state[model]

            now = time.time()
            cutoff = now - 60.0  # 1 minute ago
            state["request_times"] = [t for t in state["request_times"] if t > cutoff]
            state["token_records"] = [(t, n) for t, n in state["token_records"] if t > cutoff]

            tokens_in_window = sum(n for t, n in state["token_records"] if t > cutoff)
            wait_time = 0.0

            # Check rpm: need fewer than rpm requests in the window
            if len(state["request_times"]) >= rpm:
                oldest = min(state["request_times"])
                wait_time = max(0.0, (oldest + 60.0) - time.time())
            # Check tpm: need tokens_in_window < tpm
            elif tokens_in_window >= tpm:
                if not state["token_records"]:
                    state["request_times"].append(time.time())
                    return
                oldest_ts = min(t for t, _ in state["token_records"])
                wait_time = max(0.0, (oldest_ts + 60.0) - time.time())
            else:
                # Reserve our request slot
                state["request_times"].append(time.time())
                return

        if wait_time > 0:
            time.sleep(wait_time)


def _record_usage(model: str, tokens: int) -> None:
    """Record token usage for a completed request (request slot was reserved in _wait_for_rate_limit)."""
    limits = RATE_LIMITS.get(model)
    if not limits:
        return
    with _rate_limit_lock:
        if model not in _rate_limit_state:
            _rate_limit_state[model] = {"request_times": [], "token_records": []}
        _rate_limit_state[model]["token_records"].append((time.time(), tokens))


def _accumulate_usage(usage_accumulator: dict, model: str, response: litellm.ModelResponse) -> None:
    """Add usage from a single response to the per-model accumulator."""
    # parse usage tokens
    usage = response.usage
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None) or 0
    output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None) or 0
    total_tokens = getattr(usage, "total_tokens", None) or (input_tokens + output_tokens) or 0
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}

    # record
    if model not in usage_accumulator:
        usage_accumulator[model] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_accumulator[model]["input_tokens"] += usage["input_tokens"]
    usage_accumulator[model]["output_tokens"] += usage["output_tokens"]
    usage_accumulator[model]["total_tokens"] += usage["total_tokens"]


# ── Core LLM call ────────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16000,
    expect_json: bool = True,
    max_retries: int = 3,
    tools: list | None = None,
    tool_cache: dict | None = None,
    max_tool_calls: int = 10,
    usage_accumulator: dict | None = None,
    **kwargs,
) -> tuple[dict | str, dict[str, dict[str, int]]]:
    """
    Call an LLM with optional tool use.

    If `tools` is provided, runs a tool-use loop:
      1. Send messages to model
      2. If model calls a tool, execute it and append result
      3. Repeat until model produces a text response or max_tool_calls reached

    Tool call audit logs are injected into the returned dict under "_tool_call_log".

    Returns:
        (result, token_usage) where token_usage is {model: {"input_tokens", "output_tokens", "total_tokens}}.
        Nested calls (e.g. fetch_and_extract) add to usage_accumulator when passed via tool_cache.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    usage_acc = usage_accumulator if usage_accumulator is not None else {}
    cache = tool_cache if tool_cache is not None else {}
    cache["_usage_accumulator"] = usage_acc
    tool_call_logs = []
    tool_calls_made = 0

    call_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs,
    }
    if model is None:
        call_kwargs["model"] = DEFAULT_MODEL
    if tools:
        call_kwargs["tools"] = tools
    if not _is_commercial_api_model(call_kwargs["model"]):
        call_kwargs["api_base"] = "http://localhost:30009/v1"

    response = _completion_with_retry(max_retries=max_retries, **call_kwargs)
    _accumulate_usage(usage_acc, model, response)
    msg = response.choices[0].message

    if "<tool_call>" in msg.content and msg.tool_calls is None:
        tool_call = _custom_parse_tool_call(msg.content)
        msg.tool_calls = [tool_call]

    # Tool-use loop
    while (tools and hasattr(msg, "tool_calls") and msg.tool_calls and tool_calls_made < max_tool_calls):
        # Add the assistant's tool call to the conversation history
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute the tool calls, adding results to the conversation history
        tool_results = []
        for tc in msg.tool_calls:
            tool_calls_made += 1
            try:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

            result_str = execute_tool(tc.function.name, args, cache)
            tool_call_logs.append({
                "tool": tc.function.name,
                "args": args,
                "result": result_str,
            })
            tool_results.append({"role": "tool", "content": result_str})

            if tool_calls_made >= max_tool_calls:
                tool_results.append({
                    "role": "user",
                    "content": (
                        f"[System: tool call limit ({max_tool_calls}) reached. "
                        "Please produce your final JSON response now using the "
                        "information gathered so far.]"
                    ),
                })
                break

        messages.extend(tool_results)
        # Call the LLM with the updated conversation history (each call retried independently)
        call_kwargs["messages"] = messages
        response = _completion_with_retry(max_retries=max_retries, **call_kwargs)
        _accumulate_usage(usage_acc, model, response)
        msg = response.choices[0].message

    res_text = msg.content or ""

    if not expect_json:
        return (res_text, usage_acc)

    try:
        parsed = _text_to_json(res_text)
        if tool_call_logs:
            parsed["_tool_call_log"] = tool_call_logs
        return (parsed, usage_acc)
    except json.JSONDecodeError as e:
        return ({
            "_raw": res_text,
            "_parse_error": str(e),
            "_tool_call_log": tool_call_logs,
        }, usage_acc)


def _completion_with_retry(max_retries: int, **call_kwargs) -> litellm.ModelResponse:
    """
    Call litellm.completion with retries for API errors only.
    Retries on RateLimitError, retriable HTTP errors, and ServiceUnavailableError.
    Enforces RATE_LIMITS (rpm, tpm) per model before each request.
    """
    model = call_kwargs.get("model", DEFAULT_MODEL)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            _wait_for_rate_limit(model)
            response = litellm.completion(**call_kwargs)
            tokens = getattr(response.usage, "total_tokens", None) or 0
            _record_usage(model, tokens)
            return response
        except (litellm.exceptions.RateLimitError, httpx.HTTPStatusError, litellm.ServiceUnavailableError) as e:
            last_error = _handle_llm_error(e, attempt, max_retries)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts. Last error: {last_error}")


def _text_to_json(text: str) -> dict:
    """Strip markdown code fences and parse JSON."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text)


def _is_retriable_http_error(status: int) -> bool:
    """Check if HTTP status indicates a retriable error."""
    return status in (429, 500, 502, 503, 504)


def _handle_llm_error(e: Exception, attempt: int, max_retries: int) -> str:
    """
    Handle retriable LLM error: sleep, optionally print, return error message.
    Re-raises for non-retriable errors.
    """
    if isinstance(e, litellm.exceptions.RateLimitError):
        time.sleep(2 ** attempt)
        return f"Rate limit: {e}"

    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if not _is_retriable_http_error(status):
            raise
        wait = 2 ** attempt
        print(f"LLM HTTP {status}. Retrying in {wait}s... ({attempt + 1}/{max_retries})")
        time.sleep(wait)
        return f"HTTP {status}: {e}"

    if isinstance(e, litellm.ServiceUnavailableError):
        status = e.response.status_code if e.response is not None else "unknown"
        wait = 2 ** attempt
        print(f"LLM HTTP {status}. Retrying in {wait}s... ({attempt + 1}/{max_retries})")
        time.sleep(wait)
        return f"HTTP {status}: {e}"

    raise

def _is_commercial_api_model(model_name: str) -> bool:
    """Detect if this is a commercial API model vs a self-hosted model

    Commercial API models (OpenAI, Claude, etc.) use chat completion APIs and don't need tokenizers.
    Self-hosted models (vLLM) use text completion APIs and benefit from tokenizers.

    From https://github.com/rlresearch/dr-tulu
    """
    # OpenAI model patterns
    openai_patterns = [
        "gpt-",
        "o1-",
        "text-",
        "davinci",
        "curie",
        "babbage",
        "ada",
        "chatgpt",
        "gpt4",
        "turbo",
    ]

    # Anthropic Claude patterns
    claude_patterns = ["claude-", "sonnet", "haiku", "opus"]

    # Other commercial API patterns
    commercial_patterns = [
        "gemini",
        "palm",  # Google
        "command",
        "coral",  # Cohere
    ]

    all_commercial_patterns = (
        openai_patterns + claude_patterns + commercial_patterns
    )
    model_lower = model_name.lower()

    return any(pattern in model_lower for pattern in all_commercial_patterns)



def _custom_parse_tool_call(msg: str) -> ChatCompletionMessageToolCall:
    """Parse a tool call from a message."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", msg, re.DOTALL)
    if match:
        try:
            tool_call = json.loads(match.group(1))
            function = Function(
                name=tool_call["name"],
                arguments=json.dumps(tool_call["arguments"]),
            )
            return ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                function=function,
            )
        except json.JSONDecodeError:
            return None
    return None



# ── Tool Execution ───────────────────────────────────────────────────────────
def execute_tool(tool_name: str, tool_args: dict, cache: dict) -> str:
    """Dispatch a tool call from the reasoning model."""
    if tool_name == "fetch_and_extract":
        url = tool_args.get("url", "")
        question = tool_args.get("question", "")
        result = fetch_and_extract(url, question, cache)
        return result
    return f"error: Unknown tool: {tool_name}"


# ── LLM Tools ──────────────────────────────────────────────────────

FETCH_AND_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_and_extract",
        "description": (
            "Fetch a webpage or paper and extract information relevant to a specific question. "
            "Use this when a search result preview (title/URL) looks relevant but you need "
            "the actual content to answer your question. Returns a compact evidence extract, "
            "not the full text. Call this only for sources likely to contain the specific "
            "evidence you need — prefer PubMed/PMC/OMIM URLs for mechanistic evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "URL to fetch. For PubMed and PMC article IDs, use the URL format "
                        "https://pubmed.ncbi.nlm.nih.gov/{pmid}/ or https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"
                        "For OMIM numbers, use the URL format https://www.omim.org/entry/{omim_number}/"
                    ),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "Specific question this source should answer. E.g.: 'Does this paper provide "
                        "experimental evidence for haploinsufficiency of BRCA2?' or "
                        "'What splice effect was observed for variant c.123C>T in gene BRCA1 in patient RNA?'"
                    ),
                },
            },
            "required": ["url", "question"],
        },
    },
}


def fetch_and_extract(url: str, question: str, cache: dict) -> str:
    """
    Execute a fetch_and_extract tool call.

    Returns: tool result string
    """
    cache_key = f"{url}::{question}"

    if cache_key in cache:
        cached = cache[cache_key]
        return _json_to_text(cached["extract_dict"])

    try:
        content = None

        # first, try using special PubMed/PMC parsing
        if pmid := url_to_pmid(url):
            results = fetch_pubmed([pmid])
            if results:
                content = json.dumps(results[0], indent=2)
        
        elif pmcid := url_to_pmcid(url):
            results = fetch_pmc_fulltext([pmcid], ignore_sections=["ref", "comp_int", "auth_cont"])
            if pmcid in results:
                content = json.dumps(results[pmcid], indent=2)
        
        elif mim_number := url_to_mim(url):
            results = search_omim(mim_number)
            if results:
                content = json.dumps(results, indent=2)
        
        # otherwise, use raw webpage browsing
        if content is None:
            resp = asyncio.run(browse_webpage(url))
            content = json.dumps(resp, indent=2)
        
        # truncate the content to the maximum allowed length, but record the original
        raw_chars = len(content)
        content = content[:MAX_RAW_CONTENT_CHARS]
    
    except Exception as e:
        error_msg = f"Fetch failed: {type(e).__name__}: {e}"
        return f"error: {error_msg}"


    try:
        user_msg = f"QUESTION: {question}\n RAW CONTENT:\n{content}"
        usage_acc = cache.get("_usage_accumulator")
        response, _ = call_llm(
            SYSTEM_PROMPTS["evidence_extraction"],
            user_msg,
            model=HELPER_MODEL,
            expect_json=True,
            max_tokens=32000,
            usage_accumulator=usage_acc,
        )

        if "_parse_error" in response:
            raw = response["_raw"]
            extract_dict = {
                "url": url,
                "question": question,
                "answers_question": False,
                "key_finding": raw[:300],
                "evidence_type": "unclear",
                "confidence": "low",
                "_parse_error": response["_parse_error"],
            }
        else:
            extract_dict = {"url": url, "question": question, **response}
    
    except Exception as e:
        error_msg = f"Extraction failed: {type(e).__name__}: {e}"
        return f"error: {error_msg}"

    extract_str = _json_to_text(extract_dict)
    cache[cache_key] = {
        "extract": extract_str,
        "extract_dict": extract_dict,
        "raw_chars": raw_chars,
    }
    return extract_str


def _json_to_text(json_obj: dict, required_keys: list[str] = []) -> str:
    """Format extracted evidence as a compact string for the reasoning model."""
    lines = []
    if len(required_keys) == 0:
        required_keys = json_obj.keys()
    for k in required_keys:
        if isinstance(json_obj[k], dict):
            lines.append(f"{k}: {_json_to_text(json_obj[k])}")
        else:
            lines.append(f"{k}: {json_obj[k]}")
    
    if "_parse_error" in json_obj:
        # NOTE: assumes that all parsing errors contain keys "_parse_error" and "_raw"
        lines.append(f"Original had a JSON parse error: {json_obj['_parse_error']}")
        lines.append(f"Original text: {json_obj['_raw']}")
    
    return "\n".join(lines)


def url_to_mim(url: str) -> str | None:
    """Extract MIM number from OMIM URL."""
    match = re.search(r"omim\.org/entry/(\d+)", url)
    if match:
        return match.group(1)
    return None

if __name__ == "__main__":
    # Test the fetch_and_extract tool
    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10428568/"
    question = "What is the evidence for the efficacy of ASO therapy in patients with ABCA4 mutations?"
    result = fetch_and_extract(url, question, {})
    print(result)
    with open("dumps/fetch_and_extract_tool.txt", "w") as f:
        f.write(result)