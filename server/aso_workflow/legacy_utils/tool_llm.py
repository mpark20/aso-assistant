import json
import litellm
import logging
import tiktoken
import re
from typing import Optional, Any, List, Dict, TypedDict
from pydantic import BaseModel
from aso_workflow.legacy_utils.tools import BaseTool, ToolOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class GenerateWithToolsOutput(TypedDict):
    generated_text: str
    total_tokens: int
    tool_call_count: int
    stopped_reason: str
    final_answer: Optional[str]
    tool_calls: List[Dict[str, Any]]

class ToolLLM:
    def __init__(
        self,
        model: str,
        tools: Optional[list[BaseTool]] = [],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        is_commercial: bool = True,
        tokenizer = None
    ):
        self.model = model
        self.base_url = base_url
        self.tools = {tool.name: tool for tool in tools}
        self.tokenizer = tokenizer
        self.api_key = api_key
        self.is_commercial = is_commercial
    
    def _parse_tool(self, tool_str: str) -> BaseTool | str:
        try:
            tool_str = tool_str.replace("```json", "").replace("```", "").strip()
            tool_call = json.loads(tool_str)
            if not "name" in tool_call or not "parameters" in tool_call:
                raise ValueError("Tool call must have a name and parameters")
            tool_name = tool_call["name"]
            tool = self.tools.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found. Available tools: {list(self.tools.keys())}")
            tool_desc = tool.get_description()
            for rp in tool_desc.get("required_parameters", []):
                if rp not in tool_call.get("parameters", {}):
                    raise ValueError(f"Tool {tool_name} requires parameter {rp}")
            
            return tool, tool_call.get("parameters", {})

        except Exception as e:
            return "Malformed tool call: " + str(e)


    async def run(
        self,
        messages: list[dict[str, str]],
        max_tool_calls: int = 10,
        max_tokens: Optional[int] = 8192,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
        verbose: bool = False,
        **kwargs
    ) -> GenerateWithToolsOutput:

        tool_calls = []
        original_message_count = len(messages)
        current_messages = messages.copy()
        tool_call_count = 0
        base_max_tokens = max_tokens
        iter = 0

        while True:
            iter += 1
            remaining_token_count = self._calculate_dynamic_max_tokens(
                current_messages, base_max_tokens
            )
            current_token_count = self._count_tokens(current_messages)
            if verbose:
                logger.info(f"\n--- Commercial API Iteration {iter} ---")
                logger.info(f"Tool calls made so far: {tool_call_count}")
                logger.info(f"Current messages count: {len(current_messages)}")
                logger.info(f"Current message tokens: {current_token_count}")
            
            # don't generate if the token limit was surpassed
            if current_token_count > remaining_token_count:
                logger.info(f"Hit token limit before generation. ({current_token_count}/{current_token_count}), stopping.")
                break
            
            response = await self.generate_step(
                current_messages,
                max_tokens=remaining_token_count,
                temperature=temperature,
                stop_sequences=stop_sequences,
                **kwargs
            )
            # fix any incomplete tags
            if "<solution>" in response and not "</solution>" in response:
                response = response + "</solution>"
            if "<call_tool>" in response and not "</call_tool>" in response:
                response = response + "</call_tool>"
            current_messages.append({"role": "assistant", "content": response})

            # end if we hit the token limit after generation
            new_token_count = self._count_tokens(current_messages)
            if new_token_count >= base_max_tokens:
                if verbose:
                    logger.info(f"Hit token limit after generation ({new_token_count}/{base_max_tokens}), stopping.")
                break
            
            tool_match = re.search(r"<call_tool>(.*?)</call_tool>", response, re.DOTALL)

            if not tool_match:
                logger.info("No tool call found in response, stopping.")
                break
            
            # Handle tool calls
            if tool_call_count >= max_tool_calls:
                logger.info(f"Hit max tool calls ({max_tool_calls}), stopping.")
                error_message = "Exceeded allowed tool call requests."
                tool_calls.append(error_message)
                break
            
            tool_str = tool_match.group(1)
            tool_result_or_error = self._parse_tool(tool_str)
            if isinstance(tool_result_or_error, str):
                logger.info(f"Error parsing tool call: {tool_result_or_error}")
                error_message = tool_result_or_error
                tool_calls.append(error_message)
                break
            
            tool_obj, tool_args = tool_result_or_error
            tool_output = await tool_obj(**tool_args)
            tool_call_count += 1
            tool_calls.append(tool_output)
            tool_output["result"] = tool_output["result"][:10000]
            current_messages.append({
                "role": "user",
                "content": f"<tool_output>{tool_output}</tool_output>",
            })

            # Check token limit again after adding tool results
            final_token_count = self._count_tokens(current_messages)
            if final_token_count >= base_max_tokens:
                if verbose:
                    logger.info(f"Hit token limit after tool execution ({final_token_count}/{base_max_tokens}), stopping.")
                break

        # Calculate generated text (everything after original messages)
        generated_text = ""
        for msg in current_messages[original_message_count:]:
            generated_text += msg["content"]
        final_token_count = self._count_tokens(generated_text)
        answer_match = re.search(r"<solution>(.*?)</solution>", generated_text, re.DOTALL)
        final_answer = None
        if answer_match:
            final_answer = answer_match.group(1)

        output = GenerateWithToolsOutput(
            tool_calls=tool_calls,
            generated_text=generated_text,
            total_tokens=final_token_count,
            tool_call_count=tool_call_count,
            stopped_reason=(
                "max_tokens" if final_token_count >= base_max_tokens else "natural"
            ),
        )
        if final_answer:
            output["final_answer"] = final_answer
        return output

    async def run_stream(
        self,
        messages: list[dict[str, str]],
        max_tool_calls: int = 10,
        max_tokens: Optional[int] = 8192,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
        verbose: bool = False,
        **kwargs
    ):
        """Async generator that yields progress updates (as dicts) per iteration.
        Yields newline-delimited JSON-compatible dicts to be streamed to clients.
        """
        tool_calls: List[Any] = []
        original_message_count = len(messages)
        current_messages = messages.copy()
        tool_call_count = 0
        base_max_tokens = max_tokens
        iteration = 0

        while True:
            iteration += 1
            remaining_token_count = self._calculate_dynamic_max_tokens(
                current_messages, base_max_tokens
            )
            current_token_count = self._count_tokens(current_messages)
            if verbose:
                logger.info(f"\n--- Streaming API Iteration {iteration} ---")
                logger.info(f"Tool calls made so far: {tool_call_count}")
                logger.info(f"Current messages count: {len(current_messages)}")
                logger.info(f"Current message tokens: {current_token_count}")

            # don't generate if the token limit was surpassed
            if current_token_count > remaining_token_count:
                logger.info("Hit token limit before generation. stopping.")
                yield {
                    "type": "stopped",
                    "reason": "token_limit_before_generation",
                    "iteration": iteration,
                    "generated_text": "",
                    "tool_calls": tool_calls,
                    "tool_call_count": tool_call_count,
                }
                break

            response = await self.generate_step(
                current_messages,
                max_tokens=remaining_token_count,
                temperature=temperature,
                stop_sequences=stop_sequences,
                **kwargs,
            )

            # fix any incomplete tags
            if "<solution>" in response and not "</solution>" in response:
                response = response + "</solution>"
            if "<call_tool>" in response and not "</call_tool>" in response:
                response = response + "</call_tool>"
            current_messages.append({"role": "assistant", "content": response})

            # yield after generation step
            generated_text = ""
            for msg in current_messages[original_message_count:]:
                generated_text += msg["content"]
            final_token_count = self._count_tokens(generated_text)
            answer_match = re.search(r"<solution>(.*?)</solution>", generated_text, re.DOTALL)
            final_answer = answer_match.group(1) if answer_match else None

            yield {
                "type": "generation",
                "iteration": iteration,
                "generated_text": msg["content"],
                "total_tokens": final_token_count,
                "tool_call_count": tool_call_count,
            }

            # end if we hit the token limit after generation
            new_token_count = self._count_tokens(current_messages)
            if new_token_count >= base_max_tokens:
                if verbose:
                    logger.info(f"Hit token limit after generation ({new_token_count}/{base_max_tokens}), stopping.")
                break

            tool_match = re.search(r"<call_tool>(.*?)</call_tool>", response, re.DOTALL)

            if not tool_match:
                logger.info("No tool call found in response, stopping.")
                break

            # Handle tool calls
            if tool_call_count >= max_tool_calls:
                logger.info(f"Hit max tool calls ({max_tool_calls}), stopping.")
                error_message = "Exceeded allowed tool call requests."
                tool_calls.append(error_message)
                yield {
                    "type": "error",
                    "iteration": iteration,
                    "error": error_message,
                }
                break

            tool_str = tool_match.group(1)
            tool_result_or_error = self._parse_tool(tool_str)
            if isinstance(tool_result_or_error, str):
                logger.info(f"Error parsing tool call: {tool_result_or_error}")
                error_message = tool_result_or_error
                tool_calls.append(error_message)
                yield {
                    "type": "error",
                    "iteration": iteration,
                    "error": error_message,
                }
                break

            tool_obj, tool_args = tool_result_or_error
            tool_output = await tool_obj(**tool_args)
            tool_call_count += 1
            tool_output["result"] = tool_output["result"][:10000]
            tool_calls.append(tool_output)
            current_messages.append({
                "role": "user",
                "content": f"<tool_output>{tool_output}</tool_output>",
            })

            # yield after tool execution
            generated_text = ""
            for msg in current_messages[original_message_count:]:
                generated_text += msg["content"]
            final_token_count = self._count_tokens(generated_text)

            yield {
                "type": "tool_output",
                "tool_name": tool_output.get("tool_name"),
                "iteration": iteration,
                "total_tokens": final_token_count,
                "tool_calls": tool_calls[-1] if len(tool_calls) else [],
                "tool_call_count": tool_call_count,
            }

            # Check token limit again after adding tool results
            final_token_count = self._count_tokens(current_messages)
            if final_token_count >= base_max_tokens:
                if verbose:
                    logger.info(f"Hit token limit after tool execution ({final_token_count}/{base_max_tokens}), stopping.")
                break

        # Final summary
        generated_text = ""
        for msg in current_messages[original_message_count:]:
            generated_text += msg["content"]
        final_token_count = self._count_tokens(generated_text)
        answer_match = re.search(r"<solution>(.*?)</solution>", generated_text, re.DOTALL)
        final_answer = answer_match.group(1) if answer_match else None

        # Add citation section
        # TODO: this is system prompt specific, move accordingly
        snippet_info = self.parse_snippets(generated_text)
        cite_ids = []
        for match in re.findall(r'<cite\s+id="([^"]+)"', final_answer):
            cite_ids.extend(id_.strip() for id_ in match.split(','))
        citations = {k:v for k,v in snippet_info.items() if k in cite_ids}
        if len(citations):
            cite_str = "\n\n## Citations\n"
            for k, v in citations.items():
                if v.get("url") and v.get("title"):
                    cite_str += f"[{k}]: [{v['title']}]({v['url']})\n"
                elif v.get("url"):
                    cite_str += f"[{k}]: [{v['url']}]({v['url']})\n"
                elif v.get("title"):
                    cite_str += f"[{k}]: {v['title']}\n"
            final_answer = final_answer + cite_str
            generated_text = generated_text + cite_str
        
        cite_links = ", ".join([f"[{k}]({v['url']})" for k,v in citations.items() if v.get('url')])

        output = {
            # "tool_calls": tool_calls,
            "generated_text": current_messages[-1]["content"],
            "total_tokens": final_token_count,
            "tool_call_count": tool_call_count,
            "stopped_reason": (
                "max_tokens" if final_token_count >= base_max_tokens else "natural"
            ),
            "final_answer": final_answer,
            "sources": cite_links,
            "type": "final",
        }
        with open("dumps/tmp.json", "w") as f:
            json.dump(output, f, indent=2)
        yield output

    async def generate_step(self, messages: list[dict[str, str]], **kwargs) -> str:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", None),
            "max_tokens": kwargs.pop("max_tokens", None),
            "stop": kwargs.pop("stop_sequences", None),
        }
        if "gpt-" in self.model:
            params["max_completion_tokens"] = params.pop("max_tokens", None)
        # Add API credentials if available
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url
        params.update(kwargs)
        response = await litellm.acompletion(**params)
        res = response.choices[0].message.content
        print(res)
        return res
    
    def _calculate_dynamic_max_tokens(
        self, current_context: str | list[dict[str, str]], base_max_tokens: int
    ) -> int:
        """Calculate remaining max tokens based on current context."""
        if self.is_commercial:
            # For commercial API models, don't do dynamic calculation - use fixed max_tokens
            return base_max_tokens

        current_token_count = len(self._count_tokens(current_context))
        remaining_tokens = base_max_tokens - current_token_count

        # Ensure we have at least some tokens for generation (minimum 100)
        return max(100, remaining_tokens)

    def _count_tokens(self, content: list[dict[str, str]] | str) -> int:
        try:
            if self.is_commercial:
                # For commercial API models, use litellm.utils.token_counter with model name
                counter_args = {"text": content} if isinstance(content, str) else {"messages": content}
                return litellm.utils.token_counter(model=self.model, **counter_args)
            
            content = content if isinstance(content, str) else "".join(msg["content"] for msg in content)
            if self.tokenizer:
                return len(self.tokenizer.encode(content))
            else:
                # Fallback: rough estimate (4 characters per token)
                return len(content) // 4
        except Exception as e:
            logger.info(f"Warning: Could not count tokens: {e}")
            content = content if isinstance(content, str) else "".join(msg["content"] for msg in content)
            return len(content) // 4
    
    @staticmethod
    def parse_snippets(text: str) -> dict:
        """
        Parse snippet-formatted search results and return a mapping
        of snippet_id -> {url: str | None, title: str | None}.
        """
        results = {}

        # Match each <snippet ...>...</snippet> block
        snippet_pattern = re.compile(
            r'<snippet\s+id=([^\s>]+)>(.*?)</snippet>',
            re.DOTALL | re.IGNORECASE
        )

        for snippet_id, body in snippet_pattern.findall(text):
            url_match = re.search(r'url:\s*(\S+)', body)
            title_match = re.search(r'title:\s*(.+)', body)

            if url_match or title_match:
                results[snippet_id] = {
                    "url": url_match.group(1).split("\\n")[0].strip() if url_match else None,
                    "title": title_match.group(1).split("\\n")[0].strip() if title_match else None,
                }
        return results

async def main():
    from aso_workflow.legacy_utils.tools import WebSearchTool, BrowseWebpageTool, NCBISearchTool, UniProtSearchTool, BrowseUniProtTool

    tools = [NCBISearchTool(), UniProtSearchTool(), BrowseUniProtTool(), WebSearchTool(), BrowseWebpageTool()]
    schema_str = ""
    for tool in tools:
        schema = tool.get_description()
        name, desc = schema.get("name"), schema.pop("description")
        schema_str += f"Tool: {name}\nDescription: {desc}\n"
        if schema.get("required_parameters"):
            schema_str += f"Required Parameters: {schema.pop('required_parameters')}\n"
        schema_str += f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
    print(schema_str)

    system_prompt = f"""
    You are a helpful medical research assistant that can use iterative reasoning and search to answer questions.

    ## INSTRUCTIONS
    - Use <think>...</think> tags to show your reasoning at any point. Start each response by thinking briefly about the question and how you plan to answer it.
    - To call a tool, use <call_tool>...</call_tool> tags, where the content is the json schema defined for the desired tool (see SEARCH TOOLS below).
    - You may alternate between thinking and calling tools until you have a final answer.
    - To provide a final answer, use <solution>...</solution> tags, where the content is your final answer. Only provide an answer when you have enough information to do so.
    - Support every non-trivial claim with retrieved evidence. Wrap the exact claim span in <cite id="ID1,ID2">...</cite> tags, where id are the snippet IDs from the search results. NEVER invent IDs in citations, they must be from tool output messages.
    - Not all questions require a tool call. If you can answer the question without a tool call confidently, do so.

    ## SEARCH TOOLS
    Here is a description of the tools you can use, as well as the json schema you must use to call them.
    {schema_str}

    ## EXAMPLE WORKFLOW
    User: What traits are associated with the genetic variant NM_000123.4:c.567C>T?
    Assistant:
    <think>First, I need to find the traits associated with the genetic variant NM_000123.4:c.567C>T. I can use the ClinVar to find some background information.</think>
    <call_tool>
    {json.dumps({
        "name": "ncbi_search",
        "parameters": {
            "database": "clinvar",
            "query": "NM_000123 AND c.567C>T"
        }
    }, indent=2)}
    </call_tool>
    <tool_output>[<snippet id=S1>NM_000123.4:c.567C>T is associated with trait xyz</snippet></tool_output>
    <solution>
    <cite id="S1">The variant NM_000123.4:c.567C>T is associated with trait xyz</cite>
    </solution>
    """
    
    tool_llm = ToolLLM(
        model="gpt-4o-mini",
        tools=tools,
        is_commercial=True,
    )

    response = await tool_llm.run(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What exon does variant NM_000350.3:c.2626C>T lie on, and what protein domains overlap this exon? Evaluate the importance of such domains, and the impact of downregulating them."},
        ],
        stop_sequences=["</solution>", "</call_tool>"],
        max_tool_calls=10,
        verbose=True,
    )
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())