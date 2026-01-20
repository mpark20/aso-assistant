from src.tool_llm import ToolLLM
from src.tools.general import ScholarSearchTool, WebSearchTool, BrowseWebpageTool
from src.tools.genetics import NCBISearchTool
from src.tools._apis import search_mutalyzer
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from fastapi import Cookie, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


app = FastAPI(title="ASO Variant Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "gpt-4o"
MAX_TOOL_CALLS = 10
MAX_TOKENS = 16000

with open(Path(__file__).parent / "aso_workflow" / "system_prompt.txt", "r") as f:
    SYSTEM_PROMPT_TMPL = f.read()

with open(Path(__file__).parent / "aso_workflow" / "prompts.json", "r") as f:
    PROTOCOL = json.load(f)

class Variant(BaseModel):
    transcript: str
    coding_change: str
    gene: str

class WorkflowRequest(BaseModel):
    step: str
    variant: Variant

class WorkflowResponse(BaseModel):
    generated_text: str
    total_tokens: int
    tool_call_count: int
    stopped_reason: str
    tool_calls: List[Dict[str, Any]]

@app.post("/chat")
async def run_workflow(request: WorkflowRequest, stream: bool = False):
    # get background variant info
    hgvs = request.variant.transcript + ":" + request.variant.coding_change
    variant_info = search_mutalyzer(hgvs)
    variant_info["gene"] = request.variant.gene

    # get tool descriptions
    tools = [WebSearchTool(), BrowseWebpageTool(), ScholarSearchTool(), NCBISearchTool()]
    tool_descriptions = ""
    for tool in tools:
        schema = tool.get_description()
        name, desc = schema.get("name"), schema.pop("description")
        tool_descriptions += f"Tool: {name}\nDescription: {desc}\n"
        if schema.get("required_parameters"):
            tool_descriptions += f"Required Parameters: {schema.pop('required_parameters')}\n"
        tool_descriptions += f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
    
    # assemble prompt
    system_prompt = SYSTEM_PROMPT_TMPL.replace("<<tool_descriptions>>", tool_descriptions)
    curr_step = PROTOCOL[request.step]
    prompt_tmpl = curr_step["prompt"]
    prompt = prompt_tmpl
    prompt = prompt.format(variant=hgvs, gene=request.variant.gene)
    if curr_step.get("additional_instructions"):
        prompt += "\nAdditional Instructions: " + curr_step["additional_instructions"]
    
    prompt += "\nHere's some background info on the variant of interest:\n" + json.dumps(variant_info, indent=2)
    print(prompt)

    # call LLM with tools
    tool_llm = ToolLLM(
        model=MODEL,
        tools=tools,
    )

    if stream:
        # Stream newline-delimited JSON updates per iteration to the client
        async def event_stream():
            async for update in tool_llm.run_stream(
                messages=[
                    {"role": "user", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tool_calls=MAX_TOOL_CALLS,
                max_tokens=MAX_TOKENS,
                stop_sequences=["</call_tool>", "</solution>"],
                verbose=True,
            ):
                yield json.dumps(update) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    return await tool_llm.run(
        messages=[
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tool_calls=MAX_TOOL_CALLS,
        max_tokens=MAX_TOKENS,
        stop_sequences=["</call_tool>", "</solution>"],
        verbose=True,
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "endpoints": ["/chat"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # import asyncio
    # variant = Variant(
    #     transcript="NM_000350.3",
    #     coding_change="c.2626C>T",
    #     gene="ABCA4",
    # )
    # request = WorkflowRequest(
    #     step="aso_check",
    #     variant=variant,
    # )
    # response = asyncio.run(run_workflow(request))
    # with open("tool_llm_sample.json", "w") as f:
    #     json.dump(response, f, indent=2)