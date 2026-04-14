"""
FastAPI service for the N1C VARIANT ASO assessment pipeline.

Each pipeline phase exposed as its own endpoint so clients can run steps
serially and pass the accumulated AssessmentContext between calls.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aso_workflow.data_model import (
    AssessmentContext,
    ASOAssessmentReport,
    EligibilityClassification,
    InheritancePattern,
    Pathomechanism,
    StepResult,
)
from aso_workflow.pipeline import ASOAssessmentPipeline

app = FastAPI(
    title="ASO Variant Assistant",
    description="HTTP API for ASOAssessmentPipeline steps and full runs.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mirrors ASOAssessmentPipeline.STEP_MAP keys (single source for route list)
PIPELINE_STEP_NAMES: tuple[str, ...] = tuple(ASOAssessmentPipeline.STEP_MAP.keys())
DEFAULT_MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"

class PipelineOptions(BaseModel):
    """LLM and data-source options; matches ASOAssessmentPipeline constructor."""

    model_name: Optional[str] = DEFAULT_MODEL_NAME
    llm_only: bool = False
    use_web_search: bool = False
    verbose: bool = False


class StepRequest(BaseModel):
    """Run a single pipeline step."""

    hgvs: str = Field(..., description='e.g. "NM_000350.3(ABCA4):c.2626C>T"')
    context: Optional[dict[str, Any]] = Field(
        None,
        description="Serialized AssessmentContext from a prior step; omit to start fresh.",
    )
    options: Optional[PipelineOptions] = None


class RoutingRequest(BaseModel):
    """Compute Step 4 section routing from the current context (no LLM)."""

    hgvs: str
    context: Optional[dict[str, Any]] = None
    options: Optional[PipelineOptions] = None


class FinalReportRequest(BaseModel):
    """Synthesize the final report from completed step results."""

    hgvs: str
    context: Optional[dict[str, Any]] = None
    step_results: dict[str, dict[str, Any]] = Field(
        ...,
        description="Map of step name → serialized StepResult (as returned by step endpoints).",
    )
    options: Optional[PipelineOptions] = None


class FullRunRequest(BaseModel):
    """Run the full assessment in one request (same as ASOAssessmentPipeline.run)."""

    hgvs: str
    steps_to_run: Optional[list[str]] = Field(
        None,
        description="If set, only these steps run (see /health for valid names).",
    )
    options: Optional[PipelineOptions] = None


def _pipeline_from_options(opts: Optional[PipelineOptions]) -> ASOAssessmentPipeline:
    o = opts or PipelineOptions()
    return ASOAssessmentPipeline(
        model_name=o.model_name,
        llm_only=o.llm_only,
        use_web_search=o.use_web_search,
        verbose=o.verbose,
    )


def assessment_context_from_dict(
    data: Optional[dict[str, Any]], hgvs_fallback: str
) -> AssessmentContext:
    if not data:
        return AssessmentContext(hgvs_input=hgvs_fallback)
    d = dict(data)
    d.setdefault("hgvs_input", hgvs_fallback)
    if d.get("inheritance_pattern") is not None and isinstance(
        d["inheritance_pattern"], str
    ):
        d["inheritance_pattern"] = InheritancePattern(d["inheritance_pattern"])
    if d.get("pathomechanism") is not None and isinstance(d["pathomechanism"], str):
        d["pathomechanism"] = Pathomechanism(d["pathomechanism"])
    valid = {f.name for f in fields(AssessmentContext)}
    kwargs = {k: v for k, v in d.items() if k in valid}
    return AssessmentContext(**kwargs)


def assessment_context_to_dict(ctx: AssessmentContext) -> dict[str, Any]:
    raw = asdict(ctx)
    if raw.get("inheritance_pattern") is not None and hasattr(
        raw["inheritance_pattern"], "value"
    ):
        raw["inheritance_pattern"] = raw["inheritance_pattern"].value
    if raw.get("pathomechanism") is not None and hasattr(
        raw["pathomechanism"], "value"
    ):
        raw["pathomechanism"] = raw["pathomechanism"].value
    return raw


def step_result_to_dict(sr: StepResult) -> dict[str, Any]:
    d = asdict(sr)
    d["classification"] = sr.classification.value
    return d


def step_result_from_dict(data: dict[str, Any]) -> StepResult:
    return StepResult(
        step_name=data["step_name"],
        classification=EligibilityClassification(data["classification"]),
        summary=data.get("summary", ""),
        reasoning=data.get("reasoning", ""),
        data_used=data.get("data_used") or {},
        metadata=data.get("metadata") or {},
        error=data.get("error"),
        token_usage=data.get("token_usage") or {},
    )


def serialize_report(report: ASOAssessmentReport) -> dict[str, Any]:
    out = report.to_dict()
    if report.context is not None:
        out["context"] = assessment_context_to_dict(report.context)
    return out


def _run_single_step(step_name: str, body: StepRequest) -> dict[str, Any]:
    if step_name not in ASOAssessmentPipeline.STEP_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown step {step_name!r}. Valid: {list(PIPELINE_STEP_NAMES)}",
        )
    pipeline = _pipeline_from_options(body.options)
    ctx = assessment_context_from_dict(body.context, body.hgvs)
    result = pipeline.run_step(step_name, body.hgvs, ctx)
    return {
        "step": step_name,
        "step_result": step_result_to_dict(result),
        "context": assessment_context_to_dict(ctx),
    }


@app.post("/assessment/run")
async def assessment_full_run(body: FullRunRequest) -> dict[str, Any]:
    """Run the full pipeline (or a subset via steps_to_run)."""
    pipeline = _pipeline_from_options(body.options)
    report = pipeline.run(body.hgvs, steps_to_run=body.steps_to_run)
    return jsonable_encoder(serialize_report(report))


@app.post("/assessment/steps/routing")
async def assessment_routing(body: RoutingRequest) -> dict[str, Any]:
    """Step 4: which sections (exon skipping / knockdown / WT upregulation) apply."""
    pipeline = _pipeline_from_options(body.options)
    ctx = assessment_context_from_dict(body.context, body.hgvs)
    sections = pipeline._route_to_sections(ctx)
    explanation = pipeline._explain_routing(ctx)
    return {
        "sections": sections,
        "explanation": explanation,
        "context": assessment_context_to_dict(ctx),
    }


@app.post("/assessment/steps/final_report")
async def assessment_final_report(body: FinalReportRequest) -> dict[str, Any]:
    """Final synthesis step: requires hgvs, context, and all step_results gathered so far."""
    pipeline = _pipeline_from_options(body.options)
    ctx = assessment_context_from_dict(body.context, body.hgvs)
    try:
        step_results = {k: step_result_from_dict(v) for k, v in body.step_results.items()}
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid step_results payload: {e}"
        ) from e
    report = pipeline.make_final_report(body.hgvs, ctx, step_results)
    return jsonable_encoder(serialize_report(report))


# Register one POST route per named pipeline step
for _step_name in PIPELINE_STEP_NAMES:

    def _make_step_handler(sn: str):
        async def _handler(body: StepRequest) -> dict[str, Any]:
            return jsonable_encoder(_run_single_step(sn, body))

        _handler.__name__ = f"step_{sn}"
        return _handler

    app.post(
        f"/assessment/steps/{_step_name}",
        name=f"assessment_step_{_step_name}",
        tags=["pipeline-steps"],
    )(_make_step_handler(_step_name))


@app.get("/health")
async def health_check() -> dict[str, Any]:
    step_paths = [f"/assessment/steps/{name}" for name in PIPELINE_STEP_NAMES]
    return {
        "status": "ok",
        "endpoints": {
            "full_run": "/assessment/run",
            "routing": "/assessment/steps/routing",
            "final_report": "/assessment/steps/final_report",
            "steps": step_paths,
        },
        "pipeline_step_names": list(PIPELINE_STEP_NAMES),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
