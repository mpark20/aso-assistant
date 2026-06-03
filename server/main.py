"""
FastAPI service for the N1C VARIANT ASO assessment pipeline.

Each pipeline phase exposed as its own endpoint so clients can run steps
serially and pass the accumulated AssessmentContext between calls.
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
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
DEFAULT_MODEL_NAME = "claude-sonnet-4-6" #"gpt-5"

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
    response_format: Literal["json", "text"] = Field(
        "json",
        description='Use "text" for a plain-text body (Content-Type: text/plain) instead of JSON.',
    )


class FullRunRequest(BaseModel):
    """Run the full assessment in one request (same as ASOAssessmentPipeline.run)."""

    hgvs: str
    steps_to_run: Optional[list[str]] = Field(
        None,
        description="If set, only these steps run (see /health for valid names).",
    )
    options: Optional[PipelineOptions] = None


class StepApprovalRequest(BaseModel):
    """Accept a user-approved (possibly edited) step result and merge it into context."""

    hgvs: str
    context: Optional[dict[str, Any]] = Field(
        None,
        description="Serialized AssessmentContext after the step LLM run.",
    )
    step_result: dict[str, Any] = Field(
        ...,
        description="Serialized StepResult; summary/reasoning/classification may be edited.",
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
    raw_cls = data["classification"]
    if isinstance(raw_cls, str):
        cls_norm = raw_cls.strip().lower().replace(" ", "_").replace("-", "_")
    else:
        cls_norm = raw_cls
    return StepResult(
        step_name=data["step_name"],
        classification=EligibilityClassification(cls_norm),
        summary=data.get("summary", ""),
        reasoning=data.get("reasoning", ""),
        data_used=data.get("data_used") or {},
        metadata=data.get("metadata") or {},
        error=data.get("error"),
        token_usage=data.get("token_usage") or {},
        edits=data.get("edits") or [],
    )


def _fmt_enum_label(value: str) -> str:
    return value.replace("_", " ").strip()

def report_to_json(report: ASOAssessmentReport) -> dict[str, Any]:
    out = report.to_dict()
    if report.context is not None:
        out["context"] = assessment_context_to_dict(report.context)
    return out

def report_to_plain_text(report: ASOAssessmentReport) -> str:
    """Human-readable report for downloads or text clients (not JSON)."""
    lines: list[str] = []

    lines.append("ASO ASSESSMENT REPORT")
    lines.append("=" * 72)
    lines.append(f"HGVS: {report.hgvs}")
    if report.gene_id:
        lines.append(f"Gene: {report.gene_id}")
    if report.date:
        lines.append(f"Date: {report.date}")
    if report.model_name:
        lines.append(f"Model: {report.model_name}")
    lines.append("")

    lines.append("Strategy classifications")
    lines.append("-" * 72)
    lines.append(f"  Splice correction:    {_fmt_enum_label(report.splice_correction.value)}")
    lines.append(f"  Exon skipping:        {_fmt_enum_label(report.exon_skipping.value)}")
    lines.append(f"  Transcript knockdown: {_fmt_enum_label(report.transcript_knockdown.value)}")
    lines.append(f"  WT upregulation:      {_fmt_enum_label(report.wt_upregulation.value)}")
    lines.append("")

    summary = report.summary
    if isinstance(summary, dict):
        lines.append("Clinical synthesis")
        lines.append("=" * 72)
        narrative_fields = [
            ("overall_summary", "Overall summary"),
            ("variant_description", "Variant description"),
            ("inheritance_summary", "Inheritance"),
            ("pathomechanism_summary", "Pathomechanism"),
            ("splicing_summary", "Splicing"),
        ]
        for key, heading in narrative_fields:
            val = summary.get(key)
            if val:
                lines.append(heading)
                lines.append("-" * len(heading))
                lines.append(str(val).strip())
                lines.append("")

        assessments = summary.get("strategy_assessments")
        if isinstance(assessments, dict) and assessments:
            lines.append("Per-strategy detail")
            lines.append("-" * 72)
            for strat, block in assessments.items():
                title = _fmt_enum_label(str(strat)).title()
                lines.append(title)
                lines.append("~" * min(len(title), 72))
                if isinstance(block, dict):
                    for fk, fv in block.items():
                        if fv in (None, ""):
                            continue
                        label = _fmt_enum_label(str(fk)).title()
                        lines.append(f"  {label}: {fv}")
                else:
                    lines.append(f"  {block}")
                lines.append("")

        for key, heading in (
            ("recommended_next_steps", "Recommended next steps"),
            ("important_caveats", "Important caveats"),
        ):
            val = summary.get(key)
            if not val:
                continue
            lines.append(heading)
            lines.append("-" * len(heading))
            if isinstance(val, list):
                for item in val:
                    lines.append(f"  • {item}")
            else:
                lines.append(f"  {val}")
            lines.append("")

        used = {
            "overall_summary",
            "variant_description",
            "inheritance_summary",
            "pathomechanism_summary",
            "splicing_summary",
            "strategy_assessments",
            "recommended_next_steps",
            "important_caveats",
        }
        extra = {k: v for k, v in summary.items() if k not in used}
        if extra:
            lines.append("Additional fields")
            lines.append("-" * 72)
            lines.append(json.dumps(extra, indent=2, ensure_ascii=False))
            lines.append("")
    elif summary:
        lines.append("Summary")
        lines.append("-" * 72)
        lines.append(str(summary).strip())
        lines.append("")

    lines.append("Pipeline steps (summary)")
    lines.append("=" * 72)
    for name in sorted(report.step_results.keys()):
        sr = report.step_results[name]
        lines.append(name)
        lines.append("-" * min(len(name), 72))
        lines.append(f"  Classification: {_fmt_enum_label(sr.classification.value)}")
        if sr.summary:
            lines.append(f"  Summary: {sr.summary.strip()}")
        if sr.error:
            lines.append(f"  Error: {sr.error}")
        if sr.reasoning:
            lines.append("  Reasoning:")
            for reasoning_line in sr.reasoning.strip().splitlines():
                lines.append(f"    {reasoning_line}")
        lines.append("")

    if report.context is not None:
        ctx = assessment_context_to_dict(report.context)
        skip = {"raw_cache"}
        lines.append("Assessment context")
        lines.append("=" * 72)
        for k in sorted(ctx.keys()):
            if k in skip:
                continue
            v = ctx[k]
            if v in (None, "", [], {}):
                continue
            lines.append(f"  {k}: {v}")
        lines.append("")

    if report.total_token_usage:
        lines.append("Token usage")
        lines.append("-" * 72)
        lines.append(json.dumps(report.total_token_usage, indent=2))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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


def _approve_step(step_name: str, body: StepApprovalRequest) -> dict[str, Any]:
    if step_name not in ASOAssessmentPipeline.STEP_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown step {step_name!r}. Valid: {list(PIPELINE_STEP_NAMES)}",
        )
    pipeline = _pipeline_from_options(body.options)
    ctx = assessment_context_from_dict(body.context, body.hgvs)
    try:
        sr = step_result_from_dict(body.step_result)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid step_result: {e}",
        ) from e
    try:
        approved = pipeline.merge_step_result(step_name, sr, ctx)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "step": step_name,
        "step_result": step_result_to_dict(approved),
        "context": assessment_context_to_dict(ctx),
    }


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
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid step_results payload: {e}"
        ) from e
    pathomechanism_result = step_results.get("pathomechanism")
    if ctx.pathomechanism == Pathomechanism.UNKNOWN:
        if pathomechanism_result is None:
            pathomechanism_result = StepResult(
                step_name="pathomechanism",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="Pathomechanism unknown.",
                reasoning="",
                data_used={},
            )
        report = pipeline._make_early_exit_report(
            body.hgvs, ctx, step_results, pathomechanism_result
        )
    else:
        report = pipeline.make_final_report(body.hgvs, ctx, step_results)
    #return PlainTextResponse(report_to_plain_text(report))
    return jsonable_encoder(report_to_json(report))


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


@app.post("/assessment/steps/{step_name}/approve")
async def assessment_step_approve(
    step_name: str, body: StepApprovalRequest
) -> dict[str, Any]:
    """Record an approved (optionally user-edited) step result and refresh context."""
    return jsonable_encoder(_approve_step(step_name, body))


@app.get("/health")
async def health_check() -> dict[str, Any]:
    step_paths = [f"/assessment/steps/{name}" for name in PIPELINE_STEP_NAMES]
    approve_paths = [f"/assessment/steps/{name}/approve" for name in PIPELINE_STEP_NAMES]
    return {
        "status": "ok",
        "endpoints": {
            "routing": "/assessment/steps/routing",
            "final_report": "/assessment/steps/final_report",
            "steps": step_paths,
            "step_approve": approve_paths,
        },
        "pipeline_step_names": list(PIPELINE_STEP_NAMES),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
