"""
ASOAssessmentPipeline - Main orchestrator for the N1C VARIANT ASO eligibility assessment.

Runs Steps 0-3 in sequence, routes to applicable Sections A/B/C, 
and generates a final report. Supports running individual steps in isolation.

Usage:
    pipeline = ASOAssessmentPipeline()

    # Run full assessment
    report = pipeline.run("NM_000350.3(ABCA4):c.2626C>T")

    # Run individual steps
    context = AssessmentContext(hgvs_input="NM_000350.3(ABCA4):c.2626C>T")
    step0_result = pipeline.run_step("variant_check", context=context)
    step1_result = pipeline.run_step("inheritance_pattern", context=context)
    # etc.
"""
import json
import traceback
from typing import Optional
from datetime import datetime

from aso_workflow.data_model import (
    AssessmentContext,
    ASOAssessmentReport,
    EligibilityClassification,
    StepResult,
)
from aso_workflow.steps import (
    run_aso_check,
    run_variant_check,
    run_inheritance_pattern,
    run_pathomechanism,
    run_splicing_effects,
    route_to_sections,
    explain_routing,
    assess_exon_skipping,
    assess_knockdown,
    assess_wt_upregulation,
)
from aso_workflow.utils.llm import call_llm
from aso_workflow.prompts import SYSTEM_PROMPTS


class ASOAssessmentPipeline:
    """
    Orchestrates the full N1C VARIANT ASO eligibility assessment.

    Steps mirror the protocol exactly:
      Step 0 → Variant check (early exit if invalid)
      ASO Check → Search for existing ASO literature (informational)
      Step 1 → Inheritance pattern
      Step 2 → Pathomechanism + haploinsufficiency
      Step 3 → Splicing effects → splice correction classification
      Step 4 → Routes to Section A and/or B and/or C
      Section A → Exon skipping eligibility
      Section B → Transcript knockdown eligibility
      Section C → WT upregulation assessment
      Final   → Synthesized report

    All steps are independently callable via run_step().
    """

    STEP_MAP = {
        "aso_check": run_aso_check,
        "variant_check": run_variant_check,
        "inheritance_pattern": run_inheritance_pattern,
        "pathomechanism": run_pathomechanism,
        "splicing_effects": run_splicing_effects,
        "exon_skipping": assess_exon_skipping,
        "knockdown": assess_knockdown,
        "wt_upregulation": assess_wt_upregulation,
    }

    def __init__(self, model_name: str | None = None, verbose: bool = True, llm_only: bool = False):
        """
        Args:
            model_name: Optional model name for LLM calls
            verbose: If True, print progress to stdout during pipeline execution
            llm_only: If True, bypass database calls in all steps; only gene, norm_hgvs, and instruction are added to prompts
        """
        self.model_name = model_name
        self.verbose = verbose
        self.llm_only = llm_only


    def run(
        self,
        hgvs: str,
        steps_to_run: Optional[list[str]] = None,
    ) -> ASOAssessmentReport:
        """
        Run the full (or partial) ASO assessment pipeline for a given HGVS variant.

        Args:
            hgvs: HGVS variant string (e.g., "NM_000350.3(ABCA4):c.2626C>T")
            steps_to_run: Optional list of step names to run. If None, runs all steps.
                          Valid values: "variant_check", "aso_check", "inheritance_pattern", "pathomechanism",
                                        "splicing_effects", "exon_skipping", "knockdown", "wt_upregulation"

        Returns:
            ASOAssessmentReport with all classifications and reasoning
        """
        context = AssessmentContext(hgvs_input=hgvs)
        step_results: dict[str, StepResult] = {}

        self._log(f"\n{'='*60}")
        self._log(f"N1C VARIANT ASO Assessment Pipeline")
        self._log(f"Variant: {hgvs}")
        self._log(f"{'='*60}\n")

        # ── Step 0: Variant Check ─────────────────────────────────
        if self._should_run("variant_check", steps_to_run):
            self._log("Step 0: Variant Check...")
            result = self._safe_run_step(run_variant_check, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
            step_results["variant_check"] = result

            if result.classification == EligibilityClassification.UNABLE_TO_ASSESS:
                self._log("  ⚠ Variant check failed — stopping pipeline.")
                return self._make_early_exit_report(hgvs, context, step_results, result)
            self._log(f"  ✓ {result.summary}")

        # ── ASO Check (existing ASO literature) ────────────────────
        if self._should_run("aso_check", steps_to_run):
            self._log("ASO Check: Searching for existing ASO studies...")
            result = self._safe_run_step(run_aso_check, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
            step_results["aso_check"] = result
            self._log(f"  ✓ {result.summary}")

        # ── CNV Early Routing (from Step 0) ──────────────────────
        if context.is_cnv_gain and self._should_run("knockdown", steps_to_run):
            self._log("  → CNV Gain detected: routing directly to Section B")
        if context.is_cnv_loss and self._should_run("wt_upregulation", steps_to_run):
            self._log("  → CNV Loss detected: routing directly to Section C")

        # ── Step 1: Inheritance Pattern ───────────────────────────
        if self._should_run("inheritance_pattern", steps_to_run):
            self._log("Step 1: Inheritance Pattern...")
            result = self._safe_run_step(run_inheritance_pattern, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
            step_results["inheritance_pattern"] = result
            self._log(f"  ✓ {result.summary}")

        # ── Step 2: Pathomechanism + Haploinsufficiency ───────────
        if self._should_run("pathomechanism", steps_to_run):
            self._log("Step 2: Pathomechanism + Haploinsufficiency...")
            result = self._safe_run_step(run_pathomechanism, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
            step_results["pathomechanism"] = result

            if result.classification == EligibilityClassification.UNABLE_TO_ASSESS:
                self._log(f"  ⚠ Unable to assess pathomechanism: {result.summary}")
            else:
                self._log(f"  ✓ {result.summary}")

        # ── Step 3: Splicing Effects ──────────────────────────────
        if self._should_run("splicing_effects", steps_to_run):
            self._log("Step 3: Splicing Effects Evaluation...")
            result = self._safe_run_step(run_splicing_effects, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
            step_results["splicing_effects"] = result
            self._log(f"  ✓ Splice correction: {result.classification.value}")
            self._log(f"    {result.summary}")

        # ── Step 4: Route to Sections ─────────────────────────────
        self._log("Step 4: Routing to relevant sections...")
        sections = route_to_sections(context)
        routing_explanation = explain_routing(context)
        self._log(f"  {routing_explanation}")

        # Override routing if explicit steps_to_run provided
        if steps_to_run:
            sections["exon_skipping"] = "exon_skipping" in steps_to_run
            sections["knockdown"] = "knockdown" in steps_to_run
            sections["wt_upregulation"] = "wt_upregulation" in steps_to_run

        # ── Section A: Exon Skipping ──────────────────────────────
        if sections.get("exon_skipping"):
            self._log("Section A: Canonical Exon Skipping...")
            result = self._safe_run_step(assess_exon_skipping, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
            step_results["exon_skipping"] = result
            self._log(f"  ✓ Exon skipping: {result.classification.value}")
            self._log(f"    {result.summary}")

        # ── Section B: Transcript Knockdown ──────────────────────
        if sections.get("knockdown"):
            self._log("Section B: Transcript Knockdown...")
            result = self._safe_run_step(assess_knockdown, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
            step_results["knockdown"] = result
            self._log(f"  ✓ Knockdown: {result.classification.value}")
            self._log(f"    {result.summary}")

        # ── Section C: WT Upregulation ────────────────────────────
        if sections.get("wt_upregulation"):
            self._log("Section C: Wildtype Allele Upregulation...")
            result = self._safe_run_step(assess_wt_upregulation, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
            step_results["wt_upregulation"] = result
            self._log(f"  ✓ WT upregulation: {result.classification.value}")
            self._log(f"    {result.summary}")

        # ── Final Report ──────────────────────────────────────────
        self._log("\nGenerating final report...")
        report = self.make_final_report(hgvs, context, step_results)

        self._log(f"\n{'='*60}")
        self._log("ASSESSMENT COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"Splice correction:    {report.splice_correction.value}")
        self._log(f"Exon skipping:        {report.exon_skipping.value}")
        self._log(f"Transcript knockdown: {report.transcript_knockdown.value}")
        self._log(f"WT upregulation:      {report.wt_upregulation.value}")
        self._log(f"\nSummary: {report.summary}")
        self._log(f"{'='*60}\n")

        return report

    def run_step(
        self,
        step_name: str,
        hgvs: str,
        context: Optional[AssessmentContext] = None,
    ) -> StepResult:
        """
        Run a single step in isolation.

        Args:
            step_name: One of "variant_check", "aso_check", "inheritance_pattern", "pathomechanism",
                       "splicing_effects", "exon_skipping", "knockdown", "wt_upregulation"
            hgvs: HGVS variant string
            context: Optional pre-populated context. Created fresh if None.

        Returns:
            StepResult for the requested step
        """
        if step_name not in self.STEP_MAP:
            raise ValueError(
                f"Unknown step '{step_name}'. Valid steps: {list(self.STEP_MAP.keys())}"
            )

        if context is None:
            context = AssessmentContext(hgvs_input=hgvs)

        step_fn = self.STEP_MAP[step_name]
        return self._safe_run_step(step_fn, hgvs, context, model_name=self.model_name, llm_only=self.llm_only)
    

    def make_final_report(
        self,
        hgvs: str,
        context: AssessmentContext,
        step_results: dict[str, StepResult],
    ) -> ASOAssessmentReport:
        """
        Generate the final ASO assessment report.

        Args:
            hgvs: Original input HGVS
            context: Completed assessment context
            step_results: Dict of all step results

        Returns:
            Populated ASOAssessmentReport
        """
        # ── Compile step summaries for LLM ───────────────────────────
        step_summaries = {}
        for name, result in step_results.items():
            step_summaries[name] = {
                "classification": result.classification.value,
                "summary": result.summary,
                "reasoning": result.reasoning[:800] if result.reasoning else "",
                "metadata": result.metadata,
                "error": result.error,
            }

        user_msg = f"""Please generate the final ASO assessment report.

VARIANT: {hgvs}
NORMALIZED HGVS: {context.hgvs_normalized}
GENE: {context.gene_id}

STEP RESULTS SUMMARY:
{json.dumps(step_summaries, indent=2)}

CONTEXT SUMMARY:
- Inheritance: {context.inheritance_pattern.value if context.inheritance_pattern else 'unknown'}
- Pathomechanism: {context.pathomechanism.value if context.pathomechanism else 'unknown'}
- Haploinsufficiency: {context.is_haploinsufficient}
- Splicing evidence: {context.has_splicing_evidence}
- CNV Gain: {context.is_cnv_gain}, CNV Loss: {context.is_cnv_loss}

Please synthesize these results into the final report JSON.
"""

        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["final_report"],
            user_message=user_msg,
            expect_json=True,
            model=self.model_name,
        )

        # ── Extract classifications from step results ─────────────────
        def get_classification(step_name: str) -> EligibilityClassification:
            r = step_results.get(step_name)
            return r.classification if r else EligibilityClassification.NOT_APPLICABLE

        splice_correction = get_classification("splicing_effects")
        exon_skipping = get_classification("exon_skipping")
        knockdown = get_classification("knockdown")
        wt_upregulation = get_classification("wt_upregulation")

        # remove the aso_check from the raw_cache, as now it's in step_results
        if "aso_check" in context.raw_cache:
            context.raw_cache.pop("aso_check")

        return ASOAssessmentReport(
            hgvs=hgvs,
            gene_id=context.gene_id,
            splice_correction=splice_correction,
            exon_skipping=exon_skipping,
            transcript_knockdown=knockdown,
            wt_upregulation=wt_upregulation,
            summary=result,
            step_results=step_results,
            context=context,
            total_token_usage=_aggregate_token_usage(step_results),
            date=datetime.now().strftime("%Y-%m-%d"),
            model_name=self.model_name,
        )

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _safe_run_step(self, step_fn, hgvs: str, context: AssessmentContext, model_name: str | None = None, llm_only: bool = False) -> StepResult:
        """
        Run a step function with full error handling.
        Returns an UNABLE_TO_ASSESS result if the step crashes.
        """
        try:
            return step_fn(hgvs, context, model_name=model_name, llm_only=llm_only)
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"  ✗ Step crashed: {e}")
            return StepResult(
                step_name=step_fn.__name__,
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary=f"Step failed with error: {type(e).__name__}: {e}",
                reasoning="",
                data_used={},
                error=f"{type(e).__name__}: {e}\n{tb[:500]}",
            )

    def _should_run(self, step_name: str, steps_to_run: Optional[list[str]]) -> bool:
        """Check if a step should be executed given the steps_to_run filter."""
        if steps_to_run is None:
            return True
        return step_name in steps_to_run

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _make_early_exit_report(
        self,
        hgvs: str,
        context: AssessmentContext,
        step_results: dict,
        failing_result: StepResult,
    ) -> ASOAssessmentReport:
        """Create a report for early exits (e.g., invalid variant)."""
        return ASOAssessmentReport(
            hgvs=hgvs,
            gene_id=context.gene_id,
            step_results=step_results,
            splice_correction=EligibilityClassification.UNABLE_TO_ASSESS,
            exon_skipping=EligibilityClassification.UNABLE_TO_ASSESS,
            transcript_knockdown=EligibilityClassification.UNABLE_TO_ASSESS,
            wt_upregulation=EligibilityClassification.UNABLE_TO_ASSESS,
            summary=f"Assessment stopped at {failing_result.step_name}: {failing_result.summary}",
            context=context,
            total_token_usage=_aggregate_token_usage(step_results),
        )


def _aggregate_token_usage(step_results: dict[str, StepResult]) -> dict[str, dict[str, int]]:
    """Combine token usage from all steps into a single per-model breakdown."""
    combined: dict[str, dict[str, int]] = {}
    for result in step_results.values():
        for model, usage in result.token_usage.items():
            if model not in combined:
                combined[model] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            combined[model]["input_tokens"] += usage.get("input_tokens", 0)
            combined[model]["output_tokens"] += usage.get("output_tokens", 0)
            combined[model]["total_tokens"] += usage.get("total_tokens", 0)
    return combined


if __name__ == "__main__":
    from dataclasses import asdict

    pipeline = ASOAssessmentPipeline()
    hgvs = "NM_000350.3:c.2626C>T"
    report = pipeline.run(hgvs)

    with open(f"dumps/{hgvs}.json", "w") as f:
        json.dump(asdict(report), f, indent=2)