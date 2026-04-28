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
import pdb
import json
from pathlib import Path
import traceback
from typing import Any, Optional
from datetime import datetime

from aso_workflow.data_model import (
    AssessmentContext,
    ASOAssessmentReport,
    EligibilityClassification,
    InheritancePattern,
    Pathomechanism,
    StepResult,
)
from aso_workflow.utils.apis import (
    search_mutalyzer,
    search_gnomad,
    search_serper,
    search_ensembl_vep,
    search_ncbi,
    search_alt_splicing_events,
)
from aso_workflow.utils.llm import call_llm, FETCH_AND_EXTRACT_TOOL
from aso_workflow.utils.tasks import fetch_protein_context, fetch_transcript_context, fetch_clinical_context
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
        "aso_check": "run_aso_check",
        "variant_check": "run_variant_check",
        "inheritance_pattern": "run_inheritance_pattern",
        "pathomechanism": "run_pathomechanism",
        "splicing_effects": "run_splicing_effects",
        "exon_skipping": "assess_exon_skipping",
        "knockdown": "assess_knockdown",
        "wt_upregulation": "assess_wt_upregulation",
    }

    def __init__(
        self,
        model_name: str | None = None,
        llm_only: bool = False,
        use_web_search: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            model_name: Optional model name for LLM calls
            verbose: If True, print progress to stdout during pipeline execution
            llm_only: If True, bypass database calls in all steps; only gene, norm_hgvs, and instruction are added to prompts
        """
        # call_llm args
        self.model_name = model_name
        self.use_web_search = use_web_search

        # baseline: prompt llm with protocol instructions
        self.llm_only = llm_only

        # logging
        self.verbose = verbose

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
            result = self.run_step("variant_check", hgvs, context)
            step_results["variant_check"] = result

            if result.classification == EligibilityClassification.UNABLE_TO_ASSESS:
                self._log("  ⚠ Variant check failed — stopping pipeline.")
                return self._make_early_exit_report(hgvs, context, step_results, result)
            self._log(f"  ✓ {result.summary}")

        # ── ASO Check (existing ASO literature) ────────────────────
        if self._should_run("aso_check", steps_to_run):
            self._log("ASO Check: Searching for existing ASO studies...")
            result = self.run_step("aso_check", hgvs, context)
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
            result = self.run_step("inheritance_pattern", hgvs, context)
            step_results["inheritance_pattern"] = result
            self._log(f"  ✓ {result.summary}")

        # ── Step 2: Pathomechanism + Haploinsufficiency ───────────
        if self._should_run("pathomechanism", steps_to_run):
            self._log("Step 2: Pathomechanism + Haploinsufficiency...")
            result = self.run_step("pathomechanism", hgvs, context)
            step_results["pathomechanism"] = result

            if result.classification == EligibilityClassification.UNABLE_TO_ASSESS:
                self._log(f"  ⚠ Unable to assess pathomechanism: {result.summary}")
            else:
                self._log(f"  ✓ {result.summary}")

        # ── Step 3: Splicing Effects ──────────────────────────────
        if self._should_run("splicing_effects", steps_to_run):
            self._log("Step 3: Splicing Effects Evaluation...")
            result = self.run_step("splicing_effects", hgvs, context)
            step_results["splicing_effects"] = result
            self._log(f"  ✓ Splice correction: {result.classification.value}")
            self._log(f"    {result.summary}")

        # ── Step 4: Route to Sections ─────────────────────────────
        self._log("Step 4: Routing to relevant sections...")
        sections = self._route_to_sections(context)
        routing_explanation = self._explain_routing(context)
        self._log(f"  {routing_explanation}")

        # Override routing if explicit steps_to_run provided
        if steps_to_run:
            sections["exon_skipping"] = "exon_skipping" in steps_to_run
            sections["knockdown"] = "knockdown" in steps_to_run
            sections["wt_upregulation"] = "wt_upregulation" in steps_to_run

        # ── Section A: Exon Skipping ──────────────────────────────
        if sections.get("exon_skipping"):
            self._log("Section A: Canonical Exon Skipping...")
            result = self.run_step("exon_skipping", hgvs, context)
            step_results["exon_skipping"] = result
            self._log(f"  ✓ Exon skipping: {result.classification.value}")
            self._log(f"    {result.summary}")

        # ── Section B: Transcript Knockdown ──────────────────────
        if sections.get("knockdown"):
            self._log("Section B: Transcript Knockdown...")
            result = self.run_step("knockdown", hgvs, context)
            step_results["knockdown"] = result
            self._log(f"  ✓ Knockdown: {result.classification.value}")
            self._log(f"    {result.summary}")

        # ── Section C: WT Upregulation ────────────────────────────
        if sections.get("wt_upregulation"):
            self._log("Section C: Wildtype Allele Upregulation...")
            result = self.run_step("wt_upregulation", hgvs, context)
            step_results["wt_upregulation"] = result
            self._log(f"  ✓ WT upregulation: {result.classification.value}")
            self._log(f"    {result.summary}")
        
        # Return early if only running invidual steps
        if steps_to_run:
            return step_results

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

        method_name = self.STEP_MAP[step_name]
        return self._safe_run_step(method_name, hgvs, context)

    # ─────────────────────────────────────────────────────────────
    # Step 0: Variant Check
    # ─────────────────────────────────────────────────────────────

    def run_variant_check(self, hgvs: str, context: AssessmentContext) -> StepResult:
        """Execute Step 0: Variant Check.
        By default, this step does NOT use open web search. Instead, we use variant lookup databases.
        If self.llm_only and self.use_web_search are both True, the llm's native web search will be used (if supported).
        """
        mutalyzer_data = search_mutalyzer(hgvs)
        norm_hgvs = mutalyzer_data.get("normalized")
        gene = mutalyzer_data.get("gene_id")

        raw_data = {
            "input_hgvs": hgvs,
            "mutalyzer": mutalyzer_data,
            "mutalyzer_error": mutalyzer_data.get("error"),
        }

        if self.llm_only:
            mutalyzer_data = {"normalized": norm_hgvs, "gene_id": gene}
            user_msg = f"""Please evaluate this HGVS variant for Step 0 of the N1C VARIANT Guidelines.

GENE: {gene}
HGVS: {norm_hgvs}

Apply Step 0 criteria and return your JSON assessment.
"""
        else:
            user_msg = f"""Please evaluate this HGVS variant for Step 0 of the N1C VARIANT Guidelines.

INPUT HGVS: {hgvs}

MUTALYZER NORMALIZATION RESULT:
{mutalyzer_data}

Based on the normalization result and the input variant, apply Step 0 criteria and return 
your JSON assessment.
"""     
        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["variant_check"],
            user_message=user_msg,
            expect_json=True,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        if "_parse_error" in result:
            return StepResult(
                step_name="variant_check",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="LLM response could not be parsed.",
                reasoning=result.get("_raw", ""),
                data_used=raw_data,
                error=result.get("_parse_error"),
                token_usage=usage,
            )

        classification = EligibilityClassification(result.get("classification", "unable_to_assess"))

        context.hgvs_normalized = norm_hgvs
        context.gene_id = mutalyzer_data.get("gene_id")
        context.intronic_or_exonic = "intronic" if mutalyzer_data.get("intronic") else "exonic"
        context.refseq_id = mutalyzer_data.get("refseq_id")

        context.variant_valid = result.get("variant_valid", False)
        context.is_cnv_gain = result.get("is_cnv_gain", False)
        context.is_cnv_loss = result.get("is_cnv_loss", False)

        if mutalyzer_data:
            context.raw_cache["mutalyzer"] = mutalyzer_data
        
        tool_call_logs = result.pop("_tool_call_log", [])

        return StepResult(
            step_name="variant_check",
            classification=classification,
            summary=result.get("reason", ""),
            reasoning=json.dumps(result, indent=2),
            data_used=raw_data,
            metadata={
                "variant_type": result.get("variant_type"),
                "hgvs_normalized": result.get("hgvs_normalized"),
                "gene_id": result.get("gene_id"),
                "warnings": result.get("warnings", []),
                "is_cnv_gain": result.get("is_cnv_gain", False),
                "is_cnv_loss": result.get("is_cnv_loss", False),
                "_tool_call_log": tool_call_logs,
            },
            token_usage=usage,
        )

    # ─────────────────────────────────────────────────────────────
    # ASO Check
    # ─────────────────────────────────────────────────────────────

    def run_aso_check(self, hgvs: str, context: AssessmentContext) -> StepResult | dict[str, Any]:
        """Execute ASO Check: Search for existing ASO literature."""
        if "mutalyzer" not in context.raw_cache:
            context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
        mutalyzer_data = context.raw_cache.get("mutalyzer") or {}
        norm_hgvs = mutalyzer_data.get("normalized", hgvs)
        gene = mutalyzer_data.get("gene_id")

        if self.llm_only:
            raw_data = {}
            mutalyzer_data = {"normalized": norm_hgvs, "gene_id": gene}
            user_msg = f"""Please assess if there are existing studies relating to the use of ASO therapy for the given gene variant.
This can mean that: 
1. an ASO has been developed for the specific variant (see VARIANT LEVEL PAPERS).
2. an ASO has been developed for an exon skipping approach for an exon this variant is located in (see EXON LEVEL PAPERS). Note that this doesn't necessarily need to mention the exact variant in question, but rather those within the same exon.
3. a gapmer ASO or siRNA is available for the gene in question or allele specific for a SNP that is in phase (located on the same chromosome copy) with the pathogenic variant.
IMPORTANT: Exon skipping therapies are important to consider, even though the word "ASO" might not be used in the title.

GENE: {gene}
HGVS: {norm_hgvs}

Apply the ASO check criteria and return your JSON assessment.
"""
        else:
            if "clinvar" not in context.raw_cache:
                clinical_context = fetch_clinical_context(hgvs)
                clingen_data = clinical_context.get("clingen")
                clinvar_data = clinical_context.get("clinvar")
                context.raw_cache["clingen"] = clingen_data
                context.raw_cache["clinvar"] = clinvar_data
            clinvar_data = context.raw_cache.get("clinvar")
            clingen_data = context.raw_cache.get("clingen")

            gene_level_query = f"{gene} AND ((ASO) OR (AON) OR (antisense oligonucleotide) OR (AOs) OR (siRNA) OR (RNAi) OR (gapmer) or (knockdown))"

            gene_level_lit = self._get_pubmed_pmc_results(gene_level_query)

            exon_level_query = gene_level_query[:]
            exon_level_lit = None
            if mutalyzer_data.get("nearest_exon"):
                exon_level_query += f" AND (exon {mutalyzer_data.get('nearest_exon')})"
                exon_level_lit = self._get_pubmed_pmc_results(exon_level_query)

            variant_level_query = gene_level_query[:]
            equiv = mutalyzer_data.get("equivalent_descriptions") or []
            synonyms = [norm_hgvs] + equiv
            if clinvar_data and clinvar_data.get("protein_change"):
                synonyms.append(clinvar_data.get("protein_change"))
            synonyms = list(set([name.split(":")[-1] for name in synonyms if name]))
            if len(synonyms) > 0:
                name_str = " OR ".join(synonyms)
                variant_level_query += f" AND ({name_str})"
            variant_level_lit = self._get_pubmed_pmc_results(variant_level_query)

            raw_data = {
                "variant_level_papers": variant_level_lit,
                "gene_level_papers": gene_level_lit,
            }
            if exon_level_lit is not None:
                raw_data["exon_level_papers"] = exon_level_lit

            raw_data["search_queries_used"] = [gene_level_query, exon_level_query, variant_level_query]

            user_msg = f"""Please assess if there are existing studies relating to the use of ASO therapy for the given gene variant.
This can mean that: 
1. an ASO has been developed for the specific variant (see VARIANT LEVEL PAPERS).
2. an ASO has been developed for an exon skipping approach for an exon this variant is located in (see EXON LEVEL PAPERS). Note that this doesn't necessarily need to mention the exact variant in question, but rather those within the same exon.
3. a gapmer ASO or siRNA is available for the gene in question or allele specific for a SNP that is in phase (located on the same chromosome copy) with the pathogenic variant.
IMPORTANT: Exon skipping therapies are important to consider, even though the word "ASO" might not be used in the title.

GENE: {gene}
HGVS: {norm_hgvs}

CLINVAR DATA:
{clinvar_data}
"""
            if variant_level_lit is not None:
                user_msg += f"VARIANT LEVEL PAPERS:\n{variant_level_lit}\n\n"
            if exon_level_lit is not None:
                user_msg += f"EXON LEVEL PAPERS:\n{exon_level_lit}\n\n"
            if gene_level_lit is not None:
                user_msg += f"GENE LEVEL PAPERS:\n{gene_level_lit}\n\n"

        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["aso_check"],
            user_message=user_msg,
            expect_json=True,
            tools=[FETCH_AND_EXTRACT_TOOL] if not self.llm_only else None,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        if "_parse_error" in result:
            return StepResult(
                step_name="aso_check",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="LLM response could not be parsed.",
                reasoning=result.get("_raw", ""),
                data_used=raw_data,
                error=result.get("_parse_error"),
                token_usage=usage,
            )

        context.raw_cache["aso_check_pubmed"] = raw_data
        try:
            # record type of aso studied
            existing_aso_found = result.get("aso_evidence_found", False)
            context.existing_aso_type = result.get("approach_used", "unknown") if existing_aso_found else "not_applicable"
            
            # record experimental success
            context.existing_aso_success = (
                existing_aso_found and
                result.get("aso_success", False)
            )
            context.existing_aso_sufficient = (
                context.existing_aso_success and 
                result.get("evidence_classification", "") == "sufficient_functional_evidence"
            )
        except Exception:
            # do not infer metadata if system failed to fetch information
            context.existing_aso_type = "not_applicable"

        return StepResult(
            step_name="aso_check",
            classification=EligibilityClassification.ELIGIBLE,
            summary=result.get("summary", ""),
            reasoning=result.get("reasoning", ""),
            data_used=raw_data,
            metadata={
                "evidence_snippets": result.get("evidence_snippets", []),
                "aso_specificity": result.get("aso_specificity", "unknown"),
                "approach_used": result.get("approach_used", "unknown"),
                "aso_success": context.existing_aso_success,
                "aso_sufficient": context.existing_aso_sufficient,
                "evidence_classification": result.get("evidence_classification", "unknown"),
                "warnings": result.get("warnings", []),
                "_tool_call_log": result.get("_tool_call_log", []),
            },
            token_usage=usage,
        )

    # ─────────────────────────────────────────────────────────────
    # Step 1: Inheritance Pattern
    # ─────────────────────────────────────────────────────────────

    def run_inheritance_pattern(self, hgvs: str, context: AssessmentContext) -> StepResult:
        """Execute Step 1: Inheritance Pattern Assessment."""
        if "mutalyzer" not in context.raw_cache:
            context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
        mutalyzer_data = context.raw_cache.get("mutalyzer") or {}
        norm_hgvs = mutalyzer_data.get("normalized", hgvs)
        gene = mutalyzer_data.get("gene_id")

        if self.llm_only:
            raw_data = {}
            mutalyzer_data = {"normalized": norm_hgvs, "gene_id": gene}
            user_msg = f"""Please assess the inheritance pattern for this variant (Step 1 of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}

Apply Step 1 criteria and return your JSON assessment.
"""
        else:
            if "clinvar" not in context.raw_cache:
                clinical_context = fetch_clinical_context(hgvs)
                clingen_data = clinical_context.get("clingen")
                clinvar_data = clinical_context.get("clinvar")
                context.raw_cache["clingen"] = clingen_data
                context.raw_cache["clinvar"] = clinvar_data
            clinvar_data = context.raw_cache.get("clinvar")
            clingen_data = context.raw_cache.get("clingen")

            gnomad_data = search_gnomad(gene, hgvsc=norm_hgvs)
            web_results = search_serper(gene + " inheritance pattern")

            raw_data = {
                "clinvar": clinvar_data,
                "gnomad_summary": gnomad_data,
                "web_search": web_results,
            }

            if clinvar_data:
                context.raw_cache["clinvar"] = clinvar_data
            if clingen_data:
                context.raw_cache["clingen"] = clingen_data
            if gnomad_data:
                context.raw_cache["gnomad"] = gnomad_data

            user_msg = f"""Please assess the inheritance pattern for this variant (Step 1 of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}

CLINVAR DATA:
{clinvar_data}

GNOMAD SUMMARY:
{gnomad_data}

WEB SEARCH RESULTS:
{web_results}

Apply Step 1 criteria and return your JSON assessment.
If you encounter cited URLs or PubMed IDs that appear to have relevant information, you use the `fetch_and_extract` tool to
get a summary of the full text in relation to a research question of interest.
"""
        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["inheritance_pattern"],
            user_message=user_msg,
            expect_json=True,
            tools=[FETCH_AND_EXTRACT_TOOL] if not self.llm_only else None,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        if "_parse_error" in result:
            return StepResult(
                step_name="inheritance_pattern",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="LLM response could not be parsed.",
                reasoning=result.get("_raw", ""),
                data_used=raw_data,
                error=result.get("_parse_error"),
                token_usage=usage,
            )

        pattern_str = result.get("inheritance_pattern", "unknown")
        try:
            context.inheritance_pattern = InheritancePattern(pattern_str)
        except ValueError:
            context.inheritance_pattern = InheritancePattern.UNKNOWN
        context.inheritance_confidence = result.get("confidence", "low")

        return StepResult(
            step_name="inheritance_pattern",
            classification=EligibilityClassification.ELIGIBLE,
            summary=result.get("evidence_summary", ""),
            reasoning=result.get("reasoning", ""),
            data_used=raw_data,
            metadata={
                "inheritance_pattern": pattern_str,
                "confidence": result.get("confidence"),
                "associated_diseases": result.get("associated_diseases", []),
                "also_associated_with_other_patterns": result.get("also_associated_with_other_patterns", False),
                "other_patterns_note": result.get("other_patterns_note", ""),
                "warnings": result.get("warnings", []),
                "_tool_call_log": result.get("_tool_call_log", []),
            },
            token_usage=usage,
        )

    # ─────────────────────────────────────────────────────────────
    # Step 2: Pathomechanism
    # ─────────────────────────────────────────────────────────────

    def run_pathomechanism(self, hgvs: str, context: AssessmentContext) -> StepResult:
        """Execute Step 2: Pathomechanism and Haploinsufficiency Assessment."""
        if "mutalyzer" not in context.raw_cache:
            context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
        mutalyzer_data = context.raw_cache.get("mutalyzer") or {}
        gene = mutalyzer_data.get("gene_id")
        norm_hgvs = mutalyzer_data.get("normalized", hgvs)

        if self.llm_only:
            raw_data = {}
            mutalyzer_data = {"normalized": norm_hgvs, "gene_id": gene}
            user_msg = f"""Please assess pathomechanism and haploinsufficiency for this variant (Step 2).

GENE: {gene}
HGVS: {norm_hgvs}

Apply Step 2 criteria and return your JSON assessment.
"""
        else:
            if "clinvar" not in context.raw_cache and "clingen" not in context.raw_cache:
                clinical_context = fetch_clinical_context(hgvs)
                clingen_data = clinical_context.get("clingen")
                clinvar_data = clinical_context.get("clinvar")
                context.raw_cache["clingen"] = clingen_data
                context.raw_cache["clinvar"] = clinvar_data

            clinvar_data = context.raw_cache.get("clinvar")
            clingen_data = context.raw_cache.get("clingen")

            gnomad_data = search_gnomad(gene, hgvsc=norm_hgvs)

            patho_str = " OR ".join(["(loss of function)", "(gain of function)", "(dominant negative)", "(loss-of-function)", "(gain-of-function)", "(dominant-negative)"])
            search_query = f"{gene} AND ({patho_str})"
            equiv = mutalyzer_data.get("equivalent_descriptions") or []
            synonyms = [norm_hgvs] + equiv
            if clinvar_data and clinvar_data.get("protein_change"):
                synonyms.append(clinvar_data.get("protein_change"))
            synonyms = list(set([name.split(":")[-1] for name in synonyms if name]))
            if len(synonyms) > 0:
                name_str = " OR ".join([f"({s})" for s in synonyms])
                search_query += f" AND ({name_str})"
            pubmed_results = self._get_pubmed_pmc_results(search_query)

            raw_data = {
                "clingen": clingen_data,
                "gnomad_sample": gnomad_data,
                "clinvar": clinvar_data,
                "pubmed": pubmed_results,
            }

            inheritance_info = f"Inheritance pattern: {context.inheritance_pattern.value if context.inheritance_pattern else 'unknown'}"

            user_msg = f"""Please assess pathomechanism and haploinsufficiency for this variant (Step 2).

GENE: {gene}
HGVS: {norm_hgvs}
{inheritance_info or ""}

CLINVAR DATA:
{clinvar_data}

CLINGEN DOSAGE SENSITIVITY:
{clingen_data}

GNOMAD SUMMARY:
{gnomad_data}

PUBMED SEARCH RESULTS:
{pubmed_results}

Apply Step 2 criteria and return your JSON assessment.
IMPORTANT: For at least one of the PubMed IDs, MIM numbers, or URLs mentioned in the context above, use the `fetch_and_extract` tool
to get a summary of the full text in relation to a research question of interest. For this, you should pick the source that you think is most relevant.
"""

        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["pathomechanism"],
            user_message=user_msg,
            expect_json=True,
            tools=[FETCH_AND_EXTRACT_TOOL] if not self.llm_only else None,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        if "_parse_error" in result:
            return StepResult(
                step_name="pathomechanism",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="LLM response could not be parsed.",
                reasoning=result.get("_raw", ""),
                data_used=raw_data,
                error=result.get("_parse_error"),
                token_usage=usage,
            )

        pmech_str = result.get("pathomechanism", "unknown")
        try:
            context.pathomechanism = Pathomechanism(pmech_str)
        except ValueError:
            context.pathomechanism = Pathomechanism.UNKNOWN

        context.is_haploinsufficient = result.get("is_haploinsufficient")
        context.haploinsufficiency_evidence = result.get("haploinsufficiency_conclusion", "")

        return StepResult(
            step_name="pathomechanism",
            classification=EligibilityClassification.ELIGIBLE,
            summary=(
                f"Pathomechanism: {pmech_str}. "
                f"Haploinsufficiency: {result.get('is_haploinsufficient')}."
            ),
            reasoning=result.get("pathomechanism_reasoning", "") + "\n\n" + result.get("haploinsufficiency_conclusion", ""),
            data_used=raw_data,
            metadata={
                "pathomechanism": pmech_str,
                "pathomechanism_confidence": result.get("pathomechanism_confidence"),
                "is_haploinsufficient": result.get("is_haploinsufficient"),
                "haploinsufficiency_evidence": result.get("haploinsufficiency_evidence"),
                "warnings": result.get("warnings", []),
                "_tool_call_log": result.get("_tool_call_log", []),
            },
            token_usage=usage,
        )

    # ─────────────────────────────────────────────────────────────
    # Step 3: Splicing Effects
    # ─────────────────────────────────────────────────────────────

    def run_splicing_effects(self, hgvs: str, context: AssessmentContext) -> StepResult:
        """Execute Step 3: Splicing Effects Evaluation."""
        if "mutalyzer" not in context.raw_cache:
            context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
        mutalyzer_data = context.raw_cache.get("mutalyzer") or {}
        gene = mutalyzer_data.get("gene_id")
        norm_hgvs = mutalyzer_data.get("normalized", hgvs)

        if self.llm_only:
            raw_data = {}
            mutalyzer_data = {"normalized": norm_hgvs, "gene_id": gene}
            user_msg = f"""Please evaluate splicing effects for this variant (Step 3 of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}

Important: Only RNAseq, qPCR, or cDNA from patient-derived cells counts as sufficient 
functional evidence. In silico predictions are NOT sufficient.

Apply Step 3 criteria (Table 3) and return your JSON assessment.
"""
        else:
            # reuse data from previous steps
            if "clinvar" not in context.raw_cache:
                clinical_context = fetch_clinical_context(hgvs)
                clinvar_data = clinical_context.get("clinvar")
                context.raw_cache["clinvar"] = clinvar_data
            clinvar_data = context.raw_cache.get("clinvar")

            if "ensembl_vep" not in context.raw_cache:
                context.raw_cache["ensembl_vep"] = search_ensembl_vep(norm_hgvs)
            vep_data = context.raw_cache.get("ensembl_vep")

            if "aso_check_pubmed" not in context.raw_cache:
                aso_check_result = self.run_aso_check(norm_hgvs, context)
                if isinstance(aso_check_result, dict):
                    context.raw_cache["aso_check_pubmed"] = aso_check_result
            aso_check = context.raw_cache.get("aso_check_pubmed")

            # fetch data specific to this step
            transcript_ctx = fetch_transcript_context(norm_hgvs)
            context.raw_cache["transcript_context"] = transcript_ctx

            search_query = f"{gene} AND (splicing)"
            equiv = mutalyzer_data.get("equivalent_descriptions") or []
            synonyms = [norm_hgvs] + equiv
            if clinvar_data and clinvar_data.get("protein_change"):
                synonyms.append(clinvar_data.get("protein_change"))
            synonyms = list(set([name.split(":")[-1] for name in synonyms if name]))
            if len(synonyms) > 0:
                name_str = " OR ".join([f"({s})" for s in synonyms])
                search_query += f" AND ({name_str})"
            pubmed_results = self._get_pubmed_pmc_results(search_query)

            raw_data = {
                "vep": vep_data,
                "transcript_context": transcript_ctx,
                "clinvar": clinvar_data,
                "pubmed": pubmed_results,
            }

            cached_info = ""
            if context.inheritance_pattern:
                cached_info += f"Inheritance pattern: {context.inheritance_pattern.value}\n"
            if context.pathomechanism:
                cached_info += f"Pathomechanism: {context.pathomechanism.value}\n"
            if len(cached_info) > 0:
                cached_info = "\n" + cached_info + "\n"

            if aso_check:
                cached_info += f"Summary of existing ASO literature: {aso_check}\n"

            user_msg = f"""Please evaluate splicing effects for this variant (Step 3 of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}
{cached_info}
TRANSCRIPT INFORMATION:
{transcript_ctx}

ENSEMBL VEP ANNOTATION:
{vep_data}

CLINVAR DATA:
{clinvar_data}

PUBMED SEARCH RESULTS:
{pubmed_results}

Important: Only RNAseq, qPCR, or cDNA from patient-derived cells counts as sufficient 
functional evidence. In silico predictions are NOT sufficient.

Apply Step 3 criteria (Table 3) and return your JSON assessment.
"""

        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["splicing_effects"],
            user_message=user_msg,
            expect_json=True,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        if "_parse_error" in result:
            return StepResult(
                step_name="splicing_effects",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="LLM response could not be parsed.",
                reasoning=result.get("_raw", ""),
                data_used=raw_data,
                error=result.get("_parse_error"),
                token_usage=usage,
            )

        context.has_splicing_evidence = result.get("has_splicing_evidence")
        context.splicing_effect_type = result.get("splicing_effect_type")
        context.canonical_splicing_destroyed = result.get("canonical_splicing_destroyed")

        classification_str = result.get("splice_correction_classification", "unable_to_assess")
        try:
            classification = EligibilityClassification(classification_str)
        except ValueError:
            classification = EligibilityClassification.UNABLE_TO_ASSESS

        return StepResult(
            step_name="splicing_effects",
            classification=classification,
            summary=(
                f"Splice correction: {classification_str}. "
                f"Splicing evidence: {result.get('has_splicing_evidence')}. "
                f"Effect type: {result.get('splicing_effect_type')}."
            ),
            reasoning=result.get("splice_correction_reasoning", ""),
            data_used=raw_data,
            metadata={
                "has_splicing_evidence": result.get("has_splicing_evidence"),
                "evidence_source": result.get("evidence_source"),
                "splicing_effect_type": result.get("splicing_effect_type"),
                "canonical_splicing_destroyed": result.get("canonical_splicing_destroyed"),
                "wildtype_transcript_detectable": result.get("wildtype_transcript_detectable"),
                "variant_distance_from_splice_site_bp": result.get("variant_distance_from_splice_site_bp"),
                "intronic_or_exonic": "intronic" if mutalyzer_data.get("intronic") else "exonic",
                "aso_evidence_found": result.get("aso_evidence_found", False),
                "aso_evidence_description": result.get("aso_evidence_description", ""),
                "warnings": result.get("warnings", []),
                "_tool_call_log": result.get("_tool_call_log", []),
            },
            token_usage=usage,
        )

    # ─────────────────────────────────────────────────────────────
    # Routing helpers
    # ─────────────────────────────────────────────────────────────

    def _route_to_sections(self, context: AssessmentContext) -> dict[str, bool]:
        """Determine which Sections (A, B, C) to evaluate based on context."""
        sections = {
            "exon_skipping": False,
            "knockdown": False,
            "wt_upregulation": False,
        }

        if context.is_cnv_gain:
            sections["knockdown"] = True
            return sections

        if context.is_cnv_loss:
            sections["wt_upregulation"] = True
            return sections

        inheritance = context.inheritance_pattern
        pathomech = context.pathomechanism
        is_hi = context.is_haploinsufficient

        if inheritance is None or inheritance == InheritancePattern.UNKNOWN:
            sections["exon_skipping"] = True
            sections["knockdown"] = True
            sections["wt_upregulation"] = True
            return sections

        if pathomech is None or pathomech == Pathomechanism.UNKNOWN or pathomech == Pathomechanism.COMPLEX:
            sections["exon_skipping"] = True
            sections["knockdown"] = True
            return sections

        is_dominant = inheritance in (
            InheritancePattern.AUTOSOMAL_DOMINANT,
            InheritancePattern.X_LINKED_DOMINANT,
        )

        if pathomech == Pathomechanism.LOSS_OF_FUNCTION:
            sections["exon_skipping"] = True
            if is_dominant:
                if is_hi or is_hi is None:
                    sections["wt_upregulation"] = True

        elif pathomech == Pathomechanism.GAIN_OF_FUNCTION:
            sections["exon_skipping"] = True
            sections["knockdown"] = True

        elif pathomech == Pathomechanism.DOMINANT_NEGATIVE:
            if is_dominant:
                sections["exon_skipping"] = True
                sections["knockdown"] = True
            else:
                sections["exon_skipping"] = True
                sections["knockdown"] = True
        
        # protocol describes assessing variants in exons
        if context.intronic_or_exonic == "intronic":
            sections["exon_skipping"] = False
        
        # override any info above with existing aso studies
        if context.existing_aso_success and context.existing_aso_sufficient:
            approach = context.existing_aso_type
            if approach in sections:
                sections[approach] = True

        return sections

    def _explain_routing(self, context: AssessmentContext) -> str:
        """Generate a human-readable explanation of the routing decision."""
        sections = self._route_to_sections(context)
        selected = [k for k, v in sections.items() if v]

        lines = [
            f"Routing based on:",
            f"  - Inheritance: {context.inheritance_pattern.value if context.inheritance_pattern else 'unknown'}",
            f"  - Pathomechanism: {context.pathomechanism.value if context.pathomechanism else 'unknown'}",
            f"  - Haploinsufficiency: {context.is_haploinsufficient}",
            f"  - CNV Gain: {context.is_cnv_gain}, CNV Loss: {context.is_cnv_loss}",
            f"  - Type: {context.intronic_or_exonic}"
            "",
            f"Sections to evaluate: {', '.join(selected) if selected else 'none (check Step 0-2 results)'}",
        ]

        if "exon_skipping" in selected:
            lines.append("  → Section A: Canonical exon skipping")
        if "knockdown" in selected:
            lines.append("  → Section B: Transcript knockdown")
        if "wt_upregulation" in selected:
            lines.append("  → Section C: WT allele upregulation")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # Section A: Exon Skipping
    # ─────────────────────────────────────────────────────────────

    def assess_exon_skipping(self, hgvs: str, context: AssessmentContext) -> StepResult:
        """Execute Section A: Canonical Exon Skipping assessment."""
        if "mutalyzer" not in context.raw_cache:
            context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
        mutalyzer_data = context.raw_cache.get("mutalyzer") or {}
        gene = mutalyzer_data.get("gene_id")
        norm_hgvs = mutalyzer_data.get("normalized", hgvs)

        if self.llm_only:
            raw_data = {}
            mutalyzer_data = {"normalized": norm_hgvs, "gene_id": gene}
            user_msg = f"""Please evaluate canonical exon skipping eligibility (Section A of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}

Apply Section A criteria (Table 4) step-by-step and return your JSON assessment.
Remember: assessment is at the EXON level, not the variant level.
"""
        else:
            if "clinvar" not in context.raw_cache:
                clinical_context = fetch_clinical_context(hgvs)
                clinvar_data = clinical_context.get("clinvar")
                context.raw_cache["clinvar"] = clinvar_data
            clinvar_data = context.raw_cache.get("clinvar")

            if "aso_check_pubmed" not in context.raw_cache:
                self.run_aso_check(hgvs, context)
            aso_lit = context.raw_cache.get("aso_check_pubmed", {}).get("exon_level_papers", [])

            transcript_ctx = fetch_transcript_context(norm_hgvs)
            protein_ctx = fetch_protein_context(norm_hgvs)

            raw_data = {
                "transcript_context": transcript_ctx,
                "protein_context": protein_ctx,
                "aso_literature": aso_lit,
            }

            tx_summary = "Transcript context unavailable."
            if transcript_ctx and isinstance(transcript_ctx, dict):
                tx_summary = (
                    f"Transcript ID: {transcript_ctx.get('transcript_id')}\n"
                    f"Chromosome: {transcript_ctx.get('chromosome')}\n"
                    f"Location: {transcript_ctx.get('location')}\n"
                    f"Total exons in transcript: {transcript_ctx.get('transcript_exons')}\n"
                )
                if transcript_ctx.get("location") == "intronic":
                    tx_summary += f"Offset from nearest coding position: {transcript_ctx.get('offset')}\n"
                    tx_summary += f"Nearest exon number: {transcript_ctx.get('exon_number')}\n"
                else:
                    tx_summary += f"Exon number: {transcript_ctx.get('exon_number')}\n"
                tx_summary += "Flanking exons (with sequences):\n" + json.dumps(transcript_ctx.get("flanking_exons", []), indent=2)

            protein_summary = "Protein domain context unavailable."
            if protein_ctx and isinstance(protein_ctx, dict):
                protein_summary = (
                    f"Gene: {protein_ctx.get('gene_id')}\n"
                    f"UniProt ID: {protein_ctx.get('uniprot_id')}\n"
                    f"Protein length: {protein_ctx.get('protein_aa_length')} aa\n"
                    f"Exon AA range: {protein_ctx.get('exon_aa_range')}\n"
                    f"Domains overlapping exon:\n"
                    + json.dumps(protein_ctx.get("domains", []), indent=2)
                )

            cached_info = ""
            if context.inheritance_pattern:
                cached_info += f"Inheritance pattern: {context.inheritance_pattern.value}\n"
            if context.pathomechanism:
                cached_info += f"Pathomechanism: {context.pathomechanism.value}\n"
            if context.is_haploinsufficient:
                cached_info += f"Haploinsufficient gene: {context.is_haploinsufficient}\n"
            if len(cached_info) > 0:
                cached_info = "\n" + cached_info + "\n"

            user_msg = f"""Please evaluate canonical exon skipping eligibility (Section A of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}

{cached_info}

TRANSCRIPT CONTEXT:
{tx_summary}

PROTEIN DOMAIN CONTEXT:
{protein_summary}

CLINVAR DATA:
{clinvar_data}

PUBMED SEARCH RESULTS:
{aso_lit}

Apply Section A criteria (Table 4) step-by-step and return your JSON assessment.
Remember: assessment is at the EXON level, not just the variant level.
"""

        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["exon_skipping"],
            user_message=user_msg,
            expect_json=True,
            tools=[FETCH_AND_EXTRACT_TOOL] if not self.llm_only else None,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        if "_parse_error" in result:
            return StepResult(
                step_name="exon_skipping",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="LLM response could not be parsed.",
                reasoning=result.get("_raw", ""),
                data_used=raw_data,
                error=result.get("_parse_error"),
                token_usage=usage,
            )

        classification_str = result.get("exon_skipping_classification", "unable_to_assess")
        try:
            classification = EligibilityClassification(classification_str)
        except ValueError:
            classification = EligibilityClassification.UNABLE_TO_ASSESS

        return StepResult(
            step_name="exon_skipping",
            classification=classification,
            summary=(
                f"Exon skipping: {classification_str}. "
                f"Exon {result.get('exon_number')} of {result.get('total_exons')}, "
                f"frame: {result.get('exon_frame')}."
            ),
            reasoning=result.get("exon_skipping_reasoning", ""),
            data_used=raw_data,
            metadata={
                "exon_number": result.get("exon_number"),
                "total_exons": result.get("total_exons"),
                "exon_frame": result.get("exon_frame"),
                "exon_phase": result.get("exon_phase"),
                "forms_stop_codon_on_skipping": result.get("forms_stop_codon_on_skipping"),
                "exon_size_percent_coding": result.get("exon_size_percent_coding"),
                "natural_skipping_evidence": result.get("natural_skipping_evidence"),
                "functional_domains": result.get("functional_domains", []),
                "domain_assessment": result.get("domain_assessment", ""),
                "allele_specific_required": result.get("allele_specific_required", False),
                "aso_evidence_found": result.get("aso_evidence_found", False),
                "aso_evidence_description": result.get("aso_evidence_description", ""),
                "warnings": result.get("warnings", []),
            },
            token_usage=usage,
        )

    # ─────────────────────────────────────────────────────────────
    # Section B: Knockdown
    # ─────────────────────────────────────────────────────────────

    def assess_knockdown(self, hgvs: str, context: AssessmentContext) -> StepResult:
        """Execute Section B: Transcript Knockdown assessment."""
        if "mutalyzer" not in context.raw_cache:
            context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
        mutalyzer_data = context.raw_cache.get("mutalyzer") or {}
        gene = mutalyzer_data.get("gene_id")
        norm_hgvs = mutalyzer_data.get("normalized", hgvs)

        if self.llm_only:
            raw_data = {}
            mutalyzer_data = {"normalized": norm_hgvs, "gene_id": gene}
            user_msg = f"""Please evaluate transcript knockdown eligibility (Section B of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}

Apply Section B criteria (Table 5) and return your JSON assessment.
"""
        else:
            if not context.raw_cache.get("clingen"):
                clinical_context = fetch_clinical_context(hgvs)
                clingen_data = clinical_context.get("clingen")
                context.raw_cache["clingen"] = clingen_data
            clingen_data = context.raw_cache.get("clingen")

            gnomad_data = search_gnomad(gene, hgvsc=norm_hgvs)

            raw_data = {
                "clingen": clingen_data,
                "gnomad_sample": gnomad_data,
            }

            cached_info = ""
            if context.inheritance_pattern:
                cached_info += f"Inheritance pattern: {context.inheritance_pattern.value}\n"
            if context.pathomechanism:
                cached_info += f"Pathomechanism: {context.pathomechanism.value}\n"
            if context.is_haploinsufficient:
                cached_info += f"Haploinsufficient gene: {context.is_haploinsufficient}\n"
            if len(cached_info) > 0:
                cached_info = "\n" + cached_info + "\n"

            user_msg = f"""Please evaluate transcript knockdown eligibility (Section B of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}
{cached_info}
CLINGEN DOSAGE SENSITIVITY:
{clingen_data}

GNOMAD SUMMARY:
{gnomad_data}

Apply Section B criteria (Table 5) and return your JSON assessment.
"""
        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["knockdown"],
            user_message=user_msg,
            expect_json=True,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        if "_parse_error" in result:
            return StepResult(
                step_name="knockdown",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="LLM response could not be parsed.",
                reasoning=result.get("_raw", ""),
                data_used=raw_data,
                error=result.get("_parse_error"),
                token_usage=usage,
            )

        classification_str = result.get("knockdown_classification", "unable_to_assess")
        try:
            classification = EligibilityClassification(classification_str)
        except ValueError:
            classification = EligibilityClassification.UNABLE_TO_ASSESS

        return StepResult(
            step_name="knockdown",
            classification=classification,
            summary=(
                f"Knockdown: {classification_str}. "
                f"HI conclusion: {result.get('haploinsufficiency_conclusion')}."
            ),
            reasoning=result.get("knockdown_reasoning", ""),
            data_used=raw_data,
            metadata={
                "pathomechanism_eligible": result.get("pathomechanism_eligible"),
                "pli_score": result.get("pli_score"),
                "loeuf_score": result.get("loeuf_score"),
                "clingen_hi_score": result.get("clingen_hi_score"),
                "haploinsufficiency_conclusion": result.get("haploinsufficiency_conclusion"),
                "allele_specific_recommended": result.get("allele_specific_recommended", False),
                "allele_specific_reason": result.get("allele_specific_reason", ""),
                "aso_evidence_found": result.get("aso_evidence_found", False),
                "aso_evidence_description": result.get("aso_evidence_description", ""),
                "warnings": result.get("warnings", []),
            },
            token_usage=usage,
        )

    # ─────────────────────────────────────────────────────────────
    # Section C: WT Upregulation
    # ─────────────────────────────────────────────────────────────

    def assess_wt_upregulation(self, hgvs: str, context: AssessmentContext) -> StepResult:
        """Execute Section C: Wildtype Allele Upregulation assessment."""
        if "mutalyzer" not in context.raw_cache:
            context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
        mutalyzer_data = context.raw_cache.get("mutalyzer") or {}
        gene = mutalyzer_data.get("gene_id")
        norm_hgvs = mutalyzer_data.get("normalized", hgvs)

        if self.llm_only:
            raw_data = {}
            mutalyzer_data = {"normalized": norm_hgvs, "gene_id": gene}
            user_msg = f"""Please evaluate wildtype allele upregulation strategies (Section C of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}

Apply Section C criteria and return your JSON assessment.
Note: This section does NOT classify as likely/unlikely eligible - only "eligible" if an upregulation strategy has already been well established in the literature.
"""
        else:
            if "aso_check_pubmed" not in context.raw_cache:
                self.run_aso_check(hgvs, context)
            aso_lit = context.raw_cache.get("aso_check_pubmed")

            if "ensembl_vep" not in context.raw_cache:
                context.raw_cache["ensembl_vep"] = search_ensembl_vep(norm_hgvs)
            vep_data = context.raw_cache.get("ensembl_vep")

            alt_splice_data = search_alt_splicing_events(gene)

            tango_query = f"{gene} AND ((poison exon) OR (TANGO) OR (antisense transcript) OR (uORF))"
            tango_papers = self._get_pubmed_pmc_results(tango_query)

            raw_data = {
                "ensembl_vep": vep_data,
                "alt_splice_data": alt_splice_data,
                "tango_specific_literature": tango_papers,
                "general_aso_literature": aso_lit,
            }

            cached_info = ""
            if context.inheritance_pattern:
                cached_info += f"Inheritance pattern: {context.inheritance_pattern.value}\n"
            if context.pathomechanism:
                cached_info += f"Pathomechanism: {context.pathomechanism.value}\n"
            if context.is_haploinsufficient:
                cached_info += f"Haploinsufficient gene: {context.is_haploinsufficient}\n"
            if context.haploinsufficiency_evidence:
                cached_info += f"Haploinsufficiency evidence: {context.haploinsufficiency_evidence}\n"
            if len(cached_info) > 0:
                cached_info = "\n" + cached_info + "\n"

            user_msg = f"""Please evaluate wildtype allele upregulation strategies (Section C of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}
{cached_info}
KEY REFERENCES TO CONSIDER:
- Mittal et al. (2022): curated lists of poison exons, NATs, and uORFs per gene
- Lim et al. (2020): poison exon data
- Felker et al. (2023): poison exon annotations

ENSEMBL VEP:
{vep_data}

GENERAL ASO LITERATURE:
{aso_lit}

TANGO/POISON EXON LITERATURE:
{tango_papers}

ALTERNATIVE SPLICING EVENTS FROM KEY REFERENCES:
{alt_splice_data}

Apply Section C criteria and return your JSON assessment.
Note: This section does NOT classify as likely/unlikely eligible - only "eligible" if an upregulation strategy has already been well established in the literature.
"""

        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["wt_upregulation"],
            user_message=user_msg,
            expect_json=True,
            tools=[FETCH_AND_EXTRACT_TOOL] if not self.llm_only else None,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        if "_parse_error" in result:
            return StepResult(
                step_name="wt_upregulation",
                classification=EligibilityClassification.UNABLE_TO_ASSESS,
                summary="LLM response could not be parsed.",
                reasoning=result.get("_raw", ""),
                data_used=raw_data,
                error=result.get("_parse_error"),
                token_usage=usage,
            )

        wt_class_str = result.get("wt_upregulation_classification", "no_strategy_identified")
        if wt_class_str == "eligible":
            classification = EligibilityClassification.ELIGIBLE
        elif wt_class_str == "strategy_available_needs_validation":
            classification = EligibilityClassification.LIKELY_ELIGIBLE
        elif wt_class_str == "not_applicable":
            classification = EligibilityClassification.NOT_APPLICABLE
        else:
            classification = EligibilityClassification.UNABLE_TO_ASSESS

        return StepResult(
            step_name="wt_upregulation",
            classification=classification,
            summary=result.get("wt_upregulation_summary", ""),
            reasoning=result.get("wt_upregulation_summary", ""),
            data_used=raw_data,
            metadata={
                "applicable": result.get("applicable"),
                "poison_exon_identified": result.get("poison_exon_identified"),
                "poison_exon_details": result.get("poison_exon_details", ""),
                "nat_identified": result.get("nat_identified"),
                "nat_details": result.get("nat_details", ""),
                "uorf_identified": result.get("uorf_identified"),
                "uorf_details": result.get("uorf_details", ""),
                "established_wt_upregulation_strategy": result.get("established_wt_upregulation_strategy"),
                "recommended_next_steps": result.get("recommended_next_steps", []),
                "warnings": result.get("warnings", []),
                "_tool_call_log": result.get("_tool_call_log", []),
            },
            token_usage=usage,
        )

    def _get_pubmed_pmc_results(self, search_term: str) -> list[dict[str, Any]]:
        """Search PubMed and PMC for a term, deduplicating by PMID."""
        combined_results = []
        seen_pmids = set()

        for db in ["pubmed", "pmc"]:
            ncbi_result = search_ncbi(
                database=db,
                search_term=search_term,
                max_results=10,
            )
            for result in (ncbi_result.get("results") or []):
                if result.get("pmid") and result["pmid"] not in seen_pmids:
                    seen_pmids.add(result.get("pmid"))
                    combined_results.append(result)
        return combined_results



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
Do not change the classification labels in the step results.
"""

        result, usage = call_llm(
            system_prompt=SYSTEM_PROMPTS["final_report"],
            user_message=user_msg,
            expect_json=True,
            model=self.model_name,
            use_web_search=self.use_web_search,
        )

        # ── Extract classifications from step results ─────────────────
        def get_classification(step_name: str) -> EligibilityClassification:
            # first, check formatted llm summary
            assessments = result.get("strategy_assessments", {})
            if assessments.get(step_name, {}).get("classification"):
                try:
                    return EligibilityClassification(assessments[step_name]["classification"].replace(" ", "_"))
                except Exception as e:
                    print(e)
                    pass
            # if report failed to generate, use raw step result
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

    def _safe_run_step(self, method_name: str, hgvs: str, context: AssessmentContext) -> StepResult:
        """
        Run a step method with full error handling.
        Returns an UNABLE_TO_ASSESS result if the step crashes.
        """
        step_fn = getattr(self, method_name)
        try:
            return step_fn(hgvs, context)
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"  ✗ Step crashed: {e}")
            return StepResult(
                step_name=method_name,
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