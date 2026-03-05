"""
Step 0 - Variant Check

Validates the HGVS variant description, normalizes it via Mutalyzer,
and determines whether the variant type is eligible for assessment
under the N1C VARIANT guidelines.

Applicable: SNVs, small indels, single/multi-exon deletions/duplications,
            single gene deletions/duplications
Excluded:   Non-coding gene variants, multi-gene CNVs, imprinting defects,
            structural rearrangements, aneuploidies, mtDNA variants
"""
import json
from typing import Any, Dict
from functools import cache
from aso_workflow.data_model import (
    ASOAssessmentReport, StepResult, AssessmentContext, EligibilityClassification, InheritancePattern, Pathomechanism,
)
from aso_workflow.utils.apis import *
from aso_workflow.utils.llm import call_llm, FETCH_AND_EXTRACT_TOOL
from aso_workflow.utils.tasks import fetch_protein_context, fetch_transcript_context, fetch_clinical_context
from aso_workflow.prompts import SYSTEM_PROMPTS



def run_variant_check(hgvs: str, context: AssessmentContext, model_name: str | None = None) -> StepResult:
    """
    Execute Step 0: Variant Check.

    Args:
        hgvs: Input HGVS string
        context: Shared assessment context (will be updated in place)

    Returns:
        StepResult with classification and metadata
    """
    # ── Fetch phase ──────────────────────────────────────────────
    mutalyzer_data = search_mutalyzer(hgvs)

    raw_data = {
        "input_hgvs": hgvs,
        "mutalyzer": mutalyzer_data,
        "mutalyzer_error": mutalyzer_data.get("error"),
    }

    # ── Reason phase ─────────────────────────────────────────────
    user_msg = f"""Please evaluate this HGVS variant for Step 0 of the N1C VARIANT Guidelines.

INPUT HGVS: {hgvs}

MUTALYZER NORMALIZATION RESULT:
{mutalyzer_data}

Based on the normalization result and the input variant, apply Step 0 criteria and return 
your JSON assessment.
"""

    call_kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPTS["variant_check"],
        "user_message": user_msg,
        "expect_json": True,
    }
    if model_name is not None:
        call_kwargs["model"] = model_name
    result, usage = call_llm(**call_kwargs)

    # ── Parse and update context ──────────────────────────────────
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

    classification = EligibilityClassification(
        result.get("classification", "unable_to_assess")
    )

    # Update shared context
    context.hgvs_normalized = result.get("hgvs_normalized") or hgvs
    context.gene_id = result.get("gene_id")
    context.refseq_id = result.get("refseq_id")
    context.variant_valid = result.get("variant_valid", False)
    context.is_cnv_gain = result.get("is_cnv_gain", False)
    context.is_cnv_loss = result.get("is_cnv_loss", False)

    # Cache Mutalyzer result for downstream steps
    if mutalyzer_data:
        context.raw_cache["mutalyzer"] = mutalyzer_data

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
        },
        token_usage=usage,
    )


def run_aso_check(hgvs: str, context: AssessmentContext, add_llm_summary: bool = True, model_name: str | None = None) -> StepResult | dict[str, any]:
    """
    Execute Step 4: ASO Check.

    Args:
        hgvs: Normalized HGVS string
        context: Shared context
        add_llm_summary: Whether to use an LLM to summarize search results. If False, the raw search results will be returned.

    Returns:
        StepResult if add_llm_summary is True, otherwise Dict[str, Any] containing the raw search results
    """
    # mutalyzer context
    if not "mutalyzer" in context.raw_cache:
        context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
    mutalyzer_data = context.raw_cache.get("mutalyzer")
    norm_hgvs = mutalyzer_data.get("normalized")
    gene = mutalyzer_data.get("gene_id")

    # clinvar context
    if not "clinvar" in context.raw_cache:
        clinical_context = fetch_clinical_context(hgvs)
        clingen_data = clinical_context.get("clingen")
        clinvar_data = clinical_context.get("clinvar")
        context.raw_cache["clingen"] = clingen_data
        context.raw_cache["clinvar"] = clinvar_data
    clinvar_data = context.raw_cache.get("clinvar")
    clingen_data = context.raw_cache.get("clingen")


    gene_level_query = f"{gene} AND ((ASO) OR (AON) OR (antisense oligonucleotide) OR (AOs) OR (siRNA) OR (RNAi) OR (gapmer) or (knockdown))"

    # gene level ASO search
    gene_level_lit = _get_pubmed_pmc_results(gene_level_query)

    # exon level ASO search
    exon_level_query = gene_level_query[:]
    exon_level_lit = None
    if mutalyzer_data.get("nearest_exon"):
        exon_level_query += f" AND (exon {mutalyzer_data.get('nearest_exon')})"
        exon_level_lit = _get_pubmed_pmc_results(exon_level_query)

    # variant-level ASO search
    variant_level_query = gene_level_query[:]
    equiv = mutalyzer_data.get("equivalent_descriptions") or []
    synonyms = [norm_hgvs] + equiv
    if clinvar_data and clinvar_data.get("protein_change"):
        synonyms.append(clinvar_data.get("protein_change"))
    synonyms = list(set([name.split(':')[-1] for name in synonyms if name]))
    if len(synonyms) > 0:
        name_str = " OR ".join(synonyms)
        variant_level_query += f" AND ({name_str})"
    variant_level_lit = _get_pubmed_pmc_results(variant_level_query)
    
    raw_data = {
        "variant_level_papers": variant_level_lit,
        "gene_level_papers": gene_level_lit,
    }
    if exon_level_lit is not None:
        raw_data["exon_level_papers"] = exon_level_lit
    
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
    
    if not add_llm_summary:
        return raw_data

    call_kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPTS["aso_check"],
        "user_message": user_msg,
        "expect_json": True,
        "tools": [FETCH_AND_EXTRACT_TOOL],
    }
    if model_name is not None:
        call_kwargs["model"] = model_name
    result, usage = call_llm(**call_kwargs)

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

    # ── Update context ────────────────────────────────────────────
    context.raw_cache["aso_check_pubmed"] = raw_data
    try:
        context.existing_aso_found = result.get("aso_evidence_found", False)
        context.existing_aso_type = result.get("approach_used", "unknown")
        context.existing_aso_success = result.get("aso_success", False)
        is_sufficient = result.get("evidence_classification", "needs_further_evaluation") == "sufficient_functional_evidence"
        context.existing_aso_sufficient = is_sufficient
    except Exception:
        context.existing_aso_found = False
        context.existing_aso_type = "unknown"
        context.existing_aso_success = False
        context.existing_aso_sufficient = False
    
    return StepResult(
        step_name="aso_check",
        classification=EligibilityClassification.ELIGIBLE,  # can still continue through assessment, regardless of ASO success
        summary=result.get("summary", ""),
        reasoning=result.get("reasoning", ""),
        data_used=raw_data,
        metadata={
            "search_queries_used": [gene_level_query, exon_level_query, variant_level_query],
            "evidence_snippets": result.get("evidence_snippets", []),
            "aso_specificity": result.get("aso_specificity", "unknown"),
            "approach_used": result.get("approach_used", "unknown"),
            "aso_success": result.get("aso_success", False),
            "evidence_classification": result.get("evidence_classification", "unknown"),
            "warnings": result.get("warnings", []),
            "_tool_call_log": result.get("_tool_call_log", []),
        },
        token_usage=usage,
    )


def run_inheritance_pattern(hgvs: str, context: AssessmentContext, model_name: str | None = None) -> StepResult:
    """
    Execute Step 1: Inheritance Pattern Assessment.

    Args:
        hgvs: Normalized HGVS string
        context: Shared context (gene_id must be populated from Step 0)

    Returns:
        StepResult with inheritance classification
    """
    if not "mutalyzer" in context.raw_cache:
        context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
    mutalyzer_data = context.raw_cache.get("mutalyzer")
    norm_hgvs = mutalyzer_data.get("normalized")
    gene = mutalyzer_data.get("gene_id")

    if not "clinvar" in context.raw_cache:
        clinical_context = fetch_clinical_context(hgvs)
        clingen_data = clinical_context.get("clingen")
        clinvar_data = clinical_context.get("clinvar")
        context.raw_cache["clingen"] = clingen_data
        context.raw_cache["clinvar"] = clinvar_data
    clinvar_data = context.raw_cache.get("clinvar")
    clingen_data = context.raw_cache.get("clingen")

    # ── Fetch phase ──────────────────────────────────────────────
    clinical_context = fetch_clinical_context(hgvs)
    clinvar_data = clinical_context.get("clinvar")
    clingen_data = clinical_context.get("clingen")
    
    gnomad_data = search_gnomad(gene, hgvsc=norm_hgvs)
    # TODO: remove this, add it into the call_llm function
    web_results = search_serper(gene + " inheritance pattern")

    raw_data = {
        "clinvar": clinvar_data,
        "gnomad_summary": gnomad_data,
        "web_search": web_results,
    }

    # Cache for downstream use
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

    call_kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPTS["inheritance_pattern"],
        "user_message": user_msg,
        "expect_json": True,
        "tools": [FETCH_AND_EXTRACT_TOOL],
    }
    if model_name is not None:
        call_kwargs["model"] = model_name
    result, usage = call_llm(**call_kwargs)

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

    # ── Update context ────────────────────────────────────────────
    pattern_str = result.get("inheritance_pattern", "unknown")
    try:
        context.inheritance_pattern = InheritancePattern(pattern_str)
    except ValueError:
        context.inheritance_pattern = InheritancePattern.UNKNOWN
    context.inheritance_confidence = result.get("confidence", "low")

    return StepResult(
        step_name="inheritance_pattern",
        classification=EligibilityClassification.ELIGIBLE,  # Step 1 doesn't classify eligibility
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


def run_pathomechanism(hgvs: str, context: AssessmentContext, model_name: str | None = None) -> StepResult:
    """
    Execute Step 2: Pathomechanism and Haploinsufficiency Assessment.

    Args:
        hgvs: Normalized HGVS string
        context: Shared context

    Returns:
        StepResult with pathomechanism and haploinsufficiency data
    """
    if not "mutalyzer" in context.raw_cache:
        context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
    mutalyzer_data = context.raw_cache.get("mutalyzer")
    gene = mutalyzer_data.get("gene_id")
    norm_hgvs = mutalyzer_data.get("normalized")

    if not "clinvar" in context.raw_cache and not "clingen" in context.raw_cache:
        clinical_context = fetch_clinical_context(hgvs)
        clingen_data = clinical_context.get("clingen")
        clinvar_data = clinical_context.get("clinvar")

        context.raw_cache["clingen"] = clingen_data
        context.raw_cache["clinvar"] = clinvar_data
    
    clinvar_data = context.raw_cache.get("clinvar")
    clingen_data = context.raw_cache.get("clingen")


    # ── Fetch phase ──────────────────────────────────────────────
    gnomad_data = search_gnomad(gene, hgvsc=norm_hgvs)

    # Search for occurences in pubmed, accounting for variant synonyms
    patho_str = " OR ".join(["(loss of function)", "(gain of function)", "(dominant negative)", "(loss-of-function)", "(gain-of-function)", "(dominant-negative)"])
    search_query = f"{gene} AND ({patho_str})"
    equiv = mutalyzer_data.get("equivalent_descriptions") or []
    synonyms = [norm_hgvs] + equiv
    if clinvar_data and clinvar_data.get("protein_change"):
        synonyms.append(clinvar_data.get("protein_change"))
    synonyms = list(set([name.split(':')[-1] for name in synonyms if name]))
    if len(synonyms) > 0:
        name_str = " OR ".join([f"({s})" for s in synonyms])
        search_query += f" AND ({name_str})"
    pubmed_results = _get_pubmed_pmc_results(search_query)

    raw_data = {
        "clingen": clingen_data,
        "gnomad_sample": gnomad_data,
        "clinvar": clinvar_data,
        "pubmed": pubmed_results,
    }

    # ── Reason phase ─────────────────────────────────────────────
    # Extract inheritance info from context for better reasoning
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

    call_kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPTS["pathomechanism"],
        "user_message": user_msg,
        "expect_json": True,
        "tools": [FETCH_AND_EXTRACT_TOOL],
    }
    if model_name is not None:
        call_kwargs["model"] = model_name
    result, usage = call_llm(**call_kwargs)

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

    # ── Update context ────────────────────────────────────────────
    pmech_str = result.get("pathomechanism", "unknown")
    try:
        context.pathomechanism = Pathomechanism(pmech_str)
    except ValueError:
        context.pathomechanism = Pathomechanism.UNKNOWN

    context.is_haploinsufficient = result.get("is_haploinsufficient")
    context.haploinsufficiency_evidence = result.get("haploinsufficiency_conclusion", "")

    return StepResult(
        step_name="pathomechanism",
        classification=EligibilityClassification.ELIGIBLE,  # Step 2 doesn't classify eligibility
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

def run_splicing_effects(hgvs: str, context: AssessmentContext, model_name: str | None = None) -> StepResult:
    """
    Execute Step 3: Splicing Effects Evaluation.

    Args:
        hgvs: Normalized HGVS string
        context: Shared context

    Returns:
        StepResult with splice correction classification
    """
    # ── Fetch phase ──────────────────────────────────────────────
    if not "mutalyzer" in context.raw_cache:
        context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
    mutalyzer_data = context.raw_cache.get("mutalyzer")
    gene = mutalyzer_data.get("gene_id")
    norm_hgvs = mutalyzer_data.get("normalized")

    if not "clinvar" in context.raw_cache:
        clinical_context = fetch_clinical_context(hgvs)
        clinvar_data = clinical_context.get("clinvar")
        context.raw_cache["clinvar"] = clinvar_data
    clinvar_data = context.raw_cache.get("clinvar")

    if not "ensembl_vep" in context.raw_cache:
        context.raw_cache["ensembl_vep"] = search_ensembl_vep(norm_hgvs)
    vep_data = context.raw_cache.get("ensembl_vep")
    aso_check = context.raw_cache.get("aso_check")
    
    raw_data = {
        "vep": vep_data,
        "aso_check": aso_check,
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

    # ── Reason phase ─────────────────────────────────────────────
    user_msg = f"""Please evaluate splicing effects for this variant (Step 3 of N1C Guidelines).

GENE: {gene}
HGVS: {norm_hgvs}
{cached_info}
ENSEMBL VEP ANNOTATION:
{vep_data}

POSITION INFO (Mutalyzer):
{mutalyzer_data}

CLINVAR DATA:
{clinvar_data}

Important: Only RNAseq, qPCR, or cDNA from patient-derived cells counts as sufficient 
functional evidence. In silico predictions are NOT sufficient.

Apply Step 3 criteria (Table 3) and return your JSON assessment.
"""

    call_kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPTS["splicing_effects"],
        "user_message": user_msg,
        "expect_json": True,
    }
    if model_name is not None:
        call_kwargs["model"] = model_name
    result, usage = call_llm(**call_kwargs)

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

    # ── Update context ────────────────────────────────────────────
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
            "intronic_or_exonic": result.get("intronic_or_exonic"),
            "aso_evidence_found": result.get("aso_evidence_found", False),
            "aso_evidence_description": result.get("aso_evidence_description", ""),
            "warnings": result.get("warnings", []),
            "_tool_call_log": result.get("_tool_call_log", []),
        },
        token_usage=usage,
    )


def route_to_sections(context: AssessmentContext) -> dict[str, bool]:
    """
    Determine which Sections (A, B, C) to evaluate based on context.

    Implements Figure 6 decision table from the N1C VARIANT guidelines:

    | Inheritance       | GoF              | LoF              | DN               |
    |-------------------|------------------|------------------|------------------|
    | Autosomal Rec.    | A (exon skip)    | A (exon skip)    | N/A              |
    |                   | B (knockdown)    |                  |                  |
    | Autosomal Dom.    | A (exon skip*)   | A (exon skip*)   | A (exon skip*)   |
    |                   | B (knockdown*)   | C (WT upregul.)  | B (knockdown*)   |
    | X-linked Rec.     | A (exon skip)    | A (exon skip)    | N/A              |
    |                   | B (knockdown)    |                  |                  |
    | X-linked Dom.     | A (exon skip*)   | A (exon skip*)   | A (exon skip*)   |
    |                   | B (knockdown*)   | C (WT upregul.)**| B (knockdown*)   |

    * allele-specific ASO considerations
    ** only for individuals with two X chromosomes

    CNV special cases (from Step 0):
    - CNV Gain → Section B (knockdown)
    - CNV Loss (one WT copy) → Section C (WT upregulation)

    Returns:
        Dict mapping section names to True/False
    """
    sections = {
        "exon_skipping": False,
        "knockdown": False,
        "wt_upregulation": False,
    }

    # currently not using ASO check to skip sections, but as supplemental info.

    # CNV special cases override normal routing
    if context.is_cnv_gain:
        sections["knockdown"] = True
        return sections

    if context.is_cnv_loss:
        sections["wt_upregulation"] = True
        return sections

    inheritance = context.inheritance_pattern
    pathomech = context.pathomechanism
    is_hi = context.is_haploinsufficient

    # Handle unknown states gracefully
    if inheritance is None or inheritance == InheritancePattern.UNKNOWN:
        # Can't route without inheritance — assess all applicable sections
        sections["exon_skipping"] = True
        sections["knockdown"] = True
        sections["wt_upregulation"] = True
        return sections

    if pathomech is None or pathomech == Pathomechanism.UNKNOWN or pathomech == Pathomechanism.COMPLEX:
        # Can't route without pathomechanism — try all
        sections["exon_skipping"] = True
        sections["knockdown"] = True
        return sections

    is_dominant = inheritance in (
        InheritancePattern.AUTOSOMAL_DOMINANT,
        InheritancePattern.X_LINKED_DOMINANT,
    )

    if pathomech == Pathomechanism.LOSS_OF_FUNCTION:
        sections["exon_skipping"] = True  # Exon skipping always considered for LoF
        if is_dominant:
            # AD/XLD LoF: consider WT upregulation if gene is haploinsufficient
            if is_hi or is_hi is None:  # include if uncertain
                sections["wt_upregulation"] = True

    elif pathomech == Pathomechanism.GAIN_OF_FUNCTION:
        sections["exon_skipping"] = True  # Exon skipping to destroy transcript
        sections["knockdown"] = True  # Knockdown

    elif pathomech == Pathomechanism.DOMINANT_NEGATIVE:
        if is_dominant:
            sections["exon_skipping"] = True  # Exon skipping (destroy mutant transcript)
            sections["knockdown"] = True  # Knockdown
        else:
            # DN should be dominant; if annotated as recessive, flag but still assess
            sections["exon_skipping"] = True
            sections["knockdown"] = True

    # Recessive GoF: both A and B
    # Recessive DN: N/A per guidelines, but include A and B for completeness

    return sections


def explain_routing(context: AssessmentContext) -> str:
    """
    Generate a human-readable explanation of the routing decision.
    """
    sections = route_to_sections(context)
    selected = [k for k, v in sections.items() if v]

    lines = [
        f"Routing based on:",
        f"  - Inheritance: {context.inheritance_pattern.value if context.inheritance_pattern else 'unknown'}",
        f"  - Pathomechanism: {context.pathomechanism.value if context.pathomechanism else 'unknown'}",
        f"  - Haploinsufficiency: {context.is_haploinsufficient}",
        f"  - CNV Gain: {context.is_cnv_gain}, CNV Loss: {context.is_cnv_loss}",
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



def assess_exon_skipping(hgvs: str, context: AssessmentContext, model_name: str | None = None) -> StepResult:
    """
    Execute Section A: Canonical Exon Skipping assessment.

    Args:
        hgvs: Normalized HGVS string
        context: Shared context

    Returns:
        StepResult with exon skipping classification
    """
    if not "mutalyzer" in context.raw_cache:
        context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
    mutalyzer_data = context.raw_cache.get("mutalyzer")
    gene = mutalyzer_data.get("gene_id")
    norm_hgvs = mutalyzer_data.get("normalized")

    if not "clinvar" in context.raw_cache:
        clinical_context = fetch_clinical_context(hgvs)
        clinvar_data = clinical_context.get("clinvar")
        context.raw_cache["clinvar"] = clinvar_data
    clinvar_data = context.raw_cache.get("clinvar")

    # ── Fetch phase ──────────────────────────────────────────────
    transcript_ctx = fetch_transcript_context(norm_hgvs)
    protein_ctx = fetch_protein_context(norm_hgvs)

    aso_lit = {"error": "Web search not available."}

    raw_data = {
        "transcript_context": transcript_ctx,
        "protein_context": protein_ctx,
        "aso_literature": aso_lit
    }

    # ── Build context summary for LLM ────────────────────────────
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
    
    # TODO: these are all necessary and should be re-ran if not present in the cache
    cached_info = ""
    if context.inheritance_pattern:
        cached_info += f"Inheritance pattern: {context.inheritance_pattern.value}\n"
    if context.pathomechanism:
        cached_info += f"Pathomechanism: {context.pathomechanism.value}\n"
    if context.is_haploinsufficient:
        cached_info += f"Haploinsufficient gene: {context.is_haploinsufficient}\n"
    if len(cached_info) > 0:
        cached_info = "\n" + cached_info + "\n"

    # ── Reason phase ─────────────────────────────────────────────
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

Apply Section A criteria (Table 4) step-by-step and return your JSON assessment.
Remember: assessment is at the EXON level, not the variant level.
"""

    call_kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPTS["exon_skipping"],
        "user_message": user_msg,
        "expect_json": True,
    }
    if model_name is not None:
        call_kwargs["model"] = model_name
    result, usage = call_llm(**call_kwargs)

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


def assess_knockdown(hgvs: str, context: AssessmentContext, model_name: str | None = None) -> StepResult:
    """
    Execute Section B: Transcript Knockdown assessment.

    Args:
        hgvs: Normalized HGVS string
        context: Shared context

    Returns:
        StepResult with knockdown eligibility classification
    """
    if not "mutalyzer" in context.raw_cache:
        context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
    mutalyzer_data = context.raw_cache.get("mutalyzer")
    gene = mutalyzer_data.get("gene_id")
    norm_hgvs = mutalyzer_data.get("normalized")

    # ── Fetch phase ──────────────────────────────────────────────
    # ClinGen and gnomAD may already be cached from Step 2
    if not context.raw_cache.get("clingen"):
        clinical_context = fetch_clinical_context(hgvs)
        clingen_data = clinical_context.get("clingen")
        context.raw_cache["clingen"] = clingen_data
    clingen_data = context.raw_cache.get("clingen")

    gnomad_data = search_gnomad(gene, hgvsc=norm_hgvs)

    aso_lit = {"error": "Web search not available."}

    raw_data = {
        "clingen": clingen_data,
        "gnomad_sample": gnomad_data,
        "aso_literature": aso_lit,
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
    
    # ── Reason phase ─────────────────────────────────────────────
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

    call_kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPTS["knockdown"],
        "user_message": user_msg,
        "expect_json": True,
    }
    if model_name is not None:
        call_kwargs["model"] = model_name
    result, usage = call_llm(**call_kwargs)

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

def assess_wt_upregulation(hgvs: str, context: AssessmentContext, model_name: str | None = None) -> StepResult:
    """
    Execute Section C: Wildtype Allele Upregulation assessment.

    Args:
        hgvs: Normalized HGVS string
        context: Shared context

    Returns:
        StepResult with WT upregulation assessment
    """
    if not "mutalyzer" in context.raw_cache:
        context.raw_cache["mutalyzer"] = search_mutalyzer(hgvs)
    mutalyzer_data = context.raw_cache.get("mutalyzer")
    gene = mutalyzer_data.get("gene_id")
    norm_hgvs = mutalyzer_data.get("normalized")

    if not "aso_check_pubmed" in context.raw_cache:
        aso_check_result = run_aso_check(hgvs, context, model_name=model_name)
    aso_lit = context.raw_cache.get("aso_check_pubmed")

    # ── Fetch phase ──────────────────────────────────────────────
    # Ensembl VEP
    if not "ensembl_vep" in context.raw_cache:
        context.raw_cache["ensembl_vep"] = search_ensembl_vep(norm_hgvs)
    vep_data = context.raw_cache.get("ensembl_vep")

    # Search recommended supplemental tables for TANGO/poison exon/upregulation regions in the
    alt_splice_data = search_alt_splicing_events(gene)

    # Search for related papers
    tango_query = f"{gene} AND ((poison exon) OR (TANGO) OR (antisense transcript) OR (uORF))"
    tango_papers = _get_pubmed_pmc_results(tango_query)

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

    # ── Reason phase ─────────────────────────────────────────────
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

    call_kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPTS["wt_upregulation"],
        "user_message": user_msg,
        "expect_json": True,
    }
    if model_name is not None:
        call_kwargs["model"] = model_name
    result, usage = call_llm(**call_kwargs)

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

    # Map to standard classification
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


if __name__ == "__main__":
    from dataclasses import asdict

    hgvs = "NM_000834.3:c.1858G>A"
    context = AssessmentContext(hgvs_input=hgvs)
    result = run_aso_check(hgvs, context)
    print(result)
    with open("dumps/run_aso_check_sample.json", "w") as f:
        json.dump(asdict(result), f, indent=2)



def _get_pubmed_pmc_results(search_term: str) -> list[dict[str, any]]:
    """
    Helper function that searches both PubMed and PMC for a given search term
    then deduplicates the results by PMID.
    NOTE: we use both databases, as PubMed has access to more papers, but PMC uses a full text search index

    Args:
        search_term: The search term to search for
        max_results: The maximum number of results to return

    Returns:
        A list of dictionaries containing the search results
    """
    combined_results = []
    seen_pmids = set()
    
    for db in ["pubmed", "pmc"]:
        ncbi_result = search_ncbi(
            database=db,
            search_term=search_term,
            max_results=10
        )
        # eliminate overlapping results
        for result in ncbi_result.get("results", []):
            if result.get("pmid") and result["pmid"] not in seen_pmids:
                seen_pmids.add(result.get("pmid"))
                combined_results.append(result)
    return combined_results
