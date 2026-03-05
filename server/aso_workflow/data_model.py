"""
Core data models for the N1C VARIANT ASO Assessment Pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


# ─────────────────────────────────────────────
# Classification Enumerations
# ─────────────────────────────────────────────

class EligibilityClassification(str, Enum):
    ELIGIBLE = "eligible"
    LIKELY_ELIGIBLE = "likely_eligible"
    UNLIKELY_ELIGIBLE = "unlikely_eligible"
    NOT_ELIGIBLE = "not_eligible"
    UNABLE_TO_ASSESS = "unable_to_assess"
    NOT_APPLICABLE = "not_applicable"  # step was skipped for this variant


class InheritancePattern(str, Enum):
    AUTOSOMAL_DOMINANT = "autosomal_dominant"
    AUTOSOMAL_RECESSIVE = "autosomal_recessive"
    X_LINKED_RECESSIVE = "x_linked_recessive"
    X_LINKED_DOMINANT = "x_linked_dominant"
    UNKNOWN = "unknown"


class Pathomechanism(str, Enum):
    LOSS_OF_FUNCTION = "loss_of_function"
    GAIN_OF_FUNCTION = "gain_of_function"
    DOMINANT_NEGATIVE = "dominant_negative"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────
# Per-Step Result
# ─────────────────────────────────────────────

@dataclass
class StepResult:
    step_name: str
    classification: EligibilityClassification
    summary: str                      # 1-2 sentence human-readable summary
    reasoning: str                    # full LLM reasoning chain
    data_used: dict[str, Any]         # raw data fetched for this step
    metadata: dict[str, Any] = field(default_factory=dict)  # step-specific structured fields
    error: Optional[str] = None
    # Token usage per model: {model: {"input_tokens": int, "output_tokens": int, "total_tokens": int}}
    token_usage: dict[str, dict[str, int]] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Shared Assessment Context (passed between steps)
# ─────────────────────────────────────────────

@dataclass
class AssessmentContext:
    """
    Accumulates structured facts discovered during the assessment.
    Steps read from and write to this object so downstream steps can
    use facts established by upstream ones.
    """
    hgvs_input: str
    hgvs_normalized: Optional[str] = None
    gene_id: Optional[str] = None
    refseq_id: Optional[str] = None

    # ASO check outputs
    existing_aso_found: Optional[bool] = None
    existing_aso_type: Optional[str] = None
    existing_aso_success: Optional[bool] = None
    existing_aso_sufficient: Optional[bool] = None
    existing_aso_summary: Optional[str] = None

    # Step 1 outputs
    inheritance_pattern: Optional[InheritancePattern] = None
    inheritance_confidence: Optional[str] = None   # "high" | "medium" | "low"

    # Step 2 outputs
    pathomechanism: Optional[Pathomechanism] = None
    is_haploinsufficient: Optional[bool] = None
    haploinsufficiency_evidence: Optional[str] = None

    # Step 3 outputs
    has_splicing_evidence: Optional[bool] = None   # True = confirmed splicing effect
    splicing_effect_type: Optional[str] = None     # "cryptic_exon", "exon_skipping", "donor_loss", "acceptor_loss", etc.
    canonical_splicing_destroyed: Optional[bool] = None
    intronic_or_exonic: Optional[str] = None    # "intronic", "exonic"

    # Step 0 outputs
    variant_valid: Optional[bool] = None
    is_cnv_gain: Optional[bool] = None
    is_cnv_loss: Optional[bool] = None

    # Sections to evaluate (set in Step 4 routing)
    evaluate_splice_correction: bool = False
    evaluate_exon_skipping: bool = False
    evaluate_knockdown: bool = False
    evaluate_wt_upregulation: bool = False

    # Raw data cache (avoids re-fetching across steps)
    raw_cache: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Final Report
# ─────────────────────────────────────────────


@dataclass
class ASOAssessmentReport:
    hgvs: str
    gene_id: Optional[str]
    step_results: dict[str, StepResult]

    # Final classifications per strategy
    splice_correction: EligibilityClassification = EligibilityClassification.NOT_APPLICABLE
    exon_skipping: EligibilityClassification = EligibilityClassification.NOT_APPLICABLE
    transcript_knockdown: EligibilityClassification = EligibilityClassification.NOT_APPLICABLE
    wt_upregulation: EligibilityClassification = EligibilityClassification.NOT_APPLICABLE

    summary: Optional[str | dict[str, Any]] = None
    context: Optional[AssessmentContext] = None
    # Combined token usage across all steps: {model: {"input_tokens", "output_tokens", "total_tokens"}}
    total_token_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    date: Optional[str] = None
    model_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "hgvs": self.hgvs,
            "gene_id": self.gene_id,
            "classifications": {
                "splice_correction": self.splice_correction.value,
                "exon_skipping": self.exon_skipping.value,
                "transcript_knockdown": self.transcript_knockdown.value,
                "wt_upregulation": self.wt_upregulation.value,
            },
            "summary": self.summary,
            "total_token_usage": self.total_token_usage,
            "steps": {
                name: {
                    "classification": r.classification.value,
                    "summary": r.summary,
                    "reasoning": r.reasoning,
                    "metadata": r.metadata,
                    "error": r.error,
                    "token_usage": r.token_usage,
                }
                for name, r in self.step_results.items()
            },
        }