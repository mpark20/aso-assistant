#!/usr/bin/env python3
"""
Evaluate the ASO assessment pipeline against expert-labeled variants.

Reads a spreadsheet with columns "hgvs" and "N1C VARIANT outcome", runs the
pipeline for each variant, and writes results to disk. Skips variants that
have already been processed.

Usage:
    python evaluate.py -m claude-sonnet-5 --split n1c_assessed_variants -n 2
"""
from collections import defaultdict
import argparse
import asyncio
import json
import os
import re
import sys
import time
import pandas as pd
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any

from aso_workflow.pipeline import ASOAssessmentPipeline

THERAPY_TYPES = ["splice_correction", "exon_skipping", "transcript_knockdown", "wt_upregulation"]

# label categories for coarse-grained scoring
NEG_LABELS = ["not_applicable", "not_eligible", "unable_to_assess", "not_eligible|unable_to_assess"]
POS_LABELS = ["likely_eligible", "eligible"]
NEUTRAL_LABELS = ["unlikely_eligible", "applicable"]


def sanitize_hgvs_for_filename(hgvs: str, max_length: int = 120) -> str:
    """Convert HGVS string to a safe filesystem filename."""
    safe = re.sub(r'[^\w\-.]', '_', hgvs)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe[:max_length] if len(safe) > max_length else safe


def load_variants(spreadsheet_path: Path, split="n1c_test_variants") -> pd.DataFrame:
    """Load variants from spreadsheet (xlsx or csv)."""
    path_str = str(spreadsheet_path)
    if path_str.endswith('.csv'):
        df = pd.read_csv(spreadsheet_path)
    else:
        df = pd.read_excel(spreadsheet_path, engine='openpyxl')
    df.rename(columns={"normalized_hgvs": "hgvs"}, inplace=True)

    required = ['hgvs', 'parsed_outcome']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Spreadsheet must have columns {required}. Missing: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    
    if split is not None:
        df = df.loc[df["source"] == split].reset_index()

    return df


def parse_outcome_str(outcome_str: str) -> dict:
    """
    Parses labels in form "approach1:label1;approach2:label2" into
    {"approach1": "label1", "approach2": "label2"}

    If a therapy has no label, it is considered "not_eligible|unable_to_assess, as the dataset doesn't explicitly include these".
    All therapies are considered "unable to assess" if the outcome_str = "unable to assess".
    """
    parts = outcome_str.split(";")
    parsed = {}
    if outcome_str == "unable_to_assess":
        return {k:"unable_to_assess" for k in THERAPY_TYPES}
    for part in parts:
        if ':' in part:
            # add underscores to match the automated pipeline output format
            key, value = part.split(':')
            parsed[key.strip().replace(" ", "_")] = value.strip().replace(" ", "_")
    
    if "knockdown" in parsed:
        kd = parsed.pop("knockdown")
        parsed["transcript_knockdown"] = kd
    return parsed


def calculate_score(true_outcome: dict, pred_outcome: dict, strict: bool = False) -> dict:
    """
    Scores a predicted outcome against a true outcome.
    Assumes both `true_outcome` and `pred_outcome` are dictionaries with keys
    "knockdown", "splice_correction", "transcript_knockdown", "wt_upregulation", "exon_skipping".
    """
    # grading scale (4 pt possible):
    # - same label = 1 pt
    # - diff label = 0 pt if strict, or potential partial credit for similar options
    scores = defaultdict(float)
    for k in true_outcome.keys():
        pred_outcome[k] = pred_outcome[k].replace("not_applicable", "not_eligible")
        if pred_outcome[k] in true_outcome[k].split('|'): # e.g. unable_to_assess|not_eligible
            scores[k] = 1
        elif strict:
            scores[k] = 0
        
        # otherwise, consider coarse-grained matches
        elif true_outcome[k] in NEG_LABELS and pred_outcome[k] in NEG_LABELS:
            scores[k] = 1
        elif true_outcome[k] in NEUTRAL_LABELS and pred_outcome[k] in NEUTRAL_LABELS:
            scores[k] = 1
        elif true_outcome[k] in POS_LABELS and pred_outcome[k] in POS_LABELS:
            scores[k] = 1
        else:
            scores[k] = 0
    return scores


@dataclass
class VariantResult:
    idx: int
    status: str  # "processed", "failed", "skipped"
    true_outcome: dict[str, Any] | None = None
    pred_outcome: dict[str, Any] | None = None
    hgvs: str | None = None
    out_path: str | None = None
    error: str | None = None


async def generate_report(
    semaphore: asyncio.Semaphore,
    pipeline: ASOAssessmentPipeline,
    idx: int,
    total: int,
    hgvs: str,
    source: str,
    true_outcome: dict,
    out_path: str,
    verbose: bool,
) -> VariantResult:
    """Run the pipeline for one variant, limited by the semaphore."""
    async with semaphore:
        if verbose:
            print(f"\n[{idx + 1}/{total}] Processing: {hgvs} (source: {source})")

        try:
            report = await asyncio.to_thread(pipeline.run, hgvs)
            parsed_report = report.to_dict()
            try:
                backup_report = {
                    "splice_correction": report.splice_correction.value,
                    "exon_skipping": report.exon_skipping.value,
                    "transcript_knockdown": report.transcript_knockdown.value,
                    "wt_upregulation": report.wt_upregulation.value,
                }
            except Exception:
                backup_report = None

            pred_outcome = parsed_report.get("classifications", backup_report)
            result = {
                "hgvs": hgvs,
                "dataset": source,
                "true_outcome": true_outcome,
                "predicted_outcome": pred_outcome,
                "pipeline_report": asdict(report),
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"  → Wrote {out_path}")

            return VariantResult(
                idx=idx,
                status="processed",
                true_outcome=true_outcome,
                pred_outcome=pred_outcome,
                hgvs=hgvs,
                out_path=out_path,
            )
        except Exception as e:
            print(f"  ✗ Failed ({hgvs}): {e}")
            return VariantResult(
                idx=idx,
                status="failed",
                hgvs=hgvs,
                error=str(e),
            )


async def generate_report_batch(
    examples: list[tuple],
    pipeline: ASOAssessmentPipeline,
    batch_size: int,
    total: int,
    verbose: bool,
) -> list[VariantResult]:
    """Run examples_to_run variants with at most batch_size concurrent pipeline runs."""
    semaphore = asyncio.Semaphore(batch_size)
    tasks = [
        generate_report(
            semaphore,
            pipeline,
            idx,
            total,
            hgvs,
            source,
            true_outcome,
            out_path,
            verbose,
        )
        for idx, hgvs, source, true_outcome, out_path in examples
    ]
    return await asyncio.gather(*tasks)


def main(args) -> None:
    """Run pipeline on each variant, writing results to disk as processed.
    
    Args:
        input_file: Path to spreadsheet (xlsx or csv) with columns 'hgvs' and 'parsed_outcome'
        model_name: Model name to use for evaluation
        num_examples: Number of examples to evaluate (default: all)
        verbose: If True, print progress to stdout during pipeline execution
        llm_only: If True, bypass database calls in all steps; only gene, norm_hgvs, and instruction are added to prompts
        use_web_search: If True, llm calls will use the model provider's native web search tool.
        batch_size: Number of cases to run concurrently (default: 1).
    """
    input_file = args.data_file
    model_name = args.model_name
    num_examples = args.num_examples
    split = args.split
    verbose = args.verbose
    llm_only = args.llm_only
    use_web_search = args.use_web_search
    batch_size = args.batch_size

    output_dir = f"outputs/{model_name.split('/')[-1]}"
    
    if llm_only:
        output_dir += "__llm-only"
    if use_web_search:
        output_dir += "__web-search"
    os.makedirs(output_dir, exist_ok=True)

    df = load_variants(input_file, split=split)
    if num_examples:
        df = df.iloc[:num_examples]
    if args.hgvs:
        df = df[df['hgvs'] == args.hgvs]
    
    pipeline = ASOAssessmentPipeline(
        model_name=model_name,
        verbose=verbose,
        llm_only=llm_only,
        use_web_search=use_web_search,
    )

    total = len(df)
    skipped = 0
    processed = 0
    failed = 0
    start_time = time.time()

    results: list[VariantResult] = []
    examples_to_run: list[tuple] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        hgvs = str(row['hgvs']).strip()
        source = str(row['source']).strip()

        if pd.isna(row['hgvs']) or not hgvs:
            if verbose:
                print(f"[{idx + 1}/{total}] Skipping empty hgvs")
            skipped += 1
            continue

        safe_name = sanitize_hgvs_for_filename(hgvs)
        out_path = os.path.join(output_dir, f"{safe_name}.json")
        true_outcome = parse_outcome_str(row['parsed_outcome'].strip())

        if os.path.exists(out_path):
            if verbose:
                print(f"[{idx + 1}/{total}] Skipping (already exists): {hgvs}")
            skipped += 1
            with open(out_path) as f:
                report = json.load(f)
            results.append(VariantResult(
                idx=idx,
                status="skipped",
                true_outcome=true_outcome,
                pred_outcome=report['predicted_outcome'],
            ))
            continue

        examples_to_run.append((idx, hgvs, source, true_outcome, out_path))

    if examples_to_run:
        if verbose and batch_size > 1:
            print(f"\nRunning {len(examples_to_run)} variants with batch_size={batch_size}")
        batch_results = asyncio.run(
            generate_report_batch(examples_to_run, pipeline, batch_size, total, verbose)
        )
        results.extend(batch_results)

    results.sort(key=lambda r: r.idx)
    true_outcomes = [r.true_outcome for r in results if r.true_outcome is not None]
    pred_outcomes = [r.pred_outcome for r in results if r.pred_outcome is not None]
    processed = sum(1 for r in results if r.status == "processed")
    failed = sum(1 for r in results if r.status == "failed")

    end_time = time.time()
    print(f"Evaluation took {end_time - start_time} seconds")
    if processed:
        print(f"Average time per example: {(end_time - start_time) / processed} seconds")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation complete: {processed} processed, {skipped} skipped, {failed} failed")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ASO pipeline against expert-labeled variants"
    )
    parser.add_argument(
        "-d", "--data-file",
        type=Path,
        help="Path to spreadsheet (xlsx or csv) with columns 'hgvs' and 'parsed_outcome'",
        default=Path("data/parsed_n1c_assessments.csv")
    )
    # this is for the MAIN model, the helper model is hardcoded to gpt-5-nano
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="gemini/gemini-3-flash-preview",
        help="Model name to use for evaluation (default: gemini/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed report progress",
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int, default=None, help="Number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--split",
        choices=["n1c_test_variants", "n1c_assessed_variants", "gene_steps_assessed_variants"],
        default="n1c_test_variants", help="Split to evaluate (default: n1c_test_variants)"
    )
    parser.add_argument(
        "--hgvs",
        type=str, default=None, help="HGVS to evaluate (default: all)"
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="If true, llm calls will use the model provider's native web search tool.",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Bypass database calls; only add gene, norm_hgvs, and instruction to prompts (for ablation experiments)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of cases to run concurrently (default: 1)",
    )
    args = parser.parse_args()

    if not args.data_file.exists():
        print(f"Error: Spreadsheet not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)

    if args.batch_size < 1:
        print("Error: --batch-size must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    print(args)

    main(args)
