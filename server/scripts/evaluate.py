#!/usr/bin/env python3
"""
Evaluate the ASO assessment pipeline against expert-labeled variants.

Reads a spreadsheet with columns "hgvs" and "N1C VARIANT outcome", runs the
pipeline for each variant, and writes results to disk. Skips variants that
have already been processed.

Usage:
    python evaluate.py path/to/spreadsheet.xlsx
    python evaluate.py path/to/spreadsheet.csv --output-dir evaluation_results
"""
import pdb
import argparse
import json
import os
import re
import sys
import time
import pandas as pd
from pathlib import Path
from dataclasses import asdict

from aso_workflow.pipeline import ASOAssessmentPipeline


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
    
    df = df.loc[df["source"] == split].reset_index()

    return df


def parse_outcome_str(outcome_str: str) -> dict:
    """
    Parses labels in form "approach1:label1;approach2:label2" into
    {"approach1": "label1", "approach2": "label2"}

    If a therapy has no label, it is considered "not applicable".
    All therapies are considered "unable to assess" if the outcome_str = "unable to assess".
    """
    parts = outcome_str.split(";")
    parsed = {k:"not_applicable" for k in ["splice_correction", "exon_skipping", "transcript_knockdown", "wt_upregulation"]}
    if outcome_str == "unable_to_assess":
        return {k:"unable_to_assess" for k in parsed.keys()}
    for part in parts:
        if ':' in part:
            # add underscores to match the automated pipeline output format
            key, value = part.split(':')
            parsed[key.strip().replace(" ", "_")] = value.strip().replace(" ", "_")
    
    if "knockdown" in parsed:
        kd = parsed.pop("knockdown")
        parsed["transcript_knockdown"] = kd
    return parsed

def score_result(true_outcome: dict, pred_outcome: dict) -> dict:
    """
    Scores a predicted outcome against a true outcome.
    Assumes both `true_outcome` and `pred_outcome` are dictionaries with keys
    "knockdown", "splice_correction", "transcript_knockdown", "wt_upregulation", "exon_skipping".
    """
    # grading scale (4 pt possible):
    # - same label = 1 pt
    # - likely eligible vs eligible = 0.5 pt
    # - not eligible vs not applicable = 1 pt (this info isn't available in ground truth assessments)
    # - all other cases = 0 pt
    neg_edge_cases = ["unable_to_assess", "not_applicable", "not_eligible"]
    pos_edge_cases = ["likely_eligible", "eligible"]
    score = 0
    for k in true_outcome.keys():
        if true_outcome[k] == pred_outcome[k]:
            score += 1
        elif true_outcome[k] in pos_edge_cases and pred_outcome[k] in pos_edge_cases:
            score += 0.5
        elif true_outcome[k] in neg_edge_cases and pred_outcome[k] in neg_edge_cases:
            score += 1
        else:
            score += 0
    return score


def main(args) -> None:
    """Run pipeline on each variant, writing results to disk as processed.
    
    Args:
        input_file: Path to spreadsheet (xlsx or csv) with columns 'hgvs' and 'parsed_outcome'
        model_name: Model name to use for evaluation
        num_examples: Number of examples to evaluate (default: all)
        verbose: If True, print progress to stdout during pipeline execution
        llm_only: If True, bypass database calls in all steps; only gene, norm_hgvs, and instruction are added to prompts
        use_web_search: If True, llm calls will use the model provider's native web search tool.
    """
    input_file = args.data_file
    model_name = args.model_name
    num_examples = args.num_examples
    verbose = not args.quiet
    llm_only = args.llm_only
    use_web_search = args.use_web_search

    output_dir = f"outputs/{model_name.split('/')[-1]}"
    
    if llm_only:
        output_dir += "__llm-only"
    if use_web_search:
        output_dir += "__web-search"
    os.makedirs(output_dir, exist_ok=True)

    df = load_variants(input_file, split="n1c_test_variants")
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

    true_outcomes = []
    pred_outcomes = []

    for idx, row in df.iterrows():
        hgvs = str(row['hgvs']).strip()
        source = str(row['source']).strip()

        if pd.isna(row['hgvs']) or not hgvs:
            if verbose:
                print(f"[{idx + 1}/{total}] Skipping empty hgvs")
            skipped += 1
            continue

        safe_name = sanitize_hgvs_for_filename(hgvs)
        out_path = os.path.join(output_dir, f"{safe_name}.json")

        if os.path.exists(out_path):
            if verbose:
                print(f"[{idx + 1}/{total}] Skipping (already exists): {hgvs}")
            skipped += 1
            report = json.load(open(out_path))
            pred_outcomes.append(report['predicted_outcome'])
            true_outcomes.append(parse_outcome_str(row['parsed_outcome'].strip()))
            continue

        if verbose:
            print(f"\n[{idx + 1}/{total}] Processing: {hgvs} (source: {source})")

        try:
            report = pipeline.run(hgvs)
            parsed_report = report.to_dict()
            try:
                backup_report = {
                    "splice_correction": report.splice_correction.value,
                    "exon_skipping": report.exon_skipping.value,
                    "transcript_knockdown": report.transcript_knockdown.value,
                    "wt_upregulation": report.wt_upregulation.value,
                }
            except Exception as e:
                backup_report = None
            
            true_outcome = parse_outcome_str(row['parsed_outcome'].strip())
            true_outcomes.append(true_outcome)
            pred_outcome = parsed_report.get("classifications", backup_report)
            pred_outcomes.append(pred_outcome)

            result = {
                "hgvs": hgvs,
                "true_outcome": true_outcome,
                "dataset": source,
                "predicted_outcome": pred_outcome,
                "pipeline_report": asdict(report),
            }
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            processed += 1
            if verbose:
                print(f"  → Wrote {out_path}")
        except Exception as e:
            failed += 1
            if verbose:
                print(f"  ✗ Failed: {e}")

            time.sleep(1)

    scores = [score_result(true_outcome, pred_outcome) for true_outcome, pred_outcome in zip(true_outcomes, pred_outcomes)]
    print(f"Average score: {sum(scores) / len(scores)}")

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
    # this is for the MAIN model, the helper model is hardcoded to gemini/gemini-3.1-flash-lite-preview
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="gemini/gemini-3-flash-preview",
        help="Model name to use for evaluation (default: gemini/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int, default=None, help="Number of examples to evaluate (default: all)"
    )
    parser.add_argument("--hgvs", type=str, default=None, help="HGVS to evaluate (default: all)")
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
    args = parser.parse_args()

    if not args.data_file.exists():
        print(f"Error: Spreadsheet not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)
    
    print(args)

    main(args)
