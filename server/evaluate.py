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


def load_variants(spreadsheet_path: Path) -> pd.DataFrame:
    """Load variants from spreadsheet (xlsx or csv)."""
    path_str = str(spreadsheet_path)
    if path_str.endswith('.csv'):
        df = pd.read_csv(spreadsheet_path)
    else:
        df = pd.read_excel(spreadsheet_path, engine='openpyxl')

    required = ['hgvs', 'parsed_outcome']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Spreadsheet must have columns {required}. Missing: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def parse_outcome_str(outcome_str: str) -> dict:
    """
    Parses labels in form "approach1:label1;approach2:label2" into
    {"approach1": "label1", "approach2": "label2"}

    If a therapy has no label, it is considered "not applicable".
    All therapies are considered "unable to assess" if the outcome_str = "unable to assess".
    """
    parts = outcome_str.split(";")
    parsed = {k:"not_applicable" for k in ["knockdown", "splice_correction", "transcript_knockdown", "wt_upregulation"]}
    if outcome_str == "unable to assess":
        return {k:"unable_to_assess" for k in parsed.keys()}
    for part in parts:
        if ':' in part:
            # add underscores to match the automated pipeline output format
            key, value = part.split(':')
            parsed[key.strip().replace(" ", "_")] = value.strip().replace(" ", "_")
    return parsed


def main(
    input_file: Path,
    model_name: str = None,
    num_examples: int = None,
    verbose: bool = True,
    llm_only: bool = False,
) -> None:
    """Run pipeline on each variant, writing results to disk as processed."""
    output_dir = f"outputs/{model_name.split('/')[-1]}"
    if llm_only:
        output_dir = f"outputs/{model_name.split('/')[-1]}__llm-only"
    os.makedirs(output_dir, exist_ok=True)

    df = load_variants(input_file)
    if num_examples:
        df = df.head(num_examples)
    pipeline = ASOAssessmentPipeline(model_name=model_name, verbose=verbose, llm_only=llm_only)

    total = len(df)
    skipped = 0
    processed = 0
    failed = 0

    for idx, row in df.iterrows():
        hgvs = str(row['hgvs']).strip()
        true_outcome = parse_outcome_str(row['parsed_outcome'].strip())
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
            
            result = {
                "hgvs": hgvs,
                "true_outcome": true_outcome,
                "dataset": source,
                "predicted_outcome": parsed_report.get("classifications", backup_report),
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

            # to account for free tier rate limits
            time.sleep(5)

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

    main(
        args.data_file,
        model_name=args.model_name,
        num_examples=args.num_examples,
        verbose=not args.quiet,
        llm_only=args.llm_only,
    )
