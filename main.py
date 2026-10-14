"""
main.py
-------
Entry point for the FreightTiger AI Shipping Cost Monitoring Assistant.

Usage
-----
  python main.py                   # run pipeline, write output.csv
  python main.py --reproduce 3     # run 3 times and compare results

Windows PowerShell example:
  cd freighttiger-ai-assistant
  python main.py
"""

from __future__ import annotations

import sys
import json
import argparse
import hashlib
import pandas as pd
from pathlib import Path

from src.pipeline import run_pipeline, OUTPUT_PATH


def _csv_fingerprint(path: Path) -> str:
    """Return a SHA-256 fingerprint of the numeric + flag columns of output.csv."""
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    # Only hash deterministic columns (not 'reason' which may vary by LLM call)
    cols = ["route", "week_of", "cost_per_tonne_km",
            "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]
    stable = df[cols].to_csv(index=False)
    return hashlib.sha256(stable.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="FreightTiger AI Shipping Cost Monitoring Assistant"
    )
    parser.add_argument(
        "--reproduce", type=int, default=1, metavar="N",
        help="Run the pipeline N times and verify numeric reproducibility (default: 1)"
    )
    parser.add_argument(
        "--no-api", action="store_true",
        help="Skip Groq API calls (force fallback explanations)"
    )
    args = parser.parse_args()

    if args.no_api:
        import os
        os.environ.pop("GROQ_API_KEY", None)
        print("[main] --no-api flag set: Groq API disabled, using fallback explanations.")

    fingerprints: list[str] = []
    all_metrics: list[dict] = []

    for run_i in range(1, args.reproduce + 1):
        if args.reproduce > 1:
            print(f"\n{'='*60}")
            print(f"  RUN {run_i} / {args.reproduce}")
            print(f"{'='*60}")

        metrics = run_pipeline(verbose=True)
        all_metrics.append(metrics)

        fp = _csv_fingerprint(OUTPUT_PATH)
        fingerprints.append(fp)

    # ── Print metrics summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE METRICS")
    print("=" * 60)
    for k, v in all_metrics[-1].items():
        print(f"  {k:<35} {v}")

    # ── Reproducibility check ────────────────────────────────────────────────
    if len(fingerprints) > 1:
        print("\n" + "=" * 60)
        print("  REPRODUCIBILITY CHECK")
        print("=" * 60)
        all_same = len(set(fingerprints)) == 1
        for i, fp in enumerate(fingerprints, 1):
            print(f"  Run {i}: {fp[:24]}…")
        if all_same:
            print("  [OK] All runs produced identical numeric results.")
        else:
            print("  [!!] Numeric results differ between runs - investigate.")

    print(f"\n  Output CSV: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
