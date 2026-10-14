"""
pipeline.py
-----------
Orchestrates the full FreightTiger AI pipeline:

  1. Load & validate shipment_records.csv
  2. Aggregate to weekly route-level cost metrics
  3. Detect anomalies (deterministic)
  4. Retrieve contextual notes (deterministic)
  5. Generate LLM explanations (Groq; cached; fallback-safe)
  6. Write output.csv with the required schema
  7. Return a metrics dict for the evaluation report
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_processor import build_weekly_summary
from src.anomaly_detector import detect_anomalies, format_pct_string
from src.note_retriever import load_notes, attach_notes
from src.groq_explainer import (
    generate_explanation,
    UsageTracker,
    _load_cache,
    _save_cache,
)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file's parent directory = project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "output.csv"

SHIPMENT_CSV = DATA_DIR / "shipment_records.csv"
NOTES_CSV = DATA_DIR / "context_notes.csv"


# ---------------------------------------------------------------------------
# Justified vs unexplained decision
# ---------------------------------------------------------------------------

def _resolve_flagged_label(row: pd.Series) -> str:
    """Return the human-readable flag label for the output CSV.

    - Non-anomalies: empty string (not flagged)
    - Anomaly with a matched note: "No (justified)"
    - Anomaly without a note:      "Yes"
    """
    if not row["anomaly_flag"]:
        return ""
    if row["matched_note_id"] and not pd.isna(row["matched_note_id"]):
        return "No (justified)"
    return "Yes"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    shipment_csv: Path | str | None = None,
    notes_csv: Path | str | None = None,
    output_path: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    """Run the complete pipeline and return a metrics dictionary.

    All path parameters default to project-root paths when not supplied.
    """
    t_start = time.time()

    shipment_csv = Path(shipment_csv) if shipment_csv else SHIPMENT_CSV
    notes_csv = Path(notes_csv) if notes_csv else NOTES_CSV
    output_path = Path(output_path) if output_path else OUTPUT_PATH

    if verbose:
        print(f"[pipeline] Loading shipment records from {shipment_csv} …")

    # ── Step 1-2: Load & aggregate ───────────────────────────────────────────
    weekly = build_weekly_summary(shipment_csv)
    n_route_weeks = len(weekly)

    if verbose:
        print(f"[pipeline] {n_route_weeks} route-week combinations found.")

    # ── Step 3: Anomaly detection ────────────────────────────────────────────
    if verbose:
        print("[pipeline] Running anomaly detection …")

    detected = detect_anomalies(weekly)
    n_anomalies = int(detected["anomaly_flag"].sum())

    if verbose:
        print(f"[pipeline] Anomalies detected: {n_anomalies}")

    # ── Step 4: Note retrieval (anomalies only) ──────────────────────────────
    if verbose:
        print("[pipeline] Retrieving contextual notes …")

    notes_df = load_notes(notes_csv)
    enriched = attach_notes(detected, notes_df)

    n_justified = int(
        (enriched["anomaly_flag"] & enriched["matched_note_id"].notna()).sum()
    )
    n_unexplained = n_anomalies - n_justified

    if verbose:
        print(f"[pipeline]   Justified by note: {n_justified}")
        print(f"[pipeline]   Unexplained:       {n_unexplained}")

    # ── Step 5: LLM explanations (anomalies only) ────────────────────────────
    if verbose:
        print("[pipeline] Generating explanations (Groq only for anomalies) …")

    cache = _load_cache()
    tracker = UsageTracker()
    explanations: list[str] = []

    anomaly_rows = enriched[enriched["anomaly_flag"]]

    for _, row in enriched.iterrows():
        if not row["anomaly_flag"]:
            explanations.append("")
            continue

        hist_pct = None if pd.isna(row.get("vs_own_history_pct")) else float(row["vs_own_history_pct"])
        peer_pct = None if pd.isna(row.get("vs_similar_routes_pct")) else float(row["vs_similar_routes_pct"])
        note_id = row["matched_note_id"] if row["matched_note_id"] and not pd.isna(row["matched_note_id"]) else None
        note_text = row["matched_note_text"] if row["matched_note_text"] and not pd.isna(row["matched_note_text"]) else None
        week_str = str(row["week_of"].date()) if hasattr(row["week_of"], "date") else str(row["week_of"])

        explanation, cache, used_api = generate_explanation(
            route=row["route"],
            week_of=week_str,
            cost=float(row["cost_per_tonne_km"]),
            hist_pct=hist_pct,
            peer_pct=peer_pct,
            flagged=row["anomaly_flag"],
            note_id=note_id,
            note_text=note_text,
            cache=cache,
        )
        tracker.record(used_api=used_api)
        explanations.append(explanation)

    _save_cache(cache)

    enriched["_reason_raw"] = explanations

    # ── Step 6: Build output DataFrame ──────────────────────────────────────
    if verbose:
        print("[pipeline] Building output CSV …")

    out_rows = []
    for _, row in enriched.iterrows():
        hist_pct = None if pd.isna(row.get("vs_own_history_pct")) else float(row["vs_own_history_pct"])
        peer_pct = None if pd.isna(row.get("vs_similar_routes_pct")) else float(row["vs_similar_routes_pct"])

        note_id_val = row["matched_note_id"]
        if not note_id_val or (isinstance(note_id_val, float) and np.isnan(note_id_val)):
            note_id_val = ""

        flagged_label = _resolve_flagged_label(row)
        reason = row["_reason_raw"] if row["anomaly_flag"] else ""

        week_str = str(row["week_of"].date()) if hasattr(row["week_of"], "date") else str(row["week_of"])

        out_rows.append({
            "route": row["route"],
            "week_of": week_str,
            "cost_per_tonne_km": round(float(row["cost_per_tonne_km"]), 2),
            "vs_own_history": format_pct_string(hist_pct, "this route's past average"),
            "vs_similar_routes": format_pct_string(peer_pct, "similar-length routes this week"),
            "flagged": flagged_label,
            "matched_note_id": note_id_val,
            "reason": reason,
        })

    out_df = pd.DataFrame(out_rows, columns=[
        "route", "week_of", "cost_per_tonne_km",
        "vs_own_history", "vs_similar_routes",
        "flagged", "matched_note_id", "reason",
    ])

    out_df.to_csv(output_path, index=False)

    t_elapsed = time.time() - t_start

    if verbose:
        print(f"[pipeline] Output written to {output_path}")
        print(f"[pipeline] Completed in {t_elapsed:.1f}s")

    # ── Metrics ──────────────────────────────────────────────────────────────
    usage = tracker.summary()
    metrics = {
        "route_weeks_processed": n_route_weeks,
        "anomalies_detected": n_anomalies,
        "justified_by_note": n_justified,
        "unexplained": n_unexplained,
        "execution_time_s": round(t_elapsed, 2),
        **usage,
    }

    return metrics
