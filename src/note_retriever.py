"""
note_retriever.py
-----------------
Lightweight contextual-note retrieval without FAISS or any vector database.

Retrieval pipeline (all four stages evaluated; first match wins in priority order):

  Stage 1 – Exact route match  (applies_to == route name)
  Stage 2 – "All Routes" scope matched to a relevant date window
  Stage 3 – Date-range proximity (note date within ±35 days of week_of)
  Stage 4 – Keyword overlap (TF-IDF cosine between note text and route+material)

A note is only considered *valid justification* for an anomaly if:
  - It describes a reason for cost increases (not a "no disruption" note), AND
  - Its date is within ±35 days of the anomalous week_of

Validation prevents irrelevant or "all clear" notes from falsely justifying anomalies.
"""

from __future__ import annotations

import re
import math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import Counter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATE_WINDOW_DAYS: int = 35   # maximum days between note date and week_of

# Keywords that imply the note could explain a cost increase
COST_INCREASE_KEYWORDS: set[str] = {
    "flood", "flooding", "disruption", "disrupted", "detour", "surcharge",
    "festival", "demand", "diesel", "price", "toll", "rise", "higher",
    "delay", "delays", "closure", "maintenance", "shortage", "cost",
    "increas", "pressure", "restrict", "divert",
}

# Keywords that suggest the note is a "no disruption / all clear" note
NO_ISSUE_KEYWORDS: set[str] = {
    "no significant", "normal", "remained normal", "stable", "no major",
    "not significantly", "without a rate change", "absorbed", "improved",
    "returned to normal",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, split into tokens."""
    return re.findall(r"[a-z]+", text.lower())


def _tfidf_cosine(query_tokens: list[str], doc_tokens: list[str],
                  corpus_token_counts: list[Counter], doc_idx: int) -> float:
    """Very lightweight TF-IDF cosine similarity between query and one document."""
    N = len(corpus_token_counts)
    if N == 0:
        return 0.0

    # IDF
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    shared = query_set & doc_set
    if not shared:
        return 0.0

    def tfidf_vec(tokens: list[str], token_set: set[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {}
        for tok in token_set:
            idf = math.log((1 + N) / (1 + sum(1 for c in corpus_token_counts if tok in c))) + 1
            vec[tok] = (tf.get(tok, 0) / max(len(tokens), 1)) * idf
        return vec

    q_vec = tfidf_vec(query_tokens, shared)
    d_vec = tfidf_vec(doc_tokens, shared)

    dot = sum(q_vec[t] * d_vec[t] for t in shared)
    q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values()))
    d_norm = math.sqrt(sum(v ** 2 for v in d_vec.values()))

    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (q_norm * d_norm)


def _note_has_cost_increase_signal(note_text: str) -> bool:
    """Return True if note text suggests a reason for a cost increase."""
    lower = note_text.lower()
    # If note is clearly "no issue", it cannot justify a cost spike
    for phrase in NO_ISSUE_KEYWORDS:
        if phrase in lower:
            return False
    # Must contain at least one cost-increase keyword
    for kw in COST_INCREASE_KEYWORDS:
        if kw in lower:
            return True
    return False


def _date_in_window(note_date: pd.Timestamp, week_of: pd.Timestamp) -> bool:
    """True if note_date is within DATE_WINDOW_DAYS of week_of."""
    delta = abs((note_date - week_of).days)
    return delta <= DATE_WINDOW_DAYS


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_notes(csv_path: str | Path) -> pd.DataFrame:
    """Load and parse context_notes.csv.

    Returns a DataFrame with columns:
        note_id, date (Timestamp), applies_to (str), note (str)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Context notes not found: {path}")

    df = pd.read_csv(path)

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Parse date; tolerate various formats
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where critical fields are missing
    df = df.dropna(subset=["note_id", "date", "applies_to", "note"])
    df["applies_to"] = df["applies_to"].str.strip()
    df["note"] = df["note"].str.strip()

    # Precompute tokenised note text for TF-IDF
    df["_tokens"] = df["note"].apply(_tokenize)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve_note(
    route: str,
    week_of: pd.Timestamp,
    notes_df: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """Return (note_id, note_text) for the best-matching *valid* note, or (None, None).

    "Valid" means:
      1. Route scope matches (exact or "All Routes")
      2. Date is within ±35 days of week_of
      3. Note has a cost-increase signal (not an "all-clear" note)

    Among valid candidates, the one with the highest TF-IDF score (or earliest
    in stage priority) is returned.
    """
    if notes_df is None or notes_df.empty:
        return None, None

    corpus_token_counts = [Counter(toks) for toks in notes_df["_tokens"]]
    query_tokens = _tokenize(route)

    candidates: list[tuple[float, int]] = []  # (score, row_idx)

    for i, row in notes_df.iterrows():
        applies_to: str = row["applies_to"]
        note_date: pd.Timestamp = row["date"]
        note_text: str = row["note"]

        # ── Stage 1: route scope ────────────────────────────────────────────
        route_match = (applies_to == route) or (applies_to == "All Routes")
        if not route_match:
            continue  # no point scoring further

        # ── Stage 2: date window ────────────────────────────────────────────
        if not _date_in_window(note_date, week_of):
            continue

        # ── Stage 3: cost-increase validation ──────────────────────────────
        if not _note_has_cost_increase_signal(note_text):
            continue  # note cannot justify an anomaly

        # ── Stage 4: TF-IDF score ───────────────────────────────────────────
        tfidf = _tfidf_cosine(query_tokens, row["_tokens"], corpus_token_counts, i)

        # Boost for exact route match vs "All Routes"
        boost = 2.0 if applies_to == route else 1.0
        final_score = tfidf * boost + (1.0 if applies_to == route else 0.0)

        candidates.append((final_score, i))

    if not candidates:
        return None, None

    # Best candidate = highest score; tie-break by recency (closest date)
    candidates.sort(key=lambda x: -x[0])
    best_idx = candidates[0][1]
    best_row = notes_df.iloc[best_idx]

    return str(best_row["note_id"]), str(best_row["note"])


# ---------------------------------------------------------------------------
# Batch wrapper
# ---------------------------------------------------------------------------

def attach_notes(anomaly_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    """For every row in anomaly_df, retrieve the best matching note.

    Only anomalous rows get a note lookup (non-anomalies have no note attached).

    Adds columns: matched_note_id, matched_note_text
    """
    anomaly_df = anomaly_df.copy()
    anomaly_df["matched_note_id"] = None
    anomaly_df["matched_note_text"] = None

    for idx, row in anomaly_df.iterrows():
        if not row.get("anomaly_flag", False):
            continue  # skip non-anomalies

        note_id, note_text = retrieve_note(
            route=row["route"],
            week_of=row["week_of"],
            notes_df=notes_df,
        )
        anomaly_df.at[idx, "matched_note_id"] = note_id
        anomaly_df.at[idx, "matched_note_text"] = note_text

    return anomaly_df
