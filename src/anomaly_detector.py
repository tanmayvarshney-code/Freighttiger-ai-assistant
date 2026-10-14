"""
anomaly_detector.py
-------------------
Purely deterministic, Python-only anomaly detection.

Two signals are computed per (route, week_of):
  1. vs_own_history  – % deviation vs. the route's own 8-week rolling baseline
  2. vs_similar_routes – % deviation vs. peer routes (same route_type) in the same week

Anomaly flagging rules (consistent with the PDF):
  - FLAG if vs_own_history  >= +20% (the cost is ≥20% above its own 8-week baseline)
  - FLAG if vs_similar_routes >= +20% (the cost is ≥20% above the peer-route average)
  - A route-week is anomalous if EITHER condition is True
  - "No (justified)" is resolved downstream by the note-retriever + LLM layer

All numeric decisions are made here. Groq is never asked to perform maths.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ── Tunable thresholds (percentage points) ──────────────────────────────────
HISTORY_THRESHOLD_PCT: float = 20.0   # % above own 8-week baseline → flag
PEER_THRESHOLD_PCT: float = 20.0      # % above same-type peers this week → flag
HISTORY_WEEKS: int = 8                 # number of previous weeks used as baseline


# ---------------------------------------------------------------------------
# Own-history baseline
# ---------------------------------------------------------------------------

def _compute_own_history(weekly: pd.DataFrame) -> pd.DataFrame:
    """For every (route, week_of) compute the rolling 8-week backward baseline.

    The baseline is the mean cost_per_tonne_km of the 8 weeks *before* the
    current week (i.e. strictly excluding the current week itself to avoid
    future-data leakage).

    Returns the same DataFrame with two new columns:
        history_baseline   – mean cost of previous 8 available weeks
        vs_own_history_pct – percentage deviation from that baseline
    """
    result_rows = []

    for route, grp in weekly.groupby("route"):
        grp = grp.sort_values("week_of").copy()
        costs = grp["cost_per_tonne_km"].values
        weeks = grp["week_of"].values

        history_baselines = []
        deviations = []

        for i in range(len(grp)):
            # Take up to 8 preceding weeks (not the current week)
            past_costs = costs[max(0, i - HISTORY_WEEKS): i]
            valid_past = past_costs[~np.isnan(past_costs)]

            if len(valid_past) == 0:
                history_baselines.append(np.nan)
                deviations.append(np.nan)
            else:
                baseline = float(np.mean(valid_past))
                current = costs[i]
                if np.isnan(current) or baseline == 0:
                    deviations.append(np.nan)
                else:
                    dev_pct = (current - baseline) / baseline * 100.0
                    deviations.append(round(dev_pct, 2))
                history_baselines.append(round(baseline, 6))

        grp["history_baseline"] = history_baselines
        grp["vs_own_history_pct"] = deviations
        result_rows.append(grp)

    return pd.concat(result_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Peer / similar-route baseline
# ---------------------------------------------------------------------------

def _compute_peer_comparison(weekly: pd.DataFrame) -> pd.DataFrame:
    """For every (route, week_of) compute the peer-route average for the same week.

    Peers are routes of the same route_type, excluding the route itself.

    Returns the same DataFrame with two new columns:
        peer_baseline         – mean cost of all other same-type routes this week
        vs_similar_routes_pct – percentage deviation from that baseline
    """
    # Build a lookup: (route_type, week_of) → list of costs for each *other* route
    weekly = weekly.copy()
    weekly["peer_baseline"] = np.nan
    weekly["vs_similar_routes_pct"] = np.nan

    for idx, row in weekly.iterrows():
        same_week_same_type = weekly[
            (weekly["week_of"] == row["week_of"])
            & (weekly["route_type"] == row["route_type"])
            & (weekly["route"] != row["route"])
        ]

        valid_peers = same_week_same_type["cost_per_tonne_km"].dropna()

        if len(valid_peers) == 0:
            weekly.at[idx, "peer_baseline"] = np.nan
            weekly.at[idx, "vs_similar_routes_pct"] = np.nan
        else:
            peer_mean = float(valid_peers.mean())
            weekly.at[idx, "peer_baseline"] = round(peer_mean, 6)
            if np.isnan(row["cost_per_tonne_km"]) or peer_mean == 0:
                weekly.at[idx, "vs_similar_routes_pct"] = np.nan
            else:
                dev = (row["cost_per_tonne_km"] - peer_mean) / peer_mean * 100.0
                weekly.at[idx, "vs_similar_routes_pct"] = round(dev, 2)

    return weekly


# ---------------------------------------------------------------------------
# Anomaly flagging
# ---------------------------------------------------------------------------

def _flag_anomalies(weekly: pd.DataFrame) -> pd.DataFrame:
    """Set `anomaly_flag` = True for route-weeks exceeding EITHER threshold.

    A route-week with insufficient history (vs_own_history_pct is NaN) is
    only flagged via the peer comparison signal, and vice-versa.
    """
    weekly = weekly.copy()

    own_flag = (
        weekly["vs_own_history_pct"].notna()
        & (weekly["vs_own_history_pct"] >= HISTORY_THRESHOLD_PCT)
    )
    peer_flag = (
        weekly["vs_similar_routes_pct"].notna()
        & (weekly["vs_similar_routes_pct"] >= PEER_THRESHOLD_PCT)
    )

    weekly["anomaly_flag"] = own_flag | peer_flag
    return weekly


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_anomalies(weekly: pd.DataFrame) -> pd.DataFrame:
    """Run the full anomaly detection pipeline.

    Parameters
    ----------
    weekly : pd.DataFrame
        Output of data_processor.build_weekly_summary()

    Returns
    -------
    pd.DataFrame with all original columns plus:
        history_baseline, vs_own_history_pct,
        peer_baseline,    vs_similar_routes_pct,
        anomaly_flag
    """
    df = _compute_own_history(weekly)
    df = _compute_peer_comparison(df)
    df = _flag_anomalies(df)
    return df


def format_pct_string(value: float | None, label: str) -> str:
    """Produce the human-readable percentage string used in the output CSV.

    Examples
    --------
    format_pct_string(35.5, "this route's past average")
    → '+35.5% vs this route's past average'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A (insufficient data)"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}% vs {label}"
