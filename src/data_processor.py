"""
data_processor.py
-----------------
Reads and pre-processes shipment_records.csv.

Responsibilities:
  - Load raw CSV into a validated DataFrame
  - Derive the composite route label (origin-destination)
  - Compute cost_per_tonne_km for every shipment
  - Group into Monday-anchored ISO weeks
  - Aggregate to one row per (route, week_of)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_validate(csv_path: str | Path) -> pd.DataFrame:
    """Load shipment_records.csv, validate required columns, and return a
    clean DataFrame with derived fields already attached.

    Raises
    ------
    FileNotFoundError  – if the file does not exist
    ValueError         – if a required column is missing
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Shipment records not found: {path}")

    df = pd.read_csv(path, parse_dates=["shipment_date"])

    required_cols = {
        "shipment_id", "origin", "destination", "route_type",
        "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows where core numerics are null or non-positive
    numeric_cols = ["quantity_tonnes", "distance_km", "freight_cost_inr"]
    initial_len = len(df)
    df = df.dropna(subset=numeric_cols)
    df = df[(df["quantity_tonnes"] > 0) & (df["distance_km"] > 0) & (df["freight_cost_inr"] > 0)]

    if len(df) < initial_len:
        dropped = initial_len - len(df)
        print(f"[data_processor] Dropped {dropped} rows with invalid numerics.")

    # Derived columns
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    df["cost_per_tonne_km"] = df["freight_cost_inr"] / df["tonne_km"]

    # Monday-anchored week (ISO week start = Monday)
    df["week_of"] = df["shipment_date"].dt.to_period("W-SUN").apply(
        lambda p: p.start_time.normalize()
    )

    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shipment-level data to one row per (route, week_of).

    cost_per_tonne_km is computed from the *summed* totals so that larger
    shipments have appropriate weight – this is the correct formula:
        cost_per_tonne_km = sum(freight_cost_inr) / sum(tonne_km)

    Returns a DataFrame with columns:
        route, week_of, route_type,
        total_freight_cost, total_tonne_km, cost_per_tonne_km, shipment_count
    """
    agg = (
        df.groupby(["route", "week_of", "route_type"], as_index=False)
        .agg(
            total_freight_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
    )

    # Guard against zero denominator (shouldn't happen after validation)
    agg["cost_per_tonne_km"] = np.where(
        agg["total_tonne_km"] > 0,
        agg["total_freight_cost"] / agg["total_tonne_km"],
        np.nan,
    )

    agg = agg.sort_values(["route", "week_of"]).reset_index(drop=True)
    return agg


def build_weekly_summary(csv_path: str | Path) -> pd.DataFrame:
    """End-to-end convenience wrapper: load → validate → aggregate."""
    raw = load_and_validate(csv_path)
    weekly = aggregate_weekly(raw)
    return weekly
