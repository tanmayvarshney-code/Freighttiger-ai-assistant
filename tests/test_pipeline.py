"""
test_pipeline.py
----------------
Comprehensive pytest test suite for the FreightTiger AI pipeline.

Tests cover:
  - Data parsing and validation
  - Weekly cost aggregation
  - 8-week rolling baseline calculation
  - Similar-route peer comparison
  - Edge cases: missing data, zero denominators, single rows
  - Anomaly detection thresholds
  - Note retrieval: correct match, incorrect rejection
  - LLM response validation
  - API failure fallback
  - Output CSV schema
  - Reproducibility (numeric stability)
"""

from __future__ import annotations

import io
import os
import math
import hashlib
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processor import load_and_validate, aggregate_weekly, build_weekly_summary
from src.anomaly_detector import detect_anomalies, format_pct_string, HISTORY_THRESHOLD_PCT, PEER_THRESHOLD_PCT
from src.note_retriever import load_notes, retrieve_note, attach_notes, _note_has_cost_increase_signal
from src.groq_explainer import (
    _build_user_prompt, _validate_response, _fallback_explanation,
    _cache_key, generate_explanation,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_shipment_csv(tmp_path):
    """Create a minimal but realistic shipment CSV for testing."""
    content = (
        "shipment_id,origin,destination,route_type,material,"
        "quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter\n"
        # Week 1 (2024-01-01) – Mumbai-Pune, Short
        "SHP001,Mumbai,Pune,Short,Cement,10.0,150.0,5000,2024-01-01,Co A\n"
        "SHP002,Mumbai,Pune,Short,Steel,12.0,150.0,6000,2024-01-02,Co A\n"
        # Week 1 – Delhi-Jaipur, Short
        "SHP003,Delhi,Jaipur,Short,Cement,20.0,280.0,14000,2024-01-01,Co B\n"
        # Week 2
        "SHP004,Mumbai,Pune,Short,Cement,10.0,150.0,5100,2024-01-08,Co A\n"
        "SHP005,Delhi,Jaipur,Short,Cement,20.0,280.0,14200,2024-01-08,Co B\n"
    )
    p = tmp_path / "shipment_records.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_notes_csv(tmp_path):
    content = (
        "note_id,date,applies_to,note\n"
        "N001,2024-01-03,Mumbai-Pune,\"Heavy flooding on the Mumbai-Pune highway disrupted truck movement causing higher detour costs.\"\n"
        "N002,2024-01-03,All Routes,\"Diesel prices rose nationwide pushing transportation costs higher across all routes.\"\n"
        "N003,2024-06-01,Delhi-Jaipur,\"No significant disruptions were reported; freight movement remained normal.\"\n"
        "N004,2023-01-01,Mumbai-Pune,\"Very old note about flooding – outside any reasonable date window.\"\n"
    )
    p = tmp_path / "context_notes.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def large_shipment_csv(tmp_path):
    """Generate 10 weeks of data for two Short routes to test 8-week baseline."""
    rows = ["shipment_id,origin,destination,route_type,material,quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter"]
    base_cost = 5000.0
    idx = 1
    for week in range(10):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=week)
        # Mumbai-Pune: normal cost for weeks 0-8, spike on week 9
        cost_mp = base_cost if week < 9 else base_cost * 1.40   # +40% spike
        rows.append(f"SHP{idx:04d},Mumbai,Pune,Short,Cement,10.0,150.0,{cost_mp:.0f},{date.date()},Co A")
        idx += 1
        # Delhi-Jaipur: stable cost throughout
        rows.append(f"SHP{idx:04d},Delhi,Jaipur,Short,Cement,20.0,280.0,14000,{date.date()},Co B")
        idx += 1

    p = tmp_path / "shipment_records.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


# ===========================================================================
# 1. Data Parsing & Validation
# ===========================================================================

class TestDataParsing:
    def test_load_valid_csv(self, sample_shipment_csv):
        df = load_and_validate(sample_shipment_csv)
        assert len(df) > 0
        assert "route" in df.columns
        assert "cost_per_tonne_km" in df.columns
        assert "week_of" in df.columns

    def test_route_derived_correctly(self, sample_shipment_csv):
        df = load_and_validate(sample_shipment_csv)
        assert "Mumbai-Pune" in df["route"].values
        assert "Delhi-Jaipur" in df["route"].values

    def test_cost_per_tonne_km_formula(self, sample_shipment_csv):
        """cost_per_tonne_km = freight_cost_inr / (quantity_tonnes * distance_km)"""
        df = load_and_validate(sample_shipment_csv)
        row = df[df["shipment_id"] == "SHP001"].iloc[0]
        expected = 5000.0 / (10.0 * 150.0)
        assert abs(row["cost_per_tonne_km"] - expected) < 1e-6

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_validate(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("shipment_id,origin\nSHP001,Mumbai\n")
        # pandas raises its own error for missing parse_dates column before our check;
        # we accept any error (ValueError or otherwise) since the load must fail
        with pytest.raises(Exception):
            load_and_validate(bad_csv)

    def test_zero_quantity_dropped(self, tmp_path):
        content = (
            "shipment_id,origin,destination,route_type,material,"
            "quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter\n"
            "SHP001,Mumbai,Pune,Short,Cement,0.0,150.0,5000,2024-01-01,Co A\n"
            "SHP002,Mumbai,Pune,Short,Cement,10.0,150.0,5000,2024-01-02,Co A\n"
        )
        p = tmp_path / "s.csv"
        p.write_text(content)
        df = load_and_validate(p)
        assert len(df) == 1  # only the valid row remains

    def test_week_of_is_monday(self, sample_shipment_csv):
        df = load_and_validate(sample_shipment_csv)
        for wk in df["week_of"]:
            assert wk.weekday() == 0, f"week_of={wk} is not a Monday"


# ===========================================================================
# 2. Weekly Aggregation
# ===========================================================================

class TestWeeklyAggregation:
    def test_aggregation_reduces_rows(self, sample_shipment_csv):
        df = load_and_validate(sample_shipment_csv)
        weekly = aggregate_weekly(df)
        # Two shipments in week-1 for Mumbai-Pune → merged into 1 row
        mp_w1 = weekly[(weekly["route"] == "Mumbai-Pune") & (weekly["week_of"] == pd.Timestamp("2024-01-01"))]
        assert len(mp_w1) == 1

    def test_aggregated_cost_uses_summed_totals(self, sample_shipment_csv):
        """cost_per_tonne_km must be sum(cost)/sum(tonne_km), not mean of individual rows."""
        df = load_and_validate(sample_shipment_csv)
        weekly = aggregate_weekly(df)
        mp_w1 = weekly[(weekly["route"] == "Mumbai-Pune") & (weekly["week_of"] == pd.Timestamp("2024-01-01"))].iloc[0]
        expected = (5000 + 6000) / (10 * 150 + 12 * 150)
        assert abs(mp_w1["cost_per_tonne_km"] - expected) < 1e-4

    def test_shipment_count_column_present(self, sample_shipment_csv):
        df = load_and_validate(sample_shipment_csv)
        weekly = aggregate_weekly(df)
        assert "shipment_count" in weekly.columns


# ===========================================================================
# 3. Eight-Week Historical Baseline
# ===========================================================================

class TestEightWeekBaseline:
    def test_first_week_has_no_baseline(self, large_shipment_csv):
        weekly = build_weekly_summary(large_shipment_csv)
        detected = detect_anomalies(weekly)
        mp = detected[detected["route"] == "Mumbai-Pune"].sort_values("week_of")
        # Row 0 should have NaN baseline (no prior weeks)
        assert pd.isna(mp.iloc[0]["history_baseline"])

    def test_baseline_excludes_current_week(self, large_shipment_csv):
        weekly = build_weekly_summary(large_shipment_csv)
        detected = detect_anomalies(weekly)
        mp = detected[detected["route"] == "Mumbai-Pune"].sort_values("week_of")
        # Week index 1 should use only week 0 as baseline
        baseline_wk1 = mp.iloc[1]["history_baseline"]
        cost_wk0 = mp.iloc[0]["cost_per_tonne_km"]
        assert abs(baseline_wk1 - cost_wk0) < 1e-4

    def test_baseline_uses_max_8_weeks(self, large_shipment_csv):
        weekly = build_weekly_summary(large_shipment_csv)
        detected = detect_anomalies(weekly)
        mp = detected[detected["route"] == "Mumbai-Pune"].sort_values("week_of")
        # Row 9 (index 9) should use rows 1-8 (8 weeks) as baseline
        baseline_wk9 = mp.iloc[9]["history_baseline"]
        expected = float(mp.iloc[1:9]["cost_per_tonne_km"].mean())
        assert abs(baseline_wk9 - expected) < 1e-4

    def test_no_future_leakage(self, large_shipment_csv):
        """Baseline for week N must not include any data from week N onwards."""
        weekly = build_weekly_summary(large_shipment_csv)
        detected = detect_anomalies(weekly)
        mp = detected[detected["route"] == "Mumbai-Pune"].sort_values("week_of").reset_index(drop=True)
        for i in range(1, len(mp)):
            # Baseline for row i must be <= mean of rows 0..i-1
            past_mean = float(mp.iloc[max(0, i-8):i]["cost_per_tonne_km"].mean())
            assert abs(mp.iloc[i]["history_baseline"] - past_mean) < 1e-4

    def test_single_row_no_crash(self, tmp_path):
        content = (
            "shipment_id,origin,destination,route_type,material,"
            "quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter\n"
            "SHP001,Mumbai,Pune,Short,Cement,10.0,150.0,5000,2024-01-01,Co A\n"
        )
        p = tmp_path / "s.csv"
        p.write_text(content)
        weekly = build_weekly_summary(p)
        detected = detect_anomalies(weekly)
        assert len(detected) == 1


# ===========================================================================
# 4. Similar-Route Peer Comparison
# ===========================================================================

class TestPeerComparison:
    def test_peer_excludes_self(self, large_shipment_csv):
        weekly = build_weekly_summary(large_shipment_csv)
        detected = detect_anomalies(weekly)
        # Mumbai-Pune peer should be Delhi-Jaipur only (same type = Short)
        mp = detected[(detected["route"] == "Mumbai-Pune")].iloc[0]
        dj = detected[(detected["route"] == "Delhi-Jaipur") &
                      (detected["week_of"] == mp["week_of"])].iloc[0]
        # Peer baseline for Mumbai-Pune ≈ cost of Delhi-Jaipur in same week
        assert abs(mp["peer_baseline"] - dj["cost_per_tonne_km"]) < 1e-4

    def test_no_peer_gives_nan(self, tmp_path):
        """A route with no same-type peers should have NaN peer_baseline."""
        content = (
            "shipment_id,origin,destination,route_type,material,"
            "quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter\n"
            "SHP001,Mumbai,Pune,Short,Cement,10.0,150.0,5000,2024-01-01,Co A\n"
        )
        p = tmp_path / "s.csv"
        p.write_text(content)
        weekly = build_weekly_summary(p)
        detected = detect_anomalies(weekly)
        assert pd.isna(detected.iloc[0]["peer_baseline"])


# ===========================================================================
# 5. Anomaly Detection
# ===========================================================================

class TestAnomalyDetection:
    def test_spike_above_threshold_flagged(self, large_shipment_csv):
        """Mumbai-Pune week 9 has a +40% spike – must be flagged."""
        weekly = build_weekly_summary(large_shipment_csv)
        detected = detect_anomalies(weekly)
        mp = detected[detected["route"] == "Mumbai-Pune"].sort_values("week_of")
        last_row = mp.iloc[-1]
        assert last_row["anomaly_flag"] == True
        assert last_row["vs_own_history_pct"] >= HISTORY_THRESHOLD_PCT

    def test_normal_week_not_flagged(self, large_shipment_csv):
        """Stable weeks (0-8) on Delhi-Jaipur must not be flagged."""
        weekly = build_weekly_summary(large_shipment_csv)
        detected = detect_anomalies(weekly)
        dj = detected[detected["route"] == "Delhi-Jaipur"].sort_values("week_of")
        # Only rows with sufficient history (row index >= 1) are testable
        for _, row in dj.iloc[1:].iterrows():
            assert row["anomaly_flag"] == False

    def test_format_pct_string_positive(self):
        s = format_pct_string(35.5, "this route's past average")
        assert s == "+35.5% vs this route's past average"

    def test_format_pct_string_negative(self):
        s = format_pct_string(-10.2, "similar-length routes this week")
        assert s == "-10.2% vs similar-length routes this week"

    def test_format_pct_string_nan(self):
        s = format_pct_string(float("nan"), "label")
        assert "N/A" in s or "insufficient" in s.lower()


# ===========================================================================
# 6. Note Retrieval
# ===========================================================================

class TestNoteRetrieval:
    def test_correct_note_returned(self, sample_notes_csv):
        notes = load_notes(sample_notes_csv)
        note_id, note_text = retrieve_note(
            route="Mumbai-Pune",
            week_of=pd.Timestamp("2024-01-01"),
            notes_df=notes,
        )
        assert note_id == "N001"
        assert note_text is not None
        assert len(note_text) > 10

    def test_all_routes_note_returned_when_no_specific(self, sample_notes_csv):
        """Delhi-Jaipur has no specific note; should match 'All Routes' N002."""
        notes = load_notes(sample_notes_csv)
        note_id, note_text = retrieve_note(
            route="Delhi-Jaipur",
            week_of=pd.Timestamp("2024-01-01"),
            notes_df=notes,
        )
        # N002 is the 'All Routes' note with a cost-increase signal
        assert note_id == "N002"

    def test_no_clear_note_rejected(self, sample_notes_csv):
        """N003 says 'no significant disruptions' – must not be returned as justification."""
        notes = load_notes(sample_notes_csv)
        note_id, _ = retrieve_note(
            route="Delhi-Jaipur",
            week_of=pd.Timestamp("2024-06-01"),
            notes_df=notes,
        )
        assert note_id != "N003"

    def test_old_note_outside_date_window_rejected(self, sample_notes_csv):
        """N004 is from 2023-01-01 – far outside ±35 days of 2024-01-01."""
        notes = load_notes(sample_notes_csv)
        # Force retrieval only with old notes
        old_notes = notes[notes["note_id"] == "N004"].copy()
        note_id, _ = retrieve_note("Mumbai-Pune", pd.Timestamp("2024-01-01"), old_notes)
        assert note_id is None

    def test_empty_notes_returns_none(self):
        empty_notes = pd.DataFrame(columns=["note_id", "date", "applies_to", "note", "_tokens"])
        note_id, note_text = retrieve_note("Mumbai-Pune", pd.Timestamp("2024-01-01"), empty_notes)
        assert note_id is None
        assert note_text is None

    def test_cost_increase_signal_detection(self):
        assert _note_has_cost_increase_signal("Flooding caused higher detour costs") is True
        assert _note_has_cost_increase_signal("Diesel prices rose nationwide") is True
        assert _note_has_cost_increase_signal("No significant disruptions; movement remained normal") is False
        assert _note_has_cost_increase_signal("Freight movement remained stable") is False

    def test_load_notes_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_notes(tmp_path / "missing_notes.csv")


# ===========================================================================
# 7. Groq Explainer – Prompt, Validation, Fallback
# ===========================================================================

class TestGroqExplainer:
    def test_prompt_contains_required_facts(self):
        prompt = _build_user_prompt(
            route="Mumbai-Pune",
            week_of="2024-01-01",
            cost=3.33,
            hist_pct=35.5,
            peer_pct=21.0,
            flagged=True,
            note_id="N001",
            note_text="Flooding caused higher costs.",
        )
        assert "Mumbai-Pune" in prompt
        assert "2024-01-01" in prompt
        assert "3.3300" in prompt
        assert "+35.5%" in prompt
        assert "+21.0%" in prompt
        assert "N001" in prompt

    def test_valid_response_accepted(self):
        is_valid, cleaned = _validate_response(
            "The cost spike is due to N001 flooding disruption on this route.",
            note_id="N001",
        )
        assert is_valid
        assert len(cleaned) > 0

    def test_invented_note_id_rejected(self):
        is_valid, _ = _validate_response(
            "This matches note N999 which explains the cost rise.",
            note_id="N001",
        )
        assert not is_valid

    def test_empty_response_rejected(self):
        is_valid, _ = _validate_response("", note_id=None)
        assert not is_valid

    def test_fallback_with_note(self):
        text = _fallback_explanation(
            route="Mumbai-Pune",
            week_of="2024-01-01",
            hist_pct=35.5,
            peer_pct=21.0,
            flagged=True,
            note_id="N001",
            note_text="Flooding caused higher costs.",
        )
        assert "N001" in text
        assert len(text) > 20

    def test_fallback_without_note_flagged(self):
        text = _fallback_explanation(
            route="Delhi-Jaipur",
            week_of="2024-11-11",
            hist_pct=35.5,
            peer_pct=21.0,
            flagged=True,
            note_id=None,
            note_text=None,
        )
        assert "flagged" in text.lower() or "review" in text.lower() or "unexplained" in text.lower() or "35.5" in text

    def test_cache_key_is_stable(self):
        k1 = _cache_key("model", "Mumbai-Pune", "2024-01-01", 3.33, 35.5, 21.0, "N001", "flood")
        k2 = _cache_key("model", "Mumbai-Pune", "2024-01-01", 3.33, 35.5, 21.0, "N001", "flood")
        assert k1 == k2

    def test_cache_key_differs_on_different_inputs(self):
        k1 = _cache_key("model", "Mumbai-Pune", "2024-01-01", 3.33, 35.5, 21.0, "N001", "flood")
        k2 = _cache_key("model", "Mumbai-Pune", "2024-01-01", 3.33, 36.0, 21.0, "N001", "flood")
        assert k1 != k2

    def test_api_failure_returns_fallback(self, tmp_path):
        """When Groq raises an exception, a fallback is returned (not a crash)."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key", "GROQ_MODEL": "llama-3.1-8b-instant"}):
            # Groq is imported lazily inside generate_explanation; patch at the groq package level
            with patch("groq.Groq") as mock_groq_cls:
                mock_groq_cls.return_value.chat.completions.create.side_effect = Exception("Connection error")
                explanation, cache, used_api = generate_explanation(
                    route="Mumbai-Pune",
                    week_of="2024-01-01",
                    cost=3.33,
                    hist_pct=35.5,
                    peer_pct=21.0,
                    flagged=True,
                    note_id=None,
                    note_text=None,
                    cache={},
                )
        assert len(explanation) > 0
        assert used_api is False

    def test_missing_api_key_uses_fallback(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        explanation, cache, used_api = generate_explanation(
            route="Mumbai-Pune",
            week_of="2024-01-01",
            cost=3.33,
            hist_pct=35.5,
            peer_pct=None,
            flagged=True,
            note_id=None,
            note_text=None,
            cache={},
        )
        assert len(explanation) > 0
        assert used_api is False


# ===========================================================================
# 8. Output CSV Schema
# ===========================================================================

class TestOutputSchema:
    REQUIRED_COLUMNS = [
        "route", "week_of", "cost_per_tonne_km",
        "vs_own_history", "vs_similar_routes",
        "flagged", "matched_note_id", "reason",
    ]

    def test_output_has_correct_columns(self, tmp_path, large_shipment_csv, sample_notes_csv):
        from src.pipeline import run_pipeline
        output_path = tmp_path / "output.csv"
        run_pipeline(
            shipment_csv=large_shipment_csv,
            notes_csv=sample_notes_csv,
            output_path=output_path,
            verbose=False,
        )
        df = pd.read_csv(output_path)
        for col in self.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_output_column_order_matches_spec(self, tmp_path, large_shipment_csv, sample_notes_csv):
        from src.pipeline import run_pipeline
        output_path = tmp_path / "output.csv"
        run_pipeline(
            shipment_csv=large_shipment_csv,
            notes_csv=sample_notes_csv,
            output_path=output_path,
            verbose=False,
        )
        df = pd.read_csv(output_path)
        assert list(df.columns) == self.REQUIRED_COLUMNS

    def test_no_blank_routes_in_output(self, tmp_path, large_shipment_csv, sample_notes_csv):
        from src.pipeline import run_pipeline
        output_path = tmp_path / "output.csv"
        run_pipeline(
            shipment_csv=large_shipment_csv,
            notes_csv=sample_notes_csv,
            output_path=output_path,
            verbose=False,
        )
        df = pd.read_csv(output_path)
        assert df["route"].notna().all()


# ===========================================================================
# 9. Reproducibility
# ===========================================================================

class TestReproducibility:
    def test_numeric_results_identical_on_repeat(self, tmp_path, large_shipment_csv, sample_notes_csv):
        """Running the pipeline twice must yield byte-for-byte identical numeric columns."""
        from src.pipeline import run_pipeline

        def fingerprint(path):
            df = pd.read_csv(path)
            cols = ["route", "week_of", "cost_per_tonne_km",
                    "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]
            return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()

        p1 = tmp_path / "out1.csv"
        p2 = tmp_path / "out2.csv"

        run_pipeline(large_shipment_csv, sample_notes_csv, p1, verbose=False)
        run_pipeline(large_shipment_csv, sample_notes_csv, p2, verbose=False)

        assert fingerprint(p1) == fingerprint(p2), "Numeric results differ between runs!"

    def test_pipeline_runs_without_api_key(self, tmp_path, large_shipment_csv, sample_notes_csv, monkeypatch):
        """Pipeline must complete and write output.csv even without a Groq API key."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from src.pipeline import run_pipeline
        output_path = tmp_path / "output_no_api.csv"
        metrics = run_pipeline(large_shipment_csv, sample_notes_csv, output_path, verbose=False)
        assert output_path.exists()
        assert metrics["route_weeks_processed"] > 0
