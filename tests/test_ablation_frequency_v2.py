"""Tests for the Phase 5 ablation experiment (Task 5.4).

Verifies:
  - run_knob_condition returns a valid dict for each knob × frequency combination.
  - The primary variable (input_sampling_hz at 5 Hz) produces miss_rate = 1.0.
  - All knobs at 5 Hz produce miss_rate = 1.0 (because 5 Hz × 200 ms accumulation
    = exactly 1 gate-fire at tick 0, where cortex evidence is still neutral).
  - All knobs at ≥ 10 Hz produce miss_rate = 0.0.
  - run_baseline returns miss_rate = 0.0.
  - Output JSON is saved correctly and contains the expected records.

Scientific note (primary variable):
  input_sampling_hz is the theoretically designated primary variable (PROJECT_MEMORY §5.1).
  Empirically, all four knobs show the same 5/10 Hz threshold in this experiment because
  the accumulation window (200 ms) equals the 5 Hz period (200 ticks), so any gate at 5 Hz
  fires only at tick 0 (neutral evidence). The isolation of input_sampling_hz as the
  mechanistic primary variable is confirmed by the pipeline architecture: it alone gates
  which cortical state enters the BG; the other three gates operate on already-sampled
  evidence and are therefore downstream of the bottleneck.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Experiments are not installed as a package; add the experiments dir to the path.
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from ablation_frequency_v2 import (
    FIXED_HZ,
    KNOB_NAMES,
    SWEEP_FREQUENCIES_HZ,
    run_baseline,
    run_knob_condition,
)

# --- Baseline tests ---


def test_baseline_returns_dict():
    """run_baseline returns a dict with required keys."""
    result = run_baseline()
    assert isinstance(result, dict)
    assert result["condition"] == "baseline"
    assert result["knob_name"] == "all"
    assert result["freq_hz"] == FIXED_HZ


def test_baseline_miss_rate_zero():
    """At 160 Hz, the closed loop succeeds on all go trials: miss_rate == 0.0."""
    result = run_baseline()
    assert result["miss_rate"] == pytest.approx(0.0)


def test_baseline_n_go_positive():
    """Baseline must have a non-zero count of go trials."""
    result = run_baseline()
    assert result["n_go_trials"] > 0


# --- run_knob_condition structural tests ---


def test_run_knob_condition_returns_dict():
    """run_knob_condition returns a dict with the expected keys."""
    result = run_knob_condition("input_sampling_hz", 160.0)
    assert set(result.keys()) == {"condition", "knob_name", "freq_hz", "miss_rate", "n_go_trials"}


def test_run_knob_condition_knob_name_preserved():
    """Knob name is preserved verbatim in the returned dict."""
    for knob in KNOB_NAMES:
        result = run_knob_condition(knob, 160.0)
        assert result["knob_name"] == knob


def test_run_knob_condition_freq_hz_preserved():
    """Frequency value is preserved in the returned dict."""
    for freq in SWEEP_FREQUENCIES_HZ:
        result = run_knob_condition("output_emission_hz", freq)
        assert result["freq_hz"] == freq


def test_run_knob_condition_condition_label():
    """The condition field is always 'ablation'."""
    result = run_knob_condition("integration_step_hz", 40.0)
    assert result["condition"] == "ablation"


def test_run_knob_condition_miss_rate_in_range():
    """miss_rate is in [0.0, 1.0] or NaN for all conditions."""
    result = run_knob_condition("commitment_update_hz", 20.0)
    mr = result["miss_rate"]
    assert math.isnan(mr) or 0.0 <= mr <= 1.0


# --- Primary variable: input_sampling_hz ---


def test_input_sampling_5hz_all_miss():
    """At 5 Hz input_sampling, BG reads only tick-0 neutral evidence → miss_rate == 1.0.

    Mechanism (PROJECT_MEMORY §20.6): period_ticks = round(1000/(5*1)) = 200.
    The accumulation window is 200 ms = 200 ticks. Gate 1 fires only at tick 0
    where the cortical ramp is still at [0.5, 0.5] (neutral). BGModel cannot
    select → committed_decision.selected_channel = -1 → all go trials miss.
    """
    result = run_knob_condition("input_sampling_hz", 5.0)
    assert result["miss_rate"] == pytest.approx(1.0), (
        f"Expected miss_rate=1.0 at 5 Hz input_sampling, got {result['miss_rate']}"
    )


def test_input_sampling_10hz_no_miss():
    """At 10 Hz input_sampling, Gate 1 fires at tick 0 (no select) and tick 100
    (salience risen above selection threshold) → miss_rate == 0.0."""
    result = run_knob_condition("input_sampling_hz", 10.0)
    assert result["miss_rate"] == pytest.approx(0.0), (
        f"Expected miss_rate=0.0 at 10 Hz input_sampling, got {result['miss_rate']}"
    )


def test_input_sampling_160hz_no_miss():
    """At 160 Hz (max), BG reads all ticks → miss_rate == 0.0."""
    result = run_knob_condition("input_sampling_hz", 160.0)
    assert result["miss_rate"] == pytest.approx(0.0)


# --- All knobs share the 5 Hz / 10 Hz boundary ---


@pytest.mark.parametrize("knob_name", KNOB_NAMES)
def test_all_knobs_5hz_miss_rate_one(knob_name: str):
    """Every knob at 5 Hz produces miss_rate == 1.0.

    At 5 Hz, period_ticks = 200, which equals the accumulation window length.
    The gate fires exactly once (tick 0) where evidence is neutral → no BG
    selection → all go trials miss.  This holds for all four knobs because each
    gate operates as the last bottleneck in the pipeline at 5 Hz.
    """
    result = run_knob_condition(knob_name, 5.0)
    assert result["miss_rate"] == pytest.approx(1.0), (
        f"Expected miss_rate=1.0 for {knob_name} at 5 Hz, got {result['miss_rate']}"
    )


@pytest.mark.parametrize("knob_name", KNOB_NAMES)
def test_all_knobs_10hz_no_miss(knob_name: str):
    """Every knob at 10 Hz produces miss_rate == 0.0.

    At 10 Hz, period_ticks = 100 so the gate fires at tick 0 and tick 100.
    By tick 100 the cortical ramp has risen above the BG selection threshold
    → the committed decision selects channel 0 → go trials succeed.
    """
    result = run_knob_condition(knob_name, 10.0)
    assert result["miss_rate"] == pytest.approx(0.0), (
        f"Expected miss_rate=0.0 for {knob_name} at 10 Hz, got {result['miss_rate']}"
    )


@pytest.mark.parametrize("knob_name", KNOB_NAMES)
def test_all_knobs_160hz_no_miss(knob_name: str):
    """Every knob at 160 Hz produces miss_rate == 0.0."""
    result = run_knob_condition(knob_name, 160.0)
    assert result["miss_rate"] == pytest.approx(0.0), (
        f"Expected miss_rate=0.0 for {knob_name} at 160 Hz, got {result['miss_rate']}"
    )


# --- JSON output tests ---


def test_json_output_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """main() saves a valid JSON file to the results path."""
    import ablation_frequency_v2 as mod

    results_path = tmp_path / "ablation_frequency_v2.json"
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "RESULTS_PATH", results_path)

    mod.main()

    assert results_path.exists(), "JSON output file was not created"
    data = json.loads(results_path.read_text())
    assert isinstance(data, list)
    # 1 baseline + 4 knobs × 6 frequencies = 25 records total
    assert len(data) == 25

    # Every record has the required keys
    required_keys = {"condition", "knob_name", "freq_hz", "miss_rate", "n_go_trials"}
    for rec in data:
        assert required_keys <= rec.keys(), f"Missing keys in record: {rec}"


def test_json_output_has_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """JSON output contains exactly one baseline record."""
    import ablation_frequency_v2 as mod

    results_path = tmp_path / "ablation_frequency_v2.json"
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "RESULTS_PATH", results_path)

    mod.main()

    data = json.loads(results_path.read_text())
    baselines = [r for r in data if r["condition"] == "baseline"]
    assert len(baselines) == 1


def test_json_output_all_knobs_covered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """JSON output covers all four knobs across all sweep frequencies."""
    import ablation_frequency_v2 as mod

    results_path = tmp_path / "ablation_frequency_v2.json"
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "RESULTS_PATH", results_path)

    mod.main()

    data = json.loads(results_path.read_text())
    ablation_records = [r for r in data if r["condition"] == "ablation"]

    for knob in KNOB_NAMES:
        knob_freqs = {r["freq_hz"] for r in ablation_records if r["knob_name"] == knob}
        assert knob_freqs == set(SWEEP_FREQUENCIES_HZ), (
            f"Knob {knob} missing frequencies: {set(SWEEP_FREQUENCIES_HZ) - knob_freqs}"
        )
