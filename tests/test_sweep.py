"""Tests for the Phase 5 sweep condition runner (Task 5.1)."""

from __future__ import annotations

import pytest

from nrp_bga_sb.sweep import (
    CONFLICT_PEAK_SALIENCE,
    SweepConditionResult,
    run_condition,
)

# --- CONFLICT_PEAK_SALIENCE sanity ---


def test_conflict_peak_salience_ordering():
    """low > medium > high peak salience."""
    assert CONFLICT_PEAK_SALIENCE["low"] > CONFLICT_PEAK_SALIENCE["medium"]
    assert CONFLICT_PEAK_SALIENCE["medium"] > CONFLICT_PEAK_SALIENCE["high"]


def test_conflict_peak_salience_bounds():
    for level, val in CONFLICT_PEAK_SALIENCE.items():
        assert 0.0 < val < 1.0, f"{level}: {val} out of (0, 1)"


# --- run_condition return type ---


def test_run_condition_go_nogo_returns_result():
    result = run_condition(40.0, "low", "go_nogo", n_trials=10, seed=42)
    assert isinstance(result, SweepConditionResult)
    assert result.frequency_hz == 40.0
    assert result.conflict_level == "low"
    assert result.paradigm == "go_nogo"
    assert result.seed == 42
    assert result.n_trials == 10


def test_run_condition_two_choice_returns_result():
    result = run_condition(40.0, "medium", "two_choice", n_trials=10, seed=42)
    assert isinstance(result, SweepConditionResult)
    assert result.paradigm == "two_choice"
    assert result.n_trials == 10


# --- Phase-5 metrics are populated ---


def test_go_nogo_has_miss_rate_and_go_success_rate():
    result = run_condition(40.0, "low", "go_nogo", n_trials=20, seed=42)
    assert result.miss_rate is not None
    assert result.go_success_rate is not None
    assert result.timeout_rate is None  # go_nogo does not produce timeouts


def test_two_choice_has_timeout_rate():
    result = run_condition(40.0, "low", "two_choice", n_trials=20, seed=42)
    assert result.timeout_rate is not None
    assert result.miss_rate is None  # two_choice does not produce misses


# --- Determinism ---


def test_run_condition_deterministic():
    """Same seed must produce identical results."""
    r1 = run_condition(40.0, "medium", "go_nogo", n_trials=10, seed=99)
    r2 = run_condition(40.0, "medium", "go_nogo", n_trials=10, seed=99)
    assert r1.miss_rate == r2.miss_rate
    assert r1.go_success_rate == r2.go_success_rate
    assert r1.n_trials == r2.n_trials


# --- Frequency × conflict behavioral effects ---


def test_high_freq_low_conflict_go_nogo_minimal_misses():
    """160 Hz + low conflict: go trials should nearly always succeed."""
    result = run_condition(160.0, "low", "go_nogo", n_trials=30, seed=42)
    # At 160 Hz, BG sees >150 evidence ticks; low conflict peak=0.85 selects at tick~88
    assert result.miss_rate is not None
    assert result.miss_rate < 0.1, f"miss_rate={result.miss_rate} too high at 160Hz/low"


def test_low_freq_medium_conflict_go_nogo_misses():
    """10 Hz + medium conflict: BG fires at tick 0,100 → max salience 0.610 < 0.65 → all misses."""
    result = run_condition(10.0, "medium", "go_nogo", n_trials=30, seed=42)
    assert result.miss_rate is not None
    # All go trials miss because tick 100 gives [0.610, 0.390] below GPR threshold
    assert result.miss_rate > 0.8, f"miss_rate={result.miss_rate} too low at 10Hz/medium"


def test_frequency_conflict_interaction():
    """Low conflict succeeds at 10 Hz; medium conflict fails at 10 Hz."""
    low_10hz = run_condition(10.0, "low", "go_nogo", n_trials=30, seed=42)
    med_10hz = run_condition(10.0, "medium", "go_nogo", n_trials=30, seed=42)
    assert low_10hz.go_success_rate > med_10hz.go_success_rate


def test_false_alarm_rate_low_regardless_of_frequency():
    """No-go trials: cortex gives neutral evidence → BG withholds → no false alarms."""
    for freq in [10.0, 160.0]:
        result = run_condition(freq, "low", "go_nogo", n_trials=30, seed=42)
        # false_alarm_rate should be 0.0 (no directed evidence for no_go cues)
        assert result.false_alarm_rate is not None
        assert result.false_alarm_rate == 0.0, f"false_alarm_rate != 0 at {freq}Hz"


# --- Invalid inputs ---


def test_invalid_conflict_level_raises():
    with pytest.raises((KeyError, ValueError)):
        run_condition(40.0, "extreme", "go_nogo", n_trials=5, seed=1)  # type: ignore[arg-type]


def test_invalid_paradigm_raises():
    with pytest.raises((ValueError, KeyError)):
        run_condition(40.0, "low", "stop_signal", n_trials=5, seed=1)  # type: ignore[arg-type]
