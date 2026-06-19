"""Tests for the Phase 5 statistics module (Task 5.2)."""

from __future__ import annotations

import pytest

from nrp_bga_sb.stats import (
    aggregate_by_frequency,
    bootstrap_ci,
    fit_frequency_slope,
    format_sweep_report,
    reproducibility_check,
)
from nrp_bga_sb.sweep import SweepConditionResult

# --- bootstrap_ci ---


def test_bootstrap_ci_returns_tuple():
    lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_bootstrap_ci_lo_le_hi():
    lo, hi = bootstrap_ci([0.3] * 10)
    assert lo <= hi


def test_bootstrap_ci_mean_between_bounds():
    values = [0.0, 0.5, 1.0] * 10
    lo, hi = bootstrap_ci(values, n_bootstrap=1000, rng_seed=42)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_deterministic():
    v = [0.1, 0.4, 0.9, 0.2, 0.7]
    r1 = bootstrap_ci(v, rng_seed=7)
    r2 = bootstrap_ci(v, rng_seed=7)
    assert r1 == r2


def test_bootstrap_ci_wider_with_more_variance():
    lo_narrow, hi_narrow = bootstrap_ci([0.5] * 30, rng_seed=42)
    lo_wide, hi_wide = bootstrap_ci([0.0, 1.0] * 15, rng_seed=42)
    assert (hi_wide - lo_wide) > (hi_narrow - lo_narrow)


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


# --- aggregate_by_frequency ---


def _make_results(
    freqs: list[float],
    metric_val: float,
    paradigm: str = "go_nogo",
    conflict: str = "low",
) -> list[SweepConditionResult]:
    """Build minimal SweepConditionResult objects for testing."""
    results = []
    for i, freq in enumerate(freqs):
        results.append(
            SweepConditionResult(
                frequency_hz=freq,
                conflict_level=conflict,  # type: ignore[arg-type]
                paradigm=paradigm,  # type: ignore[arg-type]
                seed=i,
                n_trials=20,
                reaction_time_mean=None,
                wrong_action_rate=0.0,
                wrong_target_rate=0.0,
                false_alarm_rate=0.0,
                miss_rate=metric_val,
                timeout_rate=None,
                go_success_rate=1.0 - metric_val,
                bg_commitment_latency_mean=0.15,
                bg_commitment_latency_std=None,
            )
        )
    return results


def test_aggregate_by_frequency_groups_correctly():
    results = _make_results([10.0, 10.0, 20.0, 20.0], 0.5)
    curves = aggregate_by_frequency(results, "miss_rate")
    assert set(curves.keys()) == {10.0, 20.0}
    assert curves[10.0]["n"] == 2
    assert curves[20.0]["n"] == 2


def test_aggregate_by_frequency_correct_mean():
    results = _make_results([10.0, 10.0], 0.6) + _make_results([10.0, 10.0], 0.4)
    curves = aggregate_by_frequency(results, "miss_rate")
    assert abs(curves[10.0]["mean"] - 0.5) < 1e-9


def test_aggregate_filters_paradigm():
    go_nogo = _make_results([40.0], 0.3, paradigm="go_nogo")
    two_choice = _make_results([40.0], 0.9, paradigm="two_choice")
    curves = aggregate_by_frequency(go_nogo + two_choice, "miss_rate", paradigm="go_nogo")
    assert abs(curves[40.0]["mean"] - 0.3) < 1e-9


def test_aggregate_filters_conflict():
    low = _make_results([40.0], 0.1, conflict="low")
    high = _make_results([40.0], 0.9, conflict="high")
    curves = aggregate_by_frequency(low + high, "miss_rate", conflict_level="low")
    assert abs(curves[40.0]["mean"] - 0.1) < 1e-9


def test_aggregate_omits_all_none_metric():
    # timeout_rate is None in go_nogo results
    results = _make_results([40.0, 80.0], 0.5)
    curves = aggregate_by_frequency(results, "timeout_rate")
    assert len(curves) == 0


# --- fit_frequency_slope ---


def test_fit_frequency_slope_positive_trend():
    curves = {
        10.0: {"mean": 0.1},
        40.0: {"mean": 0.5},
        160.0: {"mean": 0.9},
    }
    slope = fit_frequency_slope(curves)
    assert slope > 0.0


def test_fit_frequency_slope_negative_trend():
    curves = {
        10.0: {"mean": 0.9},
        40.0: {"mean": 0.5},
        160.0: {"mean": 0.1},
    }
    slope = fit_frequency_slope(curves)
    assert slope < 0.0


def test_fit_frequency_slope_single_point_zero():
    curves = {40.0: {"mean": 0.5}}
    assert fit_frequency_slope(curves) == 0.0


# --- reproducibility_check ---


def test_reproducibility_check_identical_passes():
    r = _make_results([10.0, 20.0], 0.5)
    assert reproducibility_check(r, r[:]) is True


def test_reproducibility_check_different_fails():
    r1 = _make_results([10.0], 0.5)
    r2 = _make_results([10.0], 0.6)
    assert reproducibility_check(r1, r2) is False


def test_reproducibility_check_different_keys_fails():
    r1 = _make_results([10.0], 0.5)
    r2 = _make_results([20.0], 0.5)
    assert reproducibility_check(r1, r2) is False


# --- format_sweep_report ---


def test_format_sweep_report_returns_string():
    results = _make_results(
        [10.0, 10.0, 20.0], 0.8, paradigm="go_nogo", conflict="medium"
    ) + _make_results([10.0, 10.0, 20.0], 0.0, paradigm="two_choice", conflict="low")
    report = format_sweep_report(results, [10.0, 20.0], ["low", "medium", "high"])
    assert isinstance(report, str)
    assert "go_nogo" in report
    assert "two_choice" in report
