"""Tests for stop_signal_sweep module (Task 7.4).

Uses small n (n_trials_per_seed=10, n_seeds=2) for speed.
Asserts structure and types only — does not assert specific metric values
that depend on internal BG calibration.
"""

from __future__ import annotations

import pytest

from nrp_bga_sb.stop_signal_metrics import StopSignalMetrics, StopSignalValidityReport
from nrp_bga_sb.stop_signal_sweep import (
    FREQUENCIES_HZ,
    N_SEEDS,
    N_TRIALS_PER_SEED,
    StopSignalSweepResult,
    format_sweep_report,
    run_stop_signal_condition,
)

# --- Constants ---


def test_frequencies_hz_has_five_elements():
    assert len(FREQUENCIES_HZ) == 5


def test_trial_count_meets_m5_requirement():
    # M5 acceptance criterion: >= 500 trials per frequency condition.
    assert N_SEEDS * N_TRIALS_PER_SEED >= 500


# --- run_stop_signal_condition return type and structure ---


@pytest.fixture(scope="module")
def result_80hz():
    """Run a small 80 Hz condition once and reuse across tests in this module."""
    return run_stop_signal_condition(80.0, n_trials_per_seed=10, n_seeds=2)


@pytest.fixture(scope="module")
def result_5hz():
    """Run a small 5 Hz condition once and reuse across tests in this module."""
    return run_stop_signal_condition(5.0, n_trials_per_seed=10, n_seeds=2)


def test_run_stop_signal_condition_returns_sweep_result(result_80hz):
    assert isinstance(result_80hz, StopSignalSweepResult)


def test_n_trials_equals_n_trials_per_seed_times_n_seeds(result_80hz):
    assert result_80hz.n_trials == 10 * 2


def test_n_seeds_field_matches_argument(result_80hz):
    assert result_80hz.n_seeds == 2


def test_frequency_hz_field_matches_argument(result_80hz):
    assert result_80hz.frequency_hz == 80.0


def test_metrics_is_stop_signal_metrics_instance(result_80hz):
    assert isinstance(result_80hz.metrics, StopSignalMetrics)


def test_validity_is_stop_signal_validity_report_instance(result_80hz):
    assert isinstance(result_80hz.validity, StopSignalValidityReport)


def test_n_stop_trials_greater_than_zero(result_80hz):
    # stop_proportion=0.25, 20 trials → expect ~5 stop trials
    assert result_80hz.metrics.n_stop_trials > 0


def test_go_trials_exist(result_80hz):
    # n_go_trials must be positive (there are go trials in the aggregated batch)
    assert result_80hz.metrics.n_go_trials > 0


def test_5hz_stop_failure_rate_is_float(result_5hz):
    # At 5 Hz BG cannot select → likely all stop failures; value can be 0 or 1 — just check type.
    # stop_failure_rate is None only when n_stop_trials == 0, which should not happen.
    assert result_5hz.metrics.stop_failure_rate is not None
    assert isinstance(result_5hz.metrics.stop_failure_rate, float)


def test_aggregated_n_trials_matches_seeds_times_trials(result_5hz):
    # Determinism check: aggregate trial count must equal n_seeds * n_trials_per_seed
    # regardless of stochastic per-seed distributions.
    assert result_5hz.n_trials == 10 * 2


# --- format_sweep_report ---


def test_format_sweep_report_returns_non_empty_string(result_80hz):
    report = format_sweep_report([result_80hz])
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_sweep_report_contains_hz(result_80hz):
    report = format_sweep_report([result_80hz])
    assert "Hz" in report


def test_format_sweep_report_contains_ssrt(result_80hz):
    report = format_sweep_report([result_80hz])
    # Brief requires "SSRT" or "ssrt" to appear in the report.
    assert "SSRT" in report or "ssrt" in report


def test_format_sweep_report_sorted_by_frequency():
    # Run two conditions at different frequencies and verify report order.
    r5 = run_stop_signal_condition(5.0, n_trials_per_seed=5, n_seeds=1)
    r80 = run_stop_signal_condition(80.0, n_trials_per_seed=5, n_seeds=1)
    report = format_sweep_report([r80, r5])  # deliberately out of order
    idx_5 = report.index("5")
    idx_80 = report.index("80")
    assert idx_5 < idx_80, "5 Hz entry should appear before 80 Hz entry in the report"
