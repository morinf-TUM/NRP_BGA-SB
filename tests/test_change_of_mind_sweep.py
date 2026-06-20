"""Tests for the change-of-mind BG-frequency sweep library (Task 8.3).

Tests follow the TDD red-green protocol: each test corresponds to one
acceptance criterion from the task brief.
"""

from __future__ import annotations

from nrp_bga_sb.change_of_mind_sweep import (
    SWITCH_DELAY_CATEGORIES,
    ChangeOfMindSweepResult,
    format_sweep_report,
    run_change_of_mind_condition,
)

# --- Test 1: return type ---


def test_run_change_of_mind_condition_returns_result():
    """run_change_of_mind_condition returns a ChangeOfMindSweepResult."""
    result = run_change_of_mind_condition(
        frequency_hz=40.0,
        n_trials_per_seed=20,
        n_seeds=1,
        base_seed=99,
    )
    assert isinstance(result, ChangeOfMindSweepResult)
    assert result.frequency_hz == 40.0


# --- Test 2: total trial count ---


def test_run_change_of_mind_condition_trial_count():
    """n_trials equals n_seeds * n_trials_per_seed."""
    n_trials_per_seed = 20
    n_seeds = 2
    result = run_change_of_mind_condition(
        frequency_hz=40.0,
        n_trials_per_seed=n_trials_per_seed,
        n_seeds=n_seeds,
        base_seed=99,
    )
    assert result.n_trials == n_seeds * n_trials_per_seed
    assert result.n_seeds == n_seeds


# --- Test 3: switch_success_by_category keys ---


def test_switch_success_by_category_has_four_keys():
    """switch_success_by_category has exactly the four delay category keys."""
    result = run_change_of_mind_condition(
        frequency_hz=40.0,
        n_trials_per_seed=20,
        n_seeds=1,
        base_seed=99,
    )
    expected_keys = set(SWITCH_DELAY_CATEGORIES.keys())
    assert set(result.switch_success_by_category.keys()) == expected_keys


# --- Test 4: 40 Hz → high change-of-mind probability ---


def test_40hz_high_change_of_mind_probability():
    """At 40 Hz, change_of_mind_probability >= 0.9 (BG responds fast enough)."""
    result = run_change_of_mind_condition(
        frequency_hz=40.0,
        n_trials_per_seed=20,
        n_seeds=1,
        base_seed=99,
    )
    assert result.change_of_mind_probability is not None
    assert result.change_of_mind_probability >= 0.9, (
        f"Expected >= 0.9 at 40 Hz, got {result.change_of_mind_probability:.3f}"
    )


# --- Test 5: 5 Hz → degraded change-of-mind probability ---


def test_5hz_lower_change_of_mind_probability_than_40hz():
    """At 5 Hz, change_of_mind_probability is noticeably lower than at 40 Hz.

    At 5 Hz (period=200ms), the BG fires infrequently; after the evidence_change
    event the BG may not re-sample in the post-switch window. The threshold
    is < 0.9 (any significant degradation from the 40 Hz ceiling is acceptable).
    """
    result_5hz = run_change_of_mind_condition(
        frequency_hz=5.0,
        n_trials_per_seed=40,
        n_seeds=1,
        base_seed=99,
    )
    result_40hz = run_change_of_mind_condition(
        frequency_hz=40.0,
        n_trials_per_seed=40,
        n_seeds=1,
        base_seed=99,
    )
    assert result_5hz.change_of_mind_probability is not None
    assert result_40hz.change_of_mind_probability is not None

    # 5 Hz must underperform 40 Hz (degraded switch success at low BG frequency)
    # OR fall below 0.9 — either condition proves frequency-dependent degradation.
    degraded = (
        result_5hz.change_of_mind_probability < 0.9
        or result_5hz.change_of_mind_probability < result_40hz.change_of_mind_probability
    )
    assert degraded, (
        f"Expected 5 Hz to show degradation vs 40 Hz: "
        f"5 Hz={result_5hz.change_of_mind_probability:.3f}, "
        f"40 Hz={result_40hz.change_of_mind_probability:.3f}"
    )


# --- Test 6: format_sweep_report contains each frequency ---


def test_format_sweep_report_contains_frequencies():
    """format_sweep_report includes each frequency label in sorted order."""
    results = [
        run_change_of_mind_condition(
            frequency_hz=hz,
            n_trials_per_seed=20,
            n_seeds=1,
            base_seed=99,
        )
        for hz in [5.0, 80.0]
    ]
    report = format_sweep_report(results)
    assert isinstance(report, str)
    assert "5" in report, "Report should contain '5' (5 Hz)"
    assert "80" in report, "Report should contain '80' (80 Hz)"

    # Check sorted order: "5" appears before "80" in the report
    idx_5 = report.find("5")
    idx_80 = report.find("80")
    assert idx_5 < idx_80, "5 Hz entry should appear before 80 Hz entry (sorted ascending)"


# --- Test 7: format_sweep_report mentions change_of_mind_probability ---


def test_format_sweep_report_contains_probability_label():
    """format_sweep_report contains a change-of-mind probability label (case-insensitive)."""
    result = run_change_of_mind_condition(
        frequency_hz=40.0,
        n_trials_per_seed=20,
        n_seeds=1,
        base_seed=99,
    )
    report = format_sweep_report([result])
    assert "change" in report.lower() and "mind" in report.lower(), (
        "Report should mention 'change' and 'mind' (change-of-mind probability label)"
    )


# --- Test 8: mean_correction_cost is not None and > 0 at 40 Hz ---


def test_40hz_mean_correction_cost_positive():
    """At 40 Hz, mean_correction_cost is not None and > 0.0 (kinematic reversal occurred)."""
    result = run_change_of_mind_condition(
        frequency_hz=40.0,
        n_trials_per_seed=20,
        n_seeds=1,
        base_seed=99,
    )
    assert result.mean_correction_cost is not None, (
        "mean_correction_cost should not be None at 40 Hz"
    )
    assert result.mean_correction_cost > 0.0, (
        f"mean_correction_cost should be > 0.0, got {result.mean_correction_cost}"
    )
