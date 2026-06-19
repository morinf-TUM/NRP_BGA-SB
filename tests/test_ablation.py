"""Tests for the frequency ablation sweep (Task 3.3).

Covers build_sweep_config, run_condition, run_knob_sweep, summarize_ablation,
and the run_full_ablation integration path.

The key scientific assertion (Test 18) verifies the null result: in the abstract
single-call constant-evidence model, all frequency conditions produce identical
metrics for the same engine and seed.
"""

from __future__ import annotations

import pytest

from nrp_bga_sb.ablation import (
    KNOB_NAMES,
    SWEEP_FREQUENCIES_HZ,
    build_sweep_config,
    run_condition,
    run_full_ablation,
    run_knob_sweep,
    summarize_ablation,
)
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import Metrics

# --- build_sweep_config tests ---


def test_build_sweep_config_output_emission_hz():
    """Knob output_emission_hz set to target; all others remain at BASELINE_HZ."""
    cfg = build_sweep_config("output_emission_hz", 40.0)
    assert cfg.output_emission_hz == 40.0
    assert cfg.input_sampling_hz == 160.0
    assert cfg.integration_step_hz == 160.0
    assert cfg.commitment_update_hz == 160.0


def test_build_sweep_config_commitment_update_hz():
    """Knob commitment_update_hz set to 10.0; all others remain at 160.0."""
    cfg = build_sweep_config("commitment_update_hz", 10.0)
    assert cfg.commitment_update_hz == 10.0
    assert cfg.input_sampling_hz == 160.0
    assert cfg.integration_step_hz == 160.0
    assert cfg.output_emission_hz == 160.0


def test_build_sweep_config_invalid_knob():
    """Unknown knob name raises ValueError."""
    with pytest.raises(ValueError, match="unknown knob"):
        build_sweep_config("nonexistent_hz", 40.0)


def test_build_sweep_config_all_knobs_roundtrip():
    """All four KNOB_NAMES produce a FrequencyConfig with the correct field set."""
    for knob in KNOB_NAMES:
        cfg = build_sweep_config(knob, 20.0)
        assert isinstance(cfg, FrequencyConfig)
        assert getattr(cfg, knob) == 20.0
        # All other knobs should be at baseline
        for other in KNOB_NAMES:
            if other != knob:
                assert getattr(cfg, other) == 160.0, (
                    f"Expected {other}=160.0 when sweeping {knob}, "
                    f"got {getattr(cfg, other)}"
                )


# --- run_condition tests ---


def test_run_condition_go_nogo_returns_metrics():
    """run_condition returns a Metrics object for go_nogo engine at 160 Hz."""
    freq_cfg = build_sweep_config("output_emission_hz", 160.0)
    result = run_condition("go_nogo", freq_cfg, n_trials=10, seed=42)
    assert isinstance(result, Metrics)
    assert result.n_trials == 10


def test_run_condition_two_choice_returns_metrics():
    """run_condition returns a Metrics object for two_choice engine at 10 Hz."""
    freq_cfg = build_sweep_config("output_emission_hz", 10.0)
    result = run_condition("two_choice", freq_cfg, n_trials=10, seed=42)
    assert isinstance(result, Metrics)
    assert result.n_trials == 10


def test_run_condition_stop_signal_returns_metrics():
    """run_condition returns a Metrics object for stop_signal engine at 40 Hz."""
    freq_cfg = build_sweep_config("output_emission_hz", 40.0)
    result = run_condition("stop_signal", freq_cfg, n_trials=10, seed=42)
    assert isinstance(result, Metrics)
    assert result.n_trials == 10


def test_run_condition_change_of_mind_returns_metrics():
    """run_condition returns a Metrics object for change_of_mind engine at 80 Hz."""
    freq_cfg = build_sweep_config("output_emission_hz", 80.0)
    result = run_condition("change_of_mind", freq_cfg, n_trials=10, seed=42)
    assert isinstance(result, Metrics)
    assert result.n_trials == 10


# --- run_knob_sweep tests ---


def test_run_knob_sweep_correct_keys():
    """run_knob_sweep returns dict with correct engine and frequency keys."""
    results = run_knob_sweep(
        "output_emission_hz", engine_names=["go_nogo"], n_trials=5, seed=42
    )
    assert "go_nogo" in results
    assert set(results["go_nogo"].keys()) == set(SWEEP_FREQUENCIES_HZ)


def test_run_knob_sweep_all_metrics_valid():
    """All returned Metrics objects are valid (not None, have a condition_id)."""
    results = run_knob_sweep(
        "output_emission_hz", engine_names=["go_nogo", "two_choice"], n_trials=5, seed=42
    )
    for engine_name, freq_map in results.items():
        for freq, metrics in freq_map.items():
            assert metrics is not None, f"None Metrics for {engine_name} @ {freq} Hz"
            assert isinstance(metrics, Metrics)
            assert metrics.condition_id == engine_name


def test_run_knob_sweep_null_result_same_distribution():
    """Null result: go_nogo metrics are identical across all five frequency conditions.

    In the abstract tick-0 model, all gates fire at tick 0 regardless of frequency,
    so constant-evidence trials produce identical outcomes across the sweep.
    """
    results = run_knob_sweep(
        "output_emission_hz", engine_names=["go_nogo"], n_trials=20, seed=42
    )
    freq_map = results["go_nogo"]
    freqs = list(freq_map.keys())
    baseline_metrics = freq_map[freqs[0]]

    for freq in freqs[1:]:
        m = freq_map[freq]
        assert m.reaction_time_mean == baseline_metrics.reaction_time_mean, (
            f"reaction_time_mean differs at {freq} Hz vs {freqs[0]} Hz"
        )
        assert m.false_alarm_rate == baseline_metrics.false_alarm_rate, (
            f"false_alarm_rate differs at {freq} Hz vs {freqs[0]} Hz"
        )


# --- summarize_ablation tests ---


def _minimal_ablation_results() -> dict:
    """Run a minimal ablation (5 trials, one engine) for summary tests."""
    return run_full_ablation(engine_names=["go_nogo"], n_trials=5, seed=42)


def test_summarize_ablation_returns_nonempty_string():
    """summarize_ablation returns a non-empty string."""
    results = _minimal_ablation_results()
    report = summarize_ablation(results)
    assert isinstance(report, str)
    assert len(report) > 0


def test_summarize_ablation_contains_output_emission_hz():
    """Report must mention output_emission_hz (primary variable)."""
    results = _minimal_ablation_results()
    report = summarize_ablation(results)
    assert "output_emission_hz" in report


def test_summarize_ablation_contains_primary_variable():
    """Report must state the primary variable assignment (case-insensitive)."""
    results = _minimal_ablation_results()
    report = summarize_ablation(results)
    assert "primary variable" in report.lower()


def test_summarize_ablation_documents_null_result():
    """Report must explain why metrics are flat (tick-0 / constant evidence)."""
    results = _minimal_ablation_results()
    report = summarize_ablation(results)
    found = any(
        phrase in report.lower()
        for phrase in ["tick-0", "tick 0", "constant evidence"]
    )
    assert found, (
        "Report must mention 'tick-0', 'tick 0', or 'constant evidence' "
        "to document the null result."
    )


# --- Integration tests ---


def test_run_full_ablation_returns_all_four_knobs():
    """run_full_ablation with n_trials=5 returns dict with all four knobs."""
    results = run_full_ablation(engine_names=["go_nogo"], n_trials=5, seed=42)
    assert set(results.keys()) == set(KNOB_NAMES)


def test_run_full_ablation_result_structure():
    """result['output_emission_hz']['go_nogo'][160.0] is a Metrics object."""
    results = run_full_ablation(engine_names=["go_nogo"], n_trials=5, seed=42)
    metrics = results["output_emission_hz"]["go_nogo"][160.0]
    assert isinstance(metrics, Metrics)


def test_run_full_ablation_null_result_across_frequencies():
    """Key scientific assertion: same engine + seed → identical metrics across all frequencies.

    Confirms that in the abstract model, the four frequency knobs do not yet produce
    behavioral differentiation. The tick-0 guarantee means all gates fire on the first
    tick regardless of period, so constant-evidence trials yield identical outcomes.
    """
    results = run_full_ablation(engine_names=["go_nogo"], n_trials=20, seed=42)
    freq_sweep = results["output_emission_hz"]["go_nogo"]

    freqs = list(freq_sweep.keys())
    m0 = freq_sweep[freqs[0]]

    for freq in freqs[1:]:
        m = freq_sweep[freq]
        assert m.reaction_time_mean == m0.reaction_time_mean, (
            f"mean_rt differs at {freq} Hz (null result violated)"
        )
        assert m.wrong_action_rate == m0.wrong_action_rate, (
            f"wrong_action_rate differs at {freq} Hz (null result violated)"
        )
        assert m.false_alarm_rate == m0.false_alarm_rate, (
            f"false_alarm_rate differs at {freq} Hz (null result violated)"
        )
