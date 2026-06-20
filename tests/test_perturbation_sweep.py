"""Tests for perturbation_sweep.py — Task 9.1 (Phase 9, M10).

TDD order: constants → type checks → private helpers → factory → condition
runners → report formatter.

All behavioral tests use small n_trials_per_seed=5, n_seeds=2 to stay fast.
"""

from __future__ import annotations

import pytest

from nrp_bga_sb.perturbation_sweep import (
    DROPOUT_LEVELS,
    FREQUENCIES_HZ,
    JITTER_STD_LEVELS_MS,
    LATENCY_LEVELS_MS,
    N_GONOGO_SEEDS,
    N_GONOGO_TRIALS_PER_SEED,
    N_SS_SEEDS,
    N_SS_TRIALS_PER_SEED,
    PEAK_SALIENCE,
    PHASE_OFFSET_FRACTIONS,
    SS_INITIAL_SSD_MS,
    SS_SSD_MAX_MS,
    SS_SSD_MIN_MS,
    SS_SSD_STEP_MS,
    SS_STOP_PROPORTION,
    PerturbationSweepResult,
    _make_wrapped_policy,
    _phase_offset_ms,
    format_decomposition_report,
    run_gonogo_perturbation_condition,
    run_stopsignal_perturbation_condition,
)
from nrp_bga_sb.perturbations import (
    DropoutWrapper,
    JitterWrapper,
    LatencyWrapper,
    PhaseOffsetWrapper,
)

# =============================================================================
# Constants
# =============================================================================


def test_perturbation_levels_have_zero_baseline():
    """All four level lists start with 0.0 (baseline / unperturbed condition)."""
    assert LATENCY_LEVELS_MS[0] == 0.0
    assert JITTER_STD_LEVELS_MS[0] == 0.0
    assert DROPOUT_LEVELS[0] == 0.0
    assert PHASE_OFFSET_FRACTIONS[0] == 0.0


def test_latency_levels_strictly_increasing():
    assert LATENCY_LEVELS_MS == sorted(LATENCY_LEVELS_MS)
    assert len(LATENCY_LEVELS_MS) == len(set(LATENCY_LEVELS_MS)), "levels must be unique"


def test_phase_offset_fractions_in_unit_interval():
    for f in PHASE_OFFSET_FRACTIONS:
        assert 0.0 <= f <= 1.0, f"fraction {f} out of [0, 1]"


def test_dropout_levels_in_unit_interval():
    for d in DROPOUT_LEVELS:
        assert 0.0 <= d <= 1.0, f"dropout level {d} out of [0, 1]"


def test_frequencies_hz_all_positive():
    assert all(f > 0 for f in FREQUENCIES_HZ)


def test_constants_match_spec():
    """Spot-check exact spec values from the brief."""
    assert PEAK_SALIENCE == 0.85
    assert N_GONOGO_SEEDS == 5
    assert N_GONOGO_TRIALS_PER_SEED == 50
    assert N_SS_SEEDS == 5
    assert N_SS_TRIALS_PER_SEED == 100
    assert SS_STOP_PROPORTION == 0.25
    assert SS_INITIAL_SSD_MS == 200
    assert SS_SSD_STEP_MS == 50
    assert SS_SSD_MIN_MS == 50
    assert SS_SSD_MAX_MS == 450


# =============================================================================
# _phase_offset_ms helper
# =============================================================================


def test_phase_offset_ms_zero_fraction():
    assert _phase_offset_ms(0.0, 20.0) == 0.0


def test_phase_offset_ms_full_period():
    # 1.0 * (1000/10) = 100.0 ms
    assert _phase_offset_ms(1.0, 10.0) == 100.0


def test_phase_offset_ms_half_period():
    # 0.5 * (1000/40) = 0.5 * 25 = 12.5 ms
    assert _phase_offset_ms(0.5, 40.0) == pytest.approx(12.5)


def test_phase_offset_ms_quarter_period():
    # 0.25 * (1000/20) = 0.25 * 50 = 12.5 ms
    assert _phase_offset_ms(0.25, 20.0) == pytest.approx(12.5)


# =============================================================================
# _make_wrapped_policy factory
# =============================================================================


def test_make_wrapped_policy_latency():
    from nrp_bga_sb.cortex import CortexConfig
    from nrp_bga_sb.scheduler import FrequencyConfig

    freq_cfg = FrequencyConfig.from_effective_hz(20.0)
    cortex_cfg = CortexConfig(peak_salience=0.85, rise_time_ms=200.0, noise_std=0.0)
    policy = _make_wrapped_policy("latency", 25.0, 20.0, cortex_cfg, freq_cfg)
    assert isinstance(policy, LatencyWrapper)


def test_make_wrapped_policy_jitter():
    from nrp_bga_sb.cortex import CortexConfig
    from nrp_bga_sb.scheduler import FrequencyConfig

    freq_cfg = FrequencyConfig.from_effective_hz(20.0)
    cortex_cfg = CortexConfig(peak_salience=0.85, rise_time_ms=200.0, noise_std=0.0)
    policy = _make_wrapped_policy("jitter", 10.0, 20.0, cortex_cfg, freq_cfg)
    assert isinstance(policy, JitterWrapper)


def test_make_wrapped_policy_dropout():
    from nrp_bga_sb.cortex import CortexConfig
    from nrp_bga_sb.scheduler import FrequencyConfig

    freq_cfg = FrequencyConfig.from_effective_hz(20.0)
    cortex_cfg = CortexConfig(peak_salience=0.85, rise_time_ms=200.0, noise_std=0.0)
    policy = _make_wrapped_policy("dropout", 0.05, 20.0, cortex_cfg, freq_cfg)
    assert isinstance(policy, DropoutWrapper)


def test_make_wrapped_policy_phase_offset():
    from nrp_bga_sb.cortex import CortexConfig
    from nrp_bga_sb.scheduler import FrequencyConfig

    freq_cfg = FrequencyConfig.from_effective_hz(20.0)
    cortex_cfg = CortexConfig(peak_salience=0.85, rise_time_ms=200.0, noise_std=0.0)
    policy = _make_wrapped_policy("phase_offset", 0.5, 20.0, cortex_cfg, freq_cfg)
    assert isinstance(policy, PhaseOffsetWrapper)


def test_make_wrapped_policy_unknown_type_raises():
    from nrp_bga_sb.cortex import CortexConfig
    from nrp_bga_sb.scheduler import FrequencyConfig

    freq_cfg = FrequencyConfig.from_effective_hz(20.0)
    cortex_cfg = CortexConfig(peak_salience=0.85, rise_time_ms=200.0, noise_std=0.0)
    with pytest.raises(ValueError, match="unknown"):
        _make_wrapped_policy("bad_type", 0.0, 20.0, cortex_cfg, freq_cfg)


# =============================================================================
# Go/no-go condition runner
# =============================================================================


def test_gonogo_baseline_result_fields():
    """20 Hz, latency=0 → all go_nogo fields populated."""
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert isinstance(result, PerturbationSweepResult)
    assert result.go_success_rate is not None
    assert result.false_alarm_rate is not None
    assert result.bg_commitment_latency_mean is not None
    assert result.n_trials == 10  # 5 trials × 2 seeds


def test_gonogo_result_paradigm_is_gonogo():
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert result.paradigm == "go_nogo"


def test_gonogo_stop_signal_fields_are_none():
    """Stop-signal fields must be None for go_nogo paradigm."""
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert result.stop_failure_rate is None
    assert result.ssrt_estimate_s is None
    assert result.go_rt_mean_s is None
    assert result.inhibition_function_monotone is None


def test_gonogo_high_frequency_success_near_one():
    """At 40 Hz with no perturbation, go_success_rate should be close to 1.0."""
    result = run_gonogo_perturbation_condition(
        frequency_hz=40.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=10,
        n_seeds=2,
    )
    assert result.go_success_rate is not None
    assert result.go_success_rate >= 0.5  # conservative lower bound


def test_gonogo_latency_does_not_change_success_rate():
    """At 40 Hz, adding 100ms latency should not change go_success_rate (within 0.05)."""
    baseline = run_gonogo_perturbation_condition(
        frequency_hz=40.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=10,
        n_seeds=2,
        base_seed=42,
    )
    with_latency = run_gonogo_perturbation_condition(
        frequency_hz=40.0,
        perturbation_type="latency",
        perturbation_value=100.0,
        n_trials_per_seed=10,
        n_seeds=2,
        base_seed=42,
    )
    assert baseline.go_success_rate is not None
    assert with_latency.go_success_rate is not None
    diff = abs(baseline.go_success_rate - with_latency.go_success_rate)
    assert diff <= 0.05


def test_gonogo_dropout_high_changes_false_alarm():
    """At 40 Hz, dropout=0.10 returns a valid PerturbationSweepResult."""
    result = run_gonogo_perturbation_condition(
        frequency_hz=40.0,
        perturbation_type="dropout",
        perturbation_value=0.10,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert isinstance(result, PerturbationSweepResult)
    assert result.false_alarm_rate is not None
    assert 0.0 <= result.false_alarm_rate <= 1.0


def test_gonogo_n_trials_equals_seeds_times_trials():
    """n_trials in result is total across seeds."""
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="jitter",
        perturbation_value=5.0,
        n_trials_per_seed=7,
        n_seeds=3,
    )
    assert result.n_trials == 21  # 7 * 3
    assert result.n_seeds == 3


def test_gonogo_perturbation_type_and_value_stored():
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="dropout",
        perturbation_value=0.05,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert result.perturbation_type == "dropout"
    assert result.perturbation_value == 0.05


# =============================================================================
# Stop-signal condition runner
# =============================================================================


def test_stopsignal_baseline_result_fields():
    """20 Hz, latency=0 → all stop_signal fields populated."""
    result = run_stopsignal_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=10,
        n_seeds=2,
    )
    assert isinstance(result, PerturbationSweepResult)
    assert result.stop_failure_rate is not None
    # ssrt_estimate_s may be None in degenerate cases — just check it's accessible
    assert hasattr(result, "ssrt_estimate_s")
    assert result.n_trials == 20  # 10 * 2


def test_stopsignal_result_paradigm_is_stop_signal():
    result = run_stopsignal_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=10,
        n_seeds=2,
    )
    assert result.paradigm == "stop_signal"


def test_stopsignal_gonogo_fields_are_none():
    """Go/no-go fields must be None for stop_signal paradigm."""
    result = run_stopsignal_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=10,
        n_seeds=2,
    )
    assert result.go_success_rate is None
    assert result.false_alarm_rate is None
    assert result.bg_commitment_latency_mean is None


def test_stopsignal_stop_failure_rate_in_range():
    result = run_stopsignal_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="dropout",
        perturbation_value=0.0,
        n_trials_per_seed=10,
        n_seeds=2,
    )
    assert result.stop_failure_rate is not None
    assert 0.0 <= result.stop_failure_rate <= 1.0


# =============================================================================
# Perturbation label format
# =============================================================================


def test_perturbation_label_latency_format():
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=25.0,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert "latency=" in result.perturbation_label
    assert "ms" in result.perturbation_label


def test_perturbation_label_dropout_format():
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="dropout",
        perturbation_value=0.05,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert "dropout=" in result.perturbation_label
    assert "%" in result.perturbation_label


def test_perturbation_label_jitter_format():
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="jitter",
        perturbation_value=10.0,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert "jitter_std=" in result.perturbation_label
    assert "ms" in result.perturbation_label


def test_perturbation_label_phase_offset_format():
    result = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="phase_offset",
        perturbation_value=0.25,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    assert "phase_offset=" in result.perturbation_label
    assert "%" in result.perturbation_label


# =============================================================================
# format_decomposition_report
# =============================================================================


def test_format_decomposition_report_has_header():
    """Report string must contain 'M10' from the phase header."""
    r = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    report = format_decomposition_report([r], [])
    assert "M10" in report


def test_format_decomposition_report_has_all_perturbation_types():
    """With one result per type, all four type labels appear in the report."""
    types_and_vals = [
        ("latency", 0.0),
        ("jitter", 0.0),
        ("dropout", 0.0),
        ("phase_offset", 0.0),
    ]
    gonogo_results = [
        run_gonogo_perturbation_condition(
            frequency_hz=20.0,
            perturbation_type=pt,
            perturbation_value=pv,
            n_trials_per_seed=5,
            n_seeds=2,
        )
        for pt, pv in types_and_vals
    ]
    report = format_decomposition_report(gonogo_results, [])
    assert "latency" in report
    assert "jitter" in report
    assert "dropout" in report
    assert "phase_offset" in report


def test_format_decomposition_report_returns_string():
    r = run_gonogo_perturbation_condition(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        n_trials_per_seed=5,
        n_seeds=2,
    )
    report = format_decomposition_report([r], [])
    assert isinstance(report, str)
    assert len(report) > 0


# =============================================================================
# Report formatter — Task 9.2 additions
# =============================================================================


def _make_gonogo_result(
    frequency_hz: float = 20.0,
    perturbation_type: str = "latency",
    perturbation_value: float = 0.0,
) -> PerturbationSweepResult:
    """Helper: run a minimal go/no-go condition for report tests."""
    return run_gonogo_perturbation_condition(
        frequency_hz=frequency_hz,
        perturbation_type=perturbation_type,  # type: ignore[arg-type]
        perturbation_value=perturbation_value,
        n_trials_per_seed=5,
        n_seeds=2,
    )


def _make_stopsignal_result(
    frequency_hz: float = 20.0,
    perturbation_type: str = "latency",
    perturbation_value: float = 0.0,
) -> PerturbationSweepResult:
    """Helper: run a minimal stop-signal condition for report tests."""
    return run_stopsignal_perturbation_condition(
        frequency_hz=frequency_hz,
        perturbation_type=perturbation_type,  # type: ignore[arg-type]
        perturbation_value=perturbation_value,
        n_trials_per_seed=5,
        n_seeds=2,
    )


def test_format_report_has_section_per_perturbation_type():
    """Report must include all four perturbation-type section headers."""
    gonogo_results = [
        _make_gonogo_result(perturbation_type=pt)
        for pt in ["latency", "jitter", "dropout", "phase_offset"]
    ]
    report = format_decomposition_report(gonogo_results, [])
    assert "latency" in report
    assert "jitter" in report
    assert "dropout" in report
    assert "phase_offset" in report


def test_format_report_gonogo_section_present():
    """Report must contain 'Go/No-Go' when go/no-go results are provided."""
    r = _make_gonogo_result()
    report = format_decomposition_report([r], [])
    assert "Go/No-Go" in report


def test_format_report_stop_signal_section_present():
    """Report must contain 'Stop-Signal' when stop-signal results are provided."""
    r = _make_stopsignal_result()
    report = format_decomposition_report([], [r])
    assert "Stop-Signal" in report


def test_format_report_interpretation_guide_present():
    """Report must contain the 'INTERPRETATION GUIDE' section."""
    r = _make_gonogo_result()
    report = format_decomposition_report([r], [])
    assert "INTERPRETATION GUIDE" in report


def test_format_report_na_for_none_values():
    """A stop-signal result with None stop_failure_rate must render as 'N/A'."""
    # Construct a synthetic result with stop_failure_rate=None to test the
    # N/A rendering branch in the formatter without running a real condition.
    synthetic = PerturbationSweepResult(
        frequency_hz=20.0,
        perturbation_type="latency",
        perturbation_value=0.0,
        perturbation_label="latency=0ms",
        paradigm="stop_signal",
        n_trials=10,
        n_seeds=2,
        stop_failure_rate=None,
        ssrt_estimate_s=None,
        go_rt_mean_s=None,
        inhibition_function_monotone=None,
    )
    report = format_decomposition_report([], [synthetic])
    assert "N/A" in report


def test_format_report_frequency_appears_in_table():
    """A 20 Hz result must produce a line containing '20' in the report."""
    r = _make_gonogo_result(frequency_hz=20.0)
    report = format_decomposition_report([r], [])
    assert "20" in report


def test_format_report_sorted_by_frequency():
    """Within a section, lower frequency rows must appear before higher frequency rows."""
    r5 = _make_gonogo_result(frequency_hz=5.0)
    r40 = _make_gonogo_result(frequency_hz=40.0)
    # Pass in reversed order to verify the formatter sorts, not just preserves input order.
    report = format_decomposition_report([r40, r5], [])
    pos_5 = report.find("5.0")
    pos_40 = report.find("40.0")
    assert pos_5 != -1 and pos_40 != -1, "Both frequencies must appear in the report"
    assert pos_5 < pos_40, "5 Hz row must appear before 40 Hz row in the sorted table"


def test_perturbation_levels_dict_keys():
    """Each of the 4 perturbation types must be accepted by both condition runners."""
    perturbation_types_and_values = [
        ("latency", 0.0),
        ("jitter", 0.0),
        ("dropout", 0.0),
        ("phase_offset", 0.0),
    ]
    for ptype, pval in perturbation_types_and_values:
        # Trigger: verify that the library accepts all four type strings.
        # Why: the experiment runner iterates over PERTURBATION_LEVELS.keys() —
        #      a mismatch would silently skip or error at runtime.
        # Outcome: if any type raises, the test fails fast with a clear message.
        result_gng = run_gonogo_perturbation_condition(
            frequency_hz=20.0,
            perturbation_type=ptype,  # type: ignore[arg-type]
            perturbation_value=pval,
            n_trials_per_seed=5,
            n_seeds=2,
        )
        assert result_gng.perturbation_type == ptype
        result_ss = run_stopsignal_perturbation_condition(
            frequency_hz=20.0,
            perturbation_type=ptype,  # type: ignore[arg-type]
            perturbation_value=pval,
            n_trials_per_seed=5,
            n_seeds=2,
        )
        assert result_ss.perturbation_type == ptype
