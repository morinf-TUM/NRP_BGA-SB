"""Tests for ReacherConditionResult and run_reacher_condition (Task 6.3)."""
import pytest


def test_reacher_condition_result_has_required_fields():
    from nrp_bga_sb.reacher_sweep import ReacherConditionResult
    # Verify all expected fields exist with correct defaults/types
    r = ReacherConditionResult(
        frequency_hz=160.0,
        conflict_level="low",
        paradigm="go_nogo",
        seed=0,
        n_trials=10,
        miss_rate=0.0,
        go_success_rate=1.0,
        timeout_rate=None,
        bg_commitment_latency_mean=0.1,
        movement_onset_rate=1.0,
        mean_endpoint_error=0.0,
        mean_partial_amplitude=1.0,
        mean_peak_velocity=0.005,
    )
    assert r.frequency_hz == 160.0
    assert r.movement_onset_rate == 1.0


def test_high_freq_low_conflict_low_miss_rate():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=160.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=20,
        seed=42,
    )
    assert result.miss_rate is not None
    assert result.miss_rate == pytest.approx(0.0, abs=0.15)


def test_low_freq_high_miss_rate():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=5.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=20,
        seed=42,
    )
    assert result.miss_rate is not None
    assert result.miss_rate == pytest.approx(1.0, abs=0.15)


def test_movement_onset_rate_matches_go_success_rate():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=40.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=20,
        seed=7,
    )
    if result.go_success_rate is not None:
        assert result.movement_onset_rate == pytest.approx(
            result.go_success_rate, abs=0.05
        )


def test_movement_metrics_nonnegative():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=40.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=10,
        seed=0,
    )
    assert result.mean_endpoint_error >= 0.0
    assert result.mean_partial_amplitude >= 0.0
    assert result.mean_peak_velocity >= 0.0
    assert 0.0 <= result.movement_onset_rate <= 1.0


def test_high_freq_low_conflict_endpoint_error():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=160.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=10,
        seed=0,
    )
    # At 160 Hz low conflict: BG selects on all go trials with a partial gate
    # (margin ≈ 0.2, full_open_threshold=0.3 → gain ≈ 0.6).
    # All go trials reach the same target with the same partial gain → endpoint_error
    # is non-zero and consistent.  movement_onset_rate must be 1.0 (all go trials move).
    assert result.movement_onset_rate == pytest.approx(1.0, abs=0.05)
    assert result.mean_endpoint_error > 0.0
    assert result.mean_endpoint_error < 1.5  # bounded by target distance (1.0) × 2


def test_two_choice_paradigm_returns_timeout_rate():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=160.0,
        conflict_level="low",
        paradigm="two_choice",
        n_trials=10,
        seed=0,
    )
    assert result.timeout_rate is not None
    assert result.miss_rate is None
