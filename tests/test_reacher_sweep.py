"""Tests for ReacherConditionResult and run_reacher_condition (Task 6.3).

Extended in Task 8.2 with change-of-mind trajectory and sweep tests.
"""
import numpy as np
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


# =============================================================================
# Task 8.2: simulate_change_of_mind and run_change_of_mind_reacher_condition
# =============================================================================


def _make_motor_commands(ch0: int, ch1: int, gate_gain: float = 1.0):
    """Build a pair of MotorCommand stubs for two-channel change-of-mind tests."""
    from nrp_bga_sb.schemas import MotorCommand

    def _cmd(ch: int, t_s: float, trial_id: int = 0) -> MotorCommand:
        command = [0.0, 0.0]
        command[ch] = gate_gain
        return MotorCommand(
            sim_time=t_s,
            trial_id=trial_id,
            command=command,
            gate_state="open",
            gate_gain=gate_gain,
        )

    return [_cmd(ch0, t_s=0.30, trial_id=0), _cmd(ch1, t_s=0.55, trial_id=0)]


# --- simulate_change_of_mind tests ---


def test_simulate_com_phase2_ends_near_ch1_target():
    """Phase 2 endpoint should be within 10 % of the post-switch target."""
    from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig

    cfg = ReacherConfig()
    reacher = KinematicReacher(cfg)
    cmds = _make_motor_commands(ch0=0, ch1=1)

    traj = reacher.simulate_change_of_mind(
        motor_commands=cmds,
        pre_switch_onset_ms=320.0,
        switch_time_ms=500.0,
        total_duration_ms=1500.0,
    )

    positions = np.array(traj.positions_xy)
    final_pos = positions[-1]
    target = np.array(cfg.target_positions[1])  # ch1 target
    # After 1000 ms post-switch with movement_duration_ms=300, should be at target
    assert np.linalg.norm(final_pos - target) < 0.1
    assert traj.selected_channel == 1
    assert traj.onset_time_ms == pytest.approx(320.0)


def test_simulate_com_early_switch_near_origin_at_switch():
    """With an early switch (20 ms of phase 1), position at switch_time_ms ≈ origin."""
    from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig

    cfg = ReacherConfig()
    reacher = KinematicReacher(cfg)
    cmds = _make_motor_commands(ch0=0, ch1=1)

    pre_switch_onset_ms = 320.0
    switch_time_ms = 340.0  # only 20 ms of phase 1

    traj = reacher.simulate_change_of_mind(
        motor_commands=cmds,
        pre_switch_onset_ms=pre_switch_onset_ms,
        switch_time_ms=switch_time_ms,
        total_duration_ms=1500.0,
    )

    # Find position at switch_time_ms
    times = traj.times_ms
    positions = traj.positions_xy
    switch_idx = min(range(len(times)), key=lambda i: abs(times[i] - switch_time_ms))
    switch_pos = np.array(positions[switch_idx])

    # 20 ms into a 300 ms movement: minimum-jerk is near zero
    assert np.linalg.norm(switch_pos) < 0.05


def test_simulate_com_raises_wrong_command_count():
    """ValueError when motor_commands list does not have exactly 2 entries."""
    from nrp_bga_sb.reacher import KinematicReacher

    reacher = KinematicReacher()
    cmds = _make_motor_commands(ch0=0, ch1=1)

    with pytest.raises(ValueError, match="2"):
        reacher.simulate_change_of_mind(
            motor_commands=[cmds[0]],  # only 1 command
            pre_switch_onset_ms=320.0,
            switch_time_ms=400.0,
        )

    with pytest.raises(ValueError, match="2"):
        reacher.simulate_change_of_mind(
            motor_commands=[],
            pre_switch_onset_ms=320.0,
            switch_time_ms=400.0,
        )


def test_simulate_com_switch_pos_between_origin_and_ch0_target():
    """Position at switch_time_ms is strictly between origin and ch0 target.

    A non-trivial phase 1 (120 ms, movement_duration_ms=300) should produce
    a position that is farther from origin than 0 but shorter than the full target.
    """
    from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig

    cfg = ReacherConfig()
    reacher = KinematicReacher(cfg)
    cmds = _make_motor_commands(ch0=0, ch1=1)

    pre_switch_onset_ms = 320.0
    switch_time_ms = 440.0  # 120 ms of phase 1 (40 % of movement_duration_ms=300)

    traj = reacher.simulate_change_of_mind(
        motor_commands=cmds,
        pre_switch_onset_ms=pre_switch_onset_ms,
        switch_time_ms=switch_time_ms,
        total_duration_ms=1500.0,
    )

    times = traj.times_ms
    positions = traj.positions_xy
    switch_idx = min(range(len(times)), key=lambda i: abs(times[i] - switch_time_ms))
    switch_pos = np.array(positions[switch_idx])

    target_ch0 = np.array(cfg.target_positions[0])
    dist_to_origin = np.linalg.norm(switch_pos)
    dist_to_target = np.linalg.norm(switch_pos - target_ch0)

    # Should have moved some distance from origin but not yet at target
    assert dist_to_origin > 0.01
    assert dist_to_target < np.linalg.norm(target_ch0)


def test_simulate_com_raises_closed_gate():
    """ValueError when either motor command has gate_state == 'closed'."""
    from nrp_bga_sb.reacher import KinematicReacher
    from nrp_bga_sb.schemas import MotorCommand

    reacher = KinematicReacher()
    closed_cmd = MotorCommand(
        sim_time=0.3,
        trial_id=0,
        command=[0.0, 0.0],
        gate_state="closed",
        gate_gain=0.0,
    )
    open_cmd = MotorCommand(
        sim_time=0.55,
        trial_id=0,
        command=[1.0, 0.0],
        gate_state="open",
        gate_gain=1.0,
    )

    with pytest.raises(ValueError, match="closed"):
        reacher.simulate_change_of_mind(
            motor_commands=[closed_cmd, open_cmd],
            pre_switch_onset_ms=300.0,
            switch_time_ms=400.0,
        )


# --- run_change_of_mind_reacher_condition tests ---


def test_com_reacher_condition_returns_result_structure():
    """At 40 Hz: result has 4 switch categories and n_switch_trials == n_trials."""
    from nrp_bga_sb.reacher_sweep import (
        ChangeOfMindReacherResult,
        run_change_of_mind_reacher_condition,
    )

    result = run_change_of_mind_reacher_condition(
        frequency_hz=40.0,
        n_trials=40,
        seed=42,
    )

    assert isinstance(result, ChangeOfMindReacherResult)
    # no_switch_proportion=0.0 → all trials are switch trials
    assert result.n_switch_trials == result.n_trials
    assert len(result.switch_success_by_category) == 4
    assert set(result.switch_success_by_category.keys()) == {
        "early", "medium", "late", "very_late"
    }


def test_com_reacher_condition_correction_cost_positive():
    """At 40 Hz: mean_correction_cost > 0 (arm moved in wrong direction before reversing)."""
    from nrp_bga_sb.reacher_sweep import run_change_of_mind_reacher_condition

    result = run_change_of_mind_reacher_condition(
        frequency_hz=40.0,
        n_trials=40,
        seed=42,
    )

    # correction_cost = total_path - straight_dist; must be > 0 for any reversal trajectory
    if result.mean_correction_cost is not None:
        assert result.mean_correction_cost > 0.0


def test_correction_cost_increases_with_switch_delay():
    """Later switch → arm travels farther in wrong direction → higher correction cost.

    Tests simulate_change_of_mind directly so we can control the exact timing.
    go_cue_onset_ms=300, initial_decision_point_ms=20 → pre_switch_onset_ms=320.
    Early switch: switch_time_ms=350 (30 ms of phase 1).
    Very late switch: switch_time_ms=750 (430 ms of phase 1 >> movement_duration_ms).
    """
    from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig

    cfg = ReacherConfig()
    reacher = KinematicReacher(cfg)

    cmds = _make_motor_commands(ch0=0, ch1=1, gate_gain=1.0)

    def _correction_cost(switch_time_ms: float) -> float:
        traj = reacher.simulate_change_of_mind(
            motor_commands=cmds,
            pre_switch_onset_ms=320.0,
            switch_time_ms=switch_time_ms,
            total_duration_ms=1500.0,
        )
        positions = np.array(traj.positions_xy)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_path = float(steps.sum())
        straight_dist = float(np.linalg.norm(positions[-1]))
        return total_path - straight_dist

    early_cost = _correction_cost(350.0)
    very_late_cost = _correction_cost(750.0)

    # Later switch = more phase-1 travel in wrong direction = higher correction cost
    assert very_late_cost > early_cost
