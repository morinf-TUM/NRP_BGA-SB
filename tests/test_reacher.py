"""Tests for KinematicReacher and ReacherTrajectory (Task 6.1)."""
import pytest

from nrp_bga_sb.schemas import MotorCommand


def _make_motor_command(gate_state, gate_gain, channel=0, n_channels=2):
    """Helper: build a MotorCommand consistent with ThalamusGate output."""
    command = [0.0] * n_channels
    if gate_state != "closed":
        command[channel] = gate_gain
    return MotorCommand(
        sim_time=0.0,
        trial_id=1,
        command=command,
        gate_state=gate_state,
        gate_gain=gate_gain,
    )


# --- ReacherConfig ---

def test_reacher_config_defaults():
    from nrp_bga_sb.reacher import ReacherConfig
    cfg = ReacherConfig()
    assert cfg.n_channels == 2
    assert cfg.target_positions == [[-1.0, 0.0], [1.0, 0.0]]
    assert cfg.movement_duration_ms == 300.0
    assert cfg.dt_ms == 1.0


def test_reacher_config_rejects_position_count_mismatch():
    from nrp_bga_sb.reacher import ReacherConfig
    with pytest.raises(Exception):
        ReacherConfig(n_channels=2, target_positions=[[0.0, 1.0]])


# --- _minimum_jerk_scalar ---

def test_minimum_jerk_zero_at_start():
    from nrp_bga_sb.reacher import _minimum_jerk_scalar
    assert _minimum_jerk_scalar(0.0, 300.0) == pytest.approx(0.0)


def test_minimum_jerk_half_at_midpoint():
    from nrp_bga_sb.reacher import _minimum_jerk_scalar
    assert _minimum_jerk_scalar(150.0, 300.0) == pytest.approx(0.5)


def test_minimum_jerk_one_at_end():
    from nrp_bga_sb.reacher import _minimum_jerk_scalar
    assert _minimum_jerk_scalar(300.0, 300.0) == pytest.approx(1.0)


def test_minimum_jerk_saturates_past_end():
    from nrp_bga_sb.reacher import _minimum_jerk_scalar
    assert _minimum_jerk_scalar(600.0, 300.0) == pytest.approx(1.0)


# --- KinematicReacher.simulate ---

def test_simulate_zero_trajectory_empty_commands():
    from nrp_bga_sb.reacher import KinematicReacher
    r = KinematicReacher()
    traj = r.simulate([], onset_time_ms=0.0)
    assert traj.selected_channel == -1
    assert traj.onset_time_ms is None
    assert all(p == [0.0, 0.0] for p in traj.positions_xy)


def test_simulate_zero_trajectory_closed_gate():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("closed", 0.0)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=0.0)
    assert traj.selected_channel == -1
    assert traj.onset_time_ms is None
    assert all(p == [0.0, 0.0] for p in traj.positions_xy)


def test_simulate_full_movement_ch0_reaches_target():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("open", 1.0, channel=0)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=0.0, total_duration_ms=500.0)
    assert traj.selected_channel == 0
    assert traj.onset_time_ms == pytest.approx(0.0)
    final = traj.positions_xy[-1]
    assert final[0] == pytest.approx(-1.0, abs=1e-6)
    assert final[1] == pytest.approx(0.0, abs=1e-6)


def test_simulate_full_movement_ch1_reaches_target():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("open", 1.0, channel=1)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=0.0, total_duration_ms=500.0)
    assert traj.selected_channel == 1
    final = traj.positions_xy[-1]
    assert final[0] == pytest.approx(1.0, abs=1e-6)


def test_simulate_partial_movement_gate_gain_half():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("partial", 0.5, channel=0)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=0.0, total_duration_ms=500.0)
    final = traj.positions_xy[-1]
    # Effective endpoint = 0.5 × target(-1,0) = (-0.5, 0)
    assert final[0] == pytest.approx(-0.5, abs=1e-6)


def test_simulate_onset_respected_positions_before_onset_are_zero():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("open", 1.0, channel=1)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=100.0, total_duration_ms=500.0)
    # All positions before t=100ms should be (0,0)
    for t, p in zip(traj.times_ms, traj.positions_xy):
        if t < 100.0:
            assert p == [0.0, 0.0], f"position at t={t} should be zero before onset"


def test_simulate_trajectory_length():
    from nrp_bga_sb.reacher import KinematicReacher
    r = KinematicReacher()
    traj = r.simulate([], onset_time_ms=None, total_duration_ms=200.0)
    # n_steps = int(round(200.0 / 1.0)) + 1 = 201
    assert len(traj.times_ms) == 201
    assert len(traj.positions_xy) == 201


def test_simulate_raises_on_channel_out_of_range():
    from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
    cmd = _make_motor_command("open", 1.0, channel=0, n_channels=5)
    # Config only has 2 targets
    r = KinematicReacher(ReacherConfig(n_channels=2))
    with pytest.raises(ValueError, match="channel"):
        r.simulate([cmd], onset_time_ms=0.0)


def test_simulate_none_onset_defaults_to_zero():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("open", 1.0, channel=1)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=None, total_duration_ms=500.0)
    # onset_time_ms=None → default 0.0, movement starts immediately
    assert traj.onset_time_ms == pytest.approx(0.0)
    # First position is at t=0, which equals onset, so s=_minimum_jerk(0,300)=0 → still (0,0)
    assert traj.positions_xy[0] == [pytest.approx(0.0), pytest.approx(0.0)]
