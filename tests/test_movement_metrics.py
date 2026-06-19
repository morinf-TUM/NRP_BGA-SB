"""Tests for MovementMetrics and compute_movement_metrics (Task 6.2)."""
import pytest

from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.schemas import MotorCommand


def _make_motor_command(gate_state, gate_gain, channel=0, n_channels=2):
    command = [0.0] * n_channels
    if gate_state != "closed":
        command[channel] = gate_gain
    return MotorCommand(
        sim_time=0.0, trial_id=1,
        command=command, gate_state=gate_state, gate_gain=gate_gain,
    )


def _traj(gate_state, gate_gain, channel=0, onset_ms=0.0):
    """Build a ReacherTrajectory via KinematicReacher."""
    cmd = _make_motor_command(gate_state, gate_gain, channel)
    return KinematicReacher().simulate([cmd], onset_time_ms=onset_ms, total_duration_ms=500.0)


# --- no movement ---

def test_no_movement_closed_gate():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("closed", 0.0)
    m = compute_movement_metrics(traj, ReacherConfig())
    assert m.movement_onset_time_ms is None
    assert m.endpoint_error == pytest.approx(0.0)
    assert m.partial_movement_amplitude == pytest.approx(0.0)
    assert m.trajectory_curvature == pytest.approx(0.0)
    assert m.movement_reversal_time_ms is None
    assert m.peak_velocity == pytest.approx(0.0, abs=1e-9)


# --- full movement ---

def test_full_movement_zero_endpoint_error():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    # Full gate_gain=1.0 → reaches target (-1,0) exactly
    assert m.endpoint_error == pytest.approx(0.0, abs=1e-6)


def test_full_movement_amplitude_equals_target_distance():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=1)
    m = compute_movement_metrics(traj, ReacherConfig())
    # target (1, 0): distance from origin = 1.0
    assert m.partial_movement_amplitude == pytest.approx(1.0, abs=1e-6)


# --- partial movement ---

def test_partial_movement_endpoint_error():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("partial", 0.5, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    # endpoint = (-0.5, 0), target = (-1, 0), error = 0.5
    assert m.endpoint_error == pytest.approx(0.5, abs=1e-6)


def test_partial_movement_amplitude():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("partial", 0.5, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    assert m.partial_movement_amplitude == pytest.approx(0.5, abs=1e-6)


# --- straight-line curvature ---

def test_straight_line_curvature_is_zero():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    # Min-jerk toward (-1,0): all positions are along the x-axis → curvature = 0
    assert m.trajectory_curvature == pytest.approx(0.0, abs=1e-9)


# --- velocity ---

def test_peak_velocity_positive_for_full_movement():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=1)
    m = compute_movement_metrics(traj, ReacherConfig())
    assert m.peak_velocity > 0.0


def test_peak_velocity_min_jerk_formula():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    # v_peak = 1.875 * amplitude / T for min-jerk
    # amplitude = 1.0 (gate_gain=1, target at ±1), T = 300 ms
    traj = _traj("open", 1.0, channel=1)
    m = compute_movement_metrics(traj, ReacherConfig())
    expected_peak = 1.875 * 1.0 / 300.0
    assert m.peak_velocity == pytest.approx(expected_peak, rel=0.05)


# --- reversal ---

def test_no_reversal_for_single_command():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    # Monotone min-jerk → no reversal
    assert m.movement_reversal_time_ms is None
