# tests/test_reacher_correction.py
import math

import numpy as np
import pytest

from nrp_bga_sb.cerebellum import Cerebellum
from nrp_bga_sb.perturbation_plant import VisuomotorRotation
from nrp_bga_sb.reacher import KinematicReacher
from nrp_bga_sb.schemas import MotorCommand


def _open_cmd(channel: int, gain: float = 1.0, n: int = 2) -> MotorCommand:
    command = [0.0] * n
    command[channel] = gain
    return MotorCommand(
        command=command, gate_state="open", gate_gain=gain, sim_time=0.7, trial_id=0
    )


def _closed_cmd(n: int = 2) -> MotorCommand:
    return MotorCommand(
        command=[0.0] * n, gate_state="closed", gate_gain=0.0, sim_time=0.7, trial_id=0
    )


def test_correction_no_movement_leaves_cerebellum_untouched():
    reacher = KinematicReacher()
    cb = Cerebellum()
    cb.adaptive_filter.theta_hat = 0.5
    traj = reacher.simulate_with_correction(
        [_closed_cmd()], onset_time_ms=None,
        perturbation=VisuomotorRotation(rotation_deg=30.0), cerebellum=cb,
    )
    # guard: closed gate -> no movement, cerebellum never learns
    assert traj.selected_channel == -1
    assert traj.onset_time_ms is None
    assert cb.adaptive_filter.theta_hat == 0.5


def test_correction_perturbation_only_rotates_endpoint():
    reacher = KinematicReacher()
    traj = reacher.simulate_with_correction(
        [_open_cmd(1, 1.0)], onset_time_ms=0.0, total_duration_ms=500.0,
        perturbation=VisuomotorRotation(rotation_deg=30.0), cerebellum=None,
    )
    final = traj.positions_xy[-1]
    # target for channel 1 is [1, 0]; rotated by 30 deg
    assert final[0] == pytest.approx(math.cos(math.radians(30.0)), abs=1e-3)
    assert final[1] == pytest.approx(math.sin(math.radians(30.0)), abs=1e-3)


def test_correction_online_reduces_endpoint_error():
    reacher = KinematicReacher()
    pert = VisuomotorRotation(rotation_deg=30.0)
    target = np.array([1.0, 0.0])

    uncorrected = reacher.simulate_with_correction(
        [_open_cmd(1, 1.0)], 0.0, 500.0, perturbation=pert, cerebellum=None
    )
    err_unc = float(np.linalg.norm(np.array(uncorrected.positions_xy[-1]) - target))

    cb = Cerebellum(adaptation_enabled=False, online_enabled=True, online_gain=0.6)
    corrected = reacher.simulate_with_correction(
        [_open_cmd(1, 1.0)], 0.0, 500.0, perturbation=pert, cerebellum=cb
    )
    err_cor = float(np.linalg.norm(np.array(corrected.positions_xy[-1]) - target))
    assert err_cor < err_unc


def test_correction_no_perturbation_no_cerebellum_matches_simulate():
    reacher = KinematicReacher()
    cmds = [_open_cmd(0, 0.8)]
    a = reacher.simulate(cmds, 0.0, 500.0)
    b = reacher.simulate_with_correction(cmds, 0.0, 500.0, perturbation=None, cerebellum=None)
    assert b.positions_xy[-1] == pytest.approx(a.positions_xy[-1], abs=1e-6)
    assert b.selected_channel == a.selected_channel


def test_correction_adaptation_learns_toward_perturbation():
    reacher = KinematicReacher()
    pert = VisuomotorRotation(rotation_deg=30.0)
    cb = Cerebellum(adaptation_enabled=True, online_enabled=False, learning_rate=0.3)
    for _ in range(50):
        reacher.simulate_with_correction(
            [_open_cmd(1, 1.0)], 0.0, 500.0, perturbation=pert, cerebellum=cb
        )
    assert cb.adaptive_filter.theta_hat == pytest.approx(math.radians(30.0), abs=1e-2)
