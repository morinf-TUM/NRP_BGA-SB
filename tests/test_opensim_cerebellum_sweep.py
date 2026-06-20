"""Host dry-run tests for opensim_cerebellum_sweep.py (Task 11b.2).

All tests use a fake_runner — no Docker container is needed. The fake runner
reads jobs.json written by the client and echoes back plausible trajectories
so the host-side perturbation + cerebellar logic is exercised end-to-end.
"""
import json
import math
from pathlib import Path

from nrp_bga_sb.opensim_plant import OpenSimPlantClient, OpenSimPlantConfig

# FK target endpoints used in all fake-runner responses.
_TARGETS = [[0.30, 0.40], [0.10, 0.05]]
_THETA_DEG = 30.0
_THETA_RAD = math.radians(_THETA_DEG)


def _cfg():
    return OpenSimPlantConfig(
        image="x", q0=[0.0, 0.35],
        q_target=[[1.2, 1.0], [0.8, 1.6]],
        kp=[120.0, 90.0], kd=[18.0, 14.0],
    )


def _no_move_traj(job):
    return {"trial_id": job["trial_id"], "times_ms": [0.0, 5.0],
            "positions_xy": [[0.0, 0.0], [0.0, 0.0]],
            "onset_time_ms": None, "selected_channel": -1}


def _move_traj(job, targets=_TARGETS):
    ch = job["selected_channel"]
    ep = list(targets[ch])
    return {"trial_id": job["trial_id"],
            "times_ms": [0.0, 700.0, 1300.0],
            "positions_xy": [[0.0, 0.0], [ep[0] * 0.5, ep[1] * 0.5], ep],
            "onset_time_ms": 700.0, "selected_channel": ch}


def _make_all_no_move_runner(tmp_path):
    """Fake runner: every trial returns a zero-movement trajectory."""
    def runner(argv):
        jobs = json.loads((tmp_path / "jobs.json").read_text())["jobs"]
        out = argv[argv.index("--out") + 1]
        Path(out.replace("/io", str(tmp_path))).write_text(json.dumps({
            "target_endpoints_xy": _TARGETS,
            "trajectories": [_no_move_traj(j) for j in jobs],
        }))
        return 0
    return runner


def _make_move_where_open_runner(sub_dir):
    """Fake runner: moves for open-gate specs, no-movement for closed."""
    def runner(argv):
        jobs = json.loads((sub_dir / "jobs.json").read_text())["jobs"]
        trajs = [_move_traj(j) if j["gate_state"] != "closed" else _no_move_traj(j)
                 for j in jobs]
        out = argv[argv.index("--out") + 1]
        Path(out.replace("/io", str(sub_dir))).write_text(json.dumps({
            "target_endpoints_xy": _TARGETS,
            "trajectories": trajs,
        }))
        return 0
    return runner


# --- Tests ---


def test_all_miss_trials_produce_zero_onset_and_deviation(tmp_path):
    """5 Hz all-miss: onset_rate=0, deviation=0, theta_hat=0 (filter never updated)."""
    from experiments.opensim_cerebellum_sweep import run_opensim_cerebellum_condition

    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path,
                                runner=_make_all_no_move_runner(tmp_path))
    res = run_opensim_cerebellum_condition(
        5.0, n_trials=10, client=client,
        cerebellum_enabled=True, perturbation_deg=_THETA_DEG,
    )
    assert res["opensim_movement_onset_rate"] == 0.0
    assert res["opensim_mean_endpoint_deviation"] == 0.0
    # The cerebellar filter must never update on missed trials (BG-effect guard).
    assert res["final_theta_hat"] == 0.0


def test_perturbation_causes_nonzero_deviation_without_cerebellum(tmp_path):
    """Without correction, a 30° visuomotor rotation deflects the endpoint from target."""
    from experiments.opensim_cerebellum_sweep import run_opensim_cerebellum_condition

    sub = tmp_path / "nocb"
    client = OpenSimPlantClient(_cfg(), io_dir=sub,
                                runner=_make_move_where_open_runner(sub))
    res = run_opensim_cerebellum_condition(
        40.0, n_trials=20, client=client,
        cerebellum_enabled=False, perturbation_deg=_THETA_DEG,
    )
    assert res["opensim_movement_onset_rate"] > 0.0
    assert res["opensim_mean_endpoint_deviation"] > 0.0


def test_onset_rate_identical_cerebellum_on_vs_off(tmp_path):
    """BG-effect guard: cerebellar correction must not alter the movement-onset rate."""
    from experiments.opensim_cerebellum_sweep import run_opensim_cerebellum_condition

    sub_on = tmp_path / "on"
    sub_off = tmp_path / "off"
    client_on = OpenSimPlantClient(_cfg(), io_dir=sub_on,
                                   runner=_make_move_where_open_runner(sub_on))
    client_off = OpenSimPlantClient(_cfg(), io_dir=sub_off,
                                    runner=_make_move_where_open_runner(sub_off))
    res_on = run_opensim_cerebellum_condition(40.0, n_trials=20, client=client_on,
                                              cerebellum_enabled=True)
    res_off = run_opensim_cerebellum_condition(40.0, n_trials=20, client=client_off,
                                               cerebellum_enabled=False)
    assert res_on["opensim_movement_onset_rate"] == res_off["opensim_movement_onset_rate"]


def test_cerebellum_reduces_endpoint_deviation(tmp_path):
    """After 50 trials the LMS filter converges enough to reduce mean deviation."""
    from experiments.opensim_cerebellum_sweep import run_opensim_cerebellum_condition

    sub_on = tmp_path / "on"
    sub_off = tmp_path / "off"
    client_on = OpenSimPlantClient(_cfg(), io_dir=sub_on,
                                   runner=_make_move_where_open_runner(sub_on))
    client_off = OpenSimPlantClient(_cfg(), io_dir=sub_off,
                                    runner=_make_move_where_open_runner(sub_off))
    res_on = run_opensim_cerebellum_condition(40.0, n_trials=50, client=client_on,
                                              cerebellum_enabled=True)
    res_off = run_opensim_cerebellum_condition(40.0, n_trials=50, client=client_off,
                                               cerebellum_enabled=False)
    assert res_on["opensim_mean_endpoint_deviation"] < res_off["opensim_mean_endpoint_deviation"]


def test_result_dict_has_required_keys(tmp_path):
    """run_opensim_cerebellum_condition returns a complete metrics dict."""
    from experiments.opensim_cerebellum_sweep import run_opensim_cerebellum_condition

    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path,
                                runner=_make_all_no_move_runner(tmp_path))
    res = run_opensim_cerebellum_condition(5.0, n_trials=6, client=client)
    required = {
        "frequency_hz", "n_trials", "n_go_trials",
        "opensim_movement_onset_rate", "kinematic_movement_onset_rate",
        "opensim_mean_endpoint_deviation",
        "cerebellum_enabled", "perturbation_deg", "final_theta_hat",
    }
    assert required <= res.keys()
    assert res["n_trials"] == 6
    assert res["frequency_hz"] == 5.0
