import json
from pathlib import Path

import pytest

from nrp_bga_sb.opensim_plant import (
    OpenSimPlantClient,
    OpenSimPlantConfig,
    ReachSpec,
    extract_reach_spec,
)
from nrp_bga_sb.schemas import MotorCommand


def _cmd(command, gate_state, gate_gain):
    return MotorCommand(sim_time=0.7, trial_id=0, command=command,
                        gate_state=gate_state, gate_gain=gate_gain)


def test_extract_reach_spec_open_gate_selects_argmax():
    cmds = [_cmd([0.0, 0.8], "open", 0.8)]
    spec = extract_reach_spec(cmds, onset_time_ms=700.0, trial_id="t1")
    assert spec.selected_channel == 1
    assert spec.gate_gain == 0.8
    assert spec.gate_state == "open"
    assert spec.onset_time_ms == 700.0
    assert spec.trial_id == "t1"


def test_extract_reach_spec_empty_is_no_movement():
    spec = extract_reach_spec([], onset_time_ms=None, trial_id="t2")
    assert spec.selected_channel == -1
    assert spec.onset_time_ms is None
    assert spec.gate_state == "closed"


def test_extract_reach_spec_closed_gate_is_no_movement():
    cmds = [_cmd([0.0, 0.0], "closed", 0.0)]
    spec = extract_reach_spec(cmds, onset_time_ms=None, trial_id="t3")
    assert spec.selected_channel == -1
    assert spec.gate_state == "closed"


def test_extract_reach_spec_uses_last_command():
    cmds = [_cmd([0.5, 0.0], "open", 0.5), _cmd([0.0, 0.9], "open", 0.9)]
    spec = extract_reach_spec(cmds, onset_time_ms=700.0, trial_id="t4")
    assert spec.selected_channel == 1


def test_extract_reach_spec_open_but_all_zero_raises():
    cmds = [_cmd([0.0, 0.0], "open", 0.0)]
    with pytest.raises(ValueError, match="all-zero"):
        extract_reach_spec(cmds, onset_time_ms=700.0, trial_id="t5")


FIXTURE = Path(__file__).parent / "opensim" / "fixtures" / "trajectories_ok.json"


def _cfg():
    return OpenSimPlantConfig(image="nrp-bga-opensim:4.6", q0=[0.0, 0.35],
                              q_target=[[0.6, 1.4], [0.0, 0.2]],
                              kp=[60.0, 40.0], kd=[12.0, 8.0])


def test_client_runs_and_parses_results(tmp_path):
    specs = [
        ReachSpec(trial_id="t_go", selected_channel=0, onset_time_ms=700.0,
                  gate_gain=1.0, gate_state="open"),
        ReachSpec(trial_id="t_miss", selected_channel=-1, onset_time_ms=None,
                  gate_gain=0.0, gate_state="closed"),
    ]

    def fake_runner(argv):
        # emulate the container: write the fixture to the requested --out path
        out = argv[argv.index("--out") + 1]
        # argv paths are container paths; map back to the host io_dir
        Path(out.replace("/io", str(tmp_path))).write_text(FIXTURE.read_text())
        return 0

    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path, runner=fake_runner)
    trajs, endpoints = client.run(specs)
    assert [t.trial_id for t in trajs] == ["t_go", "t_miss"]
    assert trajs[0].selected_channel == 0
    assert trajs[1].selected_channel == -1
    assert endpoints == [[0.30, 0.40], [0.10, 0.05]]


def test_client_fails_fast_on_nonzero_exit(tmp_path):
    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path, runner=lambda argv: 1)
    with pytest.raises(RuntimeError, match="container exited"):
        client.run([ReachSpec(trial_id="x", selected_channel=0, onset_time_ms=0.0,
                              gate_gain=1.0, gate_state="open")])


def test_client_fails_fast_on_trial_id_mismatch(tmp_path):
    def bad_runner(argv):
        out = argv[argv.index("--out") + 1]
        Path(out.replace("/io", str(tmp_path))).write_text(FIXTURE.read_text())
        return 0
    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path, runner=bad_runner)
    with pytest.raises(ValueError, match="trial_id"):
        client.run([ReachSpec(trial_id="not_in_fixture", selected_channel=0,
                              onset_time_ms=0.0, gate_gain=1.0, gate_state="open")])


def test_client_fails_fast_on_missing_output(tmp_path):
    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path, runner=lambda argv: 0)
    with pytest.raises(RuntimeError, match="no output"):
        client.run([ReachSpec(trial_id="x", selected_channel=0, onset_time_ms=0.0,
                              gate_gain=1.0, gate_state="open")])


# --- Task 10.4: host dry-run of the go/no-go sweep orchestration (no Docker) ---


def test_gonogo_condition_builds_specs_without_container(tmp_path):
    """Prove run_opensim_gonogo_condition assembles specs and scores trajectories
    with a fake runner echoing one no-movement trajectory per job — no container."""
    from experiments.opensim_gonogo_sweep import run_opensim_gonogo_condition

    captured = {}

    def fake_runner(argv):
        jobs = json.loads((tmp_path / "jobs.json").read_text())["jobs"]
        captured["n"] = len(jobs)
        out = argv[argv.index("--out") + 1]
        # echo a zero-movement trajectory per job (trial_id required for id match)
        trajs = [{"trial_id": j["trial_id"], "times_ms": [0.0, 5.0],
                  "positions_xy": [[0.0, 0.0], [0.0, 0.0]],
                  "onset_time_ms": None, "selected_channel": -1} for j in jobs]
        Path(out.replace("/io", str(tmp_path))).write_text(json.dumps(
            {"target_endpoints_xy": [[0.3, 0.4], [0.1, 0.05]], "trajectories": trajs}))
        return 0

    cfg = OpenSimPlantConfig(image="x", q0=[0.0, 0.35], q_target=[[1.2, 1.0], [0.8, 1.6]],
                             kp=[120.0, 90.0], kd=[18.0, 14.0])
    client = OpenSimPlantClient(cfg, io_dir=tmp_path, runner=fake_runner)
    res = run_opensim_gonogo_condition(40.0, n_trials=8, client=client)
    assert res["n_trials"] == 8
    assert captured["n"] == 8                       # all trials shipped as specs
    assert res["n_go_trials"] >= 1                  # at least one go trial scored
    assert res["opensim_movement_onset_rate"] == 0.0  # all echoed as no-movement
