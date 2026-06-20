"""Docker-gated end-to-end go/no-go embodiment smoke (Task 10.4). pytest -m opensim.

Confirms the M8 milestone survives a real OpenSim Arm26 embodiment: a low BG
frequency suppresses reaches (onset rate < 0.5) while a high frequency restores
them (onset rate > 0.5). Skips cleanly when the OpenSim image is not built.
"""
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.opensim

IMAGE = "nrp-bga-opensim:4.6"


def _docker_available() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "image", "inspect", IMAGE],
                       capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _docker_available(), reason="docker image not built")
def test_frequency_effect_survives_embodiment(tmp_path):
    from experiments.opensim_gonogo_sweep import run_opensim_gonogo_condition
    from nrp_bga_sb.opensim_plant import OpenSimPlantClient, OpenSimPlantConfig

    # TUNED plant config (Task 10.2 source of truth); total_duration 1300 ms so
    # the ~700 ms movement onset falls inside the simulation window.
    cfg = OpenSimPlantConfig(
        image=IMAGE, q0=[0.0, 0.35], q_target=[[1.2, 1.0], [0.8, 1.6]],
        kp=[120.0, 90.0], kd=[18.0, 14.0], dt_ms=2.0,
        movement_duration_ms=300.0, total_duration_ms=1300.0,
    )
    client = OpenSimPlantClient(cfg, io_dir=tmp_path)
    low = run_opensim_gonogo_condition(5.0, n_trials=10, client=client)
    high = run_opensim_gonogo_condition(40.0, n_trials=10, client=client)
    # qualitative effect: low frequency suppresses reaches, high frequency restores them
    assert low["opensim_movement_onset_rate"] < 0.5
    assert high["opensim_movement_onset_rate"] > 0.5
