"""Docker-gated e2e cerebellar correction smoke (Task 11b.2). pytest -m opensim.

Confirms M9 embodied: cerebellar adaptation reduces endpoint deviation under a
visuomotor rotation, while the BG-frequency onset signature is unchanged.
Skips cleanly when the OpenSim image is not built.
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
def test_cerebellar_correction_survives_embodiment(tmp_path):
    """M9 embodied acceptance:
    - BG-frequency onset signature is unchanged by the cerebellar layer.
    - Cerebellar adaptation reduces mean endpoint deviation vs no correction.
    """
    from experiments.opensim_cerebellum_sweep import run_opensim_cerebellum_condition
    from nrp_bga_sb.opensim_plant import OpenSimPlantClient, OpenSimPlantConfig

    cfg = OpenSimPlantConfig(
        image=IMAGE, q0=[0.0, 0.35], q_target=[[1.2, 1.0], [0.8, 1.6]],
        kp=[120.0, 90.0], kd=[18.0, 14.0], dt_ms=2.0,
        movement_duration_ms=300.0, total_duration_ms=1300.0,
    )
    # High BG frequency: movements execute, so cerebellar learning is active.
    client_on = OpenSimPlantClient(cfg, io_dir=tmp_path / "on")
    client_off = OpenSimPlantClient(cfg, io_dir=tmp_path / "off")
    res_on = run_opensim_cerebellum_condition(40.0, n_trials=50, client=client_on,
                                              cerebellum_enabled=True)
    res_off = run_opensim_cerebellum_condition(40.0, n_trials=50, client=client_off,
                                               cerebellum_enabled=False)

    # BG-effect guard: onset rate must be identical with/without cerebellum.
    assert res_on["opensim_movement_onset_rate"] == res_off["opensim_movement_onset_rate"]
    # Accuracy: cerebellar adaptation reduces deviation under perturbation.
    assert res_on["opensim_mean_endpoint_deviation"] < res_off["opensim_mean_endpoint_deviation"]
    # Low BG frequency: no movements, onset rate = 0, deviation = 0 in both conditions.
    client_low = OpenSimPlantClient(cfg, io_dir=tmp_path / "low")
    res_low = run_opensim_cerebellum_condition(5.0, n_trials=10, client=client_low,
                                               cerebellum_enabled=True)
    assert res_low["opensim_movement_onset_rate"] == 0.0
    assert res_low["opensim_mean_endpoint_deviation"] == 0.0
