"""Docker-gated Arm26 plant validation (Task 10.2). Run with: pytest -m opensim."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.opensim

IMAGE = "nrp-bga-opensim:4.6"
# Final tuned CONFIG from Task 10.2 Step 5. Downstream Tasks 10.3 / 10.4 reuse
# these exact values as the OpenSimPlantConfig defaults. Key tuning result:
# dt_ms=2.0 (a 5 ms control step under-samples the low-inertia arm and makes
# the hand zig-zag at movement onset); PD + minimum-jerk velocity feedforward
# at kp/kd below yields monotone, bounded-error reaches with peak torque ~40 Nm.
CONFIG = {
    "coordinate_names": None,
    "q0": [0.0, 0.35],
    "q_target": [[1.2, 1.0], [0.8, 1.6]],   # final tuned values from Step 5
    "kp": [120.0, 90.0], "kd": [18.0, 14.0],
    "dt_ms": 2.0, "movement_duration_ms": 300.0, "total_duration_ms": 500.0,
    "end_effector_body": None, "peak_torque_bound": 200.0,
    "endpoint_error_tol": 0.02,
}


def _docker_available() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "image", "inspect", IMAGE],
                       capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _docker_available(), reason="docker image not built")
def test_plant_validation_passes(tmp_path: Path):
    io = tmp_path
    (io / "config.json").write_text(json.dumps(CONFIG))
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{io}:/io", IMAGE,
         "validate_plant.py", "--config", "/io/config.json", "--out", "/io/report.json"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads((io / "report.json").read_text())
    assert report["passed"] is True
    assert report["deterministic"] is True
