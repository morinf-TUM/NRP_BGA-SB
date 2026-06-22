import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.nrp
def test_nrpcoresim_runs_python_engine(tmp_path):
    log = tmp_path / "smoke.log"
    env = dict(os.environ, NRP_BGA_LOG=str(log))
    # The nrp env must already be sourced in the calling shell; we invoke the
    # installed binary by name.  -d sets NRPCoreSim's experiment working directory
    # to the repo root so the relative engine/TF paths in the config resolve correctly
    # (NRPCoreSim does not use the process cwd — it changes to the config dir by default).
    proc = subprocess.run(
        ["NRPCoreSim", "-c", "nrp/configs/_smoke.json", "-d", str(REPO)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert len(lines) >= 1               # at least one timestep logged
    assert "t_ns" in lines[0]
