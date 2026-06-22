from pathlib import Path

import pytest

from nrp.config_gen import build_config
from nrp.run import run_trial

REPO = Path(__file__).resolve().parents[2]


def _go_params():
    return {"trial_id": 0, "seed": 0, "cue_identity": "go"}


def _motor_released(trace):
    """A motor command with an open/partial gate appears somewhere in the trace."""
    for rec in trace:
        m = rec.get("motor")
        if m and m.get("gate_state") in ("open", "partial") and any(m["command"]):
            return True
    return False


@pytest.mark.nrp
def test_high_freq_go_trial_releases_motor(tmp_path):
    trace = run_trial(build_config(40.0), _go_params(), tmp_path / "hi")
    assert _motor_released(trace), "40 Hz go trial should release a motor command"


@pytest.mark.nrp
def test_low_freq_go_trial_misses(tmp_path):
    trace = run_trial(build_config(5.0), _go_params(), tmp_path / "lo")
    assert not _motor_released(trace), "5 Hz go trial should NOT release a motor command"
