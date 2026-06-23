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


@pytest.mark.nrp
def test_sweep_categorical_signature(tmp_path):
    from experiments.nrp_gonogo_sweep import run_sweep, FREQUENCIES_HZ
    rates = run_sweep(FREQUENCIES_HZ, n_seeds=2, run_root=tmp_path / "sweep")
    assert rates[5.0] == 0.0                       # 5 Hz: all miss
    assert all(rates[hz] == 1.0 for hz in (10.0, 20.0, 40.0, 80.0, 160.0))
