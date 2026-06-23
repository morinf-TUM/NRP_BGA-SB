import numpy as np

from nrp_bga_sb.bg_integrator import BGIntegratorDriver
from nrp_bga_sb.bg_model import BGIntegratorState, BGModel, BGModelConfig
from nrp_bga_sb.cortex import CortexConfig, CortexEvidenceGenerator
from nrp_bga_sb.schemas import TrialLog
from nrp_bga_sb.thalamus import ThalamusConfig, ThalamusGate

EMISSION_HZ = 160.0          # the BG engine wakes at the emission rate (others pinned high)
SIM_MS = 300.0               # nrp SimulationTimeout = 0.3 s


def _first_release_ms(integration_hz: float) -> float | None:
    """Replicate the nrp pipeline on the host: drive the integrator at the emission
    cadence with the cortex ramp, and return the first elapsed time (ms) at which the
    thalamus gate opens (same release rule as nrp/score.py), or None if never."""
    drv = BGIntegratorDriver(integration_hz=integration_hz)
    cortex = CortexEvidenceGenerator(CortexConfig())
    gate = ThalamusGate(ThalamusConfig())
    trial = TrialLog(trial_id=0, seed=0, task_type="go_nogo", cue_identity="go", cue_onset_time=0.0)
    step_ms = 1000.0 / EMISSION_HZ
    t = 0.0
    while t <= SIM_MS + 1e-9:
        ev = cortex(trial, t)
        dec = drv.advance(t, ev)
        motor = gate(dec)
        if motor.gate_state in ("open", "partial") and any(motor.command):
            return t
        t += step_ms
    return None


def test_release_latency_decreases_with_rate():
    # The integration knob's nrp signature: slower integration settles (releases) LATER.
    r5, r10, r20 = _first_release_ms(5.0), _first_release_ms(10.0), _first_release_ms(20.0)
    assert r5 is not None and r10 is not None and r20 is not None  # all functional
    assert r5 > r10 > r20  # monotone settling latency where integration is the bottleneck


def test_every_rate_eventually_releases():
    # Functional, not idempotent: even 5 Hz settles within the 300 ms sim (just late).
    for hz in (5.0, 10.0, 20.0, 160.0):
        assert _first_release_ms(hz) is not None


def test_integrator_is_non_idempotent():
    # A second sweep on the same evidence changes the readout (margin grows) — the
    # property the old stateless solver lacked.
    model = BGModel(BGModelConfig())
    sal = np.array([0.65, 0.35])
    s1 = model.step(BGIntegratorState.initial(2), sal, n_sweeps=1)
    s2 = model.step(s1, sal, n_sweeps=1)
    assert s2.decision_margin > s1.decision_margin


def test_rejects_nonpositive_rate():
    import pytest
    with pytest.raises(ValueError):
        BGIntegratorDriver(integration_hz=0.0)
