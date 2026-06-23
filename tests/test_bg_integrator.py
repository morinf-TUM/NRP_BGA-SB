from nrp_bga_sb.bg_integrator import BGIntegratorDriver
from nrp_bga_sb.cortex import CortexConfig, CortexEvidenceGenerator
from nrp_bga_sb.thalamus import ThalamusConfig, ThalamusGate
from nrp_bga_sb.schemas import TrialLog

EMISSION_HZ = 160.0          # other knobs pinned high during the integration ablation
SIM_MS = 300.0               # nrp SimulationTimeout = 0.3 s


def _ever_released(integration_hz: float) -> bool:
    """Replicate the nrp pipeline on the host: drive the integrator at the emission
    cadence with the cortex ramp, and ask whether the thalamus gate ever opens
    (same release rule as nrp/score.py)."""
    drv = BGIntegratorDriver(integration_hz=integration_hz, accumulation_ms=200.0)
    cortex = CortexEvidenceGenerator(CortexConfig())
    gate = ThalamusGate(ThalamusConfig())
    trial = TrialLog(trial_id=0, seed=0, task_type="go_nogo", cue_identity="go", cue_onset_time=0.0)
    step_ms = 1000.0 / EMISSION_HZ
    t = 0.0
    released = False
    while t <= SIM_MS + 1e-9:
        ev = cortex(trial, t)
        dec = drv.advance(t, ev)
        motor = gate(dec)
        if motor.gate_state in ("open", "partial") and any(motor.command):
            released = True
        t += step_ms
    return released


def test_integration_misses_at_5hz():
    # 5 Hz: single tick at t=0 on neutral evidence; t=200 tick excluded -> never settles.
    assert _ever_released(5.0) is False


def test_integration_succeeds_at_10_20_40hz():
    assert _ever_released(10.0) is True
    assert _ever_released(20.0) is True
    assert _ever_released(40.0) is True


def test_baseline_rate_converges_like_compute():
    # 160 Hz: ~32 sweeps within the window -> converged readout -> success on the ramp.
    assert _ever_released(160.0) is True


def test_rejects_nonpositive_rate():
    import pytest
    with pytest.raises(ValueError):
        BGIntegratorDriver(integration_hz=0.0)
