import pytest

from nrp.config_gen import build_config_sampled
from nrp.run import run_trial
from nrp.score import trace_to_outcome


def _go():
    return {"trial_id": 0, "seed": 0, "cue_identity": "go"}


@pytest.mark.nrp
def test_slow_sampling_misses_despite_fast_emission(tmp_path):
    # Low input-sampling (5 Hz) starves the BG even when emission is fast (160 Hz):
    # the sampler latches evidence only at t=0 and t=0.2, so the BG integrates
    # neutral early evidence. Sampling, not emission, drives the miss.
    cfg = build_config_sampled(input_sampling_hz=5.0, output_emission_hz=160.0)
    trace = run_trial(cfg, _go(), tmp_path / "slow_sample")
    assert trace_to_outcome(trace)["motor_released"] is False


@pytest.mark.nrp
def test_fast_sampling_hits_even_with_slow_emission(tmp_path):
    # Fast sampling (160 Hz) with slow emission (10 Hz): the BG sees the ramp peak;
    # emission still publishes in time within the 300 ms window -> release.
    cfg = build_config_sampled(input_sampling_hz=160.0, output_emission_hz=10.0)
    trace = run_trial(cfg, _go(), tmp_path / "fast_sample")
    assert trace_to_outcome(trace)["motor_released"] is True
