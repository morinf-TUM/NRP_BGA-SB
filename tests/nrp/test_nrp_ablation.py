import pytest

from experiments.nrp_ablation import ablate_knob, FREQUENCIES_HZ


@pytest.mark.nrp
def test_sampling_knob_has_miss_boundary_at_5hz(tmp_path):
    # The sampling knob must reproduce the headline boundary: miss at 5 Hz,
    # release at >=10 Hz, with the other three knobs pinned high.
    rates = ablate_knob("sampling", FREQUENCIES_HZ, tmp_path / "samp", n_seeds=2)
    assert rates[5.0] == 0.0
    assert rates[10.0] == 1.0


@pytest.mark.nrp
def test_all_knobs_runnable_and_monotone(tmp_path):
    # Each knob sweep completes through the runtime and is non-decreasing in
    # frequency (no knob makes higher frequency worse).
    for knob in ("sampling", "emission", "commitment"):
        rates = ablate_knob(knob, FREQUENCIES_HZ, tmp_path / knob, n_seeds=2)
        vals = [rates[hz] for hz in FREQUENCIES_HZ]
        assert vals == sorted(vals), f"{knob} not monotone: {vals}"
