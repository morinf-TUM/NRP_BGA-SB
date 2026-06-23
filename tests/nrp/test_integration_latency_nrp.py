import pytest

pytestmark = pytest.mark.nrp

from experiments.nrp_integration_latency import measure


def test_release_latency_decreases_with_rate(tmp_path):
    """The integration knob's nrp signature: slower integration settles (releases) later.
    Monotone where integration is the bottleneck (5 > 10 > 20 Hz)."""
    r5 = measure(5.0, tmp_path)
    r10 = measure(10.0, tmp_path)
    r20 = measure(20.0, tmp_path)
    assert r5 is not None and r10 is not None and r20 is not None
    assert r5 > r10 > r20
