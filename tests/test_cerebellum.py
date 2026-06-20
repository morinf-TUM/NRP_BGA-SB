import math

import pytest

from nrp_bga_sb.cerebellum import AdaptiveFilter


def test_adaptive_filter_rejects_bad_learning_rate():
    with pytest.raises(ValueError):
        AdaptiveFilter(learning_rate=0.0)
    with pytest.raises(ValueError):
        AdaptiveFilter(learning_rate=1.5)


def test_adaptive_filter_starts_at_zero():
    assert AdaptiveFilter().theta_hat == 0.0


def test_adaptive_filter_converges_to_perturbation():
    # Simulate the closed loop: residual error each trial is (theta - theta_hat).
    theta = math.radians(30.0)
    af = AdaptiveFilter(learning_rate=0.2)
    errors = []
    for _ in range(100):
        residual = theta - af.theta_hat  # what the feedforward failed to cancel
        errors.append(abs(residual))
        af.update(residual)
    assert af.theta_hat == pytest.approx(theta, abs=1e-3)
    # error decays monotonically toward zero
    assert errors[-1] < errors[0]
    assert all(errors[i + 1] <= errors[i] + 1e-12 for i in range(len(errors) - 1))


def test_adaptive_filter_precompensate_counter_rotates():
    af = AdaptiveFilter()
    af.theta_hat = math.radians(30.0)
    out = af.precompensate([1.0, 0.0])  # rotate by -30 deg
    assert out[1] == pytest.approx(math.sin(math.radians(-30.0)))


def test_adaptive_filter_reset():
    af = AdaptiveFilter()
    af.update(0.5)
    assert af.theta_hat != 0.0
    af.reset()
    assert af.theta_hat == 0.0
