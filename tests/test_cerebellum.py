import math

import numpy as np
import pytest

from nrp_bga_sb.cerebellum import AdaptiveFilter, Cerebellum, ForwardModelController


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


def _min_jerk_s(n: int) -> list[float]:
    out = []
    for i in range(n):
        tau = i / (n - 1)
        out.append(10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    return out


def test_forward_model_rejects_bad_gain():
    with pytest.raises(ValueError):
        ForwardModelController(gain=-0.1)
    with pytest.raises(ValueError):
        ForwardModelController(gain=1.1)


def test_forward_model_gain_zero_reproduces_openloop():
    s = _min_jerk_s(50)
    D = [1.0, 0.0]
    P = [math.cos(math.radians(30)), math.sin(math.radians(30))]  # rotated endpoint
    fmc = ForwardModelController(gain=0.0)
    traj = fmc.integrate(D, P, s)
    # endpoint matches the open-loop perturbed endpoint P
    assert traj[-1] == pytest.approx(P, abs=1e-6)


def test_forward_model_gain_reduces_endpoint_error():
    s = _min_jerk_s(200)
    D = np.array([1.0, 0.0])
    P = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
    err_open = float(np.linalg.norm(P - D))
    fmc = ForwardModelController(gain=0.6)
    traj = fmc.integrate(list(D), list(P), s)
    err_corrected = float(np.linalg.norm(np.array(traj[-1]) - D))
    assert err_corrected < err_open


def test_forward_model_output_length_matches_s():
    s = _min_jerk_s(37)
    traj = ForwardModelController(gain=0.5).integrate([1.0, 0.0], [0.0, 1.0], s)
    assert len(traj) == 37


def test_cerebellum_precompensate_identity_when_adaptation_off():
    cb = Cerebellum(adaptation_enabled=False)
    cb.adaptive_filter.theta_hat = math.radians(30.0)  # would rotate if used
    assert cb.precompensate([1.0, 0.0]) == pytest.approx([1.0, 0.0])


def test_cerebellum_learn_noop_when_adaptation_off():
    cb = Cerebellum(adaptation_enabled=False)
    cb.learn(0.5)
    assert cb.adaptive_filter.theta_hat == 0.0


def test_cerebellum_learn_updates_when_adaptation_on():
    cb = Cerebellum(adaptation_enabled=True, learning_rate=0.2)
    cb.learn(1.0)
    assert cb.adaptive_filter.theta_hat == pytest.approx(0.2)


def test_cerebellum_integrate_straight_line_when_online_off():
    s = _min_jerk_s(50)
    P = [0.5, 0.5]
    cb = Cerebellum(online_enabled=False)
    traj = cb.integrate([1.0, 0.0], P, s)
    assert traj[-1] == pytest.approx(P, abs=1e-6)  # ends at open-loop endpoint


def test_cerebellum_reset_clears_filter():
    cb = Cerebellum()
    cb.learn(0.5)
    cb.reset()
    assert cb.adaptive_filter.theta_hat == 0.0
