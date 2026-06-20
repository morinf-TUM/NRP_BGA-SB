import math

import pytest

from nrp_bga_sb.perturbation_plant import VisuomotorRotation, rotate_xy, signed_angle


def test_rotate_xy_90_degrees():
    out = rotate_xy([1.0, 0.0], math.pi / 2)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] == pytest.approx(1.0, abs=1e-9)


def test_rotate_xy_zero_is_identity():
    assert rotate_xy([0.7, -0.3], 0.0) == pytest.approx([0.7, -0.3])


def test_signed_angle_positive_ccw():
    # from +x axis to +y axis is +90 degrees
    assert signed_angle([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.pi / 2)


def test_signed_angle_negative_cw():
    assert signed_angle([1.0, 0.0], [0.0, -1.0]) == pytest.approx(-math.pi / 2)


def test_signed_angle_zero_vector_returns_zero():
    assert signed_angle([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_visuomotor_rotation_applies_degrees():
    pert = VisuomotorRotation(rotation_deg=30.0)
    out = pert.apply([1.0, 0.0])
    assert out[0] == pytest.approx(math.cos(math.radians(30.0)))
    assert out[1] == pytest.approx(math.sin(math.radians(30.0)))


def test_visuomotor_rotation_zero_is_identity():
    pert = VisuomotorRotation(rotation_deg=0.0)
    assert pert.apply([-1.0, 0.0]) == pytest.approx([-1.0, 0.0])
