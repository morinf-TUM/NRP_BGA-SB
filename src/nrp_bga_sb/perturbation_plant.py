"""Motor perturbation plant + 2D geometry helpers (Phase 11, Task 11.1).

VisuomotorRotation injects a fixed angular distortion into an executed reach
endpoint — the canonical sensorimotor-adaptation perturbation the cerebellum
must learn to cancel. The two free functions (rotate_xy, signed_angle) are the
shared 2D geometry used by the cerebellar layers and the sweep metrics.
"""
from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

# --- Geometry helpers ---


def rotate_xy(vec: list[float] | np.ndarray, theta_rad: float) -> list[float]:
    """Rotate a 2D vector about the origin by theta_rad (CCW positive)."""
    v = np.asarray(vec, dtype=float)
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return [float(c * v[0] - s * v[1]), float(s * v[0] + c * v[1])]


def signed_angle(
    v_from: list[float] | np.ndarray, v_to: list[float] | np.ndarray
) -> float:
    """Signed angle (rad, CCW positive) rotating v_from onto v_to.

    Returns 0.0 if either vector is degenerate (near-zero norm): an undefined
    direction carries no angular error.
    """
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return 0.0
    # atan2 of the 2D cross and dot products gives a signed angle in (-pi, pi].
    cross = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    return float(math.atan2(cross, dot))


# --- VisuomotorRotation ---


class VisuomotorRotation(BaseModel):
    """Fixed-angle visuomotor rotation applied to an executed reach endpoint."""

    rotation_deg: float = 30.0

    def apply(self, endpoint_xy: list[float]) -> list[float]:
        """Rotate the endpoint about the origin by rotation_deg."""
        return rotate_xy(endpoint_xy, math.radians(self.rotation_deg))
