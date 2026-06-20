"""Movement metrics extracted from kinematic reacher trajectories (Task 6.2).

Metrics are derived from a ReacherTrajectory and a ReacherConfig. The config
supplies target positions for endpoint_error. All computations are stateless
numpy operations on the trajectory arrays.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.reacher import ReacherConfig, ReacherTrajectory

# --- MovementMetrics ---


class MovementMetrics(BaseModel):
    """Scalar movement metrics extracted from one ReacherTrajectory."""

    movement_onset_time_ms: float | None   # None if gate was closed (no movement)
    endpoint_error: float                  # ‖final_pos − target‖; 0.0 if no movement
    partial_movement_amplitude: float      # ‖final_pos‖ (distance from origin)
    trajectory_curvature: float            # mean perpendicular deviation (0 for straight)
    movement_reversal_time_ms: float | None  # first velocity reversal time; None if none
    peak_velocity: float                   # max instantaneous speed (position units / ms)


# --- compute_movement_metrics ---


def compute_movement_metrics(
    trajectory: ReacherTrajectory,
    config: ReacherConfig,
) -> MovementMetrics:
    """Extract movement metrics from a simulated trajectory.

    Args:
        trajectory: ReacherTrajectory from KinematicReacher.simulate.
        config: ReacherConfig supplying target_positions for endpoint_error.
    """
    positions = np.array(trajectory.positions_xy, dtype=float)  # (n, 2)
    times = np.array(trajectory.times_ms, dtype=float)          # (n,)
    final_pos = positions[-1]                                    # (2,)

    # --- partial_movement_amplitude ---
    partial_movement_amplitude = float(np.linalg.norm(final_pos))

    # --- endpoint_error ---
    if trajectory.selected_channel >= 0:
        target = np.array(
            config.target_positions[trajectory.selected_channel], dtype=float
        )
        endpoint_error = float(np.linalg.norm(final_pos - target))
    else:
        # No movement: reacher stayed at origin, no target was attempted
        endpoint_error = 0.0

    # --- trajectory_curvature ---
    # Mean absolute perpendicular deviation from the straight line origin → final_pos.
    # For single-command minimum-jerk trajectories all positions are scalar multiples
    # of the target direction, so curvature is always 0.0 in Phase 6.
    if partial_movement_amplitude < 1e-9:
        trajectory_curvature = 0.0
    else:
        unit_dir = final_pos / partial_movement_amplitude          # (2,)
        proj_scalars = positions @ unit_dir                        # (n,)
        projected_on_line = np.outer(proj_scalars, unit_dir)       # (n, 2)
        perp_deviations = np.linalg.norm(positions - projected_on_line, axis=1)
        trajectory_curvature = float(perp_deviations.mean())

    # --- peak_velocity ---
    if len(times) > 1:
        dt = np.diff(times)                      # (n-1,)
        dpos = np.diff(positions, axis=0)        # (n-1, 2)
        # Guard against zero dt (should not occur for valid simulations)
        safe_dt = np.where(dt > 0.0, dt, np.inf)
        speeds = np.linalg.norm(dpos, axis=1) / safe_dt
        peak_velocity = float(speeds.max())
    else:
        peak_velocity = 0.0

    # --- movement_reversal_time_ms ---
    # A reversal is the first timestep where projected velocity flips from
    # positive (toward target) to negative (away from target). In Phase 6 this is
    # always None because single-command min-jerk is monotone. Phase 8 change-of-mind
    # trajectories (two policy calls, initial + post-switch command) will exercise this.
    movement_reversal_time_ms: float | None = None
    if (
        trajectory.selected_channel >= 0
        and partial_movement_amplitude > 1e-9
        and len(times) > 2
    ):
        unit_dir = final_pos / partial_movement_amplitude
        dt = np.diff(times)
        dpos = np.diff(positions, axis=0)
        safe_dt = np.where(dt > 0.0, dt, np.inf)
        proj_vel = (dpos @ unit_dir) / safe_dt  # (n-1,)
        for i in range(1, len(proj_vel)):
            if (
                (proj_vel[i - 1] > 1e-9 and proj_vel[i] < -1e-9)
                or (proj_vel[i - 1] < -1e-9 and proj_vel[i] > 1e-9)
            ):
                # Trigger: velocity projection changes sign in either direction.
                # Why: single-command trajectories only reverse pos→neg (monotone min-jerk);
                #      change-of-mind two-phase trajectories reverse neg→pos (away from ch0,
                #      then toward ch1). Both directions constitute a genuine reversal.
                # Outcome: reversal_time_ms is set for all sign changes, not just pos→neg.
                movement_reversal_time_ms = float(times[i + 1])
                break

    return MovementMetrics(
        movement_onset_time_ms=trajectory.onset_time_ms,
        endpoint_error=endpoint_error,
        partial_movement_amplitude=partial_movement_amplitude,
        trajectory_curvature=trajectory_curvature,
        movement_reversal_time_ms=movement_reversal_time_ms,
        peak_velocity=peak_velocity,
    )
