"""Cerebellum on/off frequency-sweep library (Phase 11, Tasks 11.2 + 11.3).

Runs the Phase 6 go/no-go kinematic pipeline under a visuomotor-rotation
perturbation, with the cerebellum either engaged or absent, on the SAME BG
decisions. Reports the BG-selection guard metrics (onset / success rate) and
the accuracy metrics (endpoint deviation, angular error, learned theta_hat,
per-trial learning curve).
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.cerebellum import Cerebellum
from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.perturbation_plant import VisuomotorRotation, signed_angle
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.sweep import CONFLICT_PEAK_SALIENCE

FREQUENCIES_HZ: list[float] = [5.0, 10.0, 20.0, 40.0, 80.0]

# Match reacher_sweep timing so results are comparable across phases.
_RISE_TIME_MS: float = 200.0
_ACCUMULATION_MS: float = 200.0
_TOTAL_DURATION_MS: float = 1300.0


# --- CerebellumSweepResult ---


class CerebellumSweepResult(BaseModel):
    """Guard + accuracy metrics for one cerebellum sweep condition."""

    frequency_hz: float
    seed: int
    n_trials: int
    cerebellum_enabled: bool
    perturbation_deg: float
    # BG-selection guard metrics
    movement_onset_rate: float
    go_success_rate: float | None
    # Accuracy metrics (over movement trials; vs the desired gate-scaled endpoint)
    mean_endpoint_deviation: float
    mean_angular_error_rad: float
    final_theta_hat: float
    endpoint_deviation_by_trial: list[float]


# --- Public condition runner ---


def run_cerebellum_condition(
    frequency_hz: float,
    n_trials: int = 30,
    seed: int = 42,
    perturbation_deg: float = 30.0,
    cerebellum_enabled: bool = True,
    learning_rate: float = 0.1,
    online_gain: float = 0.5,
    accumulation_ms: float = _ACCUMULATION_MS,
    rise_time_ms: float = _RISE_TIME_MS,
) -> CerebellumSweepResult:
    """Run one go/no-go condition through the reacher under perturbation."""
    # --- Closed-loop policy (matches reacher_sweep, low conflict) ---
    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)
    cortex_cfg = CortexConfig(
        rise_time_ms=rise_time_ms,
        peak_salience=CONFLICT_PEAK_SALIENCE["low"],
        noise_std=0.0,
    )
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=accumulation_ms,
    )

    engine_cfg = GoNoGoConfig(
        n_trials=n_trials,
        go_probability=0.7,
        response_window_start_ms=0,
        response_window_duration_ms=600,
        fixation_duration_ms=200,
        cue_onset_ms=400,
        decision_point_ms=300,
        seed=seed,
    )
    trials = run_go_nogo_trials(engine_cfg, policy)

    # --- Reacher + perturbation + (optional) cerebellum ---
    reacher_cfg = ReacherConfig()
    reacher = KinematicReacher(reacher_cfg)
    perturbation = VisuomotorRotation(rotation_deg=perturbation_deg)
    cerebellum = (
        Cerebellum(learning_rate=learning_rate, online_gain=online_gain)
        if cerebellum_enabled
        else None
    )

    go_trials = [t for t in trials if t.cue_identity == "go"]
    n_go = len(go_trials)
    go_success_rate = (
        sum(1 for t in go_trials if t.success is True) / n_go if n_go else None
    )

    deviations: list[float] = []
    angular_errors: list[float] = []
    n_move = 0
    for trial in go_trials:
        if not trial.motor_command_series:
            continue
        onset_ms = (
            trial.movement_onset_time * 1000.0
            if trial.movement_onset_time is not None
            else None
        )
        traj = reacher.simulate_with_correction(
            trial.motor_command_series,
            onset_ms,
            _TOTAL_DURATION_MS,
            perturbation=perturbation,
            cerebellum=cerebellum,
        )
        if traj.onset_time_ms is None or traj.selected_channel < 0:
            continue
        n_move += 1
        # Desired (achievable) endpoint = target scaled by the final gate gain.
        last_cmd = trial.motor_command_series[-1]
        tx, ty = reacher_cfg.target_positions[traj.selected_channel]
        desired = [tx * last_cmd.gate_gain, ty * last_cmd.gate_gain]
        final = traj.positions_xy[-1]
        deviations.append(float(np.linalg.norm(np.array(final) - np.array(desired))))
        angular_errors.append(abs(signed_angle(desired, final)))

    movement_onset_rate = n_move / n_go if n_go else 0.0
    mean_dev = float(np.mean(deviations)) if deviations else 0.0
    mean_ang = float(np.mean(angular_errors)) if angular_errors else 0.0
    final_theta = cerebellum.adaptive_filter.theta_hat if cerebellum else 0.0

    return CerebellumSweepResult(
        frequency_hz=frequency_hz,
        seed=seed,
        n_trials=n_trials,
        cerebellum_enabled=cerebellum_enabled,
        perturbation_deg=perturbation_deg,
        movement_onset_rate=movement_onset_rate,
        go_success_rate=go_success_rate,
        mean_endpoint_deviation=mean_dev,
        mean_angular_error_rad=mean_ang,
        final_theta_hat=final_theta,
        endpoint_deviation_by_trial=deviations,
    )
