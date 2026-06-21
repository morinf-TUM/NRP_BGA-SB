# visuals/trajectory_gen.py
"""On-the-fly trajectory generation for visual deliverables.

Imports from src/nrp_bga_sb/ to run the existing closed-loop pipeline and
produce (times_ms, positions_xy) arrays for animation.
"""
from __future__ import annotations

from nrp_bga_sb.cerebellum import Cerebellum
from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.perturbation_plant import VisuomotorRotation
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.thalamus import ThalamusConfig

# --- Shared configuration ---

# Single upward target so the reach looks natural and the 30° rotation
# deflects it clearly to the upper-right before cerebellar correction.
VISUAL_REACHER_CONFIG = ReacherConfig(
    target_positions=[[0.0, 1.0], [0.0, 1.0]],
    movement_duration_ms=300.0,
    dt_ms=1.0,
)

THRESHOLD_FREQUENCIES: list[int] = [5, 10, 20, 40, 80]

# GoNoGoConfig for a single go trial; decision_point_ms=200 matches
# the 200 ms accumulation window so frequency effects are observable.
_SINGLE_GO_CONFIG = GoNoGoConfig(
    n_trials=1,
    go_probability=1.0,
    response_window_start_ms=0,
    response_window_duration_ms=500,
    fixation_duration_ms=0,
    cue_onset_ms=0,
    decision_point_ms=200,
    seed=42,
)

_TOTAL_DURATION_MS = 500.0   # enough to capture the 300 ms reach


# --- Threshold trajectory generation ---

def generate_threshold_trajectories() -> list[dict]:
    """Run one go trial per frequency; return trajectory arrays.

    Returns:
        List of dicts (one per frequency in THRESHOLD_FREQUENCIES):
            frequency_hz  : int
            times_ms      : list[float]  — simulation time axis
            positions_xy  : list[[x, y]] — hand positions (all zero for misses)
            selected_channel : int       — -1 for miss
            gate_closed   : bool         — True means no movement
    """
    reacher = KinematicReacher(VISUAL_REACHER_CONFIG)
    results = []
    for freq in THRESHOLD_FREQUENCIES:
        policy = make_closed_loop_policy(
            frequency_config=FrequencyConfig.from_effective_hz(float(freq)),
            accumulation_ms=200.0,
        )
        trials = run_go_nogo_trials(_SINGLE_GO_CONFIG, policy)
        trial = trials[0]
        onset_ms = (
            trial.movement_onset_time * 1000.0
            if trial.movement_onset_time is not None
            else None
        )
        traj = reacher.simulate(
            trial.motor_command_series,
            onset_time_ms=onset_ms,
            total_duration_ms=_TOTAL_DURATION_MS,
        )
        gate_closed = traj.selected_channel == -1
        results.append({
            "frequency_hz":    freq,
            "times_ms":        traj.times_ms,
            "positions_xy":    traj.positions_xy,
            "selected_channel": traj.selected_channel,
            "gate_closed":     gate_closed,
        })
    return results


# --- Cerebellum trajectory generation ---

def generate_cerebellum_trajectories(n_trials: int = 30) -> list[dict]:
    """Run n_trials go-trials under 30° visuomotor rotation with AdaptiveFilter only.

    ForwardModelController is disabled (online_enabled=False) so the learning
    curve is visible across trials — the arc endpoint rotates gradually back
    toward the target as theta_hat decreases from 0 → ~-0.52 rad (counter-rotation).

    Returns:
        List of dicts (one per trial):
            trial_index  : int
            times_ms     : list[float]
            positions_xy : list[[x, y]]  — perturbed + partially-corrected arc
            endpoint_xy  : list[float]   — final position [x, y]
            theta_hat    : float         — AdaptiveFilter state AFTER this trial
            is_go        : bool          — always True (go_probability=1.0)
    """
    reacher = KinematicReacher(VISUAL_REACHER_CONFIG)
    cerebellum = Cerebellum(
        learning_rate=0.1,
        adaptation_enabled=True,
        online_enabled=False,   # filter-only: learning curve visible across trials
    )
    # Negative rotation: -30° (CW) of (0,1) → (0.5, 0.866), deflecting the
    # upward reach into the upper-right quadrant.  With the standard CCW
    # rotate_xy convention, rotation_deg=30 would give (-0.5, 0.866) instead.
    perturbation = VisuomotorRotation(rotation_deg=-30.0)

    config = GoNoGoConfig(
        n_trials=n_trials,
        go_probability=1.0,
        response_window_start_ms=0,
        response_window_duration_ms=500,
        fixation_duration_ms=0,
        cue_onset_ms=0,
        decision_point_ms=200,
        seed=42,
    )

    # Build a 40 Hz policy with full_open_threshold=0.20 so the gate opens
    # fully at the actual BG margin (≈0.20) for this cortex config.
    # Trigger: default full_open_threshold=0.30 exceeds the BG margin → partial
    # gate (gain=0.6), scaling the endpoint to 0.6×target instead of target.
    # Why: visual deliverable needs the endpoint at the target to clearly show
    # the 30° deflection arc; partial gain would give ambiguous position.
    # Outcome: gate_state="open", gate_gain=1.0 on every go trial.
    thalamus_cfg = ThalamusConfig(margin_threshold=0.05, full_open_threshold=0.20)
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        accumulation_ms=200.0,
        thalamus_config=thalamus_cfg,
    )
    trials = run_go_nogo_trials(config, policy)

    results = []
    for i, trial in enumerate(trials):
        onset_ms = (
            trial.movement_onset_time * 1000.0
            if trial.movement_onset_time is not None
            else None
        )
        traj = reacher.simulate_with_correction(
            trial.motor_command_series,
            onset_time_ms=onset_ms,
            total_duration_ms=_TOTAL_DURATION_MS,
            perturbation=perturbation,
            cerebellum=cerebellum,
        )
        endpoint = traj.positions_xy[-1]
        results.append({
            "trial_index":   i,
            "times_ms":      traj.times_ms,
            "positions_xy":  traj.positions_xy,
            "endpoint_xy":   endpoint,
            "theta_hat":     cerebellum.adaptive_filter.theta_hat,
            "is_go":         traj.selected_channel >= 0,
        })
    return results
