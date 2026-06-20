"""Reacher-augmented sweep condition runner (Tasks 6.3, 8.2).

Mirrors the Phase 5 run_condition interface but attaches a KinematicReacher
to each trial, adding movement-level metrics to the per-condition result.
The engine setup replicates sweep._run_engine locally to avoid modifying
the stable Phase 5 sweep module.

Task 8.2 adds ChangeOfMindReacherResult and run_change_of_mind_reacher_condition()
for two-phase reversal trajectory kinematics over change-of-mind trials.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.change_of_mind_metrics import compute_change_of_mind_metrics
from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.change_of_mind import ChangeOfMindConfig, run_change_of_mind_trials
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.engines.two_choice import TwoChoiceConfig, run_two_choice_trials
from nrp_bga_sb.movement_metrics import compute_movement_metrics
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import EventType, TrialLog
from nrp_bga_sb.sweep import CONFLICT_PEAK_SALIENCE

# Type aliases matching sweep.py
ConflictLevel = Literal["low", "medium", "high"]
Paradigm = Literal["go_nogo", "two_choice"]

# Timing constants matching Phase 5 (see sweep.py module docstring)
_RISE_TIME_MS: float = 200.0
_ACCUMULATION_MS: float = 200.0

# Gaussian noise for two_choice (matching sweep.py)
_TWO_CHOICE_NOISE_STD: float = 0.05


# --- ReacherConditionResult ---


class ReacherConditionResult(BaseModel):
    """All metrics for one condition: Phase 5 abstract + Phase 6 movement."""

    frequency_hz: float
    conflict_level: ConflictLevel
    paradigm: Paradigm
    seed: int
    n_trials: int
    # Phase 5 abstract metrics
    miss_rate: float | None
    go_success_rate: float | None
    timeout_rate: float | None
    bg_commitment_latency_mean: float | None
    # Phase 6 movement metrics (aggregated over trials that had movement)
    movement_onset_rate: float     # fraction of trials where movement occurred
    mean_endpoint_error: float     # mean ‖endpoint − target‖ over movement trials
    mean_partial_amplitude: float  # mean ‖endpoint‖ over movement trials
    mean_peak_velocity: float      # mean peak speed over movement trials


# --- ChangeOfMindReacherResult ---


class ChangeOfMindReacherResult(BaseModel):
    """All metrics for one change-of-mind frequency condition."""

    frequency_hz: float
    seed: int
    n_trials: int
    n_switch_trials: int
    # Behavioral (from change_of_mind_metrics.compute_change_of_mind_metrics)
    change_of_mind_probability: float | None
    perseveration_rate: float | None
    # Kinematic (aggregated over responding switch trials; None if no responding switch trials)
    mean_trajectory_reversal_time_ms: float | None
    mean_correction_cost: float | None
    # Success rate broken down by switch-delay category
    switch_success_by_category: dict[str, float]


# --- Public condition runners ---


def run_change_of_mind_reacher_condition(
    frequency_hz: float,
    n_trials: int = 40,
    seed: int = 42,
    reacher_config: ReacherConfig | None = None,
    accumulation_ms: float = _ACCUMULATION_MS,
    rise_time_ms: float = _RISE_TIME_MS,
) -> ChangeOfMindReacherResult:
    """Run one change-of-mind frequency condition with kinematics.

    Args:
        frequency_hz:    BG update frequency; sets all four BG knobs via
                         FrequencyConfig.from_effective_hz.
        n_trials:        Number of trials (all switch trials; no_switch_proportion=0.0).
        seed:            Random seed (deterministic).
        reacher_config:  KinematicReacher config; defaults to ReacherConfig().
        accumulation_ms: ScheduledBGAdapter pre-decision window (ms).
        rise_time_ms:    CortexEvidenceGenerator ramp duration (ms).
    """
    # --- Engine setup ---
    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)
    cortex_cfg = CortexConfig(rise_time_ms=rise_time_ms)
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=accumulation_ms,
    )

    # no_switch_proportion=0.0: all trials are switch trials, maximising kinematic
    # data per condition. Switch categories cycle in round-robin order so each
    # category receives n_trials / 4 trials (with 4 categories).
    config = ChangeOfMindConfig(
        n_trials=n_trials,
        no_switch_proportion=0.0,
        switch_delay_categories={"early": 50, "medium": 150, "late": 300, "very_late": 450},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=550,
        response_window_duration_ms=700,
        go_cue_onset_ms=300,
        fixation_duration_ms=200,
        initial_salience=[0.8, 0.2],
        post_switch_salience=[0.2, 0.8],
        seed=seed,
    )
    trials = run_change_of_mind_trials(config, policy)

    # --- Behavioral metrics ---
    behavioral = compute_change_of_mind_metrics(trials)

    # --- Kinematic metrics over responding switch trials ---
    cfg = reacher_config or ReacherConfig()
    reacher = KinematicReacher(cfg)

    # Simulation window must accommodate go_cue_onset_ms(300) + post_switch_decision_point_ms(550)
    # + movement_duration_ms(300) = 1150 ms; 1500 ms gives headroom.
    _TOTAL_DURATION_MS = 1500.0

    reversal_times: list[float] = []
    correction_costs: list[float] = []

    for trial in trials:
        # Trigger: skip trials without exactly 2 motor commands or without movement.
        # Why: responding switch trials always have 2 commands (pre + post switch call);
        #      non-responding trials (miss) have 2 commands but movement_onset_time is None.
        # Outcome: only trials where both phases fired and movement occurred are analysed.
        cmds = trial.motor_command_series
        if len(cmds) != 2 or trial.movement_onset_time is None:
            continue

        # Extract switch_time_ms from the evidence_change event.
        switch_time_ms: float | None = None
        for ev in trial.events:
            if ev.event_type == EventType.evidence_change:
                switch_time_ms = ev.sim_time * 1000.0
                break
        if switch_time_ms is None:
            # No evidence_change → this is a no-switch trial; skip.
            continue

        pre_switch_onset_ms = cmds[0].sim_time * 1000.0

        traj = reacher.simulate_change_of_mind(
            motor_commands=cmds,
            pre_switch_onset_ms=pre_switch_onset_ms,
            switch_time_ms=switch_time_ms,
            total_duration_ms=_TOTAL_DURATION_MS,
        )

        m = compute_movement_metrics(traj, cfg)
        if m.movement_reversal_time_ms is not None:
            reversal_times.append(m.movement_reversal_time_ms)

        positions = np.array(traj.positions_xy)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_path = float(steps.sum())
        straight_dist = float(np.linalg.norm(positions[-1]))
        correction_costs.append(total_path - straight_dist)

    mean_reversal = float(np.mean(reversal_times)) if reversal_times else None
    mean_cost = float(np.mean(correction_costs)) if correction_costs else None

    # --- Per-category success rate ---
    # Build from behavioral.switch_success_by_category; fall back to 0.0 for missing keys.
    categories = list(config.switch_delay_categories.keys())
    success_by_cat: dict[str, float] = {}
    for cat in categories:
        rate = behavioral.switch_success_by_category.get(cat)
        success_by_cat[cat] = rate if rate is not None else 0.0

    return ChangeOfMindReacherResult(
        frequency_hz=frequency_hz,
        seed=seed,
        n_trials=n_trials,
        n_switch_trials=behavioral.n_switch_trials,
        change_of_mind_probability=behavioral.change_of_mind_probability,
        perseveration_rate=behavioral.perseveration_rate,
        mean_trajectory_reversal_time_ms=mean_reversal,
        mean_correction_cost=mean_cost,
        switch_success_by_category=success_by_cat,
    )


def run_reacher_condition(
    frequency_hz: float,
    conflict_level: ConflictLevel,
    paradigm: Paradigm,
    n_trials: int,
    seed: int,
    reacher_config: ReacherConfig | None = None,
    accumulation_ms: float = _ACCUMULATION_MS,
    rise_time_ms: float = _RISE_TIME_MS,
) -> ReacherConditionResult:
    """Run one sweep condition with the kinematic reacher and return all metrics.

    Args:
        frequency_hz:    BG update frequency; applied to all four knobs via
                         FrequencyConfig.from_effective_hz.
        conflict_level:  Evidence discriminability ("low", "medium", "high").
        paradigm:        Task engine ("go_nogo" or "two_choice").
        n_trials:        Number of trials per condition.
        seed:            Random seed (deterministic).
        reacher_config:  KinematicReacher config; defaults to ReacherConfig().
        accumulation_ms: ScheduledBGAdapter pre-decision window (ms).
        rise_time_ms:    CortexEvidenceGenerator ramp duration (ms).
    """
    peak_salience = CONFLICT_PEAK_SALIENCE[conflict_level]
    noise_std = _TWO_CHOICE_NOISE_STD if paradigm == "two_choice" else 0.0

    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)
    cortex_cfg = CortexConfig(
        rise_time_ms=rise_time_ms,
        peak_salience=peak_salience,
        noise_std=noise_std,
    )
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=accumulation_ms,
    )

    trials = _run_engine(paradigm, n_trials, seed, policy)
    cfg = reacher_config or ReacherConfig()
    reacher = KinematicReacher(cfg)

    abstract = _compute_abstract_metrics(trials, paradigm)
    movement = _compute_movement_aggregate(trials, reacher, cfg, paradigm)

    return ReacherConditionResult(
        frequency_hz=frequency_hz,
        conflict_level=conflict_level,
        paradigm=paradigm,
        seed=seed,
        n_trials=n_trials,
        **abstract,
        **movement,
    )


# --- Private helpers ---


def _run_engine(
    paradigm: Paradigm,
    n_trials: int,
    seed: int,
    policy,
) -> list[TrialLog]:
    """Run the task engine and return trial logs.

    Mirrors sweep._run_engine with identical config values so that condition
    results are comparable between Phase 5 and Phase 6.
    """
    if paradigm == "go_nogo":
        config = GoNoGoConfig(
            n_trials=n_trials,
            go_probability=0.7,
            response_window_start_ms=0,
            response_window_duration_ms=600,
            fixation_duration_ms=200,
            cue_onset_ms=400,
            decision_point_ms=300,
            seed=seed,
        )
        return run_go_nogo_trials(config, policy)
    elif paradigm == "two_choice":
        config = TwoChoiceConfig(
            n_trials=n_trials,
            conflict_levels={"conflict": [0.65, 0.35]},
            response_window_start_ms=0,
            response_window_duration_ms=600,
            fixation_duration_ms=200,
            target_onset_ms=400,
            decision_point_ms=300,
            seed=seed,
        )
        return run_two_choice_trials(config, policy)
    else:
        raise ValueError(f"Unsupported paradigm: {paradigm!r}")


def _compute_abstract_metrics(trials: list[TrialLog], paradigm: Paradigm) -> dict:
    """Mirror Phase 5 abstract metrics for comparison."""
    n = len(trials)

    if paradigm == "go_nogo":
        go_trials = [t for t in trials if t.cue_identity == "go"]
        if go_trials:
            miss_rate: float | None = (
                sum(1 for t in go_trials if t.failure_mode == "miss") / len(go_trials)
            )
            go_success_rate: float | None = (
                sum(1 for t in go_trials if t.success is True) / len(go_trials)
            )
        else:
            miss_rate = go_success_rate = None
        timeout_rate: float | None = None
    else:
        miss_rate = go_success_rate = None
        timeouts = sum(1 for t in trials if t.failure_mode == "timeout")
        timeout_rate = timeouts / n if n > 0 else None

    latencies = [
        t.thalamic_relay_time - t.cue_onset_time
        for t in trials
        if t.thalamic_relay_time is not None
    ]
    bg_commitment_latency_mean: float | None = (
        float(np.mean(latencies)) if latencies else None
    )

    return {
        "miss_rate": miss_rate,
        "go_success_rate": go_success_rate,
        "timeout_rate": timeout_rate,
        "bg_commitment_latency_mean": bg_commitment_latency_mean,
    }


def _compute_movement_aggregate(
    trials: list[TrialLog],
    reacher: KinematicReacher,
    config: ReacherConfig,
    paradigm: Paradigm,
) -> dict:
    """Run the reacher on each trial and aggregate movement metrics.

    For go_nogo, movement_onset_rate is computed over go trials only so that
    it tracks go_success_rate (both share the same denominator: n_go_trials).
    For two_choice, movement_onset_rate is computed over all trials.
    This alignment is required by the Phase 6 acceptance check:
      abs(movement_onset_rate - go_success_rate) < 0.05.

    total_duration_ms is set to 1300 ms to accommodate the go_nogo engine's
    movement_onset_time ≈ 700 ms (cue_onset_ms=400 + decision_point_ms=300)
    plus the default movement_duration_ms=300 ms.
    """
    # Trials to compute movement_onset_rate denominator over.
    # Trigger: go_nogo has go and no_go trials; only go trials can yield movement.
    # Why: no_go trials always produce closed gates; including them in the denominator
    #      would halve movement_onset_rate relative to go_success_rate, breaking the
    #      acceptance check that requires the two rates to be within 0.05 of each other.
    # Outcome: movement_onset_rate == go_success_rate (within floating-point) at all freqs.
    if paradigm == "go_nogo":
        denominator_trials = [t for t in trials if t.cue_identity == "go"]
    else:
        denominator_trials = trials

    n_denominator = len(denominator_trials)

    # Simulation window must be longer than movement_onset_time + movement_duration.
    # The go_nogo engine places onset at cue_onset_ms(400) + decision_point_ms(300) = 700 ms.
    # Adding movement_duration_ms(300 ms) gives a minimum window of 1000 ms; 1300 ms gives
    # headroom for any future timing changes.
    _TOTAL_DURATION_MS = 1300.0

    movement_metrics_map: dict[int, object] = {}  # trial index → MovementMetrics
    for i, trial in enumerate(denominator_trials):
        if not trial.motor_command_series:
            # Trigger: policy was never called (should not happen for ClosedLoopPolicy).
            # Why: guard against a task engine that skips the policy on some trials.
            continue
        onset_ms = (
            trial.movement_onset_time * 1000.0
            if trial.movement_onset_time is not None
            else None
        )
        traj = reacher.simulate(trial.motor_command_series, onset_ms, _TOTAL_DURATION_MS)
        m = compute_movement_metrics(traj, config)
        movement_metrics_map[i] = m

    all_metrics = list(movement_metrics_map.values())
    movement_trials = [m for m in all_metrics if m.movement_onset_time_ms is not None]
    n_move = len(movement_trials)

    return {
        "movement_onset_rate": n_move / n_denominator if n_denominator > 0 else 0.0,
        "mean_endpoint_error": (
            float(np.mean([m.endpoint_error for m in movement_trials]))
            if movement_trials else 0.0
        ),
        "mean_partial_amplitude": (
            float(np.mean([m.partial_movement_amplitude for m in movement_trials]))
            if movement_trials else 0.0
        ),
        "mean_peak_velocity": (
            float(np.mean([m.peak_velocity for m in movement_trials]))
            if movement_trials else 0.0
        ),
    }
