"""Frequency-sweep experiment: condition runner and result types (Task 5.1).

A 'condition' is a unique (frequency_hz, conflict_level, paradigm, seed) tuple.
run_condition runs one condition with ClosedLoopPolicy and returns a
SweepConditionResult carrying both standard scorer metrics and phase-5-specific
metrics computed directly from trial-level data.

Parameter decisions:
  rise_time_ms=200.0 and accumulation_ms=200.0 place the frequency-dependent
  selection boundary within the {10, 20, 40, 80, 160 Hz} sweep range.
  See docs/superpowers/plans/2026-06-19-phase5-frequency-sweep.md §Key parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.engines.two_choice import TwoChoiceConfig, run_two_choice_trials
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog
from nrp_bga_sb.scorer import score_trials

# --- Type aliases ---

ConflictLevel = Literal["low", "medium", "high"]
Paradigm = Literal["go_nogo", "two_choice"]

# --- Conflict → peak salience mapping ---

# Salience at key ticks (rise_time_ms=200, accumulation_ms=200, n_steps=200):
#   tick 100 (10 Hz second firing): salience = 0.5 + (peak-0.5) * 0.5
#   tick 150 (20 Hz third firing):  salience = 0.5 + (peak-0.5) * 0.75
# GPR selection boundary: salience > ~0.600 triggers selection (T_winner > 0).
# Verified: [0.600, 0.400] → no selection; [0.610, 0.390] → selected.
CONFLICT_PEAK_SALIENCE: dict[str, float] = {
    "low": 0.85,  # tick 100 → 0.675 → selects at 10 Hz (> GPR threshold ~0.600)
    "medium": 0.69,  # tick 100 → 0.595 → miss; tick 150 → 0.643 → selects at 20 Hz
    "high": 0.68,  # tick 100 → 0.590 → miss; tick 150 → 0.635 → selects at 20 Hz
}

# Default sweep timing parameters (see module docstring).
_SWEEP_RISE_TIME_MS: float = 200.0
_SWEEP_ACCUMULATION_MS: float = 200.0

# Gaussian noise for two_choice to make wrong-target errors possible.
# Zero for go_nogo: miss/false-alarm are frequency-driven, not noise-driven.
_TWO_CHOICE_NOISE_STD: float = 0.05


# --- Result type ---


class SweepConditionResult(BaseModel):
    """All metrics for one (frequency_hz, conflict_level, paradigm, seed) condition."""

    frequency_hz: float
    conflict_level: ConflictLevel
    paradigm: Paradigm
    seed: int
    n_trials: int
    # Standard scorer metrics
    reaction_time_mean: float | None
    wrong_action_rate: float  # go_nogo: response outside response window
    wrong_target_rate: float  # two_choice: selected wrong (lower-salience) channel
    false_alarm_rate: float | None  # go_nogo: no-go trial where BG responded
    # Phase-5-specific metrics computed directly from trial logs
    miss_rate: float | None  # go_nogo: go trial where BG returned -1 (key metric)
    timeout_rate: float | None  # two_choice: no channel selected (BG returned -1)
    go_success_rate: float | None  # go_nogo: fraction of go trials that succeeded
    bg_commitment_latency_mean: float | None  # mean(thalamic_relay_time - cue_onset_time) in s
    bg_commitment_latency_std: float | None


# --- Condition runner ---


def run_condition(
    frequency_hz: float,
    conflict_level: ConflictLevel,
    paradigm: Paradigm,
    n_trials: int,
    seed: int,
    accumulation_ms: float = _SWEEP_ACCUMULATION_MS,
    rise_time_ms: float = _SWEEP_RISE_TIME_MS,
) -> SweepConditionResult:
    """Run one sweep condition and return its metrics.

    Args:
        frequency_hz:    BG update frequency (Hz); applied to all four knobs via
                         FrequencyConfig.from_effective_hz.
        conflict_level:  Evidence discriminability ("low", "medium", "high").
        paradigm:        Task engine ("go_nogo" or "two_choice").
        n_trials:        Number of trials per condition.
        seed:            Random seed for trial generation (deterministic).
        accumulation_ms: ScheduledBGAdapter pre-decision integration window (ms).
        rise_time_ms:    CortexEvidenceGenerator ramp duration (ms).

    Returns:
        SweepConditionResult with scorer and phase-5-specific metrics.
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

    condition_id = f"{paradigm}_{conflict_level}_{frequency_hz:.0f}hz_seed{seed}"
    trials = _run_engine(paradigm, n_trials, seed, policy)
    metrics = score_trials(trials, condition_id=condition_id, bg_frequency_hz=frequency_hz)
    extra = _compute_phase5_metrics(trials, paradigm)

    return SweepConditionResult(
        frequency_hz=frequency_hz,
        conflict_level=conflict_level,
        paradigm=paradigm,
        seed=seed,
        n_trials=n_trials,
        reaction_time_mean=metrics.reaction_time_mean,
        wrong_action_rate=metrics.wrong_action_rate,
        wrong_target_rate=metrics.wrong_target_rate,
        false_alarm_rate=metrics.false_alarm_rate,
        **extra,
    )


# --- Private helpers ---


def _run_engine(
    paradigm: Paradigm,
    n_trials: int,
    seed: int,
    policy: Callable[[TrialLog, ActionEvidence], BGDecision],
) -> list[TrialLog]:
    """Dispatch to the appropriate task engine runner."""
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
        # conflict_levels is required by TwoChoiceConfig but its salience values are
        # overridden by the ClosedLoopPolicy cortex_generator at every input-sampling tick.
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
        raise ValueError(f"Unsupported paradigm: {paradigm!r}. Must be 'go_nogo' or 'two_choice'.")


def _compute_phase5_metrics(trials: list[TrialLog], paradigm: Paradigm) -> dict:
    """Compute phase-5 metrics not available in the standard scorer."""
    n = len(trials)

    # --- Paradigm-specific rate metrics ---
    if paradigm == "go_nogo":
        go_trials = [t for t in trials if t.cue_identity == "go"]
        if go_trials:
            missed = sum(1 for t in go_trials if t.failure_mode == "miss")
            succeeded = sum(1 for t in go_trials if t.success is True)
            miss_rate: float | None = missed / len(go_trials)
            go_success_rate: float | None = succeeded / len(go_trials)
        else:
            miss_rate = None
            go_success_rate = None
        timeout_rate: float | None = None
    else:
        miss_rate = None
        go_success_rate = None
        timeouts = sum(1 for t in trials if t.failure_mode == "timeout")
        timeout_rate = timeouts / n if n > 0 else None

    # --- BG commitment latency ---
    # thalamic_relay_time is set by ClosedLoopPolicy from bg_decision.sim_time,
    # which equals cue_onset_time + committed_tick * base_dt_ms / 1000.
    # The difference (relay_time - cue_onset_time) is the elapsed time (s) from
    # cue onset to when BG made its final commitment within the accumulation window.
    latencies = [
        t.thalamic_relay_time - t.cue_onset_time
        for t in trials
        if t.thalamic_relay_time is not None
    ]
    if latencies:
        lat_arr = np.array(latencies, dtype=float)
        bg_commitment_latency_mean: float | None = float(lat_arr.mean())
        bg_commitment_latency_std: float | None = (
            float(lat_arr.std(ddof=1)) if len(latencies) > 1 else None
        )
    else:
        bg_commitment_latency_mean = None
        bg_commitment_latency_std = None

    return {
        "miss_rate": miss_rate,
        "timeout_rate": timeout_rate,
        "go_success_rate": go_success_rate,
        "bg_commitment_latency_mean": bg_commitment_latency_mean,
        "bg_commitment_latency_std": bg_commitment_latency_std,
    }
