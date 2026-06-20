"""Perturbation sweep library for Phase 9 (M10) decomposition — Task 9.1.

This module sweeps timing perturbations (latency, jitter, dropout, phase
offset) against BG update frequency on the go/no-go and stop-signal
paradigms.  Each perturbation type is crossed with each BG frequency
level to produce PerturbationSweepResult objects used in the §11
decomposition report.

The results distinguish three competing accounts of BG function:
  - Selector-bottleneck: frequency drives wrong-choice / miss rate.
  - Urgency/commitment: frequency shifts RT; choices are preserved.
  - Cancellation-bottleneck: dropout raises false alarms / stop failures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.engines.stop_signal import StopSignalConfig, run_stop_signal_trials
from nrp_bga_sb.perturbations import (
    DropoutWrapper,
    JitterWrapper,
    LatencyWrapper,
    PhaseOffsetWrapper,
)
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog
from nrp_bga_sb.stop_signal_metrics import (
    compute_stop_signal_metrics,
    validate_stop_signal_data,
)

# --- Type aliases and constants ---

PerturbationType = Literal["latency", "jitter", "dropout", "phase_offset"]

LATENCY_LEVELS_MS: list[float] = [0.0, 10.0, 25.0, 50.0, 100.0]
JITTER_STD_LEVELS_MS: list[float] = [0.0, 5.0, 10.0, 25.0]
DROPOUT_LEVELS: list[float] = [0.0, 0.01, 0.05, 0.10]
PHASE_OFFSET_FRACTIONS: list[float] = [0.0, 0.25, 0.50, 0.75]
FREQUENCIES_HZ: list[float] = [5.0, 10.0, 20.0, 40.0, 80.0]

# Go/no-go sweep parameters
N_GONOGO_SEEDS: int = 5
N_GONOGO_TRIALS_PER_SEED: int = 50   # 250 trials per condition

# Stop-signal sweep parameters (match Phase 7 stop_signal_sweep.py constants)
N_SS_SEEDS: int = 5
N_SS_TRIALS_PER_SEED: int = 100      # 500 trials per condition
SS_STOP_PROPORTION: float = 0.25
SS_INITIAL_SSD_MS: int = 200
SS_SSD_STEP_MS: int = 50
SS_SSD_MIN_MS: int = 50
SS_SSD_MAX_MS: int = 450
SS_DECISION_POINT_MS: int = 500
SS_GO_CUE_ONSET_MS: int = 300

# Shared timing parameters
PEAK_SALIENCE: float = 0.85   # low conflict — go process active at >=10 Hz
RISE_TIME_MS: float = 200.0
ACCUMULATION_MS: float = 200.0


# --- PerturbationSweepResult model ---


class PerturbationSweepResult(BaseModel):
    """Aggregated metrics for one (frequency_hz, perturbation_type, perturbation_value,
    paradigm) condition — aggregated across all seeds internally."""

    frequency_hz: float
    perturbation_type: PerturbationType
    perturbation_value: float
    perturbation_label: str
    paradigm: Literal["go_nogo", "stop_signal"]
    n_trials: int
    n_seeds: int

    # go_nogo metrics (None when paradigm == "stop_signal")
    go_success_rate: float | None = None
    false_alarm_rate: float | None = None
    bg_commitment_latency_mean: float | None = None

    # stop_signal metrics (None when paradigm == "go_nogo")
    stop_failure_rate: float | None = None
    ssrt_estimate_s: float | None = None
    go_rt_mean_s: float | None = None
    inhibition_function_monotone: bool | None = None


# --- Wrapper factory ---


def _phase_offset_ms(fraction: float, frequency_hz: float) -> float:
    """Convert a phase-offset fraction [0, 1] to milliseconds for the given frequency.

    The fraction is expressed as a fraction of one BG update period
    (period_ms = 1000 / frequency_hz).  A fraction of 0.5 shifts the BG
    decision by half a period.
    """
    period_ms = 1000.0 / frequency_hz
    return fraction * period_ms


def _make_wrapped_policy(
    perturbation_type: PerturbationType,
    perturbation_value: float,
    frequency_hz: float,
    cortex_cfg: CortexConfig,
    freq_cfg: FrequencyConfig,
) -> Callable[[TrialLog, ActionEvidence], BGDecision]:
    """Build a ClosedLoopPolicy and wrap it with the requested perturbation.

    A fresh base policy is created on every call so that DropoutWrapper
    inter-call state never leaks between seeds or conditions.

    Raises ValueError for unknown perturbation_type (fail-fast contract).
    """
    base_policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=ACCUMULATION_MS,
    )

    if perturbation_type == "latency":
        return LatencyWrapper(base_policy=base_policy, latency_ms=perturbation_value)
    elif perturbation_type == "jitter":
        return JitterWrapper(base_policy=base_policy, jitter_std_ms=perturbation_value)
    elif perturbation_type == "dropout":
        return DropoutWrapper(
            base_policy=base_policy, dropout_probability=perturbation_value
        )
    elif perturbation_type == "phase_offset":
        offset_ms = _phase_offset_ms(perturbation_value, frequency_hz)
        return PhaseOffsetWrapper(base_policy=base_policy, phase_offset_ms=offset_ms)
    else:
        raise ValueError(
            f"unknown perturbation_type {perturbation_type!r}; "
            "expected one of: 'latency', 'jitter', 'dropout', 'phase_offset'"
        )


def _make_label(perturbation_type: PerturbationType, perturbation_value: float) -> str:
    """Format a human-readable label for the perturbation level."""
    if perturbation_type == "latency":
        return f"latency={perturbation_value:.0f}ms"
    elif perturbation_type == "jitter":
        return f"jitter_std={perturbation_value:.0f}ms"
    elif perturbation_type == "dropout":
        return f"dropout={perturbation_value * 100:.0f}%"
    elif perturbation_type == "phase_offset":
        return f"phase_offset={perturbation_value * 100:.0f}%"
    else:
        return f"{perturbation_type}={perturbation_value}"


# --- Go/no-go condition runner ---


def run_gonogo_perturbation_condition(
    frequency_hz: float,
    perturbation_type: PerturbationType,
    perturbation_value: float,
    n_trials_per_seed: int = N_GONOGO_TRIALS_PER_SEED,
    n_seeds: int = N_GONOGO_SEEDS,
    base_seed: int = 42,
) -> PerturbationSweepResult:
    """Run one go/no-go perturbation condition and return aggregated metrics.

    For each seed in range(n_seeds):
      - Build a fresh wrapped policy (new DropoutWrapper per seed to reset
        inter-call state).
      - Run GoNoGoConfig with the matching timing parameters.

    Aggregate all trials and compute go_success_rate, false_alarm_rate, and
    bg_commitment_latency_mean directly from trial logs.
    """
    cortex_cfg = CortexConfig(
        peak_salience=PEAK_SALIENCE,
        rise_time_ms=RISE_TIME_MS,
        noise_std=0.0,
    )
    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)

    all_trials: list[TrialLog] = []
    for seed_index in range(n_seeds):
        # Trigger: fresh policy on every seed iteration.
        # Why: DropoutWrapper holds inter-call state (_last_decision) that must
        #      not bleed across seeds — each seed is an independent mini-block.
        # Outcome: each seed starts with a clean wrapper with _last_decision=None.
        policy = _make_wrapped_policy(
            perturbation_type, perturbation_value, frequency_hz, cortex_cfg, freq_cfg
        )
        config = GoNoGoConfig(
            n_trials=n_trials_per_seed,
            go_probability=0.7,
            response_window_start_ms=0,
            response_window_duration_ms=600,
            fixation_duration_ms=200,
            cue_onset_ms=400,
            decision_point_ms=300,
            seed=base_seed + seed_index,
        )
        seed_trials = run_go_nogo_trials(config, policy)
        all_trials.extend(seed_trials)

    # --- Aggregate go/no-go metrics from all trials ---
    go_trials = [t for t in all_trials if t.cue_identity == "go"]
    if go_trials:
        go_success_rate: float | None = sum(
            1 for t in go_trials if t.success is True
        ) / len(go_trials)
    else:
        go_success_rate = None

    # False-alarm rate: fraction of no-go trials where the BG responded
    # (movement_onset_time is not None means BG responded on a no-go trial).
    no_go_trials = [t for t in all_trials if t.cue_identity == "no_go"]
    if no_go_trials:
        false_alarm_rate: float | None = sum(
            1 for t in no_go_trials if t.movement_onset_time is not None
        ) / len(no_go_trials)
    else:
        false_alarm_rate = None

    # BG commitment latency: mean(thalamic_relay_time - cue_onset_time) in seconds.
    # thalamic_relay_time is set by ClosedLoopPolicy to bg_decision.sim_time.
    latencies = [
        t.thalamic_relay_time - t.cue_onset_time
        for t in all_trials
        if t.thalamic_relay_time is not None
    ]
    if latencies:
        bg_commitment_latency_mean: float | None = float(
            np.mean(np.array(latencies, dtype=float))
        )
    else:
        bg_commitment_latency_mean = None

    label = _make_label(perturbation_type, perturbation_value)

    return PerturbationSweepResult(
        frequency_hz=frequency_hz,
        perturbation_type=perturbation_type,
        perturbation_value=perturbation_value,
        perturbation_label=label,
        paradigm="go_nogo",
        n_trials=len(all_trials),
        n_seeds=n_seeds,
        go_success_rate=go_success_rate,
        false_alarm_rate=false_alarm_rate,
        bg_commitment_latency_mean=bg_commitment_latency_mean,
    )


# --- Stop-signal condition runner ---


def run_stopsignal_perturbation_condition(
    frequency_hz: float,
    perturbation_type: PerturbationType,
    perturbation_value: float,
    n_trials_per_seed: int = N_SS_TRIALS_PER_SEED,
    n_seeds: int = N_SS_SEEDS,
    base_seed: int = 42,
) -> PerturbationSweepResult:
    """Run one stop-signal perturbation condition and return aggregated metrics.

    Mirrors run_stop_signal_condition in stop_signal_sweep.py but wraps the
    ClosedLoopPolicy with the requested perturbation before each seed block.

    For each seed:
      - Build a fresh wrapped policy.
      - Run StopSignalConfig with stop_trial_go_evidence=True and staircase=True.

    Aggregate all trials and compute stop-signal metrics via
    compute_stop_signal_metrics and validate_stop_signal_data.
    """
    cortex_cfg = CortexConfig(
        peak_salience=PEAK_SALIENCE,
        rise_time_ms=RISE_TIME_MS,
        noise_std=0.0,
    )
    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)

    all_trials: list[TrialLog] = []
    for seed_index in range(n_seeds):
        # Trigger: fresh policy on every seed iteration.
        # Why: DropoutWrapper inter-call state must not bleed across seeds;
        #      each seed is an independent mini-block of trials.
        # Outcome: clean wrapper state at the start of each seed block.
        policy = _make_wrapped_policy(
            perturbation_type, perturbation_value, frequency_hz, cortex_cfg, freq_cfg
        )
        config = StopSignalConfig(
            n_trials=n_trials_per_seed,
            stop_proportion=SS_STOP_PROPORTION,
            initial_ssd_ms=SS_INITIAL_SSD_MS,
            ssd_step_ms=SS_SSD_STEP_MS,
            ssd_min_ms=SS_SSD_MIN_MS,
            ssd_max_ms=SS_SSD_MAX_MS,
            use_staircase=True,
            stop_trial_go_evidence=True,
            decision_point_ms=SS_DECISION_POINT_MS,
            go_cue_onset_ms=SS_GO_CUE_ONSET_MS,
            seed=base_seed + seed_index,
        )
        seed_trials = run_stop_signal_trials(config, policy)
        all_trials.extend(seed_trials)

    metrics = compute_stop_signal_metrics(all_trials)
    validity = validate_stop_signal_data(all_trials, intended_stop_proportion=SS_STOP_PROPORTION)

    label = _make_label(perturbation_type, perturbation_value)

    return PerturbationSweepResult(
        frequency_hz=frequency_hz,
        perturbation_type=perturbation_type,
        perturbation_value=perturbation_value,
        perturbation_label=label,
        paradigm="stop_signal",
        n_trials=len(all_trials),
        n_seeds=n_seeds,
        stop_failure_rate=metrics.stop_failure_rate,
        ssrt_estimate_s=metrics.ssrt_estimate_s,
        go_rt_mean_s=metrics.go_rt_mean_s,
        inhibition_function_monotone=validity.inhibition_function_monotone,
    )


# --- Decomposition report formatter ---


def format_decomposition_report(
    gonogo_results: list[PerturbationSweepResult],
    stopsignal_results: list[PerturbationSweepResult],
) -> str:
    """Format a §11 decomposition report for all perturbation types.

    One section per perturbation_type (order: latency, jitter, dropout,
    phase_offset).  Within each section, rows are sorted by
    (frequency_hz, perturbation_value).

    Floats are formatted to 3 decimal places; None values show as "N/A".
    """
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("Phase 9 — Latency/Jitter/Dropout/Phase Decomposition Report (M10)")
    lines.append("=" * 64)
    lines.append("")
    lines.append("INTERPRETATION GUIDE (§11):")
    lines.append("  - Frequency drives wrong-choice / miss rate     → selector-bottleneck")
    lines.append("  - Frequency shifts RT, choices preserved        → urgency/commitment account")
    lines.append("  - Dropout increases false alarms / stop failures → cancellation-bottleneck")
    lines.append("  - Latency/jitter shift RT only, choices intact  → timing-precision effect")

    perturbation_order: list[PerturbationType] = [
        "latency", "jitter", "dropout", "phase_offset"
    ]

    for ptype in perturbation_order:
        # --- Go/No-Go section for this perturbation type ---
        gng = [r for r in gonogo_results if r.perturbation_type == ptype]
        if gng:
            lines.append("")
            lines.append(f"--- Go/No-Go: {ptype} sweep ---")
            lines.append(
                f"  {'Freq':>6}  | {'Pert Level':<18} | {'go_success%':>11} "
                f"| {'false_alarm%':>12} | {'commit_lat_ms':>13}"
            )
            for r in sorted(gng, key=lambda x: (x.frequency_hz, x.perturbation_value)):
                gs = f"{r.go_success_rate * 100:.3f}" if r.go_success_rate is not None else "N/A"
                fa = f"{r.false_alarm_rate * 100:.3f}" if r.false_alarm_rate is not None else "N/A"
                cl_ms = (
                    f"{r.bg_commitment_latency_mean * 1000:.3f}"
                    if r.bg_commitment_latency_mean is not None
                    else "N/A"
                )
                lines.append(
                    f"  {r.frequency_hz:>6.1f}  | {r.perturbation_label:<18} "
                    f"| {gs:>11} | {fa:>12} | {cl_ms:>13}"
                )

        # --- Stop-Signal section for this perturbation type ---
        ss = [r for r in stopsignal_results if r.perturbation_type == ptype]
        if ss:
            lines.append("")
            lines.append(f"--- Stop-Signal: {ptype} sweep ---")
            lines.append(
                f"  {'Freq':>6}  | {'Pert Level':<18} | {'stop_fail%':>10} "
                f"| {'SSRT_s':>8} | {'inh_mono':>8}"
            )
            for r in sorted(ss, key=lambda x: (x.frequency_hz, x.perturbation_value)):
                sfr = (
                    f"{r.stop_failure_rate * 100:.3f}"
                    if r.stop_failure_rate is not None
                    else "N/A"
                )
                ssrt = f"{r.ssrt_estimate_s:.3f}" if r.ssrt_estimate_s is not None else "N/A"
                mono_val = r.inhibition_function_monotone
                mono = str(mono_val) if mono_val is not None else "N/A"
                lines.append(
                    f"  {r.frequency_hz:>6.1f}  | {r.perturbation_label:<18} "
                    f"| {sfr:>10} | {ssrt:>8} | {mono:>8}"
                )

    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)
