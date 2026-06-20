"""Stop-signal BG-frequency sweep library (Task 7.4).

Sweeps BG oscillation frequency across the stop-signal paradigm and
aggregates inhibition function, SSRT estimates, and validity reports
per frequency condition.

Acceptance criterion M5: N_SEEDS * N_TRIALS_PER_SEED >= 500 trials per
frequency condition.
"""

from __future__ import annotations

from pydantic import BaseModel

from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.stop_signal import StopSignalConfig, run_stop_signal_trials
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import TrialLog
from nrp_bga_sb.stop_signal_metrics import (
    StopSignalMetrics,
    StopSignalValidityReport,
    compute_stop_signal_metrics,
    validate_stop_signal_data,
)

# --- Constants ---

FREQUENCIES_HZ: list[float] = [5.0, 10.0, 20.0, 40.0, 80.0]
N_SEEDS: int = 5
# 5 seeds × 100 trials = 500 per frequency condition (M5 ≥500 requirement)
N_TRIALS_PER_SEED: int = 100
STOP_PROPORTION: float = 0.25
INITIAL_SSD_MS: int = 200
SSD_STEP_MS: int = 50
SSD_MIN_MS: int = 50
SSD_MAX_MS: int = 450
PEAK_SALIENCE: float = 0.85    # low conflict: go trials always succeed at ≥10 Hz
ACCUMULATION_MS: float = 200.0
RISE_TIME_MS: float = 200.0
DECISION_POINT_MS: int = 500   # ms after go_cue onset
GO_CUE_ONSET_MS: int = 300


# --- Result model ---


class StopSignalSweepResult(BaseModel):
    """Results for one BG frequency condition (aggregated across all seeds)."""

    frequency_hz: float
    n_trials: int                 # total trials across all seeds
    n_seeds: int
    metrics: StopSignalMetrics
    validity: StopSignalValidityReport


# --- Condition runner ---


def run_stop_signal_condition(
    frequency_hz: float,
    n_trials_per_seed: int = N_TRIALS_PER_SEED,
    n_seeds: int = N_SEEDS,
    base_seed: int = 42,
) -> StopSignalSweepResult:
    """Run one BG-frequency condition and return aggregated stop-signal metrics.

    For each seed in range(n_seeds):
      - Derive per-seed seed: base_seed + seed_index (deterministic)
      - Build ClosedLoopPolicy at frequency_hz via FrequencyConfig.from_effective_hz
      - Build StopSignalConfig with stop_trial_go_evidence=True (so ClosedLoopPolicy
        generates directed go evidence on stop trials, enabling the race model)
      - Run run_stop_signal_trials(config, policy)

    Aggregate all trials across seeds into one list.
    Compute StopSignalMetrics and StopSignalValidityReport from the combined list.

    Returns StopSignalSweepResult with all aggregated data.
    """
    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)
    cortex_cfg = CortexConfig(
        peak_salience=PEAK_SALIENCE,
        rise_time_ms=RISE_TIME_MS,
        noise_std=0.0,
    )
    # A single policy instance is shared across seeds — ClosedLoopPolicy holds no
    # per-trial mutable state; all randomness is driven by per-seed StopSignalConfig.seed.
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=ACCUMULATION_MS,
    )

    all_trials: list[TrialLog] = []
    for seed_index in range(n_seeds):
        config = StopSignalConfig(
            n_trials=n_trials_per_seed,
            stop_proportion=STOP_PROPORTION,
            initial_ssd_ms=INITIAL_SSD_MS,
            ssd_step_ms=SSD_STEP_MS,
            ssd_min_ms=SSD_MIN_MS,
            ssd_max_ms=SSD_MAX_MS,
            use_staircase=True,
            # Trigger: stop_trial_go_evidence=True sets cue_identity="go" on stop trials.
            # Why: CortexEvidenceGenerator generates directed go evidence from cue_identity="go";
            #      on stop trials, the stop_signal event in trial_log.events activates the
            #      stopping mechanism. With cue_identity="stop", cortex produces neutral evidence
            #      and BG always inhibits regardless of SSD — no race model behaviour.
            # Outcome: stop trials have a genuine go process that can be stopped or not, enabling
            #          meaningful inhibition function estimation.
            stop_trial_go_evidence=True,
            decision_point_ms=DECISION_POINT_MS,
            go_cue_onset_ms=GO_CUE_ONSET_MS,
            seed=base_seed + seed_index,
        )
        seed_trials = run_stop_signal_trials(config, policy)
        all_trials.extend(seed_trials)

    metrics = compute_stop_signal_metrics(all_trials)
    validity = validate_stop_signal_data(all_trials, intended_stop_proportion=STOP_PROPORTION)

    return StopSignalSweepResult(
        frequency_hz=frequency_hz,
        n_trials=len(all_trials),
        n_seeds=n_seeds,
        metrics=metrics,
        validity=validity,
    )


# --- Report formatter ---


def format_sweep_report(results: list[StopSignalSweepResult]) -> str:
    """Format a human-readable sweep report.

    For each frequency condition (sorted by frequency_hz ascending):
      - frequency_hz
      - n_trials, n_stop_trials
      - stop_failure_rate
      - ssrt_estimate_s (or "N/A")
      - go_rt_mean_s (or "N/A")
      - rt_check_passed (True/False) and rt_check_note summary
      - inhibition_function (SSD → failure_rate, up to 5 entries)
      - inhibition_function_monotone (True/False/None)

    Returns a multi-line string. No fixed format required — readability is the goal.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Stop-Signal BG-Frequency Sweep Report")
    lines.append("=" * 60)

    for result in sorted(results, key=lambda r: r.frequency_hz):
        m = result.metrics
        v = result.validity

        n_per_seed = result.n_trials // result.n_seeds
        lines.append("")
        lines.append(
            f"Frequency: {result.frequency_hz:.0f} Hz"
            f"  ({result.n_seeds} seeds × {n_per_seed} trials"
            f" = {result.n_trials} total)"
        )
        lines.append(
            f"  Trials:            {result.n_trials}"
            f"  (go: {m.n_go_trials}, stop: {m.n_stop_trials})"
        )

        sfr = f"{m.stop_failure_rate:.3f}" if m.stop_failure_rate is not None else "N/A"
        lines.append(f"  Stop failure rate: {sfr}")

        ssrt = f"{m.ssrt_estimate_s:.4f} s" if m.ssrt_estimate_s is not None else "N/A"
        lines.append(f"  SSRT estimate:     {ssrt}")

        go_rt = f"{m.go_rt_mean_s:.4f} s" if m.go_rt_mean_s is not None else "N/A"
        lines.append(f"  Go RT (mean):      {go_rt}")

        note_summary = v.rt_check_note[:60] + ("..." if len(v.rt_check_note) > 60 else "")
        lines.append(f"  RT check passed:   {v.rt_check_passed}  ({note_summary})")

        lines.append(f"  Inhibition function monotone: {v.inhibition_function_monotone}")

        # Show up to 5 SSD entries from inhibition function.
        inh_items = sorted(m.inhibition_function.items())[:5]
        if inh_items:
            lines.append("  Inhibition function (SSD ms → failure rate):")
            for ssd_ms, rate in inh_items:
                n = m.inhibition_function_n.get(ssd_ms, "?")
                lines.append(f"    SSD {ssd_ms:4d} ms → {rate:.3f}  (n={n})")
        else:
            lines.append("  Inhibition function: (no SSD data)")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
