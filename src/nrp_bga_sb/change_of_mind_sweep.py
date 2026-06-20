"""Change-of-mind BG-frequency sweep library (Task 8.3, Milestone M6).

Sweeps BG oscillation frequency across the change-of-mind paradigm and
aggregates switch success rates, perseveration rates, revision latencies, and
kinematic reversal metrics per frequency condition.

Acceptance criterion M6 (≥300 switch trials per condition):
  N_SEEDS * N_TRIALS_PER_SEED = 5 × 80 = 400 total trials (all switch trials).
"""

from __future__ import annotations

from pydantic import BaseModel

from nrp_bga_sb.change_of_mind_metrics import compute_change_of_mind_metrics
from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.change_of_mind import ChangeOfMindConfig, run_change_of_mind_trials
from nrp_bga_sb.reacher_sweep import run_change_of_mind_reacher_condition
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import TrialLog

# --- Constants ---

FREQUENCIES_HZ: list[float] = [5.0, 10.0, 20.0, 40.0, 80.0]
N_SEEDS: int = 5
# 5 seeds × 80 trials = 400 per frequency condition (M6 ≥300 requirement)
N_TRIALS_PER_SEED: int = 80
NO_SWITCH_PROPORTION: float = 0.0   # all switch trials — maximise statistical power
SWITCH_DELAY_CATEGORIES: dict[str, int] = {
    "early": 50,
    "medium": 150,
    "late": 300,
    "very_late": 450,
}
INITIAL_DECISION_POINT_MS: int = 20
POST_SWITCH_DECISION_POINT_MS: int = 550
PEAK_SALIENCE: float = 0.85
ACCUMULATION_MS: float = 200.0
RISE_TIME_MS: float = 200.0
GO_CUE_ONSET_MS: int = 300


# --- Result model ---


class ChangeOfMindSweepResult(BaseModel):
    """Results for one BG frequency condition (aggregated across all seeds)."""

    frequency_hz: float
    n_trials: int        # total switch trials across all seeds
    n_seeds: int
    # Behavioral metrics (from compute_change_of_mind_metrics)
    change_of_mind_probability: float | None
    perseveration_rate: float | None
    mean_revision_latency_ms: float | None
    # Per-category breakdown
    switch_success_by_category: dict[str, float]
    perseveration_by_category: dict[str, float]
    revision_latency_by_category: dict[str, float]
    # Kinematic metrics (from run_change_of_mind_reacher_condition, one call per seed)
    mean_trajectory_reversal_time_ms: float | None
    mean_correction_cost: float | None


# --- Condition runner ---


def run_change_of_mind_condition(
    frequency_hz: float,
    n_trials_per_seed: int = N_TRIALS_PER_SEED,
    n_seeds: int = N_SEEDS,
    base_seed: int = 42,
) -> ChangeOfMindSweepResult:
    """Run one BG-frequency condition and return aggregated change-of-mind metrics.

    For each seed in range(n_seeds):
      - Derive per-seed seed: base_seed + seed_index (deterministic)
      - Build ClosedLoopPolicy at frequency_hz via FrequencyConfig.from_effective_hz
      - Build ChangeOfMindConfig (NO_SWITCH_PROPORTION=0.0, SWITCH_DELAY_CATEGORIES)
      - Run run_change_of_mind_trials(config, policy)

    Aggregate all trials across seeds. Compute behavioral metrics.

    For kinematic metrics, run run_change_of_mind_reacher_condition(
        frequency_hz=frequency_hz,
        n_trials=n_trials_per_seed,
        seed=base_seed,   # single-seed for kinematics (expensive to run per-seed)
        accumulation_ms=ACCUMULATION_MS,
        rise_time_ms=RISE_TIME_MS,
    ) and pull mean_trajectory_reversal_time_ms and mean_correction_cost from the result.
    """
    # --- Build shared policy ---
    # A single policy instance is reused across seeds — ClosedLoopPolicy holds no
    # per-trial mutable state; all randomness is driven by per-seed ChangeOfMindConfig.seed.
    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)
    cortex_cfg = CortexConfig(
        peak_salience=PEAK_SALIENCE,
        rise_time_ms=RISE_TIME_MS,
        noise_std=0.0,
    )
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=ACCUMULATION_MS,
    )

    # --- Collect trials across seeds ---
    all_trials: list[TrialLog] = []
    for seed_index in range(n_seeds):
        config = ChangeOfMindConfig(
            n_trials=n_trials_per_seed,
            no_switch_proportion=NO_SWITCH_PROPORTION,
            switch_delay_categories=dict(SWITCH_DELAY_CATEGORIES),
            initial_decision_point_ms=INITIAL_DECISION_POINT_MS,
            post_switch_decision_point_ms=POST_SWITCH_DECISION_POINT_MS,
            go_cue_onset_ms=GO_CUE_ONSET_MS,
            seed=base_seed + seed_index,
        )
        seed_trials = run_change_of_mind_trials(config, policy)
        all_trials.extend(seed_trials)

    # --- Behavioral metrics over all seeds ---
    behavioral = compute_change_of_mind_metrics(all_trials)

    # --- Kinematic metrics: single seed (expensive; one call suffices for the sweep) ---
    # Trigger: kinematic simulation is costly per trial; running per-seed would multiply
    #          compute time by N_SEEDS without meaningfully improving metric stability.
    # Why: the reacher is deterministic given the same BG decision sequence; the base_seed
    #      trial distribution is representative of all seeds.
    # Outcome: mean_trajectory_reversal_time_ms and mean_correction_cost come from one seed.
    kinematic = run_change_of_mind_reacher_condition(
        frequency_hz=frequency_hz,
        n_trials=n_trials_per_seed,
        seed=base_seed,
        accumulation_ms=ACCUMULATION_MS,
        rise_time_ms=RISE_TIME_MS,
    )

    return ChangeOfMindSweepResult(
        frequency_hz=frequency_hz,
        n_trials=len(all_trials),
        n_seeds=n_seeds,
        change_of_mind_probability=behavioral.change_of_mind_probability,
        perseveration_rate=behavioral.perseveration_rate,
        mean_revision_latency_ms=behavioral.mean_revision_latency_ms,
        switch_success_by_category=behavioral.switch_success_by_category,
        perseveration_by_category=behavioral.perseveration_by_category,
        revision_latency_by_category=behavioral.revision_latency_by_category,
        mean_trajectory_reversal_time_ms=kinematic.mean_trajectory_reversal_time_ms,
        mean_correction_cost=kinematic.mean_correction_cost,
    )


# --- Report formatter ---


def format_sweep_report(results: list[ChangeOfMindSweepResult]) -> str:
    """Format a human-readable sweep report.

    For each frequency condition (sorted by frequency_hz ascending):
      - frequency_hz, n_seeds, n_trials
      - change_of_mind_probability, perseveration_rate
      - mean_revision_latency_ms
      - mean_trajectory_reversal_time_ms, mean_correction_cost
      - switch_success_by_category (all 4 categories, one per line)
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Change-of-mind BG-Frequency Sweep Report")
    lines.append("=" * 60)

    for result in sorted(results, key=lambda r: r.frequency_hz):
        n_per_seed = result.n_trials // result.n_seeds if result.n_seeds > 0 else 0
        lines.append("")
        lines.append(
            f"Frequency: {result.frequency_hz:.0f} Hz"
            f"  ({result.n_seeds} seeds × {n_per_seed} trials"
            f" = {result.n_trials} total)"
        )

        # --- Behavioral ---
        com_prob = (
            f"{result.change_of_mind_probability:.3f}"
            if result.change_of_mind_probability is not None
            else "N/A"
        )
        lines.append(f"  Change-of-mind probability: {com_prob}")

        persev = (
            f"{result.perseveration_rate:.3f}"
            if result.perseveration_rate is not None
            else "N/A"
        )
        lines.append(f"  Perseveration rate:         {persev}")

        latency = (
            f"{result.mean_revision_latency_ms:.1f} ms"
            if result.mean_revision_latency_ms is not None
            else "N/A"
        )
        lines.append(f"  Mean revision latency:      {latency}")

        # --- Kinematic ---
        reversal = (
            f"{result.mean_trajectory_reversal_time_ms:.1f} ms"
            if result.mean_trajectory_reversal_time_ms is not None
            else "N/A"
        )
        lines.append(f"  Mean trajectory reversal:   {reversal}")

        cost = (
            f"{result.mean_correction_cost:.4f}"
            if result.mean_correction_cost is not None
            else "N/A"
        )
        lines.append(f"  Mean correction cost:       {cost}")

        # --- Per-category switch success ---
        lines.append("  Switch success by category:")
        for cat in ("early", "medium", "late", "very_late"):
            rate = result.switch_success_by_category.get(cat)
            rate_str = f"{rate:.3f}" if rate is not None else "N/A"
            lines.append(f"    {cat:<12} {rate_str}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
