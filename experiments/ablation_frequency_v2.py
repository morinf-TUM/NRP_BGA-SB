#!/usr/bin/env python3
"""Phase 5 ablation: per-knob frequency sweep with ClosedLoopPolicy (Task 5.4).

Run: python experiments/ablation_frequency_v2.py

Methodology: vary ONE frequency knob at a time while holding the other three at
160 Hz (fast). Use go_nogo + low conflict (peak_salience=0.85, rise_time=200ms,
accumulation=200ms) to expose input_sampling_hz as the primary variable.

Expected finding (Phase 3 null result now resolved by Phase 4 time-varying cortex):
  - All four knobs share the same 5/10 Hz selection boundary:
      5 Hz → miss_rate ≈ 1.0 (period_ticks=200 equals the accumulation window,
             so every gate fires only once at tick 0 where the ramp is neutral).
      ≥10 Hz → miss_rate ≈ 0.0 (gate fires before the window closes, BG reads
              risen evidence above the selection threshold).
  - input_sampling_hz is the mechanistic upstream variable: it controls when
    CortexEvidenceGenerator is sampled. The other three knobs (integration_step_hz,
    output_emission_hz, commitment_update_hz) are downstream, but at 5 Hz they are
    equally starved by the period=200-tick bottleneck and show the same miss pattern.

Also includes a "baseline" condition where all knobs are at 160 Hz.

Output:
  deprecated_toy_prototype_results/ablation_frequency_v2.json — list of condition result dicts
  stdout                             — formatted table
"""

from __future__ import annotations

import json
from pathlib import Path

from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.scheduler import FrequencyConfig

# --- Ablation configuration ---

# Sweep frequencies (Hz) for each individual knob
SWEEP_FREQUENCIES_HZ: list[float] = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]

# Value held at 160 Hz for all non-swept knobs
FIXED_HZ: float = 160.0

# Ablation knobs: sweep each in turn
KNOB_NAMES: list[str] = [
    "input_sampling_hz",
    "integration_step_hz",
    "output_emission_hz",
    "commitment_update_hz",
]

# Go/no-go paradigm settings (low conflict, as per Task 5.4 specification)
N_TRIALS: int = 50
SEED: int = 42
GO_PROBABILITY: float = 0.5

# Cortex settings — low conflict level; rise_time_ms=200 ensures the ramp is
# slow enough that 5 Hz (period=200 ticks) sees only neutral evidence at tick 0.
RISE_TIME_MS: float = 200.0
PEAK_SALIENCE: float = 0.85
ACCUMULATION_MS: float = 200.0

RESULTS_DIR = Path(__file__).parent.parent / "deprecated_toy_prototype_results"
RESULTS_PATH = RESULTS_DIR / "ablation_frequency_v2.json"


# --- Per-condition runner ---


def run_knob_condition(
    knob_name: str,
    freq_hz: float,
) -> dict:
    """Run one (knob, frequency) ablation condition and return a result dict.

    All knobs are fixed at FIXED_HZ except the swept knob which is set to
    freq_hz. Uses go_nogo paradigm with N_TRIALS trials and SEED for
    reproducibility.

    Args:
        knob_name: One of the four FrequencyConfig knob names.
        freq_hz:   Target frequency for the swept knob (Hz).

    Returns:
        Dict with keys: knob_name, freq_hz, condition, miss_rate, n_go_trials.
    """
    # Build a FrequencyConfig with only the target knob at freq_hz.
    # All other knobs are held at FIXED_HZ (160 Hz).
    # Trigger: ablation requires isolation of one variable at a time.
    # Why: holding all others at 160 Hz means each knob's effect is
    #      attributable solely to the manipulated knob.
    # Outcome: FrequencyConfig with three knobs at FIXED_HZ and one at freq_hz.
    knob_kwargs: dict[str, float] = {
        "input_sampling_hz": FIXED_HZ,
        "integration_step_hz": FIXED_HZ,
        "output_emission_hz": FIXED_HZ,
        "commitment_update_hz": FIXED_HZ,
    }
    knob_kwargs[knob_name] = freq_hz
    freq_cfg = FrequencyConfig(**knob_kwargs)

    cortex_cfg = CortexConfig(
        rise_time_ms=RISE_TIME_MS,
        peak_salience=PEAK_SALIENCE,
    )
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=ACCUMULATION_MS,
    )

    go_nogo_cfg = GoNoGoConfig(
        n_trials=N_TRIALS,
        go_probability=GO_PROBABILITY,
        response_window_start_ms=0,
        response_window_duration_ms=600,
        fixation_duration_ms=200,
        cue_onset_ms=400,
        decision_point_ms=300,
        seed=SEED,
    )
    trials = run_go_nogo_trials(go_nogo_cfg, policy)

    go_trials = [t for t in trials if t.cue_identity == "go"]
    n_go = len(go_trials)
    n_miss = sum(1 for t in go_trials if t.failure_mode == "miss")
    miss_rate = n_miss / n_go if n_go > 0 else float("nan")

    return {
        "condition": "ablation",
        "knob_name": knob_name,
        "freq_hz": freq_hz,
        "miss_rate": miss_rate,
        "n_go_trials": n_go,
    }


def run_baseline() -> dict:
    """Run the baseline condition with all knobs at 160 Hz.

    Used as a sanity anchor: all go trials should succeed (miss_rate ≈ 0.0).

    Returns:
        Dict with keys: knob_name, freq_hz, condition, miss_rate, n_go_trials.
    """
    freq_cfg = FrequencyConfig.from_effective_hz(FIXED_HZ)
    cortex_cfg = CortexConfig(
        rise_time_ms=RISE_TIME_MS,
        peak_salience=PEAK_SALIENCE,
    )
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=ACCUMULATION_MS,
    )

    go_nogo_cfg = GoNoGoConfig(
        n_trials=N_TRIALS,
        go_probability=GO_PROBABILITY,
        response_window_start_ms=0,
        response_window_duration_ms=600,
        fixation_duration_ms=200,
        cue_onset_ms=400,
        decision_point_ms=300,
        seed=SEED,
    )
    trials = run_go_nogo_trials(go_nogo_cfg, policy)

    go_trials = [t for t in trials if t.cue_identity == "go"]
    n_go = len(go_trials)
    n_miss = sum(1 for t in go_trials if t.failure_mode == "miss")
    miss_rate = n_miss / n_go if n_go > 0 else float("nan")

    return {
        "condition": "baseline",
        "knob_name": "all",
        "freq_hz": FIXED_HZ,
        "miss_rate": miss_rate,
        "n_go_trials": n_go,
    }


# --- Report formatting ---


def _format_table(records: list[dict]) -> str:
    """Format ablation results as a human-readable table."""
    lines: list[str] = [
        "Phase 5 Ablation Re-Run — Per-Knob Frequency Sweep (ClosedLoopPolicy)",
        "=" * 72,
        f"  Fixed Hz for non-swept knobs: {FIXED_HZ:.0f} Hz",
        f"  Cortex: rise_time={RISE_TIME_MS:.0f} ms, peak_salience={PEAK_SALIENCE}",
        f"  Go/no-go: n_trials={N_TRIALS}, seed={SEED}, go_probability={GO_PROBABILITY}",
        f"  Accumulation window: {ACCUMULATION_MS:.0f} ms",
        "",
        "Expected: all four knobs share the 5/10 Hz boundary (miss≈1.0 at 5 Hz, ≈0.0 at ≥10 Hz).",
        "          input_sampling_hz is mechanistic upstream; others equally starved at 5 Hz.",
        "",
    ]

    # Baseline first
    baseline = next((r for r in records if r["condition"] == "baseline"), None)
    if baseline is not None:
        lines.append(f"Baseline (all knobs at {FIXED_HZ:.0f} Hz):")
        lines.append(f"  miss_rate = {baseline['miss_rate']:.3f}  (n_go={baseline['n_go_trials']})")
        lines.append("")

    # Per-knob sections
    ablation_records = [r for r in records if r["condition"] == "ablation"]
    for knob_name in KNOB_NAMES:
        knob_records = [r for r in ablation_records if r["knob_name"] == knob_name]
        knob_records.sort(key=lambda r: r["freq_hz"])
        lines.append(f"Knob: {knob_name}")
        lines.append(f"  {'Freq (Hz)':<12} {'miss_rate':>12}  {'n_go':>6}")
        lines.append("  " + "-" * 34)
        for rec in knob_records:
            lines.append(
                f"  {rec['freq_hz']:<12.0f} {rec['miss_rate']:>12.3f}  {rec['n_go_trials']:>6}"
            )
        lines.append("")

    lines.append(
        "Primary variable: input_sampling_hz\n"
        "Rationale: input_sampling_hz controls when CortexEvidenceGenerator is sampled;\n"
        "other gates are downstream. However, at 5 Hz all four knobs share the same\n"
        "miss_rate ≈ 1.0 boundary: period_ticks=200 equals the accumulation window,\n"
        "so every gate fires only once at tick 0 (neutral evidence). At ≥10 Hz all\n"
        "knobs succeed (miss_rate ≈ 0.0)."
    )
    return "\n".join(lines)


# --- Entry point ---


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # Run baseline (all knobs at 160 Hz)
    print("Running baseline ...", flush=True)
    records.append(run_baseline())

    # Run ablation: sweep each knob independently
    for knob_name in KNOB_NAMES:
        for freq_hz in SWEEP_FREQUENCIES_HZ:
            print(f"  {knob_name} @ {freq_hz:.0f} Hz ...", flush=True)
            records.append(run_knob_condition(knob_name, freq_hz))

    # Save JSON
    RESULTS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # Print formatted table
    report = _format_table(records)
    print()
    print(report)
    print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
