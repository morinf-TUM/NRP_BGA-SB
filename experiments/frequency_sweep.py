#!/usr/bin/env python3
"""Phase 5 frequency-sweep experiment runner (M4 acceptance).

Run: python experiments/frequency_sweep.py

Sweep: {10, 20, 40, 80, 160} Hz × {low, medium, high} conflict
       × {go_nogo, two_choice} paradigm × 30 seeds per condition.
Total: 5 × 3 × 2 × 30 = 900 conditions.

Acceptance (M4): reproducible frequency-response curves with 95% bootstrap CIs
produced for go/no-go and two-choice on the abstract embodiment.

Output:
  deprecated_toy_prototype_results/frequency_sweep_results.json — list of condition result dicts
  stdout                               — progress + formatted report with CIs
"""

from __future__ import annotations

import json
from pathlib import Path

from nrp_bga_sb.stats import format_sweep_report, reproducibility_check
from nrp_bga_sb.sweep import (
    CONFLICT_PEAK_SALIENCE,
    SweepConditionResult,
    run_condition,
)

# --- Sweep configuration ---

FREQUENCIES_HZ: list[float] = [10.0, 20.0, 40.0, 80.0, 160.0]
CONFLICT_LEVELS: list[str] = ["low", "medium", "high"]
PARADIGMS: list[str] = ["go_nogo", "two_choice"]
N_SEEDS: int = 30
N_TRIALS: int = 30

# Reproducibility check uses a small subset to keep runtime reasonable.
# Trigger: full re-run of 900 conditions is expensive; 2×3×3×2 = 36 pairs is enough
#   to detect mutable state or broken RNG seeding.
# Why: determinism is a hard requirement; any mutable state in policy or engine
#   components would surface here.
# Outcome: PASS → commit results; FAIL → investigate before claiming M4 acceptance.
REPRO_FREQS: list[float] = [10.0, 40.0]
REPRO_SEEDS: list[int] = [0, 1, 2]

RESULTS_DIR = Path(__file__).parent.parent / "deprecated_toy_prototype_results"
RESULTS_PATH = RESULTS_DIR / "frequency_sweep_results.json"


# --- Main sweep ---


def run_sweep() -> list[SweepConditionResult]:
    """Run all 900 conditions and return the full result list."""
    results: list[SweepConditionResult] = []
    total = len(FREQUENCIES_HZ) * len(CONFLICT_LEVELS) * len(PARADIGMS) * N_SEEDS
    done = 0

    for paradigm in PARADIGMS:
        for conflict in CONFLICT_LEVELS:
            for freq in FREQUENCIES_HZ:
                for seed in range(N_SEEDS):
                    result = run_condition(
                        frequency_hz=freq,
                        conflict_level=conflict,  # type: ignore[arg-type]
                        paradigm=paradigm,  # type: ignore[arg-type]
                        n_trials=N_TRIALS,
                        seed=seed,
                    )
                    results.append(result)
                    done += 1
                    if done % 50 == 0:
                        print(f"  {done}/{total} conditions done...")

    return results


# --- Reproducibility check ---


def run_repro_check() -> tuple[list[SweepConditionResult], list[SweepConditionResult]]:
    """Re-run a small subset of conditions twice to verify deterministic output."""
    pass_a: list[SweepConditionResult] = []
    pass_b: list[SweepConditionResult] = []

    for paradigm in PARADIGMS:
        for conflict in CONFLICT_LEVELS:
            for freq in REPRO_FREQS:
                for seed in REPRO_SEEDS:
                    kwargs = dict(
                        frequency_hz=freq,
                        conflict_level=conflict,
                        paradigm=paradigm,
                        n_trials=N_TRIALS,
                        seed=seed,
                    )
                    pass_a.append(run_condition(**kwargs))  # type: ignore[arg-type]
                    pass_b.append(run_condition(**kwargs))  # type: ignore[arg-type]

    return pass_a, pass_b


# --- Persistence ---


def save_results(results: list[SweepConditionResult], path: Path) -> None:
    """Write results as a JSON array of condition dicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- Entry point ---


def main() -> None:
    print("Phase 5 frequency-sweep experiment")
    print(f"  Frequencies: {FREQUENCIES_HZ} Hz")
    print(f"  Conflict levels: {CONFLICT_LEVELS}")
    print(f"  Conflict peak saliences: {CONFLICT_PEAK_SALIENCE}")
    print(f"  Paradigms: {PARADIGMS}")
    print(f"  Seeds per condition: {N_SEEDS}")
    print(f"  Trials per condition: {N_TRIALS}")
    total = len(FREQUENCIES_HZ) * len(CONFLICT_LEVELS) * len(PARADIGMS) * N_SEEDS
    print(f"  Total conditions: {total}")
    print()

    # --- Main sweep ---
    print("Running main sweep...")
    results = run_sweep()
    print(f"  {len(results)} conditions completed.")
    print()

    # --- Reproducibility check ---
    print("Running reproducibility check...")
    pass_a, pass_b = run_repro_check()
    ok = reproducibility_check(pass_a, pass_b)
    repro_status = "PASS" if ok else "FAIL"
    print(f"  Reproducibility check: {repro_status} ({len(pass_a)} condition pairs)")
    if not ok:
        print("  WARNING: results differ between passes — RNG seeding may be broken.")
    print()

    # --- Save JSON ---
    save_results(results, RESULTS_PATH)
    print(f"  Results saved to: {RESULTS_PATH}")
    print()

    # --- Generate and print report ---
    report = format_sweep_report(results, FREQUENCIES_HZ, CONFLICT_LEVELS)
    repro_line = f"\nReproducibility check: {repro_status} ({len(pass_a)} condition pairs)\n"
    full_report = report + repro_line
    print(full_report)


if __name__ == "__main__":
    main()
