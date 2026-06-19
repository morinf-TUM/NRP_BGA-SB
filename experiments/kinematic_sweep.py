"""Phase 6 kinematic sweep runner.

Re-runs a subset of the Phase 5 frequency sweep (5 frequencies × 3 conflict
levels × 2 paradigms × 5 seeds = 150 conditions) with the KinematicReacher
attached.  Prints a report comparing movement_onset_rate to go_success_rate
(the Phase 6 acceptance check) and saves results to JSON.

Run:
    cd /home/fom/code/NRP_BGA-SB
    python experiments/kinematic_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nrp_bga_sb.reacher_sweep import ReacherConditionResult, run_reacher_condition

# --- Sweep parameters ---

FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 160.0]
CONFLICT_LEVELS = ["low", "medium", "high"]
PARADIGMS = ["go_nogo", "two_choice"]
N_SEEDS = 5
N_TRIALS = 30

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    results: list[ReacherConditionResult] = []
    total = len(FREQUENCIES_HZ) * len(CONFLICT_LEVELS) * len(PARADIGMS) * N_SEEDS
    done = 0

    for freq in FREQUENCIES_HZ:
        for conflict in CONFLICT_LEVELS:
            for paradigm in PARADIGMS:
                for seed in range(N_SEEDS):
                    result = run_reacher_condition(
                        frequency_hz=freq,
                        conflict_level=conflict,
                        paradigm=paradigm,
                        n_trials=N_TRIALS,
                        seed=seed,
                    )
                    results.append(result)
                    done += 1
                    if done % 10 == 0:
                        print(f"  {done}/{total} conditions done", flush=True)

    # Save results
    out_path = RESULTS_DIR / "kinematic_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    print(f"\nSaved {len(results)} conditions to {out_path}")

    # --- Acceptance check report ---
    _print_acceptance_report(results)


def _print_acceptance_report(results: list[ReacherConditionResult]) -> None:
    """Print movement_onset_rate vs go_success_rate by frequency (go_nogo only)."""
    print("\n=== Phase 6 Acceptance: movement_onset_rate vs go_success_rate ===")
    print(f"{'Freq (Hz)':<12} {'Conflict':<10} {'go_success':<12} {'onset_rate':<12} {'Match?'}")
    print("-" * 60)

    go_results = [r for r in results if r.paradigm == "go_nogo" and r.seed == 0]
    go_results.sort(key=lambda r: (r.conflict_level, r.frequency_hz))

    for r in go_results:
        if r.go_success_rate is None:
            continue
        match = abs(r.movement_onset_rate - r.go_success_rate) < 0.05
        print(
            f"{r.frequency_hz:<12.0f} {r.conflict_level:<10} "
            f"{r.go_success_rate:<12.3f} {r.movement_onset_rate:<12.3f} "
            f"{'✓' if match else '✗'}"
        )

    print("\n=== Endpoint error by frequency (go_nogo, low conflict, seed=0) ===")
    subset = [
        r for r in results
        if r.paradigm == "go_nogo" and r.conflict_level == "low" and r.seed == 0
    ]
    subset.sort(key=lambda r: r.frequency_hz)
    for r in subset:
        print(f"  {r.frequency_hz:>6.0f} Hz: endpoint_error={r.mean_endpoint_error:.4f}  "
              f"amplitude={r.mean_partial_amplitude:.4f}  peak_vel={r.mean_peak_velocity:.6f}")


if __name__ == "__main__":
    main()
