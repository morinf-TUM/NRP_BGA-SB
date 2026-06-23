#!/usr/bin/env python3
"""Change-of-mind BG-frequency sweep experiment (Task 8.3, Milestone M6).

Runs the change-of-mind paradigm across BG frequencies, saves JSON results,
and prints a formatted report.

5 frequencies × 5 seeds × 80 trials = 400 switch trials per condition (M6 ≥300).
"""

from __future__ import annotations

import json
import pathlib

from nrp_bga_sb.change_of_mind_sweep import (
    FREQUENCIES_HZ,
    ChangeOfMindSweepResult,
    format_sweep_report,
    run_change_of_mind_condition,
)


def main() -> None:
    results: list[ChangeOfMindSweepResult] = []
    for freq_hz in FREQUENCIES_HZ:
        print(f"Running {freq_hz:.0f} Hz ...", flush=True)
        result = run_change_of_mind_condition(freq_hz)
        results.append(result)

    out_path = pathlib.Path("deprecated_toy_prototype_results/change_of_mind_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(format_sweep_report(results))


if __name__ == "__main__":
    main()
