#!/usr/bin/env python3
"""Stop-signal BG-frequency sweep experiment (Phase 7).

Runs 5 BG frequencies × 5 seeds × 100 trials = 500 trials per condition.
Outputs results to deprecated_toy_prototype_results/stop_signal_sweep_results.json and prints a report.
"""

from __future__ import annotations

import json
import pathlib

from nrp_bga_sb.stop_signal_sweep import (
    FREQUENCIES_HZ,
    StopSignalSweepResult,
    format_sweep_report,
    run_stop_signal_condition,
)


def main() -> None:
    results: list[StopSignalSweepResult] = []
    for freq_hz in FREQUENCIES_HZ:
        print(f"Running {freq_hz:.0f} Hz ...", flush=True)
        result = run_stop_signal_condition(freq_hz)
        results.append(result)

    out_path = pathlib.Path("deprecated_toy_prototype_results/stop_signal_sweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(format_sweep_report(results))


if __name__ == "__main__":
    main()
