#!/usr/bin/env python3
"""Phase 3 ablation: sweep each BG frequency variable independently.

Run: python experiments/ablation_frequency.py
Output: data/phase3_ablation_report.txt + data/phase3_ablation_results.jsonl

For each of the four frequency knobs (input_sampling_hz, integration_step_hz,
output_emission_hz, commitment_update_hz), this script sweeps five frequency
values [10, 20, 40, 80, 160 Hz] across all four task engines and records the
resulting Metrics.

Expected finding: metrics are identical across all frequency conditions because
in the abstract single-call constant-evidence model, tick 0 satisfies the firing
condition for every gate regardless of configured Hz.  output_emission_hz is
assigned as the primary variable on theoretical grounds (nrp-core EngineTimestep
binding, §15.4 of PROJECT_MEMORY).
"""

from __future__ import annotations

import json
from pathlib import Path

from nrp_bga_sb.ablation import (
    ENGINE_RUNNERS,
    run_full_ablation,
    summarize_ablation,
)

# --- Output paths ---

DATA_DIR = Path(__file__).parent.parent / "data"
REPORT_PATH = DATA_DIR / "phase3_ablation_report.txt"
RESULTS_PATH = DATA_DIR / "phase3_ablation_results.jsonl"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Phase 3 frequency ablation sweep...")
    print(f"  Engines: {list(ENGINE_RUNNERS.keys())}")
    print("  Frequencies: [10, 20, 40, 80, 160] Hz")
    print("  Trials per condition: 20")
    print("  Seed: 42")
    print()

    # --- Run all conditions ---
    results = run_full_ablation(
        engine_names=list(ENGINE_RUNNERS.keys()),
        n_trials=20,
        seed=42,
    )

    # --- Generate and write the text report ---
    report = summarize_ablation(results)

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")

    # --- Write JSONL summary (one line per knob × engine × freq condition) ---
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        for knob_name, engine_map in results.items():
            for engine_name, freq_map in engine_map.items():
                for freq_hz, metrics in freq_map.items():
                    record = {
                        "knob_name": knob_name,
                        "engine_name": engine_name,
                        "freq_hz": freq_hz,
                        "n_trials": metrics.n_trials,
                        "condition_id": metrics.condition_id,
                        "bg_frequency_hz": metrics.bg_frequency_hz,
                        "reaction_time_mean": metrics.reaction_time_mean,
                        "reaction_time_std": metrics.reaction_time_std,
                        "wrong_action_rate": metrics.wrong_action_rate,
                        "wrong_target_rate": metrics.wrong_target_rate,
                        "false_alarm_rate": metrics.false_alarm_rate,
                        "stop_success_rate": metrics.stop_success_rate,
                        "switch_success_rate": metrics.switch_success_rate,
                    }
                    fh.write(json.dumps(record) + "\n")

    print(f"JSONL results written to: {RESULTS_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()
