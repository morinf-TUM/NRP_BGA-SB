"""Phase 2 BG-alone channel validation (Milestone M2).

Exercises BGAdapter in isolation: sweeps three salience conditions and confirms
(1) the dominant channel is reliably selected at low and medium conflict, and
(2) selection latency increases monotonically with conflict (M2 criterion).

Run:
    python experiments/bg_validation.py
Output:
    results/bg_validation.json — list of 3 dicts, one per conflict level
"""
from __future__ import annotations

import json
from pathlib import Path

from nrp_bga_sb.bg_model import BGAdapter
from nrp_bga_sb.schemas import ActionEvidence, TrialLog

# --- Salience conditions ---
# Three levels spanning the selection boundary:
#   low      (gap=0.70): reliably selects channel 0 (T_winner > threshold)
#   medium   (gap=0.30): marginally selects channel 0
#   high     (gap=0.10): gap too small for GPR winner → no selection (selected_channel=-1)
_CONDITIONS = [
    {"conflict_level": "low",    "saliences": [0.85, 0.15]},
    {"conflict_level": "medium", "saliences": [0.65, 0.35]},
    {"conflict_level": "high",   "saliences": [0.55, 0.45]},
]

# Deterministic BG (noise_std=0.0 default): one seed suffices for correctness,
# but we run 5 to document the invariant explicitly.
_N_SEEDS = 5


def run_bg_validation() -> list[dict]:
    """Run BG-alone validation across three conflict levels.

    Returns a list of result dicts with keys:
        conflict_level, salience_gap, n_seeds, n_selections,
        selection_accuracy, mean_selection_latency_ms
    """
    adapter = BGAdapter()
    results = []
    for cond in _CONDITIONS:
        latencies_ms: list[float] = []
        correct = 0
        n_sel = 0
        for seed in range(_N_SEEDS):
            trial_log = TrialLog(
                trial_id=seed,
                seed=seed,
                task_type="go_nogo",
                cue_identity="go",
                cue_onset_time=0.0,
            )
            action_evidence = ActionEvidence(
                sim_time=0.0,
                trial_id=seed,
                n_channels=2,
                channel_salience=cond["saliences"],
            )
            decision = adapter(trial_log, action_evidence)
            if decision.selected_channel >= 0:
                n_sel += 1
                latencies_ms.append(decision.selection_latency * 1000.0)
                if decision.selected_channel == 0:  # channel 0 is dominant
                    correct += 1
        results.append({
            "conflict_level": cond["conflict_level"],
            "salience_gap": round(cond["saliences"][0] - cond["saliences"][1], 2),
            "n_seeds": _N_SEEDS,
            "n_selections": n_sel,
            "selection_accuracy": correct / n_sel if n_sel > 0 else 0.0,
            "mean_selection_latency_ms": (
                round(sum(latencies_ms) / len(latencies_ms), 1)
                if latencies_ms
                else None
            ),
        })
    return results


if __name__ == "__main__":
    results = run_bg_validation()

    out_path = Path("results/bg_validation.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved → {out_path}\n")

    print(f"{'Conflict':<10} {'Gap':>6} {'Selections':>12} {'Accuracy':>10} {'Latency (ms)':>14}")
    print("-" * 56)
    for r in results:
        lat = f"{r['mean_selection_latency_ms']:.1f}" if r["mean_selection_latency_ms"] else "—"
        print(
            f"{r['conflict_level']:<10} {r['salience_gap']:>6.2f}"
            f" {r['n_selections']:>12}/{r['n_seeds']}"
            f" {r['selection_accuracy']:>10.2f} {lat:>14}"
        )
    print()
    # Monotone check: latency at low < medium (high is suppressed → no latency value)
    low_lat = next(
        r["mean_selection_latency_ms"]
        for r in results
        if r["conflict_level"] == "low"
    )
    med_lat = next(
        r["mean_selection_latency_ms"]
        for r in results
        if r["conflict_level"] == "medium"
    )
    high_sel = next(
        r["n_selections"] for r in results if r["conflict_level"] == "high"
    )
    monotone_status = "PASS" if low_lat < med_lat else "FAIL"
    print(f"M2 monotone check: low_lat={low_lat:.1f}ms < "
          f"med_lat={med_lat:.1f}ms → {monotone_status}")
    suppression_status = "PASS" if high_sel == 0 else "FAIL"
    print(f"M2 suppression check: high_conflict n_selections={high_sel} → "
          f"{suppression_status}")
