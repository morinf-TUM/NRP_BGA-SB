"""Perturbation sweep experiment: latency/jitter/dropout/phase decomposition (Phase 9, M10)."""

from __future__ import annotations

import json
from pathlib import Path

from nrp_bga_sb.perturbation_sweep import (
    DROPOUT_LEVELS,
    FREQUENCIES_HZ,
    JITTER_STD_LEVELS_MS,
    LATENCY_LEVELS_MS,
    PHASE_OFFSET_FRACTIONS,
    PerturbationSweepResult,
    format_decomposition_report,
    run_gonogo_perturbation_condition,
    run_stopsignal_perturbation_condition,
)

# --- Sweep configuration ---

# Map each perturbation type name to its ordered level list.
# The order here determines the iteration order in run_sweep().
PERTURBATION_LEVELS: dict[str, list[float]] = {
    "latency": LATENCY_LEVELS_MS,
    "jitter": JITTER_STD_LEVELS_MS,
    "dropout": DROPOUT_LEVELS,
    "phase_offset": PHASE_OFFSET_FRACTIONS,
}

# Total condition count:
#   latency:      5 levels × 5 frequencies × 2 paradigms =  50
#   jitter:       4 levels × 5 frequencies × 2 paradigms =  40
#   dropout:      4 levels × 5 frequencies × 2 paradigms =  40
#   phase_offset: 4 levels × 5 frequencies × 2 paradigms =  40
#                                                   TOTAL = 170
_TOTAL_CONDITIONS: int = sum(
    len(levels) * len(FREQUENCIES_HZ) * 2
    for levels in PERTURBATION_LEVELS.values()
)


# --- Sweep runner ---


def run_sweep() -> tuple[list[PerturbationSweepResult], list[PerturbationSweepResult]]:
    """Run all 170 conditions; return (gonogo_results, stopsignal_results).

    Iterates over every (perturbation_type, frequency_hz, level) triple and
    runs both paradigms.  Progress is printed to stdout after each condition
    so the caller can monitor a long-running sweep.
    """
    gonogo_results: list[PerturbationSweepResult] = []
    stopsignal_results: list[PerturbationSweepResult] = []
    done = 0

    for perturbation_type, levels in PERTURBATION_LEVELS.items():
        for frequency_hz in FREQUENCIES_HZ:
            for level in levels:
                label = _make_progress_label(perturbation_type, level)

                # --- Go/no-go condition ---
                gonogo_result = run_gonogo_perturbation_condition(
                    frequency_hz=frequency_hz,
                    perturbation_type=perturbation_type,  # type: ignore[arg-type]
                    perturbation_value=level,
                )
                gonogo_results.append(gonogo_result)
                done += 1
                print(
                    f"[{done}/{_TOTAL_CONDITIONS}] go_nogo"
                    f" | {perturbation_type}"
                    f" | {frequency_hz:.0f} Hz"
                    f" | {label}",
                    flush=True,
                )

                # --- Stop-signal condition ---
                stopsignal_result = run_stopsignal_perturbation_condition(
                    frequency_hz=frequency_hz,
                    perturbation_type=perturbation_type,  # type: ignore[arg-type]
                    perturbation_value=level,
                )
                stopsignal_results.append(stopsignal_result)
                done += 1
                print(
                    f"[{done}/{_TOTAL_CONDITIONS}] stop_signal"
                    f" | {perturbation_type}"
                    f" | {frequency_hz:.0f} Hz"
                    f" | {label}",
                    flush=True,
                )

    return gonogo_results, stopsignal_results


def _make_progress_label(perturbation_type: str, level: float) -> str:
    """Return a compact human-readable label for progress output.

    Mirrors the format used by _make_label() in the library, kept local so
    the experiment script does not depend on a private library function.
    """
    if perturbation_type == "latency":
        return f"latency={level:.0f}ms"
    elif perturbation_type == "jitter":
        return f"jitter_std={level:.0f}ms"
    elif perturbation_type == "dropout":
        return f"dropout={level * 100:.0f}%"
    elif perturbation_type == "phase_offset":
        return f"phase_offset={level * 100:.0f}%"
    else:
        # Fail fast: unknown type should never reach here at runtime.
        raise ValueError(f"unknown perturbation_type: {perturbation_type!r}")


# --- Persistence ---


def save_results(
    gonogo_results: list[PerturbationSweepResult],
    stopsignal_results: list[PerturbationSweepResult],
    results_dir: Path,
) -> None:
    """Write gonogo and stopsignal result arrays to separate JSON files.

    Each file is a JSON array of PerturbationSweepResult.model_dump() dicts,
    matching the PerturbationSweepResult schema exactly for downstream reload.
    """
    gonogo_path = results_dir / "perturbation_sweep_gonogo.json"
    stopsignal_path = results_dir / "perturbation_sweep_stopsignal.json"

    gonogo_path.write_text(
        json.dumps([r.model_dump() for r in gonogo_results], indent=2),
        encoding="utf-8",
    )
    stopsignal_path.write_text(
        json.dumps([r.model_dump() for r in stopsignal_results], indent=2),
        encoding="utf-8",
    )
    print(f"Saved go/no-go results    → {gonogo_path}")
    print(f"Saved stop-signal results → {stopsignal_path}")


# --- Entry point ---

if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    gonogo_results, stopsignal_results = run_sweep()
    save_results(gonogo_results, stopsignal_results, results_dir)
    report = format_decomposition_report(gonogo_results, stopsignal_results)
    (results_dir / "perturbation_sweep_report.txt").write_text(report)
    print(report)
