# Task 9.2 — Perturbation Sweep Experiment Runner + Report Tests

## Context

Phase 9 (M10) — latency/jitter/dropout/phase decomposition. Phase 9 Task 9.1
is complete (commits fd41394..714a1f7). The library module
`src/nrp_bga_sb/perturbation_sweep.py` is implemented with:
- `PerturbationSweepResult` model
- `LATENCY_LEVELS_MS`, `JITTER_STD_LEVELS_MS`, `DROPOUT_LEVELS`, `PHASE_OFFSET_FRACTIONS`, `FREQUENCIES_HZ`
- `run_gonogo_perturbation_condition(frequency_hz, perturbation_type, perturbation_value, ...)` → `PerturbationSweepResult`
- `run_stopsignal_perturbation_condition(frequency_hz, perturbation_type, perturbation_value, ...)` → `PerturbationSweepResult`
- `format_decomposition_report(gonogo_results, stopsignal_results)` → `str`

You are implementing:
1. `experiments/perturbation_sweep.py` — full experiment runner script
2. Additional tests in `tests/test_perturbation_sweep.py` covering the report formatter

---

## Specification

### Part A: `experiments/perturbation_sweep.py`

This is a standalone script (not a library). It should be runnable as:
```
python experiments/perturbation_sweep.py
```

It runs the full cross-product sweep and saves results + report.

#### Full sweep design

For **each** perturbation type in `["latency", "jitter", "dropout", "phase_offset"]`:
  For **each** frequency in `FREQUENCIES_HZ` ([5, 10, 20, 40, 80] Hz):
    For **each** level in the corresponding level list:
      - Run `run_gonogo_perturbation_condition(frequency_hz, perturbation_type, level)`
      - Run `run_stopsignal_perturbation_condition(frequency_hz, perturbation_type, level)`

Total conditions:
- latency: 5 levels × 5 frequencies × 2 paradigms = 50
- jitter: 4 levels × 5 frequencies × 2 paradigms = 40
- dropout: 4 levels × 5 frequencies × 2 paradigms = 40
- phase_offset: 4 levels × 5 frequencies × 2 paradigms = 40
= 170 condition runs total (each run is N_SEEDS × N_TRIALS_PER_SEED trials internally)

#### Progress reporting

Print to stdout as each condition completes:
```
[{done}/{total}] {paradigm} | {perturbation_type} | {frequency_hz:.0f} Hz | {label}
```

#### Output files

All output goes to `results/` directory (create if missing).

1. `results/perturbation_sweep_gonogo.json` — JSON array of all go/no-go `PerturbationSweepResult.model_dump()` entries
2. `results/perturbation_sweep_stopsignal.json` — JSON array of all stop-signal results
3. `results/perturbation_sweep_report.txt` — the formatted decomposition report

After saving, print the report to stdout.

#### Script structure

```python
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

PERTURBATION_LEVELS: dict[str, list[float]] = {
    "latency": LATENCY_LEVELS_MS,
    "jitter": JITTER_STD_LEVELS_MS,
    "dropout": DROPOUT_LEVELS,
    "phase_offset": PHASE_OFFSET_FRACTIONS,
}


def run_sweep() -> tuple[list[PerturbationSweepResult], list[PerturbationSweepResult]]:
    """Run all conditions; return (gonogo_results, stopsignal_results)."""
    ...


def save_results(
    gonogo_results: list[PerturbationSweepResult],
    stopsignal_results: list[PerturbationSweepResult],
    results_dir: Path,
) -> None:
    ...


if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    gonogo_results, stopsignal_results = run_sweep()
    save_results(gonogo_results, stopsignal_results, results_dir)
    report = format_decomposition_report(gonogo_results, stopsignal_results)
    (results_dir / "perturbation_sweep_report.txt").write_text(report)
    print(report)
```

Follow the literate-programming style (section-header comments, decision-point comments where relevant).

---

### Part B: Additional tests in `tests/test_perturbation_sweep.py`

Add ≥8 tests for the report formatter and experiment logic. Append to the existing file (do not create a new file).

Required new tests (exact names):
1. `test_format_report_has_section_per_perturbation_type` — report has all 4 perturbation type section headers: "latency", "jitter", "dropout", "phase_offset"
2. `test_format_report_gonogo_section_present` — report contains "Go/No-Go"
3. `test_format_report_stop_signal_section_present` — report contains "Stop-Signal"
4. `test_format_report_interpretation_guide_present` — report contains "INTERPRETATION GUIDE"
5. `test_format_report_na_for_none_values` — a result with None stop_failure_rate shows "N/A" in report
6. `test_format_report_frequency_appears_in_table` — report contains "20" (from a 20 Hz result)
7. `test_format_report_sorted_by_frequency` — within a section, lower frequency appears before higher
8. `test_perturbation_levels_dict_keys` — PERTURBATION_LEVELS dict (if importable from experiments script) has exactly 4 keys; OR test that calling run_gonogo_perturbation_condition with each of the 4 perturbation types does not raise

For the experiment script's `PERTURBATION_LEVELS` dict: since it lives in an `experiments/` script (not a package), import it via importlib or just test the library functions instead. Use the library approach for tests (option 2 above for test #8).

Use small n_trials_per_seed=5, n_seeds=2 for speed.

---

## Commit style

One commit:
```
feat: perturbation sweep experiment runner and report tests (Task 9.2, M10)

ChangeSet-ID: p9t2-sweep-runner
```

---

## Report contract

Write your full report to `/home/fom/code/NRP_BGA-SB/.superpowers/briefs/task-9.2-report.md`.

Return to me only:
- STATUS: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED
- Commit hash(es)
- One-line test summary (count, pass/fail)
- Any concerns
