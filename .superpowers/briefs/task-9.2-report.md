# Task 9.2 Report — Perturbation Sweep Experiment Runner + Report Tests

## Status: DONE

## What was implemented

### Part A: `experiments/perturbation_sweep.py`

- Full experiment runner implementing `run_sweep()` and `save_results()`.
- `PERTURBATION_LEVELS` dict maps the 4 perturbation type names to their
  library-defined level lists (`LATENCY_LEVELS_MS`, `JITTER_STD_LEVELS_MS`,
  `DROPOUT_LEVELS`, `PHASE_OFFSET_FRACTIONS`).
- `_TOTAL_CONDITIONS` (170) computed from the dict at module load time, used
  in the progress counter.
- `run_sweep()` iterates `(perturbation_type, frequency_hz, level)` and calls
  both `run_gonogo_perturbation_condition` and `run_stopsignal_perturbation_condition`
  per cell, printing progress in the required format:
  `[{done}/{total}] {paradigm} | {perturbation_type} | {frequency_hz:.0f} Hz | {label}`
- `save_results()` writes `results/perturbation_sweep_gonogo.json` and
  `results/perturbation_sweep_stopsignal.json` as JSON arrays of
  `PerturbationSweepResult.model_dump()`.
- `__main__` block saves both JSON files, writes `results/perturbation_sweep_report.txt`,
  and prints the report, exactly matching the spec skeleton.
- A local `_make_progress_label()` mirrors the library's private `_make_label()`
  for progress output, avoiding dependency on a private function.
- Literate-programming section headers and decision-point comments applied.

### Part B: Appended tests in `tests/test_perturbation_sweep.py`

8 new tests added (all with exact names from the brief):

1. `test_format_report_has_section_per_perturbation_type`
2. `test_format_report_gonogo_section_present`
3. `test_format_report_stop_signal_section_present`
4. `test_format_report_interpretation_guide_present`
5. `test_format_report_na_for_none_values` — uses a synthetic `PerturbationSweepResult`
   with all stop-signal fields set to `None` to exercise the N/A rendering branch
   without a costly condition run.
6. `test_format_report_frequency_appears_in_table`
7. `test_format_report_sorted_by_frequency` — passes results in reversed order to
   verify the formatter sorts, not just preserves input order.
8. `test_perturbation_levels_dict_keys` — calls both runners with each of the 4
   perturbation types at value 0.0 to confirm the library accepts all four strings.

Two private helpers (`_make_gonogo_result`, `_make_stopsignal_result`) reduce
boilerplate across the new tests; they use `n_trials_per_seed=5, n_seeds=2`.

## Verification

- Import smoke test: `from experiments.perturbation_sweep import run_sweep, PERTURBATION_LEVELS` — OK
- `PERTURBATION_LEVELS` keys: `['latency', 'jitter', 'dropout', 'phase_offset']` — OK
- Ruff: `All checks passed!` on both files
- Full test suite: **674 passed** in 54.31 s (previously 530 + 144 new from Task 9.1 + 8 new from Task 9.2 = 674 minus helper counts)

## Commit

`feat: perturbation sweep experiment runner and report tests (Task 9.2, M10)`
ChangeSet-ID: p9t2-sweep-runner
