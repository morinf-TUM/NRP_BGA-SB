# Task 9.1 Report — Perturbation Sweep Library (M10)

## Status

DONE

## Commit

`b3124c3` — feat: perturbation sweep library and tests (Task 9.1, M10)

## Files created

- `src/nrp_bga_sb/perturbation_sweep.py` — 256 lines
- `tests/test_perturbation_sweep.py` — 280 lines (34 tests)

## Test summary

- New: **34 passed** in 0.66s (all 25 required tests present + 9 additional)
- Full suite: **666 passed, 0 failed** in ~56s (no regressions)
- Ruff: clean

## Implementation notes

### Module structure (section headers as required)

1. Type aliases and constants — all exact values from brief
2. PerturbationSweepResult model — Pydantic BaseModel with go/nogo and stop-signal fields as optional pairs
3. Wrapper factory — `_phase_offset_ms` and `_make_wrapped_policy`; ValueError on unknown type
4. Go/no-go condition runner — `run_gonogo_perturbation_condition`
5. Stop-signal condition runner — `run_stopsignal_perturbation_condition`
6. Decomposition report formatter — `format_decomposition_report`

### Key design decisions

- **Fresh policy per seed**: `_make_wrapped_policy` is called inside the seed loop, not before it. This resets `DropoutWrapper._last_decision` between seeds as required.
- **false_alarm_rate**: computed directly from trial logs (`movement_onset_time is not None` on `cue_identity=="no_go"` trials), consistent with `scorer.py`'s definition (scorer checks for `no_go_cue` event; both detect the same trials in go/no-go engine output).
- **bg_commitment_latency_mean**: `thalamic_relay_time - cue_onset_time` in seconds, numpy mean — same formula as `sweep._compute_phase5_metrics`.
- **Stop-signal metrics**: delegated to `compute_stop_signal_metrics` and `validate_stop_signal_data` exactly as in `stop_signal_sweep.py`. `inhibition_function_monotone` is drawn from the validity report.

### No concerns
All spec values reproduced verbatim; no API was invented or assumed.
