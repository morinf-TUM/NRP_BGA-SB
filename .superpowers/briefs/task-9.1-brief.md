# Task 9.1 — Perturbation Sweep Library

## Context

Phase 9 (M10) of NRP_BGA-SB decomposes the effects of latency, jitter, dropout,
and phase offset from BG update frequency on action selection and stop-signal
behaviour.  Phases 0–8 are complete (632 tests, master HEAD fd41394).

You are implementing the **library module** for Phase 9:
`src/nrp_bga_sb/perturbation_sweep.py` and `tests/test_perturbation_sweep.py`.

---

## Existing infrastructure to use (do not rewrite)

- `src/nrp_bga_sb/perturbations.py` — four wrapper classes:
  - `LatencyWrapper(base_policy, latency_ms)` — adds fixed ms to selection_latency
  - `JitterWrapper(base_policy, jitter_std_ms)` — adds Normal(0, std) to selection_latency
  - `DropoutWrapper(base_policy, dropout_probability)` — replays last decision with prob p
  - `PhaseOffsetWrapper(base_policy, phase_offset_ms)` — additive latency shift (phase proxy)
  All wrappers implement `__call__(trial_log, action_evidence) -> BGDecision`.
  
- `src/nrp_bga_sb/closed_loop.py` — `make_closed_loop_policy(...)` factory:
  ```python
  make_closed_loop_policy(
      cortex_config: CortexConfig | None = None,
      frequency_config: FrequencyConfig | None = None,
      accumulation_ms: float = 200.0,
  ) -> ClosedLoopPolicy  # implements (TrialLog, ActionEvidence) -> BGDecision
  ```

- `src/nrp_bga_sb/sweep.py` — `run_condition`, `CONFLICT_PEAK_SALIENCE`, `_run_engine`:
  ```python
  CONFLICT_PEAK_SALIENCE = {"low": 0.85, "medium": 0.69, "high": 0.62}
  def _run_engine(paradigm, n_trials, seed, policy) -> list[TrialLog]  # go_nogo or two_choice
  ```
  Use these constants and the engine-dispatch pattern; do not duplicate them.

- `src/nrp_bga_sb/stop_signal_sweep.py` — stop-signal condition runner pattern:
  ```python
  # runs n_seeds * n_trials_per_seed trials, aggregates into StopSignalSweepResult
  # see: FREQUENCIES_HZ, N_SEEDS, N_TRIALS_PER_SEED, STOP_PROPORTION, etc.
  ```

- `src/nrp_bga_sb/stop_signal_metrics.py` — `compute_stop_signal_metrics`, `validate_stop_signal_data`

- `src/nrp_bga_sb/scheduler.py` — `FrequencyConfig.from_effective_hz(hz)`

- `src/nrp_bga_sb/cortex.py` — `CortexConfig(peak_salience, rise_time_ms, noise_std)`

- `src/nrp_bga_sb/scorer.py` — `score_trials(trials, condition_id, bg_frequency_hz) -> Metrics`

---

## Specification

### File to create: `src/nrp_bga_sb/perturbation_sweep.py`

#### Module docstring
One paragraph explaining purpose: sweeps timing perturbations (latency, jitter,
dropout, phase offset) against BG update frequency on go/no-go and stop-signal;
produces results for the §11 decomposition report.

#### Section structure (use `# --- Section ---` headers):
1. Type aliases and constants
2. PerturbationSweepResult model
3. Wrapper factory
4. Go/no-go condition runner
5. Stop-signal condition runner
6. Decomposition report formatter

---

#### Constants (exact names and values)

```python
LATENCY_LEVELS_MS: list[float] = [0.0, 10.0, 25.0, 50.0, 100.0]
JITTER_STD_LEVELS_MS: list[float] = [0.0, 5.0, 10.0, 25.0]
DROPOUT_LEVELS: list[float] = [0.0, 0.01, 0.05, 0.10]
PHASE_OFFSET_FRACTIONS: list[float] = [0.0, 0.25, 0.50, 0.75]
FREQUENCIES_HZ: list[float] = [5.0, 10.0, 20.0, 40.0, 80.0]

# Go/no-go sweep parameters
N_GONOGO_SEEDS: int = 5
N_GONOGO_TRIALS_PER_SEED: int = 50   # 250 trials per condition

# Stop-signal sweep parameters (match Phase 7 stop_signal_sweep.py constants)
N_SS_SEEDS: int = 5
N_SS_TRIALS_PER_SEED: int = 100      # 500 trials per condition
SS_STOP_PROPORTION: float = 0.25
SS_INITIAL_SSD_MS: int = 200
SS_SSD_STEP_MS: int = 50
SS_SSD_MIN_MS: int = 50
SS_SSD_MAX_MS: int = 450
SS_DECISION_POINT_MS: int = 500
SS_GO_CUE_ONSET_MS: int = 300

# Shared timing parameters
PEAK_SALIENCE: float = 0.85   # low conflict — go process active at >=10 Hz
RISE_TIME_MS: float = 200.0
ACCUMULATION_MS: float = 200.0
```

#### `PerturbationType` literal type

```python
PerturbationType = Literal["latency", "jitter", "dropout", "phase_offset"]
```

#### `PerturbationSweepResult` (Pydantic BaseModel)

Fields:
```python
frequency_hz: float
perturbation_type: PerturbationType
perturbation_value: float       # raw level value (ms for latency/jitter/phase_offset fraction, fraction for dropout)
perturbation_label: str         # human-readable e.g. "latency=25ms", "dropout=5%", "phase_offset=25%"
paradigm: Literal["go_nogo", "stop_signal"]
n_trials: int
n_seeds: int
# go_nogo metrics (None when paradigm == "stop_signal")
go_success_rate: float | None = None
false_alarm_rate: float | None = None
bg_commitment_latency_mean: float | None = None
# stop_signal metrics (None when paradigm == "go_nogo")
stop_failure_rate: float | None = None
ssrt_estimate_s: float | None = None
go_rt_mean_s: float | None = None
inhibition_function_monotone: bool | None = None
```

No `seed` field — conditions aggregate across seeds internally.

#### `_phase_offset_ms(fraction, frequency_hz) -> float`

Private helper. Converts phase-offset fraction [0,1] to milliseconds:
```python
period_ms = 1000.0 / frequency_hz
return fraction * period_ms
```

#### `_make_wrapped_policy(perturbation_type, perturbation_value, frequency_hz, cortex_cfg, freq_cfg) -> Callable`

Private factory. Builds a `ClosedLoopPolicy` (via `make_closed_loop_policy`) and wraps it:
- `"latency"` → `LatencyWrapper(base, perturbation_value)`
- `"jitter"` → `JitterWrapper(base, perturbation_value)`
- `"dropout"` → `DropoutWrapper(base, perturbation_value)`
- `"phase_offset"` → `PhaseOffsetWrapper(base, _phase_offset_ms(perturbation_value, frequency_hz))`

Creates a fresh base policy for each call (no shared state between conditions).
Raises `ValueError` for unknown perturbation_type.

#### `run_gonogo_perturbation_condition(frequency_hz, perturbation_type, perturbation_value, n_trials_per_seed=N_GONOGO_TRIALS_PER_SEED, n_seeds=N_GONOGO_SEEDS, base_seed=42) -> PerturbationSweepResult`

1. Build cortex/freq configs (PEAK_SALIENCE, RISE_TIME_MS, noise_std=0.0, from_effective_hz).
2. For each seed in range(n_seeds): build a fresh wrapped policy (new DropoutWrapper per seed to reset inter-call state), run GoNoGoConfig with seed=base_seed+seed_index, collect trials.
3. Aggregate all trials; compute:
   - go_success_rate: fraction of go trials (cue_identity=="go") with success==True
   - false_alarm_rate: fraction of no-go trials (cue_identity=="no_go") with success==False and failure_mode!="miss" (i.e. responded when shouldn't); use scorer.false_alarm_rate if available, but compute directly for accuracy
   - bg_commitment_latency_mean: mean(thalamic_relay_time - cue_onset_time) over trials where thalamic_relay_time is not None; in seconds
4. Generate perturbation_label (see label format below).
5. Return PerturbationSweepResult with paradigm="go_nogo".

**GoNoGoConfig parameters** (match sweep.py `_run_engine` go_nogo values exactly):
```python
GoNoGoConfig(
    n_trials=n_trials_per_seed,
    go_probability=0.7,
    response_window_start_ms=0,
    response_window_duration_ms=600,
    fixation_duration_ms=200,
    cue_onset_ms=400,
    decision_point_ms=300,
    seed=base_seed + seed_index,
)
```

**Label format:**
- latency: `f"latency={perturbation_value:.0f}ms"`
- jitter: `f"jitter_std={perturbation_value:.0f}ms"`
- dropout: `f"dropout={perturbation_value*100:.0f}%"`
- phase_offset: `f"phase_offset={perturbation_value*100:.0f}%"`

#### `run_stopsignal_perturbation_condition(frequency_hz, perturbation_type, perturbation_value, n_trials_per_seed=N_SS_TRIALS_PER_SEED, n_seeds=N_SS_SEEDS, base_seed=42) -> PerturbationSweepResult`

Mirror of `run_stop_signal_condition` in `stop_signal_sweep.py` but with a perturbation wrapper:
1. Build cortex/freq configs.
2. For each seed: build fresh wrapped policy, run StopSignalConfig with stop_trial_go_evidence=True, staircase=True, seed=base_seed+seed_index.
3. Aggregate all trials; compute stop-signal metrics via `compute_stop_signal_metrics` and `validate_stop_signal_data`.
4. Return PerturbationSweepResult with paradigm="stop_signal", stop_failure_rate, ssrt_estimate_s, go_rt_mean_s, inhibition_function_monotone.

**StopSignalConfig parameters** (match Phase 7 stop_signal_sweep.py):
```python
StopSignalConfig(
    n_trials=n_trials_per_seed,
    stop_proportion=SS_STOP_PROPORTION,
    initial_ssd_ms=SS_INITIAL_SSD_MS,
    ssd_step_ms=SS_SSD_STEP_MS,
    ssd_min_ms=SS_SSD_MIN_MS,
    ssd_max_ms=SS_SSD_MAX_MS,
    use_staircase=True,
    stop_trial_go_evidence=True,
    decision_point_ms=SS_DECISION_POINT_MS,
    go_cue_onset_ms=SS_GO_CUE_ONSET_MS,
    seed=base_seed + seed_index,
)
```

#### `format_decomposition_report(gonogo_results, stopsignal_results) -> str`

Signature:
```python
def format_decomposition_report(
    gonogo_results: list[PerturbationSweepResult],
    stopsignal_results: list[PerturbationSweepResult],
) -> str:
```

Report structure:
```
================================================================
Phase 9 — Latency/Jitter/Dropout/Phase Decomposition Report (M10)
================================================================

INTERPRETATION GUIDE (§11):
  - Frequency drives wrong-choice / miss rate     → selector-bottleneck
  - Frequency shifts RT, choices preserved        → urgency/commitment account
  - Dropout increases false alarms / stop failures → cancellation-bottleneck
  - Latency/jitter shift RT only, choices intact  → timing-precision effect

--- Go/No-Go: {perturbation_type} sweep ---
  Freq  | Pert Level        | go_success% | false_alarm% | commit_lat_ms
  ...

--- Stop-Signal: {perturbation_type} sweep ---
  Freq  | Pert Level        | stop_fail%  | SSRT_s       | inh_mono
  ...

================================================================
```

One section per perturbation_type (iterate in order: latency, jitter, dropout, phase_offset).
Within each perturbation_type section, sort by (frequency_hz, perturbation_value).
Format floats to 3 decimal places. "N/A" for None values.

---

### File to create: `tests/test_perturbation_sweep.py`

Write ≥20 tests. Use TDD: structure, constants, type checks, then behavioral.

Required tests (exact names):
1. `test_perturbation_levels_have_zero_baseline` — all four level lists start with 0.0
2. `test_latency_levels_strictly_increasing` — LATENCY_LEVELS_MS is sorted ascending
3. `test_phase_offset_fractions_in_unit_interval` — all fractions in [0,1]
4. `test_dropout_levels_in_unit_interval` — all levels in [0,1]
5. `test_phase_offset_ms_zero_fraction` — _phase_offset_ms(0.0, 20.0) == 0.0
6. `test_phase_offset_ms_full_period` — _phase_offset_ms(1.0, 10.0) == 100.0
7. `test_phase_offset_ms_half_period` — _phase_offset_ms(0.5, 40.0) == pytest.approx(12.5)
8. `test_make_wrapped_policy_latency` — returns LatencyWrapper instance
9. `test_make_wrapped_policy_jitter` — returns JitterWrapper instance
10. `test_make_wrapped_policy_dropout` — returns DropoutWrapper instance
11. `test_make_wrapped_policy_phase_offset` — returns PhaseOffsetWrapper instance
12. `test_make_wrapped_policy_unknown_type_raises` — ValueError on unknown type
13. `test_gonogo_baseline_result_fields` — run at 20 Hz, latency=0 → valid PerturbationSweepResult with all go_nogo fields set
14. `test_gonogo_result_paradigm_is_gonogo` — paradigm field == "go_nogo"
15. `test_gonogo_stop_signal_fields_are_none` — stop_failure_rate etc. are None
16. `test_gonogo_high_frequency_success_near_one` — go_success_rate ≈ 1.0 at 40 Hz, latency=0
17. `test_gonogo_latency_does_not_change_success_rate` — at 40 Hz, latency=100ms → go_success_rate same as latency=0 (within 0.05)
18. `test_gonogo_dropout_high_changes_false_alarm` — at 40 Hz, dropout=0.10 vs 0.0: false_alarm_rate may differ (just assert result is valid)
19. `test_stopsignal_baseline_result_fields` — run at 20 Hz, latency=0 → valid result with all stop_signal fields set
20. `test_stopsignal_result_paradigm_is_stop_signal` — paradigm == "stop_signal"
21. `test_stopsignal_gonogo_fields_are_none` — go_success_rate etc. are None
22. `test_perturbation_label_latency_format` — label contains "latency=" and "ms"
23. `test_perturbation_label_dropout_format` — label contains "dropout=" and "%"
24. `test_format_decomposition_report_has_header` — report string contains "M10"
25. `test_format_decomposition_report_has_all_perturbation_types` — report contains "latency", "jitter", "dropout", "phase_offset"

Use small n_trials_per_seed=5, n_seeds=2, fast seeds for speed (tests must be fast).
Import `_phase_offset_ms` and `_make_wrapped_policy` directly for testing.

---

## Key constraints

1. **Fail fast**: raise ValueError immediately for invalid perturbation_type.
2. **Fresh policy per seed**: create a new wrapped policy instance for each seed iteration so DropoutWrapper inter-call state resets between seeds.
3. **No silent fallbacks**: do not catch exceptions inside condition runners.
4. **Section-header comments** (`# --- SectionName ---`) required.
5. **Decision-point comments** (Trigger / Why / Outcome) for the fresh-policy-per-seed loop.
6. **Same timing params as prior phases**: PEAK_SALIENCE=0.85, RISE_TIME_MS=200.0, ACCUMULATION_MS=200.0.
7. **false_alarm_rate computation**: fraction of no_go trials where success==False and the trial DID respond (failure_mode != "miss"); use scorer `false_alarm_rate` field from `score_trials` output — `Metrics.false_alarm_rate` is defined in scorer.py.
8. **bg_commitment_latency_mean** is in seconds (matching thalamic_relay_time units).
9. **n_trials in result**: total trials across all seeds.

---

## Commit style

One commit:
```
feat: perturbation sweep library and tests (Task 9.1, M10)

ChangeSet-ID: p9t1-perturb-lib
```

---

## Report contract

Write your full report to `/home/fom/code/NRP_BGA-SB/.superpowers/briefs/task-9.1-report.md`.

Return to me only:
- STATUS: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED
- Commit hash(es)
- One-line test summary (count, pass/fail)
- Any concerns (if DONE_WITH_CONCERNS)
