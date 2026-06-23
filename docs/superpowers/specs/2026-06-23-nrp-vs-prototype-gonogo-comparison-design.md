# Design — Go/No-Go Prototype-vs-nrp-core Comparison

**Date:** 2026-06-23
**Status:** Approved (brainstorming complete; pending writing-plans)
**Scope:** Go/no-go paradigm only. First of a two-part effort; knob-2 modeling
(making the integration sub-step behaviorally real) is a separate later project.

## Goal

Produce a reproducible, offline comparison between the deprecated pure-Python
prototype's go/no-go results and the nrp-core binding's go/no-go results, to
determine whether the prototype's scientific signatures survive the faithful
port onto the real NRPCoreSim runtime — and to make explicit where they do not.

## Background / key facts established during brainstorming

The comparison rests on two committed prototype artifacts and two nrp-core
snapshots. Their shapes and the headline finding are already known:

- **Prototype ablation** (`deprecated_toy_prototype_results/ablation_frequency_v2.json`):
  25 records `{condition, knob_name, freq_hz, miss_rate, n_go_trials}`. Knobs:
  `input_sampling_hz`, `integration_step_hz`, `output_emission_hz`,
  `commitment_update_hz`, plus a single `all`/baseline record. Grid
  `[5,10,20,40,80,160]` Hz (per-knob); metric `miss_rate`.
- **nrp ablation** (`experiments/nrp_ablation.py` output): `{knob: {hz: go_success_rate}}`,
  knobs `sampling/integration/emission/commitment`, grid `[5,10,20,40,80,160]`.
- **Prototype frequency sweep** (`deprecated_toy_prototype_results/frequency_sweep_results.json`):
  900 records over `frequency_hz × conflict_level × paradigm × seed`; the go/no-go
  slice is 450 records over grid `[10,20,40,80,160]` (no 5 Hz), aggregating
  conflict levels `{low,medium,high}`. Aggregate go_success is **graded**
  (10→0.33, 20/40→0.67, 80/160→1.0) because it mixes conflict levels.
- **nrp frequency sweep** (`experiments/nrp_gonogo_sweep.py` output): `{hz: go_success_rate}`
  over `[5,10,20,40,80,160]`, single condition `cue=go`, 5 seeds.

**Headline finding (already visible in the raw data), go-success rate:**

```
                    5Hz   10   20   40   80  160
PROTOTYPE sampling  0.00  1.0  1.0  1.0  1.0  1.0
NRP       sampling  0.00  1.0  1.0  1.0  1.0  1.0   match
PROTOTYPE integ.    0.00  1.0  1.0  1.0  1.0  1.0
NRP       integ.    1.00  1.0  1.0  1.0  1.0  1.0   DIVERGES at 5 Hz
PROTOTYPE emission  0.00  1.0  1.0  1.0  1.0  1.0
NRP       emission  0.00  1.0  1.0  1.0  1.0  1.0   match
PROTOTYPE commit.   0.00  1.0  1.0  1.0  1.0  1.0
NRP       commit.   0.00  1.0  1.0  1.0  1.0  1.0   match
```

Three of four knobs reproduce the prototype exactly. The **integration knob
diverges at 5 Hz**: the prototype's `ScheduledBGAdapter` gates integration with
the same integer-tick driver as the other knobs (so a 5 Hz integration gate
misses, like sampling), whereas the faithful nrp-core binding implements the
integration sub-step as N repeated calls to a *stateless* steady-state BG solver,
which is idempotent and therefore never misses. This divergence is exactly the
knob-2 idempotence finding (PROJECT_MEMORY §15.7) and is the precise evidence
that motivates the separate knob-2 modeling project.

## Why a clean comparison centers on the ablation

The ablation is like-for-like: identical frequency grid, identical four knobs,
go-trial outcome on both sides (`go_success_rate = 1 − miss_rate`). The frequency
sweep is **not** directly comparable — the prototype JSON aggregates conflict
levels and is multi-paradigm/graded, while the nrp sweep is single-condition. The
frequency sweep is therefore retained only as a **caveated secondary** comparison
at the categorical-boundary level, never as an exact-rate match.

## Architecture & components

Approach A (snapshot + offline compare): the comparison reads committed inputs on
both sides and runs fast, offline, and testable without the NRPCoreSim runtime.

### 1. `nrp/results/` — new committed snapshot directory
Holds the nrp-core result snapshots that the comparison consumes:
- `nrp/results/gonogo_sweep.json`
- `nrp/results/ablation.json`

Populated by re-pointing the **final output path** of the two existing experiment
scripts from `nrp/run/…` (gitignored working area) to `nrp/results/…` (committed).
Per-trial run directories stay under `nrp/run/`. This is a one-line change to the
`result_path` in each of `experiments/nrp_gonogo_sweep.py` and
`experiments/nrp_ablation.py`. The current snapshot values (already produced by
the live runtime) are the ones shown in the headline table above.

### 2. `nrp/compare.py` — pure comparison library
No module-level I/O. Functions:
- `load_prototype_ablation(path) -> dict[str, dict[float, float]]` — parse the
  25-record list; map `knob_name → {input_sampling_hz:'sampling',
  integration_step_hz:'integration', output_emission_hz:'emission',
  commitment_update_hz:'commitment'}`; convert `miss_rate → go_success_rate`
  (`1 − miss_rate`); ignore the `all`/baseline record (kept only for the report
  footnote).
- `load_nrp_ablation(path) -> dict[str, dict[float, float]]` — parse `{knob:{str_hz:rate}}`,
  coercing string Hz keys to float.
- `load_prototype_gonogo_sweep(path) -> dict[float, float]` — filter
  `paradigm == 'go_nogo'`, aggregate `go_success_rate` per `frequency_hz` (mean
  over conflict levels and seeds).
- `load_nrp_gonogo_sweep(path) -> dict[float, float]` — parse `{str_hz:rate}`.
- `align_series(proto, nrp) -> list[Row]` — union of frequencies, each row tagged
  `common | proto_only | nrp_only`.
- `classify_regime(rate) -> 'success' | 'miss'` — success iff `rate > 0.5`
  (regime-based because seed counts differ: prototype 24–30 trials, nrp 3–5 seeds;
  exact float equality is not meaningful).
- `compare_ablation(proto, nrp) -> AblationVerdict` — per-knob regime match over
  the common grid; flags integration as divergent.
- `compare_frequency_sweep(proto, nrp) -> SweepVerdict` — reports each side's
  monotone go-success curve and the location of the success onset, but does **not**
  emit a per-frequency pass/fail. The prototype's graded sub-1.0 rates (e.g.
  10 Hz→0.33) are an artifact of aggregating conflict levels, not a true behavioral
  miss; the verdict states this explicitly and compares only the qualitative
  monotone-rise/onset shape, never regime-by-frequency equality.
- `build_verdict(ablation_verdict, sweep_verdict) -> OverallVerdict`.
- `format_report(...) -> str` — markdown.

### 3. `experiments/nrp_vs_prototype.py` — driver
`__main__` loads the two committed prototype JSONs and the two committed nrp
snapshots, runs both comparisons, writes the report to
`docs/nrp_vs_prototype_comparison.md`, and prints a one-screen summary.

### 4. `tests/nrp/test_compare.py` — host-only tests
- Unit tests on small synthetic dicts: metric normalization (`miss_rate ↔
  go_success_rate`), knob-name mapping, grid alignment (common / proto-only /
  nrp-only), regime classification at the 0.5 boundary, per-knob verdict logic,
  and integration-divergence detection.
- One regression test over the **real committed JSONs** asserting the expected
  verdict: sampling/emission/commitment hold across the common grid; integration
  diverges at 5 Hz (prototype miss, nrp success).
- All tests run under plain host pytest — no NRPCoreSim, no `.nrp_env`,
  no numpy<2 constraint.

## Data flow

```
committed prototype JSONs (deprecated_toy_prototype_results/)
committed nrp snapshots   (nrp/results/)
        │
        ▼
   nrp/compare.py  (pure functions: load → normalize → align → classify → verdict)
        │
        ├──► docs/nrp_vs_prototype_comparison.md   (committed report)
        └──► stdout summary
```

Snapshot regeneration (separate, manual, env-gated, slow ~7 min):
```
source $HOME/.local/nrp/bin/.nrp_env
python experiments/nrp_gonogo_sweep.py   # -> nrp/results/gonogo_sweep.json
python experiments/nrp_ablation.py       # -> nrp/results/ablation.json
```

## "Holds" criterion

A knob's signature **holds** iff its categorical regime (`success`/`miss`) matches
between prototype and nrp-core at every frequency on the common grid. Overall
verdict reports per-knob status. The expected, scientifically meaningful outcome
is: **3/4 knobs hold; integration diverges at 5 Hz** — reported as the key finding
and cross-referenced to PROJECT_MEMORY §15.7 and the future knob-2 project.

## Out of scope (YAGNI)

- Other paradigms (stop-signal, change-of-mind, perturbation, cerebellum, OpenSim).
- Knob-2 modeling (the stateful/incremental BG integrator) — separate project.
- Plots / visualizations — the deliverable is a tabular markdown report.
- Any change to the science layer or the existing nrp engines/TFs.
