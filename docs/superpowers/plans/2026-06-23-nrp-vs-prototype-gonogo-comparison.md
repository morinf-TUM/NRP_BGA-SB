# Go/No-Go Prototype-vs-nrp-core Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, offline comparison between the deprecated pure-Python prototype's go/no-go results and the nrp-core binding's go/no-go results, producing a committed markdown report whose headline verdict is "3 of 4 knobs reproduce the prototype exactly; the integration knob diverges at 5 Hz."

**Architecture:** Approach A (snapshot + offline compare). The two existing nrp experiment scripts are re-pointed to write their final JSON into a new *committed* `nrp/results/` directory (per-trial run dirs stay in the gitignored `nrp/run/`). A pure-function library `nrp/compare.py` loads the committed prototype JSONs and nrp snapshots, normalizes metric/grid, classifies categorical regimes, and emits per-knob verdicts. A driver `experiments/nrp_vs_prototype.py` writes the markdown report. All comparison code runs under host pytest with no NRPCoreSim, no `.nrp_env`, no numpy constraint.

**Tech Stack:** Python 3.10, stdlib only (`json`, `pathlib`, `dataclasses`), pytest. No new dependencies.

## Global Constraints

- Comparison library and tests MUST be host-only: no import of NRPCoreSim, no `.nrp_env`, no numpy<2 requirement. Pure stdlib + pytest.
- `nrp/run/` is gitignored; `nrp/results/` is NOT — committed snapshots live there.
- Metric normalization: prototype ablation stores `miss_rate`; nrp stores `go_success_rate`; `go_success_rate = 1 − miss_rate`.
- Regime classification threshold: `success` iff `rate > 0.5`, else `miss`.
- Prototype knob-name → nrp knob-label map: `input_sampling_hz→sampling`, `integration_step_hz→integration`, `output_emission_hz→emission`, `commitment_update_hz→commitment`. The prototype `all`/baseline record is excluded from per-knob series.
- Frequency-sweep comparison is qualitative only (monotone-rise/onset), never a per-frequency pass/fail — the prototype JSON aggregates conflict levels.
- Follow house style: type-safe dataclasses, fail-fast (no silent fallback / broad except / speculative getattr), small focused files, literate comments explaining *why*.
- Commit messages end with a `ChangeSet-ID:` trailer.

---

## File Structure

- Create `nrp/results/gonogo_sweep.json` — committed nrp frequency-sweep snapshot.
- Create `nrp/results/ablation.json` — committed nrp four-knob-ablation snapshot.
- Modify `experiments/nrp_gonogo_sweep.py:31-34` — re-point final JSON to `nrp/results/gonogo_sweep.json`.
- Modify `experiments/nrp_ablation.py:43-45` — re-point final JSON to `nrp/results/ablation.json`.
- Create `nrp/compare.py` — pure comparison library (loaders, regime classification, alignment, ablation/sweep comparison, verdict, report formatting).
- Create `experiments/nrp_vs_prototype.py` — driver; writes `docs/nrp_vs_prototype_comparison.md`.
- Create `tests/nrp/test_compare.py` — unit tests (synthetic) + regression test (real committed JSONs).
- Create `docs/nrp_vs_prototype_comparison.md` — generated comparison report (committed in final task).

---

### Task 1: Committed nrp snapshots + re-pointed experiment scripts

**Files:**
- Create: `nrp/results/gonogo_sweep.json`
- Create: `nrp/results/ablation.json`
- Modify: `experiments/nrp_gonogo_sweep.py:31-34`
- Modify: `experiments/nrp_ablation.py:43-45`
- Test: `tests/nrp/test_compare.py` (snapshot-shape test only in this task)

**Interfaces:**
- Consumes: nothing.
- Produces: committed snapshot files at `nrp/results/gonogo_sweep.json` (`{str_hz: rate}`) and `nrp/results/ablation.json` (`{knob: {str_hz: rate}}`). These exact values are the live-run outputs and become the regression fixtures for later tasks.

- [ ] **Step 1: Write the committed frequency-sweep snapshot**

Create `nrp/results/gonogo_sweep.json`:

```json
{
  "5.0": 0.0,
  "10.0": 1.0,
  "20.0": 1.0,
  "40.0": 1.0,
  "80.0": 1.0,
  "160.0": 1.0
}
```

- [ ] **Step 2: Write the committed ablation snapshot**

Create `nrp/results/ablation.json`:

```json
{
  "sampling":    {"5.0": 0.0, "10.0": 1.0, "20.0": 1.0, "40.0": 1.0, "80.0": 1.0, "160.0": 1.0},
  "integration": {"5.0": 1.0, "10.0": 1.0, "20.0": 1.0, "40.0": 1.0, "80.0": 1.0, "160.0": 1.0},
  "emission":    {"5.0": 0.0, "10.0": 1.0, "20.0": 1.0, "40.0": 1.0, "80.0": 1.0, "160.0": 1.0},
  "commitment":  {"5.0": 0.0, "10.0": 1.0, "20.0": 1.0, "40.0": 1.0, "80.0": 1.0, "160.0": 1.0}
}
```

- [ ] **Step 3: Re-point the frequency-sweep script output**

In `experiments/nrp_gonogo_sweep.py`, replace lines 31-34:

```python
    out_root = Path("nrp/run/gonogo_sweep")
    rates = run_sweep(FREQUENCIES_HZ, n_seeds=5, run_root=out_root)
    result_path = Path("nrp/run/nrp_gonogo_sweep.json")
    result_path.write_text(json.dumps(rates, indent=2))
```

with:

```python
    # Per-trial run dirs stay under the gitignored nrp/run/; the final snapshot
    # lands in the committed nrp/results/ so the offline comparison can consume it.
    out_root = Path("nrp/run/gonogo_sweep")
    rates = run_sweep(FREQUENCIES_HZ, n_seeds=5, run_root=out_root)
    result_path = Path("nrp/results/gonogo_sweep.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(rates, indent=2))
```

- [ ] **Step 4: Re-point the ablation script output**

In `experiments/nrp_ablation.py`, replace lines 43-45:

```python
    run_root = Path("nrp/run/ablation")
    results = {k: ablate_knob(k, FREQUENCIES_HZ, run_root) for k in KNOBS}
    Path("nrp/run/nrp_ablation.json").write_text(json.dumps(results, indent=2))
```

with:

```python
    # Per-trial run dirs stay under the gitignored nrp/run/; the final snapshot
    # lands in the committed nrp/results/ so the offline comparison can consume it.
    run_root = Path("nrp/run/ablation")
    results = {k: ablate_knob(k, FREQUENCIES_HZ, run_root) for k in KNOBS}
    result_path = Path("nrp/results/ablation.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2))
```

Also update the final print in `experiments/nrp_ablation.py` (line 50):

```python
    print("saved -> nrp/run/nrp_ablation.json")
```

to:

```python
    print(f"saved -> {result_path}")
```

- [ ] **Step 5: Write the snapshot-shape test**

Create `tests/nrp/test_compare.py`:

```python
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "nrp" / "results"


def test_committed_snapshots_lock_key_divergence():
    """The committed nrp snapshots must preserve the headline divergence:
    sampling misses at 5 Hz, integration does NOT (idempotent sub-step)."""
    ablation = json.loads((RESULTS / "ablation.json").read_text())
    assert ablation["sampling"]["5.0"] == 0.0
    assert ablation["integration"]["5.0"] == 1.0
    sweep = json.loads((RESULTS / "gonogo_sweep.json").read_text())
    assert sweep["5.0"] == 0.0
    assert sweep["10.0"] == 1.0
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/nrp/test_compare.py -v`
Expected: PASS (snapshots exist with the locked values).

- [ ] **Step 7: Commit**

```bash
git add nrp/results/gonogo_sweep.json nrp/results/ablation.json \
        experiments/nrp_gonogo_sweep.py experiments/nrp_ablation.py \
        tests/nrp/test_compare.py
git commit -m "feat: commit nrp go/no-go snapshots and re-point experiment outputs

ChangeSet-ID: nrp-compare-snapshots"
```

---

### Task 2: Loaders and metric normalization

**Files:**
- Create: `nrp/compare.py`
- Test: `tests/nrp/test_compare.py` (append)

**Interfaces:**
- Consumes: snapshot/prototype JSON files on disk (paths passed by caller).
- Produces:
  - `KNOBS: tuple[str, ...] = ("sampling", "integration", "emission", "commitment")`
  - `KNOB_NAME_MAP: dict[str, str]` (prototype name → nrp label)
  - `load_prototype_ablation(path) -> dict[str, dict[float, float]]`
  - `load_nrp_ablation(path) -> dict[str, dict[float, float]]`
  - `load_prototype_gonogo_sweep(path) -> dict[float, float]`
  - `load_nrp_gonogo_sweep(path) -> dict[float, float]`

- [ ] **Step 1: Write the failing loader tests**

Append to `tests/nrp/test_compare.py`:

```python
from nrp.compare import (
    KNOBS,
    load_nrp_ablation,
    load_nrp_gonogo_sweep,
    load_prototype_ablation,
    load_prototype_gonogo_sweep,
)

PROTO = Path(__file__).resolve().parents[2] / "deprecated_toy_prototype_results"


def test_load_prototype_ablation_normalizes_and_maps(tmp_path):
    records = [
        {"condition": "baseline", "knob_name": "all", "freq_hz": 160.0, "miss_rate": 0.0},
        {"condition": "sweep", "knob_name": "input_sampling_hz", "freq_hz": 5.0, "miss_rate": 1.0},
        {"condition": "sweep", "knob_name": "integration_step_hz", "freq_hz": 5.0, "miss_rate": 0.0},
    ]
    p = tmp_path / "ab.json"
    p.write_text(json.dumps(records))
    out = load_prototype_ablation(p)
    # 'all' baseline excluded; miss_rate -> go_success_rate; names mapped.
    assert "all" not in out
    assert out["sampling"][5.0] == 0.0
    assert out["integration"][5.0] == 1.0


def test_load_nrp_ablation_coerces_str_hz_keys():
    out = load_nrp_ablation(RESULTS / "ablation.json")
    assert out["sampling"][5.0] == 0.0
    assert out["integration"][5.0] == 1.0
    assert set(out) == set(KNOBS)


def test_load_prototype_gonogo_sweep_filters_and_aggregates(tmp_path):
    records = [
        {"paradigm": "go_nogo", "frequency_hz": 10.0, "go_success_rate": 0.0},
        {"paradigm": "go_nogo", "frequency_hz": 10.0, "go_success_rate": 1.0},
        {"paradigm": "two_choice", "frequency_hz": 10.0, "go_success_rate": 0.0},
    ]
    p = tmp_path / "fs.json"
    p.write_text(json.dumps(records))
    out = load_prototype_gonogo_sweep(p)
    # two_choice excluded; go_nogo rates averaged: (0.0 + 1.0) / 2 = 0.5
    assert out == {10.0: 0.5}


def test_load_nrp_gonogo_sweep_coerces_str_hz_keys():
    out = load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json")
    assert out[5.0] == 0.0
    assert out[10.0] == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nrp/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nrp.compare'` / import errors.

- [ ] **Step 3: Write the loaders**

Create `nrp/compare.py`:

```python
"""Offline comparison of pure-Python prototype vs nrp-core go/no-go results.

Pure functions only — no NRPCoreSim, no module-level I/O. Loaders normalize the
two sides onto a common representation (per-knob and per-frequency
go_success_rate keyed by float Hz), so the prototype's `miss_rate` and the nrp
binding's `go_success_rate` become directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

# --- Knob vocabulary ---

# Canonical nrp-side knob labels, in report order.
KNOBS: tuple[str, ...] = ("sampling", "integration", "emission", "commitment")

# Prototype FrequencyConfig field names -> canonical nrp labels.
KNOB_NAME_MAP: dict[str, str] = {
    "input_sampling_hz": "sampling",
    "integration_step_hz": "integration",
    "output_emission_hz": "emission",
    "commitment_update_hz": "commitment",
}


# --- Loaders (normalize both sides to go_success_rate keyed by float Hz) ---


def load_prototype_ablation(path: str | Path) -> dict[str, dict[float, float]]:
    """Prototype ablation: list of {knob_name, freq_hz, miss_rate}. Converts
    miss_rate -> go_success_rate and maps knob names. The 'all'/baseline record
    is not a per-knob sweep series and is excluded."""
    records = json.loads(Path(path).read_text())
    out: dict[str, dict[float, float]] = {knob: {} for knob in KNOBS}
    for r in records:
        name = r["knob_name"]
        if name == "all":
            continue
        knob = KNOB_NAME_MAP[name]
        out[knob][float(r["freq_hz"])] = 1.0 - float(r["miss_rate"])
    return out


def load_nrp_ablation(path: str | Path) -> dict[str, dict[float, float]]:
    """nrp ablation: {knob: {str_hz: go_success_rate}}. Coerce Hz keys to float."""
    raw = json.loads(Path(path).read_text())
    return {
        knob: {float(hz): float(rate) for hz, rate in series.items()}
        for knob, series in raw.items()
    }


def load_prototype_gonogo_sweep(path: str | Path) -> dict[float, float]:
    """Prototype frequency sweep: filter paradigm == 'go_nogo', average
    go_success_rate per frequency over conflict levels and seeds."""
    records = json.loads(Path(path).read_text())
    by_freq: dict[float, list[float]] = {}
    for r in records:
        if r.get("paradigm") != "go_nogo":
            continue
        by_freq.setdefault(float(r["frequency_hz"]), []).append(float(r["go_success_rate"]))
    return {hz: sum(vals) / len(vals) for hz, vals in by_freq.items()}


def load_nrp_gonogo_sweep(path: str | Path) -> dict[float, float]:
    """nrp frequency sweep: {str_hz: go_success_rate}. Coerce Hz keys to float."""
    raw = json.loads(Path(path).read_text())
    return {float(hz): float(rate) for hz, rate in raw.items()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/nrp/test_compare.py -v`
Expected: PASS (all loader tests + the Task 1 snapshot test).

- [ ] **Step 5: Commit**

```bash
git add nrp/compare.py tests/nrp/test_compare.py
git commit -m "feat: nrp/compare loaders with metric+knob normalization

ChangeSet-ID: nrp-compare-loaders"
```

---

### Task 3: Regime classification, alignment, and ablation verdict

**Files:**
- Modify: `nrp/compare.py`
- Test: `tests/nrp/test_compare.py` (append)

**Interfaces:**
- Consumes: `KNOBS`, loader outputs (`dict[str, dict[float, float]]`).
- Produces:
  - `classify_regime(rate: float) -> str` — `"success"` iff `rate > 0.5` else `"miss"`.
  - `@dataclass FreqRow(freq_hz: float, proto_rate: float | None, nrp_rate: float | None, tag: str)` where `tag ∈ {"common","proto_only","nrp_only"}`.
  - `align_series(proto: dict[float,float], nrp: dict[float,float]) -> list[FreqRow]` (sorted by freq).
  - `@dataclass KnobVerdict(knob: str, holds: bool, divergent_freqs: list[float], rows: list[FreqRow])`.
  - `@dataclass AblationVerdict(knobs: list[KnobVerdict], holds: bool)`.
  - `compare_ablation(proto, nrp) -> AblationVerdict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/nrp/test_compare.py`:

```python
from nrp.compare import (
    align_series,
    classify_regime,
    compare_ablation,
)


def test_classify_regime_threshold():
    assert classify_regime(1.0) == "success"
    assert classify_regime(0.51) == "success"
    assert classify_regime(0.5) == "miss"
    assert classify_regime(0.0) == "miss"


def test_align_series_tags_membership():
    rows = align_series({5.0: 0.0, 10.0: 1.0}, {10.0: 1.0, 20.0: 1.0})
    by_hz = {r.freq_hz: r for r in rows}
    assert by_hz[5.0].tag == "proto_only"
    assert by_hz[10.0].tag == "common"
    assert by_hz[20.0].tag == "nrp_only"
    # sorted by frequency
    assert [r.freq_hz for r in rows] == [5.0, 10.0, 20.0]


def test_compare_ablation_flags_integration_divergence():
    proto = load_prototype_ablation(PROTO / "ablation_frequency_v2.json")
    nrp = load_nrp_ablation(RESULTS / "ablation.json")
    verdict = compare_ablation(proto, nrp)
    by_knob = {kv.knob: kv for kv in verdict.knobs}
    assert by_knob["sampling"].holds is True
    assert by_knob["emission"].holds is True
    assert by_knob["commitment"].holds is True
    assert by_knob["integration"].holds is False
    assert by_knob["integration"].divergent_freqs == [5.0]
    assert verdict.holds is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nrp/test_compare.py -k "regime or align or ablation" -v`
Expected: FAIL with ImportError for `classify_regime` / `align_series` / `compare_ablation`.

- [ ] **Step 3: Implement classification, alignment, and ablation comparison**

Append to `nrp/compare.py` (add `from dataclasses import dataclass, field` to the imports at the top):

```python
# --- Categorical regime ---


def classify_regime(rate: float) -> str:
    """Coarse outcome regime. Seed counts differ across sides (prototype 24-30
    trials, nrp 3-5 seeds), so exact rates are not meaningful — only whether the
    go response is predominantly released."""
    return "success" if rate > 0.5 else "miss"


# --- Frequency-grid alignment ---


@dataclass
class FreqRow:
    freq_hz: float
    proto_rate: float | None
    nrp_rate: float | None
    tag: str  # "common" | "proto_only" | "nrp_only"


def align_series(proto: dict[float, float], nrp: dict[float, float]) -> list[FreqRow]:
    """Union of frequencies, sorted, each tagged by which side(s) cover it."""
    rows: list[FreqRow] = []
    for hz in sorted(set(proto) | set(nrp)):
        in_p, in_n = hz in proto, hz in nrp
        # Trigger: frequency present on both sides vs only one.
        # Why: only common frequencies can be judged agree/diverge; one-sided
        #      frequencies (e.g. nrp-only 5 Hz in the sweep) are reported, not graded.
        tag = "common" if in_p and in_n else ("proto_only" if in_p else "nrp_only")
        rows.append(FreqRow(freq_hz=hz, proto_rate=proto.get(hz), nrp_rate=nrp.get(hz), tag=tag))
    return rows


# --- Ablation comparison ---


@dataclass
class KnobVerdict:
    knob: str
    holds: bool
    divergent_freqs: list[float]
    rows: list[FreqRow]


@dataclass
class AblationVerdict:
    knobs: list[KnobVerdict]
    holds: bool


def compare_ablation(
    proto: dict[str, dict[float, float]],
    nrp: dict[str, dict[float, float]],
) -> AblationVerdict:
    """Per-knob regime match across the common frequency grid. A knob holds iff
    its prototype and nrp regimes agree at every shared frequency."""
    knob_verdicts: list[KnobVerdict] = []
    for knob in KNOBS:
        rows = align_series(proto.get(knob, {}), nrp.get(knob, {}))
        divergent = [
            r.freq_hz
            for r in rows
            if r.tag == "common" and classify_regime(r.proto_rate) != classify_regime(r.nrp_rate)
        ]
        knob_verdicts.append(
            KnobVerdict(knob=knob, holds=not divergent, divergent_freqs=divergent, rows=rows)
        )
    return AblationVerdict(knobs=knob_verdicts, holds=all(kv.holds for kv in knob_verdicts))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/nrp/test_compare.py -v`
Expected: PASS (all tests including the integration-divergence assertions).

- [ ] **Step 5: Commit**

```bash
git add nrp/compare.py tests/nrp/test_compare.py
git commit -m "feat: regime classification, grid alignment, ablation verdict

ChangeSet-ID: nrp-compare-ablation"
```

---

### Task 4: Frequency-sweep comparison (qualitative, caveated)

**Files:**
- Modify: `nrp/compare.py`
- Test: `tests/nrp/test_compare.py` (append)

**Interfaces:**
- Consumes: `classify_regime`, sweep loader outputs (`dict[float, float]`).
- Produces:
  - `@dataclass SweepVerdict(proto_onset_hz: float | None, nrp_onset_hz: float | None, proto_rows: dict[float,float], nrp_rows: dict[float,float], caveat: str)`.
  - `compare_frequency_sweep(proto, nrp) -> SweepVerdict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/nrp/test_compare.py`:

```python
from nrp.compare import compare_frequency_sweep


def test_compare_frequency_sweep_reports_onsets_with_caveat():
    proto = load_prototype_gonogo_sweep(PROTO / "frequency_sweep_results.json")
    nrp = load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json")
    verdict = compare_frequency_sweep(proto, nrp)
    # nrp onset is 10 Hz (5 Hz misses). Prototype onset is inflated to 20 Hz by
    # conflict-level aggregation (10 Hz aggregate = 0.333 -> miss regime).
    assert verdict.nrp_onset_hz == 10.0
    assert verdict.proto_onset_hz == 20.0
    assert "conflict" in verdict.caveat.lower()


def test_compare_frequency_sweep_onset_none_when_all_miss():
    verdict = compare_frequency_sweep({10.0: 0.0, 20.0: 0.0}, {10.0: 0.0})
    assert verdict.proto_onset_hz is None
    assert verdict.nrp_onset_hz is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nrp/test_compare.py -k "frequency_sweep" -v`
Expected: FAIL with ImportError for `compare_frequency_sweep`.

- [ ] **Step 3: Implement the sweep comparison**

Append to `nrp/compare.py`:

```python
# --- Frequency-sweep comparison (qualitative only) ---


@dataclass
class SweepVerdict:
    proto_onset_hz: float | None
    nrp_onset_hz: float | None
    proto_rows: dict[float, float]
    nrp_rows: dict[float, float]
    caveat: str


def _success_onset(series: dict[float, float]) -> float | None:
    """Lowest frequency whose regime is 'success', or None if never."""
    successes = [hz for hz in sorted(series) if classify_regime(series[hz]) == "success"]
    return successes[0] if successes else None


def compare_frequency_sweep(
    proto: dict[float, float],
    nrp: dict[float, float],
) -> SweepVerdict:
    """Qualitative comparison only. The prototype sweep JSON averages conflict
    levels (low/medium/high), which inflates its apparent success onset, so we
    report each side's monotone curve and onset but never a per-frequency
    pass/fail."""
    return SweepVerdict(
        proto_onset_hz=_success_onset(proto),
        nrp_onset_hz=_success_onset(nrp),
        proto_rows=dict(proto),
        nrp_rows=dict(nrp),
        caveat=(
            "Prototype frequency-sweep rates aggregate conflict levels "
            "(low/medium/high), inflating the apparent success onset; only the "
            "qualitative monotone rise to 1.0 is comparable, not the onset "
            "frequency or per-frequency rates."
        ),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/nrp/test_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nrp/compare.py tests/nrp/test_compare.py
git commit -m "feat: qualitative caveated frequency-sweep comparison

ChangeSet-ID: nrp-compare-sweep"
```

---

### Task 5: Overall verdict and markdown report formatting

**Files:**
- Modify: `nrp/compare.py`
- Test: `tests/nrp/test_compare.py` (append)

**Interfaces:**
- Consumes: `AblationVerdict`, `SweepVerdict`, `KNOBS`, `classify_regime`.
- Produces:
  - `@dataclass OverallVerdict(ablation: AblationVerdict, sweep: SweepVerdict, summary: str)`.
  - `build_verdict(ablation: AblationVerdict, sweep: SweepVerdict) -> OverallVerdict`.
  - `format_report(verdict: OverallVerdict) -> str` (markdown).

- [ ] **Step 1: Write the failing tests**

Append to `tests/nrp/test_compare.py`:

```python
from nrp.compare import build_verdict, format_report


def test_build_verdict_summary_names_integration():
    proto_ab = load_prototype_ablation(PROTO / "ablation_frequency_v2.json")
    nrp_ab = load_nrp_ablation(RESULTS / "ablation.json")
    proto_fs = load_prototype_gonogo_sweep(PROTO / "frequency_sweep_results.json")
    nrp_fs = load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json")
    verdict = build_verdict(
        compare_ablation(proto_ab, nrp_ab),
        compare_frequency_sweep(proto_fs, nrp_fs),
    )
    assert "3 of 4" in verdict.summary
    assert "integration" in verdict.summary.lower()


def test_format_report_contains_tables_and_callout():
    proto_ab = load_prototype_ablation(PROTO / "ablation_frequency_v2.json")
    nrp_ab = load_nrp_ablation(RESULTS / "ablation.json")
    proto_fs = load_prototype_gonogo_sweep(PROTO / "frequency_sweep_results.json")
    nrp_fs = load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json")
    verdict = build_verdict(
        compare_ablation(proto_ab, nrp_ab),
        compare_frequency_sweep(proto_fs, nrp_fs),
    )
    report = format_report(verdict)
    assert report.startswith("# Go/No-Go: Prototype vs nrp-core")
    assert "## Ablation" in report
    assert "## Frequency sweep" in report
    assert "integration" in report.lower()
    assert "5" in report  # the 5 Hz divergence appears
    assert "conflict" in report.lower()  # the sweep caveat is present
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/nrp/test_compare.py -k "verdict or format_report" -v`
Expected: FAIL with ImportError for `build_verdict` / `format_report`.

- [ ] **Step 3: Implement verdict and report**

Append to `nrp/compare.py`:

```python
# --- Overall verdict + report ---


@dataclass
class OverallVerdict:
    ablation: AblationVerdict
    sweep: SweepVerdict
    summary: str


def build_verdict(ablation: AblationVerdict, sweep: SweepVerdict) -> OverallVerdict:
    held = [kv.knob for kv in ablation.knobs if kv.holds]
    diverged = [kv.knob for kv in ablation.knobs if not kv.holds]
    summary = (
        f"{len(held)} of {len(ablation.knobs)} knobs reproduce the prototype "
        f"exactly across the common grid ({', '.join(held)})."
    )
    if diverged:
        # The expected, scientifically meaningful outcome: integration diverges
        # because the nrp sub-step is idempotent on the stateless BG solver.
        summary += (
            f" The {', '.join(diverged)} knob(s) diverge — see the per-knob table; "
            f"this is the knob-2 idempotence finding (PROJECT_MEMORY §15.7) and "
            f"motivates the separate knob-2 modeling project."
        )
    return OverallVerdict(ablation=ablation, sweep=sweep, summary=summary)


def _ablation_table(ablation: AblationVerdict) -> str:
    # Column frequencies = union across all knob rows, sorted.
    freqs = sorted({r.freq_hz for kv in ablation.knobs for r in kv.rows})
    header = "| knob | " + " | ".join(f"{f:g} Hz" for f in freqs) + " | holds |"
    sep = "|" + "---|" * (len(freqs) + 2)
    lines = [header, sep]
    for kv in ablation.knobs:
        by_hz = {r.freq_hz: r for r in kv.rows}
        cells = []
        for f in freqs:
            row = by_hz.get(f)
            if row is None or (row.proto_rate is None and row.nrp_rate is None):
                cells.append("–")
            else:
                p = "–" if row.proto_rate is None else f"{row.proto_rate:g}"
                n = "–" if row.nrp_rate is None else f"{row.nrp_rate:g}"
                mark = ""
                if row.tag == "common" and classify_regime(row.proto_rate) != classify_regime(row.nrp_rate):
                    mark = " ✗"
                cells.append(f"{p}/{n}{mark}")
        status = "✓" if kv.holds else f"✗ @ {', '.join(f'{x:g}' for x in kv.divergent_freqs)} Hz"
        lines.append(f"| {kv.knob} | " + " | ".join(cells) + f" | {status} |")
    return "\n".join(lines)


def _sweep_table(sweep: SweepVerdict) -> str:
    freqs = sorted(set(sweep.proto_rows) | set(sweep.nrp_rows))
    header = "| side | " + " | ".join(f"{f:g} Hz" for f in freqs) + " | onset |"
    sep = "|" + "---|" * (len(freqs) + 2)

    def fmt(rows: dict[float, float], onset: float | None) -> str:
        cells = [(f"{rows[f]:g}" if f in rows else "–") for f in freqs]
        onset_s = "none" if onset is None else f"{onset:g} Hz"
        return " | ".join(cells) + f" | {onset_s} |"

    return "\n".join([
        header,
        sep,
        "| prototype | " + fmt(sweep.proto_rows, sweep.proto_onset_hz),
        "| nrp-core | " + fmt(sweep.nrp_rows, sweep.nrp_onset_hz),
    ])


def format_report(verdict: OverallVerdict) -> str:
    """Render the comparison as a self-contained markdown report. Cells show
    prototype/nrp go-success; ✗ marks a common-grid regime divergence."""
    return f"""# Go/No-Go: Prototype vs nrp-core Comparison

**Verdict:** {verdict.summary}

## Ablation (primary, like-for-like)

Cells are prototype/nrp go-success rate; ✗ marks a common-grid regime divergence.

{_ablation_table(verdict.ablation)}

## Frequency sweep (secondary, qualitative)

> {verdict.sweep.caveat}

{_sweep_table(verdict.sweep)}
"""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/nrp/test_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nrp/compare.py tests/nrp/test_compare.py
git commit -m "feat: overall verdict and markdown report formatting

ChangeSet-ID: nrp-compare-report"
```

---

### Task 6: Driver script + generated report + regression lock

**Files:**
- Create: `experiments/nrp_vs_prototype.py`
- Create: `docs/nrp_vs_prototype_comparison.md` (generated by the driver)
- Test: `tests/nrp/test_compare.py` (append the end-to-end regression test)

**Interfaces:**
- Consumes: all of `nrp/compare.py`; committed prototype JSONs and nrp snapshots.
- Produces: a runnable driver and a committed markdown report.

- [ ] **Step 1: Write the driver**

Create `experiments/nrp_vs_prototype.py`:

```python
"""Offline go/no-go comparison: pure-Python prototype vs nrp-core binding.

Loads committed result snapshots from both sides (no NRPCoreSim needed), builds
the verdict, writes the markdown report, and prints a one-line summary.

Regenerate the nrp snapshots (live, env-gated, slow) before running this if the
binding changed:
    source $HOME/.local/nrp/bin/.nrp_env
    python experiments/nrp_gonogo_sweep.py   # -> nrp/results/gonogo_sweep.json
    python experiments/nrp_ablation.py       # -> nrp/results/ablation.json
"""

from __future__ import annotations

from pathlib import Path

from nrp.compare import (
    build_verdict,
    compare_ablation,
    compare_frequency_sweep,
    format_report,
    load_nrp_ablation,
    load_nrp_gonogo_sweep,
    load_prototype_ablation,
    load_prototype_gonogo_sweep,
)

REPO = Path(__file__).resolve().parents[1]
PROTO = REPO / "deprecated_toy_prototype_results"
RESULTS = REPO / "nrp" / "results"
REPORT = REPO / "docs" / "nrp_vs_prototype_comparison.md"


def main() -> None:
    ablation = compare_ablation(
        load_prototype_ablation(PROTO / "ablation_frequency_v2.json"),
        load_nrp_ablation(RESULTS / "ablation.json"),
    )
    sweep = compare_frequency_sweep(
        load_prototype_gonogo_sweep(PROTO / "frequency_sweep_results.json"),
        load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json"),
    )
    verdict = build_verdict(ablation, sweep)
    REPORT.write_text(format_report(verdict))
    print(verdict.summary)
    print(f"saved -> {REPORT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the end-to-end regression test**

Append to `tests/nrp/test_compare.py`:

```python
def test_driver_main_writes_report(tmp_path, monkeypatch):
    import experiments.nrp_vs_prototype as driver

    out = tmp_path / "report.md"
    monkeypatch.setattr(driver, "REPORT", out)
    driver.main()
    text = out.read_text()
    assert text.startswith("# Go/No-Go: Prototype vs nrp-core")
    assert "3 of 4" in text
    assert "integration" in text.lower()
```

- [ ] **Step 3: Run the regression test to verify it fails, then passes**

Run: `python -m pytest tests/nrp/test_compare.py::test_driver_main_writes_report -v`
Expected: FAIL first (`ModuleNotFoundError: experiments.nrp_vs_prototype`) until Step 1's file exists; once present, PASS.

(`experiments.nrp_vs_prototype` is importable because `pyproject.toml` sets `pythonpath = ["."]`; existing tests already do `from experiments.nrp_ablation import …` the same way. No `__init__.py` is needed.)

- [ ] **Step 4: Generate the committed report**

Run: `python experiments/nrp_vs_prototype.py`
Expected stdout: a summary line containing `3 of 4 knobs reproduce the prototype exactly` and `saved -> .../docs/nrp_vs_prototype_comparison.md`.

- [ ] **Step 5: Run the full nrp test suite**

Run: `python -m pytest tests/nrp/ -v`
Expected: PASS (all comparison tests + pre-existing nrp tests).

- [ ] **Step 6: Commit**

```bash
git add experiments/nrp_vs_prototype.py docs/nrp_vs_prototype_comparison.md tests/nrp/test_compare.py
git commit -m "feat: prototype-vs-nrp-core comparison driver and report

ChangeSet-ID: nrp-compare-driver"
```

---

## Self-Review

**1. Spec coverage:**
- `nrp/results/` committed snapshots + re-pointed scripts → Task 1. ✓
- `nrp/compare.py` loaders/normalization → Task 2. ✓
- regime classification, alignment → Task 3. ✓
- `compare_ablation` + divergence flag → Task 3. ✓
- `compare_frequency_sweep` qualitative+caveat → Task 4. ✓
- `build_verdict` + `format_report` → Task 5. ✓
- `experiments/nrp_vs_prototype.py` driver + `docs/nrp_vs_prototype_comparison.md` → Task 6. ✓
- `tests/nrp/test_compare.py` unit + real-data regression → Tasks 1–6 (regression assertions in 3, 5, 6). ✓
- Host-only / offline constraint → all tasks use stdlib + committed JSONs. ✓
- "Holds" criterion (regime match on common grid; 3/4 hold, integration diverges) → Tasks 3, 5 assert exactly this. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**3. Type consistency:** `load_*` return `dict[str, dict[float, float]]` (ablation) / `dict[float, float]` (sweep), consumed by `compare_ablation`/`compare_frequency_sweep`; `FreqRow`/`KnobVerdict`/`AblationVerdict`/`SweepVerdict`/`OverallVerdict` names and fields are reused verbatim across Tasks 3–6; `classify_regime`, `align_series`, `build_verdict`, `format_report` signatures match between definition and call sites. ✓
