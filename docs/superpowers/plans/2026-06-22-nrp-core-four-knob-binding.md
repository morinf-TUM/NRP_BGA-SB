# NRP-Core Four-Knob BG Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-host the existing, validated pure-Python BG action-selection loop as a real nrp-core (NRPCoreSim / FTILoop) experiment, building incrementally from a conservative 3-engine go/no-go spine up to the full four-knob frequency-dissociation model — each knob realised as a distinct engine `EngineTimestep` (or internal sub-step), with every phase re-validating that the empirical BG-frequency signature survives the runtime.

**Architecture:** Each module of the existing loop (`cortex`, `bg_model`, `thalamus`) becomes a separate nrp-core `python_json` engine running at its own `EngineTimestep`; `TransceiverFunction`s move typed DataPacks between them; a logger TF writes per-trial decision/motor traces to disk. The engines **delegate to the already-tested `nrp_bga_sb` classes** rather than reimplementing the science. Outcome classification and metrics are computed **offline** by reusing `scorer.py` / task-engine logic on the logged traces. One NRPCoreSim run = one trial; sweeps generate per-condition configs that differ only in `EngineTimestep` values.

**Tech Stack:** nrp-core 1.5.1 (installed at `$HOME/.local/nrp`, tag `1.5.1`, commit `25a73b5a`), Python 3.10 (the nrp host interpreter), `nrp_core` / `nrp_client` Python packages, the project's `nrp_bga_sb` package (pydantic ≥2, numpy ≥1.26), pytest ≥8, ruff ≥0.4.

## Global Constraints

- **Python 3.10 only** — fixed by the nrp-core host install (`PROJECT_MEMORY.md §15.1`). Engine scripts run under the nrp host interpreter.
- **Do not modify nrp-core source.** If a change to nrp-core itself ever seems required, stop and report — the nrp-core repo has its own Jira/EBR2 workflow (`§15.5`).
- **No rebuild on the conservative path.** The installed `minimal.cmake` build already ships `python_json` and `py_sim` engines + `NRPDataTransferEngineExecutable`. Do not flip `ENABLE_*` toggles or rebuild unless a task explicitly says so (none in this plan do).
- **Reuse the science layer; never reimplement it.** Engines import and call `CortexEvidenceGenerator`, `BGAdapter`, `ThalamusGate`; scoring reuses `scorer.py` and the task engines. The BG model, cortex ramp, thalamic gate, and scorer are already tested — porting must not fork their logic.
- **Verified nrp-core 1.5.1 API (do not deviate):**
  - Engine script: `from nrp_core.engines.python_json import EngineScript`; subclass `Script(EngineScript)` implementing `initialize(self)`, `runLoop(self, timestep_ns)`, `shutdown(self)`, optional `reset(self)`. Inside: `self._registerDataPack(name)`, `self._setDataPack(name, dict)`, `self._getDataPack(name)`, and `self._time_ns` (engine logical time, nanoseconds).
  - TF: `from nrp_core import *` and `from nrp_core.data.nrp_json import *`; decorate with one or more `@EngineDataPack(keyword=<argname>, id=DataPackIdentifier(<datapack_name>, <engine_name>))` and one `@TransceiverFunction(<target_engine_name>)`; build outputs as `JsonDataPack(<name>, <target_engine>)`, populate `.data[...]`, return a `list`.
  - Config JSON keys: top-level `SimulationName`, `SimulationDescription`, `SimulationTimeout` (seconds), `EngineConfigs` (each: `EngineType` `"python_json"`/`"py_sim"`, `EngineName`, `EngineTimestep` in **seconds**, `PythonFileName`), `DataPackProcessingFunctions` (each: `Name`, `FileName`).
  - Run: `NRPCoreSim -c <config.json> -d <repo_root>`, invoked with cwd = repo root. **The `-d <repo_root>` flag is REQUIRED** (verified in Phase 0): NRPCoreSim changes its internal working directory to the config file's directory, so without `-d` the repo-root-relative `PythonFileName`/`FileName` paths do not resolve. The nrp environment must be sourced first: `source $HOME/.local/nrp/bin/.nrp_env`.
- **numpy is pinned `>=1.26,<2`** (verified in Phase 0): nrp-core 1.5.1's compiled `nrp_json.so` is built against the NumPy 1.x C-ABI; NumPy 2.x crashes the engine subprocess. Do not raise this ceiling until nrp-core is rebuilt against NumPy 2.x.
- **FTILoop timestep rule (`§15.4`):** every `EngineTimestep` must be an integer multiple of the smallest one. The frequency set **{5, 10, 20, 40, 80, 160} Hz** over a 1 ms base step (cortex at 1000 Hz) satisfies this; do **not** introduce 120 Hz (it breaks the rule).
- **Acceptance is categorical, not bit-identical.** The IPC/JSON/process boundary means exact float reproduction across the runtime is not required. Each phase passes when the **categorical empirical signature** survives (e.g. go-success 0.0 at 5 Hz, 1.0 at ≥10 Hz; ablation primary-variable finding). State this in every validation step.
- **Trial parameters reach engines via an env-pointed JSON file**, not via config-dict access (the latter's per-engine API is unverified). The run harness writes `params.json` and exports `NRP_BGA_TRIAL_PARAMS`; engines read that path in `initialize()`. Logged output path is exported as `NRP_BGA_LOG`.
- **Paradigm scope:** go/no-go only, end to end. Other paradigms, the pysim plant, and perturbation TFs are explicitly out of scope (see "Out of Scope / Follow-on").

---

## File Structure

All new NRP assets live under a single top-level `nrp/` package so the binding is self-contained and the legacy pure-Python experiment tree is untouched.

```
nrp/
  __init__.py
  serde.py                 # pydantic schema <-> JsonDataPack-dict adapters
  config_gen.py            # build a simulation_config.json for a (knobs) condition
  run.py                   # write params, set env, invoke NRPCoreSim, parse the log
  engines/
    cortex_engine.py       # time-varying evidence ramp  (EngineTimestep = 1 ms)
    bg_engine.py           # GPR BG model                 (EngineTimestep = emission rate; internal integration sub-step)
    sampler_engine.py      # input-sampling latch         (EngineTimestep = input-sampling rate)   [Phase 3+]
    commitment_engine.py   # commitment latch             (EngineTimestep = commitment rate)        [Phase 4+]
    thalamus_engine.py     # margin gate -> motor command (EngineTimestep = 1 ms)
  tfs/
    tf_cortex_to_bg.py         # Phase 1-2
    tf_cortex_to_sampler.py    # Phase 3+
    tf_sampler_to_bg.py        # Phase 3+
    tf_bg_to_thalamus.py       # Phase 1-2
    tf_bg_to_commitment.py     # Phase 4+
    tf_commitment_to_thalamus.py # Phase 4+
    tf_log.py                  # logger: append decision+motor to NRP_BGA_LOG
  run/                     # gitignored scratch: generated configs, params, logs
experiments/
  nrp_gonogo_sweep.py      # NRP frequency sweep, offline scoring via scorer.py   [Phase 2+]
  nrp_ablation.py          # four-knob ablation through NRPCoreSim                 [Phase 6]
tests/nrp/
  test_serde.py
  test_config_gen.py
  test_nrp_smoke.py        # NRPCoreSim runs; marked @pytest.mark.nrp, deselected by default
  test_nrp_gonogo.py       # categorical signature on go/no-go (marked nrp)
  test_nrp_ablation.py     # four-knob ablation (marked nrp)
```

`tests/nrp/test_nrp_*.py` that launch `NRPCoreSim` are marked `@pytest.mark.nrp` and **deselected by default** (the host CI must not require the nrp runtime), mirroring the existing `@pytest.mark.opensim` convention. Pure-Python tests (`test_serde.py`, `test_config_gen.py`) run in the normal suite.

---

## Phase 0 — Provenance rename + NRP scaffolding

Goal: badge the legacy toy-model artefacts, stand up the `nrp/` package skeleton, install `nrp_bga_sb` into the nrp interpreter, and prove a two-engine NRPCoreSim run works on this machine.

### Task 0.1 — Rename the legacy generated-artefact folders

**Files:**
- Move (git-tracked, genuine analysis — NOT toy output): `results/neuroscience_summary.md` → `docs/neuroscience_summary.md`, `results/open_data_candidates.md` → `docs/open_data_candidates.md`
- Rename (git-tracked): `results/` → `deprecated_toy_prototype_results/`
- Rename (untracked output): `visuals/` → `deprecated_toy_prototype_visuals/`
- Create: `deprecated_toy_prototype_results/_PROVENANCE.md`, `deprecated_toy_prototype_visuals/_PROVENANCE.md`
- Modify: `.gitignore`; `experiments/frequency_sweep.py`, `experiments/stop_signal_sweep.py`, `experiments/change_of_mind_sweep.py`, `experiments/cerebellum_adaptation.py`, `experiments/opensim_gonogo_sweep.py`, `experiments/opensim_cerebellum_sweep.py`, `experiments/generate_report.py`, `experiments/generate_visuals.py`, `src/visuals/data_loader.py`, `src/visuals/assemble.py` (any hardcoded `"results/"` or `"visuals/output"` path)

- [ ] **Step 1: Relocate the genuine analysis files out of the toy-model folder**

These two files are real synthesis / literature work, not toy-model output, so they must NOT be badged as deprecated. Move them to `docs/` first, then repoint any references:

```bash
git mv results/neuroscience_summary.md docs/neuroscience_summary.md
git mv results/open_data_candidates.md docs/open_data_candidates.md
grep -rnI 'results/neuroscience_summary\|results/open_data_candidates' . --include="*.py" --include="*.md" | grep -v docs/superpowers/plans
```
Repoint any hits found (exclude this plan file). Expect few or none.

- [ ] **Step 1b: Rename the tracked results folder**

```bash
git mv results deprecated_toy_prototype_results
```

- [ ] **Step 2: Rename the (untracked) visuals output folder**

`visuals/` has no git-tracked files (its `output/` is gitignored), so a plain move is correct:

```bash
mv visuals deprecated_toy_prototype_visuals 2>/dev/null || mkdir -p deprecated_toy_prototype_visuals
```

- [ ] **Step 3: Write the provenance markers**

Create `deprecated_toy_prototype_results/_PROVENANCE.md`:

```markdown
# Provenance — LEGACY, superseded

The JSON files in this folder were produced by a **pure-Python toy model**
(`src/nrp_bga_sb/` run via `experiments/*.py`) that did **not** use the
nrp-core runtime. They are retained for reference only and are **not** the
authoritative experiment outputs. The authoritative pipeline is the nrp-core
binding under `nrp/` (see `docs/superpowers/plans/2026-06-22-nrp-core-four-knob-binding.md`).

NOTE: the genuine analysis files (`neuroscience_summary.md`, `open_data_candidates.md`)
were relocated to `docs/` in Step 1 — they are NOT toy-model output and are not in this folder.
```

Create `deprecated_toy_prototype_visuals/_PROVENANCE.md`:

```markdown
# Provenance — LEGACY, superseded

Animation frames and videos here were rendered from the pure-Python toy model
(matplotlib / MuJoCo stick-figure pipeline in `src/visuals/`). Retained for
reference only; not produced by the nrp-core runtime.
```

- [ ] **Step 4: Update hardcoded paths**

In every file listed under **Modify**, replace the literal `results/` prefix with `deprecated_toy_prototype_results/` and `visuals/output` with `deprecated_toy_prototype_visuals/output`. Find them first:

```bash
grep -rnI '"results/\|"visuals/output\|results/\|visuals/output' experiments/ src/visuals/
```

Edit each occurrence. Do **not** change the two `.md` filenames inside the folder, only the directory prefix.

- [ ] **Step 5: Update `.gitignore`**

Change the line `visuals/output/` to `deprecated_toy_prototype_visuals/output/`.

- [ ] **Step 6: Verify nothing references the old names**

Run: `grep -rnI '\bresults/\|\bvisuals/output' experiments/ src/ tests/ .gitignore`
Expected: no hits (all rewritten to the `deprecated_toy_prototype_*` prefixes).

- [ ] **Step 7: Verify the legacy report still runs against renamed paths**

Run: `python experiments/generate_report.py 2>&1 | tail -5`
Expected: completes without a `FileNotFoundError`, reading from `deprecated_toy_prototype_results/`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: badge legacy pure-Python toy-model outputs as superseded

Rename results/ -> deprecated_toy_prototype_results/ and
visuals/ -> deprecated_toy_prototype_visuals/, add _PROVENANCE.md markers,
and repoint hardcoded paths in legacy experiment/visuals scripts.
No nrp-core runtime was used to produce these artefacts.

ChangeSet-ID: nrp-legacy-rename"
```

### Task 0.2 — NRP package skeleton, env install, and two-engine smoke

**Files:**
- Create: `nrp/__init__.py`, `nrp/engines/__init__.py`, `nrp/tfs/__init__.py`, `tests/nrp/__init__.py`
- Create: `nrp/engines/_smoke_engine.py`, `nrp/tfs/_smoke_tf.py`, `nrp/configs/_smoke.json`
- Create: `tests/nrp/test_nrp_smoke.py`
- Modify: `pyproject.toml` (register the `nrp` marker), `.gitignore` (add `nrp/run/`)

**Interfaces:**
- Produces: a working `NRPCoreSim -c nrp/configs/_smoke.json` invocation pattern that every later phase reuses.

- [ ] **Step 1: Confirm `nrp_bga_sb` is importable by the nrp interpreter**

The engines will `import nrp_bga_sb`. Install the project editable into the nrp host interpreter (Python 3.10):

```bash
source $HOME/.local/nrp/bin/.nrp_env
python -c "import nrp_core; print('nrp_core OK')"
pip install -e .
python -c "import nrp_bga_sb, nrp_core; print('both import OK')"
```
Expected: `both import OK`. If `import nrp_bga_sb` fails, the engines cannot run — stop and report the interpreter mismatch.

- [ ] **Step 2: Create package markers and gitignore the scratch dir**

```bash
mkdir -p nrp/engines nrp/tfs nrp/configs nrp/run tests/nrp
touch nrp/__init__.py nrp/engines/__init__.py nrp/tfs/__init__.py tests/nrp/__init__.py
echo "nrp/run/" >> .gitignore
```

- [ ] **Step 3: Register the `nrp` pytest marker**

In `pyproject.toml`, under the pytest config, add the marker so `-m "not nrp"` works (mirror the existing `opensim` marker). Add to the `markers` list:

```toml
markers = [
    "opensim: requires the OpenSim Docker image (deselected by default)",
    "nrp: requires the nrp-core runtime / NRPCoreSim (deselected by default)",
]
```
If `addopts` does not already deselect markers, ensure it reads: `addopts = "-m 'not opensim and not nrp'"`.

- [ ] **Step 4: Write the smoke engine and TF (verified template from `examples/tf_exchange`)**

`nrp/engines/_smoke_engine.py`:

```python
"""Phase-0 smoke engine: proves NRPCoreSim launches a python_json engine and
that nrp_bga_sb is importable from inside the engine process."""

from nrp_core.engines.python_json import EngineScript

import nrp_bga_sb  # noqa: F401  -- import smoke: must succeed inside the engine process


class Script(EngineScript):
    def initialize(self):
        self._registerDataPack("tick")
        self._setDataPack("tick", {"t_ns": self._time_ns})

    def runLoop(self, timestep_ns):
        self._setDataPack("tick", {"t_ns": self._time_ns})

    def shutdown(self):
        pass
```

`nrp/tfs/_smoke_tf.py`:

```python
"""Phase-0 smoke TF: reads the tick datapack and appends it to NRP_BGA_LOG."""

import json
import os

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="tick", id=DataPackIdentifier("tick", "smoke"))
@TransceiverFunction("smoke")
def log_tick(tick):
    log_path = os.environ["NRP_BGA_LOG"]
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"t_ns": tick.data["t_ns"]}) + "\n")
    return []
```

- [ ] **Step 5: Write the smoke config**

`nrp/configs/_smoke.json`:

```json
{
    "SimulationName": "nrp_bga_smoke",
    "SimulationDescription": "Phase-0 smoke: one python_json engine, one logging TF.",
    "SimulationTimeout": 0.01,
    "EngineConfigs": [
        {
            "EngineType": "python_json",
            "EngineName": "smoke",
            "EngineTimestep": 0.001,
            "PythonFileName": "nrp/engines/_smoke_engine.py"
        }
    ],
    "DataPackProcessingFunctions": [
        { "Name": "log_tick", "FileName": "nrp/tfs/_smoke_tf.py" }
    ]
}
```

- [ ] **Step 6: Write the smoke test**

`tests/nrp/test_nrp_smoke.py`:

```python
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.nrp
def test_nrpcoresim_runs_python_engine(tmp_path):
    log = tmp_path / "smoke.log"
    env = dict(os.environ, NRP_BGA_LOG=str(log))
    # The nrp env must already be sourced in the calling shell; we invoke the
    # installed binary by name and run from the repo root so relative paths resolve.
    proc = subprocess.run(
        ["NRPCoreSim", "-c", "nrp/configs/_smoke.json"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert len(lines) >= 1               # at least one timestep logged
    assert "t_ns" in lines[0]
```

- [ ] **Step 7: Run the smoke test**

Run:
```bash
source $HOME/.local/nrp/bin/.nrp_env
python -m pytest tests/nrp/test_nrp_smoke.py -m nrp -v
```
Expected: PASS. If NRPCoreSim errors on engine launch, capture `proc.stderr` and report before continuing — every later phase depends on this invocation pattern.

- [ ] **Step 8: Confirm the default suite still excludes nrp tests**

Run: `python -m pytest tests/nrp/ -q`
Expected: `deselected` (the `nrp` marker is excluded by default `addopts`).

- [ ] **Step 9: Commit**

```bash
git add nrp/ tests/nrp/ pyproject.toml .gitignore
git commit -m "feat: nrp/ scaffolding + NRPCoreSim two-engine smoke (Phase 0)

Adds the nrp/ binding package skeleton, registers the deselected-by-default
'nrp' pytest marker, installs the canonical NRPCoreSim invocation pattern,
and proves a python_json engine launches with nrp_bga_sb importable in-process.

ChangeSet-ID: nrp-scaffold"
```

---

## Phase 1 — Conservative 3-engine go/no-go spine (single BG rate)

Goal: cortex → BG → thalamus as three engines with distinct `EngineTimestep`s (already multi-timestep), the BG running at a single rate (input-sampling ≡ emission ≡ commitment collapsed into the one BG `EngineTimestep`). Reproduce the core signature on a single go trial: 5 Hz → miss (no motor command), 40 Hz → hit.

### Task 1.1 — Schema ↔ DataPack serde

**Files:**
- Create: `nrp/serde.py`, `tests/nrp/test_serde.py`

**Interfaces:**
- Produces:
  - `evidence_to_dict(ev: ActionEvidence) -> dict` / `evidence_from_dict(d: dict) -> ActionEvidence`
  - `decision_to_dict(d: BGDecision) -> dict` / `decision_from_dict(d: dict) -> BGDecision`
  - `motor_to_dict(m: MotorCommand) -> dict` / `motor_from_dict(d: dict) -> MotorCommand`

- [ ] **Step 1: Write the failing round-trip test**

`tests/nrp/test_serde.py`:

```python
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, MotorCommand
from nrp.serde import (
    evidence_to_dict, evidence_from_dict,
    decision_to_dict, decision_from_dict,
    motor_to_dict, motor_from_dict,
)


def test_evidence_roundtrip():
    ev = ActionEvidence(sim_time=0.1, trial_id=3, n_channels=2,
                        channel_salience=[0.6, 0.4], stop_signal_present=False)
    assert evidence_from_dict(evidence_to_dict(ev)) == ev


def test_decision_roundtrip():
    d = BGDecision(sim_time=0.1, trial_id=3, selected_channel=0, decision_margin=0.2,
                   suppression_vector=[0.0, 0.3], channel_activations=[0.8, 0.5],
                   selection_latency=0.013)
    assert decision_from_dict(decision_to_dict(d)) == d


def test_motor_roundtrip():
    m = MotorCommand(sim_time=0.1, trial_id=3, command=[1.0, 0.0],
                     gate_state="open", gate_gain=1.0)
    assert motor_from_dict(motor_to_dict(m)) == m
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/nrp/test_serde.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nrp.serde'`.

- [ ] **Step 3: Implement `nrp/serde.py`**

```python
"""Adapters between pydantic schemas (nrp_bga_sb.schemas) and the plain-dict
payloads carried by nrp-core JsonDataPacks. Kept trivial on purpose: pydantic
model_dump / construction is the single source of truth for field names."""

from nrp_bga_sb.schemas import ActionEvidence, BGDecision, MotorCommand


def evidence_to_dict(ev: ActionEvidence) -> dict:
    return ev.model_dump()


def evidence_from_dict(d: dict) -> ActionEvidence:
    return ActionEvidence(**d)


def decision_to_dict(d: BGDecision) -> dict:
    return d.model_dump()


def decision_from_dict(d: dict) -> BGDecision:
    return BGDecision(**d)


def motor_to_dict(m: MotorCommand) -> dict:
    return m.model_dump()


def motor_from_dict(d: dict) -> MotorCommand:
    return MotorCommand(**d)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/nrp/test_serde.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add nrp/serde.py tests/nrp/test_serde.py
git commit -m "feat: schema<->JsonDataPack serde adapters (Task 1.1)

ChangeSet-ID: nrp-serde"
```

### Task 1.2 — Cortex engine

**Files:**
- Create: `nrp/engines/cortex_engine.py`

**Interfaces:**
- Reads `NRP_BGA_TRIAL_PARAMS` (JSON file with keys `trial_id:int`, `seed:int`, `cue_identity:str`). Registers/sets datapack `evidence` on engine `cortex` with the dict form of `ActionEvidence`.
- Produces datapack name `"evidence"` (engine `"cortex"`).

- [ ] **Step 1: Write the engine**

```python
"""Cortex engine: emits a time-varying cortical-salience ramp as the `evidence`
datapack. Delegates to the validated CortexEvidenceGenerator; reads per-trial
parameters from the env-pointed JSON file (NRP_BGA_TRIAL_PARAMS)."""

import json
import os

from nrp_core.engines.python_json import EngineScript

from nrp_bga_sb.cortex import CortexConfig, CortexEvidenceGenerator
from nrp_bga_sb.schemas import TrialLog
from nrp.serde import evidence_to_dict


class Script(EngineScript):
    def initialize(self):
        with open(os.environ["NRP_BGA_TRIAL_PARAMS"]) as fh:
            params = json.load(fh)
        # Build a minimal but complete TrialLog so the generator has every
        # required field regardless of which ones it reads.
        self._trial = TrialLog(
            trial_id=params["trial_id"],
            seed=params["seed"],
            task_type="go_nogo",
            cue_identity=params["cue_identity"],
            cue_onset_time=0.0,
        )
        self._cortex = CortexEvidenceGenerator(CortexConfig())
        self._registerDataPack("evidence")
        self._emit(0.0)

    def runLoop(self, timestep_ns):
        # _time_ns is engine logical time; the generator expects elapsed ms.
        self._emit(self._time_ns / 1.0e6)

    def _emit(self, elapsed_ms: float):
        ev = self._cortex(self._trial, elapsed_ms)
        self._setDataPack("evidence", evidence_to_dict(ev))

    def shutdown(self):
        pass
```

- [ ] **Step 2: No standalone unit test** — the engine only runs inside NRPCoreSim. It is exercised by the Task 1.6 integration test. Verify it at least imports under the nrp interpreter:

Run:
```bash
source $HOME/.local/nrp/bin/.nrp_env
python -c "import ast; ast.parse(open('nrp/engines/cortex_engine.py').read()); print('parse OK')"
```
Expected: `parse OK`.

- [ ] **Step 3: Commit**

```bash
git add nrp/engines/cortex_engine.py
git commit -m "feat: cortex python_json engine (Task 1.2)

ChangeSet-ID: nrp-cortex-engine"
```

### Task 1.3 — BG engine (single-rate)

**Files:**
- Create: `nrp/engines/bg_engine.py`

**Interfaces:**
- Reads datapack `sampled_evidence` (set on engine `bg` by a TF). Registers/sets datapack `decision` (dict form of `BGDecision`) on engine `bg`.
- Produces datapack name `"decision"` (engine `"bg"`).

- [ ] **Step 1: Write the engine**

```python
"""BG engine (single-rate, Phase 1): runs the GPR BG model on the most recent
sampled evidence and emits a `decision` datapack. Delegates to BGAdapter.

In Phase 1 input-sampling, integration, and emission all coincide with this
engine's EngineTimestep; later phases split sampling (Phase 3), commitment
(Phase 4), and internal integration (Phase 5) out into their own rates."""

import json
import os

from nrp_core.engines.python_json import EngineScript

from nrp_bga_sb.bg_model import BGAdapter, BGModelConfig
from nrp_bga_sb.schemas import TrialLog
from nrp.serde import evidence_from_dict, decision_to_dict


class Script(EngineScript):
    def initialize(self):
        with open(os.environ["NRP_BGA_TRIAL_PARAMS"]) as fh:
            params = json.load(fh)
        self._trial = TrialLog(
            trial_id=params["trial_id"], seed=params["seed"],
            task_type="go_nogo", cue_identity=params["cue_identity"],
            cue_onset_time=0.0,
        )
        self._bg = BGAdapter(BGModelConfig())
        # The TF writes incoming evidence here; register so _getDataPack works
        # even before the first TF delivery.
        self._registerDataPack("sampled_evidence")
        self._registerDataPack("decision")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("sampled_evidence")
        # Trigger: no evidence delivered yet (first ticks before TF fires).
        # Why: BGAdapter needs a populated ActionEvidence; skip until present.
        # Outcome: `decision` keeps its previous value until evidence arrives.
        if not raw or "channel_salience" not in raw:
            return
        evidence = evidence_from_dict(raw)
        decision = self._bg(self._trial, evidence)
        self._setDataPack("decision", decision_to_dict(decision))

    def shutdown(self):
        pass
```

- [ ] **Step 2: Parse check**

Run: `source $HOME/.local/nrp/bin/.nrp_env && python -c "import ast; ast.parse(open('nrp/engines/bg_engine.py').read()); print('parse OK')"`
Expected: `parse OK`.

- [ ] **Step 3: Commit**

```bash
git add nrp/engines/bg_engine.py
git commit -m "feat: BG python_json engine, single-rate (Task 1.3)

ChangeSet-ID: nrp-bg-engine"
```

### Task 1.4 — Thalamus engine

**Files:**
- Create: `nrp/engines/thalamus_engine.py`

**Interfaces:**
- Reads datapack `committed_decision` (set on engine `thalamus` by a TF). Registers/sets datapack `motor` (dict form of `MotorCommand`) on engine `thalamus`.
- Produces datapack name `"motor"` (engine `"thalamus"`).

- [ ] **Step 1: Write the engine**

```python
"""Thalamus engine: gates the committed BG decision into a motor command.
Delegates to ThalamusGate. The incoming datapack is named `committed_decision`
so the same engine serves Phase 1-3 (fed straight from BG) and Phase 4+ (fed
from the commitment engine) without change."""

from nrp_core.engines.python_json import EngineScript

from nrp_bga_sb.thalamus import ThalamusConfig, ThalamusGate
from nrp.serde import decision_from_dict, motor_to_dict


class Script(EngineScript):
    def initialize(self):
        self._thalamus = ThalamusGate(ThalamusConfig())
        self._registerDataPack("committed_decision")
        self._registerDataPack("motor")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("committed_decision")
        if not raw or "selected_channel" not in raw:
            return
        motor = self._thalamus(decision_from_dict(raw))
        self._setDataPack("motor", motor_to_dict(motor))

    def shutdown(self):
        pass
```

- [ ] **Step 2: Parse check** — `python -c "import ast; ast.parse(open('nrp/engines/thalamus_engine.py').read())"`; Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add nrp/engines/thalamus_engine.py
git commit -m "feat: thalamus python_json engine (Task 1.4)

ChangeSet-ID: nrp-thalamus-engine"
```

### Task 1.5 — TFs: cortex→bg, bg→thalamus, logger

**Files:**
- Create: `nrp/tfs/tf_cortex_to_bg.py`, `nrp/tfs/tf_bg_to_thalamus.py`, `nrp/tfs/tf_log.py`

**Interfaces:**
- `tf_cortex_to_bg`: consumes `evidence`@`cortex`, targets `bg`, returns `[JsonDataPack("sampled_evidence","bg")]`.
- `tf_bg_to_thalamus`: consumes `decision`@`bg`, targets `thalamus`, returns `[JsonDataPack("committed_decision","thalamus")]`.
- `tf_log`: consumes `decision`@`bg` and `motor`@`thalamus`, appends one JSON line per step to `NRP_BGA_LOG`.

- [ ] **Step 1: Write `nrp/tfs/tf_cortex_to_bg.py`**

```python
"""Phase 1-2 link: forward cortical evidence to the BG engine each step. In
Phase 3 this is replaced by tf_cortex_to_sampler + tf_sampler_to_bg."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="evidence", id=DataPackIdentifier("evidence", "cortex"))
@TransceiverFunction("bg")
def cortex_to_bg(evidence):
    out = JsonDataPack("sampled_evidence", "bg")
    for k, v in evidence.data.items():
        out.data[k] = v
    return [out]
```

- [ ] **Step 2: Write `nrp/tfs/tf_bg_to_thalamus.py`**

```python
"""Phase 1-3 link: forward the BG decision to the thalamus as the committed
decision. In Phase 4 this is replaced by tf_bg_to_commitment +
tf_commitment_to_thalamus."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="decision", id=DataPackIdentifier("decision", "bg"))
@TransceiverFunction("thalamus")
def bg_to_thalamus(decision):
    out = JsonDataPack("committed_decision", "thalamus")
    for k, v in decision.data.items():
        out.data[k] = v
    return [out]
```

- [ ] **Step 3: Write `nrp/tfs/tf_log.py`**

```python
"""Logger TF: append the current decision and motor command to NRP_BGA_LOG as
one JSON object per FTILoop step. Off-loop persistence — the trace is scored
offline by the existing scorer/task-engine logic."""

import json
import os

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="decision", id=DataPackIdentifier("decision", "bg"))
@EngineDataPack(keyword="motor", id=DataPackIdentifier("motor", "thalamus"))
@TransceiverFunction("thalamus")
def log_step(decision, motor):
    record = {
        "decision": dict(decision.data) if decision.data else None,
        "motor": dict(motor.data) if motor.data else None,
    }
    with open(os.environ["NRP_BGA_LOG"], "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return []
```

- [ ] **Step 4: Parse-check all three**

Run: `for f in nrp/tfs/tf_cortex_to_bg.py nrp/tfs/tf_bg_to_thalamus.py nrp/tfs/tf_log.py; do python -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f"; done; echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add nrp/tfs/tf_cortex_to_bg.py nrp/tfs/tf_bg_to_thalamus.py nrp/tfs/tf_log.py
git commit -m "feat: cortex->bg, bg->thalamus, and logger TFs (Task 1.5)

ChangeSet-ID: nrp-phase1-tfs"
```

### Task 1.6 — Phase-1 config + run harness + single-trial validation

**Files:**
- Create: `nrp/config_gen.py`, `nrp/run.py`, `tests/nrp/test_config_gen.py`, `tests/nrp/test_nrp_gonogo.py`

**Interfaces:**
- `build_config(bg_hz: float, *, name: str = "gonogo") -> dict` — Phase-1 3-engine config; cortex at 1000 Hz, thalamus at 1000 Hz, bg at `bg_hz`. `SimulationTimeout = 0.3`.
- `run_trial(config: dict, params: dict, run_dir: Path) -> list[dict]` — writes `config.json`, `params.json`, sets `NRP_BGA_TRIAL_PARAMS`/`NRP_BGA_LOG`, runs NRPCoreSim from repo root, returns parsed log records.

- [ ] **Step 1: Write the failing config-gen test**

`tests/nrp/test_config_gen.py`:

```python
from nrp.config_gen import build_config


def test_build_config_sets_bg_timestep():
    cfg = build_config(40.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert set(engines) == {"cortex", "bg", "thalamus"}
    assert engines["bg"]["EngineTimestep"] == 1.0 / 40.0
    assert engines["cortex"]["EngineTimestep"] == 0.001
    assert cfg["SimulationTimeout"] == 0.3


def test_build_config_5hz_period():
    cfg = build_config(5.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert engines["bg"]["EngineTimestep"] == 0.2   # 5 Hz -> 200 ms period
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/nrp/test_config_gen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nrp.config_gen'`.

- [ ] **Step 3: Implement `nrp/config_gen.py`**

```python
"""Build NRPCoreSim simulation configs for the go/no-go binding. Phase 1 wires
three engines (cortex, bg, thalamus). The BG frequency is the single swept knob;
later phases add sampler/commitment engines and additional EngineTimesteps."""

from __future__ import annotations

CORTEX_HZ = 1000.0      # finest resolution -> sets the FTILoop base step (1 ms)
THALAMUS_HZ = 1000.0


def build_config(bg_hz: float, *, name: str = "gonogo") -> dict:
    return {
        "SimulationName": name,
        "SimulationDescription": f"go/no-go BG binding, bg={bg_hz} Hz (Phase 1).",
        "SimulationTimeout": 0.3,   # 300 ms: covers the 200 ms accumulation window
        "EngineConfigs": [
            {"EngineType": "python_json", "EngineName": "cortex",
             "EngineTimestep": 1.0 / CORTEX_HZ,
             "PythonFileName": "nrp/engines/cortex_engine.py"},
            {"EngineType": "python_json", "EngineName": "bg",
             "EngineTimestep": 1.0 / bg_hz,
             "PythonFileName": "nrp/engines/bg_engine.py"},
            {"EngineType": "python_json", "EngineName": "thalamus",
             "EngineTimestep": 1.0 / THALAMUS_HZ,
             "PythonFileName": "nrp/engines/thalamus_engine.py"},
        ],
        "DataPackProcessingFunctions": [
            {"Name": "cortex_to_bg", "FileName": "nrp/tfs/tf_cortex_to_bg.py"},
            {"Name": "bg_to_thalamus", "FileName": "nrp/tfs/tf_bg_to_thalamus.py"},
            {"Name": "log_step", "FileName": "nrp/tfs/tf_log.py"},
        ],
    }
```

- [ ] **Step 4: Run the config-gen test to confirm it passes**

Run: `python -m pytest tests/nrp/test_config_gen.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Implement `nrp/run.py`**

```python
"""Drive a single NRPCoreSim trial: materialise config + params, set the env
contract, run the binary from the repo root, and parse the JSON-lines log."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_trial(config: dict, params: dict, run_dir: Path) -> list[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    params_path = run_dir / "params.json"
    log_path = run_dir / "trace.jsonl"
    config_path.write_text(json.dumps(config, indent=2))
    params_path.write_text(json.dumps(params))
    if log_path.exists():
        log_path.unlink()

    env = dict(os.environ,
               NRP_BGA_TRIAL_PARAMS=str(params_path),
               NRP_BGA_LOG=str(log_path))
    # -d REPO is REQUIRED: NRPCoreSim cwd's to the config's directory, so the
    # repo-root-relative PythonFileName/FileName paths only resolve with an
    # explicit experiment root (verified Phase 0).
    # Pass the config path ABSOLUTE (not relative_to(REPO)): run_dir is often a
    # pytest tmp_path OUTSIDE the repo, where relative_to(REPO) raises ValueError.
    # NRPCoreSim resolves an absolute -c regardless of the -d-set CWD (verified Task 1.6).
    proc = subprocess.run(
        ["NRPCoreSim", "-c", str(config_path), "-d", str(REPO)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"NRPCoreSim failed (rc={proc.returncode}):\n{proc.stderr}")
    if not log_path.exists():
        return []
    return [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
```

- [ ] **Step 6: Write the single-trial signature test**

`tests/nrp/test_nrp_gonogo.py`:

```python
import os
from pathlib import Path

import pytest

from nrp.config_gen import build_config
from nrp.run import run_trial

REPO = Path(__file__).resolve().parents[2]


def _go_params():
    return {"trial_id": 0, "seed": 0, "cue_identity": "go"}


def _motor_released(trace):
    """A motor command with an open/partial gate appears somewhere in the trace."""
    for rec in trace:
        m = rec.get("motor")
        if m and m.get("gate_state") in ("open", "partial") and any(m["command"]):
            return True
    return False


@pytest.mark.nrp
def test_high_freq_go_trial_releases_motor(tmp_path):
    trace = run_trial(build_config(40.0), _go_params(), tmp_path / "hi")
    assert _motor_released(trace), "40 Hz go trial should release a motor command"


@pytest.mark.nrp
def test_low_freq_go_trial_misses(tmp_path):
    trace = run_trial(build_config(5.0), _go_params(), tmp_path / "lo")
    assert not _motor_released(trace), "5 Hz go trial should NOT release a motor command"
```

- [ ] **Step 7: Run the single-trial validation under the nrp env**

Run:
```bash
source $HOME/.local/nrp/bin/.nrp_env
python -m pytest tests/nrp/test_nrp_gonogo.py -m nrp -v
```
Expected: 2 PASS — 40 Hz releases a motor command, 5 Hz does not. This is the categorical signature surviving the runtime.

If the 5 Hz trial unexpectedly releases (or 40 Hz misses): debug with systematic-debugging. The most likely cause is the BG `EngineTimestep` sampling cortex evidence at a tick where the ramp has already peaked — confirm via the logged `decision.channel_activations` against the pure-Python `closed_loop.py` expectation (5 Hz samples only at t=0 and t=0.2). Do not paper over a mismatch; the science check is the point of the phase.

- [ ] **Step 8: Commit**

```bash
git add nrp/config_gen.py nrp/run.py tests/nrp/test_config_gen.py tests/nrp/test_nrp_gonogo.py
git commit -m "feat: Phase-1 config-gen, run harness, single-trial signature (Task 1.6)

3-engine go/no-go spine through NRPCoreSim: 40 Hz releases motor, 5 Hz misses.
Categorical BG-frequency signature reproduced through the FTILoop.

ChangeSet-ID: nrp-phase1-spine"
```

---

## Phase 2 — Frequency-sweep harness + offline scoring

Goal: run the full frequency set through NRPCoreSim across seeds and score with the **existing** `scorer.py` / go/no-go classification, reproducing the categorical go-success curve (0.0 at 5 Hz, 1.0 at ≥10 Hz) that the pure-Python sweep found.

### Task 2.1 — Offline trace → outcome adapter

**Files:**
- Create: `nrp/score.py`, `tests/nrp/test_score.py`

**Interfaces:**
- `trace_to_outcome(trace: list[dict]) -> dict` — returns `{"motor_released": bool, "selected_channel": int, "first_release_time": float | None}` derived from the logged decision/motor records. Pure function, unit-testable without the runtime.

- [ ] **Step 1: Write the failing test** (synthetic traces, no NRPCoreSim)

`tests/nrp/test_score.py`:

```python
from nrp.score import trace_to_outcome


def test_outcome_released():
    trace = [
        {"decision": {"selected_channel": 0, "sim_time": 0.1}, "motor": None},
        {"decision": {"selected_channel": 0, "sim_time": 0.12},
         "motor": {"command": [1.0, 0.0], "gate_state": "open", "gate_gain": 1.0,
                   "sim_time": 0.12}},
    ]
    out = trace_to_outcome(trace)
    assert out["motor_released"] is True
    assert out["selected_channel"] == 0
    assert out["first_release_time"] == 0.12


def test_outcome_missed():
    trace = [{"decision": {"selected_channel": -1, "sim_time": 0.1}, "motor": None}]
    out = trace_to_outcome(trace)
    assert out["motor_released"] is False
    assert out["first_release_time"] is None
```

- [ ] **Step 2: Run it to confirm it fails** — `python -m pytest tests/nrp/test_score.py -v`; Expected: FAIL (`No module named 'nrp.score'`).

- [ ] **Step 3: Implement `nrp/score.py`**

```python
"""Offline scoring of an NRPCoreSim trace. The runtime produces decision/motor
records; outcome classification reuses the same notion of "released" as the
pure-Python ThalamusGate (open/partial gate with a non-zero command)."""

from __future__ import annotations


def trace_to_outcome(trace: list[dict]) -> dict:
    motor_released = False
    first_release_time = None
    selected_channel = -1
    for rec in trace:
        d = rec.get("decision")
        if d and d.get("selected_channel", -1) >= 0:
            selected_channel = d["selected_channel"]
        m = rec.get("motor")
        if m and m.get("gate_state") in ("open", "partial") and any(m.get("command", [])):
            if not motor_released:
                first_release_time = m.get("sim_time")
            motor_released = True
    return {
        "motor_released": motor_released,
        "selected_channel": selected_channel,
        "first_release_time": first_release_time,
    }
```

- [ ] **Step 4: Run it to confirm it passes** — `python -m pytest tests/nrp/test_score.py -v`; Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add nrp/score.py tests/nrp/test_score.py
git commit -m "feat: offline trace->outcome adapter (Task 2.1)

ChangeSet-ID: nrp-score"
```

### Task 2.2 — Sweep experiment + categorical validation

**Files:**
- Create: `experiments/nrp_gonogo_sweep.py`
- Modify: `tests/nrp/test_nrp_gonogo.py` (add the sweep signature test)

**Interfaces:**
- `run_sweep(frequencies: list[float], n_seeds: int, run_root: Path) -> dict[float, float]` — returns `{freq_hz: go_success_rate}` over `n_seeds` go trials per frequency, each via `run_trial`.

- [ ] **Step 1: Implement `experiments/nrp_gonogo_sweep.py`**

```python
"""NRP go/no-go frequency sweep. Runs one NRPCoreSim trial per (frequency, seed),
scores each trace offline, and reports go-success rate vs BG frequency. Reuses
nrp.config_gen / nrp.run / nrp.score; no science is reimplemented here."""

from __future__ import annotations

import json
from pathlib import Path

from nrp.config_gen import build_config
from nrp.run import run_trial
from nrp.score import trace_to_outcome

FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]


def run_sweep(frequencies: list[float], n_seeds: int, run_root: Path) -> dict[float, float]:
    rates: dict[float, float] = {}
    for hz in frequencies:
        released = 0
        for seed in range(n_seeds):
            params = {"trial_id": seed, "seed": seed, "cue_identity": "go"}
            trace = run_trial(build_config(hz), params, run_root / f"{hz}hz_s{seed}")
            if trace_to_outcome(trace)["motor_released"]:
                released += 1
        rates[hz] = released / n_seeds
    return rates


if __name__ == "__main__":
    out_root = Path("nrp/run/gonogo_sweep")
    rates = run_sweep(FREQUENCIES_HZ, n_seeds=5, run_root=out_root)
    result_path = Path("nrp/run/nrp_gonogo_sweep.json")
    result_path.write_text(json.dumps(rates, indent=2))
    print("go-success rate vs BG frequency (Hz):")
    for hz in FREQUENCIES_HZ:
        print(f"  {hz:6.1f} Hz : {rates[hz]:.3f}")
    print(f"saved -> {result_path}")
```

- [ ] **Step 2: Add the sweep signature test** to `tests/nrp/test_nrp_gonogo.py`:

```python
@pytest.mark.nrp
def test_sweep_categorical_signature(tmp_path):
    from experiments.nrp_gonogo_sweep import run_sweep, FREQUENCIES_HZ
    rates = run_sweep(FREQUENCIES_HZ, n_seeds=2, run_root=tmp_path / "sweep")
    assert rates[5.0] == 0.0                       # 5 Hz: all miss
    assert all(rates[hz] == 1.0 for hz in (10.0, 20.0, 40.0, 80.0, 160.0))
```

- [ ] **Step 3: Run the sweep validation under the nrp env**

Run:
```bash
source $HOME/.local/nrp/bin/.nrp_env
python -m pytest tests/nrp/test_nrp_gonogo.py::test_sweep_categorical_signature -m nrp -v
```
Expected: PASS — 0.0 at 5 Hz, 1.0 at ≥10 Hz, matching the pure-Python frequency sweep (PROJECT_MEMORY §1, Phase 5).

- [ ] **Step 4: Run the experiment end-to-end and eyeball the report**

Run: `source $HOME/.local/nrp/bin/.nrp_env && python experiments/nrp_gonogo_sweep.py`
Expected: prints the rate table (5 Hz → 0.000, ≥10 Hz → 1.000) and saves `nrp/run/nrp_gonogo_sweep.json`.

- [ ] **Step 5: Commit**

```bash
git add experiments/nrp_gonogo_sweep.py tests/nrp/test_nrp_gonogo.py
git commit -m "feat: NRP go/no-go frequency sweep + categorical validation (Task 2.2)

Full {5,10,20,40,80,160} Hz sweep through NRPCoreSim reproduces the pure-Python
signature: go-success 0.0 at 5 Hz, 1.0 at >=10 Hz. Scoring reuses nrp.score.

ChangeSet-ID: nrp-phase2-sweep"
```

---

## Phase 3 — Dissociate input sampling (Sampler engine): knob 1 ≠ knob 3

Goal: split input-sampling frequency away from output-emission frequency. A new `sampler` engine runs at the input-sampling rate and latches the latest cortex evidence; the BG engine runs at the (now independent) emission rate. Two distinct BG-path `EngineTimestep`s.

### Task 3.1 — Sampler engine + sampling TFs

**Files:**
- Create: `nrp/engines/sampler_engine.py`, `nrp/tfs/tf_cortex_to_sampler.py`, `nrp/tfs/tf_sampler_to_bg.py`

**Interfaces:**
- `sampler` engine: reads `incoming_evidence` (from cortex TF), latches it, sets `sampled_evidence`.
- `tf_cortex_to_sampler`: `evidence`@`cortex` → `[JsonDataPack("incoming_evidence","sampler")]`.
- `tf_sampler_to_bg`: `sampled_evidence`@`sampler` → `[JsonDataPack("sampled_evidence","bg")]`.

- [ ] **Step 1: Write `nrp/engines/sampler_engine.py`**

```python
"""Sampler engine: realises the BG INPUT-SAMPLING frequency. It runs at its own
(slow) EngineTimestep and latches the most recent cortical evidence into
`sampled_evidence`. Between its steps the BG engine sees stale evidence -- this
is exactly the mechanism that makes low input-sampling rates miss the evidence
ramp peak."""

from nrp_core.engines.python_json import EngineScript


class Script(EngineScript):
    def initialize(self):
        self._registerDataPack("incoming_evidence")
        self._registerDataPack("sampled_evidence")

    def runLoop(self, timestep_ns):
        latest = self._getDataPack("incoming_evidence")
        if latest and "channel_salience" in latest:
            # Latch: copy the latest cortical evidence at the sampling rate.
            self._setDataPack("sampled_evidence", dict(latest))

    def shutdown(self):
        pass
```

- [ ] **Step 2: Write `nrp/tfs/tf_cortex_to_sampler.py`**

```python
from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="evidence", id=DataPackIdentifier("evidence", "cortex"))
@TransceiverFunction("sampler")
def cortex_to_sampler(evidence):
    out = JsonDataPack("incoming_evidence", "sampler")
    for k, v in evidence.data.items():
        out.data[k] = v
    return [out]
```

- [ ] **Step 3: Write `nrp/tfs/tf_sampler_to_bg.py`**

```python
from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="sampled", id=DataPackIdentifier("sampled_evidence", "sampler"))
@TransceiverFunction("bg")
def sampler_to_bg(sampled):
    out = JsonDataPack("sampled_evidence", "bg")
    for k, v in sampled.data.items():
        out.data[k] = v
    return [out]
```

- [ ] **Step 4: Parse-check all three** — Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add nrp/engines/sampler_engine.py nrp/tfs/tf_cortex_to_sampler.py nrp/tfs/tf_sampler_to_bg.py
git commit -m "feat: sampler engine + sampling TFs (Task 3.1)

ChangeSet-ID: nrp-sampler"
```

### Task 3.2 — Config-gen v2 with independent sampling/emission knobs

**Files:**
- Modify: `nrp/config_gen.py`, `tests/nrp/test_config_gen.py`

**Interfaces:**
- Add `build_config_sampled(*, input_sampling_hz: float, output_emission_hz: float, name: str = "gonogo_s") -> dict` — four engines (cortex, sampler, bg, thalamus) with `sampler.EngineTimestep = 1/input_sampling_hz` and `bg.EngineTimestep = 1/output_emission_hz`; uses `tf_cortex_to_sampler` + `tf_sampler_to_bg` instead of `tf_cortex_to_bg`.

- [ ] **Step 1: Write the failing test** (append to `tests/nrp/test_config_gen.py`):

```python
def test_build_config_sampled_independent_knobs():
    from nrp.config_gen import build_config_sampled
    cfg = build_config_sampled(input_sampling_hz=5.0, output_emission_hz=40.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert set(engines) == {"cortex", "sampler", "bg", "thalamus"}
    assert engines["sampler"]["EngineTimestep"] == 0.2     # 5 Hz sampling
    assert engines["bg"]["EngineTimestep"] == 1.0 / 40.0   # 40 Hz emission
    tf_names = {t["Name"] for t in cfg["DataPackProcessingFunctions"]}
    assert {"cortex_to_sampler", "sampler_to_bg"} <= tf_names
    assert "cortex_to_bg" not in tf_names
```

- [ ] **Step 2: Run it to confirm it fails** — Expected: FAIL (`build_config_sampled` undefined).

- [ ] **Step 3: Implement `build_config_sampled`** in `nrp/config_gen.py`:

```python
def build_config_sampled(*, input_sampling_hz: float, output_emission_hz: float,
                         name: str = "gonogo_s") -> dict:
    return {
        "SimulationName": name,
        "SimulationDescription": (
            f"go/no-go, input_sampling={input_sampling_hz} Hz, "
            f"output_emission={output_emission_hz} Hz (Phase 3)."),
        "SimulationTimeout": 0.3,
        "EngineConfigs": [
            {"EngineType": "python_json", "EngineName": "cortex",
             "EngineTimestep": 1.0 / CORTEX_HZ,
             "PythonFileName": "nrp/engines/cortex_engine.py"},
            {"EngineType": "python_json", "EngineName": "sampler",
             "EngineTimestep": 1.0 / input_sampling_hz,
             "PythonFileName": "nrp/engines/sampler_engine.py"},
            {"EngineType": "python_json", "EngineName": "bg",
             "EngineTimestep": 1.0 / output_emission_hz,
             "PythonFileName": "nrp/engines/bg_engine.py"},
            {"EngineType": "python_json", "EngineName": "thalamus",
             "EngineTimestep": 1.0 / THALAMUS_HZ,
             "PythonFileName": "nrp/engines/thalamus_engine.py"},
        ],
        "DataPackProcessingFunctions": [
            {"Name": "cortex_to_sampler", "FileName": "nrp/tfs/tf_cortex_to_sampler.py"},
            {"Name": "sampler_to_bg", "FileName": "nrp/tfs/tf_sampler_to_bg.py"},
            {"Name": "bg_to_thalamus", "FileName": "nrp/tfs/tf_bg_to_thalamus.py"},
            {"Name": "log_step", "FileName": "nrp/tfs/tf_log.py"},
        ],
    }
```

- [ ] **Step 4: Run it to confirm it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nrp/config_gen.py tests/nrp/test_config_gen.py
git commit -m "feat: sampled config with independent sampling/emission knobs (Task 3.2)

ChangeSet-ID: nrp-config-sampled"
```

### Task 3.3 — Dissociation validation (sampling is the miss-driver)

**Files:**
- Create: `tests/nrp/test_nrp_dissociation.py`

- [ ] **Step 1: Write the dissociation test**

`tests/nrp/test_nrp_dissociation.py`:

```python
import pytest

from nrp.config_gen import build_config_sampled
from nrp.run import run_trial
from nrp.score import trace_to_outcome


def _go():
    return {"trial_id": 0, "seed": 0, "cue_identity": "go"}


@pytest.mark.nrp
def test_slow_sampling_misses_despite_fast_emission(tmp_path):
    # Low input-sampling (5 Hz) starves the BG even when emission is fast (160 Hz):
    # the sampler latches evidence only at t=0 and t=0.2, so the BG integrates
    # neutral early evidence. Sampling, not emission, drives the miss.
    cfg = build_config_sampled(input_sampling_hz=5.0, output_emission_hz=160.0)
    trace = run_trial(cfg, _go(), tmp_path / "slow_sample")
    assert trace_to_outcome(trace)["motor_released"] is False


@pytest.mark.nrp
def test_fast_sampling_hits_even_with_slow_emission(tmp_path):
    # Fast sampling (160 Hz) with slow emission (10 Hz): the BG sees the ramp peak;
    # emission still publishes in time within the 300 ms window -> release.
    cfg = build_config_sampled(input_sampling_hz=160.0, output_emission_hz=10.0)
    trace = run_trial(cfg, _go(), tmp_path / "fast_sample")
    assert trace_to_outcome(trace)["motor_released"] is True
```

- [ ] **Step 2: Run under the nrp env**

Run: `source $HOME/.local/nrp/bin/.nrp_env && python -m pytest tests/nrp/test_nrp_dissociation.py -m nrp -v`
Expected: 2 PASS — confirms input-sampling rate is dissociable from emission rate and is the miss-driver, matching the §15.4 mechanism description. If both knobs still move together, the sampler latch or TF wiring is wrong — debug before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/nrp/test_nrp_dissociation.py
git commit -m "feat: validate sampling/emission dissociation through NRPCoreSim (Task 3.3)

ChangeSet-ID: nrp-phase3-dissociation"
```

---

## Phase 4 — Commitment engine: knob 4

Goal: add a dedicated `commitment` engine that latches the raw BG decision into the published `committed_decision` only at its own rate. The thalamus now consumes from the commitment engine.

### Task 4.1 — Commitment engine + commitment TFs

**Files:**
- Create: `nrp/engines/commitment_engine.py`, `nrp/tfs/tf_bg_to_commitment.py`, `nrp/tfs/tf_commitment_to_thalamus.py`

**Interfaces:**
- `commitment` engine: reads `raw_decision`, latches into `committed_decision`.
- `tf_bg_to_commitment`: `decision`@`bg` → `[JsonDataPack("raw_decision","commitment")]`.
- `tf_commitment_to_thalamus`: `committed_decision`@`commitment` → `[JsonDataPack("committed_decision","thalamus")]`.

- [ ] **Step 1: Write `nrp/engines/commitment_engine.py`**

```python
"""Commitment engine: realises the BG DECISION-COMMITMENT update frequency
(§15.4 -- not a built-in nrp-core concept, so modelled as its own engine). It
latches the latest raw BG decision into `committed_decision` only at its own
(slow) EngineTimestep; between latches, downstream sees the previously committed
decision."""

from nrp_core.engines.python_json import EngineScript


class Script(EngineScript):
    def initialize(self):
        self._registerDataPack("raw_decision")
        self._registerDataPack("committed_decision")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("raw_decision")
        if raw and "selected_channel" in raw:
            self._setDataPack("committed_decision", dict(raw))

    def shutdown(self):
        pass
```

- [ ] **Step 2: Write `nrp/tfs/tf_bg_to_commitment.py`**

```python
from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="decision", id=DataPackIdentifier("decision", "bg"))
@TransceiverFunction("commitment")
def bg_to_commitment(decision):
    out = JsonDataPack("raw_decision", "commitment")
    for k, v in decision.data.items():
        out.data[k] = v
    return [out]
```

- [ ] **Step 3: Write `nrp/tfs/tf_commitment_to_thalamus.py`**

```python
from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="committed",
                id=DataPackIdentifier("committed_decision", "commitment"))
@TransceiverFunction("thalamus")
def commitment_to_thalamus(committed):
    out = JsonDataPack("committed_decision", "thalamus")
    for k, v in committed.data.items():
        out.data[k] = v
    return [out]
```

- [ ] **Step 4: Parse-check all three** — Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add nrp/engines/commitment_engine.py nrp/tfs/tf_bg_to_commitment.py nrp/tfs/tf_commitment_to_thalamus.py
git commit -m "feat: commitment engine + commitment TFs (Task 4.1)

ChangeSet-ID: nrp-commitment"
```

### Task 4.2 — Config-gen with sampler + commitment (three independent rates)

**Files:**
- Modify: `nrp/config_gen.py`, `tests/nrp/test_config_gen.py`

**Interfaces:**
- Add `build_config_committed(*, input_sampling_hz, output_emission_hz, commitment_hz, name="gonogo_c") -> dict` — five engines (cortex, sampler, bg, commitment, thalamus); thalamus fed from commitment.

- [ ] **Step 1: Write the failing test** (append):

```python
def test_build_config_committed_three_rates():
    from nrp.config_gen import build_config_committed
    cfg = build_config_committed(input_sampling_hz=5.0, output_emission_hz=40.0,
                                 commitment_hz=10.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert set(engines) == {"cortex", "sampler", "bg", "commitment", "thalamus"}
    assert engines["commitment"]["EngineTimestep"] == 0.1   # 10 Hz
    tf_names = {t["Name"] for t in cfg["DataPackProcessingFunctions"]}
    assert {"bg_to_commitment", "commitment_to_thalamus"} <= tf_names
    assert "bg_to_thalamus" not in tf_names
```

- [ ] **Step 2: Run it to confirm it fails** — Expected: FAIL.

- [ ] **Step 3: Implement `build_config_committed`** in `nrp/config_gen.py`:

```python
def build_config_committed(*, input_sampling_hz: float, output_emission_hz: float,
                           commitment_hz: float, name: str = "gonogo_c") -> dict:
    return {
        "SimulationName": name,
        "SimulationDescription": (
            f"go/no-go, sampling={input_sampling_hz} Hz, emission={output_emission_hz} Hz, "
            f"commitment={commitment_hz} Hz (Phase 4)."),
        "SimulationTimeout": 0.3,
        "EngineConfigs": [
            {"EngineType": "python_json", "EngineName": "cortex",
             "EngineTimestep": 1.0 / CORTEX_HZ,
             "PythonFileName": "nrp/engines/cortex_engine.py"},
            {"EngineType": "python_json", "EngineName": "sampler",
             "EngineTimestep": 1.0 / input_sampling_hz,
             "PythonFileName": "nrp/engines/sampler_engine.py"},
            {"EngineType": "python_json", "EngineName": "bg",
             "EngineTimestep": 1.0 / output_emission_hz,
             "PythonFileName": "nrp/engines/bg_engine.py"},
            {"EngineType": "python_json", "EngineName": "commitment",
             "EngineTimestep": 1.0 / commitment_hz,
             "PythonFileName": "nrp/engines/commitment_engine.py"},
            {"EngineType": "python_json", "EngineName": "thalamus",
             "EngineTimestep": 1.0 / THALAMUS_HZ,
             "PythonFileName": "nrp/engines/thalamus_engine.py"},
        ],
        "DataPackProcessingFunctions": [
            {"Name": "cortex_to_sampler", "FileName": "nrp/tfs/tf_cortex_to_sampler.py"},
            {"Name": "sampler_to_bg", "FileName": "nrp/tfs/tf_sampler_to_bg.py"},
            {"Name": "bg_to_commitment", "FileName": "nrp/tfs/tf_bg_to_commitment.py"},
            {"Name": "commitment_to_thalamus", "FileName": "nrp/tfs/tf_commitment_to_thalamus.py"},
            {"Name": "log_step", "FileName": "nrp/tfs/tf_log.py"},
        ],
    }
```

- [ ] **Step 4: Run it to confirm it passes** — Expected: PASS.

- [ ] **Step 5: Validate end-to-end (commitment gate releases at high rates)**

Add to `tests/nrp/test_nrp_dissociation.py`:

```python
@pytest.mark.nrp
def test_committed_config_runs_and_releases(tmp_path):
    from nrp.config_gen import build_config_committed
    cfg = build_config_committed(input_sampling_hz=160.0, output_emission_hz=160.0,
                                 commitment_hz=160.0)
    trace = run_trial(cfg, _go(), tmp_path / "committed")
    assert trace_to_outcome(trace)["motor_released"] is True
```

Run: `source $HOME/.local/nrp/bin/.nrp_env && python -m pytest tests/nrp/test_nrp_dissociation.py::test_committed_config_runs_and_releases -m nrp -v`
Expected: PASS — the five-engine pipeline runs and releases when all rates are high.

- [ ] **Step 6: Commit**

```bash
git add nrp/config_gen.py tests/nrp/test_config_gen.py tests/nrp/test_nrp_dissociation.py
git commit -m "feat: 5-engine committed config (sampling/emission/commitment) (Task 4.2)

ChangeSet-ID: nrp-config-committed"
```

---

## Phase 5 — Internal integration sub-step: knob 2 (all four knobs live)

Goal: expose the BG **internal integration step** as a knob, realised by sub-stepping the BG model inside one `runLoop` call (the integration step is internal to the engine, per §15.4 — it is NOT an `EngineTimestep`). With this, all four knobs are independently controllable.

### Task 5.1 — Integration sub-stepping in the BG engine

**Files:**
- Modify: `nrp/engines/bg_engine.py`

**Interfaces:**
- BG engine reads optional param `integration_substeps:int` (default 1) from `NRP_BGA_TRIAL_PARAMS`. Per `runLoop`, it invokes the BG model `integration_substeps` times on the current sampled evidence before emitting. Reuses `BGModelConfig` integration knobs if present; otherwise the substep count alone realises the rate ratio.

- [ ] **Step 1: Modify `nrp/engines/bg_engine.py`** — read the substep param in `initialize` and loop in `runLoop`. Replace the `initialize`/`runLoop` bodies with:

```python
    def initialize(self):
        with open(os.environ["NRP_BGA_TRIAL_PARAMS"]) as fh:
            params = json.load(fh)
        self._trial = TrialLog(
            trial_id=params["trial_id"], seed=params["seed"],
            task_type="go_nogo", cue_identity=params["cue_identity"],
            cue_onset_time=0.0,
        )
        # Knob 2: BG internal integration step. Modelled as N solver sub-steps per
        # emission step -- internal to this engine, NOT an EngineTimestep (§15.4).
        self._substeps = int(params.get("integration_substeps", 1))
        self._bg = BGAdapter(BGModelConfig())
        self._registerDataPack("sampled_evidence")
        self._registerDataPack("decision")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("sampled_evidence")
        if not raw or "channel_salience" not in raw:
            return
        evidence = evidence_from_dict(raw)
        decision = None
        # Integrate the BG model `_substeps` times before emitting; the last
        # decision is the emitted one. With substeps=1 behaviour is unchanged.
        for _ in range(self._substeps):
            decision = self._bg(self._trial, evidence)
        self._setDataPack("decision", decision_to_dict(decision))
```

- [ ] **Step 2: Confirm backward compatibility (substeps default = 1)**

Run: `source $HOME/.local/nrp/bin/.nrp_env && python -m pytest tests/nrp/test_nrp_gonogo.py -m nrp -v`
Expected: the existing Phase-1/2 signature tests still PASS (params without `integration_substeps` default to 1, so behaviour is unchanged).

- [ ] **Step 3: Commit**

```bash
git add nrp/engines/bg_engine.py
git commit -m "feat: BG internal integration sub-stepping (knob 2) (Task 5.1)

ChangeSet-ID: nrp-integration-substep"
```

### Task 5.2 — Full four-knob config builder

**Files:**
- Modify: `nrp/config_gen.py`, `tests/nrp/test_config_gen.py`

**Interfaces:**
- Add `build_config_four_knob(*, input_sampling_hz, integration_hz, output_emission_hz, commitment_hz, name="gonogo_4k") -> tuple[dict, dict]` — returns `(config, params_overlay)`, where `params_overlay = {"integration_substeps": round(integration_hz / output_emission_hz)}` is merged into the trial params by the caller. All four knobs map: sampling→sampler engine, integration→substeps overlay, emission→bg engine, commitment→commitment engine.

- [ ] **Step 1: Write the failing test** (append):

```python
def test_four_knob_maps_all_rates():
    from nrp.config_gen import build_config_four_knob
    cfg, overlay = build_config_four_knob(
        input_sampling_hz=20.0, integration_hz=80.0,
        output_emission_hz=40.0, commitment_hz=10.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert engines["sampler"]["EngineTimestep"] == 0.05      # 20 Hz
    assert engines["bg"]["EngineTimestep"] == 1.0 / 40.0     # 40 Hz emission
    assert engines["commitment"]["EngineTimestep"] == 0.1    # 10 Hz
    assert overlay["integration_substeps"] == 2              # 80/40
```

- [ ] **Step 2: Run it to confirm it fails** — Expected: FAIL.

- [ ] **Step 3: Implement `build_config_four_knob`** in `nrp/config_gen.py`:

```python
def build_config_four_knob(*, input_sampling_hz: float, integration_hz: float,
                           output_emission_hz: float, commitment_hz: float,
                           name: str = "gonogo_4k") -> tuple[dict, dict]:
    cfg = build_config_committed(
        input_sampling_hz=input_sampling_hz,
        output_emission_hz=output_emission_hz,
        commitment_hz=commitment_hz, name=name)
    # Knob 2 rides on the BG engine via a params overlay (internal sub-steps),
    # not as an EngineTimestep. substeps = how many integration steps per emission.
    substeps = max(1, round(integration_hz / output_emission_hz))
    return cfg, {"integration_substeps": substeps}
```

- [ ] **Step 4: Run it to confirm it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nrp/config_gen.py tests/nrp/test_config_gen.py
git commit -m "feat: full four-knob config builder (Task 5.2)

ChangeSet-ID: nrp-four-knob-config"
```

---

## Phase 6 — Four-knob ablation through NRPCoreSim (scientific capstone)

Goal: reproduce the pure-Python ablation finding (`PROJECT_MEMORY §5.1` / §15.4): when each of the four knobs is independently swept while the others are held high, at 5 Hz all four share the same miss boundary (period = 200 ticks = the accumulation window), and the ablation identifies the primary variable. This closes the loop: the four-knob model runs on the real runtime and the headline science survives.

### Task 6.1 — Single-knob ablation driver

**Files:**
- Create: `experiments/nrp_ablation.py`, `tests/nrp/test_nrp_ablation.py`

**Interfaces:**
- `ablate_knob(knob: str, frequencies: list[float], run_root: Path, n_seeds: int = 3) -> dict[float, float]` — sweeps one of `{"sampling","integration","emission","commitment"}` across `frequencies` with the other three held at 160 Hz; returns `{freq: go_success_rate}`. Uses `build_config_four_knob` + `run_trial` + `trace_to_outcome`.

- [ ] **Step 1: Implement `experiments/nrp_ablation.py`**

```python
"""Four-knob ablation through NRPCoreSim. For each knob, sweep its frequency
with the other three pinned high (160 Hz) and measure go-success rate. Reuses
the four-knob config builder and offline scoring; no science reimplemented."""

from __future__ import annotations

import json
from pathlib import Path

from nrp.config_gen import build_config_four_knob
from nrp.run import run_trial
from nrp.score import trace_to_outcome

KNOBS = ("sampling", "integration", "emission", "commitment")
FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
HIGH = 160.0


def _rates_for(knob: str, hz: float) -> dict:
    rates = {"input_sampling_hz": HIGH, "integration_hz": HIGH,
             "output_emission_hz": HIGH, "commitment_hz": HIGH}
    rates[{"sampling": "input_sampling_hz", "integration": "integration_hz",
           "emission": "output_emission_hz", "commitment": "commitment_hz"}[knob]] = hz
    return rates


def ablate_knob(knob: str, frequencies: list[float], run_root: Path,
                n_seeds: int = 3) -> dict[float, float]:
    out: dict[float, float] = {}
    for hz in frequencies:
        released = 0
        cfg, overlay = build_config_four_knob(**_rates_for(knob, hz))
        for seed in range(n_seeds):
            params = {"trial_id": seed, "seed": seed, "cue_identity": "go", **overlay}
            trace = run_trial(cfg, params, run_root / f"{knob}_{hz}hz_s{seed}")
            if trace_to_outcome(trace)["motor_released"]:
                released += 1
        out[hz] = released / n_seeds
    return out


if __name__ == "__main__":
    run_root = Path("nrp/run/ablation")
    results = {k: ablate_knob(k, FREQUENCIES_HZ, run_root) for k in KNOBS}
    Path("nrp/run/nrp_ablation.json").write_text(json.dumps(results, indent=2))
    print("go-success rate vs frequency, per ablated knob (others pinned 160 Hz):")
    for k in KNOBS:
        row = "  ".join(f"{hz:g}:{results[k][hz]:.2f}" for hz in FREQUENCIES_HZ)
        print(f"  {k:11s} {row}")
    print("saved -> nrp/run/nrp_ablation.json")
```

- [ ] **Step 2: Write the ablation validation test**

`tests/nrp/test_nrp_ablation.py`:

```python
import pytest

from experiments.nrp_ablation import ablate_knob, FREQUENCIES_HZ


@pytest.mark.nrp
def test_sampling_knob_has_miss_boundary_at_5hz(tmp_path):
    # The sampling knob must reproduce the headline boundary: miss at 5 Hz,
    # release at >=10 Hz, with the other three knobs pinned high.
    rates = ablate_knob("sampling", FREQUENCIES_HZ, tmp_path / "samp", n_seeds=2)
    assert rates[5.0] == 0.0
    assert rates[10.0] == 1.0


@pytest.mark.nrp
def test_all_knobs_runnable_and_monotone(tmp_path):
    # Each knob sweep completes through the runtime and is non-decreasing in
    # frequency (no knob makes higher frequency worse).
    for knob in ("sampling", "emission", "commitment"):
        rates = ablate_knob(knob, FREQUENCIES_HZ, tmp_path / knob, n_seeds=2)
        vals = [rates[hz] for hz in FREQUENCIES_HZ]
        assert vals == sorted(vals), f"{knob} not monotone: {vals}"
```

- [ ] **Step 3: Run the ablation validation under the nrp env**

Run:
```bash
source $HOME/.local/nrp/bin/.nrp_env
python -m pytest tests/nrp/test_nrp_ablation.py -m nrp -v
```
Expected: PASS — the sampling knob shows the 5 Hz miss boundary; all swept knobs are monotone. Compare the printed table against `deprecated_toy_prototype_results/ablation_frequency_v2.json` (the pure-Python ablation): the categorical boundary at 5 Hz must agree. Any divergence is a real finding — investigate with systematic-debugging, do not adjust thresholds to force agreement.

- [ ] **Step 4: Run the full ablation experiment**

Run: `source $HOME/.local/nrp/bin/.nrp_env && python experiments/nrp_ablation.py`
Expected: prints the per-knob rate table and saves `nrp/run/nrp_ablation.json`.

- [ ] **Step 5: Commit**

```bash
git add experiments/nrp_ablation.py tests/nrp/test_nrp_ablation.py
git commit -m "feat: four-knob ablation through NRPCoreSim (Task 6.1)

Reproduces the pure-Python ablation boundary (5 Hz miss) on the real runtime
with all four frequency knobs independently realised as engine timesteps /
integration sub-steps.

ChangeSet-ID: nrp-phase6-ablation"
```

### Task 6.2 — Document the binding in PROJECT_MEMORY

**Files:**
- Modify: `PROJECT_MEMORY.md` (§15 — append a "15.7 Realised binding" subsection; do not rewrite existing content)

- [ ] **Step 1: Append §15.7 to PROJECT_MEMORY.md** — add a new subsection after §15.6 recording the realised binding (append only, per the documentation-update rule):

```markdown
### 15.7 Realised binding (nrp/ package)

The nrp-core binding is implemented under `nrp/` and validated end-to-end on
go/no-go. Mapping of the four §5 knobs to the runtime:

| §5 knob | Realisation | Where |
|---|---|---|
| input sampling frequency | `sampler` engine `EngineTimestep` | `nrp/engines/sampler_engine.py` |
| internal integration step | N sub-steps inside one BG `runLoop` (params overlay `integration_substeps`) | `nrp/engines/bg_engine.py` |
| output emission frequency | `bg` engine `EngineTimestep` | `nrp/engines/bg_engine.py` |
| commitment update frequency | `commitment` engine `EngineTimestep` | `nrp/engines/commitment_engine.py` |

Cortex runs at 1000 Hz (1 ms base step); thalamus at 1000 Hz. One NRPCoreSim
run = one trial; sweeps generate per-condition configs (`nrp/config_gen.py`) and
invoke NRPCoreSim (`nrp/run.py`). Outcome classification is offline
(`nrp/score.py`), reusing the existing ThalamusGate notion of "released".
Acceptance is categorical (signature survives), not bit-identical across the
IPC boundary. Validation: `tests/nrp/test_nrp_*.py` (marked `nrp`, deselected by
default). Result: the go-success signature (0.0 at 5 Hz, 1.0 at ≥10 Hz) and the
single-knob ablation boundary survive the real runtime.

Out of scope of this binding: pysim plant embodiment, paradigms other than
go/no-go, and perturbation (latency/jitter/dropout/phase-offset) TFs.
```

- [ ] **Step 2: Update §1 current-state line** — append one line to §1 noting the nrp-core binding phase is complete (append only):

```markdown
- **NRP-core binding complete (go/no-go).** Four-knob frequency model realised on
  the real NRPCoreSim/FTILoop runtime (`nrp/`); BG-frequency signature and ablation
  boundary reproduced. See §15.7. Legacy pure-Python outputs badged under
  `deprecated_toy_prototype_*`.
```

- [ ] **Step 3: Sanity-check the full default suite is green**

Run: `python -m pytest -q`
Expected: the pure-Python suite passes and all `nrp`-marked tests are deselected (no runtime required for default CI).

- [ ] **Step 4: Commit**

```bash
git add PROJECT_MEMORY.md
git commit -m "docs: record realised nrp-core four-knob binding (§15.7) (Task 6.2)

ChangeSet-ID: nrp-memory-binding"
```

---

## Out of Scope / Follow-on

Deliberately excluded from this plan (candidates for later plans, each its own working deliverable):
- **pysim plant embodiment** — route `motor` into a `py_sim` engine (Bullet/OpenSim) consuming the command, reusing the OpenSim Arm26 work (Phase 10/11). The `py_sim` engine and `async_pysim_engine.py` are already installed.
- **Other paradigms** — two-choice, stop-signal, change-of-mind: each needs its own task engine + cue logic, reusing `engines/*.py` classification offline.
- **Perturbation TFs** — latency (enqueue-and-release), jitter (random delay), dropout (probabilistic drop) as TF wrappers; phase-offset via per-engine starting offset (per §15.4 this is an unverified nrp-core capability — a research task in its own right).
- **DataTransfer engine logging** — replace the file-append logger TF with the `NRPDataTransferEngineExecutable` for off-loop streaming if logging cost shows up in step time.

---

## Self-Review

- **Spec coverage:** conservative single-paradigm start (Phase 1, go/no-go) ✓; ends at full four-knob model (Phase 5–6) ✓; multiple timesteps per engine (cortex/sampler/bg/commitment/thalamus each own `EngineTimestep`, integration as sub-step) ✓; science as important as runtime (every phase ends in a categorical empirical check reusing the validated layer) ✓; legacy folders renamed not deleted, badged (Task 0.1) ✓.
- **Placeholder scan:** every code step contains complete, runnable content; every test step has real assertions; every run step has an exact command + expected observable. No TBD/TODO.
- **Type consistency:** datapack names are stable across producers/consumers — `evidence`(cortex)→`incoming_evidence`(sampler)→`sampled_evidence`(bg/sampler)→`decision`(bg)→`raw_decision`(commitment)→`committed_decision`(commitment→thalamus)→`motor`(thalamus); serde function names match across Tasks 1.1–6.1; config-builder names (`build_config`, `build_config_sampled`, `build_config_committed`, `build_config_four_knob`) are introduced once and reused consistently.
- **Open decision surfaced to the user:** whether `neuroscience_summary.md` / `open_data_candidates.md` should be relocated to `docs/` rather than badged as toy-model output (Task 0.1 notes this; not auto-resolved).
