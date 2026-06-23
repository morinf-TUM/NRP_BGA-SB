# Functional BG Integration Knob Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BG frequency knob 2 (internal integration sub-step) functionally dissociable on the nrp-core binding by giving the BG solver a stateful, carried-state integrator that is read out before convergence.

**Architecture:** Add an additive stateful stepper to the existing GPR solver (`BGModel.step` + `BGIntegratorState`, leaving `BGModel.compute` byte-for-byte unchanged). A new pure-Python `BGIntegratorDriver` schedules one carried Jacobi sweep per integration-tick boundary inside a half-open accumulation window `[0, accumulation_ms)`, reading out the current (possibly unsettled) state. The nrp `bg` engine delegates to this driver; `config_gen` passes the integration rate instead of a pre-baked sub-step count. At the baseline integration rate the driver converges and reproduces `compute()`; only a lowered integration rate changes behaviour — fixing the inert integration ablation row.

**Tech Stack:** Python 3.10, numpy, Pydantic v2, pytest, ruff. nrp-core (NRPCoreSim/FTILoop) for the gated runtime tests only.

## Global Constraints

- Python 3.10; pydantic ≥ 2.0; numpy ≥ 1.26 on host. **The gated `nrp` runtime needs numpy<2** (nrp_json.so ABI) and `source .nrp_env` per shell; `NRPCoreSim` must be invoked with `-d <repo_root>` (handled by `nrp/run.py`).
- Fail fast: no silent fallbacks, no broad except, no speculative getattr/casts. Validate inputs and raise `ValueError` with an explicit message.
- `BGModel.compute()` output must remain **exactly** unchanged (it backs M2 validation and every host/closed-loop test). All new behaviour is additive.
- nrp-only scope: do **not** modify `src/nrp_bga_sb/scheduler.py`, the closed-loop policy, or `deprecated_toy_prototype_results/`.
- Literate-programming comment style: explain *why* for decision points/constraints; no comments that merely restate code.
- Run the narrowest test that proves each change before widening. Commit after each task.
- Accumulation window is the half-open interval `[0, accumulation_ms)` with `accumulation_ms = 200.0` (matches the cortex ramp + prototype window, and excludes the t=200 ms tick so 5 Hz misses).
- Cortex ramp facts the boundary depends on: `CortexConfig` defaults `rise_time_ms=100`, `peak_salience=0.9`, `base_salience=0.5`; thalamic `margin_threshold=0.05`.

---

### Task 1: Stateful stepper in the GPR solver (`BGModel.step` + `BGIntegratorState`)

Refactor the per-iteration math out of `compute()` into shared helpers (keeping `compute()` output identical), add `BGIntegratorState`, `BGModel.step`, and extract the latency mapping into a shared function reused by `BGAdapter`.

**Files:**
- Modify: `src/nrp_bga_sb/bg_model.py`
- Test: `tests/test_bg_model_step.py` (new)

**Interfaces:**
- Consumes: existing `BGModelConfig`, `BGModel`, `BGAdapter`, `BGDecision`, `ActionEvidence`.
- Produces:
  - `BGIntegratorState` dataclass with fields `STN: np.ndarray`, `GPe: np.ndarray`, `GPi: np.ndarray`, `selected_channel: int`, `decision_margin: float`, `suppression_vector: list[float]`, `channel_activations: list[float]`, `T_winner: float`, `n_sweeps: int`; classmethod `initial(n: int) -> BGIntegratorState`.
  - `BGModel.step(state: BGIntegratorState, saliences: np.ndarray, n_sweeps: int = 1) -> BGIntegratorState`.
  - `selection_latency_s(config: BGModelConfig, T_winner: float) -> float` (module-level).

- [ ] **Step 1: Write the failing test**

Create `tests/test_bg_model_step.py`:

```python
import numpy as np

from nrp_bga_sb.bg_model import (
    BGAdapter,
    BGIntegratorState,
    BGModel,
    BGModelConfig,
    selection_latency_s,
)
from nrp_bga_sb.schemas import ActionEvidence, TrialLog


def test_initial_state_reads_out_channel0_zero_margin():
    # Zero state: GPi=0 -> T=thal_threshold for all channels -> argmax=0, margin=0.
    s = BGIntegratorState.initial(2)
    assert s.selected_channel == 0
    assert s.decision_margin == 0.0
    assert s.n_sweeps == 0


def test_step_is_non_idempotent_margin_grows():
    model = BGModel(BGModelConfig())
    sal = np.array([0.65, 0.35])  # medium conflict
    s0 = BGIntegratorState.initial(2)
    s1 = model.step(s0, sal, n_sweeps=1)
    s2 = model.step(s1, sal, n_sweeps=1)
    # One sweep on medium conflict has not cleared the gate; a second sweep has.
    assert s1.decision_margin < 0.05
    assert s2.decision_margin > 0.05
    assert s2.n_sweeps == 2


def test_step_converges_to_compute_fixed_point():
    model = BGModel(BGModelConfig())
    sal = np.array([0.8, 0.2])
    ref = model.compute(sal)
    s = BGIntegratorState.initial(2)
    s = model.step(s, sal, n_sweeps=50)  # well past convergence
    assert s.selected_channel == ref["selected_channel"]
    assert abs(s.decision_margin - ref["decision_margin"]) < 1e-6
    assert abs(s.T_winner - ref["T_winner"]) < 1e-6


def test_compute_output_unchanged_canonical_levels():
    # Lock compute() exactly (M2 anchor): values verified against source.
    model = BGModel(BGModelConfig())
    low = model.compute(np.array([0.8, 0.2]))
    assert low["selected_channel"] == 0
    assert abs(low["T_winner"] - 0.194) < 1e-3
    high = model.compute(np.array([0.55, 0.45]))
    assert high["selected_channel"] == -1


def test_selection_latency_matches_adapter_formula():
    cfg = BGModelConfig()
    # T_winner>0 path and the no-selection cap.
    assert abs(selection_latency_s(cfg, 0.2) - (cfg.latency_min_ms + cfg.latency_scale_ms / 0.25) / 1000.0) < 1e-12
    assert selection_latency_s(cfg, 0.0) == cfg.latency_max_ms / 1000.0


def test_adapter_still_produces_same_decision():
    # BGAdapter must be unchanged after extracting the latency helper.
    adapter = BGAdapter(BGModelConfig())
    trial = TrialLog(trial_id=1, seed=1, task_type="go_nogo", cue_identity="go", cue_onset_time=0.0)
    ev = ActionEvidence(sim_time=0.1, trial_id=1, n_channels=2, channel_salience=[0.8, 0.2])
    dec = adapter(trial, ev)
    assert dec.selected_channel == 0
    assert abs(dec.selection_latency - 0.013) < 2e-3  # ~13 ms low-conflict latency
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bg_model_step.py -q`
Expected: FAIL — `ImportError: cannot import name 'BGIntegratorState'` / `selection_latency_s`.

- [ ] **Step 3: Refactor `bg_model.py` — extract sweep/readout/latency helpers, add state + step**

In `src/nrp_bga_sb/bg_model.py`, add module-level helpers after the imports (below the `BGModelConfig` block is fine, but before `BGModel`):

```python
# --- Shared GPR primitives (used by both the converging solver and the stepper) ---


def _jacobi_update(saliences, D1, D2, GPe, cfg):
    """One GPR sweep. STN reads the *previous* GPe (feedback); GPe and GPi read the
    just-computed STN. This is exactly one iteration of the compute() fixed-point loop."""
    STN = np.maximum(0.0, saliences - cfg.w_gpe_stn * GPe + cfg.stn_offset)
    GPe_new = np.maximum(0.0, cfg.w_stn_gpe * STN - cfg.w_d2_gpe * D2 + cfg.gpe_offset)
    GPi = np.maximum(
        0.0, cfg.w_stn_gpi * float(np.mean(STN)) - cfg.w_d1_gpi * D1 + cfg.gpi_offset
    )
    return STN, GPe_new, GPi


def _readout(GPi, cfg):
    """Thalamus output + winner selection from a GPi vector (settled or not)."""
    T = np.maximum(0.0, cfg.thal_threshold - GPi)
    n = len(GPi)
    if float(np.max(T)) > 0.0:
        selected = int(np.argmax(T))
        T_sorted = np.sort(T)[::-1]
        margin = float(T_sorted[0] - T_sorted[1]) if n >= 2 else float(T_sorted[0])
        T_winner = float(T_sorted[0])
    else:
        selected, margin, T_winner = -1, 0.0, 0.0
    return selected, margin, T.tolist(), T_winner


def selection_latency_s(config, T_winner):
    """Map thalamus winner output to selection latency (seconds). Inverse
    proportionality: smaller T_winner (more conflict / less settling) -> longer latency."""
    if T_winner > 0.0:
        latency_ms = config.latency_min_ms + config.latency_scale_ms / (T_winner + config.latency_eps)
    else:
        latency_ms = config.latency_max_ms
    return latency_ms / 1000.0
```

Add the state dataclass (after the helpers, before `BGModel`):

```python
@dataclass
class BGIntegratorState:
    """Carried GPR activations plus the readout derived from them. Persists across
    integration sub-steps within a trial so repeated sub-steps are NOT idempotent."""

    STN: np.ndarray
    GPe: np.ndarray
    GPi: np.ndarray
    selected_channel: int = 0
    decision_margin: float = 0.0
    suppression_vector: list[float] = field(default_factory=list)
    channel_activations: list[float] = field(default_factory=list)
    T_winner: float = 0.0
    n_sweeps: int = 0

    @classmethod
    def initial(cls, n: int, cfg: "BGModelConfig | None" = None) -> "BGIntegratorState":
        cfg = cfg or BGModelConfig()
        GPi = np.zeros(n)
        selected, margin, acts, T_winner = _readout(GPi, cfg)
        return cls(
            STN=np.zeros(n), GPe=np.zeros(n), GPi=GPi,
            selected_channel=selected, decision_margin=margin,
            suppression_vector=GPi.tolist(), channel_activations=acts,
            T_winner=T_winner, n_sweeps=0,
        )
```

Refactor `BGModel.compute` to use the helpers (keeps the convergence loop and output identical). Replace the in-loop nuclei updates with `_jacobi_update`, and the thalamus/winner block with `_readout`:

```python
        n_iters = 0
        for iteration in range(cfg.max_iters):
            STN_prev, GPe_prev, GPi_prev = STN.copy(), GPe.copy(), GPi.copy()
            STN, GPe, GPi = _jacobi_update(saliences, D1, D2, GPe_prev, cfg)
            n_iters = iteration + 1
            if (
                float(np.max(np.abs(STN - STN_prev))) < cfg.tol
                and float(np.max(np.abs(GPe - GPe_prev))) < cfg.tol
                and float(np.max(np.abs(GPi - GPi_prev))) < cfg.tol
            ):
                break

        selected, margin, activations, T_winner = _readout(GPi, cfg)
        return {
            "selected_channel": selected,
            "decision_margin": margin,
            "suppression_vector": GPi.tolist(),
            "channel_activations": activations,
            "n_iters": n_iters,
            "T_winner": T_winner,
        }
```

Add `step` as a method on `BGModel` (after `compute`):

```python
    def step(self, state, saliences, n_sweeps=1):
        """Advance the carried integrator by n_sweeps GPR sweeps on `saliences`,
        reading out the current (possibly unsettled) state. With enough sweeps this
        converges to the same fixed point as compute()."""
        if n_sweeps < 1:
            raise ValueError(f"n_sweeps must be >= 1, got {n_sweeps}")
        cfg = self.config
        saliences = np.asarray(saliences, dtype=float)
        D1 = np.maximum(0.0, saliences - cfg.theta_d)
        D2 = np.maximum(0.0, saliences - cfg.theta_d)
        STN, GPe, GPi = state.STN, state.GPe, state.GPi
        for _ in range(n_sweeps):
            STN, GPe, GPi = _jacobi_update(saliences, D1, D2, GPe, cfg)
        selected, margin, activations, T_winner = _readout(GPi, cfg)
        return BGIntegratorState(
            STN=STN, GPe=GPe, GPi=GPi,
            selected_channel=selected, decision_margin=margin,
            suppression_vector=GPi.tolist(), channel_activations=activations,
            T_winner=T_winner, n_sweeps=state.n_sweeps + n_sweeps,
        )
```

In `BGAdapter.__call__`, replace the inline latency block with the shared helper (behaviour identical):

```python
        T_winner = result["T_winner"]
        return BGDecision(
            sim_time=action_evidence.sim_time,
            trial_id=action_evidence.trial_id,
            selected_channel=result["selected_channel"],
            decision_margin=result["decision_margin"],
            suppression_vector=result["suppression_vector"],
            channel_activations=result["channel_activations"],
            selection_latency=selection_latency_s(self.config, T_winner),
        )
```

- [ ] **Step 4: Run the new tests and the existing bg_model suite**

Run: `python -m pytest tests/test_bg_model_step.py tests/test_bg_model.py -q`
Expected: PASS (all). If `tests/test_bg_model.py` does not exist, run `python -m pytest tests/test_bg_model_step.py -q` plus `python -m pytest -k bg -q`.

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/bg_model.py tests/test_bg_model_step.py
git commit -m "feat: add stateful BGModel.step + BGIntegratorState (compute unchanged)

ChangeSet-ID: knob2-stepper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Host-testable integration driver (`BGIntegratorDriver`) — the boundary lock

A pure-Python driver that schedules one carried sweep per integration-tick boundary inside `[0, accumulation_ms)` and returns a `BGDecision`. This task contains the runtime-independent proof that the knob is functional and matches the boundary.

**Files:**
- Create: `src/nrp_bga_sb/bg_integrator.py`
- Test: `tests/test_bg_integrator.py` (new)

**Interfaces:**
- Consumes: `BGModel`, `BGModelConfig`, `BGIntegratorState`, `selection_latency_s` (Task 1); `ActionEvidence`, `BGDecision`, `TrialLog`; `CortexEvidenceGenerator`, `CortexConfig`; `ThalamusGate`, `ThalamusConfig`.
- Produces: `BGIntegratorDriver(integration_hz: float, accumulation_ms: float = 200.0, config: BGModelConfig = ...)` with method `advance(elapsed_ms: float, evidence: ActionEvidence) -> BGDecision`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bg_integrator.py`:

```python
from nrp_bga_sb.bg_integrator import BGIntegratorDriver
from nrp_bga_sb.cortex import CortexConfig, CortexEvidenceGenerator
from nrp_bga_sb.thalamus import ThalamusConfig, ThalamusGate
from nrp_bga_sb.schemas import TrialLog

EMISSION_HZ = 160.0          # other knobs pinned high during the integration ablation
SIM_MS = 300.0               # nrp SimulationTimeout = 0.3 s


def _ever_released(integration_hz: float) -> bool:
    """Replicate the nrp pipeline on the host: drive the integrator at the emission
    cadence with the cortex ramp, and ask whether the thalamus gate ever opens
    (same release rule as nrp/score.py)."""
    drv = BGIntegratorDriver(integration_hz=integration_hz, accumulation_ms=200.0)
    cortex = CortexEvidenceGenerator(CortexConfig())
    gate = ThalamusGate(ThalamusConfig())
    trial = TrialLog(trial_id=0, seed=0, task_type="go_nogo", cue_identity="go", cue_onset_time=0.0)
    step_ms = 1000.0 / EMISSION_HZ
    t = 0.0
    released = False
    while t <= SIM_MS + 1e-9:
        ev = cortex(trial, t)
        dec = drv.advance(t, ev)
        motor = gate(dec)
        if motor.gate_state in ("open", "partial") and any(motor.command):
            released = True
        t += step_ms
    return released


def test_integration_misses_at_5hz():
    # 5 Hz: single tick at t=0 on neutral evidence; t=200 tick excluded -> never settles.
    assert _ever_released(5.0) is False


def test_integration_succeeds_at_10_20_40hz():
    assert _ever_released(10.0) is True
    assert _ever_released(20.0) is True
    assert _ever_released(40.0) is True


def test_baseline_rate_converges_like_compute():
    # 160 Hz: ~32 sweeps within the window -> converged readout -> success on the ramp.
    assert _ever_released(160.0) is True


def test_rejects_nonpositive_rate():
    import pytest
    with pytest.raises(ValueError):
        BGIntegratorDriver(integration_hz=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bg_integrator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nrp_bga_sb.bg_integrator'`.

- [ ] **Step 3: Implement the driver**

Create `src/nrp_bga_sb/bg_integrator.py`:

```python
"""Stateful BG integration driver (knob 2). Carries GPR activations across
integration sub-steps within a trial and reads out the current (possibly
unsettled) state, so the integration RATE controls how far the BG has settled
by the decision deadline. This is the science-layer change that makes the
internal-integration-step knob functionally dissociable (PROJECT_MEMORY §15.7).

The nrp `bg` engine cannot run on the host (it imports nrp_core), so the
scheduling + stepping logic lives here, host-tested, and the engine delegates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nrp_bga_sb.bg_model import (
    BGIntegratorState,
    BGModel,
    BGModelConfig,
    selection_latency_s,
)
from nrp_bga_sb.schemas import ActionEvidence, BGDecision


@dataclass
class BGIntegratorDriver:
    """Schedules one carried GPR sweep per integration-tick boundary inside the
    half-open window [0, accumulation_ms). At the baseline rate the integrator
    converges and the readout equals BGModel.compute(); at low rates it is read
    out before settling, producing the miss-at-5-Hz / success-at->=10-Hz boundary.
    """

    integration_hz: float
    accumulation_ms: float = 200.0
    config: BGModelConfig = field(default_factory=BGModelConfig)

    def __post_init__(self) -> None:
        if self.integration_hz <= 0:
            raise ValueError(f"integration_hz must be > 0, got {self.integration_hz}")
        if self.accumulation_ms <= 0:
            raise ValueError(f"accumulation_ms must be > 0, got {self.accumulation_ms}")
        self._model = BGModel(self.config)
        self._period_ms = 1000.0 / self.integration_hz
        self._next_k = 0
        self._state: BGIntegratorState | None = None

    def advance(self, elapsed_ms: float, evidence: ActionEvidence) -> BGDecision:
        n = evidence.n_channels
        if self._state is None:
            self._state = BGIntegratorState.initial(n, self.config)

        saliences = evidence.channel_salience
        # Fire every integration tick whose boundary has been crossed by now and
        # lies strictly inside the accumulation window. Each fires one sweep on the
        # evidence current at this call. The strict `< accumulation_ms` bound
        # excludes the t=200 ms tick, which is what makes 5 Hz (period=200 ms) miss.
        while (
            self._next_k * self._period_ms <= elapsed_ms
            and self._next_k * self._period_ms < self.accumulation_ms
        ):
            self._state = self._model.step(self._state, saliences, n_sweeps=1)
            self._next_k += 1

        s = self._state
        return BGDecision(
            sim_time=evidence.sim_time,
            trial_id=evidence.trial_id,
            selected_channel=s.selected_channel,
            decision_margin=s.decision_margin,
            suppression_vector=s.suppression_vector,
            channel_activations=s.channel_activations,
            selection_latency=selection_latency_s(self.config, s.T_winner),
        )
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_bg_integrator.py -q`
Expected: PASS (4 tests). This is the boundary lock.

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/bg_integrator.py tests/test_bg_integrator.py
git commit -m "feat: BGIntegratorDriver locks integration boundary (miss@5Hz, success@>=10Hz)

ChangeSet-ID: knob2-driver

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `config_gen` overlay carries the integration rate

Replace the degenerate `substeps` mapping with a direct `integration_hz` overlay so the engine's driver schedules from the rate.

**Files:**
- Modify: `nrp/config_gen.py:101-111` (`build_config_four_knob`)
- Test: `tests/nrp/test_config_gen.py:43-52` (`test_four_knob_maps_all_rates`)

**Interfaces:**
- Consumes: existing `build_config_committed`.
- Produces: `build_config_four_knob(...) -> tuple[dict, dict]` whose second element (overlay) is now `{"integration_hz": float}`.

- [ ] **Step 1: Update the failing test**

In `tests/nrp/test_config_gen.py`, change the last assertion of `test_four_knob_maps_all_rates`:

```python
    assert overlay["integration_hz"] == 80.0           # rate passed through to the engine driver
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/nrp/test_config_gen.py::test_four_knob_maps_all_rates -q`
Expected: FAIL — `KeyError: 'integration_hz'` (overlay still has `integration_substeps`).

- [ ] **Step 3: Update `build_config_four_knob`**

In `nrp/config_gen.py`, replace the substeps computation (lines 108-111):

```python
    # Knob 2 rides on the BG engine via a params overlay. The engine's
    # BGIntegratorDriver schedules carried sweeps from the rate directly, decoupled
    # from emission; passing a pre-baked sub-step count collapsed to 1 across the
    # whole 5-160 Hz grid (integration_hz <= emission baseline), which is why the
    # knob was inert. See PROJECT_MEMORY §15.7.
    return cfg, {"integration_hz": float(integration_hz)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/nrp/test_config_gen.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nrp/config_gen.py tests/nrp/test_config_gen.py
git commit -m "feat: four-knob overlay carries integration_hz (not pre-baked substeps)

ChangeSet-ID: knob2-overlay

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: nrp `bg` engine delegates to the driver

Rewrite the engine `runLoop` to delegate to `BGIntegratorDriver`, removing the idempotent sub-step loop. This file imports `nrp_core`; it is verified by the gated runtime ablation in Task 5, not a host unit test.

**Files:**
- Modify: `nrp/engines/bg_engine.py`

**Interfaces:**
- Consumes: `BGIntegratorDriver` (Task 2); `params["integration_hz"]` from the overlay (Task 3); existing `evidence_from_dict`, `decision_to_dict`.
- Produces: `decision` datapack with the carried-state readout each emission step.

- [ ] **Step 1: Rewrite `bg_engine.py`**

Replace the body of `nrp/engines/bg_engine.py` with:

```python
"""BG engine: runs the GPR BG model on the most recent sampled evidence and emits
a `decision` datapack. Delegates to BGIntegratorDriver, a stateful integrator that
carries GPR activations across emission steps within a trial and reads out before
convergence -- so the internal-integration-step rate (knob 2) is functionally
dissociable (PROJECT_MEMORY §15.7), not idempotent.

Input-sampling, emission, and commitment remain EngineTimesteps (§15.4); the
integration rate rides on the params overlay (`integration_hz`)."""

import json
import os

from nrp_core.engines.python_json import EngineScript

from nrp.serde import decision_to_dict, evidence_from_dict
from nrp_bga_sb.bg_integrator import BGIntegratorDriver

# Matches the cortex ramp + prototype accumulation window; the strict upper bound
# excludes the t=200 ms tick so a 5 Hz integration rate settles only on neutral
# early evidence and misses (mirrors the other three knobs' 5 Hz boundary).
ACCUMULATION_MS = 200.0


class Script(EngineScript):
    def initialize(self):
        with open(os.environ["NRP_BGA_TRIAL_PARAMS"]) as fh:
            params = json.load(fh)
        # Knob 2: BG internal integration step, scheduled by the driver from the
        # integration rate. State is created here and carried across runLoop calls
        # (one NRPCoreSim run = one trial).
        self._driver = BGIntegratorDriver(
            integration_hz=float(params["integration_hz"]),
            accumulation_ms=ACCUMULATION_MS,
        )
        self._registerDataPack("sampled_evidence")
        self._registerDataPack("decision")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("sampled_evidence")
        # Trigger: no evidence delivered yet (first ticks before the sampler TF fires).
        # Why: the driver needs a populated ActionEvidence; skip until present.
        # Outcome: `decision` keeps its previous value until evidence arrives.
        if not raw or "channel_salience" not in raw:
            return
        evidence = evidence_from_dict(raw)
        elapsed_ms = self._time_ns / 1.0e6
        decision = self._driver.advance(elapsed_ms, evidence)
        self._setDataPack("decision", decision_to_dict(decision))

    def shutdown(self):
        pass
```

- [ ] **Step 2: Static check (host, no runtime)**

The engine can't be imported on the host (needs `nrp_core`), so just lint it.
Run: `ruff check nrp/engines/bg_engine.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add nrp/engines/bg_engine.py
git commit -m "feat: bg engine delegates to stateful BGIntegratorDriver (knob 2 functional)

ChangeSet-ID: knob2-engine

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Regenerate the nrp ablation snapshot + comparison report (gated runtime)

Run the gated nrp ablation to regenerate `nrp/results/ablation.json` (integration row should flip to `0.0 @ 5 Hz → 1.0 @ ≥10 Hz`), update the divergence-encoding tests, and regenerate the comparison report.

**Files:**
- Regenerate: `nrp/results/ablation.json`
- Regenerate: `docs/nrp_vs_prototype_comparison.md`
- Modify: `tests/nrp/test_compare.py` (the assertions encoding the divergence)

**Interfaces:**
- Consumes: the gated `experiments/nrp_ablation.py` (Task 3/4 changes); `experiments/nrp_vs_prototype.py` (report driver, unchanged); `nrp/compare.py` (data-driven, unchanged).

- [ ] **Step 1: Regenerate the nrp ablation snapshot (requires the nrp runtime)**

Run (in a shell with the runtime):
```bash
source .nrp_env
python experiments/nrp_ablation.py
```
Expected stdout: the `integration` row now reads `5:0.00  10:1.00  20:1.00  40:1.00  80:1.00  160:1.00`; the other three rows unchanged. The file `nrp/results/ablation.json` is rewritten.
Verify:
```bash
python -c "import json; d=json.load(open('nrp/results/ablation.json')); print('integration', d['integration']); print('emission', d['emission'])"
```
Expected: `integration {'5.0': 0.0, '10.0': 1.0, ...}` and `emission` unchanged from before.

> If the nrp runtime is unavailable in this environment, STOP and report: the snapshot regeneration and the gated assertions below must be run where `.nrp_env`/`NRPCoreSim` exist. Tasks 1–4 (host) are complete and independently valuable.

- [ ] **Step 2: Update the divergence-encoding tests**

In `tests/nrp/test_compare.py`, make these edits so the suite asserts the resolved (4-of-4) state:

`test_load_nrp_ablation_coerces_str_hz_keys` — integration now misses at 5 Hz:
```python
    assert out["integration"][5.0] == 0.0
```

`test_committed_snapshots_lock_key_divergence` — rename intent to lock the resolution; replace its body:
```python
def test_committed_snapshots_lock_resolved_boundary():
    """The committed nrp snapshots now reproduce the prototype on all four knobs:
    sampling AND integration both miss at 5 Hz, succeed at >=10 Hz."""
    ablation = json.loads((RESULTS / "ablation.json").read_text())
    assert ablation["sampling"]["5.0"] == 0.0
    assert ablation["integration"]["5.0"] == 0.0
    assert ablation["integration"]["10.0"] == 1.0
    sweep = json.loads((RESULTS / "gonogo_sweep.json").read_text())
    assert sweep["5.0"] == 0.0
    assert sweep["10.0"] == 1.0
```

`test_compare_ablation_flags_integration_divergence` — rename + assert integration now holds:
```python
def test_compare_ablation_all_knobs_hold():
    proto = load_prototype_ablation(PROTO / "ablation_frequency_v2.json")
    nrp = load_nrp_ablation(RESULTS / "ablation.json")
    verdict = compare_ablation(proto, nrp)
    by_knob = {kv.knob: kv for kv in verdict.knobs}
    assert by_knob["sampling"].holds is True
    assert by_knob["emission"].holds is True
    assert by_knob["commitment"].holds is True
    assert by_knob["integration"].holds is True
    assert by_knob["integration"].divergent_freqs == []
    assert verdict.holds is True
```

`test_build_verdict_summary_names_integration` — rename + assert 4-of-4 with integration held:
```python
def test_build_verdict_summary_reports_all_four_hold():
    proto_ab = load_prototype_ablation(PROTO / "ablation_frequency_v2.json")
    nrp_ab = load_nrp_ablation(RESULTS / "ablation.json")
    proto_fs = load_prototype_gonogo_sweep(PROTO / "frequency_sweep_results.json")
    nrp_fs = load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json")
    verdict = build_verdict(
        compare_ablation(proto_ab, nrp_ab),
        compare_frequency_sweep(proto_fs, nrp_fs),
    )
    assert "4 of 4" in verdict.summary
    assert "integration" in verdict.summary.lower()  # now in the HELD list
```

`test_driver_main_writes_report` — update the headline assertion:
```python
    assert "4 of 4" in text
```

(`test_format_report_contains_tables_and_callout` still passes: the report still contains "integration", "5", and the sweep "conflict" caveat. Leave it unchanged.)

- [ ] **Step 3: Regenerate the comparison report**

Run:
```bash
python experiments/nrp_vs_prototype.py
```
Expected: `docs/nrp_vs_prototype_comparison.md` is rewritten; the **Verdict** line now reads "4 of 4 knobs reproduce the prototype exactly…", the ablation `integration` row shows `0/0 … 1/1 … ✓` with no `✗`, and `holds = ✓`.

- [ ] **Step 4: Run the comparison tests**

Run (host tests; the loaders read the committed snapshots):
```bash
python -m pytest tests/nrp/test_compare.py -q
```
Expected: PASS.

- [ ] **Step 5: Run the gated nrp ablation test if present, else note**

Run:
```bash
source .nrp_env && python -m pytest tests/nrp -m nrp -q
```
Expected: PASS where the runtime exists; otherwise deselected/skipped — note it in the commit body.

- [ ] **Step 6: Commit**

```bash
git add nrp/results/ablation.json docs/nrp_vs_prototype_comparison.md tests/nrp/test_compare.py
git commit -m "feat: nrp integration knob now reproduces prototype boundary (4 of 4)

Regenerated ablation snapshot + comparison report; integration misses at 5 Hz
and succeeds at >=10 Hz, matching sampling/emission/commitment. Resolves the
§15.7 knob-2 idempotence divergence.

ChangeSet-ID: knob2-comparison

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Documentation, memory, and full-suite guard

Update PROJECT_MEMORY to record knob 2 as functional, run the full host suite + ruff, and update auto-memory.

**Files:**
- Modify: `PROJECT_MEMORY.md` §15.7 and §1
- Modify: project memory files under `/home/fom/.claude/projects/-home-fom-code-NRP-BGA-SB/memory/`

- [ ] **Step 1: Update PROJECT_MEMORY §15.7**

Append (do not delete the historical NOTE) a follow-up paragraph after the existing knob-2 NOTE, recording that the idempotence has been resolved on the nrp side:

```markdown
**UPDATE (2026-06-23) — knob 2 made functional (nrp side).** The integration
sub-step is now a stateful carried-state integrator (`BGIntegratorDriver`,
`src/nrp_bga_sb/bg_integrator.py`): GPR activations persist across emission steps
within a trial and are read out before convergence, so the integration RATE
controls settling at the decision deadline. `BGModel.step`/`BGIntegratorState`
provide the carried sweep; `BGModel.compute` is unchanged (full-convergence path,
backs M2). The nrp `bg` engine delegates to the driver and `config_gen` passes
`integration_hz`. Result: the nrp integration ablation now reproduces the prototype
boundary (0.0 @ 5 Hz → 1.0 @ ≥10 Hz) — all four knobs hold; the comparison report
`docs/nrp_vs_prototype_comparison.md` is regenerated to a 4-of-4 verdict. The
deprecated pure-Python prototype is unchanged (it already showed the boundary via a
gate-timing artifact, not stateful settling).
```

- [ ] **Step 2: Update PROJECT_MEMORY §1 current state**

Append a bullet to §1 recording sub-project completion (keep prior bullets verbatim):

```markdown
- **Knob-2 stateful integrator complete (2026-06-23).** The nrp integration
  sub-step knob is now functionally dissociable via `BGIntegratorDriver`
  (carried-state GPR integrator, readout-before-convergence). nrp integration
  ablation reproduces the prototype boundary; all four knobs hold (4-of-4
  comparison verdict). `BGModel.compute` unchanged; prototype untouched. See §15.7.
```

- [ ] **Step 3: Run the full host suite + ruff**

Run:
```bash
python -m pytest -q -m "not nrp"
ruff check .
```
Expected: all host tests pass (≥ previous 723 + the new Task 1/2 tests); ruff clean.

- [ ] **Step 4: Commit**

```bash
git add PROJECT_MEMORY.md
git commit -m "docs: record knob-2 stateful integrator completion (§15.7, §1)

ChangeSet-ID: knob2-memory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Update auto-memory**

Update `/home/fom/.claude/projects/-home-fom-code-NRP-BGA-SB/memory/project_gonogo_comparison.md` (or create a new `project_knob2_integrator.md`) noting: knob 2 made functional on the nrp side via `BGIntegratorDriver`; comparison now 4-of-4; prototype unchanged; branch `knob2-stateful-integrator` (not pushed). Add the index line to `MEMORY.md`. (No git commit needed for the auto-memory dir.)

---

## Notes for the executor

- **Do not push.** The user pushes.
- Tasks 1–3 are pure host work and fully testable now. Task 4 is a thin engine edit verified by the gated runtime in Task 5. Tasks 5–6 require the `.nrp_env` runtime for the snapshot regeneration; if unavailable, complete Tasks 1–4, commit, and report that the gated regeneration is pending the runtime.
- The boundary proof that matters is **Task 2** (host, deterministic). The gated ablation in Task 5 confirms the engine feeds the driver correctly; if its integration boundary lands off 5 Hz, the single calibration point is `ACCUMULATION_MS` in `bg_engine.py` (and the matching `accumulation_ms` in the driver test).
