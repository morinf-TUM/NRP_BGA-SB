# Functional BG Integration Knob Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Revision 2 (latency-signature framing) governs.** See the design doc Revision 2. The
> integration knob is made functional/non-idempotent, but its nrp-observable signature is decision
> LATENCY (a slow rate settles later), not a go-success miss — so the driver has NO decision window,
> and `nrp/results/ablation.json` + `docs/nrp_vs_prototype_comparison.md` + `tests/nrp/test_compare.py`
> are left unchanged. Tasks 2, 4, 5 below are the Revision-2 versions.

**Goal:** Make BG frequency knob 2 (internal integration sub-step) functionally dissociable on the nrp-core binding by giving the BG solver a stateful, carried-state integrator read out before convergence — its functional signature being decision settling latency.

**Architecture:** Add an additive stateful stepper to the existing GPR solver (`BGModel.step` + `BGIntegratorState`, leaving `BGModel.compute` byte-for-byte unchanged). A new pure-Python `BGIntegratorDriver` schedules one carried Jacobi sweep per integration-tick boundary across the trial (no window), reading out the current (possibly unsettled) state. The nrp `bg` engine delegates to this driver; `config_gen` passes the integration rate instead of a pre-baked sub-step count. A lower integration rate settles later → its effect shows up as release latency, measured by a dedicated experiment.

**Tech Stack:** Python 3.10, numpy, Pydantic v2, pytest, ruff. nrp-core (NRPCoreSim/FTILoop) for the gated runtime tests only.

## Global Constraints

- Python 3.10; pydantic ≥ 2.0; numpy ≥ 1.26 on host. **The gated `nrp` runtime needs numpy<2** (nrp_json.so ABI) and `source .nrp_env` per shell; `NRPCoreSim` must be invoked with `-d <repo_root>` (handled by `nrp/run.py`).
- Fail fast: no silent fallbacks, no broad except, no speculative getattr/casts. Validate inputs and raise `ValueError` with an explicit message.
- `BGModel.compute()` output must remain **exactly** unchanged (it backs M2 validation and every host/closed-loop test). All new behaviour is additive.
- nrp-only scope: do **not** modify `src/nrp_bga_sb/scheduler.py`, the closed-loop policy, or `deprecated_toy_prototype_results/`.
- Revision-2 invariant: do **not** modify `nrp/results/ablation.json`, `docs/nrp_vs_prototype_comparison.md`, or `tests/nrp/test_compare.py` (the go-success ablation is metric-insensitive to this knob and is left as-is).
- Literate-programming comment style: explain *why* for decision points/constraints; no comments that merely restate code.
- Run the narrowest test that proves each change before widening. Commit after each task.
- The driver has **no decision window**: one carried sweep per integration tick (`k / integration_hz` s) across the trial. A lower rate settles later (latency signature).
- Cortex ramp facts the latency depends on: `CortexConfig` defaults `rise_time_ms=100`, `peak_salience=0.9`, `base_salience=0.5`; thalamic `margin_threshold=0.05`; nrp `SimulationTimeout=0.3 s`.

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

### Task 2 (Revision 2): Host-testable integration driver (`BGIntegratorDriver`) — the latency lock

> **Revision 2 (latency-signature framing).** This REPLACES the earlier windowed Task 2 (commit
> `d927b24`). The driver has **no accumulation window**: it schedules one carried sweep per
> integration tick across the whole trial, so a slower rate settles *later* rather than not at
> all. The empirical runtime showed the windowed form regressed the sampling knob; the integration
> knob's true nrp signature is decision **latency**, not a go-success miss. This task overwrites
> `src/nrp_bga_sb/bg_integrator.py` and `tests/test_bg_integrator.py`.

A pure-Python driver that schedules one carried sweep per integration tick and returns a `BGDecision`. This task contains the runtime-independent proof that the knob is functional (non-idempotent) and that its release latency scales with the integration rate.

**Files:**
- Overwrite: `src/nrp_bga_sb/bg_integrator.py`
- Overwrite: `tests/test_bg_integrator.py`

**Interfaces:**
- Consumes: `BGModel`, `BGModelConfig`, `BGIntegratorState`, `selection_latency_s` (Task 1); `ActionEvidence`, `BGDecision`, `TrialLog`; `CortexEvidenceGenerator`, `CortexConfig`; `ThalamusGate`, `ThalamusConfig`.
- Produces: `BGIntegratorDriver(integration_hz: float, config: BGModelConfig = ...)` with method `advance(elapsed_ms: float, evidence: ActionEvidence) -> BGDecision`. **No `accumulation_ms` parameter.**

- [ ] **Step 1: Overwrite the test (failing first)**

Overwrite `tests/test_bg_integrator.py` (host-verified values: first-release ms = 200/100/50 at 5/10/20 Hz):

```python
from nrp_bga_sb.bg_integrator import BGIntegratorDriver
from nrp_bga_sb.cortex import CortexConfig, CortexEvidenceGenerator
from nrp_bga_sb.thalamus import ThalamusConfig, ThalamusGate
from nrp_bga_sb.schemas import TrialLog

EMISSION_HZ = 160.0          # the BG engine wakes at the emission rate (others pinned high)
SIM_MS = 300.0               # nrp SimulationTimeout = 0.3 s


def _first_release_ms(integration_hz: float) -> float | None:
    """Replicate the nrp pipeline on the host: drive the integrator at the emission
    cadence with the cortex ramp, and return the first elapsed time (ms) at which the
    thalamus gate opens (same release rule as nrp/score.py), or None if never."""
    drv = BGIntegratorDriver(integration_hz=integration_hz)
    cortex = CortexEvidenceGenerator(CortexConfig())
    gate = ThalamusGate(ThalamusConfig())
    trial = TrialLog(trial_id=0, seed=0, task_type="go_nogo", cue_identity="go", cue_onset_time=0.0)
    step_ms = 1000.0 / EMISSION_HZ
    t = 0.0
    while t <= SIM_MS + 1e-9:
        ev = cortex(trial, t)
        dec = drv.advance(t, ev)
        motor = gate(dec)
        if motor.gate_state in ("open", "partial") and any(motor.command):
            return t
        t += step_ms
    return None


def test_release_latency_decreases_with_rate():
    # The integration knob's nrp signature: slower integration settles (releases) LATER.
    r5, r10, r20 = _first_release_ms(5.0), _first_release_ms(10.0), _first_release_ms(20.0)
    assert r5 is not None and r10 is not None and r20 is not None  # all functional
    assert r5 > r10 > r20  # monotone settling latency where integration is the bottleneck


def test_every_rate_eventually_releases():
    # Functional, not idempotent: even 5 Hz settles within the 300 ms sim (just late).
    for hz in (5.0, 10.0, 20.0, 160.0):
        assert _first_release_ms(hz) is not None


def test_integrator_is_non_idempotent():
    # A second sweep on the same evidence changes the readout (margin grows) — the
    # property the old stateless solver lacked.
    import numpy as np
    from nrp_bga_sb.bg_model import BGIntegratorState, BGModel, BGModelConfig

    model = BGModel(BGModelConfig())
    sal = np.array([0.65, 0.35])
    s1 = model.step(BGIntegratorState.initial(2), sal, n_sweeps=1)
    s2 = model.step(s1, sal, n_sweeps=1)
    assert s2.decision_margin > s1.decision_margin


def test_rejects_nonpositive_rate():
    import pytest
    with pytest.raises(ValueError):
        BGIntegratorDriver(integration_hz=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bg_integrator.py -q`
Expected: FAIL. If the old windowed driver is still present it fails on the `accumulation_ms`-free constructor / new assertions; if absent, `ModuleNotFoundError`. Either way RED before the rewrite.

- [ ] **Step 3: Overwrite the driver (no window)**

Overwrite `src/nrp_bga_sb/bg_integrator.py`:

```python
"""Stateful BG integration driver (knob 2). Carries GPR activations across
integration sub-steps within a trial and reads out the current (possibly
unsettled) state, so the integration RATE controls how far the BG has settled
at any moment. This is the science-layer change that makes the
internal-integration-step knob functionally dissociable (PROJECT_MEMORY §15.7).

There is no decision-deadline window: the integrator keeps sweeping at the
integration rate for the whole trial. A slower rate therefore settles LATER
(its observable effect in the nrp pipeline is decision latency, not a
go-success miss — see the design doc, Revision 2).

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
    """Schedules one carried GPR sweep per integration-tick boundary
    (k / integration_hz seconds) across the trial, reading out the current
    (possibly unsettled) state. More sub-steps -> closer to the BGModel.compute()
    fixed point; a slow integration rate reaches it later.
    """

    integration_hz: float
    config: BGModelConfig = field(default_factory=BGModelConfig)

    def __post_init__(self) -> None:
        if self.integration_hz <= 0:
            raise ValueError(f"integration_hz must be > 0, got {self.integration_hz}")
        self._model = BGModel(self.config)
        self._period_ms = 1000.0 / self.integration_hz
        self._next_k = 0
        self._state: BGIntegratorState | None = None

    def advance(self, elapsed_ms: float, evidence: ActionEvidence) -> BGDecision:
        n = evidence.n_channels
        if self._state is None:
            self._state = BGIntegratorState.initial(n, self.config)

        saliences = evidence.channel_salience
        # Fire every integration tick whose boundary has been crossed since the last
        # call; each fires one carried sweep on the evidence current at this call.
        # No upper bound: at a low rate the few sweeps land late, so the decision
        # settles late (latency signature) rather than never.
        while self._next_k * self._period_ms <= elapsed_ms:
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
Expected: PASS (4 tests). The latency lock: `r5 > r10 > r20` (200 > 100 > 50 ms on the host).

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/bg_integrator.py tests/test_bg_integrator.py
git commit -m "refactor: BGIntegratorDriver drops window; latency-signature (no go-success regression)

ChangeSet-ID: knob2-driver-v2

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

> **Revision 2:** the driver takes no `accumulation_ms`; the engine constructs it with
> `integration_hz` only. No window constant.

```python
"""BG engine: runs the GPR BG model on the most recent sampled evidence and emits
a `decision` datapack. Delegates to BGIntegratorDriver, a stateful integrator that
carries GPR activations across emission steps within a trial and reads out the
current (possibly unsettled) state -- so the internal-integration-step rate
(knob 2) is functionally dissociable (PROJECT_MEMORY §15.7), not idempotent. Its
observable effect in this pipeline is decision latency (a slow rate settles late).

Input-sampling, emission, and commitment remain EngineTimesteps (§15.4); the
integration rate rides on the params overlay (`integration_hz`)."""

import json
import os

from nrp_core.engines.python_json import EngineScript

from nrp.serde import decision_to_dict, evidence_from_dict
from nrp_bga_sb.bg_integrator import BGIntegratorDriver


class Script(EngineScript):
    def initialize(self):
        with open(os.environ["NRP_BGA_TRIAL_PARAMS"]) as fh:
            params = json.load(fh)
        # Knob 2: BG internal integration step, scheduled by the driver from the
        # integration rate. State is created here and carried across runLoop calls
        # (one NRPCoreSim run = one trial); a slower rate settles later.
        self._driver = BGIntegratorDriver(
            integration_hz=float(params["integration_hz"]),
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
git commit -m "refactor: bg engine constructs windowless BGIntegratorDriver (latency signature)

ChangeSet-ID: knob2-engine-v2

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5 (Revision 2): nrp integration-latency experiment (the dissociation demonstration)

> **Revision 2.** The go-success ablation is metric-insensitive to this knob (nrp judges release
> continuously over 300 ms, so a slow integrator still releases — just later). So
> `nrp/results/ablation.json`, `docs/nrp_vs_prototype_comparison.md`, and `tests/nrp/test_compare.py`
> are **left UNCHANGED**. Instead, demonstrate the knob's functional signature — release LATENCY vs
> integration rate — with a new experiment + a host test for its pure extraction helper + a gated
> runtime test.

**Files:**
- Create: `experiments/nrp_integration_latency.py`
- Create: `tests/test_integration_latency.py` (host, pure helper)
- Create: `tests/nrp/test_integration_latency_nrp.py` (gated runtime, marked `nrp`)
- Generate: `nrp/results/integration_latency.json` (committed artifact, via the runtime)

**Interfaces:**
- Consumes: `build_config_four_knob` (Task 3 overlay), `run_trial`.
- Produces: `first_release_index(trace: list[dict]) -> int | None`; `measure(integration_hz: float, run_root: Path, seed: int = 0) -> int | None`; `FREQUENCIES_HZ`.

- [ ] **Step 1: Write the host test for the extraction helper (failing)**

Create `tests/test_integration_latency.py`:

```python
from experiments.nrp_integration_latency import first_release_index


def test_first_release_index_finds_first_open_gate():
    trace = [
        {"motor": None},
        {"motor": {"gate_state": "closed", "command": [0.0, 0.0]}},
        {"motor": {"gate_state": "partial", "command": [0.3, 0.0]}},
        {"motor": {"gate_state": "open", "command": [1.0, 0.0]}},
    ]
    assert first_release_index(trace) == 2


def test_first_release_index_none_when_never_open():
    trace = [{"motor": None}, {"motor": {"gate_state": "closed", "command": [0.0, 0.0]}}]
    assert first_release_index(trace) is None


def test_first_release_index_ignores_open_with_zero_command():
    # Mirrors nrp/score.py: an open gate with a zero command is NOT a release.
    trace = [{"motor": {"gate_state": "open", "command": [0.0, 0.0]}}]
    assert first_release_index(trace) is None
```

- [ ] **Step 2: Run it to confirm RED**

Run: `python -m pytest tests/test_integration_latency.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.nrp_integration_latency'`.

- [ ] **Step 3: Write the experiment module**

Create `experiments/nrp_integration_latency.py`:

```python
"""Integration knob (knob 2) nrp signature: first-release time vs integration rate.

The go-success ablation cannot show this knob: nrp judges the motor gate as released
if it EVER opens during the 300 ms sim, so even a slow integrator settles and releases
-- just later. Release LATENCY is the dissociating signal. A slower integration rate
settles later (PROJECT_MEMORY §15.7 / design doc Revision 2)."""

from __future__ import annotations

import json
from pathlib import Path

from nrp.config_gen import build_config_four_knob
from nrp.run import run_trial

HIGH = 160.0
FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]


def first_release_index(trace: list[dict]) -> int | None:
    """Index of the first trace record whose motor gate is open/partial with a
    non-zero command (same release rule as nrp/score.py). The thalamus logs one
    record per ms, so the index is a millisecond proxy for release latency."""
    for i, rec in enumerate(trace):
        m = rec.get("motor")
        if m and m.get("gate_state") in ("open", "partial") and any(m.get("command", [])):
            return i
    return None


def measure(integration_hz: float, run_root: Path, seed: int = 0) -> int | None:
    """Run one go trial with only the integration rate varied (others pinned 160 Hz)
    and return the first-release record index (~ms), or None if never released."""
    cfg, overlay = build_config_four_knob(
        input_sampling_hz=HIGH, integration_hz=integration_hz,
        output_emission_hz=HIGH, commitment_hz=HIGH)
    params = {"trial_id": seed, "seed": seed, "cue_identity": "go", **overlay}
    trace = run_trial(cfg, params, run_root / f"int{integration_hz}hz_s{seed}")
    return first_release_index(trace)


if __name__ == "__main__":
    run_root = Path("nrp/run/integration_latency")
    results = {hz: measure(hz, run_root) for hz in FREQUENCIES_HZ}
    out_path = Path("nrp/results/integration_latency.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({str(hz): results[hz] for hz in FREQUENCIES_HZ}, indent=2))
    print("first-release record index (~ms) vs integration rate (others pinned 160 Hz):")
    for hz in FREQUENCIES_HZ:
        print(f"  {hz:6g} Hz -> {results[hz]}")
    print(f"saved -> {out_path}")
```

- [ ] **Step 4: Confirm the host test passes**

Run: `python -m pytest tests/test_integration_latency.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Write the gated runtime test**

Create `tests/nrp/test_integration_latency_nrp.py`:

```python
import pytest

pytestmark = pytest.mark.nrp

from experiments.nrp_integration_latency import measure


def test_release_latency_decreases_with_rate(tmp_path):
    """The integration knob's nrp signature: slower integration settles (releases) later.
    Monotone where integration is the bottleneck (5 > 10 > 20 Hz)."""
    r5 = measure(5.0, tmp_path)
    r10 = measure(10.0, tmp_path)
    r20 = measure(20.0, tmp_path)
    assert r5 is not None and r10 is not None and r20 is not None
    assert r5 > r10 > r20
```

- [ ] **Step 6: Generate the committed snapshot + run the gated test (requires the nrp runtime)**

Run (the runtime is available in this environment; ~1 min for 6 trials — allow a long timeout):
```bash
python experiments/nrp_integration_latency.py
python -m pytest tests/nrp/test_integration_latency_nrp.py -m nrp -q
```
Expected stdout: a monotone-at-low-rate curve, approximately `5 Hz -> ~208`, `10 Hz -> ~108`, `20 Hz -> ~58`, then floors ~60–80 ms for ≥40 Hz; `nrp/results/integration_latency.json` written; the gated test passes (`r5 > r10 > r20`).

> If the nrp runtime is unavailable, STOP and report: Steps 1–5 (host) are complete; the snapshot + gated assertion need `NRPCoreSim`.

- [ ] **Step 7: Commit**

```bash
git add experiments/nrp_integration_latency.py tests/test_integration_latency.py \
        tests/nrp/test_integration_latency_nrp.py nrp/results/integration_latency.json
git commit -m "feat: integration-latency experiment demonstrates knob-2 dissociation

Release latency vs integration rate (208ms@5Hz -> 58ms@20Hz, monotone at low
rates). The go-success ablation is metric-insensitive (nrp judges release over
the full 300ms sim), so it + the comparison report are intentionally unchanged.

ChangeSet-ID: knob2-latency

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
**UPDATE (2026-06-23) — knob 2 made functional; signature is LATENCY, not go-success.**
The integration sub-step is now a stateful carried-state integrator
(`BGIntegratorDriver`, `src/nrp_bga_sb/bg_integrator.py`): GPR activations persist
across emission steps within a trial and are read out before convergence (one carried
Jacobi sweep per integration tick, no decision window), so the integration RATE controls
how far the BG has settled at any moment. `BGModel.step`/`BGIntegratorState` provide the
carried sweep; `BGModel.compute` is unchanged (full-convergence path, backs M2). The nrp
`bg` engine delegates to the driver and `config_gen` passes `integration_hz`.

The knob is no longer idempotent, but its observable effect in the nrp pipeline is
decision **latency**, not go-success rate: nrp scores release if the gate EVER opens
during the 300 ms sim, so a slow integrator still settles — just later. Measured
first-release time falls monotonically with rate where integration is the bottleneck
(~208 ms @5 Hz → ~108 @10 Hz → ~58 @20 Hz; floors ~60–80 ms ≥40 Hz), demonstrated by
`experiments/nrp_integration_latency.py` (→ `nrp/results/integration_latency.json`).

Why the go-success ablation is unchanged: forcing an integration go-success *miss* @5 Hz
needs a <200 ms decision window to exclude the 5 Hz tick at t=200 ms, but nrp's FTILoop
slow-sampler staleness delivers the 10 Hz sampler's strong evidence at ~207 ms, so that
same window regresses `sampling@10Hz` (success→miss) — irreconcilable with a scalar window.
The prototype's `integration@5Hz` miss is itself a 200 ms-deadline artifact (its tick loop
is exactly 200 ticks). So `nrp/results/ablation.json` and the comparison report are
intentionally left unchanged (the integration go-success row stays `1.0`); the latency
experiment carries the dissociation evidence. The deprecated pure-Python prototype is
untouched.
```

- [ ] **Step 2: Update PROJECT_MEMORY §1 current state**

Append a bullet to §1 recording sub-project completion (keep prior bullets verbatim):

```markdown
- **Knob-2 stateful integrator complete (2026-06-23).** The nrp integration
  sub-step knob is now functional (non-idempotent) via `BGIntegratorDriver`
  (carried-state GPR integrator, readout-before-convergence, no decision window).
  Its nrp signature is decision LATENCY, not go-success: release time falls with
  integration rate (~208 ms @5 Hz → ~58 @20 Hz), shown by
  `experiments/nrp_integration_latency.py`. The go-success ablation + comparison
  report are intentionally unchanged (the continuous-release metric can't exhibit
  the knob, and forcing it regresses sampling — see §15.7). `BGModel.compute`
  unchanged; prototype untouched.
```

- [ ] **Step 3: Run the full host suite + ruff**

Run:
```bash
python -m pytest -q -m "not nrp"
ruff check .
```
Expected: all host tests pass (the prior suite + the new Task 1/2/5 host tests); ruff clean. NOTE: `ruff check .` may surface ~22 pre-existing repo-wide errors that are NOT a merge gate (recorded in the prior ledger); the gate is that the files THIS branch touched are ruff-clean.

- [ ] **Step 4: Commit**

```bash
git add PROJECT_MEMORY.md
git commit -m "docs: record knob-2 stateful integrator completion (§15.7, §1)

ChangeSet-ID: knob2-memory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Update auto-memory**

Create `/home/fom/.claude/projects/-home-fom-code-NRP-BGA-SB/memory/project_knob2_integrator.md` noting: knob 2 made functional/non-idempotent on the nrp side via `BGIntegratorDriver` (carried-state, no window); signature is decision LATENCY not go-success (release ~208 ms @5 Hz → ~58 @20 Hz); go-success ablation + comparison report intentionally unchanged (continuous-release metric can't show it; windowed form regresses sampling — irreconcilable scalar window); prototype untouched; branch `knob2-stateful-integrator` (not pushed). Add the index line to `MEMORY.md`. (No git commit needed for the auto-memory dir.)

---

## Notes for the executor (Revision 2 — latency-signature framing)

- **Do not push.** The user pushes.
- Tasks 1, 3 are committed and unchanged (`9abd65f`, `2cd3cc0`). Tasks 2 and 4 are REWORKED in Revision 2 (driver loses its window; engine loses `ACCUMULATION_MS`) — they overwrite their prior commits with new commits. Task 5 is REPLACED (latency experiment, not ablation regen). Task 6 documents the latency finding.
- The dissociation proof that matters is **Task 2** (host, deterministic): release latency monotone-decreasing across 5 → 10 → 20 Hz (200/100/50 ms on the host). The gated Task 5 runtime test confirms it end-to-end (~208/108/58 ms).
- The nrp runtime IS available in this environment (`NRPCoreSim` on PATH, host numpy 1.26.4 < 2). The integration-latency experiment is ~1 min (6 trials × ~9 s); allow a long Bash timeout.
- `nrp/results/ablation.json`, `docs/nrp_vs_prototype_comparison.md`, and `tests/nrp/test_compare.py` are intentionally **unchanged** — do not edit them.
