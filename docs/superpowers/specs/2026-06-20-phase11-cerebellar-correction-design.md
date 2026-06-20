# Phase 11 — Cerebellar trajectory correction (Milestone M9)

**Date:** 2026-06-20
**Status:** Approved design, pre-implementation
**Milestone:** M9 — Cerebellar correction
**Predecessor state:** Phase 10 (M8, OpenSim Arm26 embodiment) complete and merged to master @4a7550f; 685 host tests passing, ruff clean.

---

## 1. Goal and acceptance

**Goal (IMPLEMENTATION_PLAN.md Phase 11):** add trajectory correction without erasing BG-dependent
selection effects.

**Acceptance (M9):** the cerebellar module improves movement accuracy or correction timing under
perturbation, *and* the BG-frequency selection signature survives its addition.

Concretely, this design satisfies M9 with two pieces of evidence:

1. **Accuracy improves under perturbation** — both *within a trial* (online forward-model feedback)
   and *across trials* (trial-by-trial LMS adaptation produces a learning curve toward an asymptote).
2. **BG selection effects survive** — the kinematic frequency sweep's onset-rate-vs-frequency profile
   (0.0 at 5 Hz, 1.0 at ≥10 Hz; PROJECT_MEMORY §22.7) is identical with the cerebellum on or off.

---

## 2. Scope decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Plant target | **Kinematic reacher (host-side)** | Fast, fully CI-testable, no Docker; follows the project's "validate kinematically before embodiment" ladder (PROJECT_MEMORY §3). OpenSim confirmation deferred to an optional later sub-phase / Phase 12 writeup. |
| Correction type | **Both** trial-by-trial adaptation **and** within-trial online feedback | Implemented as two separable, independently-ablatable layers. Makes the module a genuine cerebellar analog rather than a single-trick corrector. |
| Perturbation | **Visuomotor rotation** (fixed angle θ, default 30°) | The canonical cerebellar/sensorimotor adaptation paradigm (Martin, Keating, Goodkin, Bastian; Tseng et al. 2007). Corrective signal is a counter-rotation angle; natural for a 2D endpoint reach. |
| Cerebellar model | **Cerebellar adaptive-filter**: Widrow-Hoff/LMS adaptation + forward-model online feedback | The canonical minimal cerebellar model (Marr–Albus–Ito lineage; Fujita 1982; Dean, Porrill & Stone, *Nat Rev Neurosci* 2010). Both layers reduce to simple, transparent, independently-testable math. |

Rejected alternatives (documented for the record):
- **OpenSim plant** for Phase 11: most faithful but Docker-gated, slow, correction lives across the
  container boundary, and confounds learning with PD-controller dynamics. Breaks the validation ladder.
- **Two-rate state-space adaptation** (Smith, Ghazizadeh & Shadmehr 2006): more faithful to human
  curves but more parameters than "minimal" warrants and covers only the adaptation half.
- **Pure Smith predictor / internal forward model** (Miall et al. 1993): elegant for online feedback,
  but trial-by-trial adaptation becomes implicit in model weights, making the learning-curve
  demonstration harder to interpret.

---

## 3. Architecture and module boundaries

All new units are host-side and operate strictly *downstream* of the existing
BG → thalamus → motor-command pipeline. Nothing in the BG, cortex, thalamus, scheduler, or engine
code changes.

```
src/nrp_bga_sb/
    perturbation_plant.py   — VisuomotorRotation: injects a fixed rotation θ into an executed reach
    cerebellum.py           — AdaptiveFilter (trial-by-trial LMS)
                              + ForwardModelController (within-trial online feedback)
                              + Cerebellum (composes the two layers, with independent enable flags)
    reacher.py  (EXTENDED)  — KinematicReacher.simulate_with_correction(...): runs perturbed +
                              optionally cerebellum-corrected trajectories; no-movement trials untouched
    movement_metrics.py (EXTENDED) — angular_error helper; adaptation-curve aggregation
    cerebellum_sweep.py     — CerebellumSweepResult, run_cerebellum_condition
                              (cerebellum on/off × frequency, reusing the §22 sweep design)

experiments/
    cerebellum_adaptation.py — learning-curve demo + cerebellum on/off frequency sweep;
                               writes results/cerebellum_results.json
```

**Single governing invariant:** the cerebellum consumes `(motor_command_series, gate_state, target)`
from a reach and returns a corrected trajectory. It can change *where / how accurately* a movement
lands — never *whether* a movement occurs. This invariant is what protects the BG selection signature
(see §6).

### Unit responsibilities (isolation check)

- **`VisuomotorRotation`** — *what:* distorts an executed reach direction by θ. *how used:*
  `rotate(direction_vector) -> rotated_vector`. *depends on:* numpy only.
- **`AdaptiveFilter`** — *what:* learns a feedforward counter-rotation across trials. *how used:*
  `apply(command) -> corrected_command`, `update(angular_error)`. *depends on:* nothing (scalar state).
- **`ForwardModelController`** — *what:* nudges a trajectory toward target within a single trial.
  *how used:* per-step `correct(predicted, actual, target) -> correction`. *depends on:* numpy only.
- **`Cerebellum`** — *what:* composes the two layers behind one interface with independent enable
  flags. *how used:* by `simulate_with_correction`. *depends on:* the two layers above.
- **`KinematicReacher.simulate_with_correction`** — *what:* executes a (possibly perturbed, possibly
  corrected) reach. *depends on:* existing reacher machinery + the perturbation + the cerebellum.

---

## 4. Components

### 4.1 `VisuomotorRotation` (perturbation / plant distortion)

- Applies a fixed rotation θ (default 30°, configurable) to the commanded reach vector about the
  start point. This is the systematic error the cerebellum must overcome.
- θ = 0 reduces exactly to the unperturbed reacher (identity), giving a clean baseline.
- Fail-fast: validates θ is finite.

### 4.2 `AdaptiveFilter` (trial-by-trial adaptation layer)

- Holds a scalar counter-rotation estimate `θ̂` (radians), initialised to 0.
- After each *executed* reach, observes the endpoint angular error `e` (signed angle between desired
  and achieved reach direction) and updates by the Widrow-Hoff / LMS delta rule:
  `θ̂ ← θ̂ + α · e` (learning rate α, default ≈ 0.1, 0 < α ≤ 1).
- Provides a feedforward counter-rotation applied to the command *before* execution: `apply` rotates
  the command direction by `−θ̂`, so the residual executed rotation is `(θ − θ̂)` and the signed
  angular error is `e = θ − θ̂`. The LMS update `θ̂ ← θ̂ + α·e` therefore drives `θ̂ → θ` and `e → 0`.
- Stateful across trials within a block; `reset()` clears `θ̂` between blocks / seeds.
- Mechanistic mapping: climbing-fibre error signal → parallel-fibre weight update, reduced to its
  simplest scalar form (Fujita 1982; Dean, Porrill & Stone 2010).
- Yields an exponential learning curve: `θ̂_n → θ` (and angular error → 0) over trials.
- Fail-fast: validates 0 < α ≤ 1.

### 4.3 `ForwardModelController` (within-trial online feedback layer)

- During trajectory integration, a forward model predicts the *intended* (undistorted) position at
  each step; the controller adds a proportional corrective term (gain `k`) steering the actual
  (perturbed) position back toward the intended path / target.
- Stateless across trials (no learning); operates purely within one trial.
- `k = 0` ⇒ exact no-op (recovers the uncorrected perturbed trajectory) — used as a test baseline.
- Reduces endpoint error within a single trial even before adaptation has converged.
- Mechanistic mapping: internal forward model + delay-free feedback (Miall & Wolpert; Miall et al.
  1993 Smith-predictor lineage), in its simplest proportional form.

### 4.4 `Cerebellum` (composition)

- Composes both layers behind one interface, with independent boolean flags
  (`adaptation_enabled`, `online_enabled`) so each layer can be ablated and unit-tested in isolation,
  and so the on/off sweep is a single code path.
- Holds the `AdaptiveFilter` instance (stateful) and constructs/uses the `ForwardModelController`.

---

## 5. Data flow

```
ClosedLoopPolicy → motor_command_series + gate_state   (UNCHANGED Phase 4–6 pipeline)
        │
        ▼
KinematicReacher.simulate_with_correction(commands, gate_state, target, perturbation, cerebellum)
        │
        ├─ gate closed / no movement onset?  → return zero-movement trajectory
        │                                       (cerebellum is NEVER invoked — the guard, §6)
        │
        └─ movement executed:
              1. feedforward: command direction ← AdaptiveFilter.apply(command)     [if adaptation on]
              2. VisuomotorRotation distorts the executed direction by θ
              3. integrate the min-jerk trajectory step-by-step;
                 ForwardModelController nudges each step toward intended path        [if online on]
              4. compute endpoint, signed angular error
              5. AdaptiveFilter.update(angular error)  → learning for the next trial [if adaptation on]
        │
        ▼
compute_movement_metrics → endpoint_error, angular_error (+ adaptation-curve aggregation across trials)
```

When the perturbation is absent (θ = 0) and the cerebellum is off, `simulate_with_correction`
reproduces the existing `KinematicReacher.simulate` behaviour, preserving backward comparability with
the Phase 6 sweep.

---

## 6. The BG-effect guard (M9 scientific crux)

The cerebellum is invoked **only when a movement is executed** — gate non-closed *and* movement onset
present. On a 5 Hz miss trial the thalamic gate is closed (PROJECT_MEMORY §20.6, §22.6) → no movement
onset → no trajectory is integrated → the cerebellum is never called. Therefore `movement_onset_rate`
and `go_success_rate` as a function of BG frequency are **structurally identical** with the cerebellum
on or off: the corrector cannot manufacture a reach the BG never released.

This is not a tuning result; it is a property of where the module sits in the pipeline. The cerebellum
can only reshape trajectories that the BG → thalamus stage already produced.

**Demonstration (Task 11.3):** re-run the kinematic frequency sweep (the §22 design: 5/10/20/40/80 Hz)
with the cerebellum off vs on under the visuomotor rotation. Selection metrics (onset rate per
frequency) must be *bit-identical* between the two; only accuracy metrics (endpoint / angular error)
differ.

---

## 7. Error handling and conventions

House-style fail-fast (PROJECT_MEMORY §12, §16.4):

- `VisuomotorRotation` validates θ finite; `AdaptiveFilter` validates 0 < α ≤ 1; both raise `ValueError`.
- `simulate_with_correction` keeps the existing wiring-error guard (non-closed gate with an all-zero
  command vector raises `ValueError`; PROJECT_MEMORY §22.2).
- No silent fallbacks, no speculative `getattr`, no broad exception swallowing.
- The two cerebellar layers are explicit narrow objects exposing `apply`/`update` and `correct`
  capabilities — not attribute-sniffed.
- Section-header comments (`# --- ... ---`) on each multi-section module; decision-point comments
  (Trigger / Why / Outcome) on the gate-skip branch (§6 guard) and the LMS update.
- Pydantic v2 for any new result schema (`CerebellumSweepResult`), `X | None` unions, `Literal[...]`
  for fixed vocabularies, consistent with §16.4.

---

## 8. Testing strategy and acceptance evidence

All host-side, in the normal pytest suite (no Docker gating):

| Unit | Tests |
|---|---|
| `AdaptiveFilter` | LMS convergence (`θ̂ → θ`); monotone angular-error decay over trials; α-bounds validation; `reset()` semantics. |
| `ForwardModelController` | within-trial endpoint error reduced vs uncorrected on a single perturbed reach; `k = 0` ⇒ exact no-op. |
| `VisuomotorRotation` | θ = 0 identity; known-angle endpoint geometry (analytic check). |
| `Cerebellum` | independent enable flags; composition routes to the right layers. |
| `simulate_with_correction` | closed-gate / miss trial untouched (cerebellum not invoked — the guard); executed trial corrected; θ = 0 + cerebellum off reproduces `simulate`. |
| `cerebellum_sweep` | adaptation: mean error decreases over a trial block; online: error lower with cerebellum on than off; **selection guard:** onset-rate-vs-frequency identical cerebellum on vs off. |

**M9 acceptance, demonstrated by the sweep + learning-curve experiment:**

- (a) endpoint / angular error improves under the visuomotor rotation — within-trial (online layer)
  and across trials (adaptation learning curve);
- (b) the BG-frequency onset signature is unchanged by the cerebellum.

Target ≈ 30+ new tests; the full suite stays green (baseline 685) and ruff clean.

---

## 9. Out of scope (YAGNI)

- OpenSim embodiment of the cerebellum — **explicitly deferred to Phase 11b** (IMPLEMENTATION_PLAN.md),
  which re-runs this perturbation + corrector through the Dockerized Arm26 plant on the same BG
  decisions, mirroring the Phase 6 → Phase 10 kinematic → OpenSim step. Not optional; scheduled.
- Two-rate / multi-state adaptation dynamics (savings, interference).
- Adapting to perturbation *types* other than visuomotor rotation (gain, bias) — the interfaces
  should not preclude it, but only rotation is built and tested in Phase 11.
- Any change to BG, cortex, thalamus, scheduler, or task-engine code.

---

## 10. Literature anchors (to cite in PROJECT_MEMORY update)

- Cerebellar adaptive-filter model: Fujita M. (1982) *Biol Cybern*; Dean P., Porrill J., Stone J.V.
  (2010) "The cerebellar microcircuit as an adaptive filter", *Nat Rev Neurosci*.
- Internal forward model / online correction: Miall R.C., Weir D.J., Wolpert D.M., Stein J.F. (1993)
  "Is the cerebellum a Smith predictor?", *J Mot Behav*; Wolpert, Miall & Kawato (1998).
- Visuomotor rotation adaptation paradigm: Martin T.A. et al. (1996) *Brain*; Tseng Y.W. et al.
  (2007) *J Neurophysiol*.

These extend PROJECT_MEMORY §10's existing literature anchors; this is the first cerebellar reference
set in the project.
