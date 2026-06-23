# Design: Functional BG integration knob (stateful incremental integrator)

**Date:** 2026-06-23
**Branch:** `knob2-stateful-integrator`
**Sub-project:** Make BG frequency knob 2 (internal integration sub-step) functionally
dissociable.

## Problem

PROJECT_MEMORY §15.7 and `docs/nrp_vs_prototype_comparison.md` record that knob 2
(internal integration sub-step) is **idempotent** on the current BG solver: of the four
frequency knobs, only three (input sampling, output emission, commitment) are functionally
dissociable. The integration knob diverges in the comparison (prototype go-success `0.0`
vs nrp `1.0` at 5 Hz) precisely because it changes nothing.

There are **two compounding causes**:

1. **Stateless solver.** `BGModel.compute()` (`src/nrp_bga_sb/bg_model.py`) re-initialises
   `STN/GPe/GPi = 0` and runs Jacobi iteration **to full convergence** on every call.
   Calling it N times on identical evidence returns the identical fixed point — so both
   the nrp sub-step loop (`nrp/engines/bg_engine.py`, `for _ in range(self._substeps)`)
   and the prototype Gate 2 (`src/nrp_bga_sb/scheduler.py`, integration gate calling
   `base_policy`) are idempotent.
2. **Degenerate rate→work mapping.** `nrp/config_gen.py` maps
   `substeps = max(1, round(integration_hz / output_emission_hz))`. With output emission
   held at its 160 Hz baseline during the integration ablation, every integration rate from
   5→160 Hz yields `substeps = 1` (ratio ≤ 1 rounds to 0, clamped to 1). The knob would be
   inert across the whole grid even if a single sub-step were functional.

## Core idea

Introduce a **carried integrator state** `(STN, GPe, GPi)` that persists across sub-steps
*within a single trial*, advanced **one bounded Jacobi sweep at a time**, with the decision
**read out from the current (possibly unsettled) state**. The integration rate then controls
*how many sweeps have been applied — and on which evidence — by the decision deadline*:

- Low rate → few sweeps, landing on early/neutral cortical evidence → small `decision_margin`
  → thalamic gate stays closed → **miss**.
- High rate → enough sweeps on ramped evidence → margin clears `margin_threshold` →
  **success**.

The steady state of the carried Jacobi iteration is **identical** to today's `compute()`
fixed point (same equations, same convergence target). Therefore running to convergence
reproduces all current results exactly; only reading out *before* convergence is new.

### Why the boundary matches the other three knobs (no tuning)

Numeric trajectory of carried-Jacobi `decision_margin` per sweep, against the actual cortex
evidence ramp (`peak_salience=0.9`, `rise_time_ms=100`, `accumulation_ms=200`,
thalamic `margin_threshold=0.05`):

- Medium conflict `[0.65, 0.35]`: sweep 1 → margin `0.000` (miss); sweep 2 → `0.074`
  (clears 0.05 → success). Boundary at **1→2 sweeps**.
- With the **progressive** evidence ramp, the timing of sweeps matters: at low integration
  rate the single integration sweep lands at `t≈0` on neutral `[0.5,0.5]` evidence
  → margin 0 → miss; at higher rate a later sweep lands on ramped `[0.9,0.1]` evidence
  → margin clears → success.

This is the **same machinery** that already produces the 5 Hz boundary for the other three
knobs (cortical ramp + gate timing). The `0.0 @ 5 Hz → 1.0 @ ≥10 Hz` boundary therefore
emerges naturally, with no artificial threshold tuning.

## Components and changes

### `src/nrp_bga_sb/bg_model.py` — additive stateful stepper

- New `BGIntegratorState`: holds the carried `STN/GPe/GPi` arrays (zero-initialised) and the
  derived readout (`selected_channel`, `decision_margin`, `T_winner`, `suppression_vector`,
  `channel_activations`, `n_iters`).
- New `BGModel.step(state, saliences, n_sweeps) -> BGIntegratorState`: performs `n_sweeps`
  carried Jacobi sweeps on the supplied evidence and refreshes the readout from the **current**
  state. The existing zero-state readout convention applies (`GPi=0 → T=thal_threshold` for
  all channels → "channel 0, margin 0").
- `BGModel.compute()` is **left untouched** as the canonical full-converge path. It remains
  the path used by M2 validation and all direct-call policy paths; `compute()` is equivalent
  to `step` on a fresh state iterated until convergence.

### `src/nrp_bga_sb/scheduler.py` — prototype Gate 2

- Gate 2 (integration) advances a **per-call** carried integrator by **one sweep** on
  `last_sampled_evidence`, instead of calling `base_policy` to full convergence.
- Because the tick loop runs at `base_dt_ms = 1 ms`, cumulative sweeps by the deadline
  ≈ `integration_step_hz × accumulation_s` — decoupled from emission rate → **dissociable**.
- The final readout is converted to `BGDecision` (incl. the existing latency mapping) at the
  end of the call. The adapter's documented cross-call statelessness is preserved: integrator
  state lives only within one `__call__`.

### `nrp/engines/bg_engine.py` — real-runtime engine

- Carry the integrator state on the `Script` instance across `runLoop` calls (one NRPCoreSim
  run = one trial; state re-initialises at trial start in `initialize`).
- Drive the number of sweeps performed per `runLoop` off **elapsed sim time** so cumulative
  sweeps ≈ `integration_hz × elapsed_s`, mirroring the prototype tick loop. Each sweep uses
  the then-current `sampled_evidence` datapack.
- The `for _ in range(self._substeps)` idempotent loop and its NOTE comment are replaced.

### `nrp/config_gen.py` — rate→work mapping

- Replace the degenerate `substeps = max(1, round(integration_hz / output_emission_hz))` with
  an **integration-rate-driven** schedule decoupled from emission, so cumulative sweeps track
  `integration_hz × accumulation_s`. The exact parameterisation (e.g. an elapsed-time sweep
  target, or fractional sweep credit per emission window) is pinned by the ablation regression
  test so that the nrp side reproduces the prototype boundary.

## Backward-compatibility strategy

There is **one** code path (the stateful integrator) — no mode flag, no conditional fallback.
Compatibility is a property of its *output*, not of branching:

- `compute()` is untouched → **M2 validation stays exact** (13 ms low-conflict latency, correct
  channel, monotone-with-conflict).
- At the **baseline** integration rate (default `integration_step_hz = 1000 Hz`, ~200 sweeps
  over a 200 ms window) the carried state has **fully converged** before the deadline, so its
  readout is bit-identical to `compute()`. Consequently:
  - the sampling / emission / commitment ablations (which hold integration at baseline) keep
    identical numbers;
  - the 723 existing closed-loop tests, which use the default config or `from_effective_hz`
    at ≥10 Hz, are preserved.
- Behaviour changes **only** for explicitly lowered integration rates — exactly the integration
  ablation this sub-project intends to fix. A starved integrator reads out early; a well-fed one
  lands on the same fixed point.

## Testing (TDD, red-first)

- **Unit (`bg_model`):** `step` is non-idempotent (margin grows with sweeps); converges to the
  `compute()` fixed point within tolerance; zero-state readout = "channel 0, margin 0".
- **Regression-lock:** the medium-conflict 1→2-sweep margin boundary; baseline-rate convergence
  equals `compute()` for the canonical conflict levels.
- **Prototype scheduler:** integration ablation over the standard Hz grid now shows
  `0.0 @ 5 Hz → 1.0 @ ≥10 Hz`; the other three knobs unchanged.
- **nrp binding (marked `nrp`):** the integration ablation now matches the prototype boundary;
  regenerate `docs/nrp_vs_prototype_comparison.md` so **all four knobs hold** and update the
  verdict; update `tests/nrp/test_compare.py` (currently asserts the divergence).
- **Guard:** full host suite stays green; ruff clean.

## Key risks

- `FrequencyConfig.from_effective_hz` co-varies integration with the other knobs, so the
  Phase 5 frequency sweep, stop-signal, kinematic, and cerebellum experiments exercise the
  stateful path at low rates. The boundary is dominated by sampling/emission, so the headline
  signatures should be unchanged — but each must be verified green; TDD red-first on any that
  shift.
- nrp sweep-scheduling (elapsed-time target vs fractional credit) must be calibrated so the nrp
  ablation matches the prototype; this is pinned by the nrp ablation regression test.

## Out of scope

- No new BG biophysics beyond carried-state Jacobi (no leaky-Euler relaxation parameter, no
  evidence-fraction accumulation — both considered and rejected in brainstorming).
- No changes to paradigms, plants, or perturbation TFs.
