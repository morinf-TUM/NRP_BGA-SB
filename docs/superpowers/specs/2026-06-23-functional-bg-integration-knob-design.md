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

A host simulation of the proposed driver against the real cortex ramp confirms this exactly,
with the accumulation window taken as the half-open interval `[0, accumulation_ms)` so the
5 Hz tick at `t = 200 ms` is excluded (mirroring the prototype's `range(200)`):

| integration rate | integration-tick times (ms) within `[0,200)` | released? |
|---|---|---|
| 5 Hz | `{0}` (neutral) | **no → miss** |
| 10 Hz | `{0, 100}` (100 ms risen → margin 0.20) | yes → success |
| 20 Hz | `{0, 50, 100, 150}` | yes → success |
| ≥40 Hz | … | yes → success |

The nrp scorer (`nrp/score.py`) reports success if the gate *ever* opens during the run, so a
miss requires the readout margin to stay below `margin_threshold` at **every** emission readout
throughout the trial — which holds at 5 Hz because the single early sweep on neutral evidence
never advances and no later integration tick fires inside the window.

## Revised scope (nrp-only)

During planning, inspection of the committed results showed the divergence is **entirely on the
nrp side**:

- **Prototype** integration knob already shows `0.0 @ 5 Hz → 1.0 @ ≥10 Hz` — but via a
  *gate-timing / evidence-staleness* artifact (at 5 Hz the integration gate fires once at tick 0
  on neutral evidence, full-converges, and never re-fires), not via incomplete settling.
- **nrp** integration knob is `1.0` across the entire 5–160 Hz grid — the sole inert knob and the
  sole source of the comparison divergence.

The fix is therefore applied to the **nrp side only**. The deprecated pure-Python prototype
(`ScheduledBGAdapter`, `deprecated_toy_prototype_results/`) is left frozen and untouched:
migrating its Gate 2 to the stateful stepper would change its mechanism (settling vs. staleness),
is not behaviour-preserving at intermediate rates, and would risk the 723-test hot path for **zero
change to the comparison outcome**. The shared `BGModel.step` science is still introduced and used
by nrp.

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

### `src/nrp_bga_sb/bg_integrator.py` — host-testable stateful driver (new)

The nrp engine file imports `nrp_core` and cannot run on the host, so the schedule + stepping
logic lives in a pure module the engine delegates to (the established pattern: engines delegate
to validated `src/nrp_bga_sb/` science). New `BGIntegratorDriver`:

- Owns a `BGModel` and a carried `BGIntegratorState`, plus `integration_hz` and `accumulation_ms`.
- `advance(elapsed_ms, evidence) -> BGDecision`: if a new integration-tick boundary
  (`k / integration_hz` seconds, for `k·period < accumulation_ms`) has been crossed since the
  last call, performs **one** carried Jacobi sweep on `evidence`; then returns the current readout
  as a `BGDecision` (reusing the existing latency mapping). Outside the window it advances no
  further and returns the held readout.
- Because it is pure Python, host tests feed it a deterministic `(elapsed_ms, evidence)` sequence
  drawn from `CortexEvidenceGenerator` and **lock the boundary** (`5 Hz` never clears margin;
  `10 Hz` clears at the 100 ms tick) — independent of the nrp runtime.

### `nrp/engines/bg_engine.py` — real-runtime engine (thin delegation)

- Construct one `BGIntegratorDriver(integration_hz, accumulation_ms)` in `initialize` (state
  re-initialises at trial start; one NRPCoreSim run = one trial).
- In `runLoop`, pass `elapsed_ms = self._time_ns / 1e6` and the current `sampled_evidence` to
  `driver.advance(...)`, then emit the returned `BGDecision`.
- The `for _ in range(self._substeps)` idempotent loop and its NOTE comment are removed.

### `nrp/config_gen.py` — overlay carries the rate, not pre-baked sub-steps

- `build_config_four_knob` currently pre-computes the degenerate
  `substeps = max(1, round(integration_hz / output_emission_hz))` (which collapses to 1 across the
  whole grid). Replace the overlay with `{"integration_hz": integration_hz}` (and the accumulation
  window if not defaulted), so the engine's driver schedules sweeps from the rate directly,
  decoupled from emission.

## Backward-compatibility strategy

There is **one** code path (the stateful integrator) — no mode flag, no conditional fallback.
Compatibility is a property of its *output*, not of branching:

- `BGModel.compute()` is **untouched** (additive `step()` only) → **M2 validation stays exact**
  (13 ms low-conflict latency, correct channel, monotone-with-conflict), and every host/closed-loop
  test that calls `compute()`/`BGAdapter`/`ScheduledBGAdapter` is unaffected because the prototype
  path is not modified (nrp-only scope).
- The new behaviour is confined to the nrp `bg` engine via `BGIntegratorDriver`. Within that path,
  at the **baseline** integration rate the carried state has **fully converged** before the
  deadline, so its readout equals `compute()`. Consequently the nrp sampling / emission /
  commitment ablations (which hold integration at baseline `HIGH = 160 Hz`) keep identical numbers;
  only the nrp **integration** ablation changes — exactly the row this sub-project fixes.
- A starved integrator reads out early (miss); a well-fed one lands on the same fixed point.

## Testing (TDD, red-first)

- **Unit (`bg_model`):** `step` is non-idempotent (margin grows with sweeps); converges to the
  `compute()` fixed point within tolerance; zero-state readout = "channel 0, margin 0"; `compute()`
  output unchanged (exact equality on the canonical conflict levels).
- **Unit (`bg_integrator`) — the host-deterministic boundary lock:** driving
  `BGIntegratorDriver` with the `CortexEvidenceGenerator` ramp yields **miss at 5 Hz** (margin never
  reaches `margin_threshold`) and **success at 10/20/40 Hz** (margin clears at the 100/50/… ms
  tick). This is the primary, runtime-independent proof that the knob is functional and matches the
  boundary.
- **config_gen:** `build_config_four_knob` overlay now carries `integration_hz`; existing
  config-shape tests updated.
- **nrp binding (marked `nrp`, gated runtime):** regenerate `nrp/results/ablation.json` via the
  gated `experiments/nrp_ablation.py`; assert the integration row is now `0.0 @ 5 Hz → 1.0 @
  ≥10 Hz` and the other three rows are unchanged. Update `tests/nrp/test_compare.py` assertions that
  currently encode the divergence, and `nrp/compare.py` verdict text (`3 of 4` → `4 of 4`).
  Regenerate `docs/nrp_vs_prototype_comparison.md` so **all four knobs hold**.
- **Guard:** full host suite stays green; ruff clean.

## Key risks

- The exact nrp **runtime timing** (FTILoop step ordering, datapack propagation delay) is empirical
  — it is what makes emission/commitment miss at 5 Hz today. The host driver boundary lock removes
  most of this risk by proving the science deterministically; the gated `nrp` ablation then
  validates that the engine feeds the driver the right `(elapsed_ms, evidence)` sequence. If the
  runtime timing shifts the integration boundary, the `accumulation_ms` window / tick convention is
  the single calibration point, pinned by the gated ablation assertion.
- The gated `nrp` tests require the `.nrp_env` runtime (numpy<2, `source .nrp_env`, `-d <repo_root>`)
  and are deselected by default; regenerating `nrp/results/ablation.json` must be run in that
  environment.

## Out of scope

- The deprecated pure-Python prototype (`ScheduledBGAdapter`, `deprecated_toy_prototype_results/`)
  is **not** changed — see "Revised scope" above.
- No new BG biophysics beyond carried-state Jacobi (no leaky-Euler relaxation parameter, no
  evidence-fraction accumulation — both considered and rejected in brainstorming).
- No changes to paradigms, plants, or perturbation TFs.
