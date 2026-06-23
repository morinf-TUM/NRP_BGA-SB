# Design: Functional BG integration knob (stateful incremental integrator)

**Date:** 2026-06-23
**Branch:** `knob2-stateful-integrator`
**Sub-project:** Make BG frequency knob 2 (internal integration sub-step) functionally
dissociable.

---

## Revision 2 (2026-06-23) — AUTHORITATIVE: latency-signature framing

During execution, an empirical NRPCoreSim probe overturned the original "match the
go-success boundary" target. **This section governs; the boundary-match material below
(Revision 1) is retained for history but superseded where it conflicts.**

**Finding.** The integration knob *is* now functional (stateful, non-idempotent — Tasks 1–3
stand), but in the nrp pipeline its signature is **decision latency (settling time), not
go-success rate**:

- nrp scores success if the motor gate *ever* opens during the 300 ms sim (`nrp/score.py`),
  not at a 200 ms deadline. A slow integrator therefore still settles and releases — just
  *later*. Measured first-release time vs integration rate (non-clipping window):
  `~208 ms @5 Hz → 108 @10 Hz → 58 @20 Hz` (monotone where integration is the bottleneck;
  floors ~60–80 ms above 20 Hz where the cortex ramp dominates).
- Forcing a go-success *miss* at 5 Hz requires a `<200 ms` window to exclude the 5 Hz
  integration tick at t=200 ms. But nrp's FTILoop slow-sampler staleness delivers the 10 Hz
  sampler's strong evidence to the BG at **~207 ms**, so the same window also muzzles the
  (fast) integrator on the *sampling* ablation → `sampling@10Hz` regresses success→miss.
  The two decisive events straddle 200 ms in opposite directions (probe: `int@5Hz` release
  ~208 ms vs `sampling@10Hz` ~214 ms); **no scalar window separates them.** The prototype's
  `integration@5Hz` miss is itself an artifact of its tick loop being exactly 200 ticks.

**Decision (user-approved).** Adopt the latency-signature framing:

- Run the BG engine **without** the muzzling accumulation window. Consequence: `sampling`,
  `emission`, `commitment` go-success rows are preserved (no regression), and the
  `integration` go-success row stays `1.0` across the grid — so **`nrp/results/ablation.json`,
  `docs/nrp_vs_prototype_comparison.md`, and `tests/nrp/test_compare.py` are UNCHANGED**.
- Demonstrate knob-2 functionality with a new experiment measuring **first-release time vs
  integration rate** (the dissociation), plus the host-deterministic non-idempotence/latency
  test.
- `BGModel.step`/`BGIntegratorState` (Task 1) and the `integration_hz` overlay (Task 3) are
  unchanged. The `BGIntegratorDriver` (Task 2) drops its `accumulation_ms` window; the engine
  (Task 4) constructs it with `integration_hz` only.

**§15.7 narrative to record:** knob 2 is no longer idempotent — it is a stateful carried
integrator whose nrp-observable effect is settling latency. The go-success ablation cannot
exhibit it because nrp judges release continuously over 300 ms (and the prototype's go-success
miss@5 Hz is a 200 ms-deadline artifact, not a settling effect).

The components / testing / risks below are updated inline for Revision 2; the "Why the
boundary matches" table and the 200 ms-window mechanics are **superseded** (kept for history).

---

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

### Why the boundary matches the other three knobs (no tuning) — SUPERSEDED by Revision 2

> **Superseded:** the host model below is correct, but the nrp runtime judges release
> continuously over 300 ms (not at a 200 ms deadline), and the windowed mechanism that would
> reproduce a go-success miss@5 Hz regresses the sampling knob. See Revision 2. Retained for history.

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

> **Revision 2:** the driver has **no accumulation window** (the windowed form caused the
> sampling regression). It schedules one carried sweep per integration tick across the whole
> trial; the readout settles progressively, so the integration rate controls settling latency.

The nrp engine file imports `nrp_core` and cannot run on the host, so the schedule + stepping
logic lives in a pure module the engine delegates to (the established pattern: engines delegate
to validated `src/nrp_bga_sb/` science). `BGIntegratorDriver`:

- Owns a `BGModel`, a carried `BGIntegratorState`, and `integration_hz`.
- `advance(elapsed_ms, evidence) -> BGDecision`: for every integration-tick boundary
  (`k / integration_hz` seconds) crossed since the last call, performs **one** carried Jacobi
  sweep on `evidence`; then returns the current readout as a `BGDecision` (reusing
  `selection_latency_s`). No upper bound — a slow rate simply settles later.
- Pure Python, so host tests feed it a deterministic `(elapsed_ms, evidence)` sequence from
  `CortexEvidenceGenerator` and assert the **latency signature**: the first elapsed time at
  which `decision_margin` clears `margin_threshold` is monotone-decreasing across 5 → 10 → 20 Hz,
  and the integrator is non-idempotent (a second sweep changes the readout). Independent of the
  nrp runtime.

### `nrp/engines/bg_engine.py` — real-runtime engine (thin delegation)

- Construct one `BGIntegratorDriver(integration_hz=float(params["integration_hz"]))` in
  `initialize` (state re-initialises at trial start; one NRPCoreSim run = one trial). No
  `ACCUMULATION_MS`.
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

## Testing (TDD, red-first) — Revision 2

- **Unit (`bg_model`):** `step` is non-idempotent (margin grows with sweeps); converges to the
  `compute()` fixed point within tolerance; zero-state readout = "channel 0, margin 0"; `compute()`
  output unchanged (exact equality on the canonical conflict levels). *(Task 1 — done.)*
- **Unit (`bg_integrator`) — the host-deterministic latency lock:** driving `BGIntegratorDriver`
  (no window) with the `CortexEvidenceGenerator` ramp at the 160 Hz emission cadence, the first
  elapsed time at which `decision_margin` clears `margin_threshold` is **monotone-decreasing across
  5 → 10 → 20 Hz**, and every rate eventually clears (functional, not idempotent). Runtime-independent.
- **config_gen:** `build_config_four_knob` overlay carries `integration_hz`. *(Task 3 — done.)*
- **nrp binding (marked `nrp`, gated runtime):** a new `experiments/nrp_integration_latency.py`
  measures first-release time vs integration rate through NRPCoreSim and asserts the monotone
  settling-latency trend at low rates (e.g. `5 Hz > 10 Hz > 20 Hz`). `nrp/results/ablation.json`,
  `docs/nrp_vs_prototype_comparison.md`, and `tests/nrp/test_compare.py` are **unchanged** (the
  go-success ablation is metric-insensitive to this knob — see Revision 2).
- **Guard:** full host suite stays green; ruff clean.

## Key risks — Revision 2

- **(Realised, now resolved by the pivot.)** The original windowed design regressed `sampling@10Hz`
  because nrp's slow-sampler staleness collides with the 5 Hz integration period at ~200 ms. The
  latency-signature framing removes the window, so there is no go-success regression: the
  sampling/emission/commitment rows and the whole comparison report are byte-unchanged.
- The latency experiment runs in the nrp runtime (numpy<2, `NRPCoreSim -d <repo_root>`). First-release
  time has discrete jitter from tick/emission alignment at high rates (≥40 Hz it floors ~60–80 ms);
  the assertion targets the clean monotone low-rate region (5 > 10 > 20 Hz), not every adjacent pair.
- `BGModel.compute()` stays untouched → M2 and all host/closed-loop tests unaffected.

## Out of scope

- The deprecated pure-Python prototype (`ScheduledBGAdapter`, `deprecated_toy_prototype_results/`)
  is **not** changed — see "Revised scope" above.
- No new BG biophysics beyond carried-state Jacobi (no leaky-Euler relaxation parameter, no
  evidence-fraction accumulation — both considered and rejected in brainstorming).
- No changes to paradigms, plants, or perturbation TFs.
