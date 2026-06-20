# Phase 10 Design — Dockerized OpenSim Arm26 Embodiment (Milestone M8)

**Date:** 2026-06-20
**Status:** Approved design (pre-implementation)
**Milestone:** M8 — same qualitative BG-frequency effects survive embodiment
**Plan reference:** `IMPLEMENTATION_PLAN.md` Phase 10; `PROJECT_MEMORY.md` §15.5 (OpenSim is runtime-only), §22 (kinematic reacher)

---

## 1. Goal & acceptance

**Goal:** Re-run the go/no-go frequency sweep on a torque-controlled OpenSim Arm26 musculoskeletal arm and confirm the BG-frequency effect demonstrated on the kinematic reacher (Phase 6) survives embodiment.

**M8 acceptance:** at least one task (go/no-go here) runs end-to-end in OpenSim with qualitatively preserved frequency effects — i.e. 5 Hz → no reach (miss), ≥10 Hz → completed reach, matching the kinematic-reacher result in `PROJECT_MEMORY.md` §20.6 and §22.

**Out of scope for this phase (deferred):** muscle-excitation control, stop-signal and change-of-mind embodiment, 3D / MoBL-ARMS models, nrp-core `pysim` engine integration, cerebellar correction.

---

## 2. Key decisions (all user-approved)

| Decision | Choice | Rationale |
|---|---|---|
| Provisioning | **Docker** (conda-forge OpenSim image) | Neutralizes the numpy floor conflict (`numpy>=1.26` host vs OpenSim's pinned numpy); no host conda-env pollution; existing 674 tests stay green. |
| Model | **Arm26** (`arm26.osim`, 2 DOF planar) | Minimal model that adds real joint dynamics; matches the existing 2D left/right reach geometry; torque control = 2 `CoordinateActuator`s. MoBL-ARMS deferred. |
| Control law | **Reference-tracking PD + gravity compensation** | Frequency effect enters upstream (selection timing / target), exactly as in Phase 6; OpenSim adds inertia/gravity/coupling on top. Clean Phase-6-vs-OpenSim comparison (same reference, different plant). |
| Target representation | **Joint-space postures** (two fixed reach postures) | Avoids inverse kinematics; the min-jerk reference is interpolated directly in joint space. Hand-marker `(x,y)` recovered by forward kinematics for metrics. |
| Re-run scope | **Go/no-go only** | Satisfies M8 with the cleanest, most legible effect. Other paradigms are a later phase. |
| OpenSim version | **Pinned 4.6** (latest cleanly conda-installable; container Python is unconstrained) | Reproducible builds; documented version per Task 10.1. |
| Model file | **Vendored** `arm26.osim` from `opensim-org/opensim-models` | Reproducible, offline-stable build; not dependent on conda package internals. |
| Host↔container transport | **File-based batch** over a bind-mounted volume | The plant is open-loop downstream of the BG pipeline; no per-step feedback exists, so no live service is needed. |

---

## 3. Architecture

The plant runs entirely downstream of the BG pipeline (per `PROJECT_MEMORY.md` §22: `KinematicReacher` consumes a `motor_command_series`; nothing flows back into the BG). So the host↔container boundary is a **file-based batch**, not a live service.

```
HOST (existing env, untouched)                 CONTAINER (OpenSim only)
 run go/no-go pipeline per trial
 → motor_command_series + onset
 → extract compact ReachSpec per trial    ──►  read config.json + jobs.json
   {trial_id, selected_channel,                build Arm26 + CoordinateActuators
    onset_ms, gate_gain, gate_state}           disable muscle forces
 write jobs.json to bind-mounted dir            PD-track min-jerk joint reference
                                                forward-integrate each job
                                                read hand-marker (x,y) over time
 read trajectories.json ◄────────────────       write trajectories.json
 → OpenSimTrajectory per trial                   (+ torque diagnostics)
 → compute_movement_metrics
 → comparison report vs kinematic reacher
```

**Boundary minimality:** the container never imports project schemas (`BGDecision`, `MotorCommand`, etc.). The host extracts only what `KinematicReacher.simulate` actually uses — committed channel, onset, `gate_gain`, `gate_state` — into a minimal `ReachSpec`. The container speaks plain JSON.

**Rejected alternatives:**
- *Per-trial `docker run`*: container + Python + OpenSim startup × hundreds of trials = prohibitive overhead.
- *Long-lived RPC service*: overkill — there is no per-step plant→BG feedback to justify a persistent service.

---

## 4. Plant & control law (container side)

### 4.1 Model setup
- Load `arm26.osim` (2 DOF: shoulder + elbow flexion; **exact coordinate names read from the loaded model at runtime, not hardcoded**).
- **Disable the 6 Millard muscles** (`appliesForce=false`) so they contribute no force.
- **Append two `CoordinateActuator`s**, one per coordinate, for pure joint torque (Task 10.3: torque-level control, defer muscle excitation).

### 4.2 Targets and reference
- Channel 0 → `q_target[0]`, channel 1 → `q_target[1]`: two distinct reach **postures** in joint space (chosen plausible flexed postures; values fixed in config, documented).
- Reference trajectory: minimum-jerk interpolation in **joint space** from rest posture `q0` to `q_target[selected]`, scaled by `gate_gain` (partial gate → short of target), starting at `onset_ms`. Uses the same minimum-jerk scalar profile as `reacher.py` (`10τ³ − 15τ⁴ + 6τ⁵`).
- `selected_channel == -1` or `gate_state == "closed"` → no reach: arm holds at `q0`, trajectory is the rest posture for the full window, `onset_time_ms = None`, `selected_channel = -1`. (Mirrors `KinematicReacher.simulate`'s closed-gate path.)

### 4.3 Controller
Per integration step:
```
τ = Kp · (q_ref − q) + Kd · (q̇_ref − q̇) + τ_gravity(q)
```
- `Kp`, `Kd` per-coordinate gains (config; tuned in Task 10.2 for stable, well-damped tracking).
- `τ_gravity(q)` from OpenSim (gravity / inverse-dynamics term) for steady-state accuracy under gravity.
- Applied via the `CoordinateActuator`s each step using OpenSim's prescribed-controller / Manager forward-integration with a fixed solver step → deterministic.

### 4.4 Output
- Hand-marker `(x, y)` (planar) via forward kinematics at each `dt`, plus `times_ms`, `onset_time_ms`, `selected_channel` → `OpenSimTrajectory` with the **same field shape as `ReacherTrajectory`**, so `compute_movement_metrics` scores it unchanged.
- Diagnostics (for plant validation only): per-coordinate joint trajectories and peak torque.

**Coordinate-frame note:** OpenSim hand endpoints live in the arm's own frame, not the reacher's `[-1,0]/[+1,0]`. The Task 10.4 comparison is therefore on **qualitative / normalized** metrics (movement-onset rate, reach-completed vs not, curvature shape), which is exactly what "qualitatively preserved" (M8) requires — not numeric endpoint equality.

---

## 5. Components & module layout

### 5.1 Host side (pure Python, **no `opensim` import**; runs in the existing env and CI)

`src/nrp_bga_sb/opensim_plant.py`:
- `ReachSpec` (Pydantic v2) — `{trial_id: str, selected_channel: int, onset_time_ms: float | None, gate_gain: float, gate_state: Literal["open","partial","closed"]}`.
- `extract_reach_spec(motor_commands: list[MotorCommand], onset_time_ms: float | None) -> ReachSpec` — mirrors `KinematicReacher.simulate`'s command-reduction logic (uses the last command; argmax → channel; closed/all-zero handling) so kinematic and OpenSim plants receive identical drive.
- `OpenSimTrajectory` (Pydantic v2) — `{times_ms, positions_xy, onset_time_ms, selected_channel}`, field-compatible with `ReacherTrajectory`.
- `OpenSimPlantConfig` (Pydantic v2) — image tag, IO dir, `q0`, `q_target`, `Kp`, `Kd`, `dt_ms`, `movement_duration_ms`, `total_duration_ms`, integration step.
- `OpenSimPlantClient` — serialize `config.json` + `jobs.json` to the bind-mounted IO dir, invoke `docker run --rm -v <io>:<io>`, parse `trajectories.json` back into `list[OpenSimTrajectory]`. **Fail-fast** on nonzero exit, missing output file, or trial-id mismatch (no silent fallback).

`experiments/opensim_gonogo_sweep.py`:
- Orchestrates: build go/no-go trials → run BG pipeline (existing `ClosedLoopPolicy`) per frequency → `extract_reach_spec` per trial → `OpenSimPlantClient` batch → `OpenSimTrajectory` per trial → `compute_movement_metrics` → comparison report vs kinematic reacher. Saves JSON results + textual report to `results/`.

### 5.2 Container side (`docker/opensim/`, imports `opensim`; never touches the host env)

- `Dockerfile` — `FROM condaforge/miniforge3`; `conda install -y -c opensim-org -c conda-forge opensim=4.6`; copy in `models/` and the two scripts. Prebuilt binary (minutes, not a source build). **Not** the nrp-core `opensim.Dockerfile` (that is a multi-hour from-source build of OpenSim 4.4).
- `models/arm26.osim` — vendored from `opensim-org/opensim-models` (provenance URL + commit/tag recorded in a `models/PROVENANCE.md`).
- `run_plant.py` — batch reach simulator: read `config.json` + `jobs.json`; build Arm26 + actuators; PD-track per job; write `trajectories.json`.
- `validate_plant.py` — independent plant validation entrypoint (§6, Task 10.2).

---

## 6. Testing strategy

Two tiers, because OpenSim/Docker are not present in host CI.

### 6.1 Host unit tests (always run, no Docker) — the bulk of the test count
- `extract_reach_spec` reduction logic: last-command selection, argmax channel, closed-gate → `selected_channel=-1`, all-zero command fail-fast.
- `OpenSimTrajectory` / `ReachSpec` / `OpenSimPlantConfig` schema construction, validation, JSON round-trip.
- `OpenSimPlantClient` job serialization and result parsing against **fixture `trajectories.json`** (no container).
- Fail-fast behavior: malformed output, missing file, trial-id mismatch, nonzero exit → explicit errors (monkeypatched docker invocation).
- `compute_movement_metrics` accepts an `OpenSimTrajectory`-shaped object (parity with `ReacherTrajectory`).

These keep the existing 674-test suite green and CI-portable (no new host dependencies).

### 6.2 Container validation tests (Task 10.2, gated behind Docker)
`validate_plant.py` drives canonical **full reaches** to each target posture and asserts:
- smooth monotone (minimum-jerk-like) endpoint profile;
- movement duration ≈ configured `movement_duration_ms`;
- bounded endpoint error (reaches the target posture within tolerance);
- non-pathological peak torques (within a configured bound);
- **bit-identical repeated trials** (determinism).

Emits a validation report (`results/opensim_plant_validation.{json,txt}`). Marked `@pytest.mark.opensim`, **deselected by default** (`addopts = -m "not opensim"` or equivalent) so host CI never attempts it. Run explicitly with `pytest -m opensim` on a Docker-capable host.

### 6.3 End-to-end smoke (Docker-gated)
A tiny sweep (≥2 trials, frequencies {5, 40} Hz) through the real container asserting the qualitative effect: 5 Hz → no reach (miss), 40 Hz → completed reach. Also `@pytest.mark.opensim`.

---

## 7. Task breakdown (committable units)

| Task | Deliverable | Gate |
|---|---|---|
| **10.1** | `docker/opensim/Dockerfile` + vendored `arm26.osim` + `models/PROVENANCE.md`; build image; verify `import opensim` and model loads in-container; document model + OpenSim version in `PROJECT_MEMORY.md`. | Image builds; model loads. |
| **10.2** | `run_plant.py` (Arm26 + `CoordinateActuator`s + PD tracking) and `validate_plant.py`; produce the validation report. | **Plant validated (§6.2) before any BG integration.** |
| **10.3** | `src/nrp_bga_sb/opensim_plant.py` (`ReachSpec`, `extract_reach_spec`, `OpenSimTrajectory`, `OpenSimPlantConfig`, `OpenSimPlantClient`) + host unit tests (§6.1). | Host tests pass; full suite green. |
| **10.4** | `experiments/opensim_gonogo_sweep.py` + comparison report vs kinematic reacher + end-to-end smoke (§6.3). | **M8: qualitative frequency effect preserved.** |

Each task is committed independently (per project git workflow). PROJECT_MEMORY.md gets a new §26 (Phase 10 module map) at phase close.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `arm26.osim` not in the conda package | Vendored explicitly (decision §2). |
| OpenSim 4.6 conda solve fails for some reason | Fallback to 4.5.x (also on `opensim-org`); version is a single pinned line in the Dockerfile. |
| PD gains unstable / arm overshoots | Tuned in Task 10.2 against the validation assertions before any integration; gravity-comp term included for steady-state accuracy. |
| Joint-posture targets produce implausible reaches | Validation (§6.2) asserts smoothness, duration, bounded endpoint error before integration; postures adjustable in config. |
| Docker image build needs network (vendor fetch, conda) | Expected for `docker build`; image built once and reused. Build is the only network step. |
| Two-stage host→container flow adds operational complexity | Encapsulated in `OpenSimPlantClient`; experiments call it like any other plant. Fail-fast on every boundary error. |

---

## 9. Conventions honored

- Pydantic v2 schemas; `X | None`; `Literal[...]` for fixed vocabularies (per §16.4).
- Fail-fast: no silent fallback at the container boundary, on bad config, or on malformed output.
- Determinism: fixed-step OpenSim integration, no stochastic muscle activity → reproducible given identical inputs (matches the project-wide determinism convention).
- Section-header and decision-point (Trigger/Why/Outcome) comments in multi-section modules.
- Host code adds **no new host runtime dependency**; OpenSim lives only in the container.
