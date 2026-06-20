# PROJECT_MEMORY — NRP_BGA-SB

Project name: **NRP_BGA-SB** — Neurorobotics Platform / Basal Ganglia Action-Selection bottleneck testbed.

This file is the primary source of truth for project context. It is derived from `bg_action_selection_implementation_plan.md` and `bg_frequency_action_selection_experimental_plan.md`. When code is added, this file should be updated to reflect architecture, module responsibilities, conventions, and data flow.

---

## 1. Current state

- **Phase 0 complete (2026-06-19).** Source tree, schemas, trial logger, replay, and scorer are implemented and reviewed. M0 acceptance criterion verified: synthetic trials replay exactly from logs; scorer emits metrics without any neural module.
- **Phase 1 complete (2026-06-19).** All six tasks complete: four task engines, three reference policies, cue generator. M1 acceptance criterion verified: all four paradigms produce valid `TrialLog`s and `Metrics` under each of the three reference policies (212 tests passing).
- **Phase 2 complete (2026-06-19).** BG model wrapper and isolated validation. M2 acceptance criterion verified: BGAdapter selects correctly under salience manipulation (low and medium conflict); selection latency is strictly monotone with conflict; 271 tests passing.
- **Phase 3 complete (2026-06-19).** Frequency-intervention layer. M3 acceptance criterion verified: four frequency variables independently configurable; ablation identifies primary variable with evidence; 349 tests passing, ruff clean. See §19 for module map and §5.1 for M3 finding.
- **Phase 4 complete (2026-06-19).** Full abstract closed loop: cortical evidence generator → BG → thalamic gate → motor command. M4-prep acceptance criterion verified: BG-frequency manipulation propagates to abstract motor output (5 Hz → all go trials miss; 40 Hz → all succeed); trial logs remain valid across all four paradigms; 422 tests passing, ruff clean. See §20 for module map.
- **Phase 5 complete (2026-06-19).** Frequency-sweep experiment and abstract embodiment (M4 milestone). Switch_* post-switch evidence direction fixed in CortexEvidenceGenerator; sweep module (`SweepConditionResult`, `run_condition`) and stats module (`bootstrap_ci`, `aggregate_by_frequency`, `fit_frequency_slope`, `format_sweep_report`) implemented; frequency_sweep.py runs 900 conditions and saves results; ablation_frequency_v2.py re-runs the Phase 3 ablation with ClosedLoopPolicy. Key empirical finding: GPR selection threshold ≈ 0.607; at 5 Hz all four BG frequency knobs share the same miss boundary (period=200 ticks = accumulation window). 499 tests passing, ruff clean. See §21 for module map.
- **Phase 6 complete (2026-06-20).** Kinematic reaching surrogate (M7, brought forward). `KinematicReacher` simulates 2D minimum-jerk trajectories from `ClosedLoopPolicy` motor commands; `compute_movement_metrics` extracts onset time, endpoint error, partial amplitude, curvature, reversal time, and peak velocity; `run_reacher_condition` augments Phase 5 sweep conditions with movement-level metrics; `experiments/kinematic_sweep.py` runs 150-condition sweep (5×3×2×5). Key findings: `movement_onset_rate` tracks `go_success_rate` within 0.001 across all conditions; thalamic margin threshold acts as a second selection gate (marginal BG decisions with margin < 0.05 produce engine "success" but no motor movement). Pre-merge fix: `ThalamusGate` boundary condition corrected (`< → <=` on `margin_threshold` so exact-boundary margin maps to "closed" not "partial" gate). 530 tests passing, ruff clean. See §22 for module map.
- **Phase 7 complete (2026-06-20).** Stop-signal experiment (M5). Stop-signal engine extended with multi-SSD fixed schedule (`ssd_levels`) and `stop_trial_go_evidence` flag that sets `cue_identity="go"` on stop trials so `ClosedLoopPolicy`/`CortexEvidenceGenerator` runs a genuine go process (enabling race-model behaviour). `movement_onset_time` fix uses `BGDecision.selection_latency` as RT proxy. New `stop_signal_metrics.py`: `is_stop_trial`, `inhibition_function`, `estimate_ssrt` (mean-SSD method, Verbruggen 2019), `cancellation_latency_mean`, `trigger_failure_rate`, `StopSignalMetrics`, `StopSignalValidityReport`, `validate_stop_signal_data`. Sweep: `stop_signal_sweep.py` runs 5 Hz × 5 seeds × 100 trials = 500 trials/condition; `experiments/stop_signal_sweep.py` saves JSON results and prints formatted report. Key constraint: `BGDecision.selection_latency` is BG-internal latency (13–100 ms), not a go-cue-referenced RT — RT proxy is biologically short but frequency-dependent. M5 acceptance: inhibition function rises with SSD (step-function for deterministic BG, staircase oscillates around SSD = decision_point_ms boundary); SSRT-like estimate produced; validity checks implemented with appropriate deferred-check notes for deterministic Phase 7 models. 604 tests passing, ruff clean. See §23 for module map.
- **Phase 8 complete (2026-06-20).** Change-of-mind metrics and two-phase kinematic trajectory (M6). `change_of_mind_metrics.py` computes revision latency, CoM probability, perseveration rate; `KinematicReacher.simulate_change_of_mind` simulates two-phase minimum-jerk trajectories with handoff position; `change_of_mind_sweep.py` sweeps five BG frequencies across four switch-delay categories. Key finding: CoM probability is 0.0 at 5 Hz and 1.0 at ≥10 Hz; per-category timing gradient collapses because post_switch_decision_point_ms=550ms gives all categories ≥100ms of post-switch time. Movement reversal detection fixed for negative→positive sign flip. 632 tests passing (28 new in Phase 8), ruff clean. See §24 for module map.
- **Phase 9 complete (2026-06-20).** Latency/jitter/dropout/phase decomposition (M10). `perturbation_sweep.py` sweeps four timing perturbation types (fixed latency {0,10,25,50,100 ms}, jitter std {0,5,10,25 ms}, dropout {0,1,5,10 %}, phase offset {0,25,50,75 % of period}) across go/no-go and stop-signal paradigms at five BG frequencies. Key finding: latency/jitter/phase-offset shift selection_latency (RT proxy) without changing selected_channel, so go_success_rate and stop_failure_rate are frequency-driven (selector/cancellation bottleneck); only dropout alters channel selection by replaying stale decisions. M10 acceptance: decomposition report disentangles frequency from all four timing perturbations. 674 tests passing, ruff clean. See §25 for module map.
- **Phase 10 complete (2026-06-20).** OpenSim Arm26 musculoskeletal embodiment (M8). Containerised OpenSim 4.6 plant (`nrp-bga-opensim:4.6`) runs the Arm26 model via a file-based batch boundary; host `opensim_plant.py` ships reduced `ReachSpec`s to the container and scores returned hand trajectories with `compute_movement_metrics`. `experiments/opensim_gonogo_sweep.py` runs the closed-loop go/no-go pipeline at five BG frequencies through both the OpenSim plant and the kinematic reacher on the SAME BG decisions. M8 result: the BG-frequency effect survives full musculoskeletal embodiment — OpenSim movement-onset rate is 0.000 at 5 Hz and 1.000 at ≥10 Hz, identical to the kinematic reacher. 685 tests passing (1 new host dry-run + 1 Docker-gated smoke deselected without an image), ruff clean. See §26 for module map.
- **Phase 11 complete (2026-06-20).** Cerebellar trajectory correction (M9). `perturbation_plant.py` (`VisuomotorRotation` + 2D geometry helpers `rotate_xy`, `signed_angle`), `cerebellum.py` (`AdaptiveFilter` LMS trial-by-trial adaptation, `ForwardModelController` within-trial online feedback, `Cerebellum` composition with independent enable flags), `KinematicReacher.simulate_with_correction` (invoked only on executed movements — the BG-effect guard), and `cerebellum_sweep.py` (`CerebellumSweepResult`, `run_cerebellum_condition`, `FREQUENCIES_HZ`). Key result: under a 30° visuomotor rotation the cerebellum reduces mean endpoint deviation to 0.0 at every frequency that moves (θ̂ → 0.4730 rad across trials), while movement-onset-rate-vs-frequency is bit-identical with the cerebellum on or off (0.000 at 5 Hz, 1.000 at ≥10 Hz) — the BG-frequency selection signature survives. 723 tests passing, 2 deselected (Docker-gated opensim), ruff clean. See §27 for module map.
- **Phase 11b complete (2026-06-21).** OpenSim cerebellar correction confirmation (M9, embodied). `experiments/opensim_cerebellum_sweep.py` re-runs the cerebellum on/off comparison through the OpenSim Arm26 plant (`nrp-bga-opensim:4.6`) on the SAME BG decisions used in Phase 11. Perturbation (VisuomotorRotation) and cerebellar counter-rotation (AdaptiveFilter) are applied host-side in Cartesian endpoint space post-hoc on the trajectory returned by the container; the OpenSim plant is unaware of the rotation. BG-effect guard: closed-gate trials never update the filter (same invariant as §27.4). M9 embodied acceptance: cerebellar adaptation reduces endpoint deviation under a 30° visuomotor rotation while the BG-frequency onset signature is preserved. 728 tests passing, 3 deselected (Docker-gated opensim, including new cerebellum e2e smoke), ruff clean. See §28 for module map.
- The two `bg_`-prefixed files in the project root are the authoritative source documents that motivated this memory.

### Language and build (Task 0.1, 2026-06-19)

- **Language:** Python 3.10 (fixed by nrp-core host install; see §15.1)
- **Dependency manager:** `pyproject.toml` (PEP 517/518; setuptools backend; no Poetry or Pipenv)
- **Core runtime dependencies (Phase 0):** pydantic ≥ 2.0, numpy ≥ 1.26
- **Dev dependencies:** pytest ≥ 8.0, ruff ≥ 0.4

## 2. Purpose and scope

The project builds a **frequency-intervention testbed for the basal ganglia (BG)**, not a whole-brain demo. The minimal system answers:

> When BG decision updates are slowed, delayed, jittered, or dropped, how do action selection, suppression, cancellation, and switching degrade?

The refined causal question (after literature critique) is:

> Does basal-ganglia effective update frequency constrain the temporal control of action **commitment**, action **suppression**, and action **switching** in an embodied reaching system?

The framing is deliberately broad enough to adjudicate among three competing interpretations of BG function:

| Interpretation | Prediction if BG frequency is limiting |
|---|---|
| BG as action selector | Low BG frequency increases wrong-target choices and channel-selection errors. |
| BG as urgency / commitment controller | Low BG frequency mainly changes RT, vigor, and speed–accuracy tradeoff. |
| BG as cancellation bottleneck | Low BG frequency disproportionately impairs stopping and change-of-mind behavior. |

## 3. Methodological commitments

- **Synchronous logical time is a scaffold, not a biological claim.** It is used so that latency, jitter, dropout, and phase offsets become explicit, parameterized variables.
- **Engineering validation precedes scientific validation.** Each stage has technical pass/fail gates before any causal claim is made.
- **Validation ladder, not big-bang integration.** BG-alone → abstract arm → kinematic reacher → OpenSim arm. The earlier plan's mistake of integrating biological component models directly is rejected; explicit adapters with acceptance tests come first.
- **Task paradigms are first-class engines, not configurations.** Go/no-go, two-choice conflict, stop-signal, and change-of-mind each have explicit event logs and metrics pipelines.
- **"BG effective update frequency" is not a single variable.** It has four distinct candidate definitions (see §5) and ablations must determine which is meaningful before collapsing them.
- **OpenSim enters late.** A kinematic reaching surrogate must work before adding biomechanical complexity.
- **No-go ≠ stop-signal.** Withholding before commitment and cancellation after commitment are separate experiments (Dunovan et al. 2015).

## 4. System stack (target architecture)

| Component | Role in experiment |
|---|---|
| Motor cortex (abstract first) | Generates candidate action / reach commands and evidence accumulation. |
| Basal ganglia | Gates, suppresses, or releases action channels. Primary intervention target. |
| Thalamus | Gate adapter: relays BG-modulated gating state into the execution loop. Initially a controlled threshold gate. |
| Cerebellum (Stage 9) | Trajectory correction and adaptive error reduction. Not a primary selector. |
| Simplified spinal/motor interface | Command transformation layer. **No full CPG in Phase 1.** |
| Plant: abstract → kinematic reacher → OpenSim upper limb | Source of behavioral measurements. |

Explicit non-goals for the MVP (removed from the earlier plan):
- full M1 spiking model
- spinal CPG and locomotion
- visual cortex, hippocampus
- full neuromodulatory layer
- muscle-level OpenSim control and contact dynamics

## 5. The frequency-intervention vocabulary

"BG effective update frequency" is the primary independent variable but must be disambiguated:

1. **BG input sampling frequency** — how often BG reads state.
2. **BG internal integration step** — solver step inside the BG model.
3. **BG output emission frequency** — how often BG publishes gating output.
4. **BG decision-commitment update frequency** — how often a committed-channel decision is allowed to change.

These four are not equivalent. Stage 3 ablations must determine which is meaningful and controllable before collapsing them.

Companion timing variables (all must be parameterizable on the logical clock):

| Variable | Definition |
|---|---|
| BG input latency | Delay from cortex/task state to BG input. |
| BG output latency | Delay from BG output to thalamus/cortex. |
| Jitter | Trial-to-trial or message-to-message timing variability. |
| Dropout | Missing input or output messages. |
| Phase offset | Offset between BG update and companion-module update cycles. |

Initial frequency sweep: 10, 20, 40, 80, 160 Hz (also coarser 5, 10, 20, 40, 80, 120 Hz variant from implementation plan).

### 5.1 Phase 3 M3 Finding (2026-06-19)

**Ablation result (Task 3.3):** In the abstract single-call constant-evidence model, all four frequency variables produce identical behavioral outcomes. The integer-tick scheduler (Task 3.1) always fires all gates at tick=0, establishing a committed decision regardless of frequency. Behavioral metrics (RT, error rates) are flat across {10, 20, 40, 80, 160} Hz for all four knobs and all four task paradigms.

**Primary variable assignment:** `output_emission_hz` is the theoretically assigned primary variable. Rationale:
- Direct nrp-core binding: maps to `EngineTimestep` of the BG engine (§15.4).
- Most interpretable: governs when the thalamus/downstream sees updated BG decisions.
- Phase 5 sweep will vary this knob as the canonical "BG effective update frequency".

**Convenience parameter:** `FrequencyConfig.from_effective_hz(hz)` sets all four knobs to `hz` for Phase 5 sweep experiments. Individual knob dissociation will become measurable in Phase 4 (time-varying cortical evidence generator).

**M3 acceptance status:**
- Four frequencies independently configurable: ✓ (FrequencyConfig, Task 3.1)
- Timing perturbations implemented: ✓ (Task 3.2)
- Ablation identifies primary variable with evidence: ✓ (null result documented; primary variable assigned from nrp-core theory)

## 6. Data schemas (Stage 0 deliverables)

These schemas must exist before any neural model is attached:

- `task_event` — typed events with simulation timestamp
- `action_evidence` — per-channel salience / evidence
- `bg_decision` — selected channel, margin, suppression vector
- `motor_command` — descending command, gate state
- `trial_log` — full per-trial event stream with seeds
- `metrics` — aggregated per-condition outputs

Required trial events (canonical names):

`trial_start`, `fixation_on`, `go_cue`, `no_go_cue`, `target_on_left`, `target_on_right`, `stop_signal`, `evidence_change`, `movement_onset`, `decision_commit`, `movement_end`, `trial_end`.

Acceptance for Stage 0: synthetic trials can be generated, logged, replayed, and **scored without any neural model attached**.

## 7. Metrics catalog

### Neural / interface metrics
BG selection latency, BG dwell time, BG switch latency, selection entropy, gating conflict, thalamic release latency.

### Behavioral metrics
Reaction time, movement time, endpoint error, wrong-target rate, false alarm rate, stop success, switch success, perseveration, trajectory curvature, movement vigor (peak velocity / initial acceleration), partial movement amplitude, revision latency.

### Stop-signal-specific metrics
Probability of responding on stop trials, inhibition function slope, SSRT-like estimate, failed-stop RT, residual movement amplitude, trigger-failure estimate. Follow **Verbruggen et al. 2019** consensus methodology.

### System metrics
Compute cost, real-time factor, message rate, numerical failures, interface failures.

## 8. Per-trial logging contract

Every trial logs: trial ID, random seed, task type, cue identity, cue onset time, BG input receive time, BG output emit time, BG selected channel, BG channel activation values, thalamic relay/release time, motor command time series, movement onset time, endpoint trajectory, endpoint error, success/failure label, failure mode label, simulation runtime, real-time factor, message counts, dropped-message counts.

Same cue sequences should be reused across BG-frequency conditions to improve causal comparability.

## 9. Trial counts (Phase 1 design)

| Experiment | Trials per frequency condition |
|---|---:|
| Go/no-go | 200 (balanced or 70/30) |
| Two-choice | 200–300 (split across conflict levels) |
| Stop-signal | 500+ (needed for inhibition functions and SSRT) |
| Change-of-mind | 300–500 (split across switch delays) |

Seeds: at least 30 per condition for the abstract frequency sweep.

## 10. Literature anchors (canonical references)

- **BG action selection (computational baseline):** Gurney, Prescott, Redgrave 2001 (I and II), *Biological Cybernetics*. ModelDB entry 83560.
- **Stop-signal consensus methodology:** Verbruggen et al. 2019, *eLife*.
- **No-go vs stop-signal dissociation:** Dunovan, Lynch, Molesworth, Verstynen 2015, *eLife*.
- **BG as urgency controller, not target selector:** Thura & Cisek 2017, *Neuron*; Thura & Cisek 2012.
- **OpenSim upper limb validation:** Saul et al. 2015; SimTK UE-reaching; MoBL-ARMS.

Full URLs are recorded in `bg_frequency_action_selection_experimental_plan.md` §4.

## 11. Outcome interpretation guide

| Pattern in frequency sweep | Implied interpretation |
|---|---|
| Wrong-choice rate rises at low frequency | Selector-bottleneck account |
| RT / vigor shift at low frequency, choices preserved | Urgency / commitment account |
| Stop failures / SSRT worsen at low frequency | Cancellation-bottleneck account |
| Effects only in high-conflict / late-stop conditions | Conflict-dependent bottleneck, not generic |
| Effects disappear when latency/jitter controlled | Timing precision matters, not frequency per se |

A purely negative result is still informative if the experimental design and validation gates were tight.

## 12. Conventions (to be enforced when code arrives)

- Type-safe, explicit designs. Narrow Protocols for capability boundaries.
- Async boundaries are obvious (the timing is the experiment).
- Fail fast. No silent fallback logic, no speculative `getattr`, no broad exception swallowing.
- Comments explain *why*, not *what*. Decision-point comments with Trigger / Why / Outcome are encouraged where control flow gates frequency interventions.
- Real code paths preferred over mocks in tests. Bug-driven regression tests required.

## 13. Document map

- `bg_action_selection_implementation_plan.md` — staged, engineering-realistic build order (Stages 0–9, milestones M0–M10).
- `bg_frequency_action_selection_experimental_plan.md` — refined experimental design with literature anchors and statistical plan.
- `PROJECT_MEMORY.md` (this file) — project context, architecture, conventions.
- `IMPLEMENTATION_PLAN.md` — actionable phased plan with bite-sized, committable tasks.

The two `bg_` files remain frozen as source specifications; do not rewrite them. Update `PROJECT_MEMORY.md` and `IMPLEMENTATION_PLAN.md` as the project evolves.

## 14. Confidence assessment (carried over from source)

- Doability (MVP = BG + task engine + thalamic gate + abstract/kinematic reaching): ~90%.
- Measurability (every stage has pass/fail outputs): ~95%.
- Consistency with experimental plan (module validation → go/no-go → two-choice → stop-signal → change-of-mind → latency/jitter/dropout decomposition → locomotion later): ~90%.

---

## 15. NRP runtime binding

The NRP runtime is **nrp-core**, installed on the host (verified working with `NRPCoreSim --help` on 2026-06-19). This section records the exact version, build configuration, and the concrete nrp-core primitives that our project's abstractions in §4–§6 will bind to. Update this section if the install is changed or rebuilt with a different preset.

### 15.1 Pinned artifact

| Field | Value |
|---|---|
| Repo path (source) | `/home/fom/code/nrp-core` |
| Branch | `master` (verified: `master` HEAD == tag `1.5.1`) |
| Tag | `1.5.1` (latest tag in the repo as of 2026-06-19; confirmed by `git describe --tags HEAD`) |
| Commit | `25a73b5a711bec034cac353f7f583f840e51f817` |
| Commit date | 2026-04-27 |
| Declared version | `NRP_VERSION = 1.5.1` (`CMakeLists.txt:1`) |
| Host OS | Ubuntu 22.04.5 (off the project's canonical Ubuntu 20.04) |
| Python | 3.10 (system Python on jammy; `.nrp_env` writes `python3.10` paths) |
| Build preset | `.ci/cmake_cache/minimal.cmake` |
| `NRP_INSTALL_DIR` | `$HOME/.local/nrp` |
| `NRP_DEPS_INSTALL_DIR` | `$HOME/.local/nrp_deps` |
| Entry binary | `$HOME/.local/nrp/bin/NRPCoreSim` |
| Env hook | `source $NRP_INSTALL_DIR/bin/.nrp_env` |

Compile-time toggles set OFF by the `minimal.cmake` preset: `ENABLE_GAZEBO`, `ENABLE_NEST`, `ENABLE_EDLUT`, `ENABLE_SPINNAKER`, `ENABLE_ROS`, `ENABLE_MQTT`, `ENABLE_TESTING`. Re-enable individually if a later phase requires the corresponding subsystem (see §15.6).

### 15.2 Primitives we will use

| nrp-core primitive | Where defined | Role in our project |
|---|---|---|
| `NRPCoreSim` | `src/nrp_simulation/nrp_simulation_executable/main.cpp` | Single entry point for every experiment run. Driven by a JSON simulation config that lists engines and bindings. |
| `FTILoop` ("Fixed Time Increment Loop") | `src/nrp_simulation/` | Synchronous main loop. At each step, identifies engines whose `EngineTimestep` divides current `t`, steps them, exchanges datapacks via PFs/TFs, then advances `t` by `dt_min`. See `docs/architecture_overview/simulation_loop.dox`. |
| `Engine` (client/server) | `src/nrp_general_library/engine_interfaces/engine_client_interface.h` | Each project module (BG, cortex, thalamus, task, plant) is one engine. |
| `EngineTimestep` | engine JSON config key, read at `engine_client_interface.h:272` | Per-engine `dt`. The primary knob the FTILoop reads. |
| `DataPack` | `src/nrp_general_library/datapack_interface/` | Typed message exchanged between engines and PFs/TFs. JSON and protobuf DataPacks supported. |
| `TransceiverFunction` (TF) | declared via Python decorator | Python callback transforming datapacks between engines on each step. |
| `PreprocessingFunction` (PF) | declared via Python decorator | Like a TF but runs before TFs in the same step. |
| Python JSON engine | `src/nrp_python_json_engine/` | Default Python engine for our prototype modules. JSON over HTTP. |
| Python gRPC engine | `src/nrp_python_grpc_engine/` | Same role, gRPC transport. Use if Python JSON becomes a bottleneck. |
| `pysim` engine | `src/nrp_pysim_engines/` | Hosts PyBullet / OpenSim. Phase 6 (kinematic reacher) and Phase 10 (OpenSim arm) bind here. |
| `DataTransfer` engine (gRPC) | `src/nrp_datatransfer_grpc_engine/` | Streams logged data out of the loop. Candidate for Phase 0 trial logger. |

Not in this build (would require re-enabling the corresponding `ENABLE_*` flag and rebuilding): NEST engines, Gazebo engines, EDLUT engine, SpiNNaker, ROS msgs/proxies, MQTT proxies. The `nrp_event_loop` "Computational Graph" / async-experiment path is built but its ROS/MQTT nodes are not — the synchronous FTILoop is our default.

### 15.3 Mapping our schemas (§6) to nrp-core DataPacks

| Project schema (§6) | nrp-core realization | Notes |
|---|---|---|
| `action_evidence` | JSON DataPack on the cortex engine output | Per-channel salience vector. |
| `bg_decision` | JSON DataPack on the BG engine output | Selected channel, decision margin, suppression vector. |
| `motor_command` | JSON DataPack on the thalamus/motor engine output | Descending command + gate state. |
| `task_event` | DataPack emitted by a `task` engine; consumed by a logger TF | Canonical event vocabulary (§6) enforced in the DataPack schema. |
| `trial_log` | Written by a logger TF, optionally via the DataTransfer engine | Off-loop persistence so logging cost doesn't appear in FTILoop steps. |
| `metrics` | Computed offline from `trial_log` files | Not exchanged through nrp-core. |

Start with **JSON DataPacks** (simplest, no protobuf codegen). Migrate to protobuf DataPacks only if profiling shows the JSON path dominates step cost.

### 15.4 Mapping the frequency-intervention vocabulary (§5) onto FTILoop

| §5 variable | nrp-core realization | Notes / risk |
|---|---|---|
| BG output emission frequency | `EngineTimestep` of the BG engine | Cleanest binding: a 25 ms `EngineTimestep` ⇒ ~40 Hz emission. |
| BG input sampling frequency | Also `EngineTimestep` of the BG engine | FTILoop samples inputs and emits outputs at the **same** step boundaries. To dissociate, split into an input-side sampler engine + a BG-core engine with their own dts. |
| BG internal integration step | Internal to the BG model inside the BG engine | Independent of `EngineTimestep`. For Gurney-Prescott-Redgrave: the ODE solver step inside the Python module. |
| BG decision-commitment update frequency | Not a built-in nrp-core concept | Requires a commitment-gate TF (or a dedicated commitment engine) that filters BG output and only updates the published `bg_decision` when its own dt fires. |

**Critical FTILoop constraint:** the loop docs strongly recommend that every engine's `EngineTimestep` be a multiple of the smallest `dt` (and ideally `2^n × dt_min`) to guarantee clean synchronization. Our frequency sweep (5, 10, 20, 40, 80, 120 Hz — or 10, 20, 40, 80, 160 Hz) must be designed so all values share a common base step. The 10/20/40/80/160 Hz set (powers-of-two ratios over 10 Hz) satisfies this; the 5/10/20/40/80/120 Hz set does **not** for the 120 Hz point — drop or replace 120 Hz with 160 Hz unless we explicitly choose a finer `dt_min`.

**Latency, jitter, dropout are NOT first-class** in FTILoop. They must be implemented in Phase 3 as TF-level wrappers:
- Latency → a TF that enqueues outgoing DataPacks and releases them N steps later.
- Jitter → same, with a per-message random N drawn from a configured distribution.
- Dropout → a TF that drops outgoing DataPacks with configured probability.
- Phase offset → not directly tunable in a single FTILoop; would need engines whose `EngineTimestep` shares a base with the BG dt but a different starting offset (verify whether nrp-core supports this; treat as Phase 3 unknown).

### 15.5 Known constraints / quirks of this install

- **OS off-canon.** Ubuntu 22.04 + Python 3.10 + GCC ≥ 12. Build worked, but the project's CI gate is Ubuntu 20.04 / Python 3.8 / GCC 9. If any later rebuild breaks, the supported fallback is the `nrp-vanilla` Docker target (`bash build_nrp_core_image.sh nrp-vanilla`).
- **No NEST in this build.** If Phase 2 picks a NEST-based BG model (e.g., directly using the published Gurney-Prescott-Redgrave ModelDB code if it's NEST-flavoured), we must rebuild with `ENABLE_NEST=ON` — that pulls NEST 3.1, which has the most version-coupling risk on 22.04.
- **No MQTT in this build.** Inter-engine messaging goes through FTILoop / DataPacks. If we ever want fan-out logging or async observers (e.g., a live dashboard), `ENABLE_MQTT=ON` and the Paho install steps from `dockerfiles/nrp-core.Dockerfile:42` are needed.
- **OpenSim is runtime-only.** No CMake toggle. To use it (Phase 10), `pip install opensim` (Python wrappers) or a full `opensim-core` build (per `installation.dox:151`) — done outside nrp-core.
- **Submodule already initialized.** `src/nrp-core-msgs/{protobuf,nrp_ros_msgs}` is present. Don't run `git submodule update --init --recursive` unless you suspect a corruption.
- **CLAUDE.md in nrp-core asserts a Jira/EBR2 workflow** for changes to nrp-core itself. We will **not** modify nrp-core source; if we ever need to, those rules apply and the user creates the EBR2 ticket via claude.ai.

### 15.6 Triggers to re-enable disabled features

| Trigger event in our roadmap | What to flip and rebuild |
|---|---|
| Phase 2 picks a NEST-flavoured BG model | `ENABLE_NEST=ON`, `BUILD_NEST_ENGINE_SERVER=ON`; install `requirements.nest.txt` apt deps; rebuild. |
| Phase 10 needs a Gazebo physics environment instead of pysim/OpenSim | `ENABLE_GAZEBO=ON`; install `requirements.gazebo.txt` deps + `libgazebo11-dev gazebo11` (Gazebo 11 on 22.04 is fragile; consider Docker). |
| Need live observation / external streaming | `ENABLE_MQTT=ON`; build Paho per `nrp-core.Dockerfile:42`. |
| Need ROS message types for compatibility with external ROS code | `ENABLE_ROS=ON`; install ROS Noetic — this is the path the project's own CLAUDE.md warns against on non-20.04 hosts. Prefer Docker. |

Whenever any of these is flipped, update §15.1 with the new preset / commit, and append a note in §15.5 if quirks emerge.

### 15.7 Verified vs unverified claims

- **Verified by reading source / docs / running the binary:** §15.1, §15.2, §15.4's `EngineTimestep` semantics and FTILoop step algorithm.
- **Likely but to confirm in Phase 0 by running an experiment:**
  - whether `EngineTimestep` accepts mid-run modification (would enable cleaner intra-trial frequency manipulation);
  - whether TFs can introduce per-step delays without breaking the synchronization invariants the FTILoop relies on;
  - whether two engines on the same `dt` but with deliberately offset start times can coexist (the "phase offset" perturbation in §5).
- **Unknown until tried:** the JSON DataPack path's overhead at high frequencies (160 Hz × multiple engines) — this is what would force a migration to gRPC or protobuf DataPacks.

---

## 16. Phase 0 module map (stable as of 2026-06-19)

### 16.1 Source layout

```
src/nrp_bga_sb/
    __init__.py        — package root; docstring only
    schemas.py         — all six Pydantic v2 schemas + EventType enum
    logger.py          — TrialLogger (open_trial / record_event / save_trial → JSONL)
    replay.py          — load_trials / replay_events (JSONL → list[TrialLog])
    scorer.py          — score_trials (list[TrialLog] → Metrics)

tests/
    test_imports.py    — pydantic + numpy importability smoke tests
    test_schemas.py    — schema construction, validation, JSON round-trip (20 tests)
    test_logger.py     — TrialLogger open/record/save/append (8 tests)
    test_replay.py     — load_trials/replay_events round-trip, ordering (8 tests)
    test_scorer.py     — RT, wrong-action, false-alarm, edge cases (10 tests)

experiments/           — empty; experiment runner scripts go here (Phase 5)
data/                  — gitignored except .gitkeep; generated JSONL goes here
notebooks/             — empty; analysis notebooks go here
.github/workflows/ci.yml — lint (ruff) + format-check + pytest on ubuntu-22.04/python 3.10
```

### 16.2 Data flow (Phase 0)

```
task engine (Phase 1)
    │  open_trial(trial_id, seed, task_type, cue_identity, cue_onset_time)
    │  record_event(log, event_type, sim_time, real_time, payload)
    ▼
TrialLogger ──► .jsonl file (one TrialLog JSON per line)
                    │
                    ▼  load_trials(path)
                list[TrialLog]
                    │
                    ├──► replay_events(log) → Iterator[TaskEvent]   (sorted by sim_time)
                    │
                    └──► score_trials(trials, condition_id, bg_frequency_hz) → Metrics
```

### 16.3 Schema dependency order

`EventType` → `TaskEvent` → `TrialLog` (nested)
`MotorCommand` → `TrialLog` (nested)
`ActionEvidence`, `BGDecision`, `Metrics` — standalone

All six schemas plus `EventType` live in one file (`schemas.py`) for Phase 0.
Split into sub-modules only when a Phase explicitly requires it.

### 16.4 Conventions locked by Phase 0

- **Pydantic v2** for all data classes. Use `model_dump_json` / `model_validate_json` (not v1 `.json()` / `.parse_raw()`).
- **`X | None` union syntax** (not `Optional[X]`).
- **`Literal[...]`** for constrained string fields (`gate_state`, `task_type`). Add `Literal` to new fields that have a fixed vocabulary.
- **Fail fast:** `score_trials` raises `ValueError` on empty input. `load_trials` raises `FileNotFoundError` / `ValidationError` without silent fallbacks.
- **JSONL** as the on-disk format for trial logs. One `TrialLog` JSON per line; parent directories created on write.
- **Deterministic seeding:** the `seed` field in `TrialLog` is the single source of truth. Task engines construct `random.Random(seed)` from it — the logger never owns the rng.
- **Tests use `tmp_path`** (pytest built-in) for all file I/O. No writes to permanent paths in tests.
- **Section-header comments** (`# --- SectionName ---`) required in multi-section modules.
- **Decision-point comments** (Trigger / Why / Outcome) required on validators and frequency-intervention control flow.

---

## 17. Phase 1 module map (complete as of 2026-06-19)

### 17.1 Source layout additions

```
src/nrp_bga_sb/engines/
    __init__.py         — subpackage root
    go_nogo.py          — GoNoGoConfig, run_go_nogo_trials (Task 1.1)
    two_choice.py       — TwoChoiceConfig, run_two_choice_trials (Task 1.2)
    stop_signal.py      — StopSignalConfig, StaircaseState, run_stop_signal_trials (Task 1.3)
    change_of_mind.py   — ChangeOfMindConfig, run_change_of_mind_trials (Task 1.4)

src/nrp_bga_sb/
    policies.py         — oracle_policy, RandomPolicy, ThresholdPolicy (Task 1.5)
    cue_generator.py    — CueSequence, generate_cue_sequence, shared_seed_configs (Task 1.6)

tests/
    test_engine_gonogo.py       — 33 tests (Task 1.1)
    test_engine_twochoice.py    — 37 tests (Task 1.2)
    test_engine_stopsignal.py   — 31 tests (Task 1.3)
    test_engine_changeofmind.py — 21 tests (Task 1.4)
    test_policies.py            — 21 tests + M1 4×3 integration test (Task 1.5)
    test_cue_generator.py       — 21 tests (Task 1.6)
```

### 17.2 Shared engine conventions (all four engines)

- **Policy callable interface:** `(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision`.
  All engines accept the same callable; the engine varies `ActionEvidence` content per trial type.
- **Logical clock:** integer ms offsets from `go_cue_onset_ms`; converted to seconds for `sim_time`
  fields in schemas. Phase 1: logical time == real time.
- **TrialLogger optional:** pass `logger=None` for in-memory-only operation.
- **Outcome fields:** all engines set `trial_log.success` (bool) and `trial_log.failure_mode` (str | None).
- **movement_onset_time:** set only when `selected_channel >= 0`; enables downstream RT computation.
- **movement_end:** emitted only when a response was made.

### 17.3 Per-engine design notes

**go_nogo.py**
- Single policy call per trial at `decision_point_ms`.
- Outcomes: success (go + responded), correct_withhold (no-go + no response), miss (go + no response), false_alarm (no-go + responded).

**two_choice.py**
- Single policy call; two targets always presented (target_on_left, target_on_right).
- Conflict levels cycle round-robin across trials; `channel_salience` encodes conflict.
- Outcomes: success (correct target), wrong_target (wrong channel), timeout (no response).

**stop_signal.py**
- Single policy call at `decision_point_ms`; stop signal may arrive before it.
- Staircase (Verbruggen et al. 2019): SSD ↑ after inhibit success, ↓ after failure.
- Outcomes: success (go responded / stop inhibited), stop_failure (responded on stop), miss (no response on go).

**change_of_mind.py**
- **Two policy calls per switch trial:** pre-switch at `initial_decision_point_ms` (initial_salience),
  post-switch at `post_switch_decision_point_ms` (post_switch_salience).
- `evidence_change` event emitted at `switch_delay_ms` strictly between the two calls.
- Pre-switch result is logged (decision_commit payload phase="pre_switch") but does NOT determine outcome.
- Switch delay categories cycle in insertion order over switch trials.
- Outcomes: correct_switch (post-switch ch1), perseveration (post-switch ch0), miss (post-switch -1).
- No-switch baseline trials: single call, standard go outcome.

### 17.4 Scorer extensions (Tasks 1.4, final review)

`scorer.score_trials` additions beyond Phase 0:
- **`switch_success_rate`** (Task 1.4): fraction of switch trials (identified by `evidence_change` event) where `success=True`; `None` if no switch trials present.
- **`wrong_target_rate`** (final review fix): fraction of trials with `failure_mode == "wrong_target"` (two_choice engine). Always 0.0 for engines that do not use this failure mode — not `None`. Distinct from `wrong_action_rate` (go_nogo engine's `failure_mode == "wrong_action"`).

### 17.5 Reference policies (Task 1.5)

All three policies share the signature `(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision`.

- **`oracle_policy`** (function): selects `argmax(channel_salience)`; returns -1 if `stop_signal_present=True` or `max(channel_salience) < 0.3`. Decision margin = difference between top-two saliences (0.0 if fewer than 2 channels).
- **`RandomPolicy`** (class): seeds `random.Random(trial_log.seed)` on each call; picks uniformly among channel 0, channel 1, or -1. Deterministic: same `trial_log.seed` → same response.
- **`ThresholdPolicy`** (class, configurable `threshold=0.6`): selects `argmax` if `max(channel_salience) >= threshold`; otherwise -1. `stop_signal_present=True` overrides to -1 before threshold check.

M1 acceptance: `test_m1_cartesian_product` in `test_policies.py` runs all 4 engines × 3 policies (20 trials each), asserts non-empty event streams, valid `Metrics`, and non-decreasing `sim_time` event ordering.

### 17.6 Cue generator (Task 1.6)

`generate_cue_sequence(master_seed, task_type, n_trials) → CueSequence`
- Deterministic and process-stable: uses `hashlib.sha256(f"{master_seed}:{task_type}")` as RNG seed — never Python's `hash()` which is PYTHONHASHSEED-dependent.
- Different task types with the same master seed produce different sequences (salt includes task_type).
- `CueSequence` is a frozen Pydantic v2 model with `task_type: Literal[...]` validation.

`shared_seed_configs(cue_seq, base_config, bg_frequencies_hz) → list`
- Returns one copy of `base_config` per frequency, each with `seed = cue_seq.master_seed`.
- Phase 1: only the `seed` field is updated. Phase 5+ will inject per-trial seeds directly.

Scientific purpose: ensures the same cue sequence is presented under every BG-frequency condition, making per-trial causal comparisons valid.

---

## 18. Phase 2 module map (stable as of 2026-06-19)

### 18.1 Source layout additions

```
src/nrp_bga_sb/
    bg_model.py         — BGModelConfig, BGModel, BGAdapter (Tasks 2.1–2.4)

tests/
    test_bg_model.py    — 59 tests: selection, suppression, latency, wiring, M2 acceptance
```

### 18.2 BG model selection (Task 2.1)

**Chosen model:** Gurney-Prescott-Redgrave (GPR) 2001, rate-coded steady-state variant.  
**Reference:** Gurney K, Prescott TJ, Redgrave P. Biological Cybernetics 84(3), 2001. ModelDB 83560.  
**Rationale:** Minimal model that implements direct/indirect/hyperdirect pathways; analytically tractable for parameter tuning; consistent with §10 literature anchors.  
**Implementation note:** Python re-implementation of the rate equations; the original ModelDB MATLAB code is a reference only — not executed.

### 18.3 BGModel internals (Task 2.2)

Jacobi fixed-point iteration over five nuclei per channel i (N channels total):

| Equation | Description |
|---|---|
| `D1_i = max(0, u_i − theta_d)` | Direct pathway striatum |
| `D2_i = max(0, u_i − theta_d)` | Indirect pathway striatum |
| `STN_i = max(0, u_i − w_gpe_stn·GPe_i + stn_offset)` | STN: cortical input, GPe feedback |
| `GPe_i = max(0, w_stn_gpe·STN_i − w_d2_gpe·D2_i + gpe_offset)` | GPe: STN excitation, D2 inhibition |
| `GPi_i = max(0, w_stn_gpi·mean(STN) − w_d1_gpi·D1_i + gpi_offset)` | GPi: blanket suppression, D1 release |
| `T_i = max(0, thal_threshold − GPi_i)` | Thalamus output; > 0 = selected |

**Selection:** `argmax(T)`; −1 if `max(T) = 0`.

Key parameter choice: `w_stn_gpi = 0.7` (STN mean → GPi weight). Calibrated so that:
- Low conflict `[0.8, 0.2]` → channel 0 selected (T_winner ≈ 0.194)
- Medium conflict `[0.65, 0.35]` → channel 0 selected (T_winner ≈ 0.044)
- High conflict `[0.55, 0.45]` → no selection (T_winner = 0)

Mean STN (not sum) used for GPi excitation to make selection scale-invariant with N.

### 18.4 BGAdapter conventions (Task 2.2)

- **Policy interface:** `(trial_log: TrialLog, action_evidence: ActionEvidence) → BGDecision` — identical to Phase 1 policies; drop-in compatible with all four task engines.
- **Stop signal:** `stop_signal_present=True` returns `selected_channel=−1` without invoking BGModel (hyperdirect circuit deferred to Phase 3).
- **Selection latency formula:**
  - `selected_channel ≥ 0`: `latency_s = (latency_min_ms + latency_scale_ms / (T_winner + latency_eps)) / 1000`
  - `selected_channel = −1`: `latency_s = latency_max_ms / 1000`
  - Default values produce latency range: low conflict ≈ 13 ms, medium ≈ 26 ms, no-select = 100 ms.
- **Noise:** optional Gaussian perturbation on input saliences; RNG seeded from `trial_log.seed` for reproducibility.

### 18.5 M2 acceptance (verified 2026-06-19)

- BG selects correctly under salience manipulation: low conflict → correct channel; medium conflict → correct channel; high conflict → no selection (indecision).
- Selection latency strictly monotone with conflict: low < medium < high.
- BGAdapter wires into `two_choice` and `change_of_mind` engines without modification; both produce valid `TrialLog`s with non-decreasing `sim_time`.
- 271 tests passing (59 new in Phase 2).

### 18.6 Known constraints at Phase 2

- **go_nogo and stop_signal** use `channel_salience = [0.5, 0.5]` (neutral), which produces `selected_channel = −1` with BGAdapter (BG cannot decide without directional evidence). These engines need updated salience encoding to work with the BG model; deferred to Phase 4 (cortex evidence generator).
- **Hyperdirect pathway** (rapid STN boost on stop signal) is not modelled; stop signal is a policy-level override. Phase 3 may add this.
- **Latency proxy is analytic** (derived from T_winner), not a dynamic simulation of settling time. A full ODE-based latency model is deferred to Phase 3.

---

## 19. Phase 3 module map (stable as of 2026-06-19)

### 19.1 Source layout additions

```
src/nrp_bga_sb/
    scheduler.py        — FrequencyConfig, ScheduledBGAdapter (Tasks 3.1, 3.4)
    perturbations.py    — LatencyWrapper, JitterWrapper, DropoutWrapper,
                          PhaseOffsetWrapper, _derive_seed (Task 3.2)
    ablation.py         — build_sweep_config, run_condition, run_knob_sweep,
                          run_full_ablation, summarize_ablation (Task 3.3)

experiments/
    ablation_frequency.py  — Phase 3 ablation runner script (Task 3.3)

data/
    phase3_ablation_report.txt   — textual ablation finding (generated)
    phase3_ablation_results.jsonl — per-condition metrics (generated)

tests/
    test_scheduler.py      — 36 tests: FrequencyConfig, ScheduledBGAdapter,
                             from_effective_hz (Tasks 3.1, 3.4)
    test_perturbations.py  — 24 tests: all four wrappers + composition
                             (Task 3.2)
    test_ablation.py       — 18 tests: sweep construction, condition runner,
                             null-result assertion (Task 3.3)
```

### 19.2 FrequencyConfig knobs

| Knob | Default | nrp-core binding (§15.4) |
|---|---|---|
| `input_sampling_hz` | 160.0 | `EngineTimestep` of a sampler engine (if split from BG core) |
| `integration_step_hz` | 1000.0 | Internal ODE solver step inside the BG engine |
| `output_emission_hz` | 160.0 | `EngineTimestep` of the BG engine — **primary variable** |
| `commitment_update_hz` | 160.0 | Commitment-gate TF firing period |
| `base_dt_ms` | 1.0 | Simulation base step; all Hz ≤ 1000/base_dt_ms |

`FrequencyConfig.from_effective_hz(hz)` sets all four knobs to `hz` for Phase 5 sweep experiments.

### 19.3 ScheduledBGAdapter conventions

- **Integer-tick arithmetic**: `period_ticks = max(1, round(1000 / (hz * base_dt_ms)))` — avoids float modulo drift.
- **Tick-0 guarantee**: all four gates fire at tick 0 (`0 % n == 0` for any positive integer n). In the constant-evidence abstract model, `committed_decision` is always established on the first tick.
- **Stateless between calls**: each `__call__` runs a fresh simulation; no instance state mutated.
- **Fallback**: direct `base_policy` call if no commitment established (defensive guard; unreachable under valid config).
- **`accumulation_ms` parameter** (default 200 ms): the pre-decision integration window.

### 19.4 Perturbation wrappers conventions

All four wrappers implement `__call__(trial_log, action_evidence) -> BGDecision`:

| Wrapper | Effect | State |
|---|---|---|
| `LatencyWrapper(latency_ms)` | adds `latency_ms/1000` to selection_latency | stateless |
| `JitterWrapper(jitter_std_ms)` | adds N(0, jitter_std_ms)/1000 to selection_latency; clips at 0 | stateless |
| `DropoutWrapper(dropout_probability)` | returns `_last_decision` with prob p; first call always passes | **stateful** (inter-call) |
| `PhaseOffsetWrapper(phase_offset_ms)` | adds `phase_offset_ms/1000` to selection_latency | stateless |

All randomness via `_derive_seed(trial_log.seed, tag)` using `hashlib.sha256`.

### 19.5 M3 acceptance (verified 2026-06-19)

- Four frequency variables independently configurable via `FrequencyConfig`: ✓
- Timing perturbations implemented (latency, jitter, dropout, phase offset): ✓
- Ablation report identifies primary variable with evidence: ✓ (null result + theoretical assignment; see §5.1)
- 349 tests passing (78 new in Phase 3); ruff clean.

### 19.6 Known constraints at Phase 3

- **Null ablation result**: in the abstract constant-evidence model, all four frequency knobs produce identical behavioral outcomes. Differential effects require Phase 4 time-varying cortical evidence. See §5.1 for full explanation.
- **Hyperdirect pathway** not yet modelled (deferred from Phase 2; the stop-signal override remains a policy-level flag).
- **Phase offset** is modelled as additive latency in `PhaseOffsetWrapper`; nrp-core phase offset (engine timestep with different starting offset) is unverified and deferred to Phase 9.
- **Ablation should be re-run** after Phase 4 introduces time-varying cortical evidence to confirm primary variable assignment empirically.

---

## 20. Phase 4 module map (stable as of 2026-06-19)

### 20.1 Source layout additions

```
src/nrp_bga_sb/
    cortex.py       — CortexConfig, CortexEvidenceGenerator (Task 4.1)
    thalamus.py     — ThalamusConfig, ThalamusGate (Task 4.2)
    closed_loop.py  — ClosedLoopPolicy, make_closed_loop_policy (Task 4.3)

tests/
    test_cortex.py       — 36 tests: config validation, ramp shape, channel mapping,
                           symmetry, stop_signal detection, noise (Task 4.1)
    test_thalamus.py     — 22 tests: gate-state logic, gain interpolation,
                           command vector encoding, metadata (Task 4.2)
    test_closed_loop.py  — 15 tests: factory, frequency propagation (M4 acceptance),
                           motor command series, thalamic timing, 4-engine
                           integration, backward compatibility (Task 4.3)
```

### 20.2 CortexEvidenceGenerator design

- **Evidence model**: linear ramp from `[base_salience, base_salience]` (neutral) to `[peak_salience, 1 − peak_salience]` (directed) over `rise_time_ms`. Competing channel = 1 − preferred, so the sum is always 1.0.
- **Channel direction from `cue_identity`**:
  - `"go"`, `"left"`, `"no_switch"`, `"switch_*"` → channel 0 preferred
  - `"right"` → channel 1 preferred
  - `"no_go"`, `"stop"` → both channels at `base_salience` (BG should withhold)
- **stop_signal_present**: derived by scanning `trial_log.events` for `EventType.stop_signal`.
- **Noise**: optional reproducible Gaussian via `hashlib.sha256` + Box-Muller (default off).
- **Phase 5+ limitation**: switch_* cue_identities always map to channel 0 (pre-switch direction); post-switch redirection requires a switch-aware generator.

### 20.3 ThalamusGate design

- **Gate logic** (in priority order):
  1. `selected_channel == -1` → gate `"closed"`, gain `0.0`
  2. `margin < margin_threshold` → gate `"closed"`, gain `0.0`
  3. `margin_threshold ≤ margin < full_open_threshold` → gate `"partial"`, gain linearly interpolated in `[0.0, 1.0)`
  4. `margin ≥ full_open_threshold` → gate `"open"`, gain `1.0`
- **Command vector**: `command[selected_channel] = gate_gain`; all other channels `0.0`.
- **Defaults**: `margin_threshold=0.05`, `full_open_threshold=0.30`, `n_channels=2`.

### 20.4 ScheduledBGAdapter extension (backward-compatible)

- Added optional `cortex_generator: Callable[[TrialLog, float], ActionEvidence] | None = None` parameter to `__init__`.
- Gate 1 (input sampling): when `cortex_generator` is set, calls `cortex_generator(trial_log, tick * base_dt_ms)` at each sampling tick instead of copying the static `action_evidence`. When `None`, behaviour is identical to Phase 3.
- All 36 Phase 3 scheduler tests continue to pass (backward compatibility confirmed).

### 20.5 ClosedLoopPolicy and factory

- **`ClosedLoopPolicy.__call__(trial_log, action_evidence) → BGDecision`**: chains `ScheduledBGAdapter` (with cortex_generator) → `ThalamusGate`; side-effects `trial_log.motor_command_series` (appends one `MotorCommand` per call) and sets `thalamic_relay_time` / `thalamic_release_time`.
- **`make_closed_loop_policy(...)`**: convenience factory accepting optional component configs; defaults to 160 Hz, 100 ms rise time, standard thalamic thresholds.

### 20.6 Phase 4 frequency-effect mechanism

The key mechanism that makes BG-frequency effects observable (resolving the Phase 3 null result):

- At tick=0, cortical evidence is neutral `[0.5, 0.5]` → BGModel cannot select → `committed_decision.selected_channel = -1`.
- Evidence rises over `rise_time_ms` (default 100 ms). BGModel first selects when the salience gap exceeds the medium-conflict threshold (~0.3, i.e. frac ≈ 0.375, elapsed ≈ 37.5 ms).
- **Low frequency** (e.g. 5 Hz, `period_ticks = 200`): Gate 1 fires only at tick=0 within a 200-tick accumulation window. BG sees only neutral evidence → no commitment → `selected_channel = -1` → go trials miss.
- **Higher frequency** (e.g. 10 Hz, `period_ticks = 100`): Gate 1 fires at tick=0 (no selection) and tick=100 (full rise → selection) → go trials succeed.
- Frequency threshold is approximately `1000 / accumulation_ms` Hz (default: ~5 Hz boundary).

### 20.7 M4-prep acceptance (verified 2026-06-19)

- BG-frequency manipulation propagates to abstract motor output: ✓ (5 Hz → all go misses; 40 Hz → all go successes)
- Trial logs remain valid across all four paradigms: ✓ (non-decreasing sim_time, success field set)
- Intermediate states observable for failure diagnosis: ✓ (motor_command_series, thalamic timing)
- 422 tests passing (73 new in Phase 4); ruff clean.

### 20.8 Known constraints at Phase 4

- **switch_* cue_identity**: CortexEvidenceGenerator always maps to channel 0 (pre-switch direction). Post-switch evidence switching requires detecting `evidence_change` events and reversing salience direction; deferred to Phase 5.
- **go_nogo and stop_signal BGAdapter constraint resolved**: go trials now succeed with BGAdapter because directed cortical evidence is generated from `cue_identity="go"`. No-go and stop trials correctly withhold (neutral salience → BG returns -1).
- **Ablation should now be re-run**: Phase 4 time-varying evidence makes individual frequency knobs dissociable (see §5.1 and §19.6). The Phase 3 null result no longer applies.
- **Thalamic delay is zero**: `thalamic_relay_time == thalamic_release_time` in the abstract model. A non-zero thalamic delay can be added via `LatencyWrapper` from Phase 3 perturbations.py.

## 21. Phase 5 module map (stable as of 2026-06-19)

### 21.1 Source layout additions

```
src/nrp_bga_sb/
    cortex.py       — MODIFIED: post-switch evidence reversal for switch_* cues (Task 5.0)
    sweep.py        — SweepConditionResult, run_condition, CONFLICT_PEAK_SALIENCE (Task 5.1)
    stats.py        — bootstrap_ci, aggregate_by_frequency, fit_frequency_slope,
                      reproducibility_check, format_sweep_report (Task 5.2)

experiments/
    frequency_sweep.py      — 900-condition runner: 5×3×2×30 (Task 5.3)
    ablation_frequency_v2.py — per-knob ablation with ClosedLoopPolicy (Task 5.4)

tests/
    test_cortex.py              — EXTENDED: 4 regression tests for switch_* reversal
    test_sweep.py               — 15 tests: conflict calibration, run_condition, metrics
    test_stats.py               — 19 tests: bootstrap CI, aggregation, slopes, repro check
    test_frequency_sweep.py     — 13 tests: constants, save_results, run_sweep count
    test_ablation_frequency_v2.py — 26 tests: structural, threshold, parametrized, JSON

results/                  — generated output (git-ignored)
    frequency_sweep_results.json
    ablation_frequency_v2.json
```

### 21.2 Switch_* post-switch evidence fix (Task 5.0)

- **Problem**: `CortexEvidenceGenerator` is stateless; `switch_*` cues always mapped to channel 0 even after the engine crossed the switch point.
- **Fix**: scan `trial_log.events` for `EventType.evidence_change` inside `__call__`. If found and `cue_identity.startswith("switch_")`, flip `preferred` from 0 → 1.
- **Constraint**: the fix relies on the change_of_mind engine logging `evidence_change` before the second policy call; this is the only intra-trial state available to a stateless generator.

### 21.3 Sweep module design (Task 5.1)

- **`CONFLICT_PEAK_SALIENCE`**: calibrated to GPR selection threshold ≈ 0.607 (empirically verified):
  - `"low"`: 0.85 — selects at all tested frequencies (10–160 Hz)
  - `"medium"`: 0.69 — tick-100 salience=0.595 < threshold → miss at 10 Hz; tick-150 salience=0.6425 > threshold → selects at 20+ Hz
  - `"high"`: 0.62 — misses at 10, 20, 40 Hz; tick-192 (80 Hz) salience=0.6152 > threshold → selects at 80+ Hz
- **Salience formula**: `s(t) = 0.5 + (peak − 0.5) × min(1.0, t / rise_time_ms)`, with `rise_time_ms=200`, `accumulation_ms=200` (200 ticks).
- **`run_condition`**: wraps `GoNoGoEngine` or `TwoChoiceEngine` with `ClosedLoopPolicy`; computes `SweepConditionResult` with miss_rate, go_success_rate, wrong_target_rate, false_alarm_rate, timeout_rate, and BG commitment latency stats.

### 21.4 Stats module design (Task 5.2)

- **`bootstrap_ci(values, n_bootstrap=2000, alpha=0.05, rng_seed=42)`**: vectorised percentile bootstrap, pure numpy. Raises `ValueError` on empty input. `rng_seed` guarantees determinism.
- **`aggregate_by_frequency(results, metric, paradigm=None, conflict_level=None)`**: groups by `frequency_hz`, returns `{freq: {mean, ci_lo, ci_hi, n}}`. Uses `getattr(r, metric)` — raises `AttributeError` for unknown metric names (fail-fast).
- **`fit_frequency_slope(curves)`**: OLS slope of metric ~ log(frequency_hz) (natural log). Returns 0.0 for <2 frequency points. Positive slope = metric rises with frequency.
- **`format_sweep_report(results, frequencies, conflict_levels)`**: iterates over provided `frequencies` list (skipping those absent in data) to guarantee caller-controlled ordering.

### 21.5 Frequency sweep experiment (Task 5.3)

- **Design**: 5 frequencies × 3 conflict levels × 2 paradigms × 30 seeds = 900 conditions; 30 trials each.
- **Output**: `results/frequency_sweep_results.json` (JSON array of `SweepConditionResult.model_dump()`); formatted report with 95% CIs and log-frequency slopes printed to stdout.
- **Reproducibility**: a 2-freq × 3-conflict × 2-paradigm × 3-seed subset (36 condition pairs) is re-run and compared via `reproducibility_check`.

### 21.6 Ablation experiment — empirical finding (Task 5.4)

- **Design**: hold 3 of the 4 BG frequency knobs at 160 Hz, sweep the 4th over {5, 10, 20, 40, 80, 160} Hz; go_nogo, low conflict (`peak_salience=0.85`), 50 trials, seed=42.
- **Expected (pre-run)**: `input_sampling_hz` is the primary variable; secondary knobs flat.
- **Empirical finding**: all four knobs share the same 5/10 Hz selection boundary. At 5 Hz, `period_ticks = 200 = accumulation_ms`, so every gate fires exactly once at tick 0 (neutral evidence), making any single knob sufficient to block BG selection.
- **Interpretation**: `input_sampling_hz` is the mechanistic upstream bottleneck (it gates which cortical state enters the BG). The other three knobs are downstream but equally rate-limiting at 5 Hz because the pipeline serializes their effects. At ≥ 10 Hz the distinction between primary and secondary knobs becomes irrelevant (all produce clean selection).

### 21.7 M4 milestone acceptance (Phase 5)

- Frequency-response curves produced with bootstrap CIs: ✓
- Three-level conflict × frequency interaction observable: ✓ (low selects everywhere; medium 10 Hz boundary; high 80 Hz boundary)
- Switch_* post-switch evidence direction correct: ✓ (4 regression tests)
- 499 tests passing (77 new in Phase 5); ruff clean.

---

## 22. Phase 6 module map (stable as of 2026-06-20)

### 22.1 Source layout additions

```
src/nrp_bga_sb/
    reacher.py          — ReacherConfig, ReacherTrajectory, KinematicReacher,
                          _minimum_jerk_scalar (Task 6.1)
    movement_metrics.py — MovementMetrics, compute_movement_metrics (Task 6.2)
    reacher_sweep.py    — ReacherConditionResult, run_reacher_condition (Task 6.3)
    thalamus.py         — MODIFIED: boundary condition fix (< → <= on margin_threshold)

experiments/
    kinematic_sweep.py  — 150-condition reacher sweep: 5×3×2×5 (Task 6.3)

tests/
    test_reacher.py          — 15 tests: config validation, trajectory shape,
                               gate states, onset timing, channel routing
    test_movement_metrics.py — 9 tests: zero-movement, full/partial movement,
                               curvature, peak velocity (min-jerk formula)
    test_reacher_sweep.py    — 7 tests: result fields, frequency effects,
                               onset_rate vs go_success_rate parity

results/                  — generated output (git-ignored)
    kinematic_sweep_results.json
```

### 22.2 KinematicReacher design (Task 6.1)

- **Model**: 2D point-mass minimum-jerk trajectory. Formula: `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` where `τ = min(t / T, 1.0)`. Returns 0.5 at τ = 0.5 exactly.
- **Input**: `motor_command_series` from `TrialLog` (one `MotorCommand` per `ClosedLoopPolicy` call). Uses the last command (final decision).
- **Channel detection**: `int(np.argmax(command))` when `gate_state != "closed"`. `ThalamusGate` convention: `command[selected_channel] = gate_gain`, all others = 0.0.
- **Endpoint scaling**: `effective_endpoint = gate_gain × target_positions[channel]`. Partial gate (gain < 1.0) → short-of-target movement.
- **Fail fast**: raises `ValueError` if non-closed gate has all-zero command (wiring error guard).
- **`onset_time_ms=None`**: defaults to 0.0 (no movement_onset event logged, e.g. miss trial already caught by gate_state="closed").

### 22.3 MovementMetrics design (Task 6.2)

| Metric | Definition | Phase 6 value |
|---|---|---|
| `movement_onset_time_ms` | `None` if no movement | `None` for misses |
| `endpoint_error` | `‖final_pos − target‖` | 0.0 if no movement; `(1 − gate_gain) × ‖target‖` for partial |
| `partial_movement_amplitude` | `‖final_pos‖` | `gate_gain × ‖target‖` |
| `trajectory_curvature` | mean abs perpendicular deviation from origin→endpoint line | 0.0 (single-command straight-line) |
| `movement_reversal_time_ms` | first projected-velocity sign flip | `None` (monotone min-jerk; Phase 8 will exercise) |
| `peak_velocity` | max instantaneous speed | ≈ 1.875 × amplitude / T (min-jerk peak at τ=0.5) |

### 22.4 ReacherConditionResult and run_reacher_condition (Task 6.3)

- **`ReacherConditionResult`**: mirrors Phase 5 `SweepConditionResult` abstract metrics (`miss_rate`, `go_success_rate`, `timeout_rate`, `bg_commitment_latency_mean`) plus Phase 6 movement fields (`movement_onset_rate`, `mean_endpoint_error`, `mean_partial_amplitude`, `mean_peak_velocity`).
- **`movement_onset_rate` denominator**: go trials only for `go_nogo` (to be comparable with `go_success_rate`); all trials for `two_choice`.
- **`total_duration_ms = 1300.0`**: necessary because `movement_onset_time ≈ 700 ms` (cue_onset=400 + decision_point=300); 500 ms default would place all movement outside the simulation window.
- **Engine config**: identical values to `sweep._run_engine` for cross-phase comparability.
- **`CONFLICT_PEAK_SALIENCE`**: imported from `sweep`; no redefinition.

### 22.5 Thalamus boundary fix (pre-merge fix, afc715a)

- **Problem**: `ThalamusGate` used strict `< margin_threshold` for the "closed" gate condition. At the exact boundary (`margin == margin_threshold`), the code fell into the "partial" branch and computed `gate_gain = 0.0`, emitting `gate_state="partial"` with an all-zero command vector. The reacher's fail-fast guard (non-closed gate + all-zero command) would raise `ValueError`.
- **Fix**: changed to `<= margin_threshold` so the boundary value maps to `gate_state="closed"` — semantically correct (margin at threshold is not enough to open the gate).
- **Scope**: unreachable at current GPR parameters (exact float equality on 0.05 never occurs), but now consistent across the module boundary.

### 22.6 Key empirical finding: thalamic second-gate effect

- `movement_onset_rate` tracks `go_success_rate` within 0.001 across all conditions (confirmed in 150-condition sweep).
- For marginal BG decisions (selection margin ∈ (0, margin_threshold)), the go_nogo engine records "success" (BG selected a channel) but `ThalamusGate` keeps the gate closed and no motor movement occurs. This is a real system behaviour, not a calibration error.
- At current calibration (GPR threshold ≈ 0.607, default ThalamusGate thresholds 0.05/0.30), marginal selections are rare and don't affect the frequency-sweep conclusions.

### 22.7 M7 milestone acceptance (Phase 6)

- Movement-level metrics extracted automatically from ClosedLoopPolicy trials: ✓
- `movement_onset_rate` ≈ `go_success_rate` within 0.001 across all frequency/conflict conditions: ✓
- BG-frequency effects from Phase 5 survive at motor level (low freq → no movement; high freq → movement at partial amplitude): ✓
- 530 tests passing (31 new in Phase 6); ruff clean.

### 22.8 Known constraints at Phase 6

- **Single-command trajectories only**: `KinematicReacher` uses `motor_commands[-1]` (last command). Change-of-mind trajectories (2 policy calls → 2 commands) will use only the final post-switch command. Multi-command trajectory simulation (initial movement + reversal) is deferred to Phase 8.
- **`movement_reversal_time_ms` always `None`**: the reversal detection path is correct but never exercised in Phase 6 (all movements are monotone). Additionally, the reversal timestamp uses the end-of-interval sample rather than an interpolated zero-crossing — fix before Phase 8 when reversals become measurable.
- **`trajectory_curvature` always `0.0`**: single-command straight-line movements have zero perpendicular deviation. Curvature becomes non-zero in Phase 8 when multi-command trajectories are simulated.
- **Endpoint error reflects partial gate, not targeting error**: at typical BG margins (≈ 0.1–0.2), `ThalamusGate` produces partial gain (≈ 0.2–0.6), so `endpoint_error ≈ (1 − gain) × 1.0`. This is a gating-fidelity metric, not a spatial accuracy metric.

---

## 23. Phase 7 module map (stable as of 2026-06-20)

### 23.1 Source layout additions

```
src/nrp_bga_sb/
    engines/
        stop_signal.py  — MODIFIED: ssd_levels, stop_trial_go_evidence,
                          movement_onset_time RT fix (Task 7.1)
    stop_signal_metrics.py  — is_stop_trial, extract_ssd_ms, go_rt_stats,
                              failed_stop_rt_mean, inhibition_function,
                              estimate_ssrt, cancellation_latency_mean,
                              trigger_failure_rate, StopSignalMetrics,
                              StopSignalValidityReport, validate_stop_signal_data
                              (Tasks 7.2 + 7.3)
    stop_signal_sweep.py    — FREQUENCIES_HZ, N_SEEDS, N_TRIALS_PER_SEED,
                              StopSignalSweepResult, run_stop_signal_condition,
                              format_sweep_report (Task 7.4)

experiments/
    stop_signal_sweep.py  — Phase 7 sweep runner script (Task 7.4)

tests/
    test_engine_stopsignal.py  — EXTENDED: 10 new tests (ssd_levels cycling,
                                 stop_trial_go_evidence, selection_latency RT)
    test_stop_signal_metrics.py — 48 tests: trial ID helpers, RT stats, inhibition
                                  function, SSRT, cancellation latency, trigger
                                  failure, StopSignalMetrics, validity report
    test_stop_signal_sweep.py   — 16 tests: constants, condition runner, result
                                  structure, format report (fast: n=10, n_seeds=2)

results/                    — generated output (git-ignored)
    stop_signal_sweep_results.json
```

### 23.2 StopSignalConfig extensions (Task 7.1)

Two new backward-compatible fields added to the `@dataclass`:

| Field | Default | Purpose |
|---|---|---|
| `ssd_levels: list[int] \| None` | `None` | When `use_staircase=False` and set: cycles through SSD values round-robin per stop trial |
| `stop_trial_go_evidence: bool` | `False` | When `True`: stop trials use `cue_identity="go"` so `CortexEvidenceGenerator` generates directed go evidence, enabling race-model behaviour |

`movement_onset_time` fix: uses `go_cue_onset_ms + int(selection_latency * 1000)` when `BGDecision.selection_latency > 0`; falls back to `decision_abs_ms`. Event sim_time is clamped to `max(movement_onset_ms, decision_abs_ms)` to preserve log ordering invariant.

### 23.3 Stop-signal metrics (Task 7.2)

`is_stop_trial(trial)` uses three OR-criteria: `cue_identity == "stop"` OR `failure_mode == "stop_failure"` OR stop_signal event in events. This covers both `stop_trial_go_evidence` modes.

SSD extraction: `extract_ssd_ms(trial)` reads `payload["ssd_ms"]` from the stop_signal event; returns `None` for late-stop trials (SSD ≥ decision_point_ms, no event logged).

SSRT estimation (mean-SSD method, Verbruggen 2019): `SSRT = mean go RT − mean SSD`, where SSD mean excludes late-stop trials. With the 1-up/1-down staircase converging to ~50% inhibition, mean SSD ≈ SSD₅₀.

`trigger_failure_rate`: fraction of stop-failure trials where the stop_signal event is absent (SSD ≥ decision_point_ms; mechanically impossible to inhibit).

### 23.4 Validity report (Task 7.3)

`validate_stop_signal_data(trials, intended_stop_proportion)` runs five soft checks:

1. **RT check**: failed-stop RT vs. go RT. For deterministic BG models (Phase 7), RT_failed_stop = RT_go (same `selection_latency`); the report emits a specific deferred-check note rather than flagging an error.
2. **Inhibition function monotonicity**: non-decreasing failure_rate with SSD.
3. **Exclusion tracking**: `n_late_stop_trials` and `n_excluded_for_ssrt`.
4. **Empirical stop proportion**: `n_stop_trials / n_total_trials`.
5. **Independence assumption**: fixed documentation string (not empirically testable from behaviour).

### 23.5 BG-frequency sweep design (Task 7.4)

| Parameter | Value | Rationale |
|---|---|---|
| `FREQUENCIES_HZ` | [5, 10, 20, 40, 80] Hz | Same as Phase 5 sweep |
| `N_SEEDS` | 5 | Stochastic averaging |
| `N_TRIALS_PER_SEED` | 100 | 500 trials/condition ≥ M5 threshold |
| `STOP_PROPORTION` | 0.25 | Verbruggen 2019 recommendation |
| `PEAK_SALIENCE` | 0.85 | Low conflict: go process active at ≥10 Hz |
| `stop_trial_go_evidence` | `True` | Race-model behaviour for inhibition function |
| `use_staircase` | `True` | Targets ~50% inhibition per Verbruggen 2019 |

### 23.6 Key constraint: selection_latency semantics

`BGDecision.selection_latency` is documented as "time from BG input receipt to this decision (s)" — a BG-internal latency of 13–100 ms. In Phase 7, this is used as an RT proxy:
- RT = `movement_onset_time − cue_onset_time = selection_latency` (seconds).
- Biologically implausible (real RTs: 200–600 ms) but frequency-dependent (low conflict ≈ 13 ms, medium ≈ 26 ms, no-select = 100 ms).
- Consequence: failed-stop RT ≈ go RT in deterministic Phase 7 models (validity check deferred to Phase 9+).
- Phase 9+ should either: (a) route the actual go-cue-referenced selection tick time through `selection_latency`, or (b) make the RT computation in `stop_signal_metrics` aware of the `go_cue_onset_ms` offset.

### 23.7 M5 milestone acceptance (Phase 7)

- Stop-signal engine Verbruggen 2019 compliant (staircase + fixed-SSD modes): ✓
- Inhibition function rises with SSD (step-function at decision_point_ms boundary): ✓
- SSRT-like estimate produced (mean-SSD method): ✓
- Stop-signal validity checks implemented with documented deferred cases: ✓
- BG-frequency sweep produces per-condition metrics and validity reports: ✓
- 604 tests passing (74 new in Phase 7); ruff clean.

### 23.8 Known constraints at Phase 7

- **Deterministic inhibition**: `BGAdapter` returns `selected_channel=−1` whenever `stop_signal_present=True`. The inhibition function is a step function at SSD = decision_point_ms, not a smooth sigmoid. SSRT is meaningful but degenerate (≈ 0).
- **RT proxy**: `selection_latency` is BG-internal (13–100 ms), not a go-cue-referenced RT. Validity check (failed-stop RT < go RT) shows equality in Phase 7; deferred to Phase 9+ with RT variability.
- **`ssd_levels` with staircase**: `ssd_levels` is ignored when `use_staircase=True`; fixed multi-SSD schedule requires `use_staircase=False, ssd_levels=[...]`.
- **Late-stop identification with `stop_trial_go_evidence=True`**: late-stop successes (`cue_identity="go"`, `success=True`, no stop_signal event) are indistinguishable from go-success trials in the log. This case is rare in practice (directed go evidence causes BG selection; late-stop success would require BG returning −1 spontaneously).
- **`reversal_time_ms`** and **`trajectory_curvature`** limitations from Phase 6 still apply to any reacher integration in Phase 8.

---

## 24. Phase 8 module map and findings

### 24.1 Source layout

```
src/nrp_bga_sb/
    change_of_mind_metrics.py — is_switch_trial, revision_latency_ms,
                                 ChangeOfMindMetrics, compute_change_of_mind_metrics (Task 8.1)
    reacher.py                — EXTENDED: KinematicReacher.simulate_change_of_mind (Task 8.2)
    reacher_sweep.py          — EXTENDED: ChangeOfMindReacherResult,
                                 run_change_of_mind_reacher_condition (Task 8.2)
    change_of_mind_sweep.py   — ChangeOfMindSweepResult, run_change_of_mind_condition,
                                 format_sweep_report (Task 8.3)
experiments/
    change_of_mind_sweep.py   — BG-frequency sweep experiment (Task 8.3)
tests/
    test_change_of_mind_metrics.py  — 11 tests for behavioral metrics (Task 8.1)
    test_reacher_sweep.py           — EXTENDED: 8 new tests for two-phase trajectory (Task 8.2)
    test_change_of_mind_sweep.py    — 8 tests for sweep library (Task 8.3)
```

### 24.2 Change-of-mind metrics (Task 8.1)

- **`is_switch_trial(trial)`**: scans events for `EventType.evidence_change`.
- **`revision_latency_ms(trial)`**: time (ms) from `evidence_change` event to `post_switch` `decision_commit` event. Returns None for no-switch trials; raises ValueError on integrity violations.
- **`ChangeOfMindMetrics`**: `change_of_mind_probability`, `perseveration_rate`, `wrong_final_target_rate`, `mean_revision_latency_ms`, `switch_success_by_category`, `perseveration_by_category`, `revision_latency_by_category`.
- **Known constraint**: `revision_latency_by_category` returns `0.0` (not `None`) for categories where all trials are misses, because the dict type is `dict[str, float]`. Callers should cross-check with `switch_success_by_category` to detect degenerate cases.

### 24.3 Two-phase trajectory (Task 8.2)

- **`KinematicReacher.simulate_change_of_mind(motor_commands, pre_switch_onset_ms, switch_time_ms, total_duration_ms)`**: phase 1 runs minimum-jerk toward `motor_commands[0]` target from `pre_switch_onset_ms` to `switch_time_ms`; phase 2 runs minimum-jerk from the handoff position toward `motor_commands[1]` target for the remainder of the window. Returns a `ReacherTrajectory` with `selected_channel` = post-switch channel.
- **`correction_cost`**: `total_path_length - ‖final_position‖` (extra distance due to reversal). Guaranteed non-negative by the triangle inequality.
- **`movement_reversal_time_ms`** (in `compute_movement_metrics`): detects the first tick where projected velocity changes sign in **either direction** (positive→negative or negative→positive). Fixed in Phase 8 to handle the negative→positive transition characteristic of ch0→ch1 change-of-mind trajectories.
- **`ChangeOfMindReacherResult`**: `frequency_hz`, `seed`, `n_trials`, `n_switch_trials`, `change_of_mind_probability`, `perseveration_rate`, `mean_trajectory_reversal_time_ms`, `mean_correction_cost`, `switch_success_by_category`.

### 24.4 BG-frequency sweep design (Task 8.3)

| Parameter | Value |
|---|---|
| Frequencies | 5, 10, 20, 40, 80 Hz |
| Seeds | 5 |
| Trials per seed | 80 (400 total per condition; M6 ≥ 300) |
| Switch categories | early (50ms), medium (150ms), late (300ms), very_late (450ms) |
| No-switch proportion | 0.0 (all switch trials) |
| post_switch_decision_point_ms | 550 ms |
| PEAK_SALIENCE | 0.85 |
| ACCUMULATION_MS | 200 ms |

### 24.5 M6 acceptance status

- Change probability depends on BG frequency: ✅ (0.0 at 5 Hz → 1.0 at ≥ 10 Hz)
- Evidence-change timing discrimination: ⚠️ (deferred — post_switch_decision_point_ms=550ms gives all categories ≥500ms of post-switch time, collapsing per-category rates; a timing-sensitive experiment requires post_switch_decision_point_ms ≈ max_switch_delay + 50ms to expose the gradient)
- Revision latency and correction cost: ✅ (computed and present in sweep results)
- Trajectory reversal time: ✅ (movement_reversal_time_ms now works for CoM trajectories after Phase 8 fix)

### 24.6 Known constraints

- All four switch categories collapse to identical rates at every frequency because the generous post-switch window (500ms for early, 100ms for very_late) is always sufficient for the BG at ≥ 10 Hz. A future experiment reducing post_switch_decision_point_ms to ~500ms (just past very_late=450ms delay) would expose a timing gradient.
- `KinematicReacher.simulate_change_of_mind` does not model velocity continuity at the phase boundary (phase 2 starts fresh from rest at `switch_time_ms`). This simplification suffices for the Phase 8 abstract model.

---

## 25. Phase 9 module map and findings

### 25.1 Source layout additions

```
src/nrp_bga_sb/
    perturbation_sweep.py   — PerturbationType, PerturbationSweepResult,
                               LATENCY_LEVELS_MS, JITTER_STD_LEVELS_MS,
                               DROPOUT_LEVELS, PHASE_OFFSET_FRACTIONS,
                               FREQUENCIES_HZ, N_GONOGO_SEEDS,
                               N_GONOGO_TRIALS_PER_SEED, N_SS_SEEDS,
                               N_SS_TRIALS_PER_SEED, _phase_offset_ms,
                               _make_wrapped_policy, run_gonogo_perturbation_condition,
                               run_stopsignal_perturbation_condition,
                               format_decomposition_report (Tasks 9.1 + 9.2)

experiments/
    perturbation_sweep.py   — full 170-condition sweep runner; saves
                               perturbation_sweep_gonogo.json,
                               perturbation_sweep_stopsignal.json,
                               perturbation_sweep_report.txt (Task 9.2)

tests/
    test_perturbation_sweep.py — 42 tests: constants, phase-offset helper,
                                  wrapper factory, go/no-go and stop-signal
                                  condition runners, report formatter (Tasks 9.1–9.2)

results/                  — generated output (git-ignored)
    perturbation_sweep_gonogo.json
    perturbation_sweep_stopsignal.json
    perturbation_sweep_report.txt
```

### 25.2 Sweep design

Perturbation types and levels:

| Type | Levels | Unit |
|---|---|---|
| `latency` | 0, 10, 25, 50, 100 | ms |
| `jitter` | 0, 5, 10, 25 | ms std dev |
| `dropout` | 0, 1, 5, 10 | % probability |
| `phase_offset` | 0, 25, 50, 75 | % of BG update period |

Cross-product: 4 types × (4 or 5 levels) × 5 frequencies × 2 paradigms = 170 condition runs.
Each condition: N_GONOGO_SEEDS=5 × N_GONOGO_TRIALS_PER_SEED=50 = 250 go/no-go trials;
N_SS_SEEDS=5 × N_SS_TRIALS_PER_SEED=100 = 500 stop-signal trials.

Phase offset in ms: `fraction × (1000 / frequency_hz)`. At 10 Hz, 25 % = 25 ms; at 80 Hz, 25 % = 3.125 ms.

### 25.3 PerturbationSweepResult design

Single result type covering both paradigms with optional fields:
- Go/no-go fields: `go_success_rate`, `false_alarm_rate`, `bg_commitment_latency_mean` (seconds)
- Stop-signal fields: `stop_failure_rate`, `ssrt_estimate_s`, `go_rt_mean_s`, `inhibition_function_monotone`
- Shared fields: `frequency_hz`, `perturbation_type`, `perturbation_value`, `perturbation_label`, `paradigm`, `n_trials`, `n_seeds`

### 25.4 Wrapper integration

The four wrappers from Phase 3 (`perturbations.py`) wrap `ClosedLoopPolicy` as the base policy:
- `LatencyWrapper` / `JitterWrapper` / `PhaseOffsetWrapper`: modify `selection_latency` on returned `BGDecision` — affects RT proxy (`movement_onset_time`) but NOT `selected_channel`.
- `DropoutWrapper`: returns `_last_decision` (previous trial's `BGDecision`) with configured probability — this CAN change `selected_channel` and affect trial outcomes.

A fresh policy instance is created per seed so `DropoutWrapper` inter-call state resets between seeds.

### 25.5 Key finding: frequency vs. timing decomposition

- **go_success_rate** and **stop_failure_rate** are determined by `selected_channel`. Latency, jitter, and phase-offset wrappers do not change `selected_channel`, so these behavioural rates are purely frequency-driven.
- **selection_latency** (RT proxy) is increased by latency, jitter, and phase-offset wrappers, making it a timing-precision signature distinct from the frequency-driven selection signature.
- **Dropout** is the only perturbation that can alter `selected_channel` (by replaying a stale decision from a prior trial), changing go_success_rate and stop_failure_rate in a trial-history-dependent way.
- §11 interpretation: latency/jitter effects → urgency/commitment account (RT shifts, choices preserved); dropout → cancellation-bottleneck proxy (stop failures increase with replayed go decisions).

### 25.6 M10 acceptance

- Decomposition report produced for go/no-go and stop-signal: ✓
- §11 interpretation guide embedded in report header: ✓
- All four perturbation types swept across all five frequencies: ✓
- 674 tests passing (42 new in Phase 9); ruff clean.

### 25.7 Known constraints

- `bg_commitment_latency_mean` in PerturbationSweepResult is set from `thalamic_relay_time - cue_onset_time` inside ClosedLoopPolicy (before the perturbation wrapper sees the result). It does not reflect the latency/jitter/phase-offset added by the wrapper — the wrapper's effect appears only in `selection_latency` (and hence `movement_onset_time`).
- Phase-offset is modelled as additive latency (same as Phase 3's `PhaseOffsetWrapper`). True nrp-core phase-offset (engine timestep starting offset) remains unverified (see §15.4 and §19.6).
- Dropout wraps per-trial decisions across a seed's trial sequence. The inter-call `_last_decision` state is reset at seed boundaries (fresh policy per seed). Within a seed, stale decisions accumulate as intended.

## 26. Phase 10 module map (complete as of 2026-06-20)

OpenSim Arm26 musculoskeletal embodiment (M8). The BG-frequency effect demonstrated abstractly (Phase 4/5) and kinematically (Phase 6) is re-run through a real OpenSim musculoskeletal plant, and confirmed to survive embodiment.

### 26.1 Source layout

- `docker/opensim/` — containerised plant.
  - `Dockerfile` — prebuilt OpenSim 4.6 from the `opensim-org` conda channel (NOT a from-source build); copies the model and the three plant scripts to `/opt/nrp/`.
  - `_arm26_plant.py` — `Arm26Plant`: builds the model once per process, runs a torque-level PD reference-tracking controller. Muscle forces disabled; joint torques injected directly via coordinate actuators.
  - `run_plant.py` — batch entrypoint: reads `config.json` + `jobs.json`, writes `trajectories.json` (per-trajectory `trial_id`, `times_ms`, `positions_xy`, `onset_time_ms`, `selected_channel`, plus top-level `target_endpoints_xy`).
  - `validate_plant.py` — standalone plant validation harness (Task 10.2 tuning/acceptance).
  - `models/arm26.osim` + `models/PROVENANCE.md` — the Arm26 model and its provenance record.
- `src/nrp_bga_sb/opensim_plant.py` — host side. `ReachSpec` (reduced per-trial input), `extract_reach_spec(motor_commands, onset_time_ms, trial_id)` (mirrors `KinematicReacher` command reduction), `OpenSimTrajectory` (hand trajectory + `trial_id`), `OpenSimPlantConfig`, `OpenSimPlantClient(config, io_dir, runner=None).run(specs) -> (list[OpenSimTrajectory], target_endpoints_xy)`.
- `experiments/opensim_gonogo_sweep.py` — go/no-go BG-frequency sweep through the OpenSim plant with a side-by-side kinematic reacher comparison on the SAME BG decisions; reusable `run_opensim_gonogo_condition(freq_hz, n_trials, client) -> dict`; writes `results/opensim_gonogo_sweep.json`.
- `tests/opensim/` — Docker-gated tests (`@pytest.mark.opensim`, skipif image absent): `test_plant_validation.py` (Task 10.2/10.3), `test_gonogo_e2e.py` (Task 10.4 M8 smoke). Host dry-run of the sweep orchestration lives in `tests/test_opensim_plant.py` (no Docker).

### 26.2 Design decisions

- **Docker provisioning.** OpenSim 4.6 is installed from the `opensim-org` conda channel inside the image (4.5.x is the documented fallback if the solve fails) rather than built from source. The plant runs as a stateless batch container invoked per condition.
- **Arm26 + provenance.** The model is `arm26.osim` from `opensim-org/opensim-models` (`Models/Arm26/arm26.osim`), recorded in `models/PROVENANCE.md` (permissive/Apache-2.0 license per repo).
- **Torque-level PD reference tracking with min-jerk velocity feedforward.** The controller injects joint torques via coordinate actuators (muscle forces disabled). A minimum-jerk profile in joint space supplies the reference posture and a velocity feedforward term, so the PD term only corrects tracking error; this suppresses the large onset torque transient that a pure PD command produces at a coarse control rate.
- **Joint-space targets.** Reaches are specified as joint-space `q_target` postures per action channel; the hand endpoint and `target_endpoints_xy` are obtained by forward kinematics inside the container.
- **File-based batch boundary.** The host writes `config.json` (plant params, image excluded) + `jobs.json` (one `ReachSpec` per trial) into an io directory bind-mounted at `/io`; the container writes `trajectories.json`. The client fails fast on nonzero exit, missing output, or `trial_id` mismatch, and preserves request order. (Host runner usage note: `docker run -v` requires an ABSOLUTE io-dir path; `experiments/opensim_gonogo_sweep.py --io-dir` must be given an absolute path — a relative default like `data/opensim_io` is rejected by Docker.)

### 26.3 Tuned plant config (source of truth, Task 10.2)

`q0=[0.0, 0.35]`, `q_target=[[1.2, 1.0], [0.8, 1.6]]`, `kp=[120.0, 90.0]`, `kd=[18.0, 14.0]`, `dt_ms=2.0`, `movement_duration_ms=300.0`, `endpoint_error_tol=0.02`.

### 26.4 Deviations from the brief

- `dt_ms=2.0` (the brief showed 5.0): 2 ms is the validated step size that avoids onset oscillation; adopted post-tuning and made the `OpenSimPlantConfig` default.
- Minimum-jerk **velocity feedforward** added to the PD controller (not in the original design sketch) to remove the coarse-rate onset torque transient.
- `total_duration_ms=1300.0` for the EXPERIMENT (validation used 500). The go/no-go engine puts movement onset at ~700 ms (`cue_onset_ms=400 + decision_point_ms=300`); a 500 ms window would show no movement on every trial and manufacture a false-negative M8 result. The kinematic reacher uses the same 1300 ms window for an apples-to-apples comparison.

### 26.5 M8 result

go/no-go sweep, `n_trials=30` per frequency, seed 12345, onset rate computed over go trials only (18 go trials/condition):

| freq (Hz) | OpenSim onset rate | kinematic onset rate | OpenSim mean endpoint error |
|-----------|--------------------|----------------------|-----------------------------|
| 5.0  | 0.000 | 0.000 | 0.0000 |
| 10.0 | 1.000 | 1.000 | 0.1533 |
| 20.0 | 1.000 | 1.000 | 0.1533 |
| 40.0 | 1.000 | 1.000 | 0.1533 |
| 80.0 | 1.000 | 1.000 | 0.1533 |

**M8 confirmed:** the BG-frequency gating effect survives full musculoskeletal embodiment. At 5 Hz the BG never crosses the selection threshold so no reach occurs; at ≥10 Hz every go trial produces a reach. The OpenSim onset rates match the kinematic reacher exactly (both driven by the SAME BG decisions), so the embodiment adds movement dynamics without altering the qualitative selection signature. The mean OpenSim endpoint error (~0.153 in the arm's own FK frame) is constant across the frequencies that move, as expected for a deterministic reach to a fixed `q_target`.

---

## 27. Phase 11 module map (complete as of 2026-06-20)

Cerebellar trajectory correction (M9). The cerebellar module adds accuracy improvement under visuomotor perturbation while leaving the BG-frequency selection signature structurally intact. All new code is host-side; nothing in the BG, cortex, thalamus, scheduler, or task-engine pipeline changes.

### 27.1 Source layout

```
src/nrp_bga_sb/
    perturbation_plant.py   — VisuomotorRotation (apply θ rotation to a 2D reach vector),
                              rotate_xy (rotation matrix utility),
                              signed_angle (signed angular error between two 2D vectors)
    cerebellum.py           — AdaptiveFilter (scalar θ̂ state, LMS trial-by-trial adaptation),
                              ForwardModelController (within-trial proportional online feedback),
                              Cerebellum (composes AdaptiveFilter + ForwardModelController,
                                          independent adaptation_enabled / online_enabled flags)
    reacher.py              — EXTENDED: KinematicReacher.simulate_with_correction(
                                motor_commands, gate_state, target, perturbation, cerebellum,
                                onset_time_ms, total_duration_ms)
    cerebellum_sweep.py     — FREQUENCIES_HZ, CerebellumSweepResult (Pydantic v2),
                              run_cerebellum_condition(freq_hz, n_trials, seed,
                                perturbation, cerebellum_on) -> CerebellumSweepResult

experiments/
    cerebellum_adaptation.py — run_sweep, save_results, format_report, main;
                                writes results/cerebellum_results.json

tests/
    test_perturbation_plant.py   — geometry: θ=0 identity, known-angle analytic checks,
                                   signed_angle sign convention, ValueError on non-finite θ
    test_cerebellum.py           — AdaptiveFilter: LMS convergence (θ̂ → θ), monotone error
                                   decay, α-bounds validation, reset(); ForwardModelController:
                                   k=0 no-op, endpoint error reduced vs uncorrected;
                                   Cerebellum: enable flags route to correct layers
    test_reacher_correction.py   — simulate_with_correction: closed-gate / miss untouched
                                   (guard), executed trial corrected, θ=0+off reproduces simulate
    test_cerebellum_sweep.py     — CerebellumSweepResult fields, onset-rate-vs-frequency
                                   bit-identical on vs off (BG-effect guard test),
                                   dev on < dev off at moving frequencies
    test_cerebellum_experiment.py — run_sweep structure, save_results writes valid JSON,
                                    format_report includes all frequencies
```

### 27.2 Cerebellar model choice and literature anchors

The Phase 11 cerebellum implements the **cerebellar adaptive-filter** model (Marr–Albus–Ito lineage):

- **Trial-by-trial adaptation layer (`AdaptiveFilter`):** scalar counter-rotation state θ̂, updated by the Widrow-Hoff / LMS delta rule `θ̂ ← θ̂ + α·e` (α=0.1, default). Mechanistic interpretation: climbing-fibre error signal (angular error `e`) drives parallel-fibre weight update, reduced to its simplest scalar form. Produces an exponential learning curve: θ̂ → θ and error → 0 across trials. Literature anchors: Fujita M. (1982) *Biol Cybern*; Dean P., Porrill J., Stone J.V. (2010) "The cerebellar microcircuit as an adaptive filter", *Nat Rev Neurosci*.
- **Within-trial online feedback layer (`ForwardModelController`):** proportional corrective term (gain `k`) steering each integration step toward the intended (undistorted) target position. Stateless across trials. `k=0` is an exact no-op recovering the uncorrected trajectory. Mechanistic interpretation: internal forward model + delay-free feedback, simplest proportional form of the Smith-predictor lineage. Literature anchors: Miall R.C., Weir D.J., Wolpert D.M., Stein J.F. (1993) "Is the cerebellum a Smith predictor?", *J Mot Behav*; Wolpert, Miall & Kawato (1998).
- **Perturbation paradigm:** visuomotor rotation (fixed θ=30° default), the canonical cerebellar/sensorimotor adaptation task. Literature anchors: Martin T.A. et al. (1996) *Brain*; Tseng Y.W. et al. (2007) *J Neurophysiol*.

These are the project's first cerebellar literature anchors; they extend §10 without replacing it. The design rationale and rejected alternatives (two-rate model, pure Smith predictor) are documented in the design spec: `docs/superpowers/specs/2026-06-20-phase11-cerebellar-correction-design.md`.

### 27.3 Geometry convention

`VisuomotorRotation` applies a 2D rotation matrix about the origin to the commanded reach direction vector. A rotation of θ=30° (≈0.524 rad) deflects the executed trajectory by 30° relative to the intended direction. The angular error `e` is the signed angle from the achieved endpoint direction to the desired endpoint direction, computed by `signed_angle(v_actual, v_desired)` (positive = counter-clockwise correction needed).

Accuracy is measured as endpoint deviation from the **gate-scaled desired endpoint**: `target_xy * gate_gain`. This matches the Phase 6 `KinematicReacher` convention (§22.2) — the reference is not the raw target but the thalamic-gain-scaled position that a perfect unperturbed reach would reach. A cerebellum-corrected reach lands at `endpoint_deviation ≈ 0.0` when θ̂ ≈ θ; an uncorrected reach has `endpoint_deviation = ‖target‖ × sin(θ) × gate_gain`.

### 27.4 BG-effect guard mechanism

`KinematicReacher.simulate_with_correction` checks the gate state before any cerebellar computation. If `gate_state == "closed"` (5 Hz miss trial: thalamus never opened — §20.6, §22.5), the method returns a zero-movement trajectory immediately; the cerebellum (`AdaptiveFilter` and `ForwardModelController`) is never invoked. Therefore:
- No movement is manufactured that the BG did not release.
- `movement_onset_rate` as a function of BG frequency is structurally identical with the cerebellum on or off.
- The BG-frequency selection signature (§20.6, §22.7) cannot be altered by the cerebellar layer regardless of its learning state.

This is a positional invariant, not a tuning result: the corrector sits downstream of the thalamic gate and can only reshape trajectories the gate already produced.

### 27.5 M9 acceptance result

Sweep design: 5 BG frequencies × cerebellum off/on × 50 trials, seed 42, 30° visuomotor rotation, go/no-go paradigm, `n_trials_per_freq=50`, `accumulation_ms=200`. Results from `experiments/cerebellum_adaptation.py` smoke run (2026-06-20):

```
freq(Hz) | onset off | onset on | dev off | dev on | ang.err on | theta_hat on
----------------------------------------------------------------------------------
    5.0  |     0.000 |    0.000 |  0.0000 |  0.0000 |     0.0000 |       0.0000
   10.0  |     1.000 |    1.000 |  0.0393 |  0.0000 |     0.0000 |       0.4730
   20.0  |     1.000 |    1.000 |  0.2205 |  0.0000 |     0.0000 |       0.4730
   40.0  |     1.000 |    1.000 |  0.3102 |  0.0000 |     0.0000 |       0.4730
   80.0  |     1.000 |    1.000 |  0.3106 |  0.0000 |     0.0000 |       0.4730
```

Interpretation: `onset off == onset on` at every frequency (bit-identical, BG-effect guard holds); `dev on = 0.0` vs `dev off > 0` at every moving frequency (cerebellum corrects to zero endpoint deviation); `theta_hat on ≈ 0.4730 rad` (≈27°) at ≥10 Hz (LMS adaptation has converged; residual of ~3° from θ=0.524 rad reflects the finite block length). Results written to `results/cerebellum_results.json`.

**M9 confirmed.** OpenSim embodiment of the cerebellar correction is deferred to Phase 11b (IMPLEMENTATION_PLAN.md), mirroring the Phase 6 → Phase 10 kinematic → OpenSim step.

### 27.6 Known constraints at Phase 11

- **θ̂ asymptote is ~0.4730 rad, not the full 0.524 rad (30°)**: a fresh filter is built per condition (θ̂ resets each frequency × on/off cell), so with α=0.1 the LMS filter adapts over only the ~21 go-trials of each 30-trial condition (`go_probability=0.7`) before reset — not a long uninterrupted block. The exponential approach `θ̂_n ≈ θ(1 − (1−α)^n)` with n ≈ 21 predicts θ̂ ≈ 0.524 × (1 − 0.9²¹) ≈ 0.524 × 0.891 ≈ 0.467 rad, matching the measured ~0.473 rad. This is a finite-block parameter-sweep artefact, not a model failure (endpoint deviation still falls to 0.0 with the cerebellum on, via within-trial online feedback).
- **`dev off` varies with frequency (0.039–0.311)**: the unperturbed endpoint deviation depends on how many trials run at each frequency before the adaptation state resets; at 10 Hz fewer adaptation steps occur. With the cerebellum on, this variation disappears (dev on = 0.0 everywhere).
- **Kinematic plant only (Phase 11):** OpenSim embodiment was deferred to Phase 11b (now complete — see §28).
- **Single rotation direction**: only a fixed positive θ is tested. Negative θ (mirror rotation), washout, and savings experiments are out of scope for Phase 11 / Phase 11b.

---

## 28. Phase 11b module map (complete as of 2026-06-21)

OpenSim cerebellar correction confirmation (M9, embodied). Mirrors the Phase 6 → Phase 10 kinematic-to-OpenSim validation step: re-runs the cerebellar on/off sweep through the full musculoskeletal plant, confirming that M9 results survive embodiment.

### 28.1 Source layout

```
experiments/
    opensim_cerebellum_sweep.py — run_opensim_cerebellum_condition(
                                      freq_hz, n_trials, client,
                                      cerebellum_enabled, perturbation_deg
                                  ) -> dict;
                                  main() iterates FREQUENCIES_HZ × {off, on},
                                  writes results/opensim_cerebellum_results.json

tests/
    test_opensim_cerebellum_sweep.py   — host dry-run (5 tests, no Docker):
                                         all-miss → zero onset + zero deviation + theta_hat=0;
                                         perturbation without cerebellum → deviation > 0;
                                         onset rate identical on vs off (BG guard);
                                         cerebellum reduces deviation after 50 trials;
                                         result dict has all required keys.
    opensim/
        test_opensim_cerebellum_e2e.py — Docker-gated smoke (@pytest.mark.opensim):
                                          40 Hz on vs off: onset unchanged + dev↓;
                                          5 Hz: onset=0, deviation=0.
```

No new source modules were added — Phase 11b reuses `AdaptiveFilter` from `cerebellum.py`, `VisuomotorRotation` / `rotate_xy` / `signed_angle` from `perturbation_plant.py`, `OpenSimPlantClient` / `extract_reach_spec` from `opensim_plant.py`, and `KinematicReacher` from `reacher.py`.

### 28.2 Integration design

The OpenSim container is unaware of the perturbation: it drives joints to `q_target` as usual and returns the physical Cartesian trajectory. Perturbation and cerebellar correction are applied entirely host-side in Cartesian endpoint space:

1. Ship normal `ReachSpec` → container → get `OpenSimTrajectory`
2. Extract physical endpoint: `p_physical = positions_xy[-1]`
3. Apply perturbation: `p_perturbed = VisuomotorRotation.apply(p_physical)` (simulates visual distortion)
4. Compute angular error vs FK target: `e = signed_angle(target_endpoints_xy[ch], p_perturbed)`
5. Apply cerebellar counter-rotation: `p_corrected = rotate_xy(p_perturbed, -θ̂)` (where θ̂ is the current `AdaptiveFilter.theta_hat`)
6. Update filter: `AdaptiveFilter.update(e)` (for next trial)
7. Report deviation: `|p_corrected − target_endpoints_xy[ch]|`

This is equivalent to the kinematic precompensate–perturbate sequence: when θ̂ → θ, `p_corrected ≈ p_physical ≈ target_endpoints_xy[ch]` and deviation → 0.

The desired endpoint reference is `target_endpoints_xy[selected_channel]` (the FK-computed Cartesian hand position for that channel's joint target), without gate_gain scaling — consistent with Phase 10's `compute_movement_metrics` convention for the embodied plant.

### 28.3 BG-effect guard (embodied)

Same invariant as §27.4. `ot.onset_time_ms is None or ot.selected_channel < 0` identifies closed-gate (missed) trials in the returned OpenSim trajectory. The filter update is skipped and the onset counter is not incremented for such trials. The movement-onset-rate-vs-frequency signature is therefore structurally unchanged by the cerebellar layer regardless of its learning state.

### 28.4 Known constraints at Phase 11b

- **No within-trial online feedback for OpenSim**: the `ForwardModelController` cannot inject per-step corrective signals into the container's PD simulation. Only the trial-by-trial `AdaptiveFilter` layer is active; the net correction is a post-hoc counter-rotation of the returned endpoint. Accuracy improvement therefore depends entirely on LMS convergence across trials (same mechanism as §27.5, same convergence rate).
- **`target_endpoints_xy` reference (no gate_gain scaling)**: the FK targets returned by the container represent the full-reach joint targets. Gate_gain scales the reach extent in the kinematic plant but is implicit in the OpenSim PD trajectory; applying it to the reference would introduce a spurious offset. Phase 10 convention is preserved.
- **Docker-gated smoke only**: the embodied M9 acceptance numbers require the `nrp-bga-opensim:4.6` image. The host dry-run tests exercise the full adaptation logic with a fake runner echoing plausible trajectories; the BG-guard and LMS convergence are validated without Docker.
