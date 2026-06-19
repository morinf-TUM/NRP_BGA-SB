# PROJECT_MEMORY — NRP_BGA-SB

Project name: **NRP_BGA-SB** — Neurorobotics Platform / Basal Ganglia Action-Selection bottleneck testbed.

This file is the primary source of truth for project context. It is derived from `bg_action_selection_implementation_plan.md` and `bg_frequency_action_selection_experimental_plan.md`. When code is added, this file should be updated to reflect architecture, module responsibilities, conventions, and data flow.

---

## 1. Current state

- Repository is greenfield: only specification documents exist at the time of writing (2026-06-19).
- No source tree, no dependencies declared, no build system configured.
- The two `bg_`-prefixed files in the project root are the authoritative source documents that motivated this memory.

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
