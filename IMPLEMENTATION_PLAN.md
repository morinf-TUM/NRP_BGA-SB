# IMPLEMENTATION_PLAN — NRP_BGA-SB

Derived from `bg_action_selection_implementation_plan.md` (staged build order) and `bg_frequency_action_selection_experimental_plan.md` (experimental design). Phases follow the staged implementation; tasks are bite-sized and committable. Pass/fail criteria gate each phase.

> Convention: **Phase = scientifically meaningful unit.** Each phase decomposes into tasks; each task is small enough to test and commit. Tasks have explicit acceptance checks. The user gives the green light between phases unless instructed otherwise.

---

## Phase 0 — Schemas, trial logger, repo scaffold (Milestone M0)

**Goal:** Make it possible to generate, log, replay, and score synthetic trials with no neural model attached.

Tasks:

- **0.1** Choose language / runtime and lock dependency set (Python likely; record decision in `PROJECT_MEMORY.md` §1 when done).
- **0.2** Create source tree skeleton (`src/`, `tests/`, `experiments/`, `data/`, `notebooks/`). Add `.gitignore`, `pyproject.toml` (or equivalent), CI lint+test config.
- **0.3** Implement `task_event`, `action_evidence`, `bg_decision`, `motor_command`, `trial_log`, `metrics` schemas. Use typed dataclasses or Pydantic.
- **0.4** Implement canonical event vocabulary: `trial_start`, `fixation_on`, `go_cue`, `no_go_cue`, `target_on_left`, `target_on_right`, `stop_signal`, `evidence_change`, `movement_onset`, `decision_commit`, `movement_end`, `trial_end`.
- **0.5** Implement trial logger with deterministic seeding and full per-trial fields per `PROJECT_MEMORY.md` §8.
- **0.6** Implement replay: a logged trial reproduces its event stream exactly.
- **0.7** Implement minimal scorer that ingests a `trial_log` and emits `metrics` for the abstract scoreable fields (RT, wrong-action, false-alarm rate where applicable).

**Acceptance (M0):** synthetic trials replay exactly from logs; scorer emits the metrics table without invoking any neural module.

---

## Phase 1 — Task engines (Milestone M1)

**Goal:** All four task paradigms run end-to-end with dummy policies.

Tasks:

- **1.1** Go/no-go task engine: cue scheduler, response window, success/failure classifier.
- **1.2** Two-choice conflict engine: parameterizable salience gap / target separation.
- **1.3** Stop-signal engine: variable SSD, staircase support, validity checks following Verbruggen et al. 2019.
- **1.4** Change-of-mind engine: switch-cue timing, redirection scoring.
- **1.5** Three reference policies: oracle, random, evidence-threshold dummy. All policies must produce valid trial logs.
- **1.6** Cue generator with shared-seed support (same cue sequences across BG-frequency conditions).

**Acceptance (M1):** all four paradigms produce valid `trial_log`s and `metrics` under each of the three reference policies.

---

## Phase 2 — BG module wrapper and isolated validation (Milestone M2)

**Goal:** A BG model selects among 2–4 channels under salience manipulation, in isolation.

Tasks:

- **2.1** Select BG reference model (Gurney–Prescott–Redgrave 2001, ModelDB 83560, as baseline). Record choice + version in `PROJECT_MEMORY.md`.
- **2.2** Implement BG adapter: inputs (`action_evidence`, channel salience, conflict level, noise, optional dopamine/gain) → outputs (`bg_decision` with selected channel, margin, suppression vector, selection latency).
- **2.3** Unit tests: stable selection of high-salience channel; suppression of competitors; degraded selection under small salience gaps; no pathological oscillation unless explicitly induced.
- **2.4** Wire BG adapter into Phase 1 engines as a policy option.

**Acceptance (M2):** BG selects correctly under salience manipulation; selection latency varies monotonically or interpretably with conflict. If this fails, the entire use case fails or the BG model needs to be changed.

---

## Phase 3 — Frequency-intervention layer (Milestone M3)

**Goal:** Make the four candidate "BG effective update frequency" variables independently controllable; pick the meaningful one.

Tasks:

- **3.1** Implement logical-clock scheduler exposing input-sampling, internal-integration, output-emission, and decision-commitment update frequencies as separate knobs.
- **3.2** Implement timing perturbations: fixed latency, jitter, dropout, phase offset.
- **3.3** Run ablations sweeping each of the four frequency variables independently; identify which is meaningful and controllable.
- **3.4** Document the decision and (only then) provide a convenience "effective BG frequency" parameter that maps to the chosen primary knob.

**Acceptance (M3):** the four frequencies are independently configurable; ablation report identifies the primary variable with evidence.

---

## Phase 4 — Cortex evidence + thalamic gate + abstract motor loop (combined M3+ → M4 prep)

**Goal:** Full abstract closed loop: task condition → cortical evidence generator → BG → thalamic gate → abstract motor action.

Tasks:

- **4.1** Implement abstract cortical evidence generator producing `action_evidence` from task state.
- **4.2** Implement thalamic gate adapter: gate closed unless BG decision margin exceeds threshold; gate gain modulates motor command strength.
- **4.3** Connect all stages; verify that changing BG frequency changes decision timing and action release without breaking trial validity.

**Acceptance:** BG-frequency manipulation propagates through to abstract motor output; trial logs remain valid; intermediate states are observable for failure diagnosis.

---

## Phase 5 — Frequency-sweep experiment, abstract embodiment (Milestone M4)

**Goal:** Produce reproducible frequency-response curves with confidence intervals.

Tasks:

- **5.1** Experiment runner: BG update conditions **{10, 20, 40, 80, 160 Hz}** (powers-of-two ratios over a 10 Hz base, satisfying the FTILoop `2^n × dt_min` synchronization guidance — see `PROJECT_MEMORY.md` §15.4); conflict {low, medium, high}; ≥30 seeds per condition. The earlier-considered {5, 10, 20, 40, 80, 120 Hz} set is rejected because 120 Hz does not share a power-of-two relationship with the other values, which would force a smaller `dt_min` and unnecessary sub-stepping.
- **5.2** Outputs: frequency → selection latency, frequency → wrong-action rate, frequency → no-go false alarm rate, frequency × conflict interaction, decision-margin trajectories.
- **5.3** Statistical reporting: GLMs for error probabilities; bootstrap CIs; reproducibility check (seed re-run).

**Acceptance (M4):** reproducible curves with CIs are produced for go/no-go and two-choice on the abstract embodiment.

---

## Phase 6 — Kinematic reaching surrogate (Milestone M7, brought forward)

**Goal:** Add movement-level observables without OpenSim.

Tasks:

- **6.1** Implement 2D arm or point-mass reacher.
- **6.2** Add metrics: movement onset time, trajectory curvature, endpoint error, movement reversal time, partial movement amplitude.
- **6.3** Re-run Phase 5 sweep on the kinematic reacher; confirm choice/inhibition metrics are consistent with the abstract task version.

**Acceptance:** movement-level metrics extracted automatically; qualitative BG-frequency effects from Phase 5 survive.

---

## Phase 7 — Stop-signal experiment (Milestone M5)

**Goal:** Stop failure probability increases with SSD; SSRT-like estimates are produced.

Tasks:

- **7.1** Implement standard staircase or fixed-SSD schedule per Verbruggen et al. 2019.
- **7.2** Metrics: inhibition function, stop failure probability by SSD, SSRT-like estimate, partial movement amplitude, cancellation latency, failed-stop RT vs go RT, trigger-failure estimate.
- **7.3** Validity checks: failed-stop RTs faster than go RTs; independence assumptions documented; exclusion criteria recorded.
- **7.4** Run the full BG-frequency sweep on stop-signal (≥500 trials per frequency condition).

**Acceptance (M5):** inhibition function rises with SSD; SSRT-like estimate produced; stop-signal validity checks pass.

---

## Phase 8 — Change-of-mind experiment (Milestone M6)

**Goal:** Switch behavior depends on evidence-change timing and BG frequency.

Tasks:

- **8.1** Implement switch-cue schedule (early / medium / late / very late).
- **8.2** Metrics: change-of-mind probability, revision latency, trajectory reversal time, wrong-final-target rate, correction cost, perseveration.
- **8.3** Run BG-frequency sweep.

**Acceptance (M6):** change probability depends on evidence-change timing and strength; results interpretable per the outcome guide in `PROJECT_MEMORY.md` §11.

---

## Phase 9 — Latency / jitter / dropout / phase decomposition (Milestone M10)

**Goal:** Separate the effect of update frequency from communication delay, temporal unreliability, and inter-module phase alignment.

Tasks:

- **9.1** Perturbation sweep: fixed latency {0, 10, 25, 50, 100 ms}; jitter {0, ±5, ±10, ±25 ms}; dropout {0, 1, 5, 10 %}; phase offset {0, 25, 50, 75 %}.
- **9.2** Run on go/no-go and stop-signal at minimum.
- **9.3** Apply the interpretation table from `PROJECT_MEMORY.md` §11 to classify results.

**Acceptance (M10):** decomposition report disentangles frequency from latency, jitter, dropout, and phase.

---

## Phase 10 — OpenSim embodiment (Milestone M8)

**Goal:** Same qualitative BG-frequency effects survive embodiment.

Tasks:

- **10.1** Select OpenSim upper-limb model (Arm26 or MoBL-ARMS; document choice + version).
- **10.2** Independent plant validation: smooth reaches, plausible movement durations, bounded endpoint error, non-pathological torques/activations, stable repeated trials.
- **10.3** Start at torque-level control. Defer muscle excitation.
- **10.4** Re-run Phase 6–8 experiments on OpenSim; compare against kinematic reacher results.

**Acceptance (M8):** at least one task (go/no-go or stop-signal) runs end-to-end in OpenSim with qualitatively preserved frequency effects.

---

## Phase 11 — Cerebellar correction (Milestone M9)

**Goal:** Add trajectory correction without erasing BG-dependent selection effects.

Tasks:

- **11.1** Adopt a minimal cerebellar correction module (timing adaptation, endpoint-error reduction). Document choice.
- **11.2** Verify endpoint/correction metrics improve under perturbation.
- **11.3** Confirm BG-dependent selection effects survive cerebellar addition.

**Acceptance (M9):** cerebellar module improves movement accuracy or correction timing without erasing BG-frequency effects.

---

## Phase 12 — Minimum publishable prototype writeup

**Goal:** Package the minimum strong prototype per experimental plan §14.

Required for writeup:

1. BG-alone channel validation (Phase 2).
2. Reaching plant validation (Phase 6 + Phase 10).
3. Go/no-go frequency sweep (Phase 5).
4. Stop-signal frequency sweep with Verbruggen-compliant methodology (Phase 7).
5. Latency/jitter decomposition for at least the stop-signal task (Phase 9).
6. Comparison of the three interpretations: selector / urgency / cancellation bottleneck.

Two-choice (Phase 5 part) and change-of-mind (Phase 8) are desirable but not mandatory if resources are constrained.

---

## Milestone-to-phase map

| Milestone (source plan) | Phase here | Pass/fail criterion |
|---|---|---|
| M0 — Schemas + logger | Phase 0 | Trials replay exactly from logs |
| M1 — Task engine | Phase 1 | Paradigms run with dummy policies |
| M2 — BG wrapper | Phase 2 | BG selects under salience manipulation |
| M3 — Frequency intervention layer | Phase 3 | Sampling, output, commitment frequencies independent |
| M4 — Abstract frequency experiments | Phase 5 | Curves with CIs |
| M5 — Stop-signal task | Phase 7 | Stop failure rises with SSD |
| M6 — Change-of-mind task | Phase 8 | Change probability depends on evidence shift |
| M7 — Kinematic reacher | Phase 6 | Movement metrics extracted |
| M8 — OpenSim reacher | Phase 10 | Same task runs embodied |
| M9 — Cerebellar correction | Phase 11 | Endpoint metrics improve under perturbation |
| M10 — Decomposition study | Phase 9 | Frequency / latency / jitter / dropout separated |

---

## Risk register (carried over from experimental plan §13)

| Risk | Mitigation in this plan |
|---|---|
| BG model too abstract | Phase 2 in-isolation validation; intervention reported as effective communication frequency. |
| OpenSim plant dominates results | Phases 6 then 10; compare kinematic and OpenSim results. |
| Stop-signal implementation ad hoc | Phase 7 follows Verbruggen et al. 2019 consensus. |
| Target-selection framing too simplistic | Vigor, commitment latency, speed–accuracy metrics included from Phase 5. |
| Synchronous clock misunderstood | Documented in `PROJECT_MEMORY.md` §3 as a scaffold, not a biology claim. |
| Whole-stack debugging burden | Validation ladder enforced by phase order. |
| Scope creep | Go/no-go + stop-signal are core; two-choice and change-of-mind are extensions. |

---

## Workflow notes

- Each task ends with green tests + a single commit with a clear title and a human-readable short ChangeSet-ID trailer (per global instructions).
- Pushes are the user's responsibility unless explicitly delegated.
- Update `PROJECT_MEMORY.md` at the end of each phase to reflect newly stable architecture, file paths, and choices (model versions, dependency pins).
- After completion of a phase, report state and recommend (a) keep session or start fresh; (b) Sonnet or Opus for next phase.
