# Basal Ganglia Action-Selection Bottleneck — Refined Implementation Plan

## Self-critique

The previous plan is directionally correct, but too optimistic in five places.

First, it still assumes that published component models can be coupled quickly. That is the main danger. The BG model, thalamic gate, cortex model, cerebellum model, and OpenSim arm do not naturally share compatible variables. A realistic implementation must start with explicit adapters and acceptance tests, not direct neurobiological coupling.

Second, it underdefines the behavioral task layer. The experimental plan depends on go/no-go, two-choice conflict, stop-signal, and change-of-mind paradigms. These must be implemented as first-class task engines with event logs, not informal configurations.

Third, it adds OpenSim too early. The causal question can be tested first with a kinematic reaching surrogate. OpenSim should enter only after the task logic, BG timing manipulation, and metrics pipeline are stable.

Fourth, it treats “BG frequency” as if it were a trivial scheduler parameter. It is not. We need define effective BG update frequency precisely: either input sampling rate, internal integration step, output emission rate, or decision-commitment update rate. These are not equivalent.

Fifth, the plan does not separate scientific validation from engineering validation. The first prototype can be technically correct while scientifically uninformative; the roadmap needs separate gates for both.

---

# Iterated plan with higher confidence

## Revised principle

Build a frequency-intervention testbed, not a whole-brain demo.

The minimal system should answer:

> When basal-ganglia decision updates are slowed, delayed, jittered, or dropped, how do action selection, suppression, cancellation, and switching degrade?

Synchronous logical time is used only as the scaffold within which latency, dropout, jitter, and update frequency are explicitly parameterized. It is not a claim that the biological system is globally synchronous.

---

# Version 1 — engineering-realistic build order

## Stage 0 — define interfaces and metrics before integrating models

Deliverables:

- task_event schema
- action_evidence schema
- bg_decision schema
- motor_command schema
- trial_log schema
- metrics schema

Required trial events:

- trial_start
- fixation_on
- go_cue
- no_go_cue
- target_on_left
- target_on_right
- stop_signal
- evidence_change
- movement_onset
- decision_commit
- movement_end
- trial_end

Acceptance criterion:

Synthetic trials can be generated, logged, replayed, and scored without any neural model attached.

---

## Stage 1 — BG-only module validation

Inputs:

- 2 to 4 action channels
- channel salience
- conflict level
- noise level
- optional dopamine/gain parameter

Outputs:

- selected channel
- decision margin
- suppression of non-selected channels
- selection latency
- switching instability

Frequency interventions:

- BG input sampling frequency
- BG internal integration step
- BG output emission frequency
- BG decision-commitment update frequency

Do not collapse these into one variable initially. Run ablations to determine which one is meaningful and controllable.

Acceptance criterion:

The BG module produces stable channel selection, and selection latency/error vary monotonically or interpretablly with conflict.

If this fails, the whole use case fails or needs a different BG model.

---

## Stage 2 — task engine without embodiment

Implement abstract tasks first:

- go/no-go
- two-choice conflict
- stop-signal
- change-of-mind

Motor output is initially abstract:

selected_action → virtual movement onset → virtual endpoint

Acceptance criterion:

All task paradigms produce valid trial logs and metrics under a dummy oracle policy, random policy, and BG-driven policy.

Metrics:

- reaction time
- selection latency
- wrong-action rate
- no-go false alarm rate
- stop failure rate
- SSRT-style estimate
- change-of-mind probability
- revision latency
- decision instability

---

## Stage 3 — BG + cortical evidence + thalamic gate

Loop:

task condition
→ cortical evidence generator
→ BG model
→ thalamic gate adapter
→ abstract motor action

The thalamus should initially be a controlled gate:

- gate closed unless BG decision margin exceeds threshold
- gate gain modulates motor command strength

Acceptance criterion:

Changing BG frequency changes decision timing and action release without breaking trial validity.

---

## Stage 4 — frequency-sweep experiment

Tasks:

- go/no-go
- two-choice conflict

BG effective update conditions:

- 5 Hz
- 10 Hz
- 20 Hz
- 40 Hz
- 80 Hz
- 120 Hz

Conflict:

- low
- medium
- high

Seeds:

- at least 30 per condition

Minimum outputs:

- frequency → selection latency
- frequency → wrong-action rate
- frequency → no-go false alarm rate
- frequency × conflict interaction
- decision-margin trajectories

Acceptance criterion:

The pipeline produces reproducible curves with confidence intervals.

---

## Stage 5 — add kinematic reaching surrogate

Before OpenSim, use a simple 2D arm or point-mass reacher.

Metrics added:

- movement onset time
- trajectory curvature
- endpoint error
- movement reversal time
- partial movement amplitude

Acceptance criterion:

Choice and inhibition metrics remain consistent with the abstract task version, while adding movement-level observables.

---

## Stage 6 — stop-signal task

Critical requirement:

Use a standard staircase or SSD schedule.

Metrics:

- inhibition function
- stop failure probability by SSD
- SSRT-style estimate
- partial movement amplitude
- cancellation latency

Acceptance criterion:

Stop failure probability increases with stop-signal delay.

---

## Stage 7 — change-of-mind task

Metrics:

- change-of-mind probability
- revision latency
- trajectory reversal time
- wrong-final-target rate
- correction cost

Acceptance criterion:

Change probability depends on evidence-change timing and strength.

---

## Stage 8 — add OpenSim embodiment

Start with:

- simple 2D or planar arm
- few degrees of freedom
- torque-level control first
- muscle excitation later
- no contact-rich locomotion

Acceptance criterion:

The same qualitative BG-frequency effects survive embodiment.

---

## Stage 9 — add cerebellar correction

Expected role:

- trajectory correction
- timing adaptation
- endpoint-error reduction
- not primary action selection

Acceptance criterion:

Cerebellar module improves movement accuracy or correction timing without erasing BG-dependent selection effects.

---

# What I would remove from the earlier plan

Remove for MVP:

- full M1 spiking model
- spinal CPG
- locomotion
- visual cortex
- hippocampus
- full neuromodulatory layer
- muscle-level OpenSim control
- contact dynamics

---

# Measurable implementation milestones

| Milestone | Deliverable | Pass/fail criterion |
|------------|------------|------------|
| M0 | Schemas + trial logger | Trials replay exactly from logs |
| M1 | Task engine | Go/no-go and two-choice tasks run with dummy policies |
| M2 | BG wrapper | BG selects among channels under salience manipulation |
| M3 | Frequency intervention layer | Sampling, output, and commitment frequencies independently configurable |
| M4 | Abstract BG experiments | Frequency curves generated with confidence intervals |
| M5 | Stop-signal task | Stop failure increases with SSD |
| M6 | Change-of-mind task | Change probability depends on evidence shift |
| M7 | Kinematic reacher | Movement metrics extracted automatically |
| M8 | OpenSim reacher | Same task runs in embodied system |
| M9 | Cerebellar correction | Endpoint/correction metrics improve under perturbation |
| M10 | Decomposition study | Frequency, latency, jitter, and dropout effects separated |

---

# Revised confidence assessment

## Doability: ~90%

High confidence if the MVP is defined as:

BG model + task engine + thalamic gate + abstract/kinematic reaching

## Measurability: ~95%

Progress is measurable because every stage has pass/fail outputs.

## Consistency with experimental plan: ~90%

Matches progression:

1. module validation
2. go/no-go
3. two-choice conflict
4. stop-signal
5. change-of-mind
6. latency/jitter/dropout decomposition
7. later locomotion extension

## Final refined thesis

The implementation should proceed as a staged causal-inference platform:

validated task engine
→ BG-only frequency intervention
→ abstract action-selection behavior
→ kinematic reaching
→ embodied OpenSim reaching
→ cerebellar and cortical biological refinement

That version is doable, measurable, and faithful to the basal-ganglia action-selection bottleneck use case.
