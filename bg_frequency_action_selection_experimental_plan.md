# Experimental Plan: Basal Ganglia Frequency as an Action-Selection Bottleneck

## Status

This document is the refined experimental plan for the use case **"Basal Ganglia Frequency as an Action-Selection Bottleneck"**, assuming an MVP whole-brain simulation stack composed of motor cortex, basal ganglia, thalamus, cerebellum, simplified spinal/motor interface, and an OpenSim upper-limb plant.

The design goal is not to build a visually impressive whole-brain demo. The goal is to produce a controlled causal experiment in which basal-ganglia timing can be manipulated while the rest of the system is held constant.

The target confidence level for this plan is approximately **90%** on two dimensions:

1. **Methodological viability:** the experiment can plausibly be implemented, measured, and debugged with accessible models and data.
2. **Research value:** the causal question is nontrivial, externally legible, and not reducible to a simpler non-NRP simulation.

---

## 1. Refined causal question

The initial question was:

> How does the effective update frequency of the basal ganglia constrain action-selection performance in a closed-loop motor-control system?

After critique against the literature, the better formulation is:

> **Does basal-ganglia effective update frequency constrain the temporal control of action commitment, action suppression, and action switching in an embodied reaching system?**

This formulation is deliberately broader than simple target selection. It allows the experiment to adjudicate among competing interpretations of basal-ganglia function:

| Interpretation | Prediction if BG frequency is limiting |
|---|---|
| BG as action selector | Low BG frequency increases wrong-target choices and channel-selection errors. |
| BG as urgency / commitment controller | Low BG frequency mainly changes reaction time, movement vigor, and speed-accuracy tradeoff. |
| BG as cancellation bottleneck | Low BG frequency disproportionately impairs stopping and change-of-mind behavior. |

The plan should avoid assuming that the basal ganglia directly select reach targets. Thura and Cisek argue that, in reach decisions, the basal ganglia may instead control urgency of commitment rather than the identity of the selected target. That makes movement vigor, commitment latency, and speed-accuracy tradeoff central metrics, not peripheral ones.

Reference:

- Thura D, Cisek P. 2017. *The Basal Ganglia Do Not Select Reach Targets but Control the Urgency of Commitment*. Neuron. https://pubmed.ncbi.nlm.nih.gov/28823728/

---

## 2. Why upper-limb reaching, not locomotion, for Phase 1

The first-pass plan used stable rhythmic locomotion. That was useful conceptually but too confounded for the first experiment.

### Why “stable” mattered

Stable motor execution means the plant and low-level controller maintain a baseline behavior without collapsing when the basal ganglia are not intervening. Without baseline stability, BG-frequency effects are uninterpretable: poor performance could reflect body/controller instability rather than impaired action selection.

### Why “rhythmic” mattered

Rhythmic motor execution gives the system a natural phase variable. If the task includes gait, tapping, cyclic reaching, or other periodic behavior, one can ask whether BG outputs are more or less effective depending on their phase relative to an ongoing motor cycle.

### Why locomotion is deferred

Locomotion adds avoidable confounds:

- foot-ground contact dynamics;
- balance control;
- fall detection;
- uncertain spinal CPG circuitry;
- gait-specific biomechanics;
- high sensitivity to musculoskeletal modeling errors.

These are scientifically interesting but undesirable for the first causal test. They should become Phase 2 or Phase 3, after the basic BG-frequency effect has been characterized in a cleaner upper-limb task.

### Phase 1 choice

The Phase 1 task should use **cued upper-limb reaching, go/no-go suppression, stop-signal cancellation, and change-of-mind switching**.

This preserves the causal structure:

- competing actions;
- action initiation;
- action suppression;
- action cancellation;
- mid-course action switching;
- measurable reaction time, error rate, and movement trajectory.

It removes the locomotion-specific burden.

---

## 3. MVP system stack

The Phase 1 experimental stack should be:

| Component | Role in experiment |
|---|---|
| Motor cortex | Generates candidate action or reach commands. |
| Basal ganglia | Gates, suppresses, or releases action channels. |
| Thalamus | Relays BG-modulated gating state into cortical execution loop. |
| Cerebellum | Performs trajectory correction / adaptive error reduction. |
| Simplified spinal/motor interface | Converts descending commands to joint or muscle-level control. |
| OpenSim upper limb | Physical plant and source of behavioral measurements. |

The spinal cord should be minimized in Phase 1. A full locomotor spinal CPG is unnecessary and would add an irrelevant modeling burden. The spinal/motor interface can initially be a command transformation layer, not a detailed spinal circuit.

---

## 4. Literature anchors for validation

The plan should leverage established behavioral and computational paradigms rather than inventing ad hoc task metrics.

### 4.1 Basal-ganglia action-selection model anchor

The classic Gurney-Prescott-Redgrave model treats action selection as a primary basal-ganglia function and provides an established computational reference for channel competition.

References:

- Gurney K, Prescott TJ, Redgrave P. 2001. *A computational model of action selection in the basal ganglia. I. A new functional anatomy*. Biological Cybernetics. https://pubmed.ncbi.nlm.nih.gov/11417052/
- Gurney K, Prescott TJ, Redgrave P. 2001. *A computational model of action selection in the basal ganglia. II. Analysis and simulation of behaviour*. Biological Cybernetics. https://pubmed.ncbi.nlm.nih.gov/11417053/
- ModelDB implementation: https://modeldb.science/showmodel?model=83560

Use in this plan:

- validate BG channel competition in isolation;
- verify selection under salience gaps;
- measure selection latency and winner-take-all behavior before embedding in the whole stack.

### 4.2 Stop-signal methodology anchor

The stop-signal task has a mature methodological literature. The consensus guide by Verbruggen et al. should be treated as the reference protocol for inhibition-related experiments.

Reference:

- Verbruggen F et al. 2019. *A consensus guide to capturing the ability to inhibit actions and impulsive behaviors in the stop-signal task*. eLife. https://elifesciences.org/articles/46323
- PubMed: https://pubmed.ncbi.nlm.nih.gov/31033438/

Use in this plan:

- implement go RT distributions;
- implement stop-signal delay manipulation;
- generate inhibition functions;
- estimate SSRT-like quantities;
- distinguish failed stops from ordinary go trials;
- include exclusion / validity checks.

### 4.3 No-go versus reactive stopping anchor

No-go and stop-signal tasks should not be collapsed. No-go tests withholding before commitment. Stop-signal tests cancellation after commitment.

Reference:

- Dunovan K, Lynch B, Molesworth T, Verstynen T. 2015. *Competing basal ganglia pathways determine the difference between stopping and deciding not to go*. eLife. https://pmc.ncbi.nlm.nih.gov/articles/PMC4686424/

Use in this plan:

- treat go/no-go and stop-signal as separate experiments;
- test whether BG frequency affects withholding and cancellation differently;
- interpret dissociations as meaningful, not as noise.

### 4.4 Reaching decision / urgency anchor

The reaching-decision literature challenges the simple view that BG directly chooses targets. This is important because the experiment should measure urgency and commitment dynamics.

References:

- Thura D, Cisek P. 2017. *The Basal Ganglia Do Not Select Reach Targets but Control the Urgency of Commitment*. Neuron. https://pubmed.ncbi.nlm.nih.gov/28823728/
- Thura D, Cisek P. 2012. *Decision making by urgency gating: theory and experimental support*. https://cisek.org/pavel/Pubs/ThuraCisek2012.pdf

Use in this plan:

- include movement vigor;
- include speed-accuracy tradeoff;
- include commitment latency;
- avoid overclaiming that target identity is selected by BG.

### 4.5 OpenSim upper-limb validation anchor

The motor plant must be validated independently. OpenSim upper-limb models and reaching simulations provide the biomechanical reference base.

References:

- Saul KR et al. 2015. *Benchmarking of dynamic simulation predictions in two software platforms using an upper limb musculoskeletal model*. Computer Methods in Biomechanics and Biomedical Engineering. https://pmc.ncbi.nlm.nih.gov/articles/PMC4282829/
- SimTK Upper Extremity Reaching project: https://simtk.org/projects/ue-reaching
- SimTK Upper Extremity Dynamic Model / MoBL-ARMS: https://simtk.org/projects/upexdyn
- OpenSim musculoskeletal model list: https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090607/Musculoskeletal%2BModels

Use in this plan:

- validate reach kinematics before BG manipulation;
- avoid using pathological plant behavior as neuroscience evidence;
- select a plant with a tractable complexity level.

---

## 5. Core experimental intervention

The primary independent variable is:

> **Basal-ganglia effective update frequency.**

This is the frequency at which the BG module receives state and emits gating/selection output to the rest of the simulation.

Initial sweep:

| Condition | BG update interval | Approximate frequency |
|---|---:|---:|
| Very slow | 100 ms | 10 Hz |
| Slow | 50 ms | 20 Hz |
| Medium | 25 ms | 40 Hz |
| Fast | 12.5 ms | 80 Hz |
| Very fast | 6.25 ms | 160 Hz |

The first intervention should not alter internal BG physiology. It should alter only external update cadence and communication timing. This isolates the NRP-relevant causal factor: the effective timing bandwidth at which the BG module participates in the closed loop.

---

## 6. Timing model

The NRP logical clock should expose these variables explicitly:

| Variable | Definition |
|---|---|
| BG update frequency | How often BG receives state and emits an output. |
| BG input latency | Delay from cortex/task state to BG input. |
| BG output latency | Delay from BG output to thalamus/cortex. |
| Jitter | Trial-to-trial or message-to-message timing variability. |
| Dropout | Missing input or output messages. |
| Phase offset | Offset between BG update and other module update cycles. |

Synchronous logical time is not pretending that biological reality is synchronous. It is a scaffold that lets latency, dropout, jitter, and phase offsets be explicitly parameterized. That is the methodological advantage.

---

## 7. Experimental sequence

The experiment should progress through validation gates. Do not run full-stack frequency sweeps before lower-level sanity checks pass.

---

### Experiment 0: Component and interface validation

Purpose:

> Ensure the simulation is not producing artifacts before making causal claims.

Subtests:

1. **BG-alone channel competition**
   - Input: two or more action salience channels.
   - Output: selected channel, suppression of competitors.
   - Metrics: selection accuracy, selection latency, stability under salience gap variation.

2. **OpenSim reaching without BG manipulation**
   - Input: prescribed or controller-generated reach commands.
   - Output: reach trajectory.
   - Metrics: endpoint error, movement duration, smoothness, physically plausible activation/torque profiles.

3. **Cortex-BG-thalamus loop with abstract arm**
   - Use a simple kinematic arm or point-mass endpoint before full OpenSim.
   - Purpose: debug timing, gating, and state exchange.

4. **Full OpenSim arm integration**
   - Only after the abstract version works.

Acceptance criteria:

- BG module performs channel competition above chance.
- Arm reaches simple targets with stable trajectories.
- Message timestamps are logged correctly.
- Movement onset and endpoint can be detected automatically.
- BG output can be linked to behavioral events.

No causal claim is made at this stage.

---

### Experiment 1: Go/no-go reaching

Question:

> Does BG update frequency affect action withholding before movement commitment?

Task:

- Arm begins at rest.
- A cue appears.
- On go trials, the system reaches to a target.
- On no-go trials, the system must suppress movement.

Design:

| Factor | Values |
|---|---|
| BG frequency | 10, 20, 40, 80, 160 Hz |
| Trial type | go, no-go |
| Cue-target mapping | fixed initially; randomized later |

Primary metrics:

- go reaction time;
- no-go false alarm rate;
- movement onset latency;
- BG release latency;
- movement vigor;
- endpoint error on go trials.

Predictions:

| Hypothesis | Prediction |
|---|---|
| BG as action release controller | Low frequency increases RT and reduces vigor. |
| BG as suppression controller | Low frequency increases no-go false alarms. |
| BG frequency irrelevant | RT and false alarms are unchanged once latency is controlled. |

Validation anchors:

- no-go should be treated as withholding before commitment;
- it should not be interpreted as the same mechanism as stop-signal cancellation.

---

### Experiment 2: Two-choice reaching

Question:

> Does BG update frequency affect action identity selection, or mainly commitment timing?

Task:

- Arm starts at rest.
- Two targets are available.
- A cue indicates the correct target.
- System must reach to the indicated target.

Design:

| Factor | Values |
|---|---|
| BG frequency | 10, 20, 40, 80, 160 Hz |
| Target evidence / salience gap | easy, medium, hard |
| Target separation | large, medium, small |

Primary metrics:

- wrong-target rate;
- reaction time;
- selection latency;
- trajectory curvature toward wrong target;
- commitment latency;
- movement vigor;
- endpoint error.

Critical interpretation:

- If low BG frequency mainly increases wrong choices, that supports the selector-bottleneck account.
- If low BG frequency mainly shifts RT/vigor and speed-accuracy tradeoff, that supports the urgency/commitment account.
- If effects appear only in high-conflict conditions, then BG frequency is a conflict-dependent bottleneck rather than a generic motor bottleneck.

---

### Experiment 3: Stop-signal reaching

Question:

> Does BG update frequency affect reactive cancellation of an already initiated action?

Task:

- Go cue initiates a reach.
- On a subset of trials, a stop signal appears after a variable delay.
- The system must abort the movement.

Design requirements from stop-signal literature:

- Include go trials and stop trials.
- Use variable stop-signal delay.
- Generate an inhibition function: probability of responding as a function of stop-signal delay.
- Estimate an SSRT-like quantity.
- Compare failed-stop RTs with go RTs.
- Include validity checks.

Design:

| Factor | Values |
|---|---|
| BG frequency | 10, 20, 40, 80, 160 Hz |
| Stop-signal delay | adaptive staircase or fixed bins |
| Stop trial proportion | approximately 25%, adjustable |

Primary metrics:

- probability of responding on stop trials;
- inhibition function slope;
- SSRT-like estimate;
- failed-stop RT;
- residual movement amplitude;
- stop success rate;
- trigger-failure estimate.

Predictions:

| Hypothesis | Prediction |
|---|---|
| BG cancellation bottleneck | Low BG frequency increases SSRT-like estimates and stop failures. |
| BG release-only account | Go RT changes more than stopping efficiency. |
| Timing artifact | Effects disappear when latency/jitter are controlled. |

This is likely the highest-value Phase 1 experiment because it connects directly to established inhibition literature.

---

### Experiment 4: Change-of-mind reaching

Question:

> Does BG update frequency affect mid-course re-commitment from one action to another?

Task:

- Initial cue instructs reach to target A.
- A later cue instructs switch to target B.
- System must redirect the movement.

Design:

| Factor | Values |
|---|---|
| BG frequency | 10, 20, 40, 80, 160 Hz |
| Switch cue delay | early, medium, late, very late |
| Target separation | easy, medium, hard |

Primary metrics:

- switch success rate;
- redirection latency;
- curvature toward old target;
- residual movement toward abandoned target;
- final endpoint error;
- movement duration;
- correction cost.

Interpretation:

- If low BG frequency increases perseveration, it supports the switching-bottleneck interpretation.
- If low BG frequency produces slower but still accurate redirection, it supports urgency/timing rather than selection failure.
- If late switches fail regardless of BG frequency, the limitation may be biomechanical inertia rather than BG timing.

---

## 8. Latency, jitter, dropout, and phase decomposition

After the basic frequency sweep, add timing perturbations.

| Perturbation | Values |
|---|---|
| Fixed latency | 0, 10, 25, 50, 100 ms |
| Jitter | 0, +/-5, +/-10, +/-25 ms |
| Dropout | 0%, 1%, 5%, 10% |
| Phase offset | 0%, 25%, 50%, 75% of companion module update cycle |

Purpose:

> Separate the effect of update frequency from communication delay, temporal unreliability, and inter-module phase alignment.

Interpretation table:

| Observed pattern | Interpretation |
|---|---|
| Frequency effect persists at controlled latency | BG update frequency has independent causal value. |
| Fixed delay reproduces low-frequency behavior | Latency, not frequency, is the dominant bottleneck. |
| Jitter causes larger impairment than equal mean delay | Temporal precision matters more than average timing. |
| Dropout causes abrupt failures | BG outputs are event-critical. |
| Phase offset matters only in rhythmic tasks | Reserve for locomotion/tapping Phase 2. |

---

## 9. Metrics and logging specification

Every trial should log:

- trial ID;
- random seed;
- task type;
- cue identity;
- cue onset time;
- BG input receive time;
- BG output emit time;
- BG selected channel;
- BG channel activation values;
- thalamic relay/release time;
- motor command time series;
- movement onset time;
- endpoint trajectory;
- endpoint error;
- success/failure label;
- failure mode label;
- simulation runtime;
- real-time factor;
- message counts and dropped-message counts.

### Neural/interface metrics

| Metric | Definition |
|---|---|
| BG selection latency | Cue onset to BG selected-channel output. |
| BG dwell time | Duration for which a selected channel remains active. |
| BG switch latency | Old-channel suppression to new-channel release. |
| Selection entropy | Uncertainty across action channels. |
| Gating conflict | Simultaneous activation of incompatible channels. |
| Thalamic release latency | BG output to thalamic/cortical execution state. |

### Behavioral metrics

| Metric | Definition |
|---|---|
| Reaction time | Cue onset to movement onset. |
| Movement time | Movement onset to target acquisition. |
| Endpoint error | Final hand-target distance. |
| Wrong-target rate | Incorrect target reached. |
| False alarm rate | Movement on no-go trial. |
| Stop success | Movement successfully aborted after stop signal. |
| Switch success | Movement redirected to new target. |
| Perseveration | Continued execution of old action after switch/stop cue. |
| Trajectory curvature | Partial commitment toward non-selected target. |
| Movement vigor | Peak velocity or initial acceleration, depending on plant. |

### System metrics

| Metric | Definition |
|---|---|
| Compute cost | Wall-clock time per simulated second. |
| Real-time factor | Simulated time / wall-clock time. |
| Message rate | BG input/output message count per simulated second. |
| Numerical failures | Integrator or plant failures. |
| Interface failures | Timestamp, deserialization, or synchronization errors. |

---

## 10. Trial counts and statistical design

A reasonable first design:

| Experiment | Trials per frequency condition | Notes |
|---|---:|---|
| Go/no-go | 200 | Balanced go/no-go or 70/30 go/no-go. |
| Two-choice | 200-300 | Split across salience/conflict levels. |
| Stop-signal | 500+ | More trials needed for inhibition functions and SSRT-like estimates. |
| Change-of-mind | 300-500 | Split across switch delays. |

Use repeated random seeds across frequency conditions where possible. The same cue sequences should be reused across BG-frequency conditions to improve causal comparability.

Suggested analyses:

- generalized linear models for error/failure probabilities;
- mixed-effects models if using multiple simulated subjects/plants;
- survival/time-to-event analysis for movement onset and stop failure;
- frequency-response curves with confidence intervals;
- interaction models: BG frequency x conflict, BG frequency x stop delay, BG frequency x jitter.

Avoid overfitting detailed biological interpretations from a single simulated plant.

---

## 11. Validation criteria

### 11.1 BG module validity

The BG module must show plausible channel-competition behavior before whole-stack use:

- stable selection of high-salience channel;
- suppression of lower-salience channels;
- degraded selection under small salience gaps;
- no pathological oscillations unless explicitly induced.

### 11.2 Reaching plant validity

The OpenSim arm must pass a minimal kinematic screen:

- smooth reaches;
- plausible movement durations;
- bounded endpoint error;
- non-pathological torques / activations;
- stable behavior under repeated trials.

### 11.3 Stop-signal validity

The stop-signal experiment must produce:

- sensible go RT distribution;
- inhibition function that worsens as stop-signal delay increases;
- failed-stop RTs that are generally faster than ordinary go RTs;
- valid SSRT-like estimate;
- no obvious violation of independence assumptions without explanation.

### 11.4 Whole-stack validity

The whole stack must demonstrate that BG timing manipulations affect behavior through logged intermediate states, not through hidden simulation failures.

Required evidence:

- BG output timestamps align with behavioral changes;
- thalamic relay responds to BG output;
- motor command changes follow gating changes;
- OpenSim trajectory reflects command changes;
- failures can be classified mechanistically.

---

## 12. Expected outcome classes

### Strong positive result

A reproducible critical BG frequency band appears. Below this band:

- reaction time increases;
- no-go false alarms rise;
- stop-signal failures increase;
- switch perseveration increases;
- wrong-target rate may increase under high conflict.

This supports the bottleneck use case directly.

### Strong but nuanced result

BG frequency matters only in specific regimes:

- high conflict;
- late stop signal;
- high jitter;
- small salience gap;
- rapid change-of-mind.

This is probably the most biologically plausible positive outcome.

### Interpretation-shifting result

Frequency effects are explained mainly by latency or jitter decomposition.

This would redirect the use case from “frequency bottleneck” to:

> basal-ganglia communication timing precision as an action-control bottleneck.

Still valuable.

### Negative result

BG update frequency has little effect once latency, jitter, and plant dynamics are controlled.

This is still informative if the experiment is well designed, but it weakens the specific use case. It may imply that the chosen task is too slow, the BG model is too abstract, or the intervention is not biologically meaningful.

---

## 13. Methodological risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| BG model too abstract | Frequency manipulation may be arbitrary. | Validate BG-alone behavior; report intervention as effective communication frequency, not biological oscillation. |
| OpenSim plant dominates results | Biomechanics could swamp BG effects. | Start with abstract arm, then OpenSim; compare both. |
| Stop-signal implementation ad hoc | Reviewers will reject inhibition claims. | Follow Verbruggen et al. consensus structure. |
| Target selection framing too simplistic | BG may regulate urgency, not target identity. | Include vigor, commitment latency, and speed-accuracy metrics. |
| Synchronous clock misunderstood | Could look biologically naive. | State explicitly that logical time parameterizes latency, jitter, dropout, and phase; it is not a biological synchrony claim. |
| Whole-stack debugging burden | Too many modules fail at once. | Use validation ladder: BG-alone, plant-alone, abstract arm, full OpenSim. |
| Too many experiments | Scope creep. | Treat go/no-go and stop-signal as core; two-choice and change-of-mind as staged extensions. |

---

## 14. Recommended minimum publishable prototype

The minimum strong prototype should include:

1. BG-alone channel validation.
2. OpenSim or abstract-arm reaching validation.
3. Go/no-go reaching frequency sweep.
4. Stop-signal reaching frequency sweep following consensus methodology.
5. Latency/jitter decomposition for at least the stop-signal task.
6. Clear comparison of three interpretations:
   - selector bottleneck;
   - urgency/commitment bottleneck;
   - cancellation bottleneck.

The two-choice and change-of-mind tasks are desirable but not mandatory for the first prototype if resources are constrained.

---

## 15. Locomotion as Phase 2 or Phase 3

Once upper-limb results are stable, locomotion becomes valuable for a different reason:

> It introduces a natural rhythmic phase variable.

Later locomotion questions:

- Does BG update frequency interact with gait phase?
- Are stop/turn commands more effective at particular gait phases?
- Does jitter in BG output cause freezing-like or perseverative gait failures?
- Does BG timing interact with spinal CPG phase?

Candidate locomotion tasks:

- walk / stop;
- walk / turn;
- obstacle-triggered gait switch;
- step initiation;
- freeze-like failure induction.

This should remain a later phase because it introduces contact dynamics and spinal CPG complexity.

---

## 16. Final research-value assessment

This use case remains strong under the project criteria.

### 1. High-value causal question

The role of BG timing in action commitment, suppression, and switching is a meaningful causal question. It connects to basal-ganglia theories, stop-signal methodology, motor decision-making, and clinically relevant inhibition/switching deficits.

### 2. Naturally heterogeneous system

The question intrinsically involves interacting components: cortex, BG, thalamus, cerebellum, motor interface, and body. A single isolated model would miss the closed-loop timing problem.

### 3. Synchronous logical time is an advantage

Logical time enables precise manipulation of BG update frequency, latency, jitter, dropout, and phase offsets. It is not a claim that the brain is synchronous.

### 4. External validation is feasible

Validation can use established metrics:

- go/no-go false alarms;
- go reaction time;
- stop-signal inhibition functions;
- SSRT-like estimates;
- failed-stop RT distributions;
- reaching endpoint error;
- trajectory curvature;
- movement vigor.

### 5. First prototype is accessible

The first prototype can start with an abstract arm, then use OpenSim Arm26 or another upper-limb model. It does not require full locomotion or supercomputing.

### 6. Clear user / buyer

Potential users include:

- computational neuroscience groups;
- neurorobotics researchers;
- motor-control labs;
- Parkinsonian / basal-ganglia modeling researchers;
- NRP platform developers seeking timing-sensitive benchmark use cases.

### 7. Not easily achieved with simpler tools

The central claim depends on cross-module closed-loop timing. A static BG model or simple behavioral simulator would not test how BG timing interacts with embodied execution, thalamic gating, cerebellar correction, and plant dynamics.

---

## 17. Recommended next implementation step

The next concrete deliverable should be a **benchmark harness specification**, not another high-level concept note.

It should define:

1. message schema for BG input/output;
2. trial runner;
3. cue generator;
4. BG frequency scheduler;
5. latency/jitter/dropout injector;
6. event logger;
7. movement onset detector;
8. endpoint/error classifier;
9. stop-signal analysis module;
10. plotting/reporting pipeline.

The first implementation target should be:

> **Go/no-go and stop-signal reaching with an abstract arm, followed by OpenSim substitution after validation.**

That is the cleanest route to a scientifically interpretable and technically feasible prototype.
