# NRP_BGA-SB — Neuroscience Context and Results

**The central question** is whether basal ganglia (BG) update frequency constrains the temporal control of action commitment, suppression, and switching in an embodied reaching system. Three competing accounts existed: the BG as a *channel selector* (low frequency → wrong choices), as an *urgency/commitment controller* (low frequency → delayed or absent commitment, choices preserved), or as a *cancellation bottleneck* (low frequency → impaired stopping).

---

## The model

The BG core implements the Gurney-Prescott-Redgrave (GPR) 2001 rate-coded circuit — direct, indirect, and hyperdirect pathways over two action channels — embedded in a closed-loop pipeline: cortical evidence generator → scheduled BG → thalamic gate → motor plant. The thalamic gate translates the BG selection margin into a gain signal; marginal decisions (low margin) produce partial or no motor output. The motor plant was validated at three levels of biological realism: abstract motor command, 2D minimum-jerk kinematic reacher, and OpenSim Arm26 torque-controlled musculoskeletal simulation. Four task paradigms were run: go/no-go, two-choice conflict, stop-signal, and change-of-mind.

**BG update frequency** is operationalized as the rate at which the BG samples cortical evidence within a trial. Cortical evidence ramps from neutral (0.5, 0.5) to directed over a 200 ms accumulation window. At 5 Hz, the BG sampling period equals 200 ms, so the BG samples exactly once — at time zero, before evidence has risen — and receives only neutral input. At 10 Hz it samples twice; the second sample at 100 ms catches directed evidence above threshold. This creates a sharp behavioral boundary.

---

## Primary finding: the 5/10 Hz commitment threshold

At 5 Hz, all go trials miss and all go/no-go decisions fail, regardless of conflict level or task paradigm. At ≥10 Hz, selection succeeds with rates approaching 1.0 for low conflict, with higher conflict levels requiring proportionally higher frequencies (medium conflict: threshold ~20 Hz; high conflict: threshold ~80 Hz). The same boundary holds across kinematic and OpenSim embodiments — movement-onset rate is 0.000 at 5 Hz and 1.000 at ≥10 Hz in both plants, on identical BG decisions. The BG-frequency effect is not an abstraction artifact; it survives full musculoskeletal physics.

**The selector-bottleneck account is ruled out.** Wrong-channel choice rates do not increase at low frequency. When the BG fails at 5 Hz, it fails to commit at all (selected\_channel = −1), not to the wrong target. Channel selection errors remain near zero across all frequencies tested. The BG does not become a noisy selector at low update rates; it becomes a non-selector.

**The urgency/commitment account is supported.** Timing perturbations (fixed latency, jitter, phase offset) shift the selection latency proxy — effectively the RT — without changing which channel is selected or whether go trials succeed. The frequency effect on commitment is a precision-of-timing constraint: BG needs to sample often enough to catch the evidence peak within the accumulation window. Once it does, channel identity is determined correctly.

**The cancellation bottleneck account is supported.** On stop-signal trials, the BG must receive a second, suppressive evidence signal after the go cue. Dropout perturbations — which cause the BG to replay stale go decisions rather than receiving the current evidence — selectively elevate stop-failure rates without disturbing go-trial channel selection. This dissociation (dropout impairs stopping, not selecting) directly instantiates the cancellation bottleneck: updating the commitment decision in time to cancel an already-initiated response requires temporal precision that dropout removes.

**Change-of-mind.** At 5 Hz, change-of-mind probability is 0.0: the BG commits once and cannot update within the trial. At ≥10 Hz it is 1.0. This mirrors the go/no-go boundary exactly and confirms that the frequency constraint is not stopping-specific — it is a general limit on intra-trial commitment revision.

---

## Cerebellar independence

Under a 30° visuomotor rotation, the cerebellar adaptive filter (LMS trial-by-trial learning) drives endpoint deviation to zero across approximately 30 trials (θ̂ → −0.47 rad, convergence expected at −0.52 rad with longer training). Critically, movement-onset rate versus frequency is bit-identical with the cerebellum on or off. The BG-effect guard — the cerebellum is never called on closed-gate (non-executed) trials — is a structural invariant, not a tuning choice: there is no movement error signal to learn from when no movement occurs. The frequency threshold is a BG property; the cerebellum acts strictly downstream of the commitment decision and cannot compensate for its absence.

---

## Summary of interpretation verdicts

| Account | Verdict | Evidence |
|---|---|---|
| Selector bottleneck | Not supported | Wrong-channel rate does not rise at low frequency |
| Urgency / commitment | Supported | Latency/jitter shift RT; channel identity and success rate preserved |
| Cancellation bottleneck | Supported | Dropout selectively impairs stopping without disrupting go-trial selection |

The net picture is that BG update frequency governs *when* commitment occurs, not *which* commitment is made. A BG that samples too slowly simply never accumulates enough evidence to cross the selection threshold within the decision window. The mechanistic bottleneck is temporal: the period of BG sampling relative to the cortical evidence rise time. Below the threshold, action initiation, suppression, and revision all fail — not because the BG chooses incorrectly, but because it does not choose at all.
