# Open-Source Data Presenting Phenomena the NRP Mechanism Could Explain

Three strong candidates emerged from the search, in descending order of fit.

---

## 1. Freezing of Gait in Parkinson's Disease

**Dataset**: FoG-STAR, *Scientific Data* 13, 305 (2026) — freely downloadable, 22 participants,
IMU accelerometer/gyroscope from ankles, wrist and lower back, 101 annotated freeze episodes
across walking, dual-task walking, turning, and timed-up-and-go tasks.

**The unexplained phenomenon.** FoG is formally described as a "momentary breakdown of normal
gait" most common under cognitive dual-task load. The prevailing explanation — that competing
motor, cognitive, and limbic inputs cause BG output nuclei to fire synchronously, over-inhibiting
the brainstem locomotor system — is purely descriptive. It doesn't explain the *binary* character
of the failure: why does FoG manifest as *total* commitment failure (freeze) rather than reduced
vigor or wrong-step selection? And why do rhythmic external cues (auditory beats, floor stripes)
immediately break the freeze without any change in cognitive load?

**The NRP explanation.** The experiment shows that when the BG's sampling frequency drops below
the ratio of 1/(accumulation window), it sees only neutral evidence and cannot commit to any
channel. The *binary* quality — complete failure rather than partial movement — is a direct
consequence: it is not a graded motor deficit but a commitment threshold being missed entirely.
In gait, the relevant accumulation window is the anticipatory postural adjustment (APA) period
before step initiation (~200–400 ms). Under dual-task load, the effective BG sampling frequency
of directional motor evidence drops (the directional signal is buried in competing cognitive and
limbic signals), the BG sees near-neutral evidence within the APA window, and commitment fails
completely. External rhythmic cues effectively re-synchronize the BG update cycle to the
movement-planning phase, restoring the phase relationship between sampling and evidence peak —
exactly what the NRP mechanism predicts would rescue commitment. A quantitative test is possible:
compare IMU-derived APA window durations in freeze vs. non-freeze steps against STN beta
inter-burst intervals from simultaneous LFP recordings (available in related datasets).

---

## 2. The "Evidence Accumulation but not Inhibition" Puzzle in Stop-Signal Data

**Dataset**: IMAGEN cohort (n > 1,000, ages 19 and 23, open via IMAGEN consortium); ABCD Study
(open via NDA). Both contain stop-signal behavioral data and fMRI.

**The unexplained phenomenon.** A 2026 paper in *Neuropsychopharmacology* applied a mechanistic
Racing Diffusion model to the IMAGEN cohort and found that computational parameters for *evidence
accumulation* showed robust neural and clinical correlates (fronto-BG connectivity, substance use
vulnerability), but parameters for the *inhibitory process itself* (the stop "race") showed no
such correlates. This is paradoxical under standard race-model theory, which treats stopping as a
dedicated inhibitory subprocess with its own distinct neural machinery. The finding is replicated
across the ABCD dataset (where SSRT explained only 12% of neural variance). The question — if
response inhibition is a real, BG-mediated, separable process, where is its neural signature? —
is currently open.

**The NRP explanation.** The experiment demonstrates that stopping is not architecturally distinct
from going. Both are instances of the same commitment mechanism: the BG must sample evidence (go
evidence, or stop evidence) within the decision window. What predicts stopping success is whether
the BG's update rate is sufficient to catch the stop signal within the trial's temporal budget —
i.e., evidence accumulation timing, not a separate inhibitory latch. The SSRT is therefore not the
latency of an inhibitory subprocess but the minimum time the BG needs to sample and commit to the
stop channel. This is why evidence-accumulation parameters predict outcomes and inhibition
parameters do not: there is no separate inhibitory process to find a signature of.

---

## 3. The Inaction Bias and STN Beta Dissociation under Conflict

**Dataset**: PLOS Biology, January 2025 — Ging-Jehli et al., BG components have distinct
computational roles in decision-making under conflict and uncertainty. Data and analysis scripts
openly available at <https://osf.io/k38pj>. Includes behavioral data and intraoperative
DBS-LFP recordings from human Parkinson's patients during conflict and uncertainty tasks.

**The unexplained phenomenon.** The study found that STN beta oscillations (13–30 Hz) couple to
motor cortex and produce an *inaction bias* — a tendency not to initiate actions at all —
particularly in Parkinson's patients. At 20 Hz, the BG is sampling well above the NRP's 10 Hz
threshold; it should be able to commit. Yet elevated beta produces commitment failure. The
mechanistic link between beta *amplitude* and inaction is currently unexplained. Separately, the
study found STN theta (2–8 Hz) coupled to prefrontal cortex mediates conflict-related threshold
increases — another unexplained dissociation.

**The NRP explanation.** The NRP experiment's critical variable is not raw sampling frequency but
whether the BG samples *directional* evidence during each cycle. Pathological beta synchrony in
PD means all BG channels fire together at the same phase, so even when the cortex sends
directional evidence (channel A stronger than channel B), each BG sample sees the same pooled
activation across channels. The effective evidence differential at each sample is near zero —
equivalent to the NRP's neutral-evidence condition at 5 Hz. The BG fails to commit not because it
is slow, but because synchrony has destroyed the channel-discriminability that commitment
requires. DBS at ≥130 Hz is then explained not as "faster sampling" but as *desynchronization*
of pathological beta, restoring channel-discriminable sampling at the BG's natural rate. The
specific finding that 60–80 Hz DBS helps *freezing of gait* specifically (not all PD symptoms)
maps to the narrower motor action window: FoG requires rapid commitment within a short APA
window, and 60–80 Hz desynchronization is sufficient for that window even if it does not resolve
all other beta-mediated deficits.

---

## Summary

| Dataset | Phenomenon | Why currently unexplained | NRP fit |
|---|---|---|---|
| FoG-STAR (*Sci Data* 2026, free) | Binary commitment failure under dual-task; rescued by rhythmic cues | Overload model is descriptive; doesn't predict binary vs. graded failure | BG misses evidence peak within APA window → all-or-nothing; rhythmic cues re-phase sampling |
| IMAGEN / ABCD stop-signal (open via NDA/consortium) | Evidence accumulation predicts outcomes; inhibition parameters do not | Race model predicts separable inhibitory process with neural signature — none found | Stopping = same commitment mechanism applied to stop channel; no distinct inhibitory subprocess |
| Ging-Jehli et al. OSF (osf.io/k38pj, free) | STN beta inaction bias despite sampling frequency above threshold | Beta amplitude linked to inaction, but 20 Hz > 10 Hz threshold — mechanism unclear | Pathological beta synchrony destroys channel discriminability at each sample, equivalent to neutral evidence |

The common thread across all three: the NRP demonstrates that commitment failure is not simply
"too slow" — it is specifically about whether *directional* evidence is available to the BG
within a bounded time window. That framing, tested nowhere in the existing literature as a
mechanism, makes a precise and falsifiable prediction for each of these datasets.

---

## Sources

- [FoG-STAR dataset — Scientific Data 2026](https://www.nature.com/articles/s41597-026-06645-1)
- [Model-based analysis of stop-signal data reveals evidence accumulation but not inhibition — Neuropsychopharmacology 2026](https://www.nature.com/articles/s41386-026-02401-6)
- [Basal ganglia components have distinct computational roles — PLOS Biology 2025](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002978)
- [OSF data repository for PLOS Biology 2025 paper](https://osf.io/k38pj/?view_only=5c442294fcfb4991bb42cd902c60249c)
- [Effective DBS suppresses low-frequency BG oscillations by regularizing neural firing — PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3502634/)
- [High vs mid-frequency DBS differentially modulates response inhibition — PMC 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9841351/)
- [Design issues and solutions for ABCD stop-signal data — eLife](https://elifesciences.org/articles/60185)
- [Subthalamic stimulation causally modulates voluntary decision-making — npj Parkinson's 2024](https://www.nature.com/articles/s41531-024-00807-x)
- [Freezing of gait: an overload problem? — PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144986)
- [Neurocomputational model of cognitive load on freezing of gait — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5220109/)
