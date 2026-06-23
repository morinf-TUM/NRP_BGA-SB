# Go/No-Go: Prototype vs nrp-core Comparison

**Verdict:** 3 of 4 knobs reproduce the prototype exactly across the common grid (sampling, emission, commitment). The integration knob(s) diverge — see the per-knob table; this is the knob-2 idempotence finding (PROJECT_MEMORY §15.7) and motivates the separate knob-2 modeling project.

## Ablation (primary, like-for-like)

Cells are prototype/nrp go-success rate; ✗ marks a common-grid regime divergence.

| knob | 5 Hz | 10 Hz | 20 Hz | 40 Hz | 80 Hz | 160 Hz | holds |
|---|---|---|---|---|---|---|---|
| sampling | 0/0 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | ✓ |
| integration | 0/1 ✗ | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | ✗ @ 5 Hz |
| emission | 0/0 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | ✓ |
| commitment | 0/0 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | ✓ |

## Frequency sweep (secondary, qualitative)

> Prototype frequency-sweep rates aggregate conflict levels (low/medium/high), inflating the apparent success onset; only the qualitative monotone rise to 1.0 is comparable, not the onset frequency or per-frequency rates.

| side | 5 Hz | 10 Hz | 20 Hz | 40 Hz | 80 Hz | 160 Hz | onset |
|---|---|---|---|---|---|---|---|
| prototype | – | 0.333333 | 0.666667 | 0.666667 | 1 | 1 | 20 Hz |
| nrp-core | 0 | 1 | 1 | 1 | 1 | 1 | 10 Hz |
