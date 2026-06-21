# Visual Deliverables — NRP_BGA-SB

**Date:** 2026-06-21  
**Status:** Approved  
**Venue:** Website / portfolio demo (standalone, self-explanatory)

---

## 1. Goals

Produce static scientific figures and an animated demo video that illustrate the
key results of the NRP_BGA-SB project for a general technical audience. No
presenter is required — visuals must stand alone.

---

## 2. Output deliverables

### 2.1 Static figures (PNG, light background, 1920×1080 @ 150 dpi)

| File | Content |
|------|---------|
| `fig_frequency_threshold.png` | Go-success rate vs. BG update frequency — the sharp 5→10 Hz step with 95% CI bands |
| `fig_perturbation_decomposition.png` | 4-panel: each perturbation type's effect on RT vs. go-success rate |
| `fig_cerebellum_learning.png` | Trial-by-trial endpoint deviation: cerebellum on vs. off, multiple seeds; uses filter-only config (k=0, `online_enabled=False`) so the learning curve is visible — same run as `clip_cerebellum` |
| `fig_three_interpretations.png` | §11 verdict table as a visual summary card |

### 2.2 Short clips (MP4, dark background, 1920×1080 @ 24 fps)

| File | Duration | Content |
|------|----------|---------|
| `clip_threshold.mp4` | ~20s | Arm sweeps 5→80 Hz; misses become hits at 10 Hz |
| `clip_cerebellum.mp4` | ~25s | Arm learns to correct 30° visuomotor rotation across 30 trials |
| `clip_perturbation.mp4` | ~20s | Animated 2×2 grid: latency/jitter shift RT; dropout breaks selection |
| `clip_interpretations.mp4` | ~15s | Verdict table fades in row by row |

### 2.3 Hero video (`hero_video.mp4`, dark background, ~2.5 min)

Assembly of all clips with title bridge cards:

```
[opening title] → clip_threshold → [bridge: embodiment?] →
[OpenSim table] → clip_cerebellum → [bridge: timing noise?] →
clip_perturbation → clip_interpretations → [closing verdict]
```

---

## 3. File layout

```
visuals/
  style.py          — shared colour palettes, rcParams presets
                       (DARK_THEME / LIGHT_THEME dictionaries)
  data_loader.py    — load + aggregate result JSONs from results/
  trajectory_gen.py — on-the-fly trajectory generation via ClosedLoopPolicy,
                       KinematicReacher, Cerebellum
  figures.py        — four static figure generator functions (light background)
  clips.py          — four clip frame-sequence generators + title/bridge cards
                       (dark background; writes numbered PNG dirs)
  assemble.py       — ffmpeg wrapper: PNG dirs → per-clip MP4s → hero concat

experiments/
  generate_visuals.py  — CLI entry point
                          --figures  static PNGs only
                          --clips    animated clip MP4s only
                          --hero     assemble hero_video.mp4 only
                          --all      everything
                          --dry-run  verify pipeline, no file output

visuals/output/      — gitignored; all frames, clips, hero land here
  frames/
    threshold/       — numbered PNGs for clip_threshold
    cerebellum/      — numbered PNGs for clip_cerebellum
    perturbation/    — numbered PNGs for clip_perturbation
    interpretations/ — numbered PNGs for clip_interpretations
    bridges/         — title card and bridge clip PNGs
  clip_threshold.mp4
  clip_cerebellum.mp4
  clip_perturbation.mp4
  clip_interpretations.mp4
  hero_video.mp4
  fig_frequency_threshold.png
  fig_perturbation_decomposition.png
  fig_cerebellum_learning.png
  fig_three_interpretations.png
```

---

## 4. Scene storyboard

### clip_threshold (~20s)

1. Title card fades in: *"BG update frequency governs action commitment"* (2s)
2. Dark panel: 2D target dot at right. Arm trajectory (glowing arc) attempts reach.
   Glow effect: the arc is drawn as 3–4 overlaid `ax.plot()` calls with
   decreasing linewidth and alpha (e.g. lw=8/α=0.15, lw=4/α=0.3, lw=2/α=1.0),
   giving a bloom without external image compositing.
3. Frequency label steps through 5 → 10 → 20 → 40 → 80 Hz (~3s per frequency).
4. 5 Hz: hand dot never moves — red "× MISS" label pulses.
5. 10 Hz: arc draws itself to target — green "✓ HIT" label.
6. 20–80 Hz: successful reach; BG commits progressively earlier.
7. Caption: *"Below 10 Hz the BG samples only neutral evidence"*

### clip_cerebellum (~25s)

1. Title card: *"Cerebellum corrects visuomotor rotation across trials"* (2s)
2. Target dot at expected position. Trial 1: arc deflected 30° (wrong direction).
3. Trial counter increments; each arc rotates progressively toward target.
4. By trial ~25 the arc lands exactly on target.
5. Mini-panel: *"BG onset signature: unchanged"* — onset rate vs. frequency stays
   flat at 0 for 5 Hz throughout the adaptation block.

### clip_perturbation (~20s)

1. Title card: *"Timing noise vs. signal integrity"* (2s)
2. 2×2 animated grid; each cell reveals its effect as a filling bar:
   - Latency → RT shifts, channel intact → *"urgency account"*
   - Jitter → same
   - Phase offset → RT shifts only
   - Dropout → stop-failure rate rises → *"cancellation bottleneck"*

### clip_interpretations (~15s)

1. Title card: *"Which account does the BG support?"* (2s)
2. Three-row verdict table fades in row by row, colour-coded:
   - Selector bottleneck → red ✗
   - Urgency / commitment → green ✓
   - Cancellation bottleneck → green ✓

### Hero video bridge cards

| Card | Duration | Text |
|------|----------|------|
| Opening | 5s | "NRP_BGA-SB: Basal Ganglia Frequency & Action Selection" + one-line abstract |
| Bridge 1 | 5s | "Does the effect survive full musculoskeletal embodiment?" + OpenSim onset-rate table |
| Bridge 2 | 5s | "What does timing noise reveal?" |
| Closing | 5s | Final verdict summary |

---

## 5. Data pipeline

### Static figures
Read directly from `results/*.json` — no re-simulation.

### clip_threshold trajectories
`trajectory_gen.py` runs one go trial per frequency:

```python
for freq in [5, 10, 20, 40, 80]:
    policy = make_closed_loop_policy(output_emission_hz=freq)
    trial  = run_go_nogo_trial(policy, cue="go", seed=42)
    traj   = KinematicReacher().simulate(trial.motor_command_series, ...)
    # → (times_ms, positions_xy, gate_state)
```

At 5 Hz `gate_state="closed"` — hand never moves.  
At ≥10 Hz minimum-jerk arc from origin to target.

### clip_cerebellum trajectories
30-trial block with `AdaptiveFilter` only (`ForwardModelController k=0`) so the
learning curve is visible across trials as θ̂ builds from 0→~27°:

```python
cerebellum   = Cerebellum(adaptation_enabled=True, online_enabled=False)
perturbation = VisuomotorRotation(theta_deg=30)
for _ in range(30):
    traj = KinematicReacher().simulate_with_correction(
               ..., perturbation=perturbation, cerebellum=cerebellum)
    # arc rotates toward target each trial
```

With α=0.1 and ~30 go-trials, θ̂ reaches ~0.50 rad (~96% of 30°), giving a clearly
partial-then-corrected arc sequence ideal for animation.

### clip_perturbation data
Read directly from `perturbation_sweep_gonogo.json` and
`perturbation_sweep_stopsignal.json`.

### clip_interpretations data
Hard-coded from the §11 verdict table in `PROJECT_MEMORY.md`.

---

## 6. Assembly pipeline

### Per-clip encoding

```bash
ffmpeg -framerate 24 -i visuals/output/frames/<clip>/%04d.png \
       -vf "fade=in:0:24,fade=out:<end-24>:24" \
       -c:v libx264 -pix_fmt yuv420p \
       visuals/output/<clip>.mp4
```

Each clip gets a 1-second fade-in and fade-out. `-pix_fmt yuv420p` ensures
broad browser compatibility.

Title card frames are generated by `clips.py` as plain matplotlib figures —
dark background, centred white text — no ffmpeg `drawtext` needed.

### Hero concat

```bash
# concat_manifest.txt — generated by assemble.py
file 'clip_title.mp4'
file 'clip_threshold.mp4'
file 'clip_bridge1.mp4'
file 'clip_cerebellum.mp4'
file 'clip_bridge2.mp4'
file 'clip_perturbation.mp4'
file 'clip_interpretations.mp4'
file 'clip_closing.mp4'

ffmpeg -f concat -safe 0 -i concat_manifest.txt -c copy hero_video.mp4
```

### Re-run strategy

- `--figures`: fully independent of all clip work.
- `--clips`: regenerates frame PNGs and re-encodes per-clip MP4s.
- `--hero`: re-runs concat only (clips must already exist).
- Frame directories are overwritten on each run; no stale-frame risk.

---

## 7. Dependencies

No new pip packages required beyond what is already installed:
- `matplotlib` ≥ 3.10 (already upgraded this session)
- `numpy` (project dependency)
- `ffmpeg` 4.4 CLI (present on host)
- All `src/nrp_bga_sb/` modules (used by `trajectory_gen.py`)

---

## 8. Constraints and non-goals

- No audio. Silent video with on-screen text only.
- No interactive HTML5 / JS animations (out of scope for this phase).
- No OpenSim trajectories in the animation — kinematic reacher only.
  (OpenSim onset-rate table appears as a static text card in the hero bridge.)
- `visuals/output/` is gitignored — generated artefacts are not committed.
- Tests are not added for this module (visual outputs cannot be meaningfully
  regression-tested; a `--dry-run` flag serves as the smoke check).
