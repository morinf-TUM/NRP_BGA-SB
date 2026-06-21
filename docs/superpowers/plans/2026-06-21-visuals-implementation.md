# Visual Deliverables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a matplotlib + ffmpeg pipeline that produces 4 static figures and 4 animated clips (plus a hero video) illustrating the key NRP_BGA-SB results for a website/portfolio demo.

**Architecture:** `visuals/` Python package with 6 modules (style → data_loader → trajectory_gen → figures → clips → assemble) plus an `experiments/generate_visuals.py` CLI entry point. matplotlib renders all content as PNG frame sequences; ffmpeg encodes per-clip MP4s and concatenates the hero video. No new pip dependencies.

**Tech Stack:** Python 3.10, matplotlib ≥ 3.10, numpy, ffmpeg 4.4 CLI, all `src/nrp_bga_sb/` modules.

## Global Constraints

- Python 3.10 — no walrus operator or match/case in f-strings
- No new `pip install` — matplotlib, numpy, ffmpeg are already present
- All generated output lands in `visuals/output/` (gitignored)
- Frame PNG dirs use 4-digit zero-padded names (`%04d.png`)
- Video: 1920×1080, 24 fps, libx264, pix_fmt yuv420p
- Static figures: 1920×1080 @ 150 dpi, light background
- Clips: dark background (`#0d1117` figure + axes background)
- All `src/nrp_bga_sb/` imports must be importable — run from project root
  with `PYTHONPATH=src` or install the package (`pip install -e .`)
- `visuals/output/` must be created before any file is written

---

## File Map

| File | Created/Modified | Responsibility |
|------|-----------------|----------------|
| `visuals/__init__.py` | Create | Package marker |
| `visuals/style.py` | Create | Theme dicts, colour palettes, `apply_theme()` |
| `visuals/data_loader.py` | Create | Load all result JSONs from `results/` |
| `visuals/trajectory_gen.py` | Create | On-the-fly trajectory generation via `ClosedLoopPolicy` + `KinematicReacher` + `Cerebellum` |
| `visuals/figures.py` | Create | Four static figure generator functions |
| `visuals/clips.py` | Create | Five frame-sequence writers (threshold, cerebellum, perturbation, interpretations, bridge) |
| `visuals/assemble.py` | Create | `encode_clip()`, `build_hero()` ffmpeg wrappers |
| `experiments/generate_visuals.py` | Create | CLI entry point: `--figures`, `--clips`, `--hero`, `--all`, `--dry-run` |
| `.gitignore` | Modify | Add `visuals/output/` |

---

## Task 1: Package scaffold, style, gitignore

**Files:**
- Create: `visuals/__init__.py`
- Create: `visuals/style.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `apply_theme(theme: dict) -> None`
  - `DARK_THEME: dict`  — matplotlib rcParams for dark background
  - `LIGHT_THEME: dict` — matplotlib rcParams for light background
  - `FREQ_COLORS: dict[int, str]` — per-frequency hex colours
  - `VERDICT_COLORS: dict[str, str]` — `"supported"` / `"not_supported"`

- [ ] **Step 1: Create the package marker**

```python
# visuals/__init__.py
```
(empty file)

- [ ] **Step 2: Create `visuals/style.py`**

```python
# visuals/style.py
from __future__ import annotations
import matplotlib as mpl

DARK_THEME: dict = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "text.color":        "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.alpha":        0.4,
    "lines.linewidth":   2.0,
    "font.size":         14,
    "axes.titlesize":    18,
    "axes.titleweight":  "bold",
    "figure.dpi":        96,
}

LIGHT_THEME: dict = {
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8f9fa",
    "axes.edgecolor":    "#dee2e6",
    "axes.labelcolor":   "#212529",
    "text.color":        "#212529",
    "xtick.color":       "#495057",
    "ytick.color":       "#495057",
    "grid.color":        "#dee2e6",
    "grid.alpha":        0.7,
    "lines.linewidth":   2.0,
    "font.size":         13,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
}

# Per-frequency colours: red for miss (5 Hz), green at threshold, blues/purples above
FREQ_COLORS: dict[int, str] = {
    5:   "#ef4444",   # red   — miss frequency
    10:  "#22c55e",   # green — selection threshold
    20:  "#3b82f6",   # blue
    40:  "#a855f7",   # purple
    80:  "#f59e0b",   # amber
    160: "#06b6d4",   # cyan
}

VERDICT_COLORS: dict[str, str] = {
    "supported":     "#22c55e",   # green
    "not_supported": "#ef4444",   # red
}

FIG_SIZE_1080P = (1920 / 150, 1080 / 150)   # inches at 150 dpi → 1920×1080


def apply_theme(theme: dict) -> None:
    """Apply an rcParams theme dict globally."""
    mpl.rcParams.update(theme)
```

- [ ] **Step 3: Add `visuals/output/` to `.gitignore`**

Open `.gitignore` and append:

```
# Generated visual outputs
visuals/output/
```

- [ ] **Step 4: Verify imports work**

```bash
cd /home/fom/code/NRP_BGA-SB
PYTHONPATH=src python3 -c "
from visuals.style import DARK_THEME, LIGHT_THEME, FREQ_COLORS, VERDICT_COLORS, apply_theme, FIG_SIZE_1080P
print('DARK_THEME keys:', len(DARK_THEME))
print('LIGHT_THEME keys:', len(LIGHT_THEME))
print('FREQ_COLORS:', FREQ_COLORS)
print('FIG_SIZE_1080P:', FIG_SIZE_1080P)
"
```

Expected: prints without error; `FIG_SIZE_1080P` ≈ `(12.8, 7.2)`.

- [ ] **Step 5: Commit**

```bash
git add visuals/__init__.py visuals/style.py .gitignore
git commit -m "feat: visuals/ package scaffold + style themes (Task V.1)

Dark/light rcParams presets, per-frequency palette, 1080p figure size.

ChangeSet-ID: visuals-scaffold"
```

---

## Task 2: Data loader

**Files:**
- Create: `visuals/data_loader.py`

**Interfaces:**
- Consumes: `results/*.json` (already committed)
- Produces:
  - `load_frequency_sweep() -> list[dict]`
  - `load_perturbation_gonogo() -> list[dict]`
  - `load_perturbation_stopsignal() -> list[dict]`
  - `load_cerebellum_results() -> list[dict]`
  - `load_bg_validation() -> list[dict]`
  - `load_opensim_gonogo() -> list[dict]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_loader.py
import pytest
from visuals.data_loader import (
    load_frequency_sweep,
    load_perturbation_gonogo,
    load_perturbation_stopsignal,
    load_cerebellum_results,
    load_bg_validation,
    load_opensim_gonogo,
)

def test_frequency_sweep_shape():
    data = load_frequency_sweep()
    assert len(data) == 900
    assert "frequency_hz" in data[0]
    assert "go_success_rate" in data[0]

def test_perturbation_gonogo_shape():
    data = load_perturbation_gonogo()
    assert len(data) == 85
    assert "perturbation_type" in data[0]
    assert "go_success_rate" in data[0]
    assert "bg_commitment_latency_mean" in data[0]

def test_perturbation_stopsignal_shape():
    data = load_perturbation_stopsignal()
    assert len(data) == 85
    assert "stop_failure_rate" in data[0]

def test_cerebellum_shape():
    data = load_cerebellum_results()
    assert len(data) == 50
    assert "cerebellum_enabled" in data[0]
    assert "endpoint_deviation_by_trial" in data[0]

def test_bg_validation_shape():
    data = load_bg_validation()
    assert len(data) == 3
    assert "conflict_level" in data[0]
    assert "mean_selection_latency_ms" in data[0]

def test_opensim_gonogo_shape():
    data = load_opensim_gonogo()
    assert len(data) == 5
    assert "opensim_movement_onset_rate" in data[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/fom/code/NRP_BGA-SB
PYTHONPATH=src python3 -m pytest tests/test_data_loader.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'visuals.data_loader'`

- [ ] **Step 3: Implement `visuals/data_loader.py`**

```python
# visuals/data_loader.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

# Results live at project_root/results/; this file is at project_root/visuals/
_RESULTS_DIR = Path(__file__).parent.parent / "results"


def _load(filename: str) -> Any:
    path = _RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def load_frequency_sweep() -> list[dict]:
    return _load("frequency_sweep_results.json")


def load_perturbation_gonogo() -> list[dict]:
    return _load("perturbation_sweep_gonogo.json")


def load_perturbation_stopsignal() -> list[dict]:
    return _load("perturbation_sweep_stopsignal.json")


def load_cerebellum_results() -> list[dict]:
    return _load("cerebellum_results.json")


def load_bg_validation() -> list[dict]:
    return _load("bg_validation.json")


def load_opensim_gonogo() -> list[dict]:
    return _load("opensim_gonogo_sweep.json")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src python3 -m pytest tests/test_data_loader.py -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add visuals/data_loader.py tests/test_data_loader.py
git commit -m "feat: visuals data loader + tests (Task V.2)

Thin wrappers around results/*.json with FileNotFoundError on missing files.

ChangeSet-ID: visuals-data-loader"
```

---

## Task 3: Trajectory generator

**Files:**
- Create: `visuals/trajectory_gen.py`

**Interfaces:**
- Consumes: `src/nrp_bga_sb/` — `make_closed_loop_policy`, `FrequencyConfig`, `GoNoGoConfig`, `run_go_nogo_trials`, `KinematicReacher`, `ReacherConfig`, `Cerebellum`, `VisuomotorRotation`
- Produces:
  - `VISUAL_REACHER_CONFIG: ReacherConfig` — upward target at (0, 1)
  - `THRESHOLD_FREQUENCIES: list[int]` — `[5, 10, 20, 40, 80]`
  - `generate_threshold_trajectories() -> list[dict]`  
    Each dict: `{frequency_hz, times_ms, positions_xy, selected_channel, gate_closed}`
  - `generate_cerebellum_trajectories(n_trials=30) -> list[dict]`  
    Each dict: `{trial_index, times_ms, positions_xy, endpoint_xy, theta_hat, is_go}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trajectory_gen.py
import pytest
from visuals.trajectory_gen import (
    generate_threshold_trajectories,
    generate_cerebellum_trajectories,
    THRESHOLD_FREQUENCIES,
    VISUAL_REACHER_CONFIG,
)

def test_threshold_returns_one_per_frequency():
    trials = generate_threshold_trajectories()
    assert len(trials) == len(THRESHOLD_FREQUENCIES)
    freqs = [t["frequency_hz"] for t in trials]
    assert freqs == THRESHOLD_FREQUENCIES

def test_threshold_5hz_is_miss():
    trials = generate_threshold_trajectories()
    trial_5 = next(t for t in trials if t["frequency_hz"] == 5)
    assert trial_5["gate_closed"] is True
    assert trial_5["selected_channel"] == -1
    # All positions should be [0.0, 0.0] for a miss
    assert all(p == [0.0, 0.0] for p in trial_5["positions_xy"])

def test_threshold_10hz_reaches_target():
    trials = generate_threshold_trajectories()
    trial_10 = next(t for t in trials if t["frequency_hz"] == 10)
    assert trial_10["gate_closed"] is False
    assert trial_10["selected_channel"] >= 0
    # Final position should be non-zero
    final = trial_10["positions_xy"][-1]
    assert final[0] ** 2 + final[1] ** 2 > 0.01

def test_threshold_trajectory_shape():
    trials = generate_threshold_trajectories()
    for t in trials:
        assert "times_ms" in t
        assert "positions_xy" in t
        assert len(t["times_ms"]) == len(t["positions_xy"])
        assert all(len(p) == 2 for p in t["positions_xy"])

def test_cerebellum_trajectories_count():
    trials = generate_cerebellum_trajectories(n_trials=10)
    # Only go-trials produce trajectories; with go_probability=1.0 all 10 are go
    assert len(trials) == 10

def test_cerebellum_trial_1_is_deflected():
    trials = generate_cerebellum_trajectories(n_trials=5)
    t = trials[0]
    assert t["is_go"] is True
    ep = t["endpoint_xy"]
    # At trial 1 theta_hat=0 → endpoint is at 30° rotation of (0,1) = (0.5, 0.866)
    assert abs(ep[0] - 0.5) < 0.05
    assert abs(ep[1] - 0.866) < 0.05

def test_cerebellum_theta_hat_increases():
    trials = generate_cerebellum_trajectories(n_trials=15)
    go_trials = [t for t in trials if t["is_go"]]
    assert len(go_trials) >= 2
    assert go_trials[-1]["theta_hat"] > go_trials[0]["theta_hat"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src python3 -m pytest tests/test_trajectory_gen.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'visuals.trajectory_gen'`

- [ ] **Step 3: Implement `visuals/trajectory_gen.py`**

```python
# visuals/trajectory_gen.py
"""On-the-fly trajectory generation for visual deliverables.

Imports from src/nrp_bga_sb/ to run the existing closed-loop pipeline and
produce (times_ms, positions_xy) arrays for animation.
"""
from __future__ import annotations

from nrp_bga_sb.cerebellum import Cerebellum
from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.perturbation_plant import VisuomotorRotation
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.scheduler import FrequencyConfig

# --- Shared configuration ---

# Single upward target so the reach looks natural and the 30° rotation
# deflects it clearly to the upper-right before cerebellar correction.
VISUAL_REACHER_CONFIG = ReacherConfig(
    target_positions=[[0.0, 1.0], [0.0, 1.0]],
    movement_duration_ms=300.0,
    dt_ms=1.0,
)

THRESHOLD_FREQUENCIES: list[int] = [5, 10, 20, 40, 80]

# GoNoGoConfig for a single go trial; decision_point_ms=200 matches
# the 200 ms accumulation window so frequency effects are observable.
_SINGLE_GO_CONFIG = GoNoGoConfig(
    n_trials=1,
    go_probability=1.0,
    response_window_start_ms=0,
    response_window_duration_ms=500,
    fixation_duration_ms=0,
    cue_onset_ms=0,
    decision_point_ms=200,
    seed=42,
)

_TOTAL_DURATION_MS = 500.0   # enough to capture the 300 ms reach


# --- Threshold trajectory generation ---

def generate_threshold_trajectories() -> list[dict]:
    """Run one go trial per frequency; return trajectory arrays.

    Returns:
        List of dicts (one per frequency in THRESHOLD_FREQUENCIES):
            frequency_hz  : int
            times_ms      : list[float]  — simulation time axis
            positions_xy  : list[[x, y]] — hand positions (all zero for misses)
            selected_channel : int       — -1 for miss
            gate_closed   : bool         — True means no movement
    """
    reacher = KinematicReacher(VISUAL_REACHER_CONFIG)
    results = []
    for freq in THRESHOLD_FREQUENCIES:
        policy = make_closed_loop_policy(
            frequency_config=FrequencyConfig.from_effective_hz(float(freq)),
            accumulation_ms=200.0,
        )
        trials = run_go_nogo_trials(_SINGLE_GO_CONFIG, policy)
        trial = trials[0]
        onset_ms = (
            trial.movement_onset_time * 1000.0
            if trial.movement_onset_time is not None
            else None
        )
        traj = reacher.simulate(
            trial.motor_command_series,
            onset_time_ms=onset_ms,
            total_duration_ms=_TOTAL_DURATION_MS,
        )
        gate_closed = traj.selected_channel == -1
        results.append({
            "frequency_hz":    freq,
            "times_ms":        traj.times_ms,
            "positions_xy":    traj.positions_xy,
            "selected_channel": traj.selected_channel,
            "gate_closed":     gate_closed,
        })
    return results


# --- Cerebellum trajectory generation ---

def generate_cerebellum_trajectories(n_trials: int = 30) -> list[dict]:
    """Run n_trials go-trials under 30° visuomotor rotation with AdaptiveFilter only.

    ForwardModelController is disabled (online_enabled=False) so the learning
    curve is visible across trials — the arc endpoint rotates gradually back
    toward the target as theta_hat builds from 0 → ~27°.

    Returns:
        List of dicts (one per trial):
            trial_index  : int
            times_ms     : list[float]
            positions_xy : list[[x, y]]  — perturbed + partially-corrected arc
            endpoint_xy  : list[float]   — final position [x, y]
            theta_hat    : float         — AdaptiveFilter state AFTER this trial
            is_go        : bool          — always True (go_probability=1.0)
    """
    reacher = KinematicReacher(VISUAL_REACHER_CONFIG)
    cerebellum = Cerebellum(
        learning_rate=0.1,
        adaptation_enabled=True,
        online_enabled=False,   # filter-only: learning curve visible across trials
    )
    perturbation = VisuomotorRotation(rotation_deg=30.0)

    config = GoNoGoConfig(
        n_trials=n_trials,
        go_probability=1.0,
        response_window_start_ms=0,
        response_window_duration_ms=500,
        fixation_duration_ms=0,
        cue_onset_ms=0,
        decision_point_ms=200,
        seed=42,
    )

    # Build a 40 Hz policy — reliably selects on every go trial
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        accumulation_ms=200.0,
    )
    trials = run_go_nogo_trials(config, policy)

    results = []
    for i, trial in enumerate(trials):
        onset_ms = (
            trial.movement_onset_time * 1000.0
            if trial.movement_onset_time is not None
            else None
        )
        traj = reacher.simulate_with_correction(
            trial.motor_command_series,
            onset_time_ms=onset_ms,
            total_duration_ms=_TOTAL_DURATION_MS,
            perturbation=perturbation,
            cerebellum=cerebellum,
        )
        endpoint = traj.positions_xy[-1]
        results.append({
            "trial_index":   i,
            "times_ms":      traj.times_ms,
            "positions_xy":  traj.positions_xy,
            "endpoint_xy":   endpoint,
            "theta_hat":     cerebellum.adaptive_filter.theta_hat,
            "is_go":         traj.selected_channel >= 0,
        })
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src python3 -m pytest tests/test_trajectory_gen.py -v
```

Expected: 7 PASSED. The first run may take 2–3 s (one trial per frequency + 30 cerebellar trials).

- [ ] **Step 5: Commit**

```bash
git add visuals/trajectory_gen.py tests/test_trajectory_gen.py
git commit -m "feat: visuals trajectory generator + tests (Task V.3)

generate_threshold_trajectories(): one go trial per [5,10,20,40,80] Hz.
generate_cerebellum_trajectories(): 30-trial filter-only (k=0) block under
30° visuomotor rotation — arc rotates toward target as theta_hat builds.

ChangeSet-ID: visuals-trajectory-gen"
```

---

## Task 4: Static figures

**Files:**
- Create: `visuals/figures.py`

**Interfaces:**
- Consumes: `visuals/data_loader.py` (all six loaders), `visuals/style.py`
- Produces:
  - `fig_frequency_threshold(output_dir: Path) -> Path`
  - `fig_perturbation_decomposition(output_dir: Path) -> Path`
  - `fig_cerebellum_learning(cereb_trials: list[dict], output_dir: Path) -> Path`
  - `fig_three_interpretations(output_dir: Path) -> Path`

Each function saves a PNG to `output_dir / <filename>` and returns the path.

- [ ] **Step 1: Write the smoke tests**

```python
# tests/test_figures_smoke.py
import pytest
from pathlib import Path
from visuals.figures import (
    fig_frequency_threshold,
    fig_perturbation_decomposition,
    fig_cerebellum_learning,
    fig_three_interpretations,
)
from visuals.trajectory_gen import generate_cerebellum_trajectories


def test_fig_frequency_threshold(tmp_path):
    out = fig_frequency_threshold(tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 10_000   # non-trivial file


def test_fig_perturbation_decomposition(tmp_path):
    out = fig_perturbation_decomposition(tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 10_000


def test_fig_cerebellum_learning(tmp_path):
    cereb_trials = generate_cerebellum_trajectories(n_trials=10)
    out = fig_cerebellum_learning(cereb_trials, tmp_path)
    assert out.exists()
    assert out.suffix == ".png"


def test_fig_three_interpretations(tmp_path):
    out = fig_three_interpretations(tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src python3 -m pytest tests/test_figures_smoke.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'visuals.figures'`

- [ ] **Step 3: Implement `visuals/figures.py`**

```python
# visuals/figures.py
"""Static figure generators (light background, 1920×1080 @ 150 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend: no display required
import matplotlib.pyplot as plt
import numpy as np

from visuals.data_loader import (
    load_frequency_sweep,
    load_perturbation_gonogo,
    load_perturbation_stopsignal,
    load_opensim_gonogo,
)
from visuals.style import (
    DARK_THEME,
    FREQ_COLORS,
    FIG_SIZE_1080P,
    LIGHT_THEME,
    VERDICT_COLORS,
    apply_theme,
)

_FREQS = [5, 10, 20, 40, 80, 160]


# --- Helper ---

def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Figure 1: BG frequency threshold ---

def fig_frequency_threshold(output_dir: Path) -> Path:
    """Go-success rate vs. BG update frequency — 5→10 Hz threshold step."""
    apply_theme(LIGHT_THEME)
    data = load_frequency_sweep()
    opensim = load_opensim_gonogo()

    # Aggregate: go_nogo, low conflict, mean ± std across seeds
    from collections import defaultdict
    by_freq: dict[float, list[float]] = defaultdict(list)
    for row in data:
        if row["paradigm"] == "go_nogo" and row["conflict_level"] == "low":
            by_freq[row["frequency_hz"]].append(row["go_success_rate"])

    freqs = sorted(by_freq)
    means = [np.mean(by_freq[f]) for f in freqs]
    stds  = [np.std(by_freq[f]) for f in freqs]

    fig, ax = plt.subplots(figsize=FIG_SIZE_1080P)

    # Shaded error band
    ax.fill_between(freqs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color="#3b82f6")

    # Main line
    ax.plot(freqs, means, "o-", color="#3b82f6", linewidth=2.5,
            markersize=8, label="Kinematic (go/no-go, low conflict)")

    # OpenSim overlay
    os_freqs = [r["frequency_hz"] for r in opensim]
    os_rates = [r["opensim_movement_onset_rate"] for r in opensim]
    ax.plot(os_freqs, os_rates, "s--", color="#f59e0b", linewidth=2,
            markersize=8, label="OpenSim Arm26")

    # Threshold annotation
    ax.axvline(x=10, color="#ef4444", linestyle=":", linewidth=1.5,
               label="Selection threshold (10 Hz)")

    ax.set_xscale("log")
    ax.set_xticks(freqs)
    ax.set_xticklabels([str(int(f)) for f in freqs])
    ax.set_xlabel("BG update frequency (Hz)", fontsize=14)
    ax.set_ylabel("Go-success rate", fontsize=14)
    ax.set_title("BG Update Frequency Governs Action Commitment", fontsize=16)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.4)

    return _save(fig, output_dir, "fig_frequency_threshold.png")


# --- Figure 2: Perturbation decomposition ---

def fig_perturbation_decomposition(output_dir: Path) -> Path:
    """4-panel figure: each perturbation type's effect on RT vs. channel selection."""
    apply_theme(LIGHT_THEME)
    gonogo = load_perturbation_gonogo()
    stopsig = load_perturbation_stopsignal()

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
    fig.suptitle("Timing Perturbations: RT vs. Channel Selection", fontsize=16, y=1.01)

    pert_types = [
        ("latency",      "Fixed Latency (ms)",       gonogo),
        ("jitter",       "Jitter Std (ms)",           gonogo),
        ("phase_offset", "Phase Offset (% period)",   gonogo),
        ("dropout",      "Dropout (%)",               stopsig),
    ]

    for (ptype, xlabel, dataset), ax in zip(pert_types, axes.flat):
        rows = [r for r in dataset
                if r["perturbation_type"] == ptype and r["frequency_hz"] == 40.0]
        rows.sort(key=lambda r: r["perturbation_value"])

        vals  = [r["perturbation_value"] for r in rows]

        # Normalise RT to baseline (perturbation_value == 0)
        rts = [r["bg_commitment_latency_mean"] for r in rows]
        rt0 = rts[0] if rts[0] > 0 else 1.0
        rts_norm = [v / rt0 for v in rts]

        if ptype == "dropout":
            # Stop-failure rate is the key channel-selection metric
            channel_metric = [r["stop_failure_rate"] for r in rows]
            channel_label = "Stop-failure rate"
        else:
            # go_success_rate (flat for latency/jitter/phase)
            channel_metric = [r["go_success_rate"] for r in rows]
            channel_label = "Go-success rate"

        ax.plot(vals, rts_norm, "o-", color="#3b82f6", label="RT (normalised)", linewidth=2)
        ax.plot(vals, channel_metric, "s--", color="#f59e0b",
                label=channel_label, linewidth=2)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(ptype.replace("_", " ").title(), fontsize=12)
        ax.set_ylim(-0.05, 1.6)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    axes[0, 0].set_ylabel("Metric value", fontsize=11)
    axes[1, 0].set_ylabel("Metric value", fontsize=11)
    fig.tight_layout()

    return _save(fig, output_dir, "fig_perturbation_decomposition.png")


# --- Figure 3: Cerebellum learning curve ---

def fig_cerebellum_learning(cereb_trials: list[dict], output_dir: Path) -> Path:
    """Trial-by-trial endpoint deviation: filter-only adaptation under 30° rotation."""
    apply_theme(LIGHT_THEME)

    go_trials = [t for t in cereb_trials if t["is_go"]]
    trial_nums = list(range(1, len(go_trials) + 1))

    # Unperturbed reference endpoint magnitude: target is at (0, 1) → magnitude 1.0
    target = np.array([0.0, 1.0])

    deviations = []
    thetas = []
    for t in go_trials:
        ep = np.array(t["endpoint_xy"])
        deviations.append(float(np.linalg.norm(ep - target)))
        thetas.append(t["theta_hat"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_1080P)
    fig.suptitle("Cerebellar Adaptation Under 30° Visuomotor Rotation", fontsize=16)

    # Panel 1: endpoint deviation over trials
    ax1.plot(trial_nums, deviations, "o-", color="#ef4444", linewidth=2, markersize=5)
    ax1.axhline(0, color="#22c55e", linestyle="--", linewidth=1.5, label="Target (no deviation)")
    ax1.set_xlabel("Trial number (go trials only)", fontsize=13)
    ax1.set_ylabel("Endpoint deviation (a.u.)", fontsize=13)
    ax1.set_title("Endpoint deviation decreasing across trials", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.4)

    # Panel 2: theta_hat convergence
    target_theta_deg = 30.0
    thetas_deg = [t * 180.0 / np.pi for t in thetas]
    ax2.plot(trial_nums, thetas_deg, "o-", color="#3b82f6", linewidth=2, markersize=5,
             label="θ̂ (learned counter-rotation)")
    ax2.axhline(target_theta_deg, color="#a855f7", linestyle="--", linewidth=1.5,
                label=f"Perturbation θ = {target_theta_deg}°")
    ax2.set_xlabel("Trial number (go trials only)", fontsize=13)
    ax2.set_ylabel("θ̂ (degrees)", fontsize=13)
    ax2.set_title("Adaptive filter converging toward 30°", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    return _save(fig, output_dir, "fig_cerebellum_learning.png")


# --- Figure 4: Three-interpretation verdict ---

def fig_three_interpretations(output_dir: Path) -> Path:
    """§11 verdict table as a visual summary card (light background)."""
    apply_theme(LIGHT_THEME)

    rows = [
        ("Selector bottleneck",
         "Wrong-channel choices\nrise at low frequency",
         "No wrong-channel selections\n(BG selects correctly or withholds)",
         "not_supported"),
        ("Urgency / commitment",
         "RT / vigor shift at low freq,\nchannel choice preserved",
         "✓ Latency/jitter shift RT\nwithout altering selected_channel",
         "supported"),
        ("Cancellation bottleneck",
         "Stop failures and SSRT\nworsen at low frequency",
         "✓ Flat inhibition (deterministic);\ndropout selectively impairs stopping",
         "supported"),
    ]
    col_headers = ["Account", "Prediction", "Observed", "Verdict"]

    fig, ax = plt.subplots(figsize=FIG_SIZE_1080P)
    ax.axis("off")
    fig.suptitle("Which Account Does the GPR Model Support?",
                 fontsize=18, fontweight="bold", y=0.96)

    col_widths = [0.22, 0.30, 0.35, 0.13]
    x_positions = [0.02, 0.24, 0.54, 0.89]
    y_start = 0.80
    row_h = 0.16

    # Header
    for xi, header in zip(x_positions, col_headers):
        ax.text(xi, y_start + 0.04, header,
                fontsize=13, fontweight="bold", transform=ax.transAxes,
                va="bottom")

    # Divider
    ax.axhline(y_start + 0.02, xmin=0.01, xmax=0.99,
               color="#aaaaaa", linewidth=1.0, transform=ax.transAxes)

    for i, (account, prediction, observed, verdict) in enumerate(rows):
        y = y_start - (i + 1) * row_h
        color = VERDICT_COLORS[verdict]
        verdict_sym = "✓" if verdict == "supported" else "✗"

        # Row background
        ax.add_patch(plt.Rectangle((0, y - 0.01), 1.0, row_h - 0.02,
                                    transform=ax.transAxes,
                                    color=color, alpha=0.06, linewidth=0))
        ax.text(x_positions[0], y + row_h / 2, account,
                fontsize=12, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(x_positions[1], y + row_h / 2, prediction,
                fontsize=11, transform=ax.transAxes, va="center")
        ax.text(x_positions[2], y + row_h / 2, observed,
                fontsize=11, transform=ax.transAxes, va="center")
        ax.text(x_positions[3], y + row_h / 2, verdict_sym,
                fontsize=22, color=color, fontweight="bold",
                transform=ax.transAxes, va="center", ha="center")

    return _save(fig, output_dir, "fig_three_interpretations.png")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src python3 -m pytest tests/test_figures_smoke.py -v
```

Expected: 4 PASSED (each test takes 1–3s to generate a figure).

- [ ] **Step 5: Commit**

```bash
git add visuals/figures.py tests/test_figures_smoke.py
git commit -m "feat: static figure generators + smoke tests (Task V.4)

Four 1920×1080 light-background PNGs: frequency threshold step,
perturbation decomposition 4-panel, cerebellum learning curve,
three-interpretations verdict table.

ChangeSet-ID: visuals-figures"
```

---

## Task 5: Clip frame generators

**Files:**
- Create: `visuals/clips.py`

**Interfaces:**
- Consumes: `visuals/style.py`, `visuals/trajectory_gen.py`
- Produces:
  - `write_threshold_frames(trials: list[dict], frames_dir: Path, dry_run: bool = False) -> int`
  - `write_cerebellum_frames(trials: list[dict], frames_dir: Path, dry_run: bool = False) -> int`
  - `write_perturbation_frames(gonogo: list[dict], stopsig: list[dict], frames_dir: Path, dry_run: bool = False) -> int`
  - `write_interpretations_frames(frames_dir: Path, dry_run: bool = False) -> int`
  - `write_bridge_frames(text: str, frames_dir: Path, n_frames: int = 120, dry_run: bool = False) -> int`

`dry_run=True` returns the expected frame count without writing any files.

- [ ] **Step 1: Write the dry-run tests**

```python
# tests/test_clips_dryrun.py
import pytest
from visuals.clips import (
    write_threshold_frames,
    write_cerebellum_frames,
    write_perturbation_frames,
    write_interpretations_frames,
    write_bridge_frames,
)
from visuals.trajectory_gen import (
    generate_threshold_trajectories,
    generate_cerebellum_trajectories,
)
from visuals.data_loader import load_perturbation_gonogo, load_perturbation_stopsignal


def test_threshold_dryrun_returns_positive_count():
    trials = generate_threshold_trajectories()
    n = write_threshold_frames(trials, None, dry_run=True)
    assert n > 0
    assert isinstance(n, int)


def test_cerebellum_dryrun_returns_positive_count():
    trials = generate_cerebellum_trajectories(n_trials=5)
    n = write_cerebellum_frames(trials, None, dry_run=True)
    assert n > 0


def test_perturbation_dryrun_returns_positive_count():
    gonogo  = load_perturbation_gonogo()
    stopsig = load_perturbation_stopsignal()
    n = write_perturbation_frames(gonogo, stopsig, None, dry_run=True)
    assert n > 0


def test_interpretations_dryrun_returns_positive_count():
    n = write_interpretations_frames(None, dry_run=True)
    assert n > 0


def test_bridge_dryrun_returns_n_frames():
    n = write_bridge_frames("Test text", None, n_frames=60, dry_run=True)
    assert n == 60


def test_threshold_writes_pngs(tmp_path):
    frames_dir = tmp_path / "threshold"
    trials = generate_threshold_trajectories()
    n = write_threshold_frames(trials, frames_dir, dry_run=False)
    assert n > 0
    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == n
    # Frame names are zero-padded 4-digit
    assert any(p.stem == "0000" for p in pngs)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src python3 -m pytest tests/test_clips_dryrun.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'visuals.clips'`

- [ ] **Step 3: Implement `visuals/clips.py`**

```python
# visuals/clips.py
"""Clip frame-sequence generators.

Each writer generates numbered PNGs in a directory. dry_run=True returns
the expected frame count without writing any files (used for smoke checks).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from visuals.style import (
    DARK_THEME,
    FREQ_COLORS,
    FIG_SIZE_1080P,
    VERDICT_COLORS,
    apply_theme,
)

FPS = 24
_BG = "#0d1117"     # figure background colour
_FG = "#e6edf3"     # text and axis colour


# --- Internal helpers ---

def _new_dark_fig() -> tuple[plt.Figure, plt.Axes]:
    """Return a 1080p figure + single axes with the dark theme applied."""
    apply_theme(DARK_THEME)
    fig, ax = plt.subplots(figsize=FIG_SIZE_1080P)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    return fig, ax


def _save_frame(fig: plt.Figure, frames_dir: Path, index: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(frames_dir / f"{index:04d}.png", dpi=96, bbox_inches="tight",
                facecolor=_BG)
    plt.close(fig)


def _draw_glow_arc(ax: plt.Axes,
                   xs: list[float],
                   ys: list[float],
                   color: str,
                   alpha: float = 1.0) -> None:
    """Draw a trajectory arc with a soft bloom/glow effect.

    Three passes: wide+faint outer glow, medium mid-glow, sharp core line.
    """
    for lw, a in [(10, 0.06 * alpha), (5, 0.14 * alpha), (1.5, alpha)]:
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=a,
                solid_capstyle="round")


def _title_card_frames(title_text: str, frames_dir: Path | None,
                       n_frames: int, dry_run: bool, start_index: int) -> int:
    """Write a full-screen title-card block and return its frame count."""
    if dry_run:
        return n_frames
    for i in range(n_frames):
        fig, ax = _new_dark_fig()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        alpha = min(1.0, i / max(1, FPS * 0.5))   # 0.5 s fade-in
        ax.text(0.5, 0.5, title_text, color=_FG, fontsize=24,
                ha="center", va="center", alpha=alpha,
                transform=ax.transAxes, fontweight="bold", wrap=True)
        _save_frame(fig, frames_dir, start_index + i)
    return n_frames


# --- Clip 1: Frequency threshold ---

_THRESHOLD_FRAMES_PER_FREQ = 72    # 3 s per frequency
_THRESHOLD_TITLE_FRAMES    = 48    # 2 s title card
_THRESHOLD_CAPTION_FRAMES  = 48    # 2 s final caption


def write_threshold_frames(
    trials: list[dict],
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Animate arm attempts at 5→80 Hz; MISS (5 Hz) then HIT (≥10 Hz)."""
    total = (_THRESHOLD_TITLE_FRAMES
             + len(trials) * _THRESHOLD_FRAMES_PER_FREQ
             + _THRESHOLD_CAPTION_FRAMES)
    if dry_run:
        return total

    idx = 0
    # Title card
    idx += _title_card_frames(
        "BG update frequency governs action commitment",
        frames_dir, _THRESHOLD_TITLE_FRAMES, dry_run=False, start_index=idx,
    )

    target_x, target_y = 0.0, 1.0
    hold   = FPS // 2         # 0.5 s hold before arc
    arc_f  = FPS              # 1 s arc draw
    result = FPS // 2         # 0.5 s result label hold

    for trial in trials:
        freq     = trial["frequency_hz"]
        color    = FREQ_COLORS.get(int(freq), "#e6edf3")
        hit      = not trial["gate_closed"]
        positions = trial["positions_xy"]

        # Hold: show target + frequency label only
        for i in range(hold):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.2, 1.6)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=200, zorder=5)
            ax.text(0.05, 0.95, f"{int(freq)} Hz", color=color, fontsize=20,
                    fontweight="bold", transform=ax.transAxes, va="top")
            _save_frame(fig, frames_dir, idx)
            idx += 1

        # Arc draw: progressively reveal the trajectory
        for i in range(arc_f):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.2, 1.6)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=200, zorder=5)
            ax.text(0.05, 0.95, f"{int(freq)} Hz", color=color, fontsize=20,
                    fontweight="bold", transform=ax.transAxes, va="top")
            if hit and positions:
                reveal = max(2, int((i + 1) / arc_f * len(positions)))
                xs = [p[0] for p in positions[:reveal]]
                ys = [p[1] for p in positions[:reveal]]
                _draw_glow_arc(ax, xs, ys, color)
                # Hand dot
                ax.scatter([xs[-1]], [ys[-1]], color=color, s=120, zorder=6)
            _save_frame(fig, frames_dir, idx)
            idx += 1

        # Result label
        label = "✓ HIT" if hit else "✗ MISS"
        label_color = "#22c55e" if hit else "#ef4444"
        for i in range(result):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.2, 1.6)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=200, zorder=5)
            ax.text(0.05, 0.95, f"{int(freq)} Hz", color=color, fontsize=20,
                    fontweight="bold", transform=ax.transAxes, va="top")
            if hit and positions:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                _draw_glow_arc(ax, xs, ys, color, alpha=0.7)
            ax.text(0.5, 0.15, label, color=label_color, fontsize=22,
                    fontweight="bold", ha="center", transform=ax.transAxes)
            _save_frame(fig, frames_dir, idx)
            idx += 1

    # Caption
    for i in range(_THRESHOLD_CAPTION_FRAMES):
        fig, ax = _new_dark_fig()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.2, 1.6)
        ax.axis("off")
        alpha = min(1.0, i / (FPS * 0.5))
        ax.text(0.5, 0.5,
                "Below 10 Hz the BG samples only\nneutral cortical evidence",
                color=_FG, fontsize=18, ha="center", va="center",
                transform=ax.transAxes, alpha=alpha)
        _save_frame(fig, frames_dir, idx)
        idx += 1

    return total


# --- Clip 2: Cerebellar adaptation ---

_CEREB_TITLE_FRAMES     = 48    # 2 s
_CEREB_FRAMES_PER_TRIAL = 20    # ~8.3 ms wall-clock per trial at 24 fps


def write_cerebellum_frames(
    trials: list[dict],
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Animate 30 go-trials showing arc rotating toward target as theta_hat builds."""
    go_trials = [t for t in trials if t["is_go"]]
    total = _CEREB_TITLE_FRAMES + len(go_trials) * _CEREB_FRAMES_PER_TRIAL
    if dry_run:
        return total

    idx = 0
    idx += _title_card_frames(
        "Cerebellum corrects visuomotor rotation across trials",
        frames_dir, _CEREB_TITLE_FRAMES, dry_run=False, start_index=idx,
    )

    target_x, target_y = 0.0, 1.0
    # Colour gradient: red (early, deflected) → green (late, corrected)
    n = len(go_trials)

    for ti, trial in enumerate(go_trials):
        frac   = ti / max(1, n - 1)       # 0.0 → 1.0 across trials
        r = int(239 * (1 - frac) + 34 * frac)
        g = int(68  * (1 - frac) + 197 * frac)
        b = int(68  * (1 - frac) + 94  * frac)
        color  = f"#{r:02x}{g:02x}{b:02x}"
        positions = trial["positions_xy"]
        theta_deg = trial["theta_hat"] * 180.0 / np.pi

        arc_f  = int(_CEREB_FRAMES_PER_TRIAL * 0.7)    # 70% drawing arc
        hold_f = _CEREB_FRAMES_PER_TRIAL - arc_f       # 30% hold

        for i in range(arc_f):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-0.2, 1.4)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=180, zorder=5,
                       label="Target")
            reveal = max(2, int((i + 1) / arc_f * len(positions)))
            xs = [p[0] for p in positions[:reveal]]
            ys = [p[1] for p in positions[:reveal]]
            _draw_glow_arc(ax, xs, ys, color)
            ax.scatter([xs[-1]], [ys[-1]], color=color, s=100, zorder=6)
            ax.text(0.05, 0.95, f"Trial {ti + 1}", color=_FG, fontsize=16,
                    fontweight="bold", transform=ax.transAxes, va="top")
            ax.text(0.05, 0.88, f"θ̂ = {theta_deg:.1f}°", color="#a855f7",
                    fontsize=13, transform=ax.transAxes, va="top")
            _save_frame(fig, frames_dir, idx)
            idx += 1

        for i in range(hold_f):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-0.2, 1.4)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=180, zorder=5)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            _draw_glow_arc(ax, xs, ys, color, alpha=0.6)
            ax.text(0.05, 0.95, f"Trial {ti + 1}", color=_FG, fontsize=16,
                    fontweight="bold", transform=ax.transAxes, va="top")
            ax.text(0.05, 0.88, f"θ̂ = {theta_deg:.1f}°", color="#a855f7",
                    fontsize=13, transform=ax.transAxes, va="top")
            _save_frame(fig, frames_dir, idx)
            idx += 1

    return total


# --- Clip 3: Perturbation decomposition ---

_PERT_TITLE_FRAMES  = 48
_PERT_DRAW_FRAMES   = 60    # per cell line draw
_PERT_LABEL_FRAMES  = 36    # label hold per cell
_PERT_HOLD_FRAMES   = 96    # final hold

_PERT_TYPES = [
    ("latency",      "Fixed latency (ms)",     "go_success_rate",   "Go-success rate",   False),
    ("jitter",       "Jitter std (ms)",         "go_success_rate",   "Go-success rate",   False),
    ("phase_offset", "Phase offset (% period)", "go_success_rate",   "Go-success rate",   False),
    ("dropout",      "Dropout (%)",             "stop_failure_rate", "Stop-failure rate", True),
]


def write_perturbation_frames(
    gonogo: list[dict],
    stopsig: list[dict],
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Animated 2×2 grid: each perturbation's RT vs. channel-selection metric."""
    n_cells = len(_PERT_TYPES)
    total = (_PERT_TITLE_FRAMES
             + n_cells * (_PERT_DRAW_FRAMES + _PERT_LABEL_FRAMES)
             + _PERT_HOLD_FRAMES)
    if dry_run:
        return total

    idx = 0
    idx += _title_card_frames(
        "Timing noise vs. signal integrity",
        frames_dir, _PERT_TITLE_FRAMES, dry_run=False, start_index=idx,
    )

    # Pre-compute cell data
    cells = []
    for ptype, xlabel, channel_key, channel_label, use_stopsig in _PERT_TYPES:
        dataset = stopsig if use_stopsig else gonogo
        rows = [r for r in dataset
                if r["perturbation_type"] == ptype and r["frequency_hz"] == 40.0]
        rows.sort(key=lambda r: r["perturbation_value"])
        vals = [r["perturbation_value"] for r in rows]
        rts  = [r["bg_commitment_latency_mean"] for r in rows]
        rt0  = rts[0] if rts[0] > 0 else 1e-6
        rts_norm = [v / rt0 for v in rts]
        ch = [r.get(channel_key, 0.0) or 0.0 for r in rows]
        label = ("urgency account" if ptype != "dropout"
                 else "cancellation bottleneck")
        cells.append((ptype, xlabel, vals, rts_norm, ch, channel_label, label, use_stopsig))

    def _draw_base(ax_list: list, reveal_cell: int, reveal_frac: float,
                   label_cells: set[int]) -> None:
        apply_theme(DARK_THEME)
        for ci, (ptype, xlabel, vals, rts_norm, ch, ch_label, verdict, _) in enumerate(cells):
            ax = ax_list[ci]
            ax.set_facecolor(_BG)
            ax.set_xlim(min(vals) - 0.5, max(vals) + 0.5)
            ax.set_ylim(-0.05, 1.7)
            ax.set_xlabel(xlabel, color=_FG, fontsize=10)
            ax.set_title(ptype.replace("_", " ").title(), color=_FG, fontsize=11)
            ax.tick_params(colors=_FG)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

            # Draw full lines for completed cells
            if ci < reveal_cell:
                ax.plot(vals, rts_norm, "o-", color="#3b82f6", linewidth=2,
                        label="RT (norm.)")
                ax.plot(vals, ch, "s--", color="#f59e0b", linewidth=2,
                        label=ch_label)
            elif ci == reveal_cell:
                n = max(2, int(reveal_frac * len(vals)))
                ax.plot(vals[:n], rts_norm[:n], "o-", color="#3b82f6", linewidth=2)
                ax.plot(vals[:n], ch[:n], "s--", color="#f59e0b", linewidth=2)

            if ci in label_cells:
                v_color = VERDICT_COLORS.get(
                    "not_supported" if verdict == "urgency account" else "supported",
                    "#ffffff",
                )
                # Override: both urgency and cancellation are "supported"
                v_color = "#22c55e"
                ax.text(0.97, 0.97, verdict, color=v_color, fontsize=9,
                        fontweight="bold", transform=ax.transAxes,
                        ha="right", va="top")

    for ci in range(n_cells):
        for fi in range(_PERT_DRAW_FRAMES):
            apply_theme(DARK_THEME)
            fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
            fig.patch.set_facecolor(_BG)
            _draw_base(axes.flat, ci, (fi + 1) / _PERT_DRAW_FRAMES, set())
            _save_frame(fig, frames_dir, idx)
            idx += 1

        for fi in range(_PERT_LABEL_FRAMES):
            apply_theme(DARK_THEME)
            fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
            fig.patch.set_facecolor(_BG)
            _draw_base(axes.flat, ci + 1, 0.0, set(range(ci + 1)))
            _save_frame(fig, frames_dir, idx)
            idx += 1

    # Final hold
    for fi in range(_PERT_HOLD_FRAMES):
        apply_theme(DARK_THEME)
        fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
        fig.patch.set_facecolor(_BG)
        _draw_base(axes.flat, n_cells, 1.0, set(range(n_cells)))
        _save_frame(fig, frames_dir, idx)
        idx += 1

    return total


# --- Clip 4: Interpretation verdict ---

_INTERP_TITLE_FRAMES = 48
_INTERP_ROW_FRAMES   = 72     # per row reveal
_INTERP_HOLD_FRAMES  = 96

_INTERP_ROWS = [
    ("Selector bottleneck",
     "Wrong-channel choices rise at low freq",
     "No wrong-channel selections observed",
     "not_supported", "✗"),
    ("Urgency / commitment",
     "RT shifts at low freq, choices preserved",
     "✓ Latency/jitter shift RT; channel intact",
     "supported", "✓"),
    ("Cancellation bottleneck",
     "Stop failures worsen at low frequency",
     "✓ Dropout selectively impairs stopping",
     "supported", "✓"),
]


def write_interpretations_frames(
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Verdict table fading in row by row."""
    total = (_INTERP_TITLE_FRAMES
             + len(_INTERP_ROWS) * _INTERP_ROW_FRAMES
             + _INTERP_HOLD_FRAMES)
    if dry_run:
        return total

    idx = 0
    idx += _title_card_frames(
        "Which account does the GPR model support?",
        frames_dir, _INTERP_TITLE_FRAMES, dry_run=False, start_index=idx,
    )

    def _draw_table(n_visible: int, alpha_last: float) -> plt.Figure:
        apply_theme(DARK_THEME)
        fig, ax = plt.subplots(figsize=FIG_SIZE_1080P)
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.axis("off")
        fig.suptitle("Which account does the GPR model support?",
                     color=_FG, fontsize=20, fontweight="bold", y=0.95)

        col_x = [0.03, 0.28, 0.58, 0.90]
        headers = ["Account", "Prediction", "Observed", "Verdict"]
        for cx, h in zip(col_x, headers):
            ax.text(cx, 0.80, h, color=_FG, fontsize=14, fontweight="bold",
                    transform=ax.transAxes, va="bottom")
        ax.axhline(0.78, xmin=0.01, xmax=0.99, color="#30363d", linewidth=1,
                   transform=ax.transAxes)

        row_h = 0.18
        for ri, (account, pred, obs, verdict, sym) in enumerate(_INTERP_ROWS):
            if ri >= n_visible:
                break
            a = alpha_last if ri == n_visible - 1 else 1.0
            y = 0.72 - ri * row_h
            color = VERDICT_COLORS[verdict]
            bg_alpha = 0.08 * a
            ax.add_patch(plt.Rectangle((0, y - 0.03), 1.0, row_h - 0.02,
                                        transform=ax.transAxes,
                                        color=color, alpha=bg_alpha))
            ax.text(col_x[0], y + 0.04, account, color=color,
                    fontsize=13, fontweight="bold", transform=ax.transAxes,
                    alpha=a, va="center")
            ax.text(col_x[1], y + 0.04, pred, color=_FG, fontsize=11,
                    transform=ax.transAxes, alpha=a, va="center")
            ax.text(col_x[2], y + 0.04, obs, color=_FG, fontsize=11,
                    transform=ax.transAxes, alpha=a, va="center")
            ax.text(col_x[3], y + 0.04, sym, color=color, fontsize=28,
                    fontweight="bold", transform=ax.transAxes,
                    ha="center", alpha=a, va="center")
        return fig

    for ri in range(len(_INTERP_ROWS)):
        for fi in range(_INTERP_ROW_FRAMES):
            alpha = min(1.0, (fi + 1) / (FPS * 0.75))
            fig = _draw_table(ri + 1, alpha)
            _save_frame(fig, frames_dir, idx)
            idx += 1

    for fi in range(_INTERP_HOLD_FRAMES):
        fig = _draw_table(len(_INTERP_ROWS), 1.0)
        _save_frame(fig, frames_dir, idx)
        idx += 1

    return total


# --- Bridge / title cards ---

def write_bridge_frames(
    text: str,
    frames_dir: Path | None,
    n_frames: int = 120,
    dry_run: bool = False,
) -> int:
    """Write a full-screen dark title card with centred text."""
    if dry_run:
        return n_frames
    return _title_card_frames(text, frames_dir, n_frames,
                               dry_run=False, start_index=0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src python3 -m pytest tests/test_clips_dryrun.py -v
```

Expected: 6 PASSED. The `test_threshold_writes_pngs` test will take 5–15 s (rendering ~500 PNG frames).

- [ ] **Step 5: Commit**

```bash
git add visuals/clips.py tests/test_clips_dryrun.py
git commit -m "feat: clip frame-sequence generators + dry-run tests (Task V.5)

Four animated clips: frequency threshold, cerebellar adaptation,
perturbation decomposition, interpretation verdict + bridge cards.
dry_run=True returns expected frame count without I/O.

ChangeSet-ID: visuals-clips"
```

---

## Task 6: ffmpeg assembler

**Files:**
- Create: `visuals/assemble.py`

**Interfaces:**
- Consumes: `visuals/output/frames/<clip>/` directories of PNGs
- Produces:
  - `encode_clip(frames_dir: Path, output_path: Path, fps: int = 24, fade_frames: int = 24) -> Path`
  - `build_hero(clip_paths: list[Path], output_path: Path) -> Path`

- [ ] **Step 1: Write tests for subprocess command construction**

```python
# tests/test_assemble.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from visuals.assemble import encode_clip, build_hero, _ffmpeg_encode_args, _concat_manifest


def test_ffmpeg_encode_args_contains_required_flags():
    args = _ffmpeg_encode_args(
        frames_dir=Path("/tmp/frames/threshold"),
        output_path=Path("/tmp/out/clip.mp4"),
        fps=24,
        fade_frames=24,
        n_frames=480,
    )
    cmd = " ".join(args)
    assert "-framerate" in cmd
    assert "24" in cmd
    assert "libx264" in cmd
    assert "yuv420p" in cmd
    assert "fade=in" in cmd
    assert "fade=out" in cmd
    assert "/tmp/out/clip.mp4" in cmd


def test_concat_manifest_lists_all_clips(tmp_path):
    clip_paths = [
        tmp_path / "clip_a.mp4",
        tmp_path / "clip_b.mp4",
        tmp_path / "clip_c.mp4",
    ]
    manifest = _concat_manifest(clip_paths)
    for p in clip_paths:
        assert str(p) in manifest
    assert manifest.count("file ") == 3


def test_encode_clip_calls_ffmpeg(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Create minimal dummy PNG files
    for i in range(5):
        (frames_dir / f"{i:04d}.png").write_bytes(b"PNG")

    out = tmp_path / "clip.mp4"
    with patch("visuals.assemble.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = encode_clip(frames_dir, out, fps=24, fade_frames=5)
    assert mock_run.called
    args_used = mock_run.call_args[0][0]
    assert "ffmpeg" in args_used[0]
    assert str(out) in args_used


def test_build_hero_calls_ffmpeg(tmp_path):
    clips = [tmp_path / f"clip_{i}.mp4" for i in range(3)]
    for c in clips:
        c.write_bytes(b"MP4")
    hero = tmp_path / "hero.mp4"
    with patch("visuals.assemble.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = build_hero(clips, hero)
    assert mock_run.called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src python3 -m pytest tests/test_assemble.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'visuals.assemble'`

- [ ] **Step 3: Implement `visuals/assemble.py`**

```python
# visuals/assemble.py
"""ffmpeg wrappers: PNG frame sequences → per-clip MP4s → hero concat."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _ffmpeg_encode_args(
    frames_dir: Path,
    output_path: Path,
    fps: int,
    fade_frames: int,
    n_frames: int,
) -> list[str]:
    """Build the ffmpeg argument list for encoding a frame sequence."""
    fade_out_start = max(0, n_frames - fade_frames)
    vf = f"fade=in:0:{fade_frames},fade=out:{fade_out_start}:{fade_frames}"
    return [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%04d.png"),
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]


def _concat_manifest(clip_paths: list[Path]) -> str:
    """Return the content of an ffmpeg concat manifest file."""
    return "\n".join(f"file '{p}'" for p in clip_paths)


def encode_clip(
    frames_dir: Path,
    output_path: Path,
    fps: int = 24,
    fade_frames: int = 24,
) -> Path:
    """Encode a numbered PNG frame directory into a single MP4.

    Args:
        frames_dir:   Directory containing %04d.png frames.
        output_path:  Destination MP4 path (created or overwritten).
        fps:          Output frame rate.
        fade_frames:  Number of frames for fade-in and fade-out.

    Returns:
        output_path on success.

    Raises:
        RuntimeError: if ffmpeg exits with a non-zero return code.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    args = _ffmpeg_encode_args(frames_dir, output_path, fps, fade_frames,
                                n_frames=len(pngs))
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_path


def build_hero(clip_paths: list[Path], output_path: Path) -> Path:
    """Concatenate multiple MP4 clips into a single hero video.

    Uses `ffmpeg -f concat -c copy` — no re-encoding, so all clips must
    share the same codec, resolution, and frame rate.

    Args:
        clip_paths:   Ordered list of existing MP4 files.
        output_path:  Destination hero MP4 path.

    Returns:
        output_path on success.

    Raises:
        FileNotFoundError: if any clip_path does not exist.
        RuntimeError: if ffmpeg exits with a non-zero return code.
    """
    for p in clip_paths:
        if not p.exists():
            raise FileNotFoundError(f"Clip not found: {p}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.parent / "concat_manifest.txt"
    manifest_path.write_text(_concat_manifest(clip_paths))

    args = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(manifest_path),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=src python3 -m pytest tests/test_assemble.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add visuals/assemble.py tests/test_assemble.py
git commit -m "feat: ffmpeg assembler + tests (Task V.6)

encode_clip(): PNG dir → libx264 MP4 with fade in/out.
build_hero(): concat manifest → re-encode-free hero video.
subprocess.run() calls verified via mock without running ffmpeg.

ChangeSet-ID: visuals-assemble"
```

---

## Task 7: CLI entry point + end-to-end run

**Files:**
- Create: `experiments/generate_visuals.py`

**Interfaces:**
- Consumes: all `visuals/` modules
- Produces: `visuals/output/` tree with all PNGs, clips, and hero video

- [ ] **Step 1: Implement `experiments/generate_visuals.py`**

```python
#!/usr/bin/env python3
# experiments/generate_visuals.py
"""CLI for generating all NRP_BGA-SB visual deliverables.

Usage:
    python experiments/generate_visuals.py --all
    python experiments/generate_visuals.py --figures
    python experiments/generate_visuals.py --clips
    python experiments/generate_visuals.py --hero
    python experiments/generate_visuals.py --dry-run

Run from the project root with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src/ to path if running directly (not as package)
_ROOT = Path(__file__).parent.parent
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visuals.assemble import build_hero, encode_clip
from visuals.clips import (
    write_bridge_frames,
    write_cerebellum_frames,
    write_interpretations_frames,
    write_perturbation_frames,
    write_threshold_frames,
)
from visuals.data_loader import load_perturbation_gonogo, load_perturbation_stopsignal
from visuals.figures import (
    fig_cerebellum_learning,
    fig_frequency_threshold,
    fig_perturbation_decomposition,
    fig_three_interpretations,
)
from visuals.trajectory_gen import (
    generate_cerebellum_trajectories,
    generate_threshold_trajectories,
)

OUTPUT_DIR  = _ROOT / "visuals" / "output"
FRAMES_DIR  = OUTPUT_DIR / "frames"
FPS         = 24


def _log(msg: str) -> None:
    print(f"[generate_visuals] {msg}", flush=True)


def run_figures(dry_run: bool) -> None:
    _log("Generating static figures …")
    out = OUTPUT_DIR / "figures"
    if dry_run:
        _log("  [dry-run] would write fig_frequency_threshold.png")
        _log("  [dry-run] would write fig_perturbation_decomposition.png")
        _log("  [dry-run] would write fig_cerebellum_learning.png")
        _log("  [dry-run] would write fig_three_interpretations.png")
        return
    cereb_trials = generate_cerebellum_trajectories(n_trials=30)
    fig_frequency_threshold(out)
    _log("  ✓ fig_frequency_threshold.png")
    fig_perturbation_decomposition(out)
    _log("  ✓ fig_perturbation_decomposition.png")
    fig_cerebellum_learning(cereb_trials, out)
    _log("  ✓ fig_cerebellum_learning.png")
    fig_three_interpretations(out)
    _log("  ✓ fig_three_interpretations.png")


def run_clips(dry_run: bool) -> None:
    _log("Generating clip frame sequences and encoding MP4s …")

    threshold_trials = generate_threshold_trajectories()
    cereb_trials     = generate_cerebellum_trajectories(n_trials=30)
    gonogo           = load_perturbation_gonogo()
    stopsig          = load_perturbation_stopsignal()

    # Bridge card texts
    bridges = {
        "title":    "NRP_BGA-SB\nBasal Ganglia Frequency & Action Selection",
        "bridge1":  "Does the effect survive\nfull musculoskeletal embodiment?\n\n"
                    "OpenSim Arm26 onset rate:\n5 Hz → 0.000 | ≥10 Hz → 1.000",
        "bridge2":  "What does timing noise reveal?",
        "closing":  "Conclusion:\nUrgency and cancellation-bottleneck\n"
                    "accounts supported.\nSelector-bottleneck ruled out.",
    }

    clips_spec = [
        ("threshold",      lambda d: write_threshold_frames(threshold_trials,
                                         FRAMES_DIR / "threshold", d)),
        ("cerebellum",     lambda d: write_cerebellum_frames(cereb_trials,
                                         FRAMES_DIR / "cerebellum", d)),
        ("perturbation",   lambda d: write_perturbation_frames(gonogo, stopsig,
                                         FRAMES_DIR / "perturbation", d)),
        ("interpretations",lambda d: write_interpretations_frames(
                                         FRAMES_DIR / "interpretations", d)),
    ]
    bridge_spec = [
        ("bridge_title",   bridges["title"],   120),
        ("bridge_1",       bridges["bridge1"], 120),
        ("bridge_2",       bridges["bridge2"], 120),
        ("bridge_closing", bridges["closing"], 120),
    ]

    for name, writer in clips_spec:
        t0 = time.time()
        n = writer(dry_run)
        if not dry_run:
            mp4 = OUTPUT_DIR / f"clip_{name}.mp4"
            encode_clip(FRAMES_DIR / name, mp4, fps=FPS, fade_frames=FPS)
            _log(f"  ✓ clip_{name}.mp4 ({n} frames, {time.time()-t0:.1f}s)")
        else:
            _log(f"  [dry-run] clip_{name}: {n} frames")

    for name, text, n_frames in bridge_spec:
        t0 = time.time()
        n = write_bridge_frames(text, FRAMES_DIR / name, n_frames, dry_run)
        if not dry_run:
            mp4 = OUTPUT_DIR / f"{name}.mp4"
            encode_clip(FRAMES_DIR / name, mp4, fps=FPS, fade_frames=FPS // 2)
            _log(f"  ✓ {name}.mp4 ({n} frames, {time.time()-t0:.1f}s)")
        else:
            _log(f"  [dry-run] {name}: {n} frames")


def run_hero(dry_run: bool) -> None:
    _log("Assembling hero video …")
    if dry_run:
        _log("  [dry-run] would concat all clips → hero_video.mp4")
        return
    ordered = [
        OUTPUT_DIR / "bridge_title.mp4",
        OUTPUT_DIR / "clip_threshold.mp4",
        OUTPUT_DIR / "bridge_1.mp4",
        OUTPUT_DIR / "clip_cerebellum.mp4",
        OUTPUT_DIR / "bridge_2.mp4",
        OUTPUT_DIR / "clip_perturbation.mp4",
        OUTPUT_DIR / "clip_interpretations.mp4",
        OUTPUT_DIR / "bridge_closing.mp4",
    ]
    hero = OUTPUT_DIR / "hero_video.mp4"
    build_hero(ordered, hero)
    _log(f"  ✓ hero_video.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NRP_BGA-SB visual deliverables."
    )
    parser.add_argument("--figures",  action="store_true", help="Static PNGs only")
    parser.add_argument("--clips",    action="store_true", help="Animated clip MP4s")
    parser.add_argument("--hero",     action="store_true", help="Hero video only")
    parser.add_argument("--all",      action="store_true", help="Everything")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Verify pipeline runs; no file output")
    args = parser.parse_args()

    if not any([args.figures, args.clips, args.hero, args.all]):
        parser.print_help()
        sys.exit(1)

    dry = args.dry_run
    if dry:
        _log("DRY RUN — no files will be written")

    if args.figures or args.all:
        run_figures(dry)
    if args.clips or args.all:
        run_clips(dry)
    if args.hero or args.all:
        run_hero(dry)

    _log("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — dry run**

```bash
cd /home/fom/code/NRP_BGA-SB
PYTHONPATH=src python3 experiments/generate_visuals.py --all --dry-run
```

Expected output (no errors, all lines print):
```
[generate_visuals] DRY RUN — no files will be written
[generate_visuals] Generating static figures …
[generate_visuals]   [dry-run] would write fig_frequency_threshold.png
...
[generate_visuals] Generating clip frame sequences and encoding MP4s …
[generate_visuals]   [dry-run] clip_threshold: 456 frames
...
[generate_visuals] Assembling hero video …
[generate_visuals]   [dry-run] would concat all clips → hero_video.mp4
[generate_visuals] Done.
```

- [ ] **Step 3: Generate static figures**

```bash
PYTHONPATH=src python3 experiments/generate_visuals.py --figures
```

Expected: 4 PNG files created in `visuals/output/figures/`. Each should be >100 KB and open correctly in an image viewer.

Verify:
```bash
ls -lh visuals/output/figures/
```

Expected: 4 files, each 200–600 KB.

- [ ] **Step 4: Generate all clips (long run ~5–15 min)**

```bash
PYTHONPATH=src python3 experiments/generate_visuals.py --clips
```

Expected: 8 MP4 files in `visuals/output/` (`clip_threshold.mp4`, `clip_cerebellum.mp4`, `clip_perturbation.mp4`, `clip_interpretations.mp4`, `bridge_title.mp4`, `bridge_1.mp4`, `bridge_2.mp4`, `bridge_closing.mp4`).

Verify:
```bash
ls -lh visuals/output/*.mp4
```

Play a clip to verify it looks correct:
```bash
ffplay visuals/output/clip_threshold.mp4    # or use any video player
```

- [ ] **Step 5: Assemble hero video**

```bash
PYTHONPATH=src python3 experiments/generate_visuals.py --hero
```

Expected: `visuals/output/hero_video.mp4` (~30–60 MB, ~2.5 min at 24 fps).

Verify duration:
```bash
ffprobe -v quiet -show_entries format=duration \
  -of default=noprint_wrappers=1 visuals/output/hero_video.mp4
```

Expected: `duration=140.000000` (±20s depending on actual frame counts).

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
PYTHONPATH=src python3 -m pytest tests/ -x -q --ignore=tests/opensim
```

Expected: 732 (or more) tests passed.

- [ ] **Step 7: Commit**

```bash
git add experiments/generate_visuals.py tests/test_data_loader.py \
        tests/test_trajectory_gen.py tests/test_figures_smoke.py \
        tests/test_clips_dryrun.py tests/test_assemble.py
git commit -m "feat: CLI entry point + integration smoke tests (Task V.7)

generate_visuals.py: --figures / --clips / --hero / --all / --dry-run.
Full pipeline verified: 4 static figs, 4 clips + 4 bridges → hero_video.mp4.

ChangeSet-ID: visuals-cli"
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `visuals/style.py` — dark/light themes, colour palettes | Task 1 |
| `visuals/data_loader.py` | Task 2 |
| `visuals/trajectory_gen.py` — threshold + cerebellum | Task 3 |
| 4 static figures (light, 1080p) | Task 4 |
| 4 clip frame generators + bridge cards | Task 5 |
| `visuals/assemble.py` — encode_clip, build_hero | Task 6 |
| `experiments/generate_visuals.py` CLI + --dry-run | Task 7 |
| `visuals/output/` gitignored | Task 1 Step 3 |
| No new pip dependencies | (all imports are already installed) |
| clip_threshold: 5 Hz miss + HIT/MISS labels + caption | Task 5 |
| clip_cerebellum: filter-only (k=0) 30-trial adaptation | Tasks 3 + 5 |
| clip_perturbation: 2×2 animated grid | Task 5 |
| clip_interpretations: row-by-row verdict table | Task 5 |
| Hero: title + clips + bridges + closing | Task 7 |
| Glow arc technique (3-layer overlaid plot) | Task 5 `_draw_glow_arc` |
| `fig_cerebellum_learning` uses filter-only run | Tasks 3 + 4 |

All spec requirements covered. No gaps found.

**Placeholder scan:** No TBD/TODO found. All code blocks are complete.

**Type consistency:**
- `generate_threshold_trajectories() -> list[dict]` with keys `frequency_hz, times_ms, positions_xy, selected_channel, gate_closed` — used correctly in `write_threshold_frames` (Task 5).
- `generate_cerebellum_trajectories() -> list[dict]` with keys `trial_index, times_ms, positions_xy, endpoint_xy, theta_hat, is_go` — used correctly in `write_cerebellum_frames` and `fig_cerebellum_learning`.
- `encode_clip(frames_dir, output_path, fps, fade_frames) -> Path` — called correctly in Task 7.
- `build_hero(clip_paths, output_path) -> Path` — called correctly in Task 7.
- `_ffmpeg_encode_args` is internal to `assemble.py` and tested directly in Task 6 — consistent with implementation.
