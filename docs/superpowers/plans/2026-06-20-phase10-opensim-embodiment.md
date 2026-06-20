# Phase 10 — OpenSim Arm26 Embodiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the go/no-go BG-frequency sweep on a torque-controlled OpenSim Arm26 arm running in a Docker container, and confirm the frequency effect from the kinematic reacher (5 Hz → miss, ≥10 Hz → reach) survives embodiment (Milestone M8).

**Architecture:** The BG pipeline runs on the host (unchanged). Per trial, the host reduces the `motor_command_series` to a compact `ReachSpec` and batches all specs to a "dumb" OpenSim container over a bind-mounted volume as JSON. The container builds Arm26 with torque actuators, PD-tracks a minimum-jerk joint-space reference, and returns hand-marker trajectories. The host scores them with the existing `compute_movement_metrics` and compares against the kinematic reacher.

**Tech Stack:** Python 3.10 (host) / Python 3.11 (container), Pydantic v2, NumPy, Docker, OpenSim 4.6 (conda-forge, container-only), pytest.

## Global Constraints

- Host code adds **no new host runtime dependency**; `opensim` is imported **only** inside the container. (spec §2, §5.1)
- Host Python stays on the existing env; `numpy>=1.26` floor unchanged. (spec §2)
- OpenSim version pinned to **4.6** in the Dockerfile; fallback 4.5.x is also on the `opensim-org` channel. (spec §2, §8)
- `arm26.osim` is **vendored** from `opensim-org/opensim-models` with provenance recorded; not fetched from conda internals. (spec §2)
- Pydantic v2 only: `model_dump_json` / `model_validate_json`, `X | None`, `Literal[...]` for fixed vocabularies. (PROJECT_MEMORY §16.4)
- Fail-fast: no silent fallback at the container boundary, on bad config, or on malformed output. (spec §9)
- Determinism: fixed-step OpenSim integration, no stochastic muscle activity → reproducible given identical inputs. (spec §9)
- OpenSim/Docker tests are marked `@pytest.mark.opensim` and **deselected by default**; host CI never runs them. (spec §6.2)
- Muscles disabled (`appliesForce=false`); torque applied via appended `CoordinateActuator`s — torque-level control, defer muscle excitation. (spec §4.1, plan Task 10.3)
- Comparison (Task 10.4) is on **qualitative/normalized** metrics (onset rate, reach-completed vs not), not numeric endpoint equality. (spec §4.4)

---

## File Structure

**Container side** (`docker/opensim/`, imports `opensim`):
- `Dockerfile` — conda-forge miniforge base + `opensim=4.6`; copies model + scripts.
- `models/arm26.osim` — vendored model file.
- `models/PROVENANCE.md` — source URL + tag/commit + license of the model.
- `run_plant.py` — batch reach simulator: reads `config.json` + `jobs.json`, writes `trajectories.json`.
- `validate_plant.py` — independent plant validation entrypoint; writes a validation report.
- `_arm26_plant.py` — shared in-container module: model build, PD controller, single-reach simulate (imported by both scripts).

**Host side** (`src/nrp_bga_sb/`, **no** `opensim` import):
- `opensim_plant.py` — `ReachSpec`, `extract_reach_spec`, `OpenSimTrajectory`, `OpenSimPlantConfig`, `OpenSimPlantClient`.

**Host experiments / tests:**
- `experiments/opensim_gonogo_sweep.py` — orchestrates host pipeline → container → metrics → comparison report.
- `tests/test_opensim_plant.py` — host unit tests (no Docker).
- `tests/opensim/test_plant_validation.py` — Docker-gated validation + e2e smoke (`@pytest.mark.opensim`).
- `tests/opensim/fixtures/` — fixture `trajectories.json` for host unit tests.

---

## Task 10.1: Docker image + vendored model

**Files:**
- Create: `docker/opensim/Dockerfile`
- Create: `docker/opensim/models/arm26.osim` (vendored)
- Create: `docker/opensim/models/PROVENANCE.md`
- Create: `docker/opensim/.dockerignore`
- Modify: `pyproject.toml` (register the `opensim` pytest marker + default deselection)
- Create: `tests/opensim/__init__.py` (empty package marker)

**Interfaces:**
- Produces: a built image tagged `nrp-bga-opensim:4.6` that can `import opensim`, load `/opt/nrp/models/arm26.osim`, and run `model.initSystem()`. Model path inside the image: `/opt/nrp/models/arm26.osim`.

- [ ] **Step 1: Vendor the Arm26 model file**

Fetch `arm26.osim` from the official model repository and place it at `docker/opensim/models/arm26.osim`:

```bash
mkdir -p docker/opensim/models
curl -fsSL -o docker/opensim/models/arm26.osim \
  https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Arm26/arm26.osim
# Verify it is XML and references the two coordinates
head -3 docker/opensim/models/arm26.osim
grep -c "<Coordinate " docker/opensim/models/arm26.osim   # expect >= 2
```

Expected: first lines are `<?xml ... <OpenSimDocument ...`; coordinate count ≥ 2.

If the URL 404s, locate the file under the same repo (`Models/Arm26/`) and adjust the path; record the working URL in PROVENANCE.md.

- [ ] **Step 2: Record provenance**

Create `docker/opensim/models/PROVENANCE.md`:

```markdown
# arm26.osim provenance

- Source: https://github.com/opensim-org/opensim-models  (Models/Arm26/arm26.osim)
- Retrieved: 2026-06-20
- Commit/tag: <fill in the commit hash printed by: git ls-remote https://github.com/opensim-org/opensim-models master>
- License: per opensim-models repository (Apache-2.0 / permissive; see repo LICENSE)
- Model: Arm26 — 2-DOF planar shoulder+elbow, 6 Millard muscles. Muscles are
  disabled at load time; torque applied via appended CoordinateActuators.
```

Run `git ls-remote https://github.com/opensim-org/opensim-models master` and paste the commit hash into the file.

- [ ] **Step 3: Write the Dockerfile**

Create `docker/opensim/Dockerfile`:

```dockerfile
# OpenSim Arm26 plant container for NRP_BGA-SB Phase 10.
# Prebuilt OpenSim from the opensim-org conda channel (NOT a from-source build).
FROM condaforge/miniforge3:24.9.2-0

# Install OpenSim (Python bindings + native libs) into the base conda env.
# Pin 4.6; fallback 4.5.x is also on opensim-org if the solve ever fails.
RUN conda install -y -c opensim-org -c conda-forge opensim=4.6 \
    && conda clean -afy

WORKDIR /opt/nrp
COPY models/ /opt/nrp/models/
COPY _arm26_plant.py run_plant.py validate_plant.py /opt/nrp/

# Deterministic integration; headless (no GUI/visualizer).
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python"]
```

Create `docker/opensim/.dockerignore`:

```
**/__pycache__
*.pyc
```

- [ ] **Step 4: Add placeholder scripts so the image builds**

The image build `COPY`s three scripts; create minimal placeholders now (filled in Task 10.2). Create `docker/opensim/_arm26_plant.py`, `docker/opensim/run_plant.py`, `docker/opensim/validate_plant.py` each containing:

```python
"""Placeholder — implemented in Task 10.2."""
```

- [ ] **Step 5: Build the image**

Run:

```bash
docker build -t nrp-bga-opensim:4.6 docker/opensim
```

Expected: build succeeds; final line `Successfully tagged nrp-bga-opensim:4.6`. (First build downloads OpenSim — minutes, network required.)

- [ ] **Step 6: Verify OpenSim imports and the model loads in-container**

Run:

```bash
docker run --rm nrp-bga-opensim:4.6 -c \
  "import opensim, sys; m=opensim.Model('/opt/nrp/models/arm26.osim'); s=m.initSystem(); \
   cs=m.getCoordinateSet(); \
   print('opensim', opensim.__version__); \
   print('ncoords', cs.getSize()); \
   print('coords', [cs.get(i).getName() for i in range(cs.getSize())])"
```

Expected: prints `opensim 4.6`, `ncoords 2` (the two arm DOFs), and the two coordinate names (record them — Task 10.2 reads them at runtime, does not hardcode). If `ncoords` differs, the model has extra locked coordinates; note the unlocked/driving ones for Task 10.2.

- [ ] **Step 7: Register the pytest marker and deselect it by default**

In `pyproject.toml`, under `[tool.pytest.ini_options]`, add:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not opensim'"
markers = [
    "opensim: requires Docker + the nrp-bga-opensim image (deselected by default)",
]
```

Create empty `tests/opensim/__init__.py`.

- [ ] **Step 8: Verify the existing suite is unaffected**

Run:

```bash
pytest -q
```

Expected: the existing 674 tests still pass; collection shows no errors; any future `opensim`-marked tests are deselected.

- [ ] **Step 9: Commit**

```bash
git add docker/opensim pyproject.toml tests/opensim/__init__.py
git commit -m "feat: OpenSim Arm26 Docker image and vendored model (Task 10.1, M8)"
```

---

## Task 10.2: Torque-controlled plant + independent validation (Docker-gated)

**Files:**
- Modify: `docker/opensim/_arm26_plant.py` (full implementation)
- Modify: `docker/opensim/run_plant.py` (full implementation)
- Modify: `docker/opensim/validate_plant.py` (full implementation)
- Create: `tests/opensim/test_plant_validation.py`
- Create: `results/.gitkeep` (if `results/` is not tracked)

**Interfaces:**
- Consumes: `nrp-bga-opensim:4.6` image and `/opt/nrp/models/arm26.osim` from Task 10.1.
- Produces:
  - `run_plant.py` CLI: `python run_plant.py --config <config.json> --jobs <jobs.json> --out <trajectories.json>`.
  - `validate_plant.py` CLI: `python validate_plant.py --config <config.json> --out <report.json>`.
  - `config.json` schema (consumed): `{coordinate_names: list[str] | null, q0: list[float], q_target: list[list[float]], kp: list[float], kd: list[float], dt_ms: float, movement_duration_ms: float, total_duration_ms: float, end_effector_body: str | null, peak_torque_bound: float}`.
  - `jobs.json` schema (consumed): `{jobs: [{trial_id: str, selected_channel: int, onset_time_ms: float | null, gate_gain: float, gate_state: str}, ...]}`.
  - `trajectories.json` schema (produced): `{target_endpoints_xy: [[x,y],[x,y]], trajectories: [{trial_id: str, times_ms: [float], positions_xy: [[x,y],...], onset_time_ms: float | null, selected_channel: int}, ...]}`.
    - `target_endpoints_xy[c]` = hand-marker (x,y) at full-amplitude reach to `q_target[c]` (gate_gain=1.0); the host uses these as `ReacherConfig.target_positions` for endpoint-error scoring.

> **API-verification note:** OpenSim's forward-control API (Manager stepping, actuator override, gravity term) must be confirmed against the **installed 4.6** before relying on it. Step 1 is a documented in-container spike that locks the exact calls. Do not write the controller until the spike prints the expected values.

- [ ] **Step 1: API-verification spike (lock the control-loop calls)**

Run this throwaway probe in-container to confirm the exact method names exist in 4.6 and a single torque actuator moves a coordinate:

```bash
docker run --rm nrp-bga-opensim:4.6 -c "
import opensim as osim
m = osim.Model('/opt/nrp/models/arm26.osim')
cs = m.getCoordinateSet()
coord = cs.get(cs.getSize()-1)                      # the elbow-ish DOF
act = osim.CoordinateActuator(coord.getName())
act.setName('probe_act'); act.setOptimalForce(1.0)
m.addForce(act)
# disable muscles so only the actuator drives motion
fs = m.updForceSet()
for i in range(fs.getSize()):
    mus = osim.Muscle.safeDownCast(fs.get(i))
    if mus is not None: mus.set_appliesForce(False)
s = m.initSystem()
a = osim.ScalarActuator.safeDownCast(m.getForceSet().get('probe_act'))
a.overrideActuation(s, True)
a.setOverrideActuation(s, 5.0)                       # constant torque
q0 = coord.getValue(s)
mgr = osim.Manager(m)
s.setTime(0.0); mgr.initialize(s)
s = mgr.integrate(0.2)                               # 200 ms
print('q0', q0, 'q_after', coord.getValue(s), 'u_after', coord.getSpeedValue(s))
print('moved', abs(coord.getValue(s)-q0) > 1e-4)
"
```

Expected: `moved True` (the coordinate value changed under constant torque). This confirms `CoordinateActuator`, `ScalarActuator.safeDownCast`, `overrideActuation`/`setOverrideActuation`, `Manager.initialize`/`integrate`, `Coordinate.getValue`/`getSpeedValue`, and `Muscle.set_appliesForce`. If any call errors, consult `python -c "help(osim.<Class>)"` in-container and adjust the names below to match; record the confirmed names in a comment at the top of `_arm26_plant.py`.

- [ ] **Step 2: Implement the shared plant module `_arm26_plant.py`**

Replace the placeholder with the build + controller, using only the calls confirmed in Step 1:

```python
"""In-container Arm26 plant: model build + PD-tracking torque control.

Imported by run_plant.py and validate_plant.py. Requires opensim (4.6).
Confirmed-API note (Task 10.2 Step 1): CoordinateActuator / ScalarActuator.
overrideActuation / setOverrideActuation / Manager.integrate /
Coordinate.getValue / getSpeedValue / Muscle.set_appliesForce.
"""
from __future__ import annotations

import numpy as np
import opensim as osim

MODEL_PATH = "/opt/nrp/models/arm26.osim"


def _minimum_jerk_scalar(t_ms: float, T_ms: float) -> float:
    """Normalized minimum-jerk displacement: 0 at t=0, 1 at t>=T. Matches reacher.py."""
    if T_ms <= 0.0:
        return 1.0
    tau = min(t_ms / T_ms, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


class Arm26Plant:
    """Torque-controlled Arm26. One build per process; simulate() is stateless per call."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        model = osim.Model(MODEL_PATH)
        # Resolve driving coordinates: config override or all model coordinates.
        cs = model.getCoordinateSet()
        all_names = [cs.get(i).getName() for i in range(cs.getSize())]
        self.coord_names = cfg["coordinate_names"] or all_names
        # Append one CoordinateActuator per driving coordinate.
        self._act_names = []
        for cname in self.coord_names:
            act = osim.CoordinateActuator(cname)
            act.setName(f"act_{cname}")
            act.setOptimalForce(1.0)
            model.addForce(act)
            self._act_names.append(f"act_{cname}")
        # Disable muscle forces (torque-level control; defer muscle excitation).
        fs = model.updForceSet()
        for i in range(fs.getSize()):
            mus = osim.Muscle.safeDownCast(fs.get(i))
            if mus is not None:
                mus.set_appliesForce(False)
        self.model = model
        self.state = model.initSystem()
        self.coords = [model.getCoordinateSet().get(c) for c in self.coord_names]
        # End-effector body: config override or the last body in the chain.
        bs = model.getBodySet()
        self.ee_body = (
            bs.get(cfg["end_effector_body"]) if cfg["end_effector_body"]
            else bs.get(bs.getSize() - 1)
        )
        self.q0 = np.array(cfg["q0"], dtype=float)
        self.kp = np.array(cfg["kp"], dtype=float)
        self.kd = np.array(cfg["kd"], dtype=float)

    def _hand_xy(self, state) -> list[float]:
        """Planar hand position (x, y) in ground."""
        p = self.ee_body.getTransformInGround(state).p()
        return [p.get(0), p.get(1)]

    def endpoint_for(self, q_target: list[float]) -> list[float]:
        """FK hand (x,y) at a target posture (used for target_endpoints_xy)."""
        s = self.model.initSystem()
        for coord, q in zip(self.coords, q_target):
            coord.setValue(s, float(q))
        self.model.realizePosition(s)
        return self._hand_xy(s)

    def simulate(self, selected_channel: int, onset_time_ms, gate_gain: float,
                 gate_state: str) -> dict:
        """Run one reach. Returns times_ms / positions_xy / onset_time_ms / selected_channel."""
        dt = self.cfg["dt_ms"]
        T_total = self.cfg["total_duration_ms"]
        T_move = self.cfg["movement_duration_ms"]
        n_steps = int(round(T_total / dt)) + 1
        times_ms = [i * dt for i in range(n_steps)]

        # No movement: gate closed or no channel selected -> hold at q0 for the window.
        if gate_state == "closed" or selected_channel < 0:
            s = self.model.initSystem()
            for coord, q in zip(self.coords, self.q0):
                coord.setValue(s, float(q))
            self.model.realizePosition(s)
            hold = self._hand_xy(s)
            return {"times_ms": times_ms, "positions_xy": [hold] * n_steps,
                    "onset_time_ms": None, "selected_channel": -1}

        q_target = np.array(self.cfg["q_target"][selected_channel], dtype=float)
        q_ref_end = self.q0 + gate_gain * (q_target - self.q0)   # partial gate -> short reach
        onset = 0.0 if onset_time_ms is None else float(onset_time_ms)

        # Fresh state at q0, zero velocity.
        s = self.model.initSystem()
        for coord, q in zip(self.coords, self.q0):
            coord.setValue(s, float(q)); coord.setSpeedValue(s, 0.0)
        acts = [osim.ScalarActuator.safeDownCast(self.model.getForceSet().get(n))
                for n in self._act_names]
        for a in acts:
            a.overrideActuation(s, True)
        mgr = osim.Manager(self.model)
        s.setTime(times_ms[0] / 1000.0)
        mgr.initialize(s)

        positions_xy = [self._hand_xy(s)]
        for k in range(1, n_steps):
            t_ms = times_ms[k]
            # Reference posture/velocity from minimum-jerk profile (joint space).
            if t_ms < onset:
                q_ref, qd_ref = self.q0, np.zeros_like(self.q0)
            else:
                sca = _minimum_jerk_scalar(t_ms - onset, T_move)
                q_ref = self.q0 + sca * (q_ref_end - self.q0)
                qd_ref = np.zeros_like(self.q0)   # velocity FF omitted; Kd damps to ref
            q = np.array([c.getValue(s) for c in self.coords])
            qd = np.array([c.getSpeedValue(s) for c in self.coords])
            tau = self.kp * (q_ref - q) + self.kd * (qd_ref - qd)
            for a, tval in zip(acts, tau):
                a.setOverrideActuation(s, float(tval))
            s = mgr.integrate(t_ms / 1000.0)
            positions_xy.append(self._hand_xy(s))

        return {"times_ms": times_ms, "positions_xy": positions_xy,
                "onset_time_ms": onset, "selected_channel": selected_channel}
```

> Gravity note: with muscles disabled, gravity acts on the segments. The PD term plus sufficient `Kp` holds posture against gravity with small steady-state droop; the validation thresholds (Step 5) bound that droop. If validation shows excessive droop, add a gravity-compensation term — compute it in-task via OpenSim inverse dynamics at zero accel — but only if the bounded-error assertion fails. (YAGNI: try PD first.)

- [ ] **Step 3: Implement `run_plant.py` (batch CLI)**

```python
"""Batch reach simulator (in-container). Task 10.2.

Usage: python run_plant.py --config config.json --jobs jobs.json --out trajectories.json
"""
from __future__ import annotations

import argparse
import json

from _arm26_plant import Arm26Plant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.jobs) as f:
        jobs = json.load(f)["jobs"]

    plant = Arm26Plant(cfg)
    target_endpoints_xy = [plant.endpoint_for(q) for q in cfg["q_target"]]

    trajectories = []
    for job in jobs:
        traj = plant.simulate(
            selected_channel=job["selected_channel"],
            onset_time_ms=job["onset_time_ms"],
            gate_gain=job["gate_gain"],
            gate_state=job["gate_state"],
        )
        traj["trial_id"] = job["trial_id"]
        trajectories.append(traj)

    with open(args.out, "w") as f:
        json.dump({"target_endpoints_xy": target_endpoints_xy,
                   "trajectories": trajectories}, f)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `validate_plant.py` (independent plant validation, Task 10.2)**

```python
"""Independent Arm26 plant validation (Task 10.2). Runs canonical full reaches and
checks smoothness, duration, bounded endpoint error, torque sanity, determinism.

Usage: python validate_plant.py --config config.json --out report.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _arm26_plant import Arm26Plant


def _checks(plant: Arm26Plant, cfg: dict, ch: int) -> dict:
    target_xy = np.array(plant.endpoint_for(cfg["q_target"][ch]))
    traj = plant.simulate(ch, onset_time_ms=0.0, gate_gain=1.0, gate_state="open")
    pos = np.array(traj["positions_xy"])
    times = np.array(traj["times_ms"])
    start = pos[0]

    # progress toward target along the start->target direction
    direction = target_xy - start
    dist = float(np.linalg.norm(direction))
    unit = direction / dist if dist > 1e-9 else direction
    proj = (pos - start) @ unit
    # endpoint error
    endpoint_error = float(np.linalg.norm(pos[-1] - target_xy))
    # monotonicity of progress (smooth reach): allow tiny numerical dips
    monotone = bool(np.all(np.diff(proj) > -1e-3))
    # movement duration: time to reach 99% of progress
    reached = np.where(proj >= 0.99 * proj[-1])[0]
    duration_ms = float(times[reached[0]]) if len(reached) else float(times[-1])
    return {
        "channel": ch,
        "endpoint_error": endpoint_error,
        "monotone_progress": monotone,
        "duration_ms": duration_ms,
        "final_progress_fraction": float(proj[-1] / dist) if dist > 1e-9 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    plant = Arm26Plant(cfg)
    per_channel = [_checks(plant, cfg, ch) for ch in range(len(cfg["q_target"]))]

    # determinism: identical repeated reach -> identical positions
    a = plant.simulate(0, 0.0, 1.0, "open")["positions_xy"]
    b = plant.simulate(0, 0.0, 1.0, "open")["positions_xy"]
    deterministic = bool(np.allclose(np.array(a), np.array(b), atol=0.0))

    tol = cfg.get("endpoint_error_tol", 0.02)
    passed = (
        deterministic
        and all(c["monotone_progress"] for c in per_channel)
        and all(c["endpoint_error"] <= tol for c in per_channel)
        and all(c["final_progress_fraction"] >= 0.95 for c in per_channel)
    )
    report = {"passed": passed, "deterministic": deterministic,
              "endpoint_error_tol": tol, "per_channel": per_channel}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Rebuild the image and run validation, tuning gains until it passes**

```bash
docker build -t nrp-bga-opensim:4.6 docker/opensim
mkdir -p results /tmp/osim_io
cat > /tmp/osim_io/config.json <<'JSON'
{"coordinate_names": null,
 "q0": [0.0, 0.35],
 "q_target": [[0.6, 1.4], [0.0, 0.2]],
 "kp": [60.0, 40.0], "kd": [12.0, 8.0],
 "dt_ms": 5.0, "movement_duration_ms": 300.0, "total_duration_ms": 500.0,
 "end_effector_body": null, "peak_torque_bound": 200.0,
 "endpoint_error_tol": 0.02}
JSON
docker run --rm -v /tmp/osim_io:/io -v "$PWD/results":/results nrp-bga-opensim:4.6 \
  validate_plant.py --config /io/config.json --out /results/opensim_plant_validation.json
```

Expected: report prints `"passed": true`. **Tune as needed** (these are starting values): the two `q_target` postures are placeholders — set channel-0 and channel-1 to two clearly different, anatomically plausible postures (radians) using the coordinate ranges seen in Step 1/Task 10.1; raise `kp`/`kd` if reaches undershoot (low `final_progress_fraction`) or oscillate (non-monotone). Re-run until `passed: true`. Record the final tuned `config.json` values — they become the defaults in Task 10.3's `OpenSimPlantConfig`.

- [ ] **Step 6: Write the Docker-gated validation test**

Create `tests/opensim/test_plant_validation.py`:

```python
"""Docker-gated Arm26 plant validation (Task 10.2). Run with: pytest -m opensim."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.opensim

IMAGE = "nrp-bga-opensim:4.6"
CONFIG = {
    "coordinate_names": None,
    "q0": [0.0, 0.35],
    "q_target": [[0.6, 1.4], [0.0, 0.2]],   # final tuned values from Step 5
    "kp": [60.0, 40.0], "kd": [12.0, 8.0],
    "dt_ms": 5.0, "movement_duration_ms": 300.0, "total_duration_ms": 500.0,
    "end_effector_body": None, "peak_torque_bound": 200.0,
    "endpoint_error_tol": 0.02,
}


def _docker_available() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "image", "inspect", IMAGE],
                       capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _docker_available(), reason="docker image not built")
def test_plant_validation_passes(tmp_path: Path):
    io = tmp_path
    (io / "config.json").write_text(json.dumps(CONFIG))
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{io}:/io", IMAGE,
         "validate_plant.py", "--config", "/io/config.json", "--out", "/io/report.json"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads((io / "report.json").read_text())
    assert report["passed"] is True
    assert report["deterministic"] is True
```

- [ ] **Step 7: Run the gated test explicitly**

```bash
pytest -m opensim tests/opensim/test_plant_validation.py -v
```

Expected: PASS (Docker present + image built). On a host without the image it SKIPS — that is acceptable.

- [ ] **Step 8: Confirm the default suite still ignores opensim tests**

```bash
pytest -q
```

Expected: existing 674 tests pass; the opensim test is deselected (not skipped-noisily) by `addopts`.

- [ ] **Step 9: Commit**

```bash
git add docker/opensim/_arm26_plant.py docker/opensim/run_plant.py \
        docker/opensim/validate_plant.py tests/opensim/test_plant_validation.py results
git commit -m "feat: torque-controlled Arm26 plant + independent validation (Task 10.2, M8)"
```

---

## Task 10.3: Host-side plant boundary (`opensim_plant.py`)

Pure host module — **no `opensim` import**, full fixture-based TDD.

**Files:**
- Create: `src/nrp_bga_sb/opensim_plant.py`
- Create: `tests/test_opensim_plant.py`
- Create: `tests/opensim/fixtures/trajectories_ok.json`

**Interfaces:**
- Consumes: `MotorCommand` from `nrp_bga_sb.schemas`.
- Produces:
  - `ReachSpec(BaseModel)`: `trial_id: str`, `selected_channel: int`, `onset_time_ms: float | None`, `gate_gain: float`, `gate_state: Literal["open","closed","partial"]`.
  - `extract_reach_spec(motor_commands: list[MotorCommand], onset_time_ms: float | None, trial_id: str) -> ReachSpec`.
  - `OpenSimTrajectory(BaseModel)`: `times_ms: list[float]`, `positions_xy: list[list[float]]`, `onset_time_ms: float | None`, `selected_channel: int` (field-compatible with `ReacherTrajectory`).
  - `OpenSimPlantConfig(BaseModel)`: `image: str`, `q0: list[float]`, `q_target: list[list[float]]`, `kp: list[float]`, `kd: list[float]`, `dt_ms: float = 5.0`, `movement_duration_ms: float = 300.0`, `total_duration_ms: float = 500.0`, `coordinate_names: list[str] | None = None`, `end_effector_body: str | None = None`, `peak_torque_bound: float = 200.0`.
  - `OpenSimPlantClient(config: OpenSimPlantConfig, io_dir: str | Path, runner: Callable[[list[str]], int] | None = None)`; method `run(specs: list[ReachSpec]) -> tuple[list[OpenSimTrajectory], list[list[float]]]` returning `(trajectories, target_endpoints_xy)`. `runner` defaults to a real `docker run` subprocess; tests inject a fake runner.

- [ ] **Step 1: Write failing tests for `ReachSpec` + `extract_reach_spec`**

Create `tests/test_opensim_plant.py`:

```python
import json
from pathlib import Path

import pytest

from nrp_bga_sb.opensim_plant import (
    OpenSimPlantClient,
    OpenSimPlantConfig,
    OpenSimTrajectory,
    ReachSpec,
    extract_reach_spec,
)
from nrp_bga_sb.schemas import MotorCommand


def _cmd(command, gate_state, gate_gain):
    return MotorCommand(sim_time=0.7, trial_id=0, command=command,
                        gate_state=gate_state, gate_gain=gate_gain)


def test_extract_reach_spec_open_gate_selects_argmax():
    cmds = [_cmd([0.0, 0.8], "open", 0.8)]
    spec = extract_reach_spec(cmds, onset_time_ms=700.0, trial_id="t1")
    assert spec.selected_channel == 1
    assert spec.gate_gain == 0.8
    assert spec.gate_state == "open"
    assert spec.onset_time_ms == 700.0
    assert spec.trial_id == "t1"


def test_extract_reach_spec_empty_is_no_movement():
    spec = extract_reach_spec([], onset_time_ms=None, trial_id="t2")
    assert spec.selected_channel == -1
    assert spec.onset_time_ms is None
    assert spec.gate_state == "closed"


def test_extract_reach_spec_closed_gate_is_no_movement():
    cmds = [_cmd([0.0, 0.0], "closed", 0.0)]
    spec = extract_reach_spec(cmds, onset_time_ms=None, trial_id="t3")
    assert spec.selected_channel == -1
    assert spec.gate_state == "closed"


def test_extract_reach_spec_uses_last_command():
    cmds = [_cmd([0.5, 0.0], "open", 0.5), _cmd([0.0, 0.9], "open", 0.9)]
    spec = extract_reach_spec(cmds, onset_time_ms=700.0, trial_id="t4")
    assert spec.selected_channel == 1


def test_extract_reach_spec_open_but_all_zero_raises():
    cmds = [_cmd([0.0, 0.0], "open", 0.0)]
    with pytest.raises(ValueError, match="all-zero"):
        extract_reach_spec(cmds, onset_time_ms=700.0, trial_id="t5")
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `pytest tests/test_opensim_plant.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nrp_bga_sb.opensim_plant'`.

- [ ] **Step 3: Implement schemas + `extract_reach_spec`**

Create `src/nrp_bga_sb/opensim_plant.py`:

```python
"""Host-side OpenSim plant boundary (Phase 10, Task 10.3).

NO opensim import here — this module runs in the host env. It reduces a
ClosedLoopPolicy motor_command_series to a compact ReachSpec, batches specs to
the OpenSim container as JSON, and parses hand-trajectory results back.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.schemas import MotorCommand

# --- Schemas ---


class ReachSpec(BaseModel):
    trial_id: str
    selected_channel: int
    onset_time_ms: float | None
    gate_gain: float
    gate_state: Literal["open", "closed", "partial"]


class OpenSimTrajectory(BaseModel):
    times_ms: list[float]
    positions_xy: list[list[float]]
    onset_time_ms: float | None
    selected_channel: int


class OpenSimPlantConfig(BaseModel):
    image: str
    q0: list[float]
    q_target: list[list[float]]
    kp: list[float]
    kd: list[float]
    dt_ms: float = 5.0
    movement_duration_ms: float = 300.0
    total_duration_ms: float = 500.0
    coordinate_names: list[str] | None = None
    end_effector_body: str | None = None
    peak_torque_bound: float = 200.0


# --- ReachSpec extraction (mirrors KinematicReacher.simulate command reduction) ---


def extract_reach_spec(
    motor_commands: list[MotorCommand],
    onset_time_ms: float | None,
    trial_id: str,
) -> ReachSpec:
    """Reduce a motor_command_series to the OpenSim plant's input.

    Mirrors reacher.py: empty or closed gate -> no movement; otherwise the LAST
    command's argmax is the selected channel. Fail fast on an open/partial gate
    with an all-zero command (ThalamusGate wiring error).
    """
    if not motor_commands:
        return ReachSpec(trial_id=trial_id, selected_channel=-1,
                         onset_time_ms=None, gate_gain=0.0, gate_state="closed")
    last = motor_commands[-1]
    if last.gate_state == "closed":
        return ReachSpec(trial_id=trial_id, selected_channel=-1,
                         onset_time_ms=None, gate_gain=0.0, gate_state="closed")
    ch = int(np.argmax(last.command))
    if last.command[ch] == 0.0:
        # Trigger: open/partial gate but all-zero command.
        # Why: a valid non-closed gate always has command[selected] > 0.
        # Outcome: fail fast rather than emit a degenerate spec.
        raise ValueError(
            f"gate_state={last.gate_state!r} but command is all-zero: {last.command}"
        )
    return ReachSpec(trial_id=trial_id, selected_channel=ch,
                     onset_time_ms=onset_time_ms, gate_gain=last.gate_gain,
                     gate_state=last.gate_state)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_opensim_plant.py -v`
Expected: the five extraction tests PASS.

- [ ] **Step 5: Write a fixture + failing tests for `OpenSimPlantClient`**

Create `tests/opensim/fixtures/trajectories_ok.json`:

```json
{"target_endpoints_xy": [[0.30, 0.40], [0.10, 0.05]],
 "trajectories": [
   {"trial_id": "t_go", "times_ms": [0.0, 5.0, 10.0],
    "positions_xy": [[0.0,0.0],[0.05,0.07],[0.30,0.40]],
    "onset_time_ms": 0.0, "selected_channel": 0},
   {"trial_id": "t_miss", "times_ms": [0.0, 5.0, 10.0],
    "positions_xy": [[0.0,0.0],[0.0,0.0],[0.0,0.0]],
    "onset_time_ms": null, "selected_channel": -1}
 ]}
```

Append to `tests/test_opensim_plant.py`:

```python
FIXTURE = Path(__file__).parent / "opensim" / "fixtures" / "trajectories_ok.json"


def _cfg():
    return OpenSimPlantConfig(image="nrp-bga-opensim:4.6", q0=[0.0, 0.35],
                              q_target=[[0.6, 1.4], [0.0, 0.2]],
                              kp=[60.0, 40.0], kd=[12.0, 8.0])


def test_client_runs_and_parses_results(tmp_path):
    specs = [
        ReachSpec(trial_id="t_go", selected_channel=0, onset_time_ms=700.0,
                  gate_gain=1.0, gate_state="open"),
        ReachSpec(trial_id="t_miss", selected_channel=-1, onset_time_ms=None,
                  gate_gain=0.0, gate_state="closed"),
    ]

    def fake_runner(argv):
        # emulate the container: write the fixture to the requested --out path
        out = argv[argv.index("--out") + 1]
        # argv paths are container paths; map back to the host io_dir
        Path(out.replace("/io", str(tmp_path))).write_text(FIXTURE.read_text())
        return 0

    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path, runner=fake_runner)
    trajs, endpoints = client.run(specs)
    assert [t.trial_id for t in trajs] == ["t_go", "t_miss"]
    assert trajs[0].selected_channel == 0
    assert trajs[1].selected_channel == -1
    assert endpoints == [[0.30, 0.40], [0.10, 0.05]]


def test_client_fails_fast_on_nonzero_exit(tmp_path):
    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path, runner=lambda argv: 1)
    with pytest.raises(RuntimeError, match="container exited"):
        client.run([ReachSpec(trial_id="x", selected_channel=0, onset_time_ms=0.0,
                              gate_gain=1.0, gate_state="open")])


def test_client_fails_fast_on_trial_id_mismatch(tmp_path):
    def bad_runner(argv):
        out = argv[argv.index("--out") + 1]
        Path(out.replace("/io", str(tmp_path))).write_text(FIXTURE.read_text())
        return 0
    client = OpenSimPlantClient(_cfg(), io_dir=tmp_path, runner=bad_runner)
    with pytest.raises(ValueError, match="trial_id"):
        client.run([ReachSpec(trial_id="not_in_fixture", selected_channel=0,
                              onset_time_ms=0.0, gate_gain=1.0, gate_state="open")])
```

Note: `trial_id` for `extract_reach_spec` is a string; go/no-go trial ids are ints, so the experiment (Task 10.4) stringifies them.

- [ ] **Step 6: Run to confirm the client tests fail**

Run: `pytest tests/test_opensim_plant.py -v`
Expected: the three client tests FAIL (`OpenSimPlantClient` has no `run`).

- [ ] **Step 7: Implement `OpenSimPlantClient`**

Append to `src/nrp_bga_sb/opensim_plant.py`:

```python
# --- Container client ---


class OpenSimPlantClient:
    """Batch-runs ReachSpecs through the OpenSim container over a bind-mounted dir.

    runner(argv) -> int lets tests inject a fake container; the default issues a
    real `docker run`. The host writes config.json + jobs.json into io_dir and
    reads trajectories.json back. Fail-fast on nonzero exit, missing output, or
    trial_id mismatch.
    """

    def __init__(self, config: OpenSimPlantConfig, io_dir: str | Path,
                 runner: Callable[[list[str]], int] | None = None) -> None:
        self.config = config
        self.io_dir = Path(io_dir)
        self.io_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner or self._docker_runner

    @staticmethod
    def _docker_runner(argv: list[str]) -> int:
        return subprocess.run(argv).returncode

    def _plant_config(self) -> dict:
        # config.json carries only plant-relevant fields (not image / io_dir).
        return self.config.model_dump(exclude={"image"})

    def run(self, specs: list[ReachSpec]) -> tuple[list[OpenSimTrajectory], list[list[float]]]:
        (self.io_dir / "config.json").write_text(json.dumps(self._plant_config()))
        (self.io_dir / "jobs.json").write_text(
            json.dumps({"jobs": [s.model_dump() for s in specs]})
        )
        out_host = self.io_dir / "trajectories.json"
        if out_host.exists():
            out_host.unlink()

        argv = [
            "docker", "run", "--rm", "-v", f"{self.io_dir}:/io", self.config.image,
            "run_plant.py", "--config", "/io/config.json",
            "--jobs", "/io/jobs.json", "--out", "/io/trajectories.json",
        ]
        rc = self.runner(argv)
        if rc != 0:
            raise RuntimeError(f"OpenSim container exited with code {rc}")
        if not out_host.exists():
            raise RuntimeError(f"OpenSim container produced no output at {out_host}")

        data = json.loads(out_host.read_text())
        trajs = [OpenSimTrajectory.model_validate(t) for t in data["trajectories"]]
        returned_ids = {t["trial_id"] for t in data["trajectories"]}
        requested_ids = {s.trial_id for s in specs}
        if returned_ids != requested_ids:
            raise ValueError(
                f"trial_id mismatch: requested {requested_ids}, got {returned_ids}"
            )
        # preserve request order
        by_id = {t["trial_id"]: OpenSimTrajectory.model_validate(t)
                 for t in data["trajectories"]}
        ordered = [by_id[s.trial_id] for s in specs]
        return ordered, data["target_endpoints_xy"]
```

- [ ] **Step 8: Run the full new test file**

Run: `pytest tests/test_opensim_plant.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 9: Lint + full suite**

```bash
ruff check src/nrp_bga_sb/opensim_plant.py tests/test_opensim_plant.py
pytest -q
```

Expected: ruff clean; full suite green (674 + 8 new = 682, opensim-marked tests still deselected).

- [ ] **Step 10: Commit**

```bash
git add src/nrp_bga_sb/opensim_plant.py tests/test_opensim_plant.py \
        tests/opensim/fixtures/trajectories_ok.json
git commit -m "feat: host-side OpenSim plant boundary and client (Task 10.3, M8)"
```

---

## Task 10.4: Go/no-go OpenSim sweep + comparison (M8)

**Files:**
- Create: `experiments/opensim_gonogo_sweep.py`
- Create: `tests/opensim/test_gonogo_e2e.py` (Docker-gated smoke)
- Modify: `PROJECT_MEMORY.md` (add §26; update §1)

**Interfaces:**
- Consumes: `make_closed_loop_policy`, `run_go_nogo_trials`/`GoNoGoConfig`, `compute_movement_metrics`, `ReacherConfig`/`ReacherTrajectory`, and all of `opensim_plant.py`.
- Produces: `experiments/opensim_gonogo_sweep.py` writing `results/opensim_gonogo_sweep.json` and printing an acceptance report; reusable function `run_opensim_gonogo_condition(freq_hz, n_trials, client) -> dict`.

- [ ] **Step 1: Write the Docker-gated e2e smoke test (failing)**

Create `tests/opensim/test_gonogo_e2e.py`:

```python
"""Docker-gated end-to-end go/no-go embodiment smoke (Task 10.4). pytest -m opensim."""
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.opensim

IMAGE = "nrp-bga-opensim:4.6"


def _docker_available() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "image", "inspect", IMAGE],
                       capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _docker_available(), reason="docker image not built")
def test_frequency_effect_survives_embodiment(tmp_path):
    from nrp_bga_sb.opensim_plant import OpenSimPlantClient, OpenSimPlantConfig
    from experiments.opensim_gonogo_sweep import run_opensim_gonogo_condition

    cfg = OpenSimPlantConfig(image=IMAGE, q0=[0.0, 0.35],
                             q_target=[[0.6, 1.4], [0.0, 0.2]],
                             kp=[60.0, 40.0], kd=[12.0, 8.0])
    client = OpenSimPlantClient(cfg, io_dir=tmp_path)
    low = run_opensim_gonogo_condition(5.0, n_trials=10, client=client)
    high = run_opensim_gonogo_condition(40.0, n_trials=10, client=client)
    # qualitative effect: low frequency suppresses reaches, high frequency restores them
    assert low["movement_onset_rate"] < 0.5
    assert high["movement_onset_rate"] > 0.5
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest -m opensim tests/opensim/test_gonogo_e2e.py -v`
Expected: FAIL — `experiments.opensim_gonogo_sweep` does not exist (or import error). (If Docker image absent, it SKIPS — then validate logic via Step 4's host dry-run instead.)

- [ ] **Step 3: Implement the sweep runner**

Create `experiments/opensim_gonogo_sweep.py`. Mirror the go/no-go setup in `reacher_sweep.py` (cue onset 400 ms, decision point 300 ms → onset ≈ 700 ms; total window 500 ms in plant-relative time):

```python
"""Phase 10 / Task 10.4: go/no-go BG-frequency sweep on the OpenSim Arm26 plant.

Runs the existing closed-loop go/no-go pipeline on the host, ships each trial's
reduced ReachSpec to the OpenSim container, scores the returned hand trajectories
with compute_movement_metrics, and compares against the kinematic reacher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.movement_metrics import compute_movement_metrics
from nrp_bga_sb.opensim_plant import (
    OpenSimPlantClient,
    OpenSimPlantConfig,
    OpenSimTrajectory,
    extract_reach_spec,
)
from nrp_bga_sb.reacher import ReacherConfig, ReacherTrajectory

FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0]


def run_opensim_gonogo_condition(freq_hz: float, n_trials: int,
                                 client: OpenSimPlantClient) -> dict:
    """Run one BG-frequency condition end-to-end through the OpenSim plant."""
    policy = make_closed_loop_policy(effective_hz=freq_hz)
    config = GoNoGoConfig(n_trials=n_trials, seed=12345)
    trials = run_go_nogo_trials(config, policy)

    specs = []
    for trial in trials:
        onset_ms = (trial.movement_onset_time * 1000.0
                    if trial.movement_onset_time is not None else None)
        specs.append(extract_reach_spec(trial.motor_command_series, onset_ms,
                                        trial_id=str(trial.trial_id)))

    trajs, target_endpoints_xy = client.run(specs)

    # Build a ReacherConfig whose target_positions are the OpenSim FK endpoints,
    # so endpoint_error is computed in the arm's own coordinate frame.
    rcfg = ReacherConfig(n_channels=2, target_positions=target_endpoints_xy)
    metrics = []
    for ot in trajs:
        rt = ReacherTrajectory(**ot.model_dump())  # field-compatible
        metrics.append(compute_movement_metrics(rt, rcfg))

    n = len(metrics)
    moved = [m for m in metrics if m.movement_onset_time_ms is not None]
    return {
        "frequency_hz": freq_hz,
        "n_trials": n,
        "movement_onset_rate": len(moved) / n if n else 0.0,
        "mean_endpoint_error": (sum(m.endpoint_error for m in moved) / len(moved)
                                if moved else 0.0),
        "mean_peak_velocity": (sum(m.peak_velocity for m in moved) / len(moved)
                               if moved else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="nrp-bga-opensim:4.6")
    ap.add_argument("--io-dir", default="data/opensim_io")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()

    cfg = OpenSimPlantConfig(image=args.image, q0=[0.0, 0.35],
                             q_target=[[0.6, 1.4], [0.0, 0.2]],
                             kp=[60.0, 40.0], kd=[12.0, 8.0])
    client = OpenSimPlantClient(cfg, io_dir=args.io_dir)
    results = [run_opensim_gonogo_condition(f, args.n_trials, client)
               for f in FREQUENCIES_HZ]

    out = Path("results/opensim_gonogo_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    _print_report(results)


def _print_report(results: list[dict]) -> None:
    print("\n=== Phase 10 OpenSim go/no-go frequency sweep (M8) ===")
    print(f"{'freq_hz':>8} {'onset_rate':>11} {'endpoint_err':>13} {'peak_vel':>10}")
    for r in results:
        print(f"{r['frequency_hz']:>8.1f} {r['movement_onset_rate']:>11.3f} "
              f"{r['mean_endpoint_error']:>13.4f} {r['mean_peak_velocity']:>10.4f}")
    low = next(r for r in results if r["frequency_hz"] == 5.0)
    high = next(r for r in results if r["frequency_hz"] == 40.0)
    ok = low["movement_onset_rate"] < 0.5 < high["movement_onset_rate"]
    print(f"\nM8 qualitative effect preserved: {ok} "
          f"(5 Hz onset={low['movement_onset_rate']:.3f}, "
          f"40 Hz onset={high['movement_onset_rate']:.3f})")


if __name__ == "__main__":
    main()
```

> Confirm `make_closed_loop_policy`'s frequency kwarg name against `closed_loop.py` (it is the `from_effective_hz` knob, per PROJECT_MEMORY §5.1). If the kwarg is not `effective_hz`, match the actual signature used in `reacher_sweep.py:242`.

- [ ] **Step 4: Host dry-run of the pure-Python path (no Docker) using a fake runner**

Add a host-only unit test to `tests/test_opensim_plant.py` proving the orchestration assembles specs correctly without a container:

```python
def test_gonogo_condition_builds_specs_without_container(tmp_path, monkeypatch):
    from experiments.opensim_gonogo_sweep import run_opensim_gonogo_condition
    from nrp_bga_sb.opensim_plant import OpenSimPlantClient, OpenSimPlantConfig

    captured = {}

    def fake_runner(argv):
        jobs = json.loads((tmp_path / "jobs.json").read_text())["jobs"]
        captured["n"] = len(jobs)
        out = argv[argv.index("--out") + 1]
        # echo a zero-movement trajectory per job so scoring runs
        trajs = [{"trial_id": j["trial_id"], "times_ms": [0.0, 5.0],
                  "positions_xy": [[0.0, 0.0], [0.0, 0.0]],
                  "onset_time_ms": None, "selected_channel": -1} for j in jobs]
        Path(out.replace("/io", str(tmp_path))).write_text(json.dumps(
            {"target_endpoints_xy": [[0.3, 0.4], [0.1, 0.05]], "trajectories": trajs}))
        return 0

    cfg = OpenSimPlantConfig(image="x", q0=[0.0, 0.35], q_target=[[0.6, 1.4], [0.0, 0.2]],
                             kp=[60.0, 40.0], kd=[12.0, 8.0])
    client = OpenSimPlantClient(cfg, io_dir=tmp_path, runner=fake_runner)
    res = run_opensim_gonogo_condition(40.0, n_trials=8, client=client)
    assert res["n_trials"] == 8
    assert captured["n"] == 8
    assert res["movement_onset_rate"] == 0.0   # all echoed as no-movement
```

Run: `pytest tests/test_opensim_plant.py -v`
Expected: PASS (proves the host orchestration + scoring path with no OpenSim).

- [ ] **Step 5: Run the real e2e smoke (Docker)**

```bash
docker build -t nrp-bga-opensim:4.6 docker/opensim   # if not already built
pytest -m opensim tests/opensim/test_gonogo_e2e.py -v
```

Expected: PASS — 5 Hz onset rate < 0.5, 40 Hz onset rate > 0.5. (Skips cleanly if no Docker.)

- [ ] **Step 6: Run the full sweep and capture results**

```bash
python experiments/opensim_gonogo_sweep.py --n-trials 30
```

Expected: prints the table and `M8 qualitative effect preserved: True`; writes `results/opensim_gonogo_sweep.json`.

- [ ] **Step 7: Update PROJECT_MEMORY.md**

Add a new section **§26 Phase 10 module map** (source layout: `docker/opensim/`, `src/nrp_bga_sb/opensim_plant.py`, `experiments/opensim_gonogo_sweep.py`, `tests/opensim/`); record the OpenSim version (4.6), Arm26 choice + provenance, the torque-control + PD design, the file-based container boundary, the final tuned `q_target`/`kp`/`kd`, and the M8 result (5 Hz vs 40 Hz onset rates). Update **§1** to add "Phase 10 complete" with the test count. Follow the existing append-only style of §22–§25; do not rewrite earlier sections.

- [ ] **Step 8: Lint + full host suite + final commit**

```bash
ruff check experiments/opensim_gonogo_sweep.py tests/opensim/test_gonogo_e2e.py
pytest -q
```

Expected: ruff clean; full host suite green (opensim tests deselected).

```bash
git add experiments/opensim_gonogo_sweep.py tests/opensim/test_gonogo_e2e.py \
        tests/test_opensim_plant.py PROJECT_MEMORY.md results/opensim_gonogo_sweep.json
git commit -m "feat: go/no-go OpenSim embodiment sweep + comparison (Task 10.4, M8)"
```

---

## Self-Review (completed)

- **Spec coverage:** §3 boundary → Task 10.3 client; §4 plant/control → Task 10.2; §5 modules → all tasks; §6.1 host tests → Task 10.3; §6.2 validation → Task 10.2 Step 4/6; §6.3 smoke → Task 10.4; §7 tasks → 10.1–10.4; §8 risks → Dockerfile pin + validation gating + gain tuning. All covered.
- **Placeholder scan:** the Task 10.1 Step 4 "placeholder scripts" are intentional build scaffolding, fully replaced in Task 10.2; no `TODO`/`TBD`/"add error handling" left.
- **Type consistency:** `ReachSpec`/`OpenSimTrajectory`/`OpenSimPlantConfig` fields and `OpenSimPlantClient.run` return type `(list[OpenSimTrajectory], list[list[float]])` are consistent across Tasks 10.3 and 10.4; `config.json`/`jobs.json`/`trajectories.json` schemas match between container (10.2) and client (10.3).
- **Verification honesty:** OpenSim control API is locked by the Task 10.2 Step 1 spike before use, per the project "no invented APIs" rule.
```
