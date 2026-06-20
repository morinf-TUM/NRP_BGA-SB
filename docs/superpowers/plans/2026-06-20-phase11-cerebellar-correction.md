# Phase 11 — Cerebellar Trajectory Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a host-side cerebellar correction module (LMS trial-by-trial adaptation + forward-model within-trial online feedback) that reduces a visuomotor-rotation perturbation on the kinematic reacher, without altering the BG-frequency selection signature.

**Architecture:** Three new units sit strictly *downstream* of the existing BG → thalamus → motor-command pipeline. `VisuomotorRotation` distorts an executed reach by a fixed angle θ. `Cerebellum` composes an `AdaptiveFilter` (learns a feedforward counter-rotation across trials) and a `ForwardModelController` (corrects the trajectory within a trial toward the desired path). `KinematicReacher.simulate_with_correction` orchestrates them, and is invoked only when a movement is executed — so misses (e.g. 5 Hz) never reach the cerebellum and the selection signature is structurally preserved.

**Tech Stack:** Python 3.10, numpy ≥ 1.26, Pydantic v2, pytest ≥ 8.0, ruff ≥ 0.4. No new dependencies. No Docker.

## Global Constraints

- Python 3.10; numpy ≥ 1.26; Pydantic v2 (`model_dump_json` / `model_validate_json`, never v1 `.json()`/`.parse_raw()`).
- Use `X | None` union syntax (not `Optional[X]`); `Literal[...]` for fixed-vocabulary string fields.
- Fail fast: raise `ValueError` on invalid config; no silent fallbacks, no speculative `getattr`, no broad `except`.
- Section-header comments (`# --- Name ---`) in every multi-section module; decision-point comments (Trigger / Why / Outcome) on the no-movement guard branch and the LMS update.
- Comments explain *why*, not *what*.
- All angles are in **radians** internally; configs accept **degrees** and convert at construction.
- Geometry convention: visuomotor rotation and counter-rotation are rotations about the origin (the reach start point). Rotation preserves magnitude, so the perturbation produces a pure angular endpoint error.
- Baseline before Phase 11: 685 host tests passing, ruff clean. The full suite must stay green; ruff must stay clean.
- Tests use `tmp_path` for any file I/O. Determinism: seeds are the single source of truth (`random.Random(seed)` / numpy seeded locally).

---

### Task 1: Geometry helpers + `VisuomotorRotation`

**Files:**
- Create: `src/nrp_bga_sb/perturbation_plant.py`
- Test: `tests/test_perturbation_plant.py`

**Interfaces:**
- Consumes: numpy only.
- Produces:
  - `rotate_xy(vec: list[float] | np.ndarray, theta_rad: float) -> list[float]` — rotate a 2D vector about the origin by `theta_rad`.
  - `signed_angle(v_from: list[float] | np.ndarray, v_to: list[float] | np.ndarray) -> float` — signed angle (radians, range (−π, π]) rotating `v_from` onto `v_to` (positive = counter-clockwise). Returns `0.0` if either vector has near-zero norm.
  - `class VisuomotorRotation(BaseModel)` with field `rotation_deg: float` and method `apply(self, endpoint_xy: list[float]) -> list[float]` rotating the endpoint by `rotation_deg` (converted to radians). `rotation_deg=0.0` is the identity.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_perturbation_plant.py
import math

import numpy as np
import pytest

from nrp_bga_sb.perturbation_plant import VisuomotorRotation, rotate_xy, signed_angle


def test_rotate_xy_90_degrees():
    out = rotate_xy([1.0, 0.0], math.pi / 2)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] == pytest.approx(1.0, abs=1e-9)


def test_rotate_xy_zero_is_identity():
    assert rotate_xy([0.7, -0.3], 0.0) == pytest.approx([0.7, -0.3])


def test_signed_angle_positive_ccw():
    # from +x axis to +y axis is +90 degrees
    assert signed_angle([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.pi / 2)


def test_signed_angle_negative_cw():
    assert signed_angle([1.0, 0.0], [0.0, -1.0]) == pytest.approx(-math.pi / 2)


def test_signed_angle_zero_vector_returns_zero():
    assert signed_angle([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_visuomotor_rotation_applies_degrees():
    pert = VisuomotorRotation(rotation_deg=30.0)
    out = pert.apply([1.0, 0.0])
    assert out[0] == pytest.approx(math.cos(math.radians(30.0)))
    assert out[1] == pytest.approx(math.sin(math.radians(30.0)))


def test_visuomotor_rotation_zero_is_identity():
    pert = VisuomotorRotation(rotation_deg=0.0)
    assert pert.apply([-1.0, 0.0]) == pytest.approx([-1.0, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_perturbation_plant.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nrp_bga_sb.perturbation_plant'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/nrp_bga_sb/perturbation_plant.py
"""Motor perturbation plant + 2D geometry helpers (Phase 11, Task 11.1).

VisuomotorRotation injects a fixed angular distortion into an executed reach
endpoint — the canonical sensorimotor-adaptation perturbation the cerebellum
must learn to cancel. The two free functions (rotate_xy, signed_angle) are the
shared 2D geometry used by the cerebellar layers and the sweep metrics.
"""
from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

# --- Geometry helpers ---


def rotate_xy(vec: list[float] | np.ndarray, theta_rad: float) -> list[float]:
    """Rotate a 2D vector about the origin by theta_rad (CCW positive)."""
    v = np.asarray(vec, dtype=float)
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return [float(c * v[0] - s * v[1]), float(s * v[0] + c * v[1])]


def signed_angle(
    v_from: list[float] | np.ndarray, v_to: list[float] | np.ndarray
) -> float:
    """Signed angle (rad, CCW positive) rotating v_from onto v_to.

    Returns 0.0 if either vector is degenerate (near-zero norm): an undefined
    direction carries no angular error.
    """
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return 0.0
    # atan2 of the 2D cross and dot products gives a signed angle in (-pi, pi].
    cross = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    return float(math.atan2(cross, dot))


# --- VisuomotorRotation ---


class VisuomotorRotation(BaseModel):
    """Fixed-angle visuomotor rotation applied to an executed reach endpoint."""

    rotation_deg: float = 30.0

    def apply(self, endpoint_xy: list[float]) -> list[float]:
        """Rotate the endpoint about the origin by rotation_deg."""
        return rotate_xy(endpoint_xy, math.radians(self.rotation_deg))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_perturbation_plant.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/perturbation_plant.py tests/test_perturbation_plant.py
git commit -m "feat: VisuomotorRotation perturbation + 2D geometry helpers (Task 11.1)

ChangeSet-ID: phase11-perturbation-plant

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `AdaptiveFilter` (trial-by-trial LMS adaptation)

**Files:**
- Create: `src/nrp_bga_sb/cerebellum.py`
- Test: `tests/test_cerebellum.py`

**Interfaces:**
- Consumes: `rotate_xy` from `perturbation_plant` (Task 1).
- Produces:
  - `class AdaptiveFilter` constructed as `AdaptiveFilter(learning_rate: float = 0.1)`. Raises `ValueError` unless `0 < learning_rate <= 1`.
    - attribute `theta_hat: float` (radians), initialised `0.0`.
    - `precompensate(self, endpoint_xy: list[float]) -> list[float]` — rotate the desired endpoint by `−theta_hat` (the learned feedforward counter-rotation).
    - `update(self, angular_error_rad: float) -> None` — LMS step `theta_hat += learning_rate * angular_error_rad`.
    - `reset(self) -> None` — set `theta_hat = 0.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cerebellum.py
import math

import pytest

from nrp_bga_sb.cerebellum import AdaptiveFilter


def test_adaptive_filter_rejects_bad_learning_rate():
    with pytest.raises(ValueError):
        AdaptiveFilter(learning_rate=0.0)
    with pytest.raises(ValueError):
        AdaptiveFilter(learning_rate=1.5)


def test_adaptive_filter_starts_at_zero():
    assert AdaptiveFilter().theta_hat == 0.0


def test_adaptive_filter_converges_to_perturbation():
    # Simulate the closed loop: residual error each trial is (theta - theta_hat).
    theta = math.radians(30.0)
    af = AdaptiveFilter(learning_rate=0.2)
    errors = []
    for _ in range(100):
        residual = theta - af.theta_hat  # what the feedforward failed to cancel
        errors.append(abs(residual))
        af.update(residual)
    assert af.theta_hat == pytest.approx(theta, abs=1e-3)
    # error decays monotonically toward zero
    assert errors[-1] < errors[0]
    assert all(errors[i + 1] <= errors[i] + 1e-12 for i in range(len(errors) - 1))


def test_adaptive_filter_precompensate_counter_rotates():
    af = AdaptiveFilter()
    af.theta_hat = math.radians(30.0)
    out = af.precompensate([1.0, 0.0])  # rotate by -30 deg
    assert out[1] == pytest.approx(math.sin(math.radians(-30.0)))


def test_adaptive_filter_reset():
    af = AdaptiveFilter()
    af.update(0.5)
    assert af.theta_hat != 0.0
    af.reset()
    assert af.theta_hat == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cerebellum.py -v`
Expected: FAIL with `ImportError: cannot import name 'AdaptiveFilter'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/nrp_bga_sb/cerebellum.py
"""Cerebellar adaptive-filter model (Phase 11, Task 11.1).

Two separable, independently-ablatable layers:

  AdaptiveFilter         — trial-by-trial Widrow-Hoff/LMS learning of a
                           feedforward counter-rotation (Fujita 1982;
                           Dean, Porrill & Stone 2010).
  ForwardModelController — within-trial proportional feedback steering the
                           trajectory toward the desired path (Miall et al.
                           1993 internal-forward-model / Smith-predictor line).

Cerebellum composes both behind one interface, with independent enable flags.
"""
from __future__ import annotations

import math

import numpy as np

from nrp_bga_sb.perturbation_plant import rotate_xy

# --- AdaptiveFilter ---


class AdaptiveFilter:
    """Scalar LMS filter learning a feedforward counter-rotation across trials."""

    def __init__(self, learning_rate: float = 0.1) -> None:
        # Trigger: learning_rate outside (0, 1].
        # Why: <=0 never learns; >1 overshoots and can diverge the LMS update.
        # Outcome: fail fast so a mis-tuned experiment cannot silently produce noise.
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        self.learning_rate = learning_rate
        self.theta_hat: float = 0.0

    def precompensate(self, endpoint_xy: list[float]) -> list[float]:
        """Apply the learned feedforward counter-rotation (-theta_hat)."""
        return rotate_xy(endpoint_xy, -self.theta_hat)

    def update(self, angular_error_rad: float) -> None:
        """Widrow-Hoff/LMS step toward cancelling the observed angular error."""
        # The observed error equals the residual rotation (theta - theta_hat);
        # adding learning_rate * error drives theta_hat -> theta over trials.
        self.theta_hat += self.learning_rate * angular_error_rad

    def reset(self) -> None:
        """Clear learned state (between blocks / seeds)."""
        self.theta_hat = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cerebellum.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/cerebellum.py tests/test_cerebellum.py
git commit -m "feat: AdaptiveFilter LMS trial-by-trial cerebellar adaptation (Task 11.1)

ChangeSet-ID: phase11-adaptive-filter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `ForwardModelController` (within-trial online feedback)

**Files:**
- Modify: `src/nrp_bga_sb/cerebellum.py` (append after `AdaptiveFilter`)
- Test: `tests/test_cerebellum.py` (append)

**Interfaces:**
- Consumes: numpy.
- Produces:
  - `class ForwardModelController` constructed as `ForwardModelController(gain: float = 0.5)`. Raises `ValueError` unless `0 <= gain <= 1`.
    - `integrate(self, desired_xy: list[float], openloop_xy: list[float], s_values: list[float]) -> list[list[float]]` — step-wise integrator. `desired_xy` (D) is the endpoint toward the true target; `openloop_xy` (P) is the perturbed feedforward endpoint; `s_values` is the per-timestep normalized minimum-jerk progress (0→1, non-decreasing). Returns one `[x, y]` per `s`. `gain=0` reproduces the straight open-loop line to P; `gain>0` curves the path toward D, reducing endpoint error.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cerebellum.py (append)
import numpy as np

from nrp_bga_sb.cerebellum import ForwardModelController


def _min_jerk_s(n: int) -> list[float]:
    out = []
    for i in range(n):
        tau = i / (n - 1)
        out.append(10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    return out


def test_forward_model_rejects_bad_gain():
    with pytest.raises(ValueError):
        ForwardModelController(gain=-0.1)
    with pytest.raises(ValueError):
        ForwardModelController(gain=1.1)


def test_forward_model_gain_zero_reproduces_openloop():
    s = _min_jerk_s(50)
    D = [1.0, 0.0]
    P = [math.cos(math.radians(30)), math.sin(math.radians(30))]  # rotated endpoint
    fmc = ForwardModelController(gain=0.0)
    traj = fmc.integrate(D, P, s)
    # endpoint matches the open-loop perturbed endpoint P
    assert traj[-1] == pytest.approx(P, abs=1e-6)


def test_forward_model_gain_reduces_endpoint_error():
    s = _min_jerk_s(200)
    D = np.array([1.0, 0.0])
    P = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
    err_open = float(np.linalg.norm(P - D))
    fmc = ForwardModelController(gain=0.6)
    traj = fmc.integrate(list(D), list(P), s)
    err_corrected = float(np.linalg.norm(np.array(traj[-1]) - D))
    assert err_corrected < err_open


def test_forward_model_output_length_matches_s():
    s = _min_jerk_s(37)
    traj = ForwardModelController(gain=0.5).integrate([1.0, 0.0], [0.0, 1.0], s)
    assert len(traj) == 37
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cerebellum.py -k forward_model -v`
Expected: FAIL with `ImportError: cannot import name 'ForwardModelController'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/nrp_bga_sb/cerebellum.py (append after AdaptiveFilter)

# --- ForwardModelController ---


class ForwardModelController:
    """Within-trial proportional feedback toward the desired (intended) path."""

    def __init__(self, gain: float = 0.5) -> None:
        # Trigger: gain outside [0, 1].
        # Why: gain<0 pushes away from target; gain>1 over-corrects and can ring.
        # Outcome: fail fast on a mis-specified controller.
        if not 0.0 <= gain <= 1.0:
            raise ValueError(f"gain must be in [0, 1], got {gain}")
        self.gain = gain

    def integrate(
        self,
        desired_xy: list[float],
        openloop_xy: list[float],
        s_values: list[float],
    ) -> list[list[float]]:
        """Integrate the corrected trajectory step by step.

        At each step the open-loop perturbed motion increment (toward P) is
        applied, then a proportional feedback term pulls the running position
        toward the reference D*s (where the hand should be by now). gain=0 leaves
        the open-loop straight line to P; gain in (0,1) curves the path toward D.
        The (1-gain) contraction on the running position keeps the loop stable.
        """
        D = np.asarray(desired_xy, dtype=float)
        P = np.asarray(openloop_xy, dtype=float)
        pos = np.zeros(2, dtype=float)
        prev_s = 0.0
        out: list[list[float]] = []
        for s in s_values:
            ds = s - prev_s
            openloop_increment = P * ds          # perturbed feedforward motion this step
            ref = D * s                          # desired position by this point
            feedback = self.gain * (ref - pos)   # proportional pull toward desired path
            pos = pos + openloop_increment + feedback
            out.append([float(pos[0]), float(pos[1])])
            prev_s = s
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cerebellum.py -v`
Expected: PASS (9 passed total)

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/cerebellum.py tests/test_cerebellum.py
git commit -m "feat: ForwardModelController within-trial online feedback (Task 11.1)

ChangeSet-ID: phase11-forward-model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `Cerebellum` composition

**Files:**
- Modify: `src/nrp_bga_sb/cerebellum.py` (append after `ForwardModelController`)
- Test: `tests/test_cerebellum.py` (append)

**Interfaces:**
- Consumes: `AdaptiveFilter`, `ForwardModelController` (this module).
- Produces:
  - `class Cerebellum` constructed as
    `Cerebellum(learning_rate: float = 0.1, online_gain: float = 0.5, adaptation_enabled: bool = True, online_enabled: bool = True)`.
    - attribute `adaptive_filter: AdaptiveFilter`.
    - `precompensate(self, desired_xy: list[float]) -> list[float]` — counter-rotate by the filter when `adaptation_enabled`, else identity (returns a copy).
    - `integrate(self, desired_xy, openloop_xy, s_values) -> list[list[float]]` — `ForwardModelController.integrate` when `online_enabled`, else the straight open-loop line to `openloop_xy` (`gain=0` integration).
    - `learn(self, angular_error_rad: float) -> None` — `adaptive_filter.update` when `adaptation_enabled`, else no-op.
    - `reset(self) -> None` — reset the filter.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cerebellum.py (append)
from nrp_bga_sb.cerebellum import Cerebellum


def test_cerebellum_precompensate_identity_when_adaptation_off():
    cb = Cerebellum(adaptation_enabled=False)
    cb.adaptive_filter.theta_hat = math.radians(30.0)  # would rotate if used
    assert cb.precompensate([1.0, 0.0]) == pytest.approx([1.0, 0.0])


def test_cerebellum_learn_noop_when_adaptation_off():
    cb = Cerebellum(adaptation_enabled=False)
    cb.learn(0.5)
    assert cb.adaptive_filter.theta_hat == 0.0


def test_cerebellum_learn_updates_when_adaptation_on():
    cb = Cerebellum(adaptation_enabled=True, learning_rate=0.2)
    cb.learn(1.0)
    assert cb.adaptive_filter.theta_hat == pytest.approx(0.2)


def test_cerebellum_integrate_straight_line_when_online_off():
    s = _min_jerk_s(50)
    P = [0.5, 0.5]
    cb = Cerebellum(online_enabled=False)
    traj = cb.integrate([1.0, 0.0], P, s)
    assert traj[-1] == pytest.approx(P, abs=1e-6)  # ends at open-loop endpoint


def test_cerebellum_reset_clears_filter():
    cb = Cerebellum()
    cb.learn(0.5)
    cb.reset()
    assert cb.adaptive_filter.theta_hat == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cerebellum.py -k cerebellum -v`
Expected: FAIL with `ImportError: cannot import name 'Cerebellum'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/nrp_bga_sb/cerebellum.py (append after ForwardModelController)

# --- Cerebellum (composition) ---


class Cerebellum:
    """Composes the adaptation and online-feedback layers behind one interface.

    The two layers are independently toggleable so each can be ablated and the
    cerebellum-on/off sweep is a single code path.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        online_gain: float = 0.5,
        adaptation_enabled: bool = True,
        online_enabled: bool = True,
    ) -> None:
        self.adaptive_filter = AdaptiveFilter(learning_rate=learning_rate)
        self._controller = ForwardModelController(gain=online_gain)
        self._straight = ForwardModelController(gain=0.0)
        self.adaptation_enabled = adaptation_enabled
        self.online_enabled = online_enabled

    def precompensate(self, desired_xy: list[float]) -> list[float]:
        if self.adaptation_enabled:
            return self.adaptive_filter.precompensate(desired_xy)
        return list(desired_xy)

    def integrate(
        self,
        desired_xy: list[float],
        openloop_xy: list[float],
        s_values: list[float],
    ) -> list[list[float]]:
        controller = self._controller if self.online_enabled else self._straight
        return controller.integrate(desired_xy, openloop_xy, s_values)

    def learn(self, angular_error_rad: float) -> None:
        if self.adaptation_enabled:
            self.adaptive_filter.update(angular_error_rad)

    def reset(self) -> None:
        self.adaptive_filter.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cerebellum.py -v`
Expected: PASS (14 passed total)

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/cerebellum.py tests/test_cerebellum.py
git commit -m "feat: Cerebellum composition with ablatable layers (Task 11.1)

ChangeSet-ID: phase11-cerebellum

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `KinematicReacher.simulate_with_correction`

**Files:**
- Modify: `src/nrp_bga_sb/reacher.py` (add imports at top; append method to `KinematicReacher`)
- Test: `tests/test_reacher_correction.py`

**Interfaces:**
- Consumes: `MotorCommand` (schemas), `VisuomotorRotation` + `rotate_xy` + `signed_angle` (perturbation_plant), `Cerebellum` (cerebellum), `_minimum_jerk_scalar` (this module).
- Produces:
  - `KinematicReacher.simulate_with_correction(self, motor_commands: list[MotorCommand], onset_time_ms: float | None, total_duration_ms: float = 1300.0, perturbation: VisuomotorRotation | None = None, cerebellum: Cerebellum | None = None) -> ReacherTrajectory`.
    - **Guard:** if `motor_commands` is empty or the last command's `gate_state == "closed"`, return a zero-movement `ReacherTrajectory` (`onset_time_ms=None`, `selected_channel=-1`) and **do not touch** `cerebellum`.
    - For an executed movement: desired endpoint `D = target_positions[channel] * gate_gain`; feedforward `C = cerebellum.precompensate(D)` (or `D`); open-loop perturbed `P = perturbation.apply(C)` (or `C`); integrate the trajectory via `cerebellum.integrate(D, P, s_values)` (or a straight min-jerk line to `P`); then `cerebellum.learn(signed_angle(D, P))`.
    - Reuses the wiring-error guard from `simulate` (non-closed gate + all-zero command → `ValueError`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reacher_correction.py
import math

import numpy as np
import pytest

from nrp_bga_sb.cerebellum import Cerebellum
from nrp_bga_sb.perturbation_plant import VisuomotorRotation
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.schemas import MotorCommand


def _open_cmd(channel: int, gain: float = 1.0, n: int = 2) -> MotorCommand:
    command = [0.0] * n
    command[channel] = gain
    return MotorCommand(
        command=command, gate_state="open", gate_gain=gain, sim_time=0.7
    )


def _closed_cmd(n: int = 2) -> MotorCommand:
    return MotorCommand(
        command=[0.0] * n, gate_state="closed", gate_gain=0.0, sim_time=0.7
    )


def test_correction_no_movement_leaves_cerebellum_untouched():
    reacher = KinematicReacher()
    cb = Cerebellum()
    cb.adaptive_filter.theta_hat = 0.5
    traj = reacher.simulate_with_correction(
        [_closed_cmd()], onset_time_ms=None,
        perturbation=VisuomotorRotation(rotation_deg=30.0), cerebellum=cb,
    )
    # guard: closed gate -> no movement, cerebellum never learns
    assert traj.selected_channel == -1
    assert traj.onset_time_ms is None
    assert cb.adaptive_filter.theta_hat == 0.5


def test_correction_perturbation_only_rotates_endpoint():
    reacher = KinematicReacher()
    traj = reacher.simulate_with_correction(
        [_open_cmd(1, 1.0)], onset_time_ms=0.0, total_duration_ms=500.0,
        perturbation=VisuomotorRotation(rotation_deg=30.0), cerebellum=None,
    )
    final = traj.positions_xy[-1]
    # target for channel 1 is [1, 0]; rotated by 30 deg
    assert final[0] == pytest.approx(math.cos(math.radians(30.0)), abs=1e-3)
    assert final[1] == pytest.approx(math.sin(math.radians(30.0)), abs=1e-3)


def test_correction_online_reduces_endpoint_error():
    reacher = KinematicReacher()
    pert = VisuomotorRotation(rotation_deg=30.0)
    target = np.array([1.0, 0.0])

    uncorrected = reacher.simulate_with_correction(
        [_open_cmd(1, 1.0)], 0.0, 500.0, perturbation=pert, cerebellum=None
    )
    err_unc = float(np.linalg.norm(np.array(uncorrected.positions_xy[-1]) - target))

    cb = Cerebellum(adaptation_enabled=False, online_enabled=True, online_gain=0.6)
    corrected = reacher.simulate_with_correction(
        [_open_cmd(1, 1.0)], 0.0, 500.0, perturbation=pert, cerebellum=cb
    )
    err_cor = float(np.linalg.norm(np.array(corrected.positions_xy[-1]) - target))
    assert err_cor < err_unc


def test_correction_no_perturbation_no_cerebellum_matches_simulate():
    reacher = KinematicReacher()
    cmds = [_open_cmd(0, 0.8)]
    a = reacher.simulate(cmds, 0.0, 500.0)
    b = reacher.simulate_with_correction(cmds, 0.0, 500.0, perturbation=None, cerebellum=None)
    assert b.positions_xy[-1] == pytest.approx(a.positions_xy[-1], abs=1e-6)
    assert b.selected_channel == a.selected_channel


def test_correction_adaptation_learns_toward_perturbation():
    reacher = KinematicReacher()
    pert = VisuomotorRotation(rotation_deg=30.0)
    cb = Cerebellum(adaptation_enabled=True, online_enabled=False, learning_rate=0.3)
    for _ in range(50):
        reacher.simulate_with_correction(
            [_open_cmd(1, 1.0)], 0.0, 500.0, perturbation=pert, cerebellum=cb
        )
    assert cb.adaptive_filter.theta_hat == pytest.approx(math.radians(30.0), abs=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reacher_correction.py -v`
Expected: FAIL with `AttributeError: 'KinematicReacher' object has no attribute 'simulate_with_correction'`

- [ ] **Step 3: Write minimal implementation**

First add imports near the top of `src/nrp_bga_sb/reacher.py` (after the existing `from nrp_bga_sb.schemas import MotorCommand` line):

```python
from nrp_bga_sb.cerebellum import Cerebellum
from nrp_bga_sb.perturbation_plant import VisuomotorRotation, signed_angle
```

Then append this method to the `KinematicReacher` class (after `simulate_change_of_mind`):

```python
    def simulate_with_correction(
        self,
        motor_commands: list[MotorCommand],
        onset_time_ms: float | None,
        total_duration_ms: float = 1300.0,
        perturbation: VisuomotorRotation | None = None,
        cerebellum: Cerebellum | None = None,
    ) -> ReacherTrajectory:
        """Simulate a reach under an optional perturbation + cerebellar correction.

        The cerebellum sits strictly downstream: it is invoked ONLY when a
        movement is executed. Misses (closed gate / no command) return a
        zero-movement trajectory and never reach the cerebellum, so the
        BG-frequency selection signature is preserved (Phase 11 guard).
        """
        n_steps = int(round(total_duration_ms / self.config.dt_ms)) + 1
        times_ms = [i * self.config.dt_ms for i in range(n_steps)]
        zero_positions = [[0.0, 0.0]] * n_steps

        # --- No-movement guard ---
        # Trigger: empty command series or a closed final gate.
        # Why: a downstream corrector must never manufacture a reach the BG/thalamus
        #      did not release; this is what keeps onset-rate-vs-frequency invariant
        #      to the cerebellum.
        # Outcome: return a zero trajectory and leave `cerebellum` state untouched.
        if not motor_commands or motor_commands[-1].gate_state == "closed":
            return ReacherTrajectory(
                times_ms=times_ms,
                positions_xy=zero_positions,
                onset_time_ms=None,
                selected_channel=-1,
            )

        last_cmd = motor_commands[-1]
        if len(last_cmd.command) != self.config.n_channels:
            raise ValueError(
                f"Command has {len(last_cmd.command)} channels but "
                f"config expects {self.config.n_channels}"
            )
        selected_channel = int(np.argmax(last_cmd.command))
        if last_cmd.command[selected_channel] == 0.0:
            raise ValueError(
                f"gate_state={last_cmd.gate_state!r} but command is all-zero: "
                f"{last_cmd.command}"
            )

        tx, ty = self.config.target_positions[selected_channel]
        # Desired (achievable) endpoint: target scaled by the gate gain.
        desired = [tx * last_cmd.gate_gain, ty * last_cmd.gate_gain]

        # Feedforward pre-compensation (learned counter-rotation), then the plant
        # perturbation. With no cerebellum/perturbation these are identities.
        commanded = cerebellum.precompensate(desired) if cerebellum else list(desired)
        openloop = perturbation.apply(commanded) if perturbation else list(commanded)

        actual_onset = onset_time_ms if onset_time_ms is not None else 0.0
        T = self.config.movement_duration_ms

        # Build the per-step minimum-jerk progress s(t) over the movement window;
        # steps before onset hold at the origin (progress 0).
        s_values = [
            _minimum_jerk_scalar(t_ms - actual_onset, T) if t_ms >= actual_onset else 0.0
            for t_ms in times_ms
        ]

        if cerebellum is not None:
            positions_xy = cerebellum.integrate(desired, openloop, s_values)
            # Learning is driven by the open-loop (feedforward) angular error so it
            # is not masked by within-trial online correction.
            cerebellum.learn(signed_angle(desired, openloop))
        else:
            # No cerebellum: straight minimum-jerk line to the (possibly perturbed) endpoint.
            ex, ey = openloop
            positions_xy = [[ex * s, ey * s] for s in s_values]

        return ReacherTrajectory(
            times_ms=times_ms,
            positions_xy=positions_xy,
            onset_time_ms=actual_onset,
            selected_channel=selected_channel,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reacher_correction.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/reacher.py tests/test_reacher_correction.py
git commit -m "feat: KinematicReacher.simulate_with_correction + BG-effect guard (Task 11.2)

ChangeSet-ID: phase11-simulate-with-correction

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Cerebellum sweep library

**Files:**
- Create: `src/nrp_bga_sb/cerebellum_sweep.py`
- Test: `tests/test_cerebellum_sweep.py`

**Interfaces:**
- Consumes: `make_closed_loop_policy` (closed_loop), `CortexConfig` (cortex), `FrequencyConfig` (scheduler), `GoNoGoConfig` + `run_go_nogo_trials` (engines.go_nogo), `KinematicReacher` + `ReacherConfig` (reacher), `Cerebellum` (cerebellum), `VisuomotorRotation` + `signed_angle` (perturbation_plant), `CONFLICT_PEAK_SALIENCE` (sweep).
- Produces:
  - `FREQUENCIES_HZ: list[float] = [5.0, 10.0, 20.0, 40.0, 80.0]`
  - `class CerebellumSweepResult(BaseModel)` with fields: `frequency_hz: float`, `seed: int`, `n_trials: int`, `cerebellum_enabled: bool`, `perturbation_deg: float`, `movement_onset_rate: float`, `go_success_rate: float | None`, `mean_endpoint_deviation: float`, `mean_angular_error_rad: float`, `final_theta_hat: float`, `endpoint_deviation_by_trial: list[float]`.
  - `run_cerebellum_condition(frequency_hz: float, n_trials: int = 30, seed: int = 42, perturbation_deg: float = 30.0, cerebellum_enabled: bool = True, learning_rate: float = 0.1, online_gain: float = 0.5, accumulation_ms: float = 200.0, rise_time_ms: float = 200.0) -> CerebellumSweepResult`.
    - Builds the closed-loop go/no-go policy exactly as `reacher_sweep.run_reacher_condition` does (low conflict, `peak_salience=CONFLICT_PEAK_SALIENCE["low"]`, go/no-go engine config matching `reacher_sweep._run_engine`).
    - One fresh `Cerebellum` per condition (θ̂ resets per condition). When `cerebellum_enabled=False`, passes `cerebellum=None` to the reacher.
    - `movement_onset_rate` denominator = go trials only (matches `reacher_sweep`).
    - Accuracy is measured vs the **desired** endpoint `D = target_positions[channel] * gate_gain` (isolates the rotation residual from the gate-gain shortfall).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cerebellum_sweep.py
import pytest

from nrp_bga_sb.cerebellum_sweep import (
    FREQUENCIES_HZ,
    CerebellumSweepResult,
    run_cerebellum_condition,
)


def test_frequencies_match_prior_sweeps():
    assert FREQUENCIES_HZ == [5.0, 10.0, 20.0, 40.0, 80.0]


def test_condition_returns_result_type():
    r = run_cerebellum_condition(40.0, n_trials=20, seed=1)
    assert isinstance(r, CerebellumSweepResult)
    assert r.frequency_hz == 40.0
    assert r.cerebellum_enabled is True


def test_onset_rate_identical_cerebellum_on_vs_off():
    # The guard: the cerebellum must NOT change which trials produce a movement.
    for freq in (5.0, 10.0, 40.0):
        on = run_cerebellum_condition(freq, n_trials=30, seed=7, cerebellum_enabled=True)
        off = run_cerebellum_condition(freq, n_trials=30, seed=7, cerebellum_enabled=False)
        assert on.movement_onset_rate == off.movement_onset_rate
        assert on.go_success_rate == off.go_success_rate


def test_low_frequency_has_no_movement():
    r = run_cerebellum_condition(5.0, n_trials=30, seed=7)
    assert r.movement_onset_rate == 0.0


def test_cerebellum_reduces_endpoint_deviation_when_moving():
    on = run_cerebellum_condition(40.0, n_trials=40, seed=3, cerebellum_enabled=True)
    off = run_cerebellum_condition(40.0, n_trials=40, seed=3, cerebellum_enabled=False)
    assert on.movement_onset_rate > 0.0  # sanity: trials move at 40 Hz
    assert on.mean_endpoint_deviation < off.mean_endpoint_deviation


def test_adaptation_learns_nonzero_theta_when_moving():
    r = run_cerebellum_condition(40.0, n_trials=40, seed=3, perturbation_deg=30.0)
    assert r.final_theta_hat > 0.1  # learned a counter-rotation


def test_endpoint_deviation_decays_over_trials():
    r = run_cerebellum_condition(40.0, n_trials=40, seed=3, perturbation_deg=30.0)
    series = r.endpoint_deviation_by_trial
    assert len(series) >= 10
    # later trials are more accurate than the first few (learning curve)
    assert sum(series[-3:]) / 3 < sum(series[:3]) / 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cerebellum_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nrp_bga_sb.cerebellum_sweep'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/nrp_bga_sb/cerebellum_sweep.py
"""Cerebellum on/off frequency-sweep library (Phase 11, Tasks 11.2 + 11.3).

Runs the Phase 6 go/no-go kinematic pipeline under a visuomotor-rotation
perturbation, with the cerebellum either engaged or absent, on the SAME BG
decisions. Reports the BG-selection guard metrics (onset / success rate) and
the accuracy metrics (endpoint deviation, angular error, learned theta_hat,
per-trial learning curve).
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.cerebellum import Cerebellum
from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.perturbation_plant import VisuomotorRotation, signed_angle
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.sweep import CONFLICT_PEAK_SALIENCE

FREQUENCIES_HZ: list[float] = [5.0, 10.0, 20.0, 40.0, 80.0]

# Match reacher_sweep timing so results are comparable across phases.
_RISE_TIME_MS: float = 200.0
_ACCUMULATION_MS: float = 200.0
_TOTAL_DURATION_MS: float = 1300.0


# --- CerebellumSweepResult ---


class CerebellumSweepResult(BaseModel):
    """Guard + accuracy metrics for one cerebellum sweep condition."""

    frequency_hz: float
    seed: int
    n_trials: int
    cerebellum_enabled: bool
    perturbation_deg: float
    # BG-selection guard metrics
    movement_onset_rate: float
    go_success_rate: float | None
    # Accuracy metrics (over movement trials; vs the desired gate-scaled endpoint)
    mean_endpoint_deviation: float
    mean_angular_error_rad: float
    final_theta_hat: float
    endpoint_deviation_by_trial: list[float]


# --- Public condition runner ---


def run_cerebellum_condition(
    frequency_hz: float,
    n_trials: int = 30,
    seed: int = 42,
    perturbation_deg: float = 30.0,
    cerebellum_enabled: bool = True,
    learning_rate: float = 0.1,
    online_gain: float = 0.5,
    accumulation_ms: float = _ACCUMULATION_MS,
    rise_time_ms: float = _RISE_TIME_MS,
) -> CerebellumSweepResult:
    """Run one go/no-go condition through the reacher under perturbation."""
    # --- Closed-loop policy (matches reacher_sweep, low conflict) ---
    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)
    cortex_cfg = CortexConfig(
        rise_time_ms=rise_time_ms,
        peak_salience=CONFLICT_PEAK_SALIENCE["low"],
        noise_std=0.0,
    )
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=accumulation_ms,
    )

    engine_cfg = GoNoGoConfig(
        n_trials=n_trials,
        go_probability=0.7,
        response_window_start_ms=0,
        response_window_duration_ms=600,
        fixation_duration_ms=200,
        cue_onset_ms=400,
        decision_point_ms=300,
        seed=seed,
    )
    trials = run_go_nogo_trials(engine_cfg, policy)

    # --- Reacher + perturbation + (optional) cerebellum ---
    reacher_cfg = ReacherConfig()
    reacher = KinematicReacher(reacher_cfg)
    perturbation = VisuomotorRotation(rotation_deg=perturbation_deg)
    cerebellum = (
        Cerebellum(learning_rate=learning_rate, online_gain=online_gain)
        if cerebellum_enabled
        else None
    )

    go_trials = [t for t in trials if t.cue_identity == "go"]
    n_go = len(go_trials)
    go_success_rate = (
        sum(1 for t in go_trials if t.success is True) / n_go if n_go else None
    )

    deviations: list[float] = []
    angular_errors: list[float] = []
    n_move = 0
    for trial in go_trials:
        if not trial.motor_command_series:
            continue
        onset_ms = (
            trial.movement_onset_time * 1000.0
            if trial.movement_onset_time is not None
            else None
        )
        traj = reacher.simulate_with_correction(
            trial.motor_command_series,
            onset_ms,
            _TOTAL_DURATION_MS,
            perturbation=perturbation,
            cerebellum=cerebellum,
        )
        if traj.onset_time_ms is None or traj.selected_channel < 0:
            continue
        n_move += 1
        # Desired (achievable) endpoint = target scaled by the final gate gain.
        last_cmd = trial.motor_command_series[-1]
        tx, ty = reacher_cfg.target_positions[traj.selected_channel]
        desired = [tx * last_cmd.gate_gain, ty * last_cmd.gate_gain]
        final = traj.positions_xy[-1]
        deviations.append(float(np.linalg.norm(np.array(final) - np.array(desired))))
        angular_errors.append(abs(signed_angle(desired, final)))

    movement_onset_rate = n_move / n_go if n_go else 0.0
    mean_dev = float(np.mean(deviations)) if deviations else 0.0
    mean_ang = float(np.mean(angular_errors)) if angular_errors else 0.0
    final_theta = cerebellum.adaptive_filter.theta_hat if cerebellum else 0.0

    return CerebellumSweepResult(
        frequency_hz=frequency_hz,
        seed=seed,
        n_trials=n_trials,
        cerebellum_enabled=cerebellum_enabled,
        perturbation_deg=perturbation_deg,
        movement_onset_rate=movement_onset_rate,
        go_success_rate=go_success_rate,
        mean_endpoint_deviation=mean_dev,
        mean_angular_error_rad=mean_ang,
        final_theta_hat=final_theta,
        endpoint_deviation_by_trial=deviations,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cerebellum_sweep.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/nrp_bga_sb/cerebellum_sweep.py tests/test_cerebellum_sweep.py
git commit -m "feat: cerebellum on/off frequency-sweep library + guard test (Tasks 11.2, 11.3)

ChangeSet-ID: phase11-cerebellum-sweep

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Experiment runner + M9 acceptance evidence

**Files:**
- Create: `experiments/cerebellum_adaptation.py`
- Test: `tests/test_cerebellum_experiment.py`

**Interfaces:**
- Consumes: `FREQUENCIES_HZ`, `run_cerebellum_condition`, `CerebellumSweepResult` (cerebellum_sweep).
- Produces:
  - `run_sweep(n_trials: int = 30, seeds: list[int] | None = None, perturbation_deg: float = 30.0) -> list[CerebellumSweepResult]` — runs every frequency × {cerebellum off, on} × seeds.
  - `save_results(results: list[CerebellumSweepResult], path: str) -> None` — writes a JSON array of `model_dump()`.
  - `format_report(results: list[CerebellumSweepResult]) -> str` — per-frequency table comparing on vs off (onset rate, mean endpoint deviation, mean angular error, final θ̂).
  - `main() -> None` — runs the sweep, writes `results/cerebellum_results.json`, prints the report. Guarded by `if __name__ == "__main__":`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cerebellum_experiment.py
import json

from experiments.cerebellum_adaptation import format_report, run_sweep, save_results


def test_run_sweep_covers_on_and_off():
    results = run_sweep(n_trials=20, seeds=[1])
    # 5 frequencies x 2 (on/off) x 1 seed
    assert len(results) == 10
    assert any(r.cerebellum_enabled for r in results)
    assert any(not r.cerebellum_enabled for r in results)


def test_m9_acceptance_onset_invariant_and_accuracy_improves():
    results = run_sweep(n_trials=30, seeds=[7])
    by_key = {(r.frequency_hz, r.cerebellum_enabled): r for r in results}
    for freq in (5.0, 10.0, 20.0, 40.0, 80.0):
        on = by_key[(freq, True)]
        off = by_key[(freq, False)]
        # (b) BG-frequency selection signature unchanged by the cerebellum
        assert on.movement_onset_rate == off.movement_onset_rate
        # (a) accuracy improves wherever movement actually occurs
        if on.movement_onset_rate > 0.0:
            assert on.mean_endpoint_deviation < off.mean_endpoint_deviation


def test_save_results_round_trip(tmp_path):
    results = run_sweep(n_trials=20, seeds=[1])
    out = tmp_path / "cb.json"
    save_results(results, str(out))
    loaded = json.loads(out.read_text())
    assert len(loaded) == len(results)
    assert "movement_onset_rate" in loaded[0]


def test_format_report_is_nonempty_string():
    results = run_sweep(n_trials=20, seeds=[1])
    report = format_report(results)
    assert isinstance(report, str)
    assert "5.0" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cerebellum_experiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.cerebellum_adaptation'`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/cerebellum_adaptation.py
"""Phase 11 cerebellar correction experiment (M9 acceptance evidence).

Runs the visuomotor-rotation go/no-go pipeline at five BG frequencies with the
cerebellum off vs on, on the SAME BG decisions. Demonstrates:
  (a) accuracy improves under perturbation (within-trial + across-trial), and
  (b) the BG-frequency onset signature is unchanged by the cerebellum.
"""
from __future__ import annotations

import json

from nrp_bga_sb.cerebellum_sweep import (
    FREQUENCIES_HZ,
    CerebellumSweepResult,
    run_cerebellum_condition,
)

_DEFAULT_SEEDS = [11, 22, 33, 44, 55]


def run_sweep(
    n_trials: int = 30,
    seeds: list[int] | None = None,
    perturbation_deg: float = 30.0,
) -> list[CerebellumSweepResult]:
    """Run every frequency x {cerebellum off, on} x seed condition."""
    seeds = seeds if seeds is not None else _DEFAULT_SEEDS
    results: list[CerebellumSweepResult] = []
    for freq in FREQUENCIES_HZ:
        for enabled in (False, True):
            for seed in seeds:
                results.append(
                    run_cerebellum_condition(
                        freq,
                        n_trials=n_trials,
                        seed=seed,
                        perturbation_deg=perturbation_deg,
                        cerebellum_enabled=enabled,
                    )
                )
    return results


def save_results(results: list[CerebellumSweepResult], path: str) -> None:
    """Write results as a JSON array of model dumps."""
    payload = [r.model_dump() for r in results]
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def format_report(results: list[CerebellumSweepResult]) -> str:
    """Per-frequency on-vs-off comparison table (averaged over seeds)."""
    lines = [
        "Phase 11 — Cerebellar correction (M9)",
        "freq(Hz) | onset off | onset on | endpt.dev off | endpt.dev on | ang.err on | theta_hat on",
        "-" * 92,
    ]
    for freq in FREQUENCIES_HZ:
        on = [r for r in results if r.frequency_hz == freq and r.cerebellum_enabled]
        off = [r for r in results if r.frequency_hz == freq and not r.cerebellum_enabled]

        def avg(rs: list[CerebellumSweepResult], attr: str) -> float:
            return sum(getattr(r, attr) for r in rs) / len(rs) if rs else 0.0

        lines.append(
            f"{freq:>7.1f} | "
            f"{avg(off, 'movement_onset_rate'):>9.3f} | "
            f"{avg(on, 'movement_onset_rate'):>8.3f} | "
            f"{avg(off, 'mean_endpoint_deviation'):>13.4f} | "
            f"{avg(on, 'mean_endpoint_deviation'):>12.4f} | "
            f"{avg(on, 'mean_angular_error_rad'):>10.4f} | "
            f"{avg(on, 'final_theta_hat'):>12.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    results = run_sweep()
    save_results(results, "results/cerebellum_results.json")
    print(format_report(results))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cerebellum_experiment.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add experiments/cerebellum_adaptation.py tests/test_cerebellum_experiment.py
git commit -m "feat: Phase 11 cerebellum experiment runner + M9 acceptance test (Task 11.3)

ChangeSet-ID: phase11-experiment

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Phase 11 verification, smoke run, and PROJECT_MEMORY update

**Files:**
- Modify: `PROJECT_MEMORY.md` (add §27 Phase 11 module map; append Phase 11 bullet to §1)
- Run: full test suite + ruff + the experiment script once.

**Interfaces:**
- Consumes: everything above. Produces: no new code, only verification + memory.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass — baseline was 685; expect ≈ 685 + (6+9+5+7+4) new ≈ 716 passed (exact count may differ slightly; the requirement is zero failures and the new files all green). The Docker-gated `@pytest.mark.opensim` tests remain deselected without an image.

- [ ] **Step 2: Run ruff**

Run: `ruff check src/nrp_bga_sb/perturbation_plant.py src/nrp_bga_sb/cerebellum.py src/nrp_bga_sb/cerebellum_sweep.py src/nrp_bga_sb/reacher.py experiments/cerebellum_adaptation.py tests/test_perturbation_plant.py tests/test_cerebellum.py tests/test_reacher_correction.py tests/test_cerebellum_sweep.py tests/test_cerebellum_experiment.py`
Expected: `All checks passed!`

- [ ] **Step 3: Smoke-run the experiment**

Run: `mkdir -p results && python -m experiments.cerebellum_adaptation`
Expected: prints the per-frequency table; `onset off` and `onset on` columns are identical at every frequency (0.000 at 5.0 Hz, 1.000 at ≥10 Hz); `endpt.dev on` < `endpt.dev off` at frequencies that move; `theta_hat on` is near 0.52 rad (≈30°) at ≥10 Hz. Writes `results/cerebellum_results.json`.

- [ ] **Step 4: Update PROJECT_MEMORY.md**

Append a Phase 11 completion bullet to §1 (after the Phase 10 bullet), e.g.:

```markdown
- **Phase 11 complete (2026-06-20).** Cerebellar trajectory correction (M9). `perturbation_plant.py` (`VisuomotorRotation` + 2D geometry), `cerebellum.py` (`AdaptiveFilter` LMS adaptation, `ForwardModelController` within-trial online feedback, `Cerebellum` composition), `KinematicReacher.simulate_with_correction` (invoked only on executed movements — the BG-effect guard), and `cerebellum_sweep.py` (cerebellum on/off frequency sweep). Key result: under a 30° visuomotor rotation the cerebellum reduces mean endpoint deviation at every frequency that moves (within-trial via online feedback, across-trial via adaptation: θ̂ → ~0.52 rad), while movement-onset-rate-vs-frequency is bit-identical with the cerebellum on or off (0.0 at 5 Hz, 1.0 at ≥10 Hz) — the BG-frequency selection signature survives. <N> tests passing, ruff clean. See §27 for module map. Embodied confirmation deferred to Phase 11b (IMPLEMENTATION_PLAN.md).
```

Add a new `## 27. Phase 11 module map` section documenting: the source layout (the four new modules + the reacher method), the cerebellar-model choice and literature anchors (cite the design spec), the geometry convention (rotation about origin → pure angular error; accuracy measured vs the gate-scaled desired endpoint), the BG-effect guard mechanism, and the M9 acceptance result table. Follow the structure of §26.

- [ ] **Step 5: Commit**

```bash
git add PROJECT_MEMORY.md
git commit -m "docs: PROJECT_MEMORY Phase 11 cerebellar correction module map (M9 complete)

ChangeSet-ID: phase11-project-memory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** §1 goal/acceptance → Tasks 6–8 (sweep + experiment + acceptance test). §2 plant=kinematic reacher → Task 5. §2 both correction layers → Tasks 2 (adaptation) + 3 (online). §2 visuomotor rotation → Task 1. §2 adaptive-filter model → Tasks 2–4. §3 module layout → all tasks (exact files). §4 components → Tasks 1–4. §5 data flow → Task 5 (`simulate_with_correction`). §6 BG-effect guard → Task 5 (guard branch) + Task 6 (`test_onset_rate_identical_*`) + Task 7 (M9 test). §7 error handling → Tasks 1–3 (`ValueError` validations) + Task 5 (wiring guard). §8 testing → every task's tests + Task 8 verification. §10 literature → Task 8 memory update.
- **Placeholder scan:** none — every code/test step contains complete content.
- **Type consistency:** `signed_angle`, `rotate_xy`, `VisuomotorRotation.apply`, `AdaptiveFilter.{precompensate,update,reset,theta_hat}`, `ForwardModelController.integrate(desired_xy, openloop_xy, s_values)`, `Cerebellum.{precompensate,integrate,learn,reset,adaptive_filter}`, `KinematicReacher.simulate_with_correction(...)`, `CerebellumSweepResult` fields, and `run_cerebellum_condition(...)` signature are used identically across Tasks 1–8.
- **Note on test counts:** the "≈ 716" figure in Task 8 is illustrative; the binding requirement is zero failures and all new test files green.
