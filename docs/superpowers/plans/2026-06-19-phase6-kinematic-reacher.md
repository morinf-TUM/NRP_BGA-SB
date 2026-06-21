# Phase 6: Kinematic Reaching Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 2D point-mass kinematic reacher that simulates minimum-jerk reaching trajectories from ClosedLoopPolicy motor commands, extracts movement-level metrics, and re-runs the Phase 5 frequency sweep to confirm BG-frequency effects survive at the movement level.

**Architecture:** A `KinematicReacher` converts each trial's `motor_command_series` (already populated by `ClosedLoopPolicy`) into a 2D position time-series using the minimum-jerk model. A separate `compute_movement_metrics` function extracts scalar metrics from trajectories. A `run_reacher_condition` function re-runs Phase 5 sweep conditions with the reacher attached. All code is pure Python — no nrp-core binding in this phase.

**Tech Stack:** Python 3.10, pydantic ≥ 2.0, numpy ≥ 1.26, pytest ≥ 8.0, ruff ≥ 0.4.

## Global Constraints

- Python 3.10 only (nrp-core host constraint; see PROJECT_MEMORY.md §15.1).
- Pydantic v2 throughout: `BaseModel`, `model_validator(mode="after")`, `model_dump_json`/`model_validate_json`.
- `X | None` union syntax, not `Optional[X]`.
- Fail fast: raise `ValueError` on invalid input; no silent fallbacks, no speculative `getattr`.
- Section-header comments (`# --- SectionName ---`) in multi-section modules.
- Decision-point comments (Trigger / Why / Outcome) on non-obvious branches.
- Tests use `pytest` conventions; no writes to permanent paths (use `tmp_path` if needed).
- All new modules must pass `ruff check` (no lint errors).
- Run `pytest tests/ -q` after each task commit to confirm no regressions.

---

## File Structure

### New files
| File | Responsibility |
|---|---|
| `src/nrp_bga_sb/reacher.py` | `ReacherConfig`, `ReacherTrajectory`, `KinematicReacher`, `_minimum_jerk_scalar` |
| `src/nrp_bga_sb/movement_metrics.py` | `MovementMetrics`, `compute_movement_metrics` |
| `src/nrp_bga_sb/reacher_sweep.py` | `ReacherConditionResult`, `run_reacher_condition` and private helpers |
| `tests/test_reacher.py` | Unit tests for reacher dynamics (~14 tests) |
| `tests/test_movement_metrics.py` | Unit tests for metric extraction (~9 tests) |
| `tests/test_reacher_sweep.py` | Integration tests for reacher sweep (~7 tests) |
| `experiments/kinematic_sweep.py` | Phase 6 experiment runner script |

### Modified files
None. `sweep.py`, `schemas.py`, and the task engines are untouched.

---

## Task 6.1: KinematicReacher

**Files:**
- Create: `src/nrp_bga_sb/reacher.py`
- Test: `tests/test_reacher.py`

**Interfaces:**
- Consumes: `nrp_bga_sb.schemas.MotorCommand` (existing schema; `command: list[float]`, `gate_state: Literal["open","closed","partial"]`, `gate_gain: float`)
- Produces:
  - `ReacherConfig` — config dataclass used by `KinematicReacher` and by `compute_movement_metrics`
  - `ReacherTrajectory` — Pydantic model with `times_ms: list[float]`, `positions_xy: list[list[float]]`, `onset_time_ms: float | None`, `selected_channel: int`
  - `KinematicReacher.simulate(motor_commands: list[MotorCommand], onset_time_ms: float | None, total_duration_ms: float = 500.0) -> ReacherTrajectory`
  - `_minimum_jerk_scalar(t_ms: float, T_ms: float) -> float` — module-level helper (also used in tests)

**Design notes:**
- `ThalamusGate` convention: `command[selected_channel] = gate_gain`, all other channels `= 0.0`. `selected_channel = int(np.argmax(command))` when `gate_state != "closed"`. Fail fast if `command[argmax] == 0.0` for a non-closed gate (indicates a wiring error).
- `gate_gain ∈ [0, 1]` scales the endpoint: `effective_endpoint = gate_gain × target_position`. A partial gate (gain < 1.0) produces a short-of-target movement.
- When `onset_time_ms is None` (no `movement_onset` event logged), default to `0.0` so the trajectory still runs — the caller's responsibility is to pass the correct onset.
- Minimum-jerk formula: `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` where `τ = min(t/T, 1.0)`. This is 0 at t=0, 0.5 at t=T/2, 1.0 at t≥T.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reacher.py`:

```python
"""Tests for KinematicReacher and ReacherTrajectory (Task 6.1)."""
import math
import pytest
from nrp_bga_sb.schemas import MotorCommand


def _make_motor_command(gate_state, gate_gain, channel=0, n_channels=2):
    """Helper: build a MotorCommand consistent with ThalamusGate output."""
    command = [0.0] * n_channels
    if gate_state != "closed":
        command[channel] = gate_gain
    return MotorCommand(
        sim_time=0.0,
        trial_id=1,
        command=command,
        gate_state=gate_state,
        gate_gain=gate_gain,
    )


# --- ReacherConfig ---

def test_reacher_config_defaults():
    from nrp_bga_sb.reacher import ReacherConfig
    cfg = ReacherConfig()
    assert cfg.n_channels == 2
    assert cfg.target_positions == [[-1.0, 0.0], [1.0, 0.0]]
    assert cfg.movement_duration_ms == 300.0
    assert cfg.dt_ms == 1.0


def test_reacher_config_rejects_position_count_mismatch():
    from nrp_bga_sb.reacher import ReacherConfig
    with pytest.raises(Exception):
        ReacherConfig(n_channels=2, target_positions=[[0.0, 1.0]])


# --- _minimum_jerk_scalar ---

def test_minimum_jerk_zero_at_start():
    from nrp_bga_sb.reacher import _minimum_jerk_scalar
    assert _minimum_jerk_scalar(0.0, 300.0) == pytest.approx(0.0)


def test_minimum_jerk_half_at_midpoint():
    from nrp_bga_sb.reacher import _minimum_jerk_scalar
    assert _minimum_jerk_scalar(150.0, 300.0) == pytest.approx(0.5)


def test_minimum_jerk_one_at_end():
    from nrp_bga_sb.reacher import _minimum_jerk_scalar
    assert _minimum_jerk_scalar(300.0, 300.0) == pytest.approx(1.0)


def test_minimum_jerk_saturates_past_end():
    from nrp_bga_sb.reacher import _minimum_jerk_scalar
    assert _minimum_jerk_scalar(600.0, 300.0) == pytest.approx(1.0)


# --- KinematicReacher.simulate ---

def test_simulate_zero_trajectory_empty_commands():
    from nrp_bga_sb.reacher import KinematicReacher
    r = KinematicReacher()
    traj = r.simulate([], onset_time_ms=0.0)
    assert traj.selected_channel == -1
    assert traj.onset_time_ms is None
    assert all(p == [0.0, 0.0] for p in traj.positions_xy)


def test_simulate_zero_trajectory_closed_gate():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("closed", 0.0)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=0.0)
    assert traj.selected_channel == -1
    assert traj.onset_time_ms is None
    assert all(p == [0.0, 0.0] for p in traj.positions_xy)


def test_simulate_full_movement_ch0_reaches_target():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("open", 1.0, channel=0)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=0.0, total_duration_ms=500.0)
    assert traj.selected_channel == 0
    assert traj.onset_time_ms == pytest.approx(0.0)
    final = traj.positions_xy[-1]
    assert final[0] == pytest.approx(-1.0, abs=1e-6)
    assert final[1] == pytest.approx(0.0, abs=1e-6)


def test_simulate_full_movement_ch1_reaches_target():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("open", 1.0, channel=1)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=0.0, total_duration_ms=500.0)
    assert traj.selected_channel == 1
    final = traj.positions_xy[-1]
    assert final[0] == pytest.approx(1.0, abs=1e-6)


def test_simulate_partial_movement_gate_gain_half():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("partial", 0.5, channel=0)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=0.0, total_duration_ms=500.0)
    final = traj.positions_xy[-1]
    # Effective endpoint = 0.5 × target(-1,0) = (-0.5, 0)
    assert final[0] == pytest.approx(-0.5, abs=1e-6)


def test_simulate_onset_respected_positions_before_onset_are_zero():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("open", 1.0, channel=1)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=100.0, total_duration_ms=500.0)
    # All positions before t=100ms should be (0,0)
    for t, p in zip(traj.times_ms, traj.positions_xy):
        if t < 100.0:
            assert p == [0.0, 0.0], f"position at t={t} should be zero before onset"


def test_simulate_trajectory_length():
    from nrp_bga_sb.reacher import KinematicReacher
    r = KinematicReacher()
    traj = r.simulate([], onset_time_ms=None, total_duration_ms=200.0)
    # n_steps = int(round(200.0 / 1.0)) + 1 = 201
    assert len(traj.times_ms) == 201
    assert len(traj.positions_xy) == 201


def test_simulate_raises_on_channel_out_of_range():
    from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
    cmd = _make_motor_command("open", 1.0, channel=0, n_channels=5)
    # Config only has 2 targets
    r = KinematicReacher(ReacherConfig(n_channels=2))
    with pytest.raises(ValueError, match="channel"):
        r.simulate([cmd], onset_time_ms=0.0)


def test_simulate_none_onset_defaults_to_zero():
    from nrp_bga_sb.reacher import KinematicReacher
    cmd = _make_motor_command("open", 1.0, channel=1)
    r = KinematicReacher()
    traj = r.simulate([cmd], onset_time_ms=None, total_duration_ms=500.0)
    # onset_time_ms=None → default 0.0, movement starts immediately
    assert traj.onset_time_ms == pytest.approx(0.0)
    # First position is at t=0, which equals onset, so s=_minimum_jerk(0,300)=0 → still (0,0)
    assert traj.positions_xy[0] == [pytest.approx(0.0), pytest.approx(0.0)]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/fom/code/NRP_BGA-SB && python -m pytest tests/test_reacher.py -q 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'nrp_bga_sb.reacher'`

- [ ] **Step 3: Write the implementation**

Create `src/nrp_bga_sb/reacher.py`:

```python
"""2D point-mass kinematic reacher: trajectory simulation (Task 6.1).

Converts a ClosedLoopPolicy motor_command_series into a minimum-jerk 2D
position time-series. Phase 6 uses a single motor command per trial (one
ClosedLoopPolicy call per task-engine trial). Phase 8 may extend this to
multi-command trajectories for change-of-mind reversal.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, model_validator

from nrp_bga_sb.schemas import MotorCommand

# --- ReacherConfig ---


class ReacherConfig(BaseModel):
    """Physical and timing parameters for the 2D point-mass reacher."""

    n_channels: int = 2
    # (x, y) position of each action channel's target; index i → target_positions[i]
    target_positions: list[list[float]] = [[-1.0, 0.0], [1.0, 0.0]]
    # Duration (ms) for a full-amplitude (gate_gain=1.0) minimum-jerk reach
    movement_duration_ms: float = 300.0
    # Simulation timestep (ms)
    dt_ms: float = 1.0

    @model_validator(mode="after")
    def _check_target_count(self) -> "ReacherConfig":
        # Trigger: target_positions length does not match n_channels.
        # Why: a mismatch means channel indices map to undefined targets.
        # Outcome: ValidationError raised at construction; caller must fix config.
        if len(self.target_positions) != self.n_channels:
            raise ValueError(
                f"target_positions has {len(self.target_positions)} entries "
                f"but n_channels={self.n_channels}"
            )
        return self


# --- ReacherTrajectory ---


class ReacherTrajectory(BaseModel):
    """Position time-series produced by KinematicReacher.simulate.

    positions_xy: list of [x, y] at each simulation timestep (len == len(times_ms)).
    onset_time_ms: None means no movement occurred (gate was closed).
    selected_channel: -1 means no movement.
    """

    times_ms: list[float]
    positions_xy: list[list[float]]   # each entry is [x, y]
    onset_time_ms: float | None       # None = gate closed, no movement
    selected_channel: int             # -1 = no movement


# --- Private helper ---


def _minimum_jerk_scalar(t_ms: float, T_ms: float) -> float:
    """Normalized minimum-jerk displacement: 0 at t=0, 0.5 at t=T/2, 1 at t≥T."""
    if T_ms <= 0.0:
        return 1.0
    tau = min(t_ms / T_ms, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


# --- KinematicReacher ---


class KinematicReacher:
    """Simulate a minimum-jerk reaching trajectory from a motor command series.

    Each simulate() call is stateless — no per-instance state is mutated.
    """

    def __init__(self, config: ReacherConfig | None = None) -> None:
        self.config = config or ReacherConfig()

    def simulate(
        self,
        motor_commands: list[MotorCommand],
        onset_time_ms: float | None,
        total_duration_ms: float = 500.0,
    ) -> ReacherTrajectory:
        """Simulate a reaching trajectory from a trial's motor command series.

        Args:
            motor_commands: trial_log.motor_command_series (one entry per
                            ClosedLoopPolicy call; MotorCommand.gate_state
                            determines whether movement occurs).
            onset_time_ms:  movement start time in ms (trial_log.movement_onset_time
                            × 1000); None defaults to 0.0.
            total_duration_ms: simulation window length in ms.

        Returns:
            ReacherTrajectory with positions at each dt_ms tick.
        """
        n_steps = int(round(total_duration_ms / self.config.dt_ms)) + 1
        times_ms = [i * self.config.dt_ms for i in range(n_steps)]
        zero_positions = [[0.0, 0.0]] * n_steps

        if not motor_commands:
            return ReacherTrajectory(
                times_ms=times_ms,
                positions_xy=zero_positions,
                onset_time_ms=None,
                selected_channel=-1,
            )

        # Trigger: multiple motor commands possible (e.g., change_of_mind engine
        # calls policy twice: pre-switch + post-switch). Use the last command as the
        # final committed movement direction. Phase 8 will extend to multi-command
        # trajectories for explicit reversal simulation.
        last_cmd = motor_commands[-1]

        if last_cmd.gate_state == "closed":
            return ReacherTrajectory(
                times_ms=times_ms,
                positions_xy=zero_positions,
                onset_time_ms=None,
                selected_channel=-1,
            )

        # ThalamusGate convention: command[selected_channel] = gate_gain, others = 0.0.
        # argmax recovers the selected channel; guard against all-zero (wiring error).
        selected_channel = int(np.argmax(last_cmd.command))
        if selected_channel >= self.config.n_channels:
            raise ValueError(
                f"Command selects channel {selected_channel} but "
                f"config has n_channels={self.config.n_channels}"
            )
        if last_cmd.command[selected_channel] == 0.0:
            # Trigger: gate_state is not "closed" but command vector is all-zero.
            # Why: indicates a ThalamusGate wiring error — valid partial/open gates
            #      always have command[selected_channel] > 0.
            # Outcome: fail fast rather than silently simulating a zero movement.
            raise ValueError(
                f"gate_state={last_cmd.gate_state!r} but command is all-zero: "
                f"{last_cmd.command}"
            )

        tx, ty = self.config.target_positions[selected_channel]
        # gate_gain ∈ [0, 1] scales the endpoint: partial gate → short of target.
        ex = tx * last_cmd.gate_gain
        ey = ty * last_cmd.gate_gain

        # onset_time_ms=None means no movement_onset event was logged; default 0.0
        # so the trajectory still runs (the caller's responsibility is correctness).
        actual_onset = onset_time_ms if onset_time_ms is not None else 0.0
        T = self.config.movement_duration_ms

        positions_xy: list[list[float]] = []
        for t_ms in times_ms:
            if t_ms < actual_onset:
                positions_xy.append([0.0, 0.0])
            else:
                s = _minimum_jerk_scalar(t_ms - actual_onset, T)
                positions_xy.append([ex * s, ey * s])

        return ReacherTrajectory(
            times_ms=times_ms,
            positions_xy=positions_xy,
            onset_time_ms=actual_onset,
            selected_channel=selected_channel,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/fom/code/NRP_BGA-SB && python -m pytest tests/test_reacher.py -v
```

Expected: 14 tests PASSED. If any fail, fix before proceeding.

- [ ] **Step 5: Run ruff and full suite**

```bash
cd /home/fom/code/NRP_BGA-SB && ruff check src/nrp_bga_sb/reacher.py tests/test_reacher.py && python -m pytest tests/ -q
```

Expected: ruff clean; all prior tests still pass plus 14 new ones.

- [ ] **Step 6: Commit**

```bash
cd /home/fom/code/NRP_BGA-SB && git add src/nrp_bga_sb/reacher.py tests/test_reacher.py && git commit -m "$(cat <<'EOF'
feat: 2D point-mass kinematic reacher with minimum-jerk trajectories (Task 6.1)

ReacherConfig, ReacherTrajectory, KinematicReacher. simulate() converts
ClosedLoopPolicy motor_command_series into minimum-jerk 2D position series.
gate_gain scales the endpoint; closed gate → zero trajectory.

ChangeSet-ID: phase6-reacher
EOF
)"
```

---

## Task 6.2: MovementMetrics

**Files:**
- Create: `src/nrp_bga_sb/movement_metrics.py`
- Test: `tests/test_movement_metrics.py`

**Interfaces:**
- Consumes: `ReacherConfig`, `ReacherTrajectory` from `nrp_bga_sb.reacher`
- Produces:
  - `MovementMetrics` — Pydantic model (see fields below)
  - `compute_movement_metrics(trajectory: ReacherTrajectory, config: ReacherConfig) -> MovementMetrics`

**`MovementMetrics` fields:**
| Field | Type | Description |
|---|---|---|
| `movement_onset_time_ms` | `float \| None` | None if no movement |
| `endpoint_error` | `float` | Distance from final position to target (0.0 if no movement) |
| `partial_movement_amplitude` | `float` | Euclidean distance from origin to final position |
| `trajectory_curvature` | `float` | Mean absolute perpendicular deviation from origin→endpoint line; 0.0 for straight |
| `movement_reversal_time_ms` | `float \| None` | Time of first velocity direction reversal; None if none |
| `peak_velocity` | `float` | Maximum instantaneous speed (units/ms) |

**Design notes:**
- Endpoint error = `‖final_pos − target_positions[selected_channel]‖`. For gate_gain < 1.0, final_pos = gate_gain × target, so endpoint_error = (1 − gate_gain) × ‖target‖. For gate_gain = 1.0, endpoint_error = 0.
- `trajectory_curvature`: project each position onto the unit vector along origin→final_pos; curvature = mean perpendicular deviation. For single-command min-jerk trajectories (all positions are scalar multiples of the target direction), this is always 0.0.
- `movement_reversal_time_ms`: first time where projected velocity flips from positive to negative. In Phase 6 this is always `None` (monotone min-jerk). Phase 8 change-of-mind trajectories will produce non-None values.
- Peak velocity for min-jerk: v(τ) = amplitude × (30τ² − 60τ³ + 30τ⁴) / T; maximum at τ = 0.5: v_peak = 1.875 × amplitude / T. For amplitude=1.0, T=300ms: v_peak ≈ 0.00625 units/ms.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_movement_metrics.py`:

```python
"""Tests for MovementMetrics and compute_movement_metrics (Task 6.2)."""
import math
import pytest
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.schemas import MotorCommand


def _make_motor_command(gate_state, gate_gain, channel=0, n_channels=2):
    command = [0.0] * n_channels
    if gate_state != "closed":
        command[channel] = gate_gain
    return MotorCommand(
        sim_time=0.0, trial_id=1,
        command=command, gate_state=gate_state, gate_gain=gate_gain,
    )


def _traj(gate_state, gate_gain, channel=0, onset_ms=0.0):
    """Build a ReacherTrajectory via KinematicReacher."""
    cmd = _make_motor_command(gate_state, gate_gain, channel)
    return KinematicReacher().simulate([cmd], onset_time_ms=onset_ms, total_duration_ms=500.0)


# --- no movement ---

def test_no_movement_closed_gate():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("closed", 0.0)
    m = compute_movement_metrics(traj, ReacherConfig())
    assert m.movement_onset_time_ms is None
    assert m.endpoint_error == pytest.approx(0.0)
    assert m.partial_movement_amplitude == pytest.approx(0.0)
    assert m.trajectory_curvature == pytest.approx(0.0)
    assert m.movement_reversal_time_ms is None
    assert m.peak_velocity == pytest.approx(0.0, abs=1e-9)


# --- full movement ---

def test_full_movement_zero_endpoint_error():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    # Full gate_gain=1.0 → reaches target (-1,0) exactly
    assert m.endpoint_error == pytest.approx(0.0, abs=1e-6)


def test_full_movement_amplitude_equals_target_distance():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=1)
    m = compute_movement_metrics(traj, ReacherConfig())
    # target (1, 0): distance from origin = 1.0
    assert m.partial_movement_amplitude == pytest.approx(1.0, abs=1e-6)


# --- partial movement ---

def test_partial_movement_endpoint_error():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("partial", 0.5, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    # endpoint = (-0.5, 0), target = (-1, 0), error = 0.5
    assert m.endpoint_error == pytest.approx(0.5, abs=1e-6)


def test_partial_movement_amplitude():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("partial", 0.5, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    assert m.partial_movement_amplitude == pytest.approx(0.5, abs=1e-6)


# --- straight-line curvature ---

def test_straight_line_curvature_is_zero():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    # Min-jerk toward (-1,0): all positions are along the x-axis → curvature = 0
    assert m.trajectory_curvature == pytest.approx(0.0, abs=1e-9)


# --- velocity ---

def test_peak_velocity_positive_for_full_movement():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=1)
    m = compute_movement_metrics(traj, ReacherConfig())
    assert m.peak_velocity > 0.0


def test_peak_velocity_min_jerk_formula():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    # v_peak = 1.875 * amplitude / T for min-jerk
    # amplitude = 1.0 (gate_gain=1, target at ±1), T = 300 ms
    traj = _traj("open", 1.0, channel=1)
    m = compute_movement_metrics(traj, ReacherConfig())
    expected_peak = 1.875 * 1.0 / 300.0
    assert m.peak_velocity == pytest.approx(expected_peak, rel=0.05)


# --- reversal ---

def test_no_reversal_for_single_command():
    from nrp_bga_sb.movement_metrics import compute_movement_metrics
    traj = _traj("open", 1.0, channel=0)
    m = compute_movement_metrics(traj, ReacherConfig())
    # Monotone min-jerk → no reversal
    assert m.movement_reversal_time_ms is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/fom/code/NRP_BGA-SB && python -m pytest tests/test_movement_metrics.py -q 2>&1 | head -5
```

Expected: `ModuleNotFoundError: No module named 'nrp_bga_sb.movement_metrics'`

- [ ] **Step 3: Write the implementation**

Create `src/nrp_bga_sb/movement_metrics.py`:

```python
"""Movement metrics extracted from kinematic reacher trajectories (Task 6.2).

Metrics are derived from a ReacherTrajectory and a ReacherConfig. The config
supplies target positions for endpoint_error. All computations are stateless
numpy operations on the trajectory arrays.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.reacher import ReacherConfig, ReacherTrajectory

# --- MovementMetrics ---


class MovementMetrics(BaseModel):
    """Scalar movement metrics extracted from one ReacherTrajectory."""

    movement_onset_time_ms: float | None   # None if gate was closed (no movement)
    endpoint_error: float                  # ‖final_pos − target‖; 0.0 if no movement
    partial_movement_amplitude: float      # ‖final_pos‖ (distance from origin)
    trajectory_curvature: float            # mean absolute perpendicular deviation from straight line
    movement_reversal_time_ms: float | None  # first velocity reversal time; None if none
    peak_velocity: float                   # max instantaneous speed (position units / ms)


# --- compute_movement_metrics ---


def compute_movement_metrics(
    trajectory: ReacherTrajectory,
    config: ReacherConfig,
) -> MovementMetrics:
    """Extract movement metrics from a simulated trajectory.

    Args:
        trajectory: ReacherTrajectory from KinematicReacher.simulate.
        config: ReacherConfig supplying target_positions for endpoint_error.
    """
    positions = np.array(trajectory.positions_xy, dtype=float)  # (n, 2)
    times = np.array(trajectory.times_ms, dtype=float)          # (n,)
    final_pos = positions[-1]                                    # (2,)

    # --- partial_movement_amplitude ---
    partial_movement_amplitude = float(np.linalg.norm(final_pos))

    # --- endpoint_error ---
    if trajectory.selected_channel >= 0:
        target = np.array(
            config.target_positions[trajectory.selected_channel], dtype=float
        )
        endpoint_error = float(np.linalg.norm(final_pos - target))
    else:
        # No movement: reacher stayed at origin, no target was attempted
        endpoint_error = 0.0

    # --- trajectory_curvature ---
    # Mean absolute perpendicular deviation from the straight line origin → final_pos.
    # For single-command minimum-jerk trajectories all positions are scalar multiples
    # of the target direction, so curvature is always 0.0 in Phase 6.
    if partial_movement_amplitude < 1e-9:
        trajectory_curvature = 0.0
    else:
        unit_dir = final_pos / partial_movement_amplitude          # (2,)
        proj_scalars = positions @ unit_dir                        # (n,)
        projected_on_line = np.outer(proj_scalars, unit_dir)       # (n, 2)
        perp_deviations = np.linalg.norm(positions - projected_on_line, axis=1)
        trajectory_curvature = float(perp_deviations.mean())

    # --- peak_velocity ---
    if len(times) > 1:
        dt = np.diff(times)                      # (n-1,)
        dpos = np.diff(positions, axis=0)        # (n-1, 2)
        # Guard against zero dt (should not occur for valid simulations)
        safe_dt = np.where(dt > 0.0, dt, np.inf)
        speeds = np.linalg.norm(dpos, axis=1) / safe_dt
        peak_velocity = float(speeds.max())
    else:
        peak_velocity = 0.0

    # --- movement_reversal_time_ms ---
    # A reversal is the first timestep where projected velocity flips from
    # positive (toward target) to negative (away from target). In Phase 6 this is
    # always None because single-command min-jerk is monotone. Phase 8 change-of-mind
    # trajectories (two policy calls, initial + post-switch command) will exercise this.
    movement_reversal_time_ms: float | None = None
    if (
        trajectory.selected_channel >= 0
        and partial_movement_amplitude > 1e-9
        and len(times) > 2
    ):
        unit_dir = final_pos / partial_movement_amplitude
        dt = np.diff(times)
        dpos = np.diff(positions, axis=0)
        safe_dt = np.where(dt > 0.0, dt, np.inf)
        proj_vel = (dpos @ unit_dir) / safe_dt  # (n-1,)
        for i in range(1, len(proj_vel)):
            if proj_vel[i - 1] > 1e-9 and proj_vel[i] < -1e-9:
                movement_reversal_time_ms = float(times[i + 1])
                break

    return MovementMetrics(
        movement_onset_time_ms=trajectory.onset_time_ms,
        endpoint_error=endpoint_error,
        partial_movement_amplitude=partial_movement_amplitude,
        trajectory_curvature=trajectory_curvature,
        movement_reversal_time_ms=movement_reversal_time_ms,
        peak_velocity=peak_velocity,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/fom/code/NRP_BGA-SB && python -m pytest tests/test_movement_metrics.py -v
```

Expected: 9 tests PASSED.

- [ ] **Step 5: Run ruff and full suite**

```bash
cd /home/fom/code/NRP_BGA-SB && ruff check src/nrp_bga_sb/movement_metrics.py tests/test_movement_metrics.py && python -m pytest tests/ -q
```

Expected: ruff clean; all prior tests + 14 (6.1) + 9 (6.2) pass.

- [ ] **Step 6: Commit**

```bash
cd /home/fom/code/NRP_BGA-SB && git add src/nrp_bga_sb/movement_metrics.py tests/test_movement_metrics.py && git commit -m "$(cat <<'EOF'
feat: movement metrics extraction from reacher trajectories (Task 6.2)

MovementMetrics: onset_time_ms, endpoint_error, partial_amplitude,
trajectory_curvature, reversal_time_ms, peak_velocity. compute_movement_metrics
uses numpy perpendicular-deviation and projected-velocity idioms.

ChangeSet-ID: phase6-movement-metrics
EOF
)"
```

---

## Task 6.3: Reacher Sweep

**Files:**
- Create: `src/nrp_bga_sb/reacher_sweep.py`
- Create: `experiments/kinematic_sweep.py`
- Test: `tests/test_reacher_sweep.py`

**Interfaces:**
- Consumes: `KinematicReacher`, `ReacherConfig` from `nrp_bga_sb.reacher`; `compute_movement_metrics`, `MovementMetrics` from `nrp_bga_sb.movement_metrics`; `CONFLICT_PEAK_SALIENCE` from `nrp_bga_sb.sweep`; `make_closed_loop_policy` from `nrp_bga_sb.closed_loop`
- Produces:
  - `ReacherConditionResult` — Pydantic model (fields below)
  - `run_reacher_condition(frequency_hz, conflict_level, paradigm, n_trials, seed, ...) -> ReacherConditionResult`

**`ReacherConditionResult` fields:**
| Field | Type | Description |
|---|---|---|
| `frequency_hz` | `float` | BG update frequency |
| `conflict_level` | `Literal["low","medium","high"]` | Evidence discriminability |
| `paradigm` | `Literal["go_nogo","two_choice"]` | Task engine |
| `seed` | `int` | Trial seed |
| `n_trials` | `int` | Number of trials run |
| `miss_rate` | `float \| None` | Go_nogo: fraction of go trials missed |
| `go_success_rate` | `float \| None` | Go_nogo: fraction of go trials succeeded |
| `timeout_rate` | `float \| None` | Two_choice: fraction of trials with no selection |
| `bg_commitment_latency_mean` | `float \| None` | Mean BG commitment latency (s) |
| `movement_onset_rate` | `float` | Fraction of trials where movement occurred |
| `mean_endpoint_error` | `float` | Mean endpoint error over trials with movement (0.0 if none) |
| `mean_partial_amplitude` | `float` | Mean movement amplitude (0.0 if no movement trials) |
| `mean_peak_velocity` | `float` | Mean peak velocity over trials with movement (0.0 if none) |

**Design notes:**
- `run_reacher_condition` replicates the engine-setup pattern from `sweep._run_engine` locally (no modification to sweep.py). Constants `_RISE_TIME_MS = 200.0` and `_ACCUMULATION_MS = 200.0` match Phase 5 values.
- `movement_onset_rate` should match `go_success_rate` for go_nogo (every successful trial emits one open/partial MotorCommand). This is the key acceptance check.
- At 160 Hz low conflict: `go_success_rate ≈ 1.0`, `mean_endpoint_error ≈ 0.0` (gate_gain = 1.0 for open gate). At 5 Hz: `go_success_rate ≈ 0.0`, `movement_onset_rate ≈ 0.0`.
- `ClosedLoopPolicy` always appends a `MotorCommand` (even closed-gate) so `motor_command_series` is never empty for trials that ran through the policy.
- Movement onset from trial log: `trial.movement_onset_time * 1000.0` (seconds → ms); `None` if no onset recorded (miss trial).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reacher_sweep.py`:

```python
"""Tests for ReacherConditionResult and run_reacher_condition (Task 6.3)."""
import pytest


def test_reacher_condition_result_has_required_fields():
    from nrp_bga_sb.reacher_sweep import ReacherConditionResult
    # Verify all expected fields exist with correct defaults/types
    r = ReacherConditionResult(
        frequency_hz=160.0,
        conflict_level="low",
        paradigm="go_nogo",
        seed=0,
        n_trials=10,
        miss_rate=0.0,
        go_success_rate=1.0,
        timeout_rate=None,
        bg_commitment_latency_mean=0.1,
        movement_onset_rate=1.0,
        mean_endpoint_error=0.0,
        mean_partial_amplitude=1.0,
        mean_peak_velocity=0.005,
    )
    assert r.frequency_hz == 160.0
    assert r.movement_onset_rate == 1.0


def test_high_freq_low_conflict_low_miss_rate():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=160.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=20,
        seed=42,
    )
    assert result.miss_rate is not None
    assert result.miss_rate == pytest.approx(0.0, abs=0.15)


def test_low_freq_high_miss_rate():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=5.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=20,
        seed=42,
    )
    assert result.miss_rate is not None
    assert result.miss_rate == pytest.approx(1.0, abs=0.15)


def test_movement_onset_rate_matches_go_success_rate():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=40.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=20,
        seed=7,
    )
    if result.go_success_rate is not None:
        assert result.movement_onset_rate == pytest.approx(
            result.go_success_rate, abs=0.05
        )


def test_movement_metrics_nonnegative():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=40.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=10,
        seed=0,
    )
    assert result.mean_endpoint_error >= 0.0
    assert result.mean_partial_amplitude >= 0.0
    assert result.mean_peak_velocity >= 0.0
    assert 0.0 <= result.movement_onset_rate <= 1.0


def test_high_freq_low_conflict_zero_endpoint_error():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=160.0,
        conflict_level="low",
        paradigm="go_nogo",
        n_trials=10,
        seed=0,
    )
    # At 160 Hz, gate_gain = 1.0 for all go trials → endpoint error = 0
    assert result.mean_endpoint_error == pytest.approx(0.0, abs=1e-6)


def test_two_choice_paradigm_returns_timeout_rate():
    from nrp_bga_sb.reacher_sweep import run_reacher_condition
    result = run_reacher_condition(
        frequency_hz=160.0,
        conflict_level="low",
        paradigm="two_choice",
        n_trials=10,
        seed=0,
    )
    assert result.timeout_rate is not None
    assert result.miss_rate is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/fom/code/NRP_BGA-SB && python -m pytest tests/test_reacher_sweep.py -q 2>&1 | head -5
```

Expected: `ModuleNotFoundError: No module named 'nrp_bga_sb.reacher_sweep'`

- [ ] **Step 3: Write `src/nrp_bga_sb/reacher_sweep.py`**

```python
"""Reacher-augmented sweep condition runner (Task 6.3).

Mirrors the Phase 5 run_condition interface but attaches a KinematicReacher
to each trial, adding movement-level metrics to the per-condition result.
The engine setup replicates sweep._run_engine locally to avoid modifying
the stable Phase 5 sweep module.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.engines.two_choice import TwoChoiceConfig, run_two_choice_trials
from nrp_bga_sb.movement_metrics import compute_movement_metrics
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog
from nrp_bga_sb.sweep import CONFLICT_PEAK_SALIENCE

# Type aliases matching sweep.py
ConflictLevel = Literal["low", "medium", "high"]
Paradigm = Literal["go_nogo", "two_choice"]

# Timing constants matching Phase 5 (see sweep.py module docstring)
_RISE_TIME_MS: float = 200.0
_ACCUMULATION_MS: float = 200.0

# Gaussian noise for two_choice (matching sweep.py)
_TWO_CHOICE_NOISE_STD: float = 0.05


# --- ReacherConditionResult ---


class ReacherConditionResult(BaseModel):
    """All metrics for one condition: Phase 5 abstract + Phase 6 movement."""

    frequency_hz: float
    conflict_level: ConflictLevel
    paradigm: Paradigm
    seed: int
    n_trials: int
    # Phase 5 abstract metrics
    miss_rate: float | None
    go_success_rate: float | None
    timeout_rate: float | None
    bg_commitment_latency_mean: float | None
    # Phase 6 movement metrics (aggregated over trials that had movement)
    movement_onset_rate: float     # fraction of trials where movement occurred
    mean_endpoint_error: float     # mean ‖endpoint − target‖ over movement trials
    mean_partial_amplitude: float  # mean ‖endpoint‖ over movement trials
    mean_peak_velocity: float      # mean peak speed over movement trials


# --- Public condition runner ---


def run_reacher_condition(
    frequency_hz: float,
    conflict_level: ConflictLevel,
    paradigm: Paradigm,
    n_trials: int,
    seed: int,
    reacher_config: ReacherConfig | None = None,
    accumulation_ms: float = _ACCUMULATION_MS,
    rise_time_ms: float = _RISE_TIME_MS,
) -> ReacherConditionResult:
    """Run one sweep condition with the kinematic reacher and return all metrics.

    Args:
        frequency_hz:    BG update frequency; applied to all four knobs via
                         FrequencyConfig.from_effective_hz.
        conflict_level:  Evidence discriminability ("low", "medium", "high").
        paradigm:        Task engine ("go_nogo" or "two_choice").
        n_trials:        Number of trials per condition.
        seed:            Random seed (deterministic).
        reacher_config:  KinematicReacher config; defaults to ReacherConfig().
        accumulation_ms: ScheduledBGAdapter pre-decision window (ms).
        rise_time_ms:    CortexEvidenceGenerator ramp duration (ms).
    """
    peak_salience = CONFLICT_PEAK_SALIENCE[conflict_level]
    noise_std = _TWO_CHOICE_NOISE_STD if paradigm == "two_choice" else 0.0

    freq_cfg = FrequencyConfig.from_effective_hz(frequency_hz)
    cortex_cfg = CortexConfig(
        rise_time_ms=rise_time_ms,
        peak_salience=peak_salience,
        noise_std=noise_std,
    )
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
        accumulation_ms=accumulation_ms,
    )

    trials = _run_engine(paradigm, n_trials, seed, policy)
    cfg = reacher_config or ReacherConfig()
    reacher = KinematicReacher(cfg)

    abstract = _compute_abstract_metrics(trials, paradigm)
    movement = _compute_movement_aggregate(trials, reacher, cfg)

    return ReacherConditionResult(
        frequency_hz=frequency_hz,
        conflict_level=conflict_level,
        paradigm=paradigm,
        seed=seed,
        n_trials=n_trials,
        **abstract,
        **movement,
    )


# --- Private helpers ---


def _run_engine(
    paradigm: Paradigm,
    n_trials: int,
    seed: int,
    policy,
) -> list[TrialLog]:
    """Run the task engine and return trial logs.

    Mirrors sweep._run_engine with identical config values so that condition
    results are comparable between Phase 5 and Phase 6.
    """
    if paradigm == "go_nogo":
        config = GoNoGoConfig(
            n_trials=n_trials,
            go_probability=0.7,
            response_window_start_ms=0,
            response_window_duration_ms=600,
            fixation_duration_ms=200,
            cue_onset_ms=400,
            decision_point_ms=300,
            seed=seed,
        )
        return run_go_nogo_trials(config, policy)
    elif paradigm == "two_choice":
        config = TwoChoiceConfig(
            n_trials=n_trials,
            conflict_levels={"conflict": [0.65, 0.35]},
            response_window_start_ms=0,
            response_window_duration_ms=600,
            fixation_duration_ms=200,
            target_onset_ms=400,
            decision_point_ms=300,
            seed=seed,
        )
        return run_two_choice_trials(config, policy)
    else:
        raise ValueError(f"Unsupported paradigm: {paradigm!r}")


def _compute_abstract_metrics(trials: list[TrialLog], paradigm: Paradigm) -> dict:
    """Mirror Phase 5 abstract metrics for comparison."""
    n = len(trials)

    if paradigm == "go_nogo":
        go_trials = [t for t in trials if t.cue_identity == "go"]
        if go_trials:
            miss_rate: float | None = (
                sum(1 for t in go_trials if t.failure_mode == "miss") / len(go_trials)
            )
            go_success_rate: float | None = (
                sum(1 for t in go_trials if t.success is True) / len(go_trials)
            )
        else:
            miss_rate = go_success_rate = None
        timeout_rate: float | None = None
    else:
        miss_rate = go_success_rate = None
        timeouts = sum(1 for t in trials if t.failure_mode == "timeout")
        timeout_rate = timeouts / n if n > 0 else None

    latencies = [
        t.thalamic_relay_time - t.cue_onset_time
        for t in trials
        if t.thalamic_relay_time is not None
    ]
    bg_commitment_latency_mean: float | None = (
        float(np.mean(latencies)) if latencies else None
    )

    return {
        "miss_rate": miss_rate,
        "go_success_rate": go_success_rate,
        "timeout_rate": timeout_rate,
        "bg_commitment_latency_mean": bg_commitment_latency_mean,
    }


def _compute_movement_aggregate(
    trials: list[TrialLog],
    reacher: KinematicReacher,
    config: ReacherConfig,
) -> dict:
    """Run the reacher on each trial and aggregate movement metrics."""
    movement_metrics = []

    for trial in trials:
        if not trial.motor_command_series:
            # Trigger: policy was never called (should not happen for ClosedLoopPolicy).
            # Why: guard against a task engine that skips the policy on some trials.
            continue
        onset_ms = (
            trial.movement_onset_time * 1000.0
            if trial.movement_onset_time is not None
            else None
        )
        traj = reacher.simulate(trial.motor_command_series, onset_ms)
        m = compute_movement_metrics(traj, config)
        movement_metrics.append(m)

    movement_trials = [m for m in movement_metrics if m.movement_onset_time_ms is not None]
    n_total = len(trials)
    n_move = len(movement_trials)

    return {
        "movement_onset_rate": n_move / n_total if n_total > 0 else 0.0,
        "mean_endpoint_error": (
            float(np.mean([m.endpoint_error for m in movement_trials]))
            if movement_trials else 0.0
        ),
        "mean_partial_amplitude": (
            float(np.mean([m.partial_movement_amplitude for m in movement_trials]))
            if movement_trials else 0.0
        ),
        "mean_peak_velocity": (
            float(np.mean([m.peak_velocity for m in movement_trials]))
            if movement_trials else 0.0
        ),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/fom/code/NRP_BGA-SB && python -m pytest tests/test_reacher_sweep.py -v
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Write `experiments/kinematic_sweep.py`**

```python
"""Phase 6 kinematic sweep runner.

Re-runs a subset of the Phase 5 frequency sweep (5 frequencies × 3 conflict
levels × 2 paradigms × 5 seeds = 150 conditions) with the KinematicReacher
attached.  Prints a report comparing movement_onset_rate to go_success_rate
(the Phase 6 acceptance check) and saves results to JSON.

Run:
    cd /home/fom/code/NRP_BGA-SB
    python experiments/kinematic_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nrp_bga_sb.reacher_sweep import ReacherConditionResult, run_reacher_condition

# --- Sweep parameters ---

FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 160.0]
CONFLICT_LEVELS = ["low", "medium", "high"]
PARADIGMS = ["go_nogo", "two_choice"]
N_SEEDS = 5
N_TRIALS = 30

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    results: list[ReacherConditionResult] = []
    total = len(FREQUENCIES_HZ) * len(CONFLICT_LEVELS) * len(PARADIGMS) * N_SEEDS
    done = 0

    for freq in FREQUENCIES_HZ:
        for conflict in CONFLICT_LEVELS:
            for paradigm in PARADIGMS:
                for seed in range(N_SEEDS):
                    result = run_reacher_condition(
                        frequency_hz=freq,
                        conflict_level=conflict,
                        paradigm=paradigm,
                        n_trials=N_TRIALS,
                        seed=seed,
                    )
                    results.append(result)
                    done += 1
                    if done % 10 == 0:
                        print(f"  {done}/{total} conditions done", flush=True)

    # Save results
    out_path = RESULTS_DIR / "kinematic_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    print(f"\nSaved {len(results)} conditions to {out_path}")

    # --- Acceptance check report ---
    _print_acceptance_report(results)


def _print_acceptance_report(results: list[ReacherConditionResult]) -> None:
    """Print movement_onset_rate vs go_success_rate by frequency (go_nogo only)."""
    print("\n=== Phase 6 Acceptance: movement_onset_rate vs go_success_rate ===")
    print(f"{'Freq (Hz)':<12} {'Conflict':<10} {'go_success':<12} {'onset_rate':<12} {'Match?'}")
    print("-" * 60)

    go_results = [r for r in results if r.paradigm == "go_nogo" and r.seed == 0]
    go_results.sort(key=lambda r: (r.conflict_level, r.frequency_hz))

    for r in go_results:
        if r.go_success_rate is None:
            continue
        match = abs(r.movement_onset_rate - r.go_success_rate) < 0.05
        print(
            f"{r.frequency_hz:<12.0f} {r.conflict_level:<10} "
            f"{r.go_success_rate:<12.3f} {r.movement_onset_rate:<12.3f} "
            f"{'✓' if match else '✗'}"
        )

    print("\n=== Endpoint error by frequency (go_nogo, low conflict, seed=0) ===")
    subset = [
        r for r in results
        if r.paradigm == "go_nogo" and r.conflict_level == "low" and r.seed == 0
    ]
    subset.sort(key=lambda r: r.frequency_hz)
    for r in subset:
        print(f"  {r.frequency_hz:>6.0f} Hz: endpoint_error={r.mean_endpoint_error:.4f}  "
              f"amplitude={r.mean_partial_amplitude:.4f}  peak_vel={r.mean_peak_velocity:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full test suite with ruff**

```bash
cd /home/fom/code/NRP_BGA-SB && ruff check src/nrp_bga_sb/reacher_sweep.py experiments/kinematic_sweep.py && python -m pytest tests/ -q
```

Expected: ruff clean; all tests pass (499 prior + 14 + 9 + 7 = 529).

- [ ] **Step 7: Run the experiment script to verify it produces output**

```bash
cd /home/fom/code/NRP_BGA-SB && python experiments/kinematic_sweep.py
```

Expected output (structure, not exact numbers):
```
  10/150 conditions done
  ...
Saved 150 conditions to .../results/kinematic_sweep_results.json

=== Phase 6 Acceptance: movement_onset_rate vs go_success_rate ===
Freq (Hz)    Conflict   go_success   onset_rate   Match?
------------------------------------------------------------
   5         high       0.000        0.000        ✓
   5         low        0.000        0.000        ✓
...
 160         low        1.000        1.000        ✓
```

Acceptance criterion: all Match? entries show `✓`. If any show `✗`, investigate the reacher's movement-onset detection logic.

- [ ] **Step 8: Commit**

```bash
cd /home/fom/code/NRP_BGA-SB && git add src/nrp_bga_sb/reacher_sweep.py tests/test_reacher_sweep.py experiments/kinematic_sweep.py && git commit -m "$(cat <<'EOF'
feat: reacher-augmented sweep runner and Phase 6 experiment (Task 6.3)

ReacherConditionResult mirrors SweepConditionResult + movement_onset_rate,
mean_endpoint_error, mean_partial_amplitude, mean_peak_velocity.
run_reacher_condition re-runs Phase 5 conditions with KinematicReacher.
experiments/kinematic_sweep.py: 150-condition sweep with acceptance report.

ChangeSet-ID: phase6-reacher-sweep
EOF
)"
```

---

## Self-review

### Spec coverage
- **6.1 (2D arm or point-mass reacher):** Task 6.1 — ✓ point-mass minimum-jerk reacher in `reacher.py`
- **6.2 (movement onset time, trajectory curvature, endpoint error, reversal time, partial amplitude):** Task 6.2 — ✓ all five metrics in `MovementMetrics`
- **6.3 (re-run Phase 5 sweep on kinematic reacher; confirm BG-frequency effects survive):** Task 6.3 — ✓ `run_reacher_condition` + `kinematic_sweep.py` acceptance report

### Placeholder scan
- No TBD, TODO, or "similar to" patterns.
- All code steps contain complete, runnable code.

### Type consistency
- `ReacherConfig` → `target_positions: list[list[float]]` — matches `positions_xy: list[list[float]]` in `ReacherTrajectory` ✓
- `compute_movement_metrics(trajectory: ReacherTrajectory, config: ReacherConfig)` — both types defined in Task 6.1, used in Task 6.2 ✓
- `_compute_movement_aggregate(trials, reacher, cfg)` receives `ReacherConfig` as `cfg` — matches `compute_movement_metrics` signature ✓
- `run_reacher_condition` passes `ReacherConfig | None` as `reacher_config` — default `ReacherConfig()` used throughout ✓
