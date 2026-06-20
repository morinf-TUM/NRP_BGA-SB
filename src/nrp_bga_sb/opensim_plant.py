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
    trial_id: str
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
    # dt_ms=2.0: validated step size that avoids onset oscillation (Task 10.2 tuning).
    # Task 10.4 relies on this default; the brief shows 5.0 but 2.0 was adopted post-tuning.
    dt_ms: float = 2.0
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
