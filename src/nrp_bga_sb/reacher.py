"""2D point-mass kinematic reacher: trajectory simulation (Tasks 6.1, 8.2).

Converts a ClosedLoopPolicy motor_command_series into a minimum-jerk 2D
position time-series. Phase 6 uses a single motor command per trial. Phase 8
adds simulate_change_of_mind() for two-phase reversal trajectories.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, model_validator

from nrp_bga_sb.cerebellum import Cerebellum
from nrp_bga_sb.perturbation_plant import VisuomotorRotation, signed_angle
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
    def _check_target_count(self) -> ReacherConfig:
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
        if len(last_cmd.command) != self.config.n_channels:
            raise ValueError(
                f"Command has {len(last_cmd.command)} channels but "
                f"config expects {self.config.n_channels}"
            )
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

    def simulate_change_of_mind(
        self,
        motor_commands: list[MotorCommand],
        pre_switch_onset_ms: float,
        switch_time_ms: float,
        total_duration_ms: float = 1500.0,
    ) -> ReacherTrajectory:
        """Simulate a two-phase change-of-mind trajectory.

        Phase 1 (pre_switch_onset_ms → switch_time_ms):
            Minimum-jerk movement from origin toward motor_commands[0]'s target
            (scaled by gate_gain). Uses selected_channel from argmax of command[0].

        Phase 2 (switch_time_ms → end):
            Minimum-jerk movement from the switch_position (position at switch_time_ms)
            toward motor_commands[1]'s target (scaled by gate_gain).

        Args:
            motor_commands:       Must have exactly 2 entries: [pre_switch, post_switch].
                                  Both must have gate_state != "closed".
            pre_switch_onset_ms:  Absolute ms from trial start when pre-switch movement began.
                                  Extract from motor_commands[0].sim_time * 1000.
            switch_time_ms:       Absolute ms from trial start of the evidence_change event.
            total_duration_ms:    Total simulation window length in ms.

        Returns:
            ReacherTrajectory with selected_channel = post-switch channel (motor_commands[1]).
            onset_time_ms is set to pre_switch_onset_ms.

        Raises:
            ValueError if len(motor_commands) != 2.
            ValueError if either command has gate_state == "closed".
        """
        # --- Validation ---
        # Trigger: caller passes wrong number of commands.
        # Why: this method models exactly pre-switch + post-switch; any other count
        #      means the caller has wrong data and we must not silently extrapolate.
        # Outcome: fail fast so the caller can fix the upstream data.
        if len(motor_commands) != 2:
            raise ValueError(
                f"simulate_change_of_mind requires exactly 2 motor commands, "
                f"got {len(motor_commands)}"
            )
        for i, cmd in enumerate(motor_commands):
            if cmd.gate_state == "closed":
                raise ValueError(
                    f"motor_commands[{i}] has gate_state='closed'; "
                    "change-of-mind trajectory requires open or partial gate on both commands"
                )

        n_steps = int(round(total_duration_ms / self.config.dt_ms)) + 1
        times_ms = [i * self.config.dt_ms for i in range(n_steps)]

        # --- Phase 1 setup ---
        cmd0 = motor_commands[0]
        ch0 = int(np.argmax(cmd0.command))
        tx0, ty0 = self.config.target_positions[ch0]
        # gate_gain ∈ [0, 1] scales the endpoint; partial gate → short of target.
        ex0, ey0 = tx0 * cmd0.gate_gain, ty0 * cmd0.gate_gain

        # Reuse the same movement duration for both phases so the speed profile
        # is identical in each phase (minimum-jerk over movement_duration_ms).
        T = self.config.movement_duration_ms

        # --- Phase 2 setup ---
        cmd1 = motor_commands[1]
        ch1 = int(np.argmax(cmd1.command))
        tx1, ty1 = self.config.target_positions[ch1]
        ex1, ey1 = tx1 * cmd1.gate_gain, ty1 * cmd1.gate_gain

        # --- Trajectory integration ---
        # switch_pos tracks the arm position continuously during phase 1 so the
        # last assignment captures the position exactly at switch_time_ms.
        positions_xy: list[list[float]] = []
        switch_pos = [0.0, 0.0]

        for t_ms in times_ms:
            if t_ms < pre_switch_onset_ms:
                # Before movement onset: hold at origin.
                positions_xy.append([0.0, 0.0])
            elif t_ms <= switch_time_ms:
                # Phase 1: minimum-jerk toward pre-switch scaled endpoint.
                s = _minimum_jerk_scalar(t_ms - pre_switch_onset_ms, T)
                pos = [ex0 * s, ey0 * s]
                positions_xy.append(pos)
                # Trigger: continuously update switch_pos during phase 1.
                # Why: the loop condition uses <=, so the last assignment when
                #      t_ms == switch_time_ms captures the handoff position.
                # Outcome: phase 2 begins exactly from switch_pos, not from origin.
                switch_pos = pos
            else:
                # Phase 2: minimum-jerk from switch_pos toward post-switch endpoint.
                # The movement vector is (ex1 - switch_pos[0], ey1 - switch_pos[1]);
                # s interpolates from 0 (switch_pos) to 1 (ex1, ey1) over T ms.
                t_post = t_ms - switch_time_ms
                s = _minimum_jerk_scalar(t_post, T)
                px = switch_pos[0] + (ex1 - switch_pos[0]) * s
                py = switch_pos[1] + (ey1 - switch_pos[1]) * s
                positions_xy.append([px, py])

        return ReacherTrajectory(
            times_ms=times_ms,
            positions_xy=positions_xy,
            onset_time_ms=pre_switch_onset_ms,
            selected_channel=ch1,  # post-switch channel is the final committed direction
        )

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
