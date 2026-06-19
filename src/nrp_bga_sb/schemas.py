"""Data schemas for NRP_BGA-SB.

Six Pydantic v2 BaseModel classes that form the foundation for the logger
(Task 0.5), replay (Task 0.6), and scorer (Task 0.7). All schemas support
lossless JSON round-trips via model_dump_json / model_validate_json.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --- EventType Enum ---


class EventType(str, Enum):
    """Canonical event vocabulary for a trial's event stream.

    Values are lowercase strings so they serialize to JSON identically
    to their name — EventType.go_cue serializes as "go_cue".
    """
    trial_start     = "trial_start"
    fixation_on     = "fixation_on"
    go_cue          = "go_cue"
    no_go_cue       = "no_go_cue"
    target_on_left  = "target_on_left"
    target_on_right = "target_on_right"
    stop_signal     = "stop_signal"
    evidence_change = "evidence_change"
    movement_onset  = "movement_onset"
    decision_commit = "decision_commit"
    movement_end    = "movement_end"
    trial_end       = "trial_end"


# --- TaskEvent ---


class TaskEvent(BaseModel):
    """A single typed event emitted by the task engine.

    sim_time: logical clock time in seconds.
    real_time: wall-clock time in seconds.
    payload: per-event flexible data; keys depend on event_type.
    """

    event_type: EventType
    sim_time: float           # logical clock time (s)
    real_time: float          # wall-clock time (s)
    trial_id: int
    payload: dict[str, Any]   # per-event flexible data


# --- ActionEvidence ---


class ActionEvidence(BaseModel):
    """Per-channel salience evidence arriving at the BG.

    Invariant: len(channel_salience) == n_channels.
    Enforced at construction time to catch mismatches at the system boundary.

    stop_signal_present: True when a stop signal has already arrived before
    the current decision point (stop-signal task only; False for all other tasks).
    This flag surfaces the stop_signal event from trial_log.events in a form
    the policy can act on without scanning the event list.
    """

    sim_time: float
    trial_id: int
    n_channels: int                 # number of action channels
    channel_salience: list[float]   # per-channel salience, len == n_channels
    stop_signal_present: bool = False  # True only on stop trials where SSD < decision_point

    @model_validator(mode="after")
    def _check_salience_length(self) -> ActionEvidence:
        # Trigger: channel_salience length does not match n_channels declaration.
        # Why: a mismatch signals a wiring or configuration error that must be
        #      caught immediately rather than silently propagating through BG logic.
        # Outcome: ValidationError raised; caller must fix upstream data.
        if len(self.channel_salience) != self.n_channels:
            raise ValueError(
                f"channel_salience has {len(self.channel_salience)} entries "
                f"but n_channels is {self.n_channels}"
            )
        return self


# --- BGDecision ---


class BGDecision(BaseModel):
    """Output decision emitted by the basal ganglia.

    selected_channel: index of the winning action channel; -1 if none selected.
    decision_margin: salience gap between top-1 and top-2 channels.
    selection_latency: time from BG input receipt to this decision (s).
    """

    sim_time: float
    trial_id: int
    selected_channel: int = Field(ge=-1)  # -1 if no channel selected
    decision_margin: float           # salience gap between top-1 and top-2
    suppression_vector: list[float]  # suppression applied to each channel
    channel_activations: list[float] # raw BG output activations per channel
    selection_latency: float         # time from BG input receipt to decision (s)


# --- MotorCommand ---


class MotorCommand(BaseModel):
    """Descending motor command after thalamic gating.

    gate_gain: multiplicative gate modulation factor, constrained to [0.0, 1.0].
    gate_state: coarse categorical summary ("open" | "closed" | "partial").
    """

    sim_time: float
    trial_id: int
    command: list[float]   # descending motor command vector
    gate_state: Literal["open", "closed", "partial"]
    gate_gain: float       # gate modulation factor in [0.0, 1.0]

    @field_validator("gate_gain")
    @classmethod
    def _check_gate_gain_range(cls, v: float) -> float:
        # Trigger: gate_gain outside [0.0, 1.0].
        # Why: gate_gain is a multiplicative modulation factor; values outside
        #      the unit interval are physically meaningless and indicate a
        #      model misconfiguration.
        # Outcome: ValidationError raised at construction; caller must fix the
        #          thalamic gate model.
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"gate_gain must be in [0.0, 1.0], got {v}"
            )
        return v


# --- TrialLog ---


class TrialLog(BaseModel):
    """Full per-trial event log.

    Captures every timing, decision, and movement datum for a single trial.
    Required fields carry no default; optional fields (unknown at trial start)
    default to None. List fields default to empty lists so callers can append
    incrementally during the trial.

    JSON round-trip: model_dump_json() -> model_validate_json() is lossless,
    which is required by the Task 0.6 replay system.
    """

    # --- Identity ---
    trial_id: int
    seed: int
    task_type: Literal["go_nogo", "two_choice", "stop_signal", "change_of_mind"]
    cue_identity: str
    cue_onset_time: float

    # --- BG timing (filled in during the trial) ---
    bg_input_receive_time: float | None = None
    bg_output_emit_time: float | None = None
    bg_selected_channel: int | None = None
    bg_channel_activations: list[float] = []

    # --- Thalamic relay ---
    thalamic_relay_time: float | None = None
    thalamic_release_time: float | None = None

    # --- Motor execution ---
    motor_command_series: list[MotorCommand] = []
    movement_onset_time: float | None = None
    endpoint_trajectory: list[list[float]] = []   # list of (x, y, ...) positions
    endpoint_error: float | None = None

    # --- Outcome ---
    success: bool | None = None
    failure_mode: str | None = None

    # --- System metrics ---
    sim_runtime: float = 0.0
    real_time_factor: float = 0.0
    message_counts: dict[str, int] = {}
    dropped_message_counts: dict[str, int] = {}

    # --- Full event stream ---
    events: list[TaskEvent] = []   # complete per-trial event stream


# --- Metrics ---


class Metrics(BaseModel):
    """Aggregated per-condition outputs computed offline from TrialLog files.

    Optional rate/timing fields are None when the corresponding task type was
    not run in this condition (e.g., stop_success_rate is None for go_nogo).
    """

    condition_id: str
    bg_frequency_hz: float
    n_trials: int
    reaction_time_mean: float | None = None
    reaction_time_std: float | None = None
    wrong_action_rate: float | None = None
    # two_choice engine: fraction of trials with failure_mode == "wrong_target"
    wrong_target_rate: float | None = None
    false_alarm_rate: float | None = None
    stop_success_rate: float | None = None
    switch_success_rate: float | None = None
