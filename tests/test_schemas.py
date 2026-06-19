"""Tests for nrp_bga_sb.schemas — all six Pydantic v2 schema classes.

Coverage:
  1. Construction  — each schema instantiates cleanly with valid data.
  2. Validation    — ActionEvidence and MotorCommand raise ValidationError on
                     constraint violations.
  3. JSON round-trip — TrialLog with nested MotorCommand and TaskEvent entries
                        serialises and reconstructs identically.
  4. Optional fields — TrialLog can be built with only required fields; all
                        optional fields default correctly.
"""

import pytest
from pydantic import ValidationError

from nrp_bga_sb.schemas import (
    ActionEvidence,
    BGDecision,
    Metrics,
    MotorCommand,
    TaskEvent,
    TrialLog,
)

# --- 1. Construction ---


def test_task_event_construction():
    evt = TaskEvent(
        event_type="trial_start",
        sim_time=0.0,
        real_time=1.234,
        trial_id=1,
        payload={"foo": "bar"},
    )
    assert evt.event_type == "trial_start"
    assert evt.trial_id == 1


def test_action_evidence_construction():
    ae = ActionEvidence(
        sim_time=0.05,
        trial_id=1,
        n_channels=3,
        channel_salience=[0.1, 0.5, 0.3],
    )
    assert ae.n_channels == 3
    assert len(ae.channel_salience) == 3


def test_bg_decision_construction():
    bgd = BGDecision(
        sim_time=0.1,
        trial_id=1,
        selected_channel=1,
        decision_margin=0.2,
        suppression_vector=[0.9, 0.1, 0.8],
        channel_activations=[0.3, 0.8, 0.2],
        selection_latency=0.025,
    )
    assert bgd.selected_channel == 1
    assert bgd.selection_latency == 0.025


def test_motor_command_construction():
    mc = MotorCommand(
        sim_time=0.15,
        trial_id=1,
        command=[1.0, 0.5, -0.2],
        gate_state="open",
        gate_gain=1.0,
    )
    assert mc.gate_state == "open"
    assert mc.gate_gain == 1.0


def test_trial_log_full_construction():
    mc = MotorCommand(
        sim_time=0.15, trial_id=1, command=[1.0], gate_state="open", gate_gain=0.8
    )
    evt = TaskEvent(
        event_type="go_cue", sim_time=0.0, real_time=0.0, trial_id=1, payload={}
    )
    log = TrialLog(
        trial_id=1,
        seed=42,
        task_type="go_nogo",
        cue_identity="go",
        cue_onset_time=0.0,
        bg_input_receive_time=0.05,
        bg_output_emit_time=0.1,
        bg_selected_channel=0,
        bg_channel_activations=[0.9, 0.1],
        thalamic_relay_time=0.11,
        thalamic_release_time=0.12,
        motor_command_series=[mc],
        movement_onset_time=0.2,
        endpoint_trajectory=[[0.0, 0.0], [0.1, 0.05]],
        endpoint_error=0.003,
        success=True,
        failure_mode=None,
        sim_runtime=1.0,
        real_time_factor=0.95,
        message_counts={"task": 10},
        dropped_message_counts={},
        events=[evt],
    )
    assert log.trial_id == 1
    assert log.success is True


def test_metrics_construction():
    m = Metrics(
        condition_id="go_nogo_40hz",
        bg_frequency_hz=40.0,
        n_trials=200,
        reaction_time_mean=0.35,
        reaction_time_std=0.05,
        wrong_action_rate=0.02,
    )
    assert m.bg_frequency_hz == 40.0
    assert m.stop_success_rate is None   # not applicable to go_nogo


# --- 2. Validation ---


def test_action_evidence_salience_length_mismatch():
    # Trigger: n_channels=3 but only 2 salience values provided.
    # Expected: ValidationError raised immediately at construction.
    with pytest.raises(ValidationError):
        ActionEvidence(
            sim_time=0.0,
            trial_id=1,
            n_channels=3,
            channel_salience=[0.1, 0.5],   # one element short
        )


def test_action_evidence_salience_length_excess():
    # Trigger: n_channels=2 but 3 salience values provided.
    with pytest.raises(ValidationError):
        ActionEvidence(
            sim_time=0.0,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.1, 0.5, 0.3],
        )


def test_motor_command_gate_gain_above_one():
    # Trigger: gate_gain > 1.0.
    with pytest.raises(ValidationError):
        MotorCommand(
            sim_time=0.0, trial_id=1, command=[1.0], gate_state="open", gate_gain=1.1
        )


def test_motor_command_gate_gain_below_zero():
    # Trigger: gate_gain < 0.0.
    with pytest.raises(ValidationError):
        MotorCommand(
            sim_time=0.0,
            trial_id=1,
            command=[1.0],
            gate_state="closed",
            gate_gain=-0.01,
        )


def test_motor_command_gate_gain_boundary_values():
    # Boundary values 0.0 and 1.0 must both be accepted.
    mc_zero = MotorCommand(
        sim_time=0.0, trial_id=1, command=[0.0], gate_state="closed", gate_gain=0.0
    )
    mc_one = MotorCommand(
        sim_time=0.0, trial_id=1, command=[1.0], gate_state="open", gate_gain=1.0
    )
    assert mc_zero.gate_gain == 0.0
    assert mc_one.gate_gain == 1.0


# --- 3. JSON round-trip ---


def test_trial_log_json_round_trip():
    """TrialLog with nested MotorCommand and TaskEvent must survive a full
    model_dump_json -> model_validate_json cycle without data loss."""
    mc = MotorCommand(
        sim_time=0.15, trial_id=7, command=[0.5, -0.3], gate_state="partial", gate_gain=0.6
    )
    evt = TaskEvent(
        event_type="movement_onset",
        sim_time=0.2,
        real_time=1.5,
        trial_id=7,
        payload={"velocity": 0.42},
    )
    log = TrialLog(
        trial_id=7,
        seed=99,
        task_type="two_choice",
        cue_identity="left",
        cue_onset_time=0.0,
        bg_selected_channel=0,
        motor_command_series=[mc],
        events=[evt],
        endpoint_trajectory=[[0.0, 0.0], [0.2, 0.1]],
    )

    json_str = log.model_dump_json()
    reconstructed = TrialLog.model_validate_json(json_str)
    assert reconstructed == log


# --- 4. Optional fields ---


def test_trial_log_minimal_construction():
    """TrialLog with only required fields must instantiate; all optional fields
    must carry the correct defaults."""
    log = TrialLog(
        trial_id=0,
        seed=0,
        task_type="stop_signal",
        cue_identity="stop",
        cue_onset_time=0.5,
    )
    assert log.bg_input_receive_time is None
    assert log.bg_output_emit_time is None
    assert log.bg_selected_channel is None
    assert log.bg_channel_activations == []
    assert log.thalamic_relay_time is None
    assert log.thalamic_release_time is None
    assert log.motor_command_series == []
    assert log.movement_onset_time is None
    assert log.endpoint_trajectory == []
    assert log.endpoint_error is None
    assert log.success is None
    assert log.failure_mode is None
    assert log.sim_runtime == 0.0
    assert log.real_time_factor == 0.0
    assert log.message_counts == {}
    assert log.dropped_message_counts == {}
    assert log.events == []
