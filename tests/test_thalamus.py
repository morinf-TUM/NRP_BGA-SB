"""Tests for the thalamic gate adapter (Task 4.2)."""

from __future__ import annotations

import pytest

from nrp_bga_sb.schemas import BGDecision, MotorCommand
from nrp_bga_sb.thalamus import ThalamusConfig, ThalamusGate

# --- Fixtures ---


def _make_bg_decision(
    selected_channel: int = 0,
    decision_margin: float = 0.4,
    n_channels: int = 2,
) -> BGDecision:
    return BGDecision(
        sim_time=0.3,
        trial_id=1,
        selected_channel=selected_channel,
        decision_margin=decision_margin,
        suppression_vector=[0.0] * n_channels,
        channel_activations=[0.5] * n_channels,
        selection_latency=0.02,
    )


@pytest.fixture()
def gate() -> ThalamusGate:
    return ThalamusGate(ThalamusConfig())


# --- ThalamusConfig validation ---


def test_config_defaults() -> None:
    cfg = ThalamusConfig()
    assert cfg.margin_threshold == 0.05
    assert cfg.full_open_threshold == 0.30
    assert cfg.n_channels == 2


def test_margin_threshold_must_be_nonnegative() -> None:
    with pytest.raises(Exception):
        ThalamusConfig(margin_threshold=-0.1)


def test_full_open_must_be_nonnegative() -> None:
    with pytest.raises(Exception):
        ThalamusConfig(full_open_threshold=-0.1)


def test_full_open_must_be_at_least_margin_threshold() -> None:
    with pytest.raises(Exception):
        ThalamusConfig(margin_threshold=0.2, full_open_threshold=0.1)


def test_equal_thresholds_are_valid() -> None:
    # margin_threshold == full_open_threshold is degenerate but allowed.
    cfg = ThalamusConfig(margin_threshold=0.1, full_open_threshold=0.1)
    assert cfg.margin_threshold == cfg.full_open_threshold


def test_n_channels_must_be_positive() -> None:
    with pytest.raises(Exception):
        ThalamusConfig(n_channels=0)


# --- Gate-closed cases ---


def test_gate_closed_when_no_channel_selected(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=-1, decision_margin=0.9)
    cmd = gate(dec)
    assert cmd.gate_state == "closed"
    assert cmd.gate_gain == 0.0
    assert cmd.command == [0.0, 0.0]


def test_gate_closed_when_margin_below_threshold(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.01)
    cmd = gate(dec)
    assert cmd.gate_state == "closed"
    assert cmd.gate_gain == 0.0


def test_gate_closed_at_exactly_zero_margin(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.0)
    cmd = gate(dec)
    assert cmd.gate_state == "closed"


# --- Gate-partial cases ---


def test_gate_partial_when_margin_between_thresholds(gate: ThalamusGate) -> None:
    # ThalamusConfig defaults: margin_threshold=0.05, full_open=0.30
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.175)
    cmd = gate(dec)
    assert cmd.gate_state == "partial"
    assert 0.0 < cmd.gate_gain < 1.0


def test_gate_partial_gain_is_linearly_interpolated() -> None:
    cfg = ThalamusConfig(margin_threshold=0.0, full_open_threshold=1.0)
    gate = ThalamusGate(cfg)
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.5)
    cmd = gate(dec)
    assert cmd.gate_gain == pytest.approx(0.5)


def test_gate_closed_at_margin_threshold() -> None:
    # At exactly margin_threshold, gate closes. A margin at the boundary
    # is not enough to open — it is the threshold of closure.
    cfg = ThalamusConfig(margin_threshold=0.1, full_open_threshold=0.3)
    gate = ThalamusGate(cfg)
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.1)
    cmd = gate(dec)
    assert cmd.gate_state == "closed"
    assert cmd.gate_gain == pytest.approx(0.0)


# --- Gate-open cases ---


def test_gate_open_when_margin_above_full_open_threshold(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.5)
    cmd = gate(dec)
    assert cmd.gate_state == "open"
    assert cmd.gate_gain == pytest.approx(1.0)


def test_gate_open_at_exactly_full_open_threshold() -> None:
    cfg = ThalamusConfig(margin_threshold=0.05, full_open_threshold=0.3)
    gate = ThalamusGate(cfg)
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.3)
    cmd = gate(dec)
    assert cmd.gate_state == "open"
    assert cmd.gate_gain == pytest.approx(1.0)


# --- Command vector ---


def test_command_vector_has_correct_channel_open(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.5)
    cmd = gate(dec)
    assert cmd.command[0] == pytest.approx(1.0)
    assert cmd.command[1] == pytest.approx(0.0)


def test_command_vector_channel_1_open(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=1, decision_margin=0.5)
    cmd = gate(dec)
    assert cmd.command[0] == pytest.approx(0.0)
    assert cmd.command[1] == pytest.approx(1.0)


def test_command_vector_partial_gain_on_selected_channel() -> None:
    cfg = ThalamusConfig(margin_threshold=0.0, full_open_threshold=1.0)
    gate = ThalamusGate(cfg)
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.6)
    cmd = gate(dec)
    assert cmd.command[0] == pytest.approx(0.6)
    assert cmd.command[1] == pytest.approx(0.0)


def test_command_vector_all_zero_when_closed(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=-1, decision_margin=0.9)
    cmd = gate(dec)
    assert all(v == pytest.approx(0.0) for v in cmd.command)


# --- MotorCommand metadata ---


def test_motor_command_sim_time_copied_from_bg_decision(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.5)
    dec.sim_time = 0.42
    cmd = gate(dec)
    assert cmd.sim_time == pytest.approx(0.42)


def test_motor_command_trial_id_copied(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.5)
    dec.trial_id = 99
    cmd = gate(dec)
    assert cmd.trial_id == 99


def test_motor_command_is_valid_schema(gate: ThalamusGate) -> None:
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.5)
    cmd = gate(dec)
    assert isinstance(cmd, MotorCommand)
    assert 0.0 <= cmd.gate_gain <= 1.0


# --- Boundary: equal thresholds (degenerate "snap" gate) ---


def test_snap_gate_is_closed_when_thresholds_equal() -> None:
    # When margin_threshold == full_open_threshold, there is no partial range.
    # A margin at exactly the threshold is treated as closed (at the boundary).
    # To reach "open", margin must exceed the threshold.
    cfg = ThalamusConfig(margin_threshold=0.2, full_open_threshold=0.2)
    gate = ThalamusGate(cfg)
    dec = _make_bg_decision(selected_channel=0, decision_margin=0.2)
    cmd = gate(dec)
    # margin <= margin_threshold → "closed"
    assert cmd.gate_state == "closed"
    assert cmd.gate_gain == pytest.approx(0.0)
