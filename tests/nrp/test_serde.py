from nrp_bga_sb.schemas import ActionEvidence, BGDecision, MotorCommand
from nrp.serde import (
    evidence_to_dict, evidence_from_dict,
    decision_to_dict, decision_from_dict,
    motor_to_dict, motor_from_dict,
)


def test_evidence_roundtrip():
    ev = ActionEvidence(sim_time=0.1, trial_id=3, n_channels=2,
                        channel_salience=[0.6, 0.4], stop_signal_present=False)
    assert evidence_from_dict(evidence_to_dict(ev)) == ev


def test_decision_roundtrip():
    d = BGDecision(sim_time=0.1, trial_id=3, selected_channel=0, decision_margin=0.2,
                   suppression_vector=[0.0, 0.3], channel_activations=[0.8, 0.5],
                   selection_latency=0.013)
    assert decision_from_dict(decision_to_dict(d)) == d


def test_motor_roundtrip():
    m = MotorCommand(sim_time=0.1, trial_id=3, command=[1.0, 0.0],
                     gate_state="open", gate_gain=1.0)
    assert motor_from_dict(motor_to_dict(m)) == m
