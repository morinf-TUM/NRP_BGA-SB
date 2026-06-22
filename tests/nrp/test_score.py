from nrp.score import trace_to_outcome


def test_outcome_released():
    trace = [
        {"decision": {"selected_channel": 0, "sim_time": 0.1}, "motor": None},
        {"decision": {"selected_channel": 0, "sim_time": 0.12},
         "motor": {"command": [1.0, 0.0], "gate_state": "open", "gate_gain": 1.0,
                   "sim_time": 0.12}},
    ]
    out = trace_to_outcome(trace)
    assert out["motor_released"] is True
    assert out["selected_channel"] == 0
    assert out["first_release_time"] == 0.12


def test_outcome_missed():
    trace = [{"decision": {"selected_channel": -1, "sim_time": 0.1}, "motor": None}]
    out = trace_to_outcome(trace)
    assert out["motor_released"] is False
    assert out["first_release_time"] is None
