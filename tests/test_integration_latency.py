from experiments.nrp_integration_latency import first_release_index


def test_first_release_index_finds_first_open_gate():
    trace = [
        {"motor": None},
        {"motor": {"gate_state": "closed", "command": [0.0, 0.0]}},
        {"motor": {"gate_state": "partial", "command": [0.3, 0.0]}},
        {"motor": {"gate_state": "open", "command": [1.0, 0.0]}},
    ]
    assert first_release_index(trace) == 2


def test_first_release_index_none_when_never_open():
    trace = [{"motor": None}, {"motor": {"gate_state": "closed", "command": [0.0, 0.0]}}]
    assert first_release_index(trace) is None


def test_first_release_index_ignores_open_with_zero_command():
    # Mirrors nrp/score.py: an open gate with a zero command is NOT a release.
    trace = [{"motor": {"gate_state": "open", "command": [0.0, 0.0]}}]
    assert first_release_index(trace) is None
