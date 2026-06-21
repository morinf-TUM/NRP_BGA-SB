import pytest
from visuals.data_loader import (
    load_frequency_sweep,
    load_perturbation_gonogo,
    load_perturbation_stopsignal,
    load_cerebellum_results,
    load_bg_validation,
    load_opensim_gonogo,
)

def test_frequency_sweep_shape():
    data = load_frequency_sweep()
    assert len(data) == 900
    assert "frequency_hz" in data[0]
    assert "go_success_rate" in data[0]

def test_perturbation_gonogo_shape():
    data = load_perturbation_gonogo()
    assert len(data) == 85
    assert "perturbation_type" in data[0]
    assert "go_success_rate" in data[0]
    assert "bg_commitment_latency_mean" in data[0]

def test_perturbation_stopsignal_shape():
    data = load_perturbation_stopsignal()
    assert len(data) == 85
    assert "stop_failure_rate" in data[0]

def test_cerebellum_shape():
    data = load_cerebellum_results()
    assert len(data) == 50
    assert "cerebellum_enabled" in data[0]
    assert "endpoint_deviation_by_trial" in data[0]

def test_bg_validation_shape():
    data = load_bg_validation()
    assert len(data) == 3
    assert "conflict_level" in data[0]
    assert "mean_selection_latency_ms" in data[0]

def test_opensim_gonogo_shape():
    data = load_opensim_gonogo()
    assert len(data) == 5
    assert "opensim_movement_onset_rate" in data[0]
