"""Tests for the frequency sweep experiment runner (Task 5.3).

These tests verify the experiment script's helper functions in isolation —
not the full 900-condition sweep, which is validated by running the script.
"""

from __future__ import annotations

# Import the experiment module functions directly.
# The script uses __file__-relative paths; helpers are importable as-is.
import importlib.util
import json
from pathlib import Path

import pytest

from nrp_bga_sb.sweep import SweepConditionResult


def _import_experiment():
    """Import experiments/frequency_sweep.py as a module without executing main()."""
    spec_path = Path(__file__).parent.parent / "experiments" / "frequency_sweep.py"
    spec = importlib.util.spec_from_file_location("frequency_sweep", spec_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_exp = _import_experiment()


# --- Module-level constants ---


def test_frequencies_are_five():
    assert len(_exp.FREQUENCIES_HZ) == 5
    assert _exp.FREQUENCIES_HZ == [10.0, 20.0, 40.0, 80.0, 160.0]


def test_conflict_levels_are_three():
    assert _exp.CONFLICT_LEVELS == ["low", "medium", "high"]


def test_paradigms_are_two():
    assert _exp.PARADIGMS == ["go_nogo", "two_choice"]


def test_n_seeds_is_30():
    assert _exp.N_SEEDS == 30


def test_n_trials_is_30():
    assert _exp.N_TRIALS == 30


def test_total_conditions_is_900():
    total = (
        len(_exp.FREQUENCIES_HZ) * len(_exp.CONFLICT_LEVELS) * len(_exp.PARADIGMS) * _exp.N_SEEDS
    )
    assert total == 900


def test_results_path_in_results_dir():
    # results/ directory must be a sibling of experiments/, not data/
    assert _exp.RESULTS_PATH.parent.name == "results"
    assert _exp.RESULTS_PATH.name == "frequency_sweep_results.json"


# --- save_results ---


def test_save_results_writes_json_array(tmp_path):
    """save_results must write a valid JSON array, one dict per condition."""
    result = SweepConditionResult(
        frequency_hz=40.0,
        conflict_level="low",  # type: ignore[arg-type]
        paradigm="go_nogo",  # type: ignore[arg-type]
        seed=0,
        n_trials=10,
        reaction_time_mean=None,
        wrong_action_rate=0.0,
        wrong_target_rate=0.0,
        false_alarm_rate=0.0,
        miss_rate=0.1,
        timeout_rate=None,
        go_success_rate=0.9,
        bg_commitment_latency_mean=0.1,
        bg_commitment_latency_std=None,
    )
    out_path = tmp_path / "out.json"
    _exp.save_results([result], out_path)

    data = json.loads(out_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["frequency_hz"] == 40.0
    assert data[0]["conflict_level"] == "low"
    assert data[0]["paradigm"] == "go_nogo"
    assert data[0]["miss_rate"] == pytest.approx(0.1)


def test_save_results_creates_parent_dir(tmp_path):
    """save_results must create parent directories if absent."""
    out_path = tmp_path / "nested" / "dir" / "results.json"
    _exp.save_results([], out_path)
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == []


def test_save_results_round_trip(tmp_path):
    """Saved JSON must deserialise back to identical SweepConditionResult."""
    original = SweepConditionResult(
        frequency_hz=80.0,
        conflict_level="medium",  # type: ignore[arg-type]
        paradigm="two_choice",  # type: ignore[arg-type]
        seed=7,
        n_trials=30,
        reaction_time_mean=0.25,
        wrong_action_rate=0.0,
        wrong_target_rate=0.05,
        false_alarm_rate=None,
        miss_rate=None,
        timeout_rate=0.1,
        go_success_rate=None,
        bg_commitment_latency_mean=0.12,
        bg_commitment_latency_std=0.03,
    )
    out_path = tmp_path / "rt.json"
    _exp.save_results([original], out_path)

    loaded = json.loads(out_path.read_text())
    restored = SweepConditionResult(**loaded[0])
    assert restored == original


# --- run_sweep (small reduced run) ---


def test_run_sweep_reduced():
    """run_sweep with patched constants returns correct number of results."""
    # Temporarily replace constants to make sweep fast
    orig_freqs = _exp.FREQUENCIES_HZ
    orig_conflicts = _exp.CONFLICT_LEVELS
    orig_paradigms = _exp.PARADIGMS
    orig_seeds = _exp.N_SEEDS
    orig_trials = _exp.N_TRIALS

    _exp.FREQUENCIES_HZ = [10.0, 40.0]
    _exp.CONFLICT_LEVELS = ["low"]
    _exp.PARADIGMS = ["go_nogo"]
    _exp.N_SEEDS = 2
    _exp.N_TRIALS = 5

    try:
        results = _exp.run_sweep()
    finally:
        _exp.FREQUENCIES_HZ = orig_freqs
        _exp.CONFLICT_LEVELS = orig_conflicts
        _exp.PARADIGMS = orig_paradigms
        _exp.N_SEEDS = orig_seeds
        _exp.N_TRIALS = orig_trials

    # 2 freqs × 1 conflict × 1 paradigm × 2 seeds = 4
    assert len(results) == 4
    assert all(isinstance(r, SweepConditionResult) for r in results)


# --- run_repro_check ---


def test_run_repro_check_passes():
    """Repro check must pass: same seed → same result on any two runs."""
    orig_freqs = _exp.REPRO_FREQS
    orig_seeds = _exp.REPRO_SEEDS
    orig_trials = _exp.N_TRIALS
    orig_paradigms = _exp.PARADIGMS
    orig_conflicts = _exp.CONFLICT_LEVELS

    _exp.REPRO_FREQS = [40.0]
    _exp.REPRO_SEEDS = [0]
    _exp.N_TRIALS = 5
    _exp.PARADIGMS = ["go_nogo"]
    _exp.CONFLICT_LEVELS = ["low"]

    try:
        pass_a, pass_b = _exp.run_repro_check()
    finally:
        _exp.REPRO_FREQS = orig_freqs
        _exp.REPRO_SEEDS = orig_seeds
        _exp.N_TRIALS = orig_trials
        _exp.PARADIGMS = orig_paradigms
        _exp.CONFLICT_LEVELS = orig_conflicts

    from nrp_bga_sb.stats import reproducibility_check

    assert reproducibility_check(pass_a, pass_b) is True


def test_run_repro_check_same_length():
    """pass_a and pass_b must have the same number of conditions."""
    orig_freqs = _exp.REPRO_FREQS
    orig_seeds = _exp.REPRO_SEEDS
    orig_trials = _exp.N_TRIALS
    orig_paradigms = _exp.PARADIGMS
    orig_conflicts = _exp.CONFLICT_LEVELS

    _exp.REPRO_FREQS = [10.0]
    _exp.REPRO_SEEDS = [0, 1]
    _exp.N_TRIALS = 5
    _exp.PARADIGMS = ["go_nogo"]
    _exp.CONFLICT_LEVELS = ["low"]

    try:
        pass_a, pass_b = _exp.run_repro_check()
    finally:
        _exp.REPRO_FREQS = orig_freqs
        _exp.REPRO_SEEDS = orig_seeds
        _exp.N_TRIALS = orig_trials
        _exp.PARADIGMS = orig_paradigms
        _exp.CONFLICT_LEVELS = orig_conflicts

    assert len(pass_a) == len(pass_b)
    assert len(pass_a) == 2  # 1 freq × 2 seeds × 1 conflict × 1 paradigm
