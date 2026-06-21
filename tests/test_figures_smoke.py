import pytest
from pathlib import Path
from visuals.figures import (
    fig_frequency_threshold,
    fig_perturbation_decomposition,
    fig_cerebellum_learning,
    fig_three_interpretations,
)
from visuals.trajectory_gen import generate_cerebellum_trajectories


def test_fig_frequency_threshold(tmp_path):
    out = fig_frequency_threshold(tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 10_000   # non-trivial file


def test_fig_perturbation_decomposition(tmp_path):
    out = fig_perturbation_decomposition(tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 10_000


def test_fig_cerebellum_learning(tmp_path):
    cereb_trials = generate_cerebellum_trajectories(n_trials=10)
    out = fig_cerebellum_learning(cereb_trials, tmp_path)
    assert out.exists()
    assert out.suffix == ".png"


def test_fig_three_interpretations(tmp_path):
    out = fig_three_interpretations(tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
