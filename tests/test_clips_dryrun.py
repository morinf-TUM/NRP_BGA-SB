from visuals.clips import (
    write_bridge_frames,
    write_cerebellum_frames,
    write_interpretations_frames,
    write_perturbation_frames,
    write_threshold_frames,
)
from visuals.data_loader import load_perturbation_gonogo, load_perturbation_stopsignal
from visuals.trajectory_gen import (
    generate_cerebellum_trajectories,
    generate_threshold_trajectories,
)


def test_threshold_dryrun_returns_positive_count():
    trials = generate_threshold_trajectories()
    n = write_threshold_frames(trials, None, dry_run=True)
    assert n > 0
    assert isinstance(n, int)


def test_cerebellum_dryrun_returns_positive_count():
    trials = generate_cerebellum_trajectories(n_trials=5)
    n = write_cerebellum_frames(trials, None, dry_run=True)
    assert n > 0


def test_perturbation_dryrun_returns_positive_count():
    gonogo  = load_perturbation_gonogo()
    stopsig = load_perturbation_stopsignal()
    n = write_perturbation_frames(gonogo, stopsig, None, dry_run=True)
    assert n > 0


def test_interpretations_dryrun_returns_positive_count():
    n = write_interpretations_frames(None, dry_run=True)
    assert n > 0


def test_bridge_dryrun_returns_n_frames():
    n = write_bridge_frames("Test text", None, n_frames=60, dry_run=True)
    assert n == 60


def test_threshold_writes_pngs(tmp_path):
    frames_dir = tmp_path / "threshold"
    trials = generate_threshold_trajectories()
    n = write_threshold_frames(trials, frames_dir, dry_run=False)
    assert n > 0
    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == n
    # Frame names are zero-padded 4-digit
    assert any(p.stem == "0000" for p in pngs)
