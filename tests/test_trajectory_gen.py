from visuals.trajectory_gen import (
    THRESHOLD_FREQUENCIES,
    generate_cerebellum_trajectories,
    generate_threshold_trajectories,
)


def test_threshold_returns_one_per_frequency():
    trials = generate_threshold_trajectories()
    assert len(trials) == len(THRESHOLD_FREQUENCIES)
    freqs = [t["frequency_hz"] for t in trials]
    assert freqs == THRESHOLD_FREQUENCIES

def test_threshold_5hz_is_miss():
    trials = generate_threshold_trajectories()
    trial_5 = next(t for t in trials if t["frequency_hz"] == 5)
    assert trial_5["gate_closed"] is True
    assert trial_5["selected_channel"] == -1
    # All positions should be [0.0, 0.0] for a miss
    assert all(p == [0.0, 0.0] for p in trial_5["positions_xy"])

def test_threshold_10hz_reaches_target():
    trials = generate_threshold_trajectories()
    trial_10 = next(t for t in trials if t["frequency_hz"] == 10)
    assert trial_10["gate_closed"] is False
    assert trial_10["selected_channel"] >= 0
    # Final position should be non-zero
    final = trial_10["positions_xy"][-1]
    assert final[0] ** 2 + final[1] ** 2 > 0.01

def test_threshold_trajectory_shape():
    trials = generate_threshold_trajectories()
    for t in trials:
        assert "times_ms" in t
        assert "positions_xy" in t
        assert len(t["times_ms"]) == len(t["positions_xy"])
        assert all(len(p) == 2 for p in t["positions_xy"])

def test_cerebellum_trajectories_count():
    trials = generate_cerebellum_trajectories(n_trials=10)
    # Only go-trials produce trajectories; with go_probability=1.0 all 10 are go
    assert len(trials) == 10

def test_cerebellum_trial_1_is_deflected():
    trials = generate_cerebellum_trajectories(n_trials=5)
    t = trials[0]
    assert t["is_go"] is True
    ep = t["endpoint_xy"]
    # At trial 1 theta_hat=0 → endpoint is at 30° rotation of (0,1) = (0.5, 0.866)
    assert abs(ep[0] - 0.5) < 0.05
    assert abs(ep[1] - 0.866) < 0.05

def test_cerebellum_theta_hat_increases():
    trials = generate_cerebellum_trajectories(n_trials=15)
    go_trials = [t for t in trials if t["is_go"]]
    assert len(go_trials) >= 2
    # VisuomotorRotation(rotation_deg=-30) rotates (0,1) CW to (0.5, 0.866).
    # signed_angle(desired, openloop) = -30° → theta_hat decreases (becomes
    # more negative) as the filter learns the -30° correction.
    # The magnitude of the correction grows, so abs(theta_hat) increases, but
    # the scalar value decreases (goes from ~-0.05 toward ~-0.52 rad).
    assert go_trials[-1]["theta_hat"] < go_trials[0]["theta_hat"]
