from nrp_bga_sb.cerebellum_sweep import (
    FREQUENCIES_HZ,
    CerebellumSweepResult,
    run_cerebellum_condition,
)


def test_frequencies_match_prior_sweeps():
    assert FREQUENCIES_HZ == [5.0, 10.0, 20.0, 40.0, 80.0]


def test_condition_returns_result_type():
    r = run_cerebellum_condition(40.0, n_trials=20, seed=1)
    assert isinstance(r, CerebellumSweepResult)
    assert r.frequency_hz == 40.0
    assert r.cerebellum_enabled is True


def test_onset_rate_identical_cerebellum_on_vs_off():
    # The guard: the cerebellum must NOT change which trials produce a movement.
    for freq in (5.0, 10.0, 40.0):
        on = run_cerebellum_condition(freq, n_trials=30, seed=7, cerebellum_enabled=True)
        off = run_cerebellum_condition(freq, n_trials=30, seed=7, cerebellum_enabled=False)
        assert on.movement_onset_rate == off.movement_onset_rate
        assert on.go_success_rate == off.go_success_rate


def test_low_frequency_has_no_movement():
    r = run_cerebellum_condition(5.0, n_trials=30, seed=7)
    assert r.movement_onset_rate == 0.0


def test_cerebellum_reduces_endpoint_deviation_when_moving():
    on = run_cerebellum_condition(40.0, n_trials=40, seed=3, cerebellum_enabled=True)
    off = run_cerebellum_condition(40.0, n_trials=40, seed=3, cerebellum_enabled=False)
    assert on.movement_onset_rate > 0.0  # sanity: trials move at 40 Hz
    assert on.mean_endpoint_deviation < off.mean_endpoint_deviation


def test_adaptation_learns_nonzero_theta_when_moving():
    r = run_cerebellum_condition(40.0, n_trials=40, seed=3, perturbation_deg=30.0)
    assert r.final_theta_hat > 0.1  # learned a counter-rotation


def test_endpoint_deviation_decays_over_trials():
    r = run_cerebellum_condition(40.0, n_trials=40, seed=3, perturbation_deg=30.0)
    series = r.endpoint_deviation_by_trial
    assert len(series) >= 10
    # later trials are more accurate than the first few (learning curve)
    assert sum(series[-3:]) / 3 < sum(series[:3]) / 3
