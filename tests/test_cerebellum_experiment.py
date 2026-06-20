# tests/test_cerebellum_experiment.py
import json

from experiments.cerebellum_adaptation import format_report, run_sweep, save_results


def test_run_sweep_covers_on_and_off():
    results = run_sweep(n_trials=20, seeds=[1])
    # 5 frequencies x 2 (on/off) x 1 seed
    assert len(results) == 10
    assert any(r.cerebellum_enabled for r in results)
    assert any(not r.cerebellum_enabled for r in results)


def test_m9_acceptance_onset_invariant_and_accuracy_improves():
    results = run_sweep(n_trials=30, seeds=[7])
    by_key = {(r.frequency_hz, r.cerebellum_enabled): r for r in results}
    for freq in (5.0, 10.0, 20.0, 40.0, 80.0):
        on = by_key[(freq, True)]
        off = by_key[(freq, False)]
        # (b) BG-frequency selection signature unchanged by the cerebellum
        assert on.movement_onset_rate == off.movement_onset_rate
        # (a) accuracy improves wherever movement actually occurs
        if on.movement_onset_rate > 0.0:
            assert on.mean_endpoint_deviation < off.mean_endpoint_deviation


def test_save_results_round_trip(tmp_path):
    results = run_sweep(n_trials=20, seeds=[1])
    out = tmp_path / "cb.json"
    save_results(results, str(out))
    loaded = json.loads(out.read_text())
    assert len(loaded) == len(results)
    assert "movement_onset_rate" in loaded[0]


def test_format_report_is_nonempty_string():
    results = run_sweep(n_trials=20, seeds=[1])
    report = format_report(results)
    assert isinstance(report, str)
    assert "5.0" in report
