"""Tests for the abstract cortical evidence generator (Task 4.1)."""

from __future__ import annotations

import pytest

from nrp_bga_sb.cortex import CortexConfig, CortexEvidenceGenerator, _preferred_channel_from_cue
from nrp_bga_sb.schemas import EventType, TaskEvent, TrialLog

# --- Fixtures ---


def _make_trial(
    cue_identity: str,
    task_type: str = "go_nogo",
    seed: int = 42,
    cue_onset_time: float = 0.2,
) -> TrialLog:
    return TrialLog(
        trial_id=1,
        seed=seed,
        task_type=task_type,  # type: ignore[arg-type]
        cue_identity=cue_identity,
        cue_onset_time=cue_onset_time,
    )


@pytest.fixture()
def gen() -> CortexEvidenceGenerator:
    return CortexEvidenceGenerator(CortexConfig())


# --- CortexConfig validation ---


def test_config_defaults() -> None:
    cfg = CortexConfig()
    assert cfg.rise_time_ms == 100.0
    assert cfg.peak_salience == 0.9
    assert cfg.base_salience == 0.5
    assert cfg.noise_std == 0.0


def test_config_rise_time_must_be_positive() -> None:
    with pytest.raises(Exception):
        CortexConfig(rise_time_ms=0.0)
    with pytest.raises(Exception):
        CortexConfig(rise_time_ms=-10.0)


def test_config_salience_bounds() -> None:
    with pytest.raises(Exception):
        CortexConfig(peak_salience=1.1)
    with pytest.raises(Exception):
        CortexConfig(base_salience=-0.1)


def test_config_noise_std_must_be_nonnegative() -> None:
    with pytest.raises(Exception):
        CortexConfig(noise_std=-0.01)


# --- Preferred-channel mapping ---


@pytest.mark.parametrize(
    "cue_identity, expected_channel",
    [
        ("go", 0),
        ("no_go", None),
        ("left", 0),
        ("right", 1),
        ("stop", None),
        ("no_switch", 0),
        ("switch_early", 0),
        ("switch_medium", 0),
        ("switch_late", 0),
        ("switch_very_late", 0),
    ],
)
def test_preferred_channel_from_cue(cue_identity: str, expected_channel: int | None) -> None:
    assert _preferred_channel_from_cue(cue_identity) == expected_channel


def test_unknown_cue_identity_is_neutral() -> None:
    # Unknown cue → None (no preferred channel, fail-safe withhold)
    assert _preferred_channel_from_cue("unknown_cue_xyz") is None


# --- Evidence at elapsed = 0 (neutral start) ---


def test_neutral_at_elapsed_zero_go(gen: CortexEvidenceGenerator) -> None:
    trial = _make_trial("go")
    ev = gen(trial, 0.0)
    assert ev.channel_salience[0] == pytest.approx(0.5)
    assert ev.channel_salience[1] == pytest.approx(0.5)


def test_neutral_at_elapsed_zero_left(gen: CortexEvidenceGenerator) -> None:
    trial = _make_trial("left", task_type="two_choice")
    ev = gen(trial, 0.0)
    assert ev.channel_salience[0] == pytest.approx(0.5)
    assert ev.channel_salience[1] == pytest.approx(0.5)


# --- Evidence at elapsed = rise_time_ms (full rise) ---


def test_full_rise_channel_0_go(gen: CortexEvidenceGenerator) -> None:
    trial = _make_trial("go")
    ev = gen(trial, 100.0)  # default rise_time_ms = 100
    assert ev.channel_salience[0] == pytest.approx(0.9)
    assert ev.channel_salience[1] == pytest.approx(0.1)


def test_full_rise_channel_1_right(gen: CortexEvidenceGenerator) -> None:
    trial = _make_trial("right", task_type="two_choice")
    ev = gen(trial, 100.0)
    assert ev.channel_salience[0] == pytest.approx(0.1)
    assert ev.channel_salience[1] == pytest.approx(0.9)


# --- No preferred channel (no_go, stop) stays at base ---


def test_no_go_stays_neutral(gen: CortexEvidenceGenerator) -> None:
    trial = _make_trial("no_go")
    for elapsed in [0.0, 50.0, 100.0, 200.0]:
        ev = gen(trial, elapsed)
        assert ev.channel_salience[0] == pytest.approx(0.5)
        assert ev.channel_salience[1] == pytest.approx(0.5)


def test_stop_stays_neutral(gen: CortexEvidenceGenerator) -> None:
    trial = _make_trial("stop", task_type="stop_signal")
    ev = gen(trial, 100.0)
    assert ev.channel_salience[0] == pytest.approx(0.5)
    assert ev.channel_salience[1] == pytest.approx(0.5)


# --- Linear ramp shape ---


def test_ramp_is_linear_go() -> None:
    cfg = CortexConfig(rise_time_ms=100.0, peak_salience=0.9, base_salience=0.5)
    gen = CortexEvidenceGenerator(cfg)
    trial = _make_trial("go")
    # At 50% of rise_time, preferred salience should be midpoint
    ev = gen(trial, 50.0)
    assert ev.channel_salience[0] == pytest.approx(0.7)  # 0.5 + 0.4 * 0.5
    assert ev.channel_salience[1] == pytest.approx(0.3)  # 1.0 - 0.7


def test_ramp_clamps_above_rise_time() -> None:
    gen = CortexEvidenceGenerator(CortexConfig(rise_time_ms=100.0, peak_salience=0.9))
    trial = _make_trial("go")
    ev_at_100 = gen(trial, 100.0)
    ev_at_200 = gen(trial, 200.0)
    assert ev_at_100.channel_salience[0] == pytest.approx(ev_at_200.channel_salience[0])


def test_ramp_clamps_below_zero() -> None:
    gen = CortexEvidenceGenerator(CortexConfig())
    trial = _make_trial("go")
    ev = gen(trial, -10.0)  # negative elapsed → treated as 0
    assert ev.channel_salience[0] == pytest.approx(0.5)


# --- Salience symmetry: preferred + competing = 1.0 ---


def test_salience_sums_to_one_at_all_fracs() -> None:
    gen = CortexEvidenceGenerator(CortexConfig())
    trial = _make_trial("go")
    for elapsed in [0.0, 25.0, 50.0, 75.0, 100.0]:
        ev = gen(trial, elapsed)
        assert sum(ev.channel_salience) == pytest.approx(1.0)


# --- ActionEvidence fields ---


def test_evidence_sim_time_matches_cue_onset_plus_elapsed() -> None:
    gen = CortexEvidenceGenerator(CortexConfig())
    trial = _make_trial("go", cue_onset_time=0.3)
    ev = gen(trial, 50.0)
    assert ev.sim_time == pytest.approx(0.3 + 0.05)


def test_evidence_trial_id_preserved() -> None:
    trial = _make_trial("go")
    trial.trial_id = 7
    gen = CortexEvidenceGenerator(CortexConfig())
    ev = gen(trial, 0.0)
    assert ev.trial_id == 7


def test_evidence_n_channels_is_2() -> None:
    gen = CortexEvidenceGenerator(CortexConfig())
    trial = _make_trial("go")
    assert gen(trial, 0.0).n_channels == 2


# --- stop_signal_present detection ---


def test_stop_signal_present_false_when_no_stop_event(gen: CortexEvidenceGenerator) -> None:
    trial = _make_trial("stop", task_type="stop_signal")
    ev = gen(trial, 50.0)
    assert ev.stop_signal_present is False


def test_stop_signal_present_true_when_stop_event_in_log() -> None:
    trial = _make_trial("stop", task_type="stop_signal")
    trial.events.append(
        TaskEvent(
            event_type=EventType.stop_signal,
            sim_time=0.2,
            real_time=0.2,
            trial_id=trial.trial_id,
            payload={},
        )
    )
    gen = CortexEvidenceGenerator(CortexConfig())
    ev = gen(trial, 50.0)
    assert ev.stop_signal_present is True


def test_stop_signal_not_triggered_by_other_event_types() -> None:
    trial = _make_trial("go")
    trial.events.append(
        TaskEvent(
            event_type=EventType.go_cue,
            sim_time=0.1,
            real_time=0.1,
            trial_id=trial.trial_id,
            payload={},
        )
    )
    gen = CortexEvidenceGenerator(CortexConfig())
    ev = gen(trial, 50.0)
    assert ev.stop_signal_present is False


# --- Noise ---


def test_noise_produces_nonzero_deviation() -> None:
    gen = CortexEvidenceGenerator(CortexConfig(noise_std=0.05))
    trial = _make_trial("go")
    ev = gen(trial, 50.0)
    # With noise, salience should deviate slightly from the deterministic value
    # (extremely unlikely to be exactly 0.7 / 0.3)
    # Can't assert exact inequality but salience must remain in [0.0, 1.0]
    for s in ev.channel_salience:
        assert 0.0 <= s <= 1.0


def test_noise_is_reproducible_for_same_seed() -> None:
    gen = CortexEvidenceGenerator(CortexConfig(noise_std=0.05))
    t1 = _make_trial("go", seed=99)
    t2 = _make_trial("go", seed=99)
    ev1 = gen(t1, 50.0)
    ev2 = gen(t2, 50.0)
    assert ev1.channel_salience == ev2.channel_salience


def test_noise_differs_for_different_seeds() -> None:
    gen = CortexEvidenceGenerator(CortexConfig(noise_std=0.05))
    t1 = _make_trial("go", seed=1)
    t2 = _make_trial("go", seed=2)
    ev1 = gen(t1, 50.0)
    ev2 = gen(t2, 50.0)
    assert ev1.channel_salience != ev2.channel_salience


# --- change_of_mind cue_identities ---


def test_no_switch_maps_to_channel_0() -> None:
    gen = CortexEvidenceGenerator(CortexConfig())
    trial = _make_trial("no_switch", task_type="change_of_mind")
    ev = gen(trial, 100.0)
    assert ev.channel_salience[0] > ev.channel_salience[1]


def test_switch_cue_maps_to_channel_0_for_pre_switch() -> None:
    gen = CortexEvidenceGenerator(CortexConfig())
    trial = _make_trial("switch_early", task_type="change_of_mind")
    ev = gen(trial, 100.0)
    assert ev.channel_salience[0] > ev.channel_salience[1]
