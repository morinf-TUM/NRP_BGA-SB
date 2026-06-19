"""Tests for the minimal scorer (Task 0.7)."""

import pytest

from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.schemas import EventType, Metrics, TrialLog
from nrp_bga_sb.scorer import score_trials


def make_trial(
    trial_id: int,
    seed: int,
    task_type: str = "go_nogo",
    cue_identity: str = "go",
    cue_onset_time: float = 1.0,
    movement_onset_time: float | None = 1.5,
    failure_mode: str | None = None,
    events: list[EventType] | None = None,
) -> TrialLog:
    """Create a minimal TrialLog for testing.

    Helper that allows tests to specify only the fields they care about.
    """
    if events is None:
        events = []

    trial = TrialLog(
        trial_id=trial_id,
        seed=seed,
        task_type=task_type,
        cue_identity=cue_identity,
        cue_onset_time=cue_onset_time,
        movement_onset_time=movement_onset_time,
        failure_mode=failure_mode,
    )

    # Add events to the trial
    for event_type in events:
        trial.events.append(
            TrialLogger(None).record_event(
                trial,
                event_type=event_type,
                sim_time=cue_onset_time,
                real_time=0.0,
            )
        )

    return trial


def test_score_empty_raises():
    """Test that score_trials raises ValueError on empty trial list."""
    with pytest.raises(ValueError, match="at least one trial"):
        score_trials([], condition_id="test", bg_frequency_hz=40.0)


def test_score_returns_metrics():
    """Test that score_trials returns a Metrics instance."""
    trial = make_trial(trial_id=1, seed=100)
    result = score_trials([trial], condition_id="test", bg_frequency_hz=40.0)
    assert isinstance(result, Metrics)


def test_reaction_time_computed():
    """Test that reaction_time_mean equals movement_onset_time - cue_onset_time."""
    trial = make_trial(
        trial_id=1,
        seed=100,
        cue_onset_time=1.0,
        movement_onset_time=1.5,
    )
    result = score_trials([trial], condition_id="test", bg_frequency_hz=40.0)
    assert result.reaction_time_mean == pytest.approx(0.5)


def test_reaction_time_single_trial_no_std():
    """Test that reaction_time_std is None for a single trial with RT."""
    trial = make_trial(trial_id=1, seed=100)
    result = score_trials([trial], condition_id="test", bg_frequency_hz=40.0)
    assert result.reaction_time_mean is not None
    assert result.reaction_time_std is None


def test_reaction_time_multiple_trials():
    """Test that both reaction_time_mean and reaction_time_std are computed for n >= 2."""
    trials = [
        make_trial(trial_id=1, seed=100, cue_onset_time=1.0, movement_onset_time=1.5),
        make_trial(trial_id=2, seed=101, cue_onset_time=2.0, movement_onset_time=2.4),
    ]
    result = score_trials(trials, condition_id="test", bg_frequency_hz=40.0)

    # Expected: RTs are [0.5, 0.4], mean = 0.45, std = sqrt(((0.5-0.45)^2 + (0.4-0.45)^2) / (2-1))
    expected_mean = 0.45
    expected_std = ((0.5 - 0.45) ** 2 + (0.4 - 0.45) ** 2) / 1
    expected_std = expected_std ** 0.5

    assert result.reaction_time_mean == pytest.approx(expected_mean)
    assert result.reaction_time_std == pytest.approx(expected_std)


def test_reaction_time_none_when_no_response():
    """Test that both RT fields are None when all trials have movement_onset_time = None."""
    trials = [
        make_trial(trial_id=1, seed=100, movement_onset_time=None),
        make_trial(trial_id=2, seed=101, movement_onset_time=None),
    ]
    result = score_trials(trials, condition_id="test", bg_frequency_hz=40.0)
    assert result.reaction_time_mean is None
    assert result.reaction_time_std is None


def test_wrong_action_rate():
    """Test that wrong_action_rate is computed correctly."""
    trials = [
        make_trial(trial_id=1, seed=100, failure_mode="wrong_action"),
        make_trial(trial_id=2, seed=101, failure_mode="wrong_action"),
        make_trial(trial_id=3, seed=102, failure_mode=None),
        make_trial(trial_id=4, seed=103, failure_mode=None),
    ]
    result = score_trials(trials, condition_id="test", bg_frequency_hz=40.0)
    assert result.wrong_action_rate == pytest.approx(0.5)


def test_false_alarm_rate_with_no_go_trials():
    """Test that false_alarm_rate is computed correctly when no_go_cue events exist."""
    trials = [
        make_trial(
            trial_id=1,
            seed=100,
            cue_identity="no_go",
            movement_onset_time=1.5,
            events=[EventType.no_go_cue],
        ),
        make_trial(
            trial_id=2,
            seed=101,
            cue_identity="no_go",
            movement_onset_time=None,
            events=[EventType.no_go_cue],
        ),
        make_trial(
            trial_id=3,
            seed=102,
            cue_identity="go",
            movement_onset_time=1.5,
            events=[EventType.go_cue],
        ),
    ]
    result = score_trials(trials, condition_id="test", bg_frequency_hz=40.0)
    # 2 no_go_trials, 1 with movement_onset_time -> false_alarm_rate = 1/2
    assert result.false_alarm_rate == pytest.approx(0.5)


def test_false_alarm_rate_none_without_no_go_trials():
    """Test that false_alarm_rate is None when there are no no_go_cue events."""
    trials = [
        make_trial(
            trial_id=1,
            seed=100,
            cue_identity="go",
            events=[EventType.go_cue],
        ),
        make_trial(
            trial_id=2,
            seed=101,
            cue_identity="go",
            events=[EventType.go_cue],
        ),
    ]
    result = score_trials(trials, condition_id="test", bg_frequency_hz=40.0)
    assert result.false_alarm_rate is None


def test_condition_fields():
    """Test that condition_id, bg_frequency_hz, and n_trials are passed through correctly."""
    trials = [
        make_trial(trial_id=1, seed=100),
        make_trial(trial_id=2, seed=101),
        make_trial(trial_id=3, seed=102),
    ]
    result = score_trials(trials, condition_id="cond_40hz", bg_frequency_hz=40.0)
    assert result.condition_id == "cond_40hz"
    assert result.bg_frequency_hz == 40.0
    assert result.n_trials == 3
