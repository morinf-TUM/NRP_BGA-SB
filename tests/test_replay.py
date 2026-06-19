"""Tests for replay: load_trials and replay_events."""

import pytest

from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.replay import load_trials, replay_events
from nrp_bga_sb.schemas import EventType, TrialLog

# --- Helpers ---


def make_logger(tmp_path, filename="trials.jsonl") -> TrialLogger:
    return TrialLogger(tmp_path / filename)


def open_basic_trial(logger: TrialLogger, trial_id: int = 1, seed: int = 42) -> TrialLog:
    return logger.open_trial(
        trial_id=trial_id,
        seed=seed,
        task_type="go_nogo",
        cue_identity="left",
        cue_onset_time=0.5,
    )


# --- test_load_trials_empty_file ---


def test_load_trials_empty_file(tmp_path):
    """Empty file returns []."""
    filepath = tmp_path / "empty.jsonl"
    filepath.write_text("", encoding="utf-8")

    result = load_trials(filepath)

    assert result == []


# --- test_load_trials_single_trial ---


def test_load_trials_single_trial(tmp_path):
    """One written trial; loaded trial matches on trial_id, seed, task_type."""
    logger = make_logger(tmp_path)
    log = open_basic_trial(logger, trial_id=5, seed=99)
    logger.save_trial(log)

    loaded = load_trials(logger.output_path)

    assert len(loaded) == 1
    assert loaded[0].trial_id == 5
    assert loaded[0].seed == 99
    assert loaded[0].task_type == "go_nogo"


# --- test_load_trials_multiple_trials ---


def test_load_trials_multiple_trials(tmp_path):
    """Two written trials; loaded list has length 2 with correct trial_ids."""
    logger = make_logger(tmp_path)

    log1 = open_basic_trial(logger, trial_id=10, seed=100)
    logger.save_trial(log1)

    log2 = open_basic_trial(logger, trial_id=20, seed=200)
    logger.save_trial(log2)

    loaded = load_trials(logger.output_path)

    assert len(loaded) == 2
    assert loaded[0].trial_id == 10
    assert loaded[1].trial_id == 20


# --- test_round_trip_exact (M0 acceptance test) ---


def test_round_trip_exact(tmp_path):
    """M0: A logged trial is field-for-field identical when round-tripped.

    Create a TrialLog with ≥3 events using TrialLogger, save, load,
    and assert loaded[0] == original (Pydantic BaseModel equality).
    """
    logger = make_logger(tmp_path)
    original = open_basic_trial(logger, trial_id=7, seed=555)

    # Add ≥3 events in random order to ensure sim_time ordering is tested
    logger.record_event(original, EventType.trial_start, sim_time=0.0, real_time=0.01)
    logger.record_event(
        original,
        EventType.go_cue,
        sim_time=0.5,
        real_time=0.52,
        payload={"target": "left"},
    )
    logger.record_event(
        original,
        EventType.movement_onset,
        sim_time=0.2,
        real_time=0.21,
    )
    logger.record_event(
        original,
        EventType.movement_end,
        sim_time=1.1,
        real_time=1.11,
    )

    # Save and load
    logger.save_trial(original)
    loaded = load_trials(logger.output_path)

    # Pydantic BaseModel equality: field-for-field match
    assert loaded[0] == original
    assert loaded[0].trial_id == original.trial_id
    assert loaded[0].seed == original.seed
    assert len(loaded[0].events) == len(original.events)
    for orig_event, loaded_event in zip(original.events, loaded[0].events):
        assert orig_event.event_type == loaded_event.event_type
        assert orig_event.sim_time == loaded_event.sim_time
        assert orig_event.real_time == loaded_event.real_time
        assert orig_event.trial_id == loaded_event.trial_id
        assert orig_event.payload == loaded_event.payload


# --- test_replay_events_order ---


def test_replay_events_order(tmp_path):
    """Given events added out of order, replay_events yields them in ascending sim_time."""
    logger = make_logger(tmp_path)
    log = open_basic_trial(logger)

    # Add events out of sim_time order
    logger.record_event(log, EventType.movement_end, sim_time=2.5, real_time=2.51)
    logger.record_event(log, EventType.trial_start, sim_time=0.0, real_time=0.01)
    logger.record_event(log, EventType.movement_onset, sim_time=1.0, real_time=1.01)
    logger.record_event(log, EventType.go_cue, sim_time=0.5, real_time=0.51)

    # replay_events should yield in ascending sim_time order
    replayed = list(replay_events(log))

    assert len(replayed) == 4
    assert replayed[0].event_type == EventType.trial_start
    assert replayed[0].sim_time == 0.0

    assert replayed[1].event_type == EventType.go_cue
    assert replayed[1].sim_time == 0.5

    assert replayed[2].event_type == EventType.movement_onset
    assert replayed[2].sim_time == 1.0

    assert replayed[3].event_type == EventType.movement_end
    assert replayed[3].sim_time == 2.5


# --- test_replay_events_empty_trial ---


def test_replay_events_empty_trial(tmp_path):
    """list(replay_events(empty_log)) is []."""
    logger = make_logger(tmp_path)
    empty_log = open_basic_trial(logger)

    result = list(replay_events(empty_log))

    assert result == []


# --- test_load_trials_skips_blank_lines ---


def test_load_trials_skips_blank_lines(tmp_path):
    """A file with a blank line between two trials loads both correctly."""
    filepath = tmp_path / "with_blanks.jsonl"

    logger = make_logger(tmp_path)
    trial1 = open_basic_trial(logger, trial_id=1, seed=10)
    trial2 = open_basic_trial(logger, trial_id=2, seed=20)

    # Write manually with a blank line in between
    line1 = trial1.model_dump_json()
    line2 = trial2.model_dump_json()

    filepath.write_text(f"{line1}\n\n{line2}\n", encoding="utf-8")

    loaded = load_trials(filepath)

    assert len(loaded) == 2
    assert loaded[0].trial_id == 1
    assert loaded[1].trial_id == 2


# --- test_load_trials_file_not_found ---


def test_load_trials_file_not_found(tmp_path):
    """Raises FileNotFoundError for a non-existent path."""
    nonexistent = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError):
        load_trials(nonexistent)
