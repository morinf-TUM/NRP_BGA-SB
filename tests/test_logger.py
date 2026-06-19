"""Tests for TrialLogger: open_trial, record_event, save_trial."""

from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.schemas import EventType, TaskEvent, TrialLog

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


# --- open_trial ---


def test_open_trial_fields(tmp_path):
    logger = make_logger(tmp_path)
    log = logger.open_trial(
        trial_id=7,
        seed=99,
        task_type="stop_signal",
        cue_identity="right",
        cue_onset_time=1.25,
    )
    assert log.trial_id == 7
    assert log.seed == 99
    assert log.task_type == "stop_signal"
    assert log.cue_identity == "right"
    assert log.cue_onset_time == 1.25
    # Optional fields must be at their schema defaults
    assert log.bg_input_receive_time is None
    assert log.bg_output_emit_time is None
    assert log.bg_selected_channel is None
    assert log.bg_channel_activations == []
    assert log.thalamic_relay_time is None
    assert log.thalamic_release_time is None
    assert log.motor_command_series == []
    assert log.movement_onset_time is None
    assert log.endpoint_trajectory == []
    assert log.endpoint_error is None
    assert log.success is None
    assert log.failure_mode is None
    assert log.events == []


# --- record_event ---


def test_record_event_appends(tmp_path):
    logger = make_logger(tmp_path)
    log = open_basic_trial(logger)

    logger.record_event(log, EventType.trial_start, sim_time=0.0, real_time=0.01)
    logger.record_event(
        log,
        EventType.go_cue,
        sim_time=0.5,
        real_time=0.51,
        payload={"target": "left"},
    )

    assert len(log.events) == 2

    e0 = log.events[0]
    assert e0.event_type == EventType.trial_start
    assert e0.sim_time == 0.0
    assert e0.real_time == 0.01
    assert e0.trial_id == 1
    assert e0.payload == {}

    e1 = log.events[1]
    assert e1.event_type == EventType.go_cue
    assert e1.sim_time == 0.5
    assert e1.real_time == 0.51
    assert e1.trial_id == 1
    assert e1.payload == {"target": "left"}


def test_record_event_returns_task_event(tmp_path):
    logger = make_logger(tmp_path)
    log = open_basic_trial(logger)

    returned = logger.record_event(log, EventType.movement_onset, sim_time=0.8, real_time=0.82)

    # The returned event is exactly what was appended
    assert isinstance(returned, TaskEvent)
    assert returned is log.events[0]


# --- save_trial ---


def test_save_trial_creates_file(tmp_path):
    logger = make_logger(tmp_path)
    log = open_basic_trial(logger)

    assert not logger.output_path.exists()
    logger.save_trial(log)
    assert logger.output_path.exists()


def test_save_trial_valid_jsonl(tmp_path):
    logger = make_logger(tmp_path)
    log = open_basic_trial(logger, trial_id=3, seed=7)
    logger.record_event(log, EventType.trial_start, sim_time=0.0, real_time=0.0)

    logger.save_trial(log)

    lines = logger.output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    # Each line must be valid JSON and round-trip through TrialLog
    parsed = TrialLog.model_validate_json(lines[0])
    assert parsed.trial_id == 3
    assert parsed.seed == 7
    assert len(parsed.events) == 1
    assert parsed.events[0].event_type == EventType.trial_start


def test_save_trial_appends(tmp_path):
    logger = make_logger(tmp_path)

    log1 = open_basic_trial(logger, trial_id=1, seed=10)
    logger.save_trial(log1)

    log2 = open_basic_trial(logger, trial_id=2, seed=20)
    logger.save_trial(log2)

    lines = logger.output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    parsed1 = TrialLog.model_validate_json(lines[0])
    parsed2 = TrialLog.model_validate_json(lines[1])
    assert parsed1.trial_id == 1
    assert parsed2.trial_id == 2


def test_deterministic_seed_stored(tmp_path):
    logger = make_logger(tmp_path)

    log_a = logger.open_trial(
        trial_id=1, seed=12345, task_type="two_choice",
        cue_identity="left", cue_onset_time=0.0,
    )
    log_b = logger.open_trial(
        trial_id=2, seed=12345, task_type="two_choice",
        cue_identity="right", cue_onset_time=0.0,
    )

    assert log_a.seed == 12345
    assert log_b.seed == 12345


def test_save_trial_creates_parent_dirs(tmp_path):
    # Parent directory does not exist yet
    nested_path = tmp_path / "deep" / "nested" / "subdir" / "trials.jsonl"
    logger = TrialLogger(nested_path)

    log = open_basic_trial(logger)
    logger.save_trial(log)

    assert nested_path.exists()
    lines = nested_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
