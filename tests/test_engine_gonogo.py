"""Tests for the go/no-go task engine."""

import pytest

from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, EventType, TrialLog

# --- Fixtures and Helpers ---


def oracle_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
    """Oracle policy: selects action 0 if go trial, -1 (no action) if no-go trial.

    This is a perfect policy for testing: it always makes the correct decision.
    """
    is_go_trial = trial_log.cue_identity == "go"
    selected_channel = 0 if is_go_trial else -1

    return BGDecision(
        sim_time=action_evidence.sim_time,
        trial_id=trial_log.trial_id,
        selected_channel=selected_channel,
        decision_margin=0.0,
        suppression_vector=[0.0, 0.0],
        channel_activations=[0.0, 0.0],
        selection_latency=0.0,
    )


def always_act_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
    """Always selects action 0, regardless of trial type.

    Used to test false-alarm classification on no-go trials.
    """
    return BGDecision(
        sim_time=action_evidence.sim_time,
        trial_id=trial_log.trial_id,
        selected_channel=0,  # Always act
        decision_margin=0.0,
        suppression_vector=[0.0, 0.0],
        channel_activations=[0.0, 0.0],
        selection_latency=0.0,
    )


def never_act_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
    """Always selects channel -1 (no action), regardless of trial type.

    Used to test miss classification on go trials.
    """
    return BGDecision(
        sim_time=action_evidence.sim_time,
        trial_id=trial_log.trial_id,
        selected_channel=-1,  # Never act
        decision_margin=0.0,
        suppression_vector=[0.0, 0.0],
        channel_activations=[0.0, 0.0],
        selection_latency=0.0,
    )


def default_config() -> GoNoGoConfig:
    """Return a default configuration for testing."""
    return GoNoGoConfig(
        n_trials=10,
        go_probability=0.5,
        response_window_start_ms=100,
        response_window_duration_ms=500,
        fixation_duration_ms=500,
        cue_onset_ms=1000,
        decision_point_ms=100,  # 100 ms after cue onset
        seed=12345,
    )


# --- Test: Engine Returns Correct Trial Count ---


def test_engine_returns_correct_trial_count():
    """Test that the engine returns exactly n_trials trials."""
    config = default_config()
    config.n_trials = 20
    trials = run_go_nogo_trials(config, oracle_policy)
    assert len(trials) == 20


def test_engine_returns_valid_trial_logs():
    """Test that all returned trials are valid TrialLog objects."""
    config = default_config()
    trials = run_go_nogo_trials(config, oracle_policy)
    assert all(isinstance(t, TrialLog) for t in trials)


# --- Test: Trial Identification ---


def test_trial_ids_are_sequential():
    """Test that trial_id increments from 1 to n_trials."""
    config = default_config()
    config.n_trials = 5
    trials = run_go_nogo_trials(config, oracle_policy)
    assert [t.trial_id for t in trials] == [1, 2, 3, 4, 5]


def test_trial_seeds_are_unique():
    """Test that each trial has a unique seed."""
    config = default_config()
    config.n_trials = 20
    trials = run_go_nogo_trials(config, oracle_policy)
    seeds = [t.seed for t in trials]
    assert len(set(seeds)) == len(seeds), "All seeds should be unique"


def test_trial_seeds_are_deterministic():
    """Test that the same master seed produces the same trial seeds."""
    config1 = default_config()
    config1.seed = 999
    config1.n_trials = 5
    trials1 = run_go_nogo_trials(config1, oracle_policy)

    config2 = default_config()
    config2.seed = 999
    config2.n_trials = 5
    trials2 = run_go_nogo_trials(config2, oracle_policy)

    assert [t.seed for t in trials1] == [t.seed for t in trials2]


# --- Test: Go/No-Go Trial Distribution ---


def test_go_probability_balanced():
    """Test that go_probability=0.5 produces approximately 50/50 split."""
    config = default_config()
    config.n_trials = 100
    config.go_probability = 0.5
    config.seed = 42
    trials = run_go_nogo_trials(config, oracle_policy)

    go_count = sum(1 for t in trials if t.cue_identity == "go")
    no_go_count = sum(1 for t in trials if t.cue_identity == "no_go")

    assert go_count + no_go_count == 100
    # Allow ±20% deviation for randomness
    assert 30 <= go_count <= 70


def test_go_probability_70_30():
    """Test that go_probability=0.7 produces approximately 70/30 split."""
    config = default_config()
    config.n_trials = 100
    config.go_probability = 0.7
    config.seed = 42
    trials = run_go_nogo_trials(config, oracle_policy)

    go_count = sum(1 for t in trials if t.cue_identity == "go")
    no_go_count = sum(1 for t in trials if t.cue_identity == "no_go")

    assert go_count + no_go_count == 100
    # Allow ±15% deviation for randomness
    assert 55 <= go_count <= 85


# --- Test: Events Are Emitted Correctly ---


def test_all_trials_have_events():
    """Test that every trial has an events list."""
    config = default_config()
    trials = run_go_nogo_trials(config, oracle_policy)
    assert all(isinstance(t.events, list) for t in trials)
    assert all(len(t.events) > 0 for t in trials)


def test_trial_has_canonical_events():
    """Test that each trial contains the expected canonical events.

    Trial_start, fixation_on, decision_commit, and trial_end are emitted on all trials.
    Movement_end is only emitted when the agent responds (channel >= 0).
    """
    config = default_config()
    trials = run_go_nogo_trials(config, oracle_policy)

    always_present_events = {
        EventType.trial_start,
        EventType.fixation_on,
        EventType.decision_commit,
        EventType.trial_end,
    }

    for trial in trials:
        event_types = {e.event_type for e in trial.events}
        assert always_present_events.issubset(event_types)

        # movement_end is present iff the agent responded
        movement_end_events = [e for e in trial.events if e.event_type == EventType.movement_end]
        if trial.movement_onset_time is not None:
            # Agent responded: movement_end must be present
            assert len(movement_end_events) == 1
        else:
            # Agent did not respond (no_go correct_withhold): movement_end absent
            assert len(movement_end_events) == 0


def test_go_trial_emits_go_cue():
    """Test that go trials emit go_cue event."""
    config = default_config()
    config.n_trials = 20
    config.go_probability = 1.0  # Force all go trials
    trials = run_go_nogo_trials(config, oracle_policy)

    for trial in trials:
        assert trial.cue_identity == "go"
        event_types = {e.event_type for e in trial.events}
        assert EventType.go_cue in event_types


def test_no_go_trial_emits_no_go_cue():
    """Test that no-go trials emit no_go_cue event."""
    config = default_config()
    config.n_trials = 20
    config.go_probability = 0.0  # Force all no-go trials
    trials = run_go_nogo_trials(config, oracle_policy)

    for trial in trials:
        assert trial.cue_identity == "no_go"
        event_types = {e.event_type for e in trial.events}
        assert EventType.no_go_cue in event_types


# --- Test: Success/Failure Classification ---


def test_oracle_policy_all_success():
    """Test that oracle policy achieves 100% success rate on all trials."""
    config = default_config()
    config.n_trials = 50
    config.seed = 100
    trials = run_go_nogo_trials(config, oracle_policy)

    success_count = sum(1 for t in trials if t.success)
    assert success_count == 50


def test_always_act_policy_correct_on_go_trials():
    """Test that always_act policy succeeds on all go trials."""
    config = default_config()
    config.n_trials = 100
    config.go_probability = 1.0
    trials = run_go_nogo_trials(config, always_act_policy)

    for trial in trials:
        assert trial.success is True
        assert trial.failure_mode is None


def test_always_act_policy_fails_on_no_go_trials():
    """Test that always_act policy fails all no-go trials (false alarm)."""
    config = default_config()
    config.n_trials = 100
    config.go_probability = 0.0
    trials = run_go_nogo_trials(config, always_act_policy)

    for trial in trials:
        assert trial.success is False
        assert trial.failure_mode == "false_alarm"


def test_never_act_policy_fails_on_go_trials():
    """Test that never_act policy fails all go trials (miss)."""
    config = default_config()
    config.n_trials = 100
    config.go_probability = 1.0
    trials = run_go_nogo_trials(config, never_act_policy)

    for trial in trials:
        assert trial.success is False
        assert trial.failure_mode == "miss"


def test_never_act_policy_correct_on_no_go_trials():
    """Test that never_act policy succeeds on all no-go trials."""
    config = default_config()
    config.n_trials = 100
    config.go_probability = 0.0
    trials = run_go_nogo_trials(config, never_act_policy)

    for trial in trials:
        assert trial.success is True
        assert trial.failure_mode is None


# --- Test: Response Timing ---


def test_movement_onset_emitted_only_on_response():
    """Test that movement_onset is emitted only when agent responds."""
    config = default_config()
    config.n_trials = 50
    config.go_probability = 1.0
    trials = run_go_nogo_trials(config, oracle_policy)

    for trial in trials:
        movement_onset_events = [
            e for e in trial.events if e.event_type == EventType.movement_onset
        ]
        assert len(movement_onset_events) == 1
        assert trial.movement_onset_time is not None
        # RT must be positive: movement onset must come strictly after cue onset.
        assert trial.movement_onset_time > trial.cue_onset_time, (
            f"Trial {trial.trial_id}: RT is non-positive "
            f"(movement_onset={trial.movement_onset_time}, cue_onset={trial.cue_onset_time})"
        )


def test_movement_onset_not_emitted_on_no_response():
    """Test that movement_onset is not emitted when agent does not respond."""
    config = default_config()
    config.n_trials = 50
    config.go_probability = 0.0
    trials = run_go_nogo_trials(config, never_act_policy)

    for trial in trials:
        movement_onset_events = [
            e for e in trial.events if e.event_type == EventType.movement_onset
        ]
        assert len(movement_onset_events) == 0
        assert trial.movement_onset_time is None


# --- Test: Trial Logger Integration ---


def test_engine_with_logger(tmp_path):
    """Test that the engine correctly uses TrialLogger."""
    log_path = tmp_path / "trials.jsonl"
    logger = TrialLogger(log_path)

    config = default_config()
    config.n_trials = 5
    trials = run_go_nogo_trials(config, oracle_policy, logger=logger)

    # Check file was created with correct number of lines
    assert log_path.exists()
    with log_path.open() as f:
        lines = f.readlines()
    assert len(lines) == 5

    # Check that returned trials match logged trials
    assert len(trials) == 5
    assert all(t.success is True for t in trials)


def test_engine_without_logger():
    """Test that the engine works without a logger (in-memory only)."""
    config = default_config()
    config.n_trials = 5
    trials = run_go_nogo_trials(config, oracle_policy, logger=None)

    assert len(trials) == 5
    assert all(isinstance(t, TrialLog) for t in trials)


# --- Test: Task Metadata ---


def test_all_trials_task_type_is_go_nogo():
    """Test that all trials are labeled as go_nogo task."""
    config = default_config()
    trials = run_go_nogo_trials(config, oracle_policy)
    assert all(t.task_type == "go_nogo" for t in trials)


def test_cue_onset_time_in_seconds():
    """Test that cue_onset_time is correctly converted to seconds."""
    config = default_config()
    config.cue_onset_ms = 2000
    trials = run_go_nogo_trials(config, oracle_policy)
    assert all(t.cue_onset_time == 2.0 for t in trials)


# --- Test: Events Have Correct Timing ---


def test_trial_start_event_at_zero():
    """Test that trial_start event is at sim_time = 0."""
    config = default_config()
    trials = run_go_nogo_trials(config, oracle_policy)

    for trial in trials:
        trial_start_events = [
            e for e in trial.events if e.event_type == EventType.trial_start
        ]
        assert len(trial_start_events) == 1
        assert trial_start_events[0].sim_time == 0.0


def test_fixation_on_event_timing():
    """Test that fixation_on event is emitted at correct time."""
    config = default_config()
    config.fixation_duration_ms = 750
    trials = run_go_nogo_trials(config, oracle_policy)

    for trial in trials:
        fixation_events = [
            e for e in trial.events if e.event_type == EventType.fixation_on
        ]
        assert len(fixation_events) == 1
        assert fixation_events[0].sim_time == pytest.approx(0.75)


def test_cue_event_timing():
    """Test that go_cue/no_go_cue event is emitted at correct time."""
    config = default_config()
    config.cue_onset_ms = 1500
    trials = run_go_nogo_trials(config, oracle_policy)

    for trial in trials:
        cue_events = [
            e for e in trial.events
            if e.event_type in (EventType.go_cue, EventType.no_go_cue)
        ]
        assert len(cue_events) == 1
        assert cue_events[0].sim_time == pytest.approx(1.5)


# --- Test: Policy Receives Correct Evidence ---


def test_policy_receives_correct_trial_log():
    """Test that policy receives the trial_log with correct trial_id."""
    received_logs = []

    def recording_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
        received_logs.append(trial_log)
        return BGDecision(
            sim_time=action_evidence.sim_time,
            trial_id=trial_log.trial_id,
            selected_channel=0,
            decision_margin=0.0,
            suppression_vector=[0.0, 0.0],
            channel_activations=[0.0, 0.0],
            selection_latency=0.0,
        )

    config = default_config()
    config.n_trials = 5
    run_go_nogo_trials(config, recording_policy)

    assert len(received_logs) == 5
    assert [log.trial_id for log in received_logs] == [1, 2, 3, 4, 5]


def test_policy_receives_correct_action_evidence():
    """Test that policy receives ActionEvidence with correct fields."""
    received_evidence = []

    def recording_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
        received_evidence.append(action_evidence)
        return BGDecision(
            sim_time=action_evidence.sim_time,
            trial_id=trial_log.trial_id,
            selected_channel=0,
            decision_margin=0.0,
            suppression_vector=[0.0, 0.0],
            channel_activations=[0.0, 0.0],
            selection_latency=0.0,
        )

    config = default_config()
    config.n_trials = 3
    config.decision_point_ms = 200
    run_go_nogo_trials(config, recording_policy)

    assert len(received_evidence) == 3
    for evidence in received_evidence:
        assert evidence.n_channels == 2
        assert len(evidence.channel_salience) == 2
        # decision_point_ms is an offset from cue_onset_ms; absolute sim_time is
        # (cue_onset_ms=1000 + decision_point_ms=200) / 1000 = 1.2 s
        assert evidence.sim_time == pytest.approx(1.2)


# --- Test: Response Window Logic ---


def test_response_window_start_offset():
    """Test that response window respects start offset.

    Response window is defined relative to cue onset, not absolute time.
    Window: [response_window_start_ms, response_window_start_ms + response_window_duration_ms)
    Decision at decision_point_ms (relative to cue onset) must fall within this window.

    Config:
    - response_window_start_ms = 500 (window opens 500 ms after cue onset)
    - response_window_duration_ms = 1000 (window duration is 1000 ms)
    - decision_point_ms = 300 (decision made 300 ms after cue onset)
    - Window: [500, 1500) ms relative to cue onset
    - Decision at 300 ms is OUTSIDE window (before window start)

    Outcome: oracle acts on go trials, but is outside window → wrong_action failure.
    """
    config = default_config()
    config.response_window_start_ms = 500
    config.response_window_duration_ms = 1000
    config.decision_point_ms = 300  # Before window start

    # Test go trials: oracle acts, but outside window → wrong_action
    config.go_probability = 1.0
    config.n_trials = 10
    trials = run_go_nogo_trials(config, oracle_policy)
    for trial in trials:
        assert trial.success is False
        assert trial.failure_mode == "wrong_action"

    # Test no-go trials: oracle does not act → correct_withhold
    config.go_probability = 0.0
    config.n_trials = 10
    trials = run_go_nogo_trials(config, oracle_policy)
    for trial in trials:
        assert trial.success is True
        assert trial.failure_mode is None


def test_response_window_respects_duration():
    """Test that decisions outside window are penalized.

    Response window is relative to cue onset, not absolute time.
    Window: [response_window_start_ms, response_window_start_ms + response_window_duration_ms)
    Decision at decision_point_ms (relative to cue onset) must fall within this window.

    Config:
    - response_window_start_ms = 200 (window opens 200 ms after cue onset)
    - response_window_duration_ms = 300 (window duration is 300 ms)
    - decision_point_ms = 600 (decision made 600 ms after cue onset)
    - Window: [200, 500) ms relative to cue onset
    - Decision at 600 ms is OUTSIDE window (after window end)

    Outcome: oracle acts on go trials, but outside window → wrong_action failure.
    """
    config = default_config()
    config.response_window_start_ms = 200
    config.response_window_duration_ms = 300
    config.decision_point_ms = 600  # After window end (window ends at 500)

    # Test go trials: oracle acts, but outside window → wrong_action
    config.go_probability = 1.0
    config.n_trials = 10
    trials = run_go_nogo_trials(config, oracle_policy)
    for trial in trials:
        assert trial.success is False
        assert trial.failure_mode == "wrong_action"

    # Test no-go trials: oracle does not act → correct_withhold
    config.go_probability = 0.0
    config.n_trials = 10
    trials = run_go_nogo_trials(config, oracle_policy)
    for trial in trials:
        assert trial.success is True
        assert trial.failure_mode is None


# --- Test: Edge Cases ---


def test_engine_with_single_trial():
    """Test that engine handles n_trials = 1 correctly."""
    config = default_config()
    config.n_trials = 1
    trials = run_go_nogo_trials(config, oracle_policy)
    assert len(trials) == 1
    assert trials[0].trial_id == 1


def test_engine_with_go_probability_zero():
    """Test that go_probability = 0 produces all no-go trials."""
    config = default_config()
    config.n_trials = 20
    config.go_probability = 0.0
    trials = run_go_nogo_trials(config, oracle_policy)
    assert all(t.cue_identity == "no_go" for t in trials)


def test_engine_with_go_probability_one():
    """Test that go_probability = 1.0 produces all go trials."""
    config = default_config()
    config.n_trials = 20
    config.go_probability = 1.0
    trials = run_go_nogo_trials(config, oracle_policy)
    assert all(t.cue_identity == "go" for t in trials)


# --- Test: Integration with Scorer ---


def test_trials_are_scorable():
    """Test that returned trials can be scored without error."""
    from nrp_bga_sb.scorer import score_trials as score_function

    config = default_config()
    config.n_trials = 20
    config.go_probability = 0.5
    config.seed = 42
    trials = run_go_nogo_trials(config, oracle_policy)

    metrics = score_function(trials, condition_id="test", bg_frequency_hz=40.0)
    assert metrics.n_trials == 20
    assert metrics.condition_id == "test"
    assert metrics.bg_frequency_hz == 40.0
