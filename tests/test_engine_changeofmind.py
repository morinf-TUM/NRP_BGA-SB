"""Tests for the change-of-mind task engine.

Coverage:
1.  Config validation — __post_init__ rejects invalid proportions and timing.
2.  Switch trial outcome — correct switch (channel 1 post-switch → success).
3.  Perseveration — post-switch selects channel 0 → perseveration failure.
4.  Miss — post-switch returns -1 → miss failure.
5.  No-switch baseline — single policy call; success if responded, miss if not.
6.  Switch delay categories — cycling in insertion order over switch trials.
7.  Pre-switch decision logged but not used for outcome.
8.  evidence_change event emitted between the two policy calls on switch trials.
9.  Pre-switch and post-switch calls receive different ActionEvidence (saliences).
10. movement_end only emitted when a response was made.
11. Logger integration — trials persisted to JSONL and reload correctly.
12. Scorer integration — switch_success_rate computed from completed trials.
13. Determinism — same seed produces identical trial sequences.
"""

from __future__ import annotations

import pytest

from nrp_bga_sb.engines.change_of_mind import (
    ChangeOfMindConfig,
    run_change_of_mind_trials,
)
from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.replay import load_trials
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, EventType, TrialLog
from nrp_bga_sb.scorer import score_trials

# --- Shared policy helpers ---


def _make_decision(
    trial_log: TrialLog,
    action_evidence: ActionEvidence,
    channel: int,
) -> BGDecision:
    return BGDecision(
        sim_time=action_evidence.sim_time,
        trial_id=action_evidence.trial_id,
        selected_channel=channel,
        decision_margin=0.5 if channel >= 0 else 0.0,
        suppression_vector=[0.0, 0.0],
        channel_activations=[0.8, 0.3] if channel == 0 else [0.3, 0.8],
        selection_latency=0.01,
    )


def always_channel1_policy(
    trial_log: TrialLog, action_evidence: ActionEvidence
) -> BGDecision:
    """Always selects channel 1 — oracle for switch trials."""
    return _make_decision(trial_log, action_evidence, 1)


def always_channel0_policy(
    trial_log: TrialLog, action_evidence: ActionEvidence
) -> BGDecision:
    """Always selects channel 0 — perseveration on switch trials."""
    return _make_decision(trial_log, action_evidence, 0)


def always_no_response_policy(
    trial_log: TrialLog, action_evidence: ActionEvidence
) -> BGDecision:
    """Always returns -1 — miss on all trials."""
    return _make_decision(trial_log, action_evidence, -1)


# --- Salience-tracking policy ---

# Accumulates the channel_salience seen on each call so tests can verify
# that pre-switch and post-switch calls carry different evidence.

class SalienceTracker:
    """Policy that records every ActionEvidence it receives and responds on channel 1."""

    def __init__(self) -> None:
        self.calls: list[list[float]] = []

    def __call__(
        self, trial_log: TrialLog, action_evidence: ActionEvidence
    ) -> BGDecision:
        self.calls.append(list(action_evidence.channel_salience))
        return _make_decision(trial_log, action_evidence, 1)


# --- Minimal configs for tests ---

MINIMAL_CONFIG = ChangeOfMindConfig(
    n_trials=20,
    no_switch_proportion=0.0,   # all switch trials
    switch_delay_categories={"early": 50, "medium": 150},
    initial_decision_point_ms=20,
    post_switch_decision_point_ms=200,
    response_window_duration_ms=700,
    seed=42,
)

NO_SWITCH_CONFIG = ChangeOfMindConfig(
    n_trials=10,
    no_switch_proportion=1.0,   # all no-switch (baseline) trials
    switch_delay_categories={"early": 50},
    initial_decision_point_ms=20,
    post_switch_decision_point_ms=200,
    response_window_duration_ms=700,
    seed=7,
)


# --- 1. Config validation ---


def test_config_invalid_no_switch_proportion_high() -> None:
    with pytest.raises(ValueError, match="no_switch_proportion"):
        ChangeOfMindConfig(
            n_trials=10,
            no_switch_proportion=1.5,
            switch_delay_categories={"early": 50},
            initial_decision_point_ms=20,
            post_switch_decision_point_ms=100,
        )


def test_config_invalid_no_switch_proportion_negative() -> None:
    with pytest.raises(ValueError, match="no_switch_proportion"):
        ChangeOfMindConfig(
            n_trials=10,
            no_switch_proportion=-0.1,
            switch_delay_categories={"early": 50},
            initial_decision_point_ms=20,
            post_switch_decision_point_ms=100,
        )


def test_config_initial_decision_point_too_late() -> None:
    """initial_decision_point_ms must be < min switch delay (50)."""
    with pytest.raises(ValueError, match="initial_decision_point_ms"):
        ChangeOfMindConfig(
            n_trials=10,
            no_switch_proportion=0.25,
            switch_delay_categories={"early": 50, "medium": 150},
            initial_decision_point_ms=50,   # equal to min → invalid
            post_switch_decision_point_ms=200,
        )


def test_config_post_switch_decision_point_too_early() -> None:
    """post_switch_decision_point_ms must be > max switch delay (150)."""
    with pytest.raises(ValueError, match="post_switch_decision_point_ms"):
        ChangeOfMindConfig(
            n_trials=10,
            no_switch_proportion=0.25,
            switch_delay_categories={"early": 50, "medium": 150},
            initial_decision_point_ms=20,
            post_switch_decision_point_ms=150,   # equal to max → invalid
        )


def test_config_empty_categories() -> None:
    with pytest.raises(ValueError, match="switch_delay_categories"):
        ChangeOfMindConfig(
            n_trials=10,
            no_switch_proportion=0.25,
            switch_delay_categories={},
            initial_decision_point_ms=20,
            post_switch_decision_point_ms=200,
        )


# --- 2. Correct switch outcome ---


def test_correct_switch_outcome() -> None:
    """Post-switch channel 1 → success, no failure_mode."""
    trials = run_change_of_mind_trials(MINIMAL_CONFIG, always_channel1_policy)
    for t in trials:
        assert t.success is True
        assert t.failure_mode is None


# --- 3. Perseveration outcome ---


def test_perseveration_outcome() -> None:
    """Post-switch channel 0 → perseveration failure."""
    trials = run_change_of_mind_trials(MINIMAL_CONFIG, always_channel0_policy)
    for t in trials:
        assert t.success is False
        assert t.failure_mode == "perseveration"


# --- 4. Miss outcome ---


def test_miss_outcome_on_switch_trial() -> None:
    """Post-switch channel -1 → miss failure."""
    trials = run_change_of_mind_trials(MINIMAL_CONFIG, always_no_response_policy)
    for t in trials:
        assert t.success is False
        assert t.failure_mode == "miss"


# --- 5. No-switch baseline trials ---


def test_no_switch_baseline_success() -> None:
    """No-switch trial with a response → success."""
    trials = run_change_of_mind_trials(NO_SWITCH_CONFIG, always_channel1_policy)
    for t in trials:
        assert t.cue_identity == "no_switch"
        assert t.success is True
        assert t.failure_mode is None


def test_no_switch_baseline_miss() -> None:
    """No-switch trial with no response → miss."""
    trials = run_change_of_mind_trials(NO_SWITCH_CONFIG, always_no_response_policy)
    for t in trials:
        assert t.success is False
        assert t.failure_mode == "miss"


def test_no_switch_trial_has_no_evidence_change_event() -> None:
    """Baseline (no-switch) trials must not contain an evidence_change event."""
    trials = run_change_of_mind_trials(NO_SWITCH_CONFIG, always_channel1_policy)
    for t in trials:
        ev_types = [e.event_type for e in t.events]
        assert EventType.evidence_change not in ev_types


# --- 6. Switch delay category cycling ---


def test_switch_delay_categories_cycle() -> None:
    """Switch delay categories cycle in insertion order across switch trials."""
    config = ChangeOfMindConfig(
        n_trials=8,
        no_switch_proportion=0.0,    # all switch
        switch_delay_categories={"early": 50, "medium": 150, "late": 300},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=400,
        seed=0,
    )
    trials = run_change_of_mind_trials(config, always_channel1_policy)
    expected_cycle = ["early", "medium", "late", "early", "medium", "late", "early", "medium"]
    for trial, expected_cat in zip(trials, expected_cycle):
        # cue_identity encodes the category: "switch_{category}"
        assert trial.cue_identity == f"switch_{expected_cat}"


def test_switch_delay_event_timing() -> None:
    """evidence_change sim_time equals go_cue_onset + switch_delay for each category."""
    config = ChangeOfMindConfig(
        n_trials=2,
        no_switch_proportion=0.0,
        switch_delay_categories={"early": 50, "medium": 150},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=0,
    )
    trials = run_change_of_mind_trials(config, always_channel1_policy)

    for t in trials:
        ev_change = next(
            e for e in t.events if e.event_type == EventType.evidence_change
        )
        delay_ms = ev_change.payload["switch_delay_ms"]
        expected_time = (config.go_cue_onset_ms + delay_ms) / 1000.0
        assert abs(ev_change.sim_time - expected_time) < 1e-9


# --- 7. Pre-switch decision logged but not used for outcome ---


def test_pre_switch_decision_logged_not_outcome() -> None:
    """Pre-switch call is recorded in events but the post-switch call determines outcome."""
    # Use a policy that on the first call returns channel 0 and on the second returns channel 1.
    # If pre-switch drove the outcome, success would be False (perseveration); it must be True.
    call_count = 0

    def first_channel0_then_channel1(
        trial_log: TrialLog, action_evidence: ActionEvidence
    ) -> BGDecision:
        nonlocal call_count
        call_count += 1
        ch = 0 if (call_count % 2 == 1) else 1
        return _make_decision(trial_log, action_evidence, ch)

    config = ChangeOfMindConfig(
        n_trials=1,
        no_switch_proportion=0.0,
        switch_delay_categories={"medium": 150},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=0,
    )
    trials = run_change_of_mind_trials(config, first_channel0_then_channel1)
    assert len(trials) == 1
    t = trials[0]
    # Second call returned channel 1 → correct switch → success
    assert t.success is True
    assert t.failure_mode is None


# --- 8. evidence_change event emitted between the two policy calls ---


def test_evidence_change_between_policy_calls() -> None:
    """evidence_change sim_time is strictly between pre-switch and post-switch decision points."""
    config = ChangeOfMindConfig(
        n_trials=4,
        no_switch_proportion=0.0,
        switch_delay_categories={"early": 50, "medium": 150},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=99,
    )
    trials = run_change_of_mind_trials(config, always_channel1_policy)

    pre_abs = (config.go_cue_onset_ms + config.initial_decision_point_ms) / 1000.0
    post_abs = (config.go_cue_onset_ms + config.post_switch_decision_point_ms) / 1000.0

    for t in trials:
        ev_change = next(
            e for e in t.events if e.event_type == EventType.evidence_change
        )
        assert pre_abs < ev_change.sim_time < post_abs


# --- 9. Pre-switch and post-switch calls receive different saliences ---


def test_pre_and_post_switch_saliences_differ() -> None:
    """Policy receives initial_salience on pre-switch call and post_switch_salience after."""
    config = ChangeOfMindConfig(
        n_trials=2,
        no_switch_proportion=0.0,
        switch_delay_categories={"medium": 150},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=1,
        initial_salience=[0.8, 0.2],
        post_switch_salience=[0.2, 0.8],
    )
    tracker = SalienceTracker()
    run_change_of_mind_trials(config, tracker)

    # 2 trials × 2 calls each = 4 total calls
    assert len(tracker.calls) == 4
    # Odd-indexed calls (0, 2) are pre-switch; even-indexed (1, 3) are post-switch
    for i, salience in enumerate(tracker.calls):
        if i % 2 == 0:
            assert salience == [0.8, 0.2], f"pre-switch call {i} should be initial_salience"
        else:
            assert salience == [0.2, 0.8], f"post-switch call {i} should be post_switch_salience"


# --- 10. movement_end only emitted when response was made ---


def test_movement_end_only_when_responded() -> None:
    """movement_end event absent when post-switch decision is -1 (miss)."""
    config = ChangeOfMindConfig(
        n_trials=4,
        no_switch_proportion=0.0,
        switch_delay_categories={"early": 50},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=100,
        seed=3,
    )
    # No-response trials should have no movement_end
    trials = run_change_of_mind_trials(config, always_no_response_policy)
    for t in trials:
        ev_types = [e.event_type for e in t.events]
        assert EventType.movement_end not in ev_types

    # Responding trials should have movement_end
    trials2 = run_change_of_mind_trials(config, always_channel1_policy)
    for t in trials2:
        ev_types = [e.event_type for e in t.events]
        assert EventType.movement_end in ev_types


# --- 11. Logger integration ---


def test_logger_integration(tmp_path) -> None:
    """Trials saved to JSONL and reloaded match original TrialLog objects."""
    from pathlib import Path
    log_file = tmp_path / "com_trials.jsonl"
    logger = TrialLogger(Path(log_file))

    config = ChangeOfMindConfig(
        n_trials=6,
        no_switch_proportion=0.0,
        switch_delay_categories={"early": 50, "medium": 150},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=11,
    )
    original_trials = run_change_of_mind_trials(config, always_channel1_policy, logger)
    reloaded_trials = load_trials(log_file)

    assert len(reloaded_trials) == len(original_trials)
    for orig, reloaded in zip(original_trials, reloaded_trials):
        assert orig.trial_id == reloaded.trial_id
        assert orig.success == reloaded.success
        assert orig.failure_mode == reloaded.failure_mode
        assert len(orig.events) == len(reloaded.events)


# --- 12. Scorer integration ---


def test_scorer_switch_success_rate() -> None:
    """score_trials computes switch_success_rate correctly from channel-1 outcomes."""
    config = ChangeOfMindConfig(
        n_trials=10,
        no_switch_proportion=0.0,
        switch_delay_categories={"medium": 150},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=42,
    )
    all_switch_trials = run_change_of_mind_trials(config, always_channel1_policy)
    metrics = score_trials(all_switch_trials, condition_id="test", bg_frequency_hz=40.0)

    assert metrics.n_trials == 10
    assert metrics.switch_success_rate == pytest.approx(1.0)


def test_scorer_perseveration_gives_zero_switch_rate() -> None:
    """All perseveration outcomes → switch_success_rate == 0."""
    config = ChangeOfMindConfig(
        n_trials=8,
        no_switch_proportion=0.0,
        switch_delay_categories={"medium": 150},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=42,
    )
    trials = run_change_of_mind_trials(config, always_channel0_policy)
    metrics = score_trials(trials, condition_id="test", bg_frequency_hz=40.0)
    assert metrics.switch_success_rate == pytest.approx(0.0)


# --- 13. Determinism ---


def test_determinism() -> None:
    """Same seed produces identical trial sequences."""
    config = ChangeOfMindConfig(
        n_trials=12,
        no_switch_proportion=0.3,
        switch_delay_categories={"early": 50, "medium": 150, "late": 300},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=400,
        seed=77,
    )
    trials_a = run_change_of_mind_trials(config, always_channel1_policy)
    trials_b = run_change_of_mind_trials(config, always_channel1_policy)

    assert len(trials_a) == len(trials_b)
    for a, b in zip(trials_a, trials_b):
        assert a.trial_id == b.trial_id
        assert a.cue_identity == b.cue_identity
        assert a.seed == b.seed
        assert a.success == b.success
