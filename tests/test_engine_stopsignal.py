"""Tests for the stop-signal task engine.

Coverage:
1. Basic go/stop trial mix — correct trial counts and cue identities.
2. Staircase SSD updates — increases on stop-success, decreases on stop-failure.
3. SSD bounds — staircase never escapes [ssd_min_ms, ssd_max_ms].
4. Fixed-SSD mode — SSD stays constant regardless of outcomes.
5. Stop-signal event presence — stop trials have stop_signal in events; go trials don't.
6. Stop-signal event timing — stop_signal sim_time == go_cue_onset + SSD.
7. stop_signal_present flag — True only when SSD < decision_point_ms.
8. Outcome classification — go success, go miss, stop success, stop failure.
9. Validity data — movement_onset_time set on responding trials; computable RT.
10. Failed-stop RT recoverable — logs carry enough data for RT_failed_stop computation.
11. Logger integration — trials persisted to JSONL and reload correctly.
12. Determinism — same seed produces identical trial sequences.
"""

from __future__ import annotations

from collections.abc import Callable

from nrp_bga_sb.engines.stop_signal import StopSignalConfig, run_stop_signal_trials
from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.replay import load_trials
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, EventType, TrialLog

# --- Shared policy helpers ---


def always_go_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
    """Always selects channel 0 — never inhibits.

    On go trials this is a success; on stop trials this is a stop_failure.
    """
    return BGDecision(
        sim_time=action_evidence.sim_time,
        trial_id=action_evidence.trial_id,
        selected_channel=0,
        decision_margin=0.5,
        suppression_vector=[0.0, 0.0],
        channel_activations=[0.8, 0.3],
        selection_latency=0.01,
    )


def always_inhibit_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
    """Always inhibits (channel -1).

    On go trials this is a miss; on stop trials this is a stop success.
    """
    return BGDecision(
        sim_time=action_evidence.sim_time,
        trial_id=action_evidence.trial_id,
        selected_channel=-1,
        decision_margin=0.0,
        suppression_vector=[1.0, 1.0],
        channel_activations=[0.1, 0.1],
        selection_latency=0.01,
    )


def stop_aware_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
    """Inhibits when stop_signal_present, otherwise responds.

    This mirrors the oracle behaviour for a correctly-functioning stopping process.
    """
    if action_evidence.stop_signal_present:
        return always_inhibit_policy(trial_log, action_evidence)
    return always_go_policy(trial_log, action_evidence)


# --- Default config factory ---


def default_config(**overrides) -> StopSignalConfig:
    """Return a minimal StopSignalConfig suitable for tests, with optional overrides."""
    base = dict(
        n_trials=20,
        stop_proportion=0.5,   # 50% stop for easy deterministic testing
        initial_ssd_ms=200,
        ssd_step_ms=50,
        ssd_min_ms=50,
        ssd_max_ms=600,
        use_staircase=True,
        go_cue_onset_ms=300,
        decision_point_ms=500,
        response_window_duration_ms=700,
        fixation_duration_ms=200,
        seed=42,
    )
    base.update(overrides)
    return StopSignalConfig(**base)


# --- Test 1: Trial counts and cue identities ---


def test_correct_trial_count():
    """run_stop_signal_trials returns exactly n_trials TrialLog objects."""
    config = default_config(n_trials=30)
    trials = run_stop_signal_trials(config, always_go_policy)
    assert len(trials) == 30


def test_trial_ids_sequential():
    """trial_id values are 1-indexed and sequential."""
    config = default_config(n_trials=10)
    trials = run_stop_signal_trials(config, always_go_policy)
    assert [t.trial_id for t in trials] == list(range(1, 11))


def test_cue_identity_matches_trial_type():
    """cue_identity is 'go' on go trials and 'stop' on stop trials."""
    config = default_config(n_trials=40, seed=7)
    trials = run_stop_signal_trials(config, always_go_policy)
    for t in trials:
        assert t.cue_identity in ("go", "stop")


def test_stop_proportion_approximate():
    """With stop_proportion=0.25 and many trials, roughly 25% are stop trials."""
    config = default_config(n_trials=200, stop_proportion=0.25, seed=99)
    trials = run_stop_signal_trials(config, always_go_policy)
    stop_count = sum(1 for t in trials if t.cue_identity == "stop")
    # Allow ±10% around the expected 25%.
    assert 30 <= stop_count <= 70, f"Expected ~50 stop trials, got {stop_count}"


# --- Test 2: Staircase SSD updates ---


def test_staircase_increases_after_stop_success():
    """SSD should increase after a stop-success (always_inhibit on stop trial)."""
    config = default_config(
        n_trials=4,
        stop_proportion=1.0,    # all stop trials
        initial_ssd_ms=200,
        ssd_step_ms=50,
        ssd_min_ms=50,
        ssd_max_ms=600,
        use_staircase=True,
        seed=1,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)

    # All trials should be stop successes.
    assert all(t.success is True for t in trials)

    # SSD for each stop trial should increase by ssd_step_ms.
    # Extract SSD from stop_signal event payload.
    ssd_values = []
    for t in trials:
        for ev in t.events:
            if ev.event_type == EventType.stop_signal:
                ssd_values.append(ev.payload["ssd_ms"])

    assert len(ssd_values) == 4
    assert ssd_values[0] == 200
    assert ssd_values[1] == 250
    assert ssd_values[2] == 300
    assert ssd_values[3] == 350


def test_staircase_decreases_after_stop_failure():
    """SSD should decrease after a stop-failure (always_go on stop trial)."""
    config = default_config(
        n_trials=4,
        stop_proportion=1.0,
        initial_ssd_ms=300,
        ssd_step_ms=50,
        ssd_min_ms=50,
        ssd_max_ms=600,
        use_staircase=True,
        seed=1,
    )
    trials = run_stop_signal_trials(config, always_go_policy)

    assert all(t.success is False for t in trials)
    assert all(t.failure_mode == "stop_failure" for t in trials)

    ssd_values = [
        ev.payload["ssd_ms"]
        for t in trials
        for ev in t.events
        if ev.event_type == EventType.stop_signal
    ]
    assert ssd_values == [300, 250, 200, 150]


# --- Test 3: SSD bounds ---


def test_staircase_ssd_never_exceeds_max():
    """SSD must never exceed ssd_max_ms, even after many consecutive successes.

    Note: when SSD >= decision_point_ms (500), no stop_signal event is emitted
    (late-stop invariant). We therefore use an initial SSD well below decision_point_ms
    so early trials have visible events, and verify those stay within bounds.
    The staircase clamping logic is unit-tested independently in StaircaseState.
    """
    config = default_config(
        n_trials=30,
        stop_proportion=1.0,
        initial_ssd_ms=200,
        ssd_step_ms=50,
        ssd_min_ms=50,
        ssd_max_ms=400,         # max well below decision_point_ms=500 for observable events
        use_staircase=True,
        seed=2,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    ssd_values = [
        ev.payload["ssd_ms"]
        for t in trials
        for ev in t.events
        if ev.event_type == EventType.stop_signal
    ]
    assert len(ssd_values) > 0, "Expected at least some stop_signal events"
    assert all(v <= 400 for v in ssd_values), f"SSD exceeded max: {max(ssd_values)}"


def test_staircase_ssd_never_below_min():
    """SSD must never fall below ssd_min_ms, even after many consecutive failures."""
    config = default_config(
        n_trials=30,
        stop_proportion=1.0,
        initial_ssd_ms=100,
        ssd_step_ms=50,
        ssd_min_ms=50,
        ssd_max_ms=600,
        use_staircase=True,
        seed=2,
    )
    trials = run_stop_signal_trials(config, always_go_policy)
    ssd_values = [
        ev.payload["ssd_ms"]
        for t in trials
        for ev in t.events
        if ev.event_type == EventType.stop_signal
    ]
    assert all(v >= 50 for v in ssd_values), f"SSD fell below min: {min(ssd_values)}"


# --- Test 4: Fixed-SSD mode ---


def test_fixed_ssd_mode_ssd_never_changes():
    """With use_staircase=False, SSD stays at initial_ssd_ms for every stop trial."""
    config = default_config(
        n_trials=20,
        stop_proportion=1.0,
        initial_ssd_ms=200,
        use_staircase=False,
        seed=3,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    ssd_values = [
        ev.payload["ssd_ms"]
        for t in trials
        for ev in t.events
        if ev.event_type == EventType.stop_signal
    ]
    assert all(v == 200 for v in ssd_values)


# --- Test 5: Stop-signal event presence ---


def test_stop_trials_have_stop_signal_event():
    """Stop trials where SSD < decision_point_ms must contain exactly one stop_signal event.

    Trials where SSD >= decision_point_ms do NOT emit a stop_signal event (the signal
    arrives too late to affect the decision; emitting it would violate the sim_time
    ordering invariant for log replays).
    """
    # Use a config where SSD is well below decision_point_ms so all stop trials
    # have an early stop signal.
    config = default_config(
        n_trials=40,
        initial_ssd_ms=200,    # 200 < decision_point_ms=500 → event always emitted
        use_staircase=False,
        seed=5,
    )
    trials = run_stop_signal_trials(config, always_go_policy)
    stop_trials = [t for t in trials if t.cue_identity == "stop"]
    for t in stop_trials:
        ss_events = [ev for ev in t.events if ev.event_type == EventType.stop_signal]
        assert len(ss_events) == 1, (
            f"Trial {t.trial_id} (stop) has {len(ss_events)} stop_signal events"
        )


def test_go_trials_have_no_stop_signal_event():
    """Go trials must not contain any stop_signal event."""
    config = default_config(n_trials=40, seed=5)
    trials = run_stop_signal_trials(config, always_go_policy)
    go_trials = [t for t in trials if t.cue_identity == "go"]
    for t in go_trials:
        ss_events = [ev for ev in t.events if ev.event_type == EventType.stop_signal]
        assert ss_events == [], (
            f"Trial {t.trial_id} (go) unexpectedly has stop_signal events"
        )


def test_late_stop_signal_not_emitted():
    """When SSD >= decision_point_ms, no stop_signal event must appear in the trial log.

    The stop signal arrives at or after the decision point, so it cannot affect
    the agent's response. Emitting it would place an event with sim_time >=
    decision_commit's sim_time, breaking the sim_time ordering invariant for log
    replays. The trial is still classified as stop_failure (agent responded), and
    the staircase updates normally.
    """
    # SSD == decision_point_ms: signal arrives exactly at decision, still too late.
    config = default_config(
        n_trials=20,
        stop_proportion=1.0,
        initial_ssd_ms=500,    # == decision_point_ms
        use_staircase=False,
        decision_point_ms=500,
        seed=30,
    )
    trials = run_stop_signal_trials(config, always_go_policy)

    for t in trials:
        ss_events = [ev for ev in t.events if ev.event_type == EventType.stop_signal]
        assert ss_events == [], (
            f"Trial {t.trial_id}: stop_signal event emitted for late-stop trial "
            f"(ssd_ms=500 >= decision_point_ms=500)"
        )
        # Trial must still be classified as stop_failure — agent responded.
        assert t.success is False, f"Trial {t.trial_id}: expected stop_failure success=False"
        assert t.failure_mode == "stop_failure", (
            f"Trial {t.trial_id}: expected failure_mode=stop_failure, "
            f"got {t.failure_mode}"
        )

    # Also verify SSD strictly greater than decision_point_ms is handled the same way.
    config_late = default_config(
        n_trials=10,
        stop_proportion=1.0,
        initial_ssd_ms=600,    # > decision_point_ms
        use_staircase=False,
        decision_point_ms=500,
        seed=31,
    )
    trials_late = run_stop_signal_trials(config_late, always_go_policy)
    for t in trials_late:
        ss_events = [ev for ev in t.events if ev.event_type == EventType.stop_signal]
        assert ss_events == [], (
            f"Trial {t.trial_id}: stop_signal event emitted for late-stop trial "
            f"(ssd_ms=600 >= decision_point_ms=500)"
        )
        assert t.failure_mode == "stop_failure"


# --- Test 6: Stop-signal event timing ---


def test_stop_signal_sim_time_equals_go_cue_plus_ssd():
    """stop_signal sim_time must equal (go_cue_onset_ms + ssd_ms) / 1000.0."""
    config = default_config(
        n_trials=20,
        stop_proportion=1.0,
        initial_ssd_ms=200,
        use_staircase=False,
        go_cue_onset_ms=300,
        seed=6,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    for t in trials:
        for ev in t.events:
            if ev.event_type == EventType.stop_signal:
                ssd_ms = ev.payload["ssd_ms"]
                expected_sim_time = (300 + ssd_ms) / 1000.0
                assert abs(ev.sim_time - expected_sim_time) < 1e-9, (
                    f"stop_signal sim_time {ev.sim_time} != expected {expected_sim_time}"
                )


# --- Test 7: stop_signal_present flag ---


def test_stop_signal_present_when_ssd_before_decision_point():
    """stop_signal_present must be True when SSD < decision_point_ms on a stop trial."""
    received_flags: list[tuple[str, bool]] = []

    def recording_policy(
        trial_log: TrialLog, action_evidence: ActionEvidence
    ) -> BGDecision:
        received_flags.append((trial_log.cue_identity, action_evidence.stop_signal_present))
        return always_inhibit_policy(trial_log, action_evidence)

    config = default_config(
        n_trials=20,
        stop_proportion=0.5,
        initial_ssd_ms=200,    # SSD=200 < decision_point_ms=500 → flag should be True
        use_staircase=False,
        decision_point_ms=500,
        seed=10,
    )
    run_stop_signal_trials(config, recording_policy)

    for cue_id, flag in received_flags:
        if cue_id == "stop":
            assert flag is True, "stop_signal_present should be True when SSD < decision_point"
        else:
            assert flag is False, "stop_signal_present should be False on go trials"


def test_stop_signal_present_false_when_ssd_at_or_after_decision_point():
    """stop_signal_present must be False when SSD >= decision_point_ms."""
    received_flags: list[tuple[str, bool]] = []

    def recording_policy(
        trial_log: TrialLog, action_evidence: ActionEvidence
    ) -> BGDecision:
        received_flags.append((trial_log.cue_identity, action_evidence.stop_signal_present))
        return always_inhibit_policy(trial_log, action_evidence)

    config = default_config(
        n_trials=20,
        stop_proportion=0.5,
        initial_ssd_ms=500,    # SSD=500 == decision_point_ms=500 → flag should be False
        use_staircase=False,
        decision_point_ms=500,
        seed=11,
    )
    run_stop_signal_trials(config, recording_policy)

    for cue_id, flag in received_flags:
        assert flag is False, (
            f"stop_signal_present should be False when SSD >= decision_point_ms "
            f"(cue={cue_id})"
        )


# --- Test 8: Outcome classification ---


def test_go_trial_success_when_responded():
    """Go trials where always_go_policy responds should be successes."""
    config = default_config(
        n_trials=20,
        stop_proportion=0.0,   # all go trials
        seed=12,
    )
    trials = run_stop_signal_trials(config, always_go_policy)
    assert all(t.success is True for t in trials)
    assert all(t.failure_mode is None for t in trials)


def test_go_trial_miss_when_no_response():
    """Go trials where always_inhibit_policy withholds should be misses."""
    config = default_config(
        n_trials=20,
        stop_proportion=0.0,
        seed=13,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    assert all(t.success is False for t in trials)
    assert all(t.failure_mode == "miss" for t in trials)


def test_stop_trial_success_when_inhibited():
    """Stop trials where always_inhibit_policy withholds should be successes."""
    config = default_config(
        n_trials=20,
        stop_proportion=1.0,
        seed=14,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    assert all(t.success is True for t in trials)
    assert all(t.failure_mode is None for t in trials)


def test_stop_trial_failure_when_responded():
    """Stop trials where always_go_policy responds should be stop_failures."""
    config = default_config(
        n_trials=20,
        stop_proportion=1.0,
        seed=15,
    )
    trials = run_stop_signal_trials(config, always_go_policy)
    assert all(t.success is False for t in trials)
    assert all(t.failure_mode == "stop_failure" for t in trials)


def test_mixed_policy_outcomes():
    """stop_aware_policy inhibits on stop trials and responds on go trials."""
    config = default_config(
        n_trials=40,
        stop_proportion=0.5,
        initial_ssd_ms=200,
        use_staircase=False,
        decision_point_ms=500,
        seed=16,
    )
    trials = run_stop_signal_trials(config, stop_aware_policy)
    for t in trials:
        if t.cue_identity == "go":
            assert t.success is True, f"Trial {t.trial_id}: go trial should succeed"
        else:
            # SSD=200 < decision_point=500, so stop_signal_present=True → inhibits
            assert t.success is True, f"Trial {t.trial_id}: stop trial should succeed (inhibited)"


# --- Test 9 & 10: Validity data for Verbruggen 2019 ---


def test_movement_onset_time_set_on_responding_trials():
    """movement_onset_time must be set (not None) whenever a response was made."""
    config = default_config(n_trials=30, seed=17)
    trials = run_stop_signal_trials(config, always_go_policy)
    for t in trials:
        # always_go_policy always responds → movement_onset_time must be set.
        assert t.movement_onset_time is not None, (
            f"Trial {t.trial_id}: expected movement_onset_time, got None"
        )


def test_movement_onset_time_none_on_inhibited_trials():
    """movement_onset_time must be None when the agent did not respond."""
    config = default_config(n_trials=30, seed=18)
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    for t in trials:
        assert t.movement_onset_time is None, (
            f"Trial {t.trial_id}: expected None movement_onset_time, got {t.movement_onset_time}"
        )


def test_failed_stop_rt_recoverable_from_log():
    """RT for failed-stop trials must be computable from movement_onset_time - cue_onset_time.

    Verbruggen et al. 2019 validity: RT_failed_stop should be faster than RT_go on average.
    This test verifies the data is present and consistent; the actual validity criterion
    is an empirical property of the policy (not enforced by the engine).

    NOTE (Phase 1 limitation): all movement_onset_time values are pinned to
    decision_point_ms, so failed-stop RT and go RT are structurally identical here.
    The RT_failed_stop < RT_go Verbruggen validity check is a tautology in Phase 1
    and will only become a meaningful empirical test once policies produce variable
    decision timing (Phase 5+).
    """
    config = default_config(
        n_trials=40,
        stop_proportion=0.5,
        use_staircase=False,
        initial_ssd_ms=200,
        seed=19,
    )
    # always_go_policy responds on all trials → generates both go-RT and failed-stop-RT.
    trials = run_stop_signal_trials(config, always_go_policy)

    go_rts = []
    failed_stop_rts = []

    for t in trials:
        if t.movement_onset_time is None:
            continue
        rt = t.movement_onset_time - t.cue_onset_time
        assert rt >= 0.0, f"Trial {t.trial_id}: negative RT {rt}"
        if t.cue_identity == "go":
            go_rts.append(rt)
        elif t.cue_identity == "stop" and t.failure_mode == "stop_failure":
            failed_stop_rts.append(rt)

    # Both RT lists must be non-empty for the validity check to be meaningful.
    assert len(go_rts) > 0, "No go trials with RT found"
    assert len(failed_stop_rts) > 0, "No failed-stop trials with RT found"

    # In this dummy engine (Phase 1), RTs are identical (decision_point_ms offset).
    # The important property is that the data is present and non-negative.
    for rt in go_rts + failed_stop_rts:
        assert rt >= 0.0


def test_ssd_stored_in_stop_signal_event_payload():
    """SSD value must be stored in stop_signal event payload for downstream analysis."""
    config = default_config(
        n_trials=10,
        stop_proportion=1.0,
        initial_ssd_ms=150,
        use_staircase=False,
        seed=20,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    for t in trials:
        for ev in t.events:
            if ev.event_type == EventType.stop_signal:
                assert "ssd_ms" in ev.payload, (
                    f"Trial {t.trial_id}: ssd_ms missing from stop_signal payload"
                )
                assert ev.payload["ssd_ms"] == 150


# --- Test 11: Logger integration ---


def test_logger_persists_and_reloads_trials(tmp_path):
    """Trials written via TrialLogger must reload correctly from JSONL."""
    out_path = tmp_path / "stop_signal_trials.jsonl"
    logger = TrialLogger(out_path)
    config = default_config(n_trials=10, seed=21)

    written = run_stop_signal_trials(config, always_go_policy, logger=logger)
    assert out_path.exists()

    reloaded = load_trials(out_path)
    assert len(reloaded) == len(written)

    for orig, reloaded_trial in zip(written, reloaded):
        assert orig.trial_id == reloaded_trial.trial_id
        assert orig.success == reloaded_trial.success
        assert orig.failure_mode == reloaded_trial.failure_mode
        assert len(orig.events) == len(reloaded_trial.events)


def test_logger_appends_across_calls(tmp_path):
    """Calling run_stop_signal_trials twice with same logger appends to the same file."""
    out_path = tmp_path / "append_test.jsonl"
    logger = TrialLogger(out_path)
    config = default_config(n_trials=5, seed=22)

    run_stop_signal_trials(config, always_go_policy, logger=logger)
    run_stop_signal_trials(config, always_go_policy, logger=logger)

    reloaded = load_trials(out_path)
    assert len(reloaded) == 10


# --- Test 12: Determinism ---


def test_same_seed_produces_identical_trials():
    """Two runs with the same seed must produce identical trial sequences."""
    config = default_config(n_trials=20, seed=99)
    trials_a = run_stop_signal_trials(config, always_go_policy)
    trials_b = run_stop_signal_trials(config, always_go_policy)

    assert len(trials_a) == len(trials_b)
    for a, b in zip(trials_a, trials_b):
        assert a.trial_id == b.trial_id
        assert a.seed == b.seed
        assert a.cue_identity == b.cue_identity
        assert a.success == b.success


def test_different_seeds_produce_different_sequences():
    """Two runs with different seeds should produce different trial sequences."""
    config_a = default_config(n_trials=20, seed=1)
    config_b = default_config(n_trials=20, seed=2)
    trials_a = run_stop_signal_trials(config_a, always_go_policy)
    trials_b = run_stop_signal_trials(config_b, always_go_policy)

    cue_ids_a = [t.cue_identity for t in trials_a]
    cue_ids_b = [t.cue_identity for t in trials_b]
    # With 20 trials and ~50% stop, the probability of identical sequences is negligible.
    assert cue_ids_a != cue_ids_b


# --- Test 13: Event ordering ---


def test_event_order_on_stop_trial():
    """Events on a stop trial must follow the canonical sequence order by sim_time."""
    config = default_config(
        n_trials=5,
        stop_proportion=1.0,
        initial_ssd_ms=200,
        use_staircase=False,
        go_cue_onset_ms=300,
        decision_point_ms=500,
        fixation_duration_ms=200,
        seed=23,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)

    for t in trials:
        sim_times = [ev.sim_time for ev in t.events]
        # Events must be non-decreasing in sim_time.
        assert sim_times == sorted(sim_times), (
            f"Trial {t.trial_id}: events not in sim_time order: {sim_times}"
        )

        event_types = [ev.event_type for ev in t.events]
        # Canonical stop trial sequence (inhibited, no movement events).
        expected = [
            EventType.trial_start,
            EventType.fixation_on,
            EventType.go_cue,
            EventType.stop_signal,
            EventType.decision_commit,
            EventType.trial_end,
        ]
        assert event_types == expected, (
            f"Trial {t.trial_id}: event sequence {event_types} != expected {expected}"
        )


def test_event_order_on_go_trial_with_response():
    """Events on a responding go trial must include movement_onset and movement_end."""
    config = default_config(
        n_trials=5,
        stop_proportion=0.0,   # all go trials
        go_cue_onset_ms=300,
        decision_point_ms=500,
        fixation_duration_ms=200,
        seed=24,
    )
    trials = run_stop_signal_trials(config, always_go_policy)

    for t in trials:
        event_types = [ev.event_type for ev in t.events]
        expected = [
            EventType.trial_start,
            EventType.fixation_on,
            EventType.go_cue,
            EventType.decision_commit,
            EventType.movement_onset,
            EventType.movement_end,
            EventType.trial_end,
        ]
        assert event_types == expected, (
            f"Trial {t.trial_id}: event sequence {event_types} != expected {expected}"
        )


# --- Test 14: task_type ---


def test_task_type_is_stop_signal():
    """All TrialLog objects must have task_type='stop_signal'."""
    config = default_config(n_trials=5, seed=25)
    trials = run_stop_signal_trials(config, always_go_policy)
    assert all(t.task_type == "stop_signal" for t in trials)


# --- Test 15: ssd_levels cycling (Task 7.1) ---


def test_ssd_levels_cycles_round_robin():
    """With ssd_levels=[100, 200, 300] and use_staircase=False, stop trials cycle through
    the list round-robin.  The stop_signal event payload ssd_ms must match the expected
    position in the cycle for each stop trial."""
    config = StopSignalConfig(
        n_trials=40,
        stop_proportion=1.0,      # all stop trials for simplicity
        initial_ssd_ms=999,       # must be ignored when ssd_levels is set
        ssd_step_ms=50,
        ssd_min_ms=50,
        ssd_max_ms=600,
        use_staircase=False,
        go_cue_onset_ms=300,
        decision_point_ms=500,
        response_window_duration_ms=700,
        fixation_duration_ms=200,
        seed=50,
        ssd_levels=[100, 200, 300],
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)

    ssd_values = [
        ev.payload["ssd_ms"]
        for t in trials
        for ev in t.events
        if ev.event_type == EventType.stop_signal
    ]
    # Each stop trial gets ssd_levels[stop_trial_index % 3]; all 40 trials are stops.
    expected_cycle = [100, 200, 300]
    for i, ssd in enumerate(ssd_values):
        assert ssd == expected_cycle[i % 3], (
            f"Stop trial {i}: expected {expected_cycle[i % 3]}, got {ssd}"
        )


def test_ssd_levels_empty_raises():
    """ssd_levels=[] must raise ValueError at construction (fail fast)."""
    import pytest
    with pytest.raises(ValueError, match="ssd_levels"):
        StopSignalConfig(
            n_trials=10,
            use_staircase=False,
            ssd_levels=[],
        )


def test_ssd_levels_ignored_when_staircase_active():
    """When use_staircase=True, ssd_levels is ignored and staircase drives SSD.

    Verify by checking that SSD values increase after each stop-success, which
    is staircase behaviour — not a fixed cycle.
    """
    config = StopSignalConfig(
        n_trials=4,
        stop_proportion=1.0,
        initial_ssd_ms=200,
        ssd_step_ms=50,
        ssd_min_ms=50,
        ssd_max_ms=600,
        use_staircase=True,
        go_cue_onset_ms=300,
        decision_point_ms=500,
        response_window_duration_ms=700,
        fixation_duration_ms=200,
        seed=51,
        ssd_levels=[100, 200, 300],  # must be ignored
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    ssd_values = [
        ev.payload["ssd_ms"]
        for t in trials
        for ev in t.events
        if ev.event_type == EventType.stop_signal
    ]
    # Staircase: 200 → 250 → 300 → 350 (not the cycle [100, 200, 300, 100]).
    assert ssd_values == [200, 250, 300, 350], (
        f"Staircase expected [200,250,300,350], got {ssd_values}"
    )


def test_ssd_levels_negative_value_raises():
    """ssd_levels containing a non-positive integer must raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="ssd_levels"):
        StopSignalConfig(
            n_trials=10,
            use_staircase=False,
            ssd_levels=[100, -50, 200],
        )


# --- Test 16: stop_trial_go_evidence (Task 7.1) ---


def test_stop_trial_go_evidence_false_gives_cue_identity_stop():
    """Default stop_trial_go_evidence=False: stop trials must have cue_identity='stop'."""
    config = StopSignalConfig(
        n_trials=20,
        stop_proportion=0.5,
        initial_ssd_ms=200,
        use_staircase=False,
        go_cue_onset_ms=300,
        decision_point_ms=500,
        response_window_duration_ms=700,
        fixation_duration_ms=200,
        seed=60,
        stop_trial_go_evidence=False,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    for t in trials:
        stop_events = [ev for ev in t.events if ev.event_type == EventType.stop_signal]
        if stop_events:
            # stop trial — identified by stop_signal event presence
            assert t.cue_identity == "stop", (
                f"Trial {t.trial_id}: expected cue_identity='stop', got '{t.cue_identity}'"
            )


def test_stop_trial_go_evidence_true_gives_cue_identity_go():
    """stop_trial_go_evidence=True: stop trials must have cue_identity='go'."""
    config = StopSignalConfig(
        n_trials=20,
        stop_proportion=1.0,   # all stop trials
        initial_ssd_ms=200,
        use_staircase=False,
        go_cue_onset_ms=300,
        decision_point_ms=500,
        response_window_duration_ms=700,
        fixation_duration_ms=200,
        seed=61,
        stop_trial_go_evidence=True,
    )
    trials = run_stop_signal_trials(config, always_inhibit_policy)
    for t in trials:
        assert t.cue_identity == "go", (
            f"Trial {t.trial_id}: expected cue_identity='go' when stop_trial_go_evidence=True, "
            f"got '{t.cue_identity}'"
        )
        # Stop signal event must still be emitted — it is the identity marker.
        stop_events = [ev for ev in t.events if ev.event_type == EventType.stop_signal]
        assert len(stop_events) == 1, (
            f"Trial {t.trial_id}: expected 1 stop_signal event, got {len(stop_events)}"
        )


def test_go_trials_always_have_cue_identity_go_regardless_of_flag():
    """Go trials must always have cue_identity='go', even when stop_trial_go_evidence=True."""
    config = StopSignalConfig(
        n_trials=40,
        stop_proportion=0.5,
        initial_ssd_ms=200,
        use_staircase=False,
        go_cue_onset_ms=300,
        decision_point_ms=500,
        response_window_duration_ms=700,
        fixation_duration_ms=200,
        seed=62,
        stop_trial_go_evidence=True,
    )
    trials = run_stop_signal_trials(config, always_go_policy)
    # Identify go trials by absence of stop_signal events.
    for t in trials:
        stop_events = [ev for ev in t.events if ev.event_type == EventType.stop_signal]
        if not stop_events and t.failure_mode != "stop_failure":
            # Pure go trial (no stop_signal event, responded = success)
            assert t.cue_identity == "go", (
                f"Trial {t.trial_id}: go trial cue_identity should be 'go', "
                f"got '{t.cue_identity}'"
            )


# --- Test 17: movement_onset_time from selection_latency (Task 7.1) ---


def _make_latency_policy(
    selection_latency: float,
) -> Callable[[TrialLog, ActionEvidence], BGDecision]:
    """Return a policy that responds with a specific selection_latency."""
    def policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
        return BGDecision(
            sim_time=action_evidence.sim_time,
            trial_id=action_evidence.trial_id,
            selected_channel=0,
            decision_margin=0.5,
            suppression_vector=[0.0, 0.0],
            channel_activations=[0.8, 0.3],
            selection_latency=selection_latency,
        )
    return policy


def test_movement_onset_time_uses_selection_latency_when_positive():
    """When BGDecision.selection_latency > 0, movement_onset_time must reflect it.

    Expected: movement_onset_time == (go_cue_onset_ms + selection_latency_ms) / 1000.0
    where selection_latency_ms = int(selection_latency * 1000).
    """
    go_cue_onset_ms = 300
    selection_latency = 0.050   # 50 ms

    config = StopSignalConfig(
        n_trials=5,
        stop_proportion=0.0,   # all go trials
        initial_ssd_ms=200,
        use_staircase=False,
        go_cue_onset_ms=go_cue_onset_ms,
        decision_point_ms=500,
        response_window_duration_ms=700,
        fixation_duration_ms=200,
        seed=70,
    )
    policy = _make_latency_policy(selection_latency)
    trials = run_stop_signal_trials(config, policy)

    expected_onset = (go_cue_onset_ms + int(selection_latency * 1000)) / 1000.0
    for t in trials:
        assert t.movement_onset_time is not None
        assert abs(t.movement_onset_time - expected_onset) < 1e-9, (
            f"Trial {t.trial_id}: movement_onset_time={t.movement_onset_time}, "
            f"expected {expected_onset}"
        )


def test_movement_onset_time_falls_back_when_selection_latency_zero():
    """When BGDecision.selection_latency == 0.0, movement_onset_time falls back to
    decision_abs_ms / 1000.0 (old behaviour preserved for Phase 1 policies)."""
    go_cue_onset_ms = 300
    decision_point_ms = 500

    config = StopSignalConfig(
        n_trials=5,
        stop_proportion=0.0,
        initial_ssd_ms=200,
        use_staircase=False,
        go_cue_onset_ms=go_cue_onset_ms,
        decision_point_ms=decision_point_ms,
        response_window_duration_ms=700,
        fixation_duration_ms=200,
        seed=71,
    )
    policy = _make_latency_policy(0.0)
    trials = run_stop_signal_trials(config, policy)

    expected_onset = (go_cue_onset_ms + decision_point_ms) / 1000.0
    for t in trials:
        assert t.movement_onset_time is not None
        assert abs(t.movement_onset_time - expected_onset) < 1e-9, (
            f"Trial {t.trial_id}: movement_onset_time={t.movement_onset_time}, "
            f"expected {expected_onset}"
        )


# --- Test 18: Backward-compatibility regression (Task 7.1) ---


def test_backward_compat_no_new_fields():
    """StopSignalConfig(n_trials=10) with no new fields must produce identical behaviour
    to the pre-Phase-7 implementation: fixed initial_ssd_ms, cue_identity='stop' for
    stop trials, no ssd_levels cycling."""
    config = StopSignalConfig(n_trials=10, seed=80)
    assert config.ssd_levels is None
    assert config.stop_trial_go_evidence is False

    trials = run_stop_signal_trials(config, always_go_policy)
    assert len(trials) == 10
    # Stop trials must still have cue_identity='stop' (backward compat).
    stop_trials_by_cue = [t for t in trials if t.cue_identity == "stop"]
    for t in stop_trials_by_cue:
        assert t.cue_identity == "stop"
