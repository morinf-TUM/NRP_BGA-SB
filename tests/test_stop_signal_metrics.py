"""Tests for stop_signal_metrics.py — TDD suite for Task 7.2.

Synthetic TrialLog objects are constructed directly; no engine invocation is needed.
All timing values use seconds (matching TrialLog schema fields).
"""

from __future__ import annotations

import math

import pytest

from nrp_bga_sb.schemas import EventType, TaskEvent, TrialLog
from nrp_bga_sb.stop_signal_metrics import (
    StopSignalMetrics,
    StopSignalValidityReport,
    _has_stop_signal_event,
    _rt_s,
    cancellation_latency_mean,
    compute_stop_signal_metrics,
    estimate_ssrt,
    extract_ssd_ms,
    failed_stop_rt_mean,
    go_rt_stats,
    inhibition_function,
    is_go_trial,
    is_stop_trial,
    trigger_failure_rate,
    validate_stop_signal_data,
)

# ---------------------------------------------------------------------------
# Synthetic TrialLog helpers
# ---------------------------------------------------------------------------

_CUE_ONSET = 0.300  # go_cue onset at 300 ms — used as the cue_onset_time baseline


def _make_event(
    event_type: EventType,
    sim_time: float,
    trial_id: int = 1,
    payload: dict | None = None,
) -> TaskEvent:
    return TaskEvent(
        event_type=event_type,
        sim_time=sim_time,
        real_time=sim_time,
        trial_id=trial_id,
        payload=payload or {},
    )


def make_go_success(movement_onset_time: float, cue_onset_time: float = _CUE_ONSET) -> TrialLog:
    """A go trial where the agent responded successfully."""
    return TrialLog(
        trial_id=1,
        seed=0,
        task_type="stop_signal",
        cue_identity="go",
        cue_onset_time=cue_onset_time,
        success=True,
        failure_mode=None,
        movement_onset_time=movement_onset_time,
        events=[
            _make_event(EventType.go_cue, cue_onset_time),
            _make_event(EventType.decision_commit, cue_onset_time + 0.050),
            _make_event(EventType.movement_onset, movement_onset_time),
        ],
    )


def make_go_miss() -> TrialLog:
    """A go trial where the agent failed to respond (miss)."""
    return TrialLog(
        trial_id=2,
        seed=1,
        task_type="stop_signal",
        cue_identity="go",
        cue_onset_time=_CUE_ONSET,
        success=False,
        failure_mode="miss",
        movement_onset_time=None,
        events=[
            _make_event(EventType.go_cue, _CUE_ONSET),
            _make_event(EventType.decision_commit, _CUE_ONSET + 0.050),
        ],
    )


def make_stop_success_early(
    ssd_ms: int,
    decision_point_ms: int = 500,
    cue_onset_time: float = _CUE_ONSET,
) -> TrialLog:
    """A stop trial where: SSD < decision_point_ms, agent successfully inhibited.

    The stop_signal event is present (early stop). The decision_commit event
    follows the stop_signal event, representing the moment the engine checked.
    """
    stop_signal_time = cue_onset_time + ssd_ms / 1000.0
    decision_commit_time = cue_onset_time + decision_point_ms / 1000.0
    return TrialLog(
        trial_id=3,
        seed=2,
        task_type="stop_signal",
        cue_identity="stop",
        cue_onset_time=cue_onset_time,
        success=True,
        failure_mode=None,
        movement_onset_time=None,  # inhibited — no movement
        events=[
            _make_event(EventType.go_cue, cue_onset_time),
            _make_event(EventType.stop_signal, stop_signal_time, payload={"ssd_ms": ssd_ms}),
            _make_event(EventType.decision_commit, decision_commit_time),
        ],
    )


def make_stop_failure_early(
    ssd_ms: int,
    movement_onset_time: float,
    cue_onset_time: float = _CUE_ONSET,
    decision_point_ms: int = 500,
) -> TrialLog:
    """A stop trial where: SSD < decision_point_ms, agent responded (stop failure).

    The stop_signal event is present because the signal arrived before the decision.
    """
    stop_signal_time = cue_onset_time + ssd_ms / 1000.0
    decision_commit_time = cue_onset_time + decision_point_ms / 1000.0
    return TrialLog(
        trial_id=4,
        seed=3,
        task_type="stop_signal",
        cue_identity="stop",
        cue_onset_time=cue_onset_time,
        success=False,
        failure_mode="stop_failure",
        movement_onset_time=movement_onset_time,
        events=[
            _make_event(EventType.go_cue, cue_onset_time),
            _make_event(EventType.stop_signal, stop_signal_time, payload={"ssd_ms": ssd_ms}),
            _make_event(EventType.decision_commit, decision_commit_time),
            _make_event(EventType.movement_onset, movement_onset_time),
        ],
    )


def make_stop_failure_late() -> TrialLog:
    """A stop trial where the stop signal arrived at or after the decision point.

    No stop_signal event is logged (engine omits it when SSD >= decision_point_ms).
    The trial is still classified as stop_failure because the agent responded.
    This simulates a mechanically unavoidable failure (trigger failure).
    """
    cue_onset_time = _CUE_ONSET
    decision_commit_time = cue_onset_time + 0.500
    movement_onset_time = cue_onset_time + 0.500  # responded at decision point
    return TrialLog(
        trial_id=5,
        seed=4,
        task_type="stop_signal",
        # cue_identity is "go" when stop_trial_go_evidence=True, but the trial
        # is still identified as a stop trial by failure_mode="stop_failure".
        cue_identity="go",
        cue_onset_time=cue_onset_time,
        success=False,
        failure_mode="stop_failure",
        movement_onset_time=movement_onset_time,
        events=[
            _make_event(EventType.go_cue, cue_onset_time),
            _make_event(EventType.decision_commit, decision_commit_time),
            _make_event(EventType.movement_onset, movement_onset_time),
            # No stop_signal event — it arrived at or after the decision point.
        ],
    )


# ---------------------------------------------------------------------------
# Test 1: is_stop_trial detects by cue_identity="stop"
# ---------------------------------------------------------------------------

def test_is_stop_trial_by_cue_identity() -> None:
    trial = TrialLog(
        trial_id=1, seed=0, task_type="stop_signal",
        cue_identity="stop", cue_onset_time=0.3,
        success=True, failure_mode=None,
    )
    assert is_stop_trial(trial) is True
    assert is_go_trial(trial) is False


# ---------------------------------------------------------------------------
# Test 2: is_stop_trial detects by failure_mode="stop_failure"
# ---------------------------------------------------------------------------

def test_is_stop_trial_by_failure_mode() -> None:
    # cue_identity="go" but failure_mode marks it as a stop trial
    trial = TrialLog(
        trial_id=1, seed=0, task_type="stop_signal",
        cue_identity="go", cue_onset_time=0.3,
        success=False, failure_mode="stop_failure",
    )
    assert is_stop_trial(trial) is True


# ---------------------------------------------------------------------------
# Test 3: is_stop_trial detects by stop_signal event presence
# ---------------------------------------------------------------------------

def test_is_stop_trial_by_event() -> None:
    # Neither cue_identity nor failure_mode marks it — but event is present
    trial = TrialLog(
        trial_id=1, seed=0, task_type="stop_signal",
        cue_identity="go", cue_onset_time=0.3,
        success=True, failure_mode=None,
        events=[
            _make_event(EventType.stop_signal, 0.45, payload={"ssd_ms": 150}),
        ],
    )
    assert is_stop_trial(trial) is True


# ---------------------------------------------------------------------------
# Test 4: is_go_trial correctly negates is_stop_trial
# ---------------------------------------------------------------------------

def test_is_go_trial_plain_go() -> None:
    trial = make_go_success(movement_onset_time=0.350)
    assert is_go_trial(trial) is True
    assert is_stop_trial(trial) is False


# ---------------------------------------------------------------------------
# Test 5: extract_ssd_ms returns ssd from payload; None when no event
# ---------------------------------------------------------------------------

def test_extract_ssd_ms_present() -> None:
    trial = make_stop_success_early(ssd_ms=150)
    assert extract_ssd_ms(trial) == 150


def test_extract_ssd_ms_absent() -> None:
    trial = make_go_success(movement_onset_time=0.350)
    assert extract_ssd_ms(trial) is None


def test_extract_ssd_ms_late_stop_no_event() -> None:
    # Late-stop trial: stop_failure but no stop_signal event → None
    trial = make_stop_failure_late()
    assert extract_ssd_ms(trial) is None


# ---------------------------------------------------------------------------
# Test 6: _rt_s returns None when no movement; correct value otherwise
# ---------------------------------------------------------------------------

def test_rt_s_none_when_no_movement() -> None:
    trial = make_go_miss()
    assert _rt_s(trial) is None


def test_rt_s_correct_value() -> None:
    cue = 0.300
    onset = 0.350
    trial = make_go_success(movement_onset_time=onset, cue_onset_time=cue)
    rt = _rt_s(trial)
    assert rt is not None
    assert math.isclose(rt, 0.050, abs_tol=1e-9)


def test_rt_s_clamp_negative_to_zero() -> None:
    # Pathological: movement_onset_time < cue_onset_time → clamp to 0.0
    trial = TrialLog(
        trial_id=1, seed=0, task_type="stop_signal",
        cue_identity="go", cue_onset_time=0.500,
        success=True, failure_mode=None,
        movement_onset_time=0.400,  # before cue onset — should not happen normally
    )
    assert _rt_s(trial) == 0.0


# ---------------------------------------------------------------------------
# Test 7: go_rt_stats correct mean/std; None for <2 trials
# ---------------------------------------------------------------------------

def test_go_rt_stats_correct() -> None:
    # Two go trials: RT = 0.050 s and 0.100 s
    t1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    t2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.100)
    stats = go_rt_stats([t1, t2])
    assert stats["n"] == 2
    assert math.isclose(stats["mean_s"], 0.075, abs_tol=1e-9)
    expected_std = math.sqrt(((0.050 - 0.075) ** 2 + (0.100 - 0.075) ** 2) / 2)
    assert math.isclose(stats["std_s"], expected_std, abs_tol=1e-9)


def test_go_rt_stats_none_for_single_trial() -> None:
    t1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    stats = go_rt_stats([t1])
    assert stats["mean_s"] is None
    assert stats["std_s"] is None
    assert stats["n"] == 1


def test_go_rt_stats_none_for_zero_trials() -> None:
    stats = go_rt_stats([])
    assert stats["mean_s"] is None
    assert stats["std_s"] is None
    assert stats["n"] == 0


# ---------------------------------------------------------------------------
# Test 8: go_rt_stats excludes stop-trial logs
# ---------------------------------------------------------------------------

def test_go_rt_stats_excludes_stop_trials() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.100)
    stop = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.030)
    stats = go_rt_stats([go1, go2, stop])
    assert stats["n"] == 2
    assert math.isclose(stats["mean_s"], 0.075, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 9: failed_stop_rt_mean correct; None when no failures
# ---------------------------------------------------------------------------

def test_failed_stop_rt_mean_correct() -> None:
    f1 = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.040)
    f2 = make_stop_failure_early(ssd_ms=200, movement_onset_time=_CUE_ONSET + 0.060)
    mean = failed_stop_rt_mean([f1, f2])
    assert mean is not None
    assert math.isclose(mean, 0.050, abs_tol=1e-9)


def test_failed_stop_rt_mean_none_when_no_failures() -> None:
    go = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    assert failed_stop_rt_mean([go]) is None


# ---------------------------------------------------------------------------
# Test 10: inhibition_function bins by SSD; excludes late-stops (ssd=None)
# ---------------------------------------------------------------------------

def test_inhibition_function_bins_by_ssd() -> None:
    # SSD 100 ms: 1 success, 1 failure → failure_rate = 0.5
    s1 = make_stop_success_early(ssd_ms=100)
    f1 = make_stop_failure_early(ssd_ms=100, movement_onset_time=_CUE_ONSET + 0.040)
    # SSD 200 ms: 1 failure → failure_rate = 1.0
    f2 = make_stop_failure_early(ssd_ms=200, movement_onset_time=_CUE_ONSET + 0.050)
    # Late stop: no ssd → excluded from dict
    late = make_stop_failure_late()

    result = inhibition_function([s1, f1, f2, late])
    assert set(result.keys()) == {100, 200}
    assert math.isclose(result[100]["failure_rate"], 0.5, abs_tol=1e-9)
    assert result[100]["n"] == 2
    assert math.isclose(result[200]["failure_rate"], 1.0, abs_tol=1e-9)
    assert result[200]["n"] == 1


def test_inhibition_function_excludes_late_stop_from_dict() -> None:
    late = make_stop_failure_late()
    go = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    result = inhibition_function([late, go])
    # Late stop has no SSD → not included in any SSD bin
    assert result == {}


# ---------------------------------------------------------------------------
# Test 11: inhibition_function failure_rate == 0.0 when all succeed at SSD
# ---------------------------------------------------------------------------

def test_inhibition_function_all_success() -> None:
    s1 = make_stop_success_early(ssd_ms=150)
    s2 = make_stop_success_early(ssd_ms=150)
    result = inhibition_function([s1, s2])
    assert 150 in result
    assert result[150]["failure_rate"] == 0.0
    assert result[150]["n"] == 2


# ---------------------------------------------------------------------------
# Test 12: inhibition_function failure_rate == 1.0 when all fail at SSD
# ---------------------------------------------------------------------------

def test_inhibition_function_all_fail() -> None:
    f1 = make_stop_failure_early(ssd_ms=300, movement_onset_time=_CUE_ONSET + 0.050)
    f2 = make_stop_failure_early(ssd_ms=300, movement_onset_time=_CUE_ONSET + 0.060)
    result = inhibition_function([f1, f2])
    assert 300 in result
    assert result[300]["failure_rate"] == 1.0
    assert result[300]["n"] == 2


# ---------------------------------------------------------------------------
# Test 13: estimate_ssrt = mean_go_rt - mean_ssd; None when <2 go trials
# ---------------------------------------------------------------------------

def test_estimate_ssrt_correct() -> None:
    # Go trials: RT = 0.050 s and 0.100 s → mean_go_rt = 0.075 s
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.100)
    # Stop trials: SSD = 100 ms and 200 ms → mean_ssd = 150 ms = 0.150 s
    s1 = make_stop_success_early(ssd_ms=100)
    f1 = make_stop_failure_early(ssd_ms=200, movement_onset_time=_CUE_ONSET + 0.040)

    ssrt = estimate_ssrt([go1, go2, s1, f1])
    assert ssrt is not None
    # SSRT = 0.075 - 0.150 (ms→s) = 0.075 - 0.150 = -0.075
    # mean_ssd = (100 + 200) / 2 = 150 ms = 0.150 s
    assert math.isclose(ssrt, 0.075 - 0.150, abs_tol=1e-9)


def test_estimate_ssrt_none_when_fewer_than_2_go_trials() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    s1 = make_stop_success_early(ssd_ms=150)
    assert estimate_ssrt([go1, s1]) is None


def test_estimate_ssrt_none_when_no_stop_trials() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    assert estimate_ssrt([go1, go2]) is None


# ---------------------------------------------------------------------------
# Test 14: estimate_ssrt excludes late-stop trials from SSD mean
# ---------------------------------------------------------------------------

def test_estimate_ssrt_excludes_late_stops_from_ssd_mean() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.100)
    # One early stop with SSD=100 ms, one late stop (no SSD)
    s1 = make_stop_success_early(ssd_ms=100)
    late = make_stop_failure_late()

    ssrt = estimate_ssrt([go1, go2, s1, late])
    assert ssrt is not None
    # mean_ssd uses only s1 → 100 ms = 0.100 s
    # mean_go_rt = 0.075 s
    # SSRT = 0.075 - 0.100 = -0.025
    assert math.isclose(ssrt, 0.075 - 0.100, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 15: cancellation_latency_mean correct
# ---------------------------------------------------------------------------

def test_cancellation_latency_mean_correct() -> None:
    # stop_signal at 0.450 s, decision_commit at 0.800 s → latency = 0.350 s
    cue = 0.300
    ssd_ms = 150   # stop_signal at 0.300 + 0.150 = 0.450 s
    dp_ms = 500    # decision_commit at 0.300 + 0.500 = 0.800 s
    trial = make_stop_success_early(ssd_ms=ssd_ms, decision_point_ms=dp_ms, cue_onset_time=cue)
    mean = cancellation_latency_mean([trial])
    assert mean is not None
    assert math.isclose(mean, 0.350, abs_tol=1e-9)


def test_cancellation_latency_mean_two_trials() -> None:
    # Trial 1: stop_signal=0.450, decision_commit=0.800 → latency=0.350
    # Trial 2: stop_signal=0.400, decision_commit=0.800 → latency=0.400
    t1 = make_stop_success_early(ssd_ms=150, decision_point_ms=500, cue_onset_time=0.300)
    t2 = make_stop_success_early(ssd_ms=100, decision_point_ms=500, cue_onset_time=0.300)
    mean = cancellation_latency_mean([t1, t2])
    assert mean is not None
    assert math.isclose(mean, (0.350 + 0.400) / 2, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 16: cancellation_latency_mean None when no early-stop successes
# ---------------------------------------------------------------------------

def test_cancellation_latency_mean_none_when_no_qualifying() -> None:
    go = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    fail = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.040)
    late = make_stop_failure_late()
    assert cancellation_latency_mean([go, fail, late]) is None


# ---------------------------------------------------------------------------
# Test 17: trigger_failure_rate = fraction without stop_signal event
# ---------------------------------------------------------------------------

def test_trigger_failure_rate_correct() -> None:
    # 2 early stop failures (have stop_signal event), 1 late stop failure (no event)
    f1 = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.040)
    f2 = make_stop_failure_early(ssd_ms=200, movement_onset_time=_CUE_ONSET + 0.050)
    late = make_stop_failure_late()
    rate = trigger_failure_rate([f1, f2, late])
    assert rate is not None
    # 1 out of 3 stop failures lack a stop_signal event
    assert math.isclose(rate, 1 / 3, abs_tol=1e-9)


def test_trigger_failure_rate_all_late() -> None:
    late1 = make_stop_failure_late()
    late2 = make_stop_failure_late()
    rate = trigger_failure_rate([late1, late2])
    assert rate is not None
    assert rate == 1.0


def test_trigger_failure_rate_none_late() -> None:
    # All stop failures have stop_signal events → rate = 0.0
    f1 = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.040)
    rate = trigger_failure_rate([f1])
    assert rate is not None
    assert rate == 0.0


# ---------------------------------------------------------------------------
# Test 18: trigger_failure_rate None when no stop failures
# ---------------------------------------------------------------------------

def test_trigger_failure_rate_none_when_no_failures() -> None:
    go = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    success = make_stop_success_early(ssd_ms=150)
    assert trigger_failure_rate([go, success]) is None


# ---------------------------------------------------------------------------
# Test 19: compute_stop_signal_metrics raises ValueError on empty list
# ---------------------------------------------------------------------------

def test_compute_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_stop_signal_metrics([])


# ---------------------------------------------------------------------------
# Test 20: compute_stop_signal_metrics integration test — mixed trial list
# ---------------------------------------------------------------------------

def test_compute_stop_signal_metrics_integration() -> None:
    """Full integration: mixed go/stop/late-stop trials → all fields populated correctly."""
    # Go trials: RT = 0.050, 0.080 → mean = 0.065 s
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    go_miss = make_go_miss()

    # Stop trials:
    #   SSD 100 ms: 1 success (early stop)
    #   SSD 200 ms: 1 failure (early stop, RT = 0.040 s)
    #   Late stop: 1 failure (no stop_signal event)
    s_early_success = make_stop_success_early(ssd_ms=100, decision_point_ms=500)
    s_early_failure = make_stop_failure_early(
        ssd_ms=200, movement_onset_time=_CUE_ONSET + 0.040
    )
    s_late_failure = make_stop_failure_late()

    trials = [go1, go2, go_miss, s_early_success, s_early_failure, s_late_failure]
    m = compute_stop_signal_metrics(trials)

    assert isinstance(m, StopSignalMetrics)

    # Counts
    assert m.n_trials == 6
    assert m.n_go_trials == 3
    assert m.n_stop_trials == 3

    # Go RT: only responding go trials = go1, go2 (RT = 0.050, 0.080)
    assert m.go_rt_mean_s is not None
    assert math.isclose(m.go_rt_mean_s, 0.065, abs_tol=1e-9)

    # Failed stop RT: early failure (RT=0.040) + late failure (RT=0.500) → mean=0.270
    # make_stop_failure_late: movement_onset_time = _CUE_ONSET + 0.500 = 0.800 s
    # RT_late = 0.800 - 0.300 = 0.500 s
    # mean = (0.040 + 0.500) / 2 = 0.270 s
    assert m.failed_stop_rt_mean_s is not None
    assert math.isclose(m.failed_stop_rt_mean_s, 0.270, abs_tol=1e-9)

    # Stop failure rate: 2 failures out of 3 stop trials
    assert m.stop_failure_rate is not None
    assert math.isclose(m.stop_failure_rate, 2 / 3, abs_tol=1e-9)

    # Inhibition function: SSD bins 100 and 200 (late-stop excluded)
    assert set(m.inhibition_function.keys()) == {100, 200}
    assert m.inhibition_function[100] == 0.0   # 1 success, 0 failures at SSD 100
    assert m.inhibition_function[200] == 1.0   # 0 successes, 1 failure at SSD 200

    # SSRT: mean_go_rt - mean_ssd_s
    # mean_go_rt = 0.065 s; mean_ssd = (100 + 200) / 2 = 150 ms = 0.150 s
    # (late-stop excluded from SSD mean)
    assert m.ssrt_estimate_s is not None
    assert math.isclose(m.ssrt_estimate_s, 0.065 - 0.150, abs_tol=1e-9)
    assert m.mean_ssd_ms is not None
    assert math.isclose(m.mean_ssd_ms, 150.0, abs_tol=1e-9)

    # Cancellation latency: only s_early_success qualifies
    # stop_signal at 0.300 + 0.100 = 0.400 s, decision_commit at 0.300 + 0.500 = 0.800 s
    assert m.cancellation_latency_mean_s is not None
    assert math.isclose(m.cancellation_latency_mean_s, 0.400, abs_tol=1e-9)

    # Trigger failure rate: 1 late-stop failure out of 2 total stop failures
    assert m.trigger_failure_rate is not None
    assert math.isclose(m.trigger_failure_rate, 1 / 2, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Test: _has_stop_signal_event helper
# ---------------------------------------------------------------------------

def test_has_stop_signal_event_true() -> None:
    trial = make_stop_success_early(ssd_ms=150)
    assert _has_stop_signal_event(trial) is True


def test_has_stop_signal_event_false() -> None:
    trial = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    assert _has_stop_signal_event(trial) is False


# ---------------------------------------------------------------------------
# Task 7.3 Tests: validate_stop_signal_data and StopSignalValidityReport
# ---------------------------------------------------------------------------

_INDEPENDENCE_NOTE = (
    "Race model independence assumed. In Phase 7, go and stop processes share"
    " the CortexEvidenceGenerator; true statistical independence is not guaranteed"
    " but is not empirically testable from behavioural data alone."
)


# Test V1: raises ValueError on empty list
def test_validate_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        validate_stop_signal_data([])


# Test V2: rt_check_passed=True when failed_stop_rt < go_rt
def test_rt_check_passed_when_failed_stop_rt_lt_go_rt() -> None:
    # go RT = 0.080 s (two trials: 0.060 + 0.100 → mean = 0.080)
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.060)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.100)
    # failed stop RT = 0.030 s (shorter than go RT)
    fail = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.030)

    report = validate_stop_signal_data([go1, go2, fail])

    assert isinstance(report, StopSignalValidityReport)
    assert report.rt_check_passed is True
    assert report.rt_check_note == "Failed-stop RT < go RT: race model prediction satisfied."


# Test V3: rt_check_passed=False with deterministic note when failed_stop_rt == go_rt
def test_rt_check_failed_equal_rts_deterministic_note() -> None:
    # go RT = 0.050 s (two trials with same onset)
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    # failed stop RT = exactly 0.050 s (same as go RT)
    fail = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.050)

    report = validate_stop_signal_data([go1, go2, fail])

    assert report.rt_check_passed is False
    assert "deterministic BG model" in report.rt_check_note
    assert "Phase 9+" in report.rt_check_note


# Test V4: rt_check_passed=False with "unexpected" note when failed_stop_rt > go_rt
def test_rt_check_failed_unexpected_when_failed_stop_rt_gt_go_rt() -> None:
    # go RT = 0.030 s (mean of two short trials)
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.020)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.040)
    # failed stop RT = 0.100 s (greater than go RT = 0.030)
    fail = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.100)

    report = validate_stop_signal_data([go1, go2, fail])

    assert report.rt_check_passed is False
    assert "unexpected" in report.rt_check_note.lower()


# Test V5: rt_check_note "No stop failures..." when no stop-failure trials
def test_rt_check_note_no_stop_failures() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    success = make_stop_success_early(ssd_ms=150)

    report = validate_stop_signal_data([go1, go2, success])

    assert report.failed_stop_rt_mean_s is None
    assert "No stop failures" in report.rt_check_note


# Test V6: rt_check_note "No responding go trials..." when no go responses
def test_rt_check_note_no_responding_go_trials() -> None:
    # Only stop failures — no go responding trials
    fail = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.050)
    miss = make_go_miss()

    report = validate_stop_signal_data([fail, miss])

    assert report.go_rt_mean_s is None
    assert "No responding go trials" in report.rt_check_note


# Test V7: n_late_stop_trials counts stop trials without stop_signal event
def test_n_late_stop_trials_correct() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    early_fail = make_stop_failure_early(ssd_ms=150, movement_onset_time=_CUE_ONSET + 0.040)
    late1 = make_stop_failure_late()
    late2 = make_stop_failure_late()

    report = validate_stop_signal_data([go1, go2, early_fail, late1, late2])

    assert report.n_late_stop_trials == 2


# Test V8: n_excluded_for_ssrt equals n_late_stop_trials
def test_n_excluded_for_ssrt_equals_n_late_stop() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    late = make_stop_failure_late()
    early_success = make_stop_success_early(ssd_ms=100)

    report = validate_stop_signal_data([go1, go2, late, early_success])

    assert report.n_excluded_for_ssrt == report.n_late_stop_trials


# Test V9: independence_assumption_note is the exact fixed string
def test_independence_assumption_note_is_fixed_string() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)

    report = validate_stop_signal_data([go1, go2])

    assert report.independence_assumption_note == _INDEPENDENCE_NOTE


# Test V10: empirical_stop_proportion = n_stop_trials / n_total_trials
def test_empirical_stop_proportion_correct() -> None:
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    go3 = make_go_success(movement_onset_time=_CUE_ONSET + 0.060)
    stop1 = make_stop_success_early(ssd_ms=150)

    report = validate_stop_signal_data([go1, go2, go3, stop1])

    assert math.isclose(report.empirical_stop_proportion, 1 / 4, abs_tol=1e-9)
    assert report.n_total_trials == 4
    assert report.n_stop_trials == 1
    assert report.n_go_trials == 3


# Test V11: inhibition_function_monotone=True when failure_rate rises with SSD
def test_inhibition_function_monotone_true() -> None:
    # SSD 100: 1 success, 0 failures → rate 0.0
    # SSD 200: 1 failure, 0 successes → rate 1.0
    # 0.0 → 1.0 is non-decreasing → monotone=True
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    s_success = make_stop_success_early(ssd_ms=100)
    s_fail = make_stop_failure_early(ssd_ms=200, movement_onset_time=_CUE_ONSET + 0.040)

    report = validate_stop_signal_data([go1, go2, s_success, s_fail])

    assert report.inhibition_function_monotone is True


# Test V12: inhibition_function_monotone=False when failure_rate decreases at any step
def test_inhibition_function_monotone_false() -> None:
    # SSD 100: all fail → rate 1.0
    # SSD 200: all succeed → rate 0.0
    # 1.0 → 0.0 is decreasing → monotone=False
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    s_fail_100 = make_stop_failure_early(ssd_ms=100, movement_onset_time=_CUE_ONSET + 0.040)
    s_success_200 = make_stop_success_early(ssd_ms=200)

    report = validate_stop_signal_data([go1, go2, s_fail_100, s_success_200])

    assert report.inhibition_function_monotone is False


# Test V13: inhibition_function_monotone=None when fewer than 2 SSD levels
def test_inhibition_function_monotone_none_when_single_ssd() -> None:
    # Only one SSD level — cannot assess monotonicity
    go1 = make_go_success(movement_onset_time=_CUE_ONSET + 0.050)
    go2 = make_go_success(movement_onset_time=_CUE_ONSET + 0.080)
    s_success = make_stop_success_early(ssd_ms=150)

    report = validate_stop_signal_data([go1, go2, s_success])

    assert report.inhibition_function_monotone is None
