"""Tests for the closed-loop policy and Task 4.3 acceptance verification.

The key frequency-propagation test (M4 prep):
  At 5 Hz input_sampling_hz with accumulation_ms=200 ms and rise_time_ms=100 ms,
  Gate 1 fires only at tick=0 (elapsed=0 ms → neutral evidence [0.5, 0.5]).
  BGModel cannot select → committed_decision.selected_channel = -1 → go trials miss.

  At 40 Hz (period=25 ticks), Gate 1 fires at ticks 0, 25, 50, 75, 100, ...
  Tick 50 → elapsed=50 ms → salience [0.7, 0.3] (gap=0.4 > medium-conflict threshold).
  BGModel selects channel 0 → committed_decision.selected_channel = 0 → go trials succeed.
"""

from __future__ import annotations

from nrp_bga_sb.closed_loop import ClosedLoopPolicy, make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.change_of_mind import ChangeOfMindConfig, run_change_of_mind_trials
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.engines.stop_signal import StopSignalConfig, run_stop_signal_trials
from nrp_bga_sb.engines.two_choice import TwoChoiceConfig, run_two_choice_trials
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import TrialLog
from nrp_bga_sb.thalamus import ThalamusConfig

# --- Helpers ---


def _go_nogo_config(seed: int = 1, n_trials: int = 20) -> GoNoGoConfig:
    return GoNoGoConfig(
        n_trials=n_trials,
        go_probability=1.0,  # all-go for frequency effect tests
        response_window_start_ms=0,
        response_window_duration_ms=500,
        fixation_duration_ms=200,
        cue_onset_ms=300,
        decision_point_ms=200,
        seed=seed,
    )


def _two_choice_config(seed: int = 1, n_trials: int = 12) -> TwoChoiceConfig:
    return TwoChoiceConfig(
        n_trials=n_trials,
        conflict_levels={"low": [0.8, 0.2], "medium": [0.65, 0.35]},
        response_window_start_ms=0,
        response_window_duration_ms=500,
        fixation_duration_ms=200,
        target_onset_ms=300,
        decision_point_ms=200,
        seed=seed,
    )


def _stop_signal_config(seed: int = 1, n_trials: int = 20) -> StopSignalConfig:
    return StopSignalConfig(
        n_trials=n_trials,
        stop_proportion=0.3,
        initial_ssd_ms=100,
        ssd_step_ms=50,
        response_window_duration_ms=500,
        fixation_duration_ms=200,
        go_cue_onset_ms=300,
        decision_point_ms=200,
        seed=seed,
    )


def _change_of_mind_config(seed: int = 1, n_trials: int = 20) -> ChangeOfMindConfig:
    return ChangeOfMindConfig(
        n_trials=n_trials,
        no_switch_proportion=0.25,
        switch_delay_categories={"early": 50, "late": 150},
        initial_decision_point_ms=30,
        post_switch_decision_point_ms=200,
        response_window_duration_ms=600,
        go_cue_onset_ms=300,
        fixation_duration_ms=200,
        seed=seed,
    )


def _valid_trial_logs(trials: list[TrialLog]) -> bool:
    """Check that every trial has non-decreasing sim_time events and an outcome."""
    for t in trials:
        times = [e.sim_time for e in t.events]
        if times != sorted(times):
            return False
        if t.success is None:
            return False
    return True


# --- make_closed_loop_policy factory ---


def test_factory_returns_closed_loop_policy() -> None:
    policy = make_closed_loop_policy()
    assert isinstance(policy, ClosedLoopPolicy)


def test_factory_with_explicit_configs() -> None:
    policy = make_closed_loop_policy(
        cortex_config=CortexConfig(rise_time_ms=80.0),
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        thalamus_config=ThalamusConfig(margin_threshold=0.02),
    )
    assert isinstance(policy, ClosedLoopPolicy)


# --- Task 4.3 M4 Acceptance: frequency propagation ---
# At 5 Hz (period=200 ticks), Gate 1 fires only at tick=0 (evidence neutral).
# BGModel cannot commit a positive selection → all go trials are misses.
# At 40 Hz (period=25 ticks), Gate 1 fires at tick=50 (evidence risen to [0.7, 0.3]).
# BGModel selects channel 0 → go trials succeed.


def test_5hz_go_trials_all_miss() -> None:
    """At 5 Hz with accumulation_ms=200ms and rise_time_ms=100ms, BG never sees
    sufficient salience; committed_decision has selected_channel=-1 on every trial."""
    policy_5hz = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(5.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    cfg = _go_nogo_config(n_trials=10)
    trials = run_go_nogo_trials(cfg, policy_5hz)
    # All are go trials (go_probability=1.0); at 5 Hz no selection → miss
    miss_count = sum(1 for t in trials if t.failure_mode == "miss")
    assert miss_count == len(trials)


def test_40hz_go_trials_all_succeed() -> None:
    """At 40 Hz, Gate 1 fires at tick=50 (elapsed=50ms → frac=0.5 → evidence [0.7, 0.3]).
    The salience gap 0.4 exceeds the BGModel medium-conflict threshold; BG commits
    channel 0; go trials succeed."""
    policy_40hz = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    cfg = _go_nogo_config(n_trials=10)
    trials = run_go_nogo_trials(cfg, policy_40hz)
    success_count = sum(1 for t in trials if t.success is True)
    assert success_count == len(trials)


def test_frequency_effect_on_success_rate() -> None:
    """Success rate increases monotonically from 5 Hz to 40 Hz."""
    configs = {
        5: FrequencyConfig.from_effective_hz(5.0),
        10: FrequencyConfig.from_effective_hz(10.0),
        40: FrequencyConfig.from_effective_hz(40.0),
    }
    cortex = CortexConfig(rise_time_ms=100.0)
    rates = {}
    for hz, fr_cfg in configs.items():
        policy = make_closed_loop_policy(
            frequency_config=fr_cfg,
            cortex_config=cortex,
            accumulation_ms=200.0,
        )
        trials = run_go_nogo_trials(_go_nogo_config(n_trials=10), policy)
        rates[hz] = sum(1 for t in trials if t.success) / len(trials)

    assert rates[5] < rates[10] <= rates[40]


# --- Thalamic fields populated ---


def test_motor_command_series_populated_on_success() -> None:
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    trials = run_go_nogo_trials(_go_nogo_config(n_trials=5), policy)
    successful = [t for t in trials if t.success]
    assert len(successful) > 0
    for t in successful:
        assert len(t.motor_command_series) == 1
        cmd = t.motor_command_series[0]
        assert cmd.gate_state in ("open", "partial")
        assert cmd.gate_gain > 0.0


def test_motor_command_series_empty_on_miss() -> None:
    # At 5 Hz all go trials miss; motor_command_series should still be non-empty
    # because ClosedLoopPolicy always appends a MotorCommand (gate may be closed).
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(5.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    trials = run_go_nogo_trials(_go_nogo_config(n_trials=5), policy)
    for t in trials:
        # ClosedLoopPolicy appends one MotorCommand per call, even if gate=closed
        assert len(t.motor_command_series) == 1
        assert t.motor_command_series[0].gate_state == "closed"


def test_thalamic_relay_and_release_times_set() -> None:
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    trials = run_go_nogo_trials(_go_nogo_config(n_trials=5), policy)
    for t in trials:
        assert t.thalamic_relay_time is not None
        assert t.thalamic_release_time is not None
        assert t.thalamic_relay_time >= 0.0
        assert t.thalamic_release_time >= 0.0


# --- Trial log validity across all four engines ---


def test_go_nogo_trials_valid_with_closed_loop() -> None:
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    cfg = GoNoGoConfig(
        n_trials=20,
        go_probability=0.6,
        response_window_start_ms=0,
        response_window_duration_ms=500,
        fixation_duration_ms=200,
        cue_onset_ms=300,
        decision_point_ms=200,
        seed=42,
    )
    trials = run_go_nogo_trials(cfg, policy)
    assert _valid_trial_logs(trials)


def test_two_choice_trials_valid_with_closed_loop() -> None:
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    trials = run_two_choice_trials(_two_choice_config(), policy)
    assert _valid_trial_logs(trials)


def test_stop_signal_trials_valid_with_closed_loop() -> None:
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    trials = run_stop_signal_trials(_stop_signal_config(), policy)
    assert _valid_trial_logs(trials)


def test_change_of_mind_trials_valid_with_closed_loop() -> None:
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    trials = run_change_of_mind_trials(_change_of_mind_config(), policy)
    assert _valid_trial_logs(trials)


# --- No_go withholding works correctly ---


def test_no_go_withhold_correct_with_closed_loop() -> None:
    """No-go cortical evidence stays neutral → BG cannot select → correct_withhold."""
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    cfg = GoNoGoConfig(
        n_trials=10,
        go_probability=0.0,  # all no-go trials
        response_window_start_ms=0,
        response_window_duration_ms=500,
        fixation_duration_ms=200,
        cue_onset_ms=300,
        decision_point_ms=200,
        seed=7,
    )
    trials = run_go_nogo_trials(cfg, policy)
    withhold_count = sum(1 for t in trials if t.success is True)
    assert withhold_count == len(trials)


# --- Intermediate state observable for failure diagnosis ---


def test_bg_decision_observable_via_motor_command() -> None:
    """BG decision and motor command together expose intermediate state."""
    policy = make_closed_loop_policy(
        frequency_config=FrequencyConfig.from_effective_hz(40.0),
        cortex_config=CortexConfig(rise_time_ms=100.0),
        accumulation_ms=200.0,
    )
    trials = run_go_nogo_trials(_go_nogo_config(n_trials=5), policy)
    for t in trials:
        # Motor command carries gate_state, gate_gain, and selected channel info
        assert len(t.motor_command_series) == 1
        cmd = t.motor_command_series[0]
        assert cmd.gate_state in ("open", "partial", "closed")
        assert len(cmd.command) == 2


# --- Backward compatibility: scheduler still works without cortex_generator ---


def test_scheduler_without_cortex_generator_unchanged() -> None:
    """ScheduledBGAdapter with no cortex_generator still works as in Phase 3."""
    from nrp_bga_sb.bg_model import BGAdapter, BGModelConfig
    from nrp_bga_sb.scheduler import FrequencyConfig, ScheduledBGAdapter
    from nrp_bga_sb.schemas import ActionEvidence, TrialLog

    bg = BGAdapter(BGModelConfig())
    sched = ScheduledBGAdapter(
        base_policy=bg,
        config=FrequencyConfig.from_effective_hz(40.0),
        accumulation_ms=200.0,
        # cortex_generator=None (default)
    )
    trial = TrialLog(
        trial_id=1, seed=1, task_type="two_choice",
        cue_identity="left", cue_onset_time=0.3,
    )
    evidence = ActionEvidence(
        sim_time=0.5, trial_id=1, n_channels=2,
        channel_salience=[0.8, 0.2],
    )
    decision = sched(trial, evidence)
    # Static [0.8, 0.2] evidence → BG selects channel 0 (as in Phase 3)
    assert decision.selected_channel == 0
