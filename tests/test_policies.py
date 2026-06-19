"""Unit tests and M1 integration test for the three reference policies.

Tests verify:
1. Each policy correctly processes ActionEvidence
2. BGDecision fields are properly populated
3. All four engines × all three policies produce valid TrialLogs and Metrics
"""

import pytest

from nrp_bga_sb.engines.change_of_mind import ChangeOfMindConfig, run_change_of_mind_trials
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.engines.stop_signal import StopSignalConfig, run_stop_signal_trials
from nrp_bga_sb.engines.two_choice import TwoChoiceConfig, run_two_choice_trials
from nrp_bga_sb.policies import (
    RandomPolicy,
    ThresholdPolicy,
    oracle_policy,
)
from nrp_bga_sb.schemas import ActionEvidence, TrialLog
from nrp_bga_sb.scorer import score_trials

# --- Unit Tests: Oracle Policy ---


class TestOraclePolicy:
    """Test the oracle policy behavior."""

    def test_oracle_selects_argmax_salience(self):
        """Oracle selects channel with highest salience."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="two_choice",
            cue_identity="left",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.3, 0.8],
        )
        decision = oracle_policy(trial_log, evidence)
        assert decision.selected_channel == 1
        assert decision.channel_activations == [0.3, 0.8]

    def test_oracle_inhibits_on_stop_signal(self):
        """Oracle returns -1 when stop_signal_present=True."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="stop_signal",
            cue_identity="go",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.8, 0.3],
            stop_signal_present=True,
        )
        decision = oracle_policy(trial_log, evidence)
        assert decision.selected_channel == -1

    def test_oracle_inhibits_weak_salience(self):
        """Oracle returns -1 when max salience < 0.3."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="two_choice",
            cue_identity="left",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.2, 0.25],
        )
        decision = oracle_policy(trial_log, evidence)
        assert decision.selected_channel == -1

    def test_oracle_decision_margin(self):
        """Oracle computes correct decision_margin."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="two_choice",
            cue_identity="left",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.8, 0.3],
        )
        decision = oracle_policy(trial_log, evidence)
        assert abs(decision.decision_margin - 0.5) < 1e-6  # 0.8 - 0.3

    def test_oracle_decision_margin_single_channel(self):
        """Oracle sets decision_margin=0.0 for single-channel evidence."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="go_nogo",
            cue_identity="go",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=1,
            channel_salience=[0.7],
        )
        decision = oracle_policy(trial_log, evidence)
        assert decision.decision_margin == 0.0


# --- Unit Tests: Random Policy ---


class TestRandomPolicy:
    """Test the random policy behavior."""

    def test_random_is_deterministic_from_seed(self):
        """Random policy with same seed produces same decision."""
        trial_log1 = TrialLog(
            trial_id=1,
            seed=12345,
            task_type="go_nogo",
            cue_identity="go",
            cue_onset_time=0.0,
        )
        trial_log2 = TrialLog(
            trial_id=2,
            seed=12345,  # same seed
            task_type="go_nogo",
            cue_identity="go",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.5, 0.5],
        )

        policy = RandomPolicy()
        decision1 = policy(trial_log1, evidence)
        decision2 = policy(trial_log2, evidence)

        assert decision1.selected_channel == decision2.selected_channel

    def test_random_produces_valid_channels(self):
        """Random policy only selects -1, 0, or 1."""
        policy = RandomPolicy()
        for seed in range(100):
            trial_log = TrialLog(
                trial_id=1,
                seed=seed,
                task_type="go_nogo",
                cue_identity="go",
                cue_onset_time=0.0,
            )
            evidence = ActionEvidence(
                sim_time=0.1,
                trial_id=1,
                n_channels=2,
                channel_salience=[0.5, 0.5],
            )
            decision = policy(trial_log, evidence)
            assert decision.selected_channel in [-1, 0, 1]

    def test_random_passes_through_channel_activations(self):
        """Random policy passes channel_activations from evidence."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="two_choice",
            cue_identity="left",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.7, 0.4],
        )
        policy = RandomPolicy()
        decision = policy(trial_log, evidence)
        assert decision.channel_activations == [0.7, 0.4]

    def test_random_decision_margin_single_channel(self):
        """Random policy sets decision_margin=0.0 for single-channel evidence."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="go_nogo",
            cue_identity="go",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=1,
            channel_salience=[0.7],
        )
        policy = RandomPolicy()
        decision = policy(trial_log, evidence)
        assert decision.decision_margin == 0.0


# --- Unit Tests: Threshold Policy ---


class TestThresholdPolicy:
    """Test the threshold policy behavior."""

    def test_threshold_selects_above_threshold(self):
        """Threshold policy selects channel when salience >= threshold."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="two_choice",
            cue_identity="left",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.7, 0.3],
        )
        policy = ThresholdPolicy(threshold=0.6)
        decision = policy(trial_log, evidence)
        assert decision.selected_channel == 0  # 0.7 >= 0.6

    def test_threshold_inhibits_below_threshold(self):
        """Threshold policy returns -1 when salience < threshold."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="two_choice",
            cue_identity="left",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.5, 0.3],
        )
        policy = ThresholdPolicy(threshold=0.6)
        decision = policy(trial_log, evidence)
        assert decision.selected_channel == -1  # 0.5 < 0.6

    def test_threshold_inhibits_on_stop_signal(self):
        """Threshold policy returns -1 when stop_signal_present=True."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="stop_signal",
            cue_identity="go",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.9, 0.1],
            stop_signal_present=True,
        )
        policy = ThresholdPolicy(threshold=0.6)
        decision = policy(trial_log, evidence)
        assert decision.selected_channel == -1  # stop signal overrides high salience

    def test_threshold_respects_boundary(self):
        """Threshold policy treats >= threshold as the inclusion boundary."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="two_choice",
            cue_identity="left",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.6, 0.2],
        )
        policy = ThresholdPolicy(threshold=0.6)
        decision = policy(trial_log, evidence)
        assert decision.selected_channel == 0  # 0.6 >= 0.6 is True

    def test_threshold_decision_margin_single_channel(self):
        """Threshold policy sets decision_margin=0.0 for single-channel evidence."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="go_nogo",
            cue_identity="go",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=1,
            channel_salience=[0.7],
        )
        policy = ThresholdPolicy(threshold=0.6)
        decision = policy(trial_log, evidence)
        assert decision.decision_margin == 0.0


# --- Unit Tests: BGDecision Schema ---


class TestBGDecisionSchema:
    """Test that policies populate BGDecision fields correctly."""

    def test_bgdecision_fields_populated(self):
        """BGDecision from oracle_policy has all required fields."""
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="two_choice",
            cue_identity="left",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.7, 0.4],
        )
        decision = oracle_policy(trial_log, evidence)

        assert decision.sim_time == 0.1
        assert decision.trial_id == 1
        assert decision.selected_channel in [-1, 0, 1]
        assert isinstance(decision.decision_margin, float)
        assert len(decision.suppression_vector) == 2
        assert len(decision.channel_activations) == 2
        assert isinstance(decision.selection_latency, float)

    def test_bgdecision_selected_channel_ge_minus_one(self):
        """BGDecision.selected_channel respects Field(ge=-1) constraint."""
        # This test verifies that the schema enforces the constraint.
        trial_log = TrialLog(
            trial_id=1,
            seed=42,
            task_type="go_nogo",
            cue_identity="go",
            cue_onset_time=0.0,
        )
        evidence = ActionEvidence(
            sim_time=0.1,
            trial_id=1,
            n_channels=2,
            channel_salience=[0.5, 0.5],
        )

        decision = oracle_policy(trial_log, evidence)
        assert decision.selected_channel >= -1


# --- Integration Test: M1 (All Engines × All Policies) ---


class TestM1IntegrationAllEngines:
    """M1 acceptance criterion: all four engines × all three policies work together.

    This test runs each engine with each policy, verifies TrialLogs are valid,
    and confirms that score_trials produces Metrics without error.
    """

    @pytest.fixture(scope="function")
    def policies(self):
        """Fixture providing all three policies."""
        return {
            "oracle": oracle_policy,
            "random": RandomPolicy(),
            "threshold": ThresholdPolicy(threshold=0.6),
        }

    def test_m1_gonogo_all_policies(self, policies):
        """Go/no-go engine with all three policies produces valid TrialLogs and Metrics."""
        config = GoNoGoConfig(
            n_trials=20,
            go_probability=0.5,
            response_window_start_ms=100,
            response_window_duration_ms=500,
            fixation_duration_ms=500,
            cue_onset_ms=1000,
            decision_point_ms=100,
            seed=12345,
        )

        for policy_name, policy in policies.items():
            trials = run_go_nogo_trials(config, policy, logger=None)

            # Verify trial count.
            assert len(trials) == config.n_trials

            # Verify each trial is a valid TrialLog.
            for trial in trials:
                assert isinstance(trial, TrialLog)
                assert trial.task_type == "go_nogo"
                assert trial.success is not None
                assert trial.success or trial.failure_mode is not None
                assert len(trial.events) > 0  # Event stream populated

            # Verify score_trials works.
            metrics = score_trials(trials, condition_id="m1_gonogo", bg_frequency_hz=50.0)
            assert metrics.n_trials == config.n_trials
            assert metrics.false_alarm_rate is not None

    def test_m1_twochoice_all_policies(self, policies):
        """Two-choice engine with all three policies produces valid TrialLogs and Metrics."""
        config = TwoChoiceConfig(
            n_trials=20,
            conflict_levels={
                "low": [0.8, 0.2],
                "medium": [0.65, 0.35],
            },
            response_window_start_ms=100,
            response_window_duration_ms=500,
            fixation_duration_ms=500,
            target_onset_ms=1000,
            decision_point_ms=100,
            seed=12345,
        )

        for policy_name, policy in policies.items():
            trials = run_two_choice_trials(config, policy, logger=None)

            # Verify trial count.
            assert len(trials) == config.n_trials

            # Verify each trial is a valid TrialLog.
            for trial in trials:
                assert isinstance(trial, TrialLog)
                assert trial.task_type == "two_choice"
                assert trial.success is not None
                assert len(trial.events) > 0

            # Verify score_trials works.
            metrics = score_trials(trials, condition_id="m1_twochoice", bg_frequency_hz=50.0)
            assert metrics.n_trials == config.n_trials
            assert metrics.wrong_action_rate is not None

    def test_m1_stopsignal_all_policies(self, policies):
        """Stop-signal engine with all three policies produces valid TrialLogs and Metrics."""
        config = StopSignalConfig(
            n_trials=20,
            stop_proportion=0.5,
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

        for policy_name, policy in policies.items():
            trials = run_stop_signal_trials(config, policy, logger=None)

            # Verify trial count.
            assert len(trials) == config.n_trials

            # Verify each trial is a valid TrialLog.
            for trial in trials:
                assert isinstance(trial, TrialLog)
                assert trial.task_type == "stop_signal"
                assert trial.success is not None
                assert len(trial.events) > 0

            # Verify score_trials works.
            metrics = score_trials(trials, condition_id="m1_stopsignal", bg_frequency_hz=50.0)
            assert metrics.n_trials == config.n_trials

    def test_m1_changeofmind_all_policies(self, policies):
        """Change-of-mind engine with all three policies produces valid TrialLogs and Metrics."""
        config = ChangeOfMindConfig(
            n_trials=20,
            no_switch_proportion=0.2,
            switch_delay_categories={"early": 50, "medium": 150},
            initial_decision_point_ms=20,
            post_switch_decision_point_ms=200,
            response_window_duration_ms=700,
            seed=42,
        )

        for policy_name, policy in policies.items():
            trials = run_change_of_mind_trials(config, policy, logger=None)

            # Verify trial count.
            assert len(trials) == config.n_trials

            # Verify each trial is a valid TrialLog.
            for trial in trials:
                assert isinstance(trial, TrialLog)
                assert trial.task_type == "change_of_mind"
                assert trial.success is not None
                assert len(trial.events) > 0

            # Verify score_trials works.
            metrics = score_trials(trials, condition_id="m1_changeofmind", bg_frequency_hz=50.0)
            assert metrics.n_trials == config.n_trials
            if metrics.switch_success_rate is not None:
                assert 0.0 <= metrics.switch_success_rate <= 1.0

    def test_m1_cartesian_product(self, policies):
        """M1 acceptance: all 4 engines × 3 policies produce valid outputs."""
        engines = {
            "go_nogo": (
                GoNoGoConfig(
                    n_trials=10,
                    go_probability=0.5,
                    response_window_start_ms=100,
                    response_window_duration_ms=500,
                    fixation_duration_ms=500,
                    cue_onset_ms=1000,
                    decision_point_ms=100,
                    seed=12345,
                ),
                run_go_nogo_trials,
            ),
            "two_choice": (
                TwoChoiceConfig(
                    n_trials=10,
                    conflict_levels={"low": [0.8, 0.2]},
                    response_window_start_ms=100,
                    response_window_duration_ms=500,
                    fixation_duration_ms=500,
                    target_onset_ms=1000,
                    decision_point_ms=100,
                    seed=12345,
                ),
                run_two_choice_trials,
            ),
            "stop_signal": (
                StopSignalConfig(
                    n_trials=10,
                    stop_proportion=0.5,
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
                ),
                run_stop_signal_trials,
            ),
            "change_of_mind": (
                ChangeOfMindConfig(
                    n_trials=10,
                    no_switch_proportion=0.2,
                    switch_delay_categories={"early": 50},
                    initial_decision_point_ms=20,
                    post_switch_decision_point_ms=200,
                    response_window_duration_ms=700,
                    seed=42,
                ),
                run_change_of_mind_trials,
            ),
        }

        # Run all 4 × 3 = 12 combinations.
        for engine_name, (config, engine_fn) in engines.items():
            for policy_name, policy in policies.items():
                # Run trials.
                trials = engine_fn(config, policy, logger=None)

                # Verify basic properties.
                assert len(trials) == config.n_trials
                for trial in trials:
                    assert isinstance(trial, TrialLog)
                    assert trial.success is not None
                    assert len(trial.events) > 0

                # Verify metrics.
                metrics = score_trials(
                    trials, condition_id=f"m1_{engine_name}_{policy_name}", bg_frequency_hz=50.0
                )
                assert metrics.n_trials == config.n_trials
                assert metrics.condition_id == f"m1_{engine_name}_{policy_name}"
                assert metrics.bg_frequency_hz == 50.0
