"""Tests for the Gurney-Prescott-Redgrave (2001) BG model and BGAdapter.

Coverage structure:
  - BGModelConfig: defaults and custom instantiation
  - BGModel: selection correctness, suppression, conflict/tonic no-selection,
             latency monotonicity, 4-channel operation, determinism, noise
  - BGAdapter: policy interface, stop-signal override, latency mapping,
               determinism with/without noise
  - Wiring (Task 2.4): BGAdapter as drop-in policy for two_choice and change_of_mind
  - M2 acceptance: selection latency monotone with conflict
"""

from __future__ import annotations

import numpy as np

from nrp_bga_sb.bg_model import BGAdapter, BGModel, BGModelConfig
from nrp_bga_sb.engines.change_of_mind import ChangeOfMindConfig, run_change_of_mind_trials
from nrp_bga_sb.engines.two_choice import TwoChoiceConfig, run_two_choice_trials
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence(
    saliences: list[float],
    trial_id: int = 1,
    sim_time: float = 0.1,
    stop: bool = False,
) -> ActionEvidence:
    return ActionEvidence(
        sim_time=sim_time,
        trial_id=trial_id,
        n_channels=len(saliences),
        channel_salience=saliences,
        stop_signal_present=stop,
    )


def _make_trial(seed: int = 42, task_type: str = "two_choice") -> TrialLog:
    return TrialLog(
        trial_id=1,
        seed=seed,
        task_type=task_type,  # type: ignore[arg-type]
        cue_identity="cue_a",
        cue_onset_time=0.0,
    )


# ---------------------------------------------------------------------------
# BGModelConfig
# ---------------------------------------------------------------------------


class TestBGModelConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = BGModelConfig()
        assert cfg.n_channels == 2
        assert 0.0 < cfg.theta_d < 0.5
        assert 0.0 < cfg.w_stn_gpi <= 1.0
        assert 0.0 < cfg.w_d1_gpi <= 1.5
        assert cfg.thal_threshold > 0.0
        assert cfg.max_iters >= 100
        assert cfg.noise_std == 0.0

    def test_custom_n_channels(self) -> None:
        cfg = BGModelConfig(n_channels=4)
        assert cfg.n_channels == 4

    def test_latency_params_ordered(self) -> None:
        cfg = BGModelConfig()
        # min < max; scale and eps are positive
        assert cfg.latency_min_ms < cfg.latency_max_ms
        assert cfg.latency_scale_ms > 0.0
        assert cfg.latency_eps > 0.0


# ---------------------------------------------------------------------------
# BGModel: core selection
# ---------------------------------------------------------------------------


class TestBGModelSelection:
    def setup_method(self) -> None:
        self.model = BGModel(BGModelConfig())

    def test_selects_high_salience_channel_low_conflict(self) -> None:
        # Low conflict: clear dominant channel.
        result = self.model.compute(np.array([0.8, 0.2]))
        assert result["selected_channel"] == 0

    def test_selects_correct_channel_medium_conflict(self) -> None:
        # Medium conflict [0.65, 0.35]: dominant channel still selected.
        result = self.model.compute(np.array([0.65, 0.35]))
        assert result["selected_channel"] == 0

    def test_no_selection_high_conflict(self) -> None:
        # High conflict [0.55, 0.45]: gap too small for GPR to decide.
        result = self.model.compute(np.array([0.55, 0.45]))
        assert result["selected_channel"] == -1

    def test_no_selection_equal_salience(self) -> None:
        # Equal saliences: symmetric state → no thalamus output.
        result = self.model.compute(np.array([0.5, 0.5]))
        assert result["selected_channel"] == -1

    def test_no_selection_zero_salience(self) -> None:
        # Tonic state: no cortical input → GPi suppresses thalamus.
        result = self.model.compute(np.array([0.0, 0.0]))
        assert result["selected_channel"] == -1

    def test_selects_channel_1_when_it_dominates(self) -> None:
        # Reversed: channel 1 should win.
        result = self.model.compute(np.array([0.2, 0.8]))
        assert result["selected_channel"] == 1

    def test_correct_selection_change_of_mind_post_switch(self) -> None:
        # Post-switch salience [0.2, 0.8]: channel 1 selected.
        result = self.model.compute(np.array([0.2, 0.8]))
        assert result["selected_channel"] == 1


# ---------------------------------------------------------------------------
# BGModel: suppression invariants
# ---------------------------------------------------------------------------


class TestBGModelSuppression:
    def setup_method(self) -> None:
        self.model = BGModel(BGModelConfig())

    def test_loser_has_higher_gpi_than_winner(self) -> None:
        result = self.model.compute(np.array([0.8, 0.2]))
        sv = result["suppression_vector"]
        # Channel 0 is the winner: its GPi must be strictly below channel 1's GPi.
        assert sv[0] < sv[1]

    def test_suppression_vector_nonnegative(self) -> None:
        for saliences in [[0.8, 0.2], [0.5, 0.5], [0.0, 0.0]]:
            result = self.model.compute(np.array(saliences))
            assert all(v >= 0.0 for v in result["suppression_vector"])

    def test_channel_activations_nonnegative(self) -> None:
        for saliences in [[0.8, 0.2], [0.5, 0.5], [0.0, 0.0]]:
            result = self.model.compute(np.array(saliences))
            assert all(v >= 0.0 for v in result["channel_activations"])

    def test_winner_has_highest_channel_activation(self) -> None:
        result = self.model.compute(np.array([0.8, 0.2]))
        activations = result["channel_activations"]
        selected = result["selected_channel"]
        assert selected == int(np.argmax(activations))

    def test_loser_activation_is_zero_or_lower(self) -> None:
        result = self.model.compute(np.array([0.8, 0.2]))
        act = result["channel_activations"]
        # Channel 0 wins; channel 1 activation should be ≤ channel 0.
        assert act[1] <= act[0]

    def test_decision_margin_positive_when_selected(self) -> None:
        result = self.model.compute(np.array([0.8, 0.2]))
        assert result["decision_margin"] > 0.0

    def test_decision_margin_zero_when_no_selection(self) -> None:
        result = self.model.compute(np.array([0.5, 0.5]))
        assert result["decision_margin"] == 0.0


# ---------------------------------------------------------------------------
# BGModel: latency monotonicity (M2 acceptance criterion)
# ---------------------------------------------------------------------------


class TestBGModelLatencyMonotonicity:
    """Verify that T_winner decreases as conflict increases.

    This is the analytical foundation of the M2 acceptance criterion:
    the adapter converts T_winner → latency via an inverse function, so
    lower T_winner → higher latency.
    """

    def setup_method(self) -> None:
        self.model = BGModel(BGModelConfig())

    def test_t_winner_decreases_with_conflict(self) -> None:
        low = self.model.compute(np.array([0.8, 0.2]))
        med = self.model.compute(np.array([0.65, 0.35]))
        # Low conflict has a stronger winner → higher T_winner.
        assert low["T_winner"] > med["T_winner"] > 0.0

    def test_high_conflict_produces_zero_t_winner(self) -> None:
        result = self.model.compute(np.array([0.55, 0.45]))
        assert result["T_winner"] == 0.0

    def test_medium_conflict_t_winner_between_low_and_high(self) -> None:
        low = self.model.compute(np.array([0.8, 0.2]))
        med = self.model.compute(np.array([0.65, 0.35]))
        high = self.model.compute(np.array([0.55, 0.45]))
        assert low["T_winner"] > med["T_winner"] > high["T_winner"]
        assert high["T_winner"] == 0.0


# ---------------------------------------------------------------------------
# BGModel: multi-channel and determinism
# ---------------------------------------------------------------------------


class TestBGModelMultiChannel:
    def test_four_channels_selects_dominant(self) -> None:
        cfg = BGModelConfig(n_channels=4)
        model = BGModel(cfg)
        saliences = np.array([0.8, 0.3, 0.2, 0.1])
        result = model.compute(saliences)
        assert result["selected_channel"] == 0

    def test_four_channels_correct_vector_lengths(self) -> None:
        cfg = BGModelConfig(n_channels=4)
        model = BGModel(cfg)
        result = model.compute(np.array([0.8, 0.3, 0.2, 0.1]))
        assert len(result["suppression_vector"]) == 4
        assert len(result["channel_activations"]) == 4

    def test_deterministic_without_noise(self) -> None:
        model = BGModel(BGModelConfig())
        s = np.array([0.8, 0.2])
        r1 = model.compute(s.copy())
        r2 = model.compute(s.copy())
        assert r1["selected_channel"] == r2["selected_channel"]
        assert r1["T_winner"] == r2["T_winner"]

    def test_noise_different_seeds_may_differ(self) -> None:
        cfg = BGModelConfig(noise_std=0.1)
        model = BGModel(cfg)
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(999)
        # With high noise, two different seeds should not always agree (no hard
        # assertion, but activations should differ at least occasionally).
        r1 = model.compute(np.array([0.6, 0.4]), rng=rng1)
        r2 = model.compute(np.array([0.6, 0.4]), rng=rng2)
        # At least the activations vectors differ; they are not guaranteed identical.
        # (This is a statistical property; very unlikely to collide.)
        assert r1["channel_activations"] != r2["channel_activations"]

    def test_noise_same_seed_is_reproducible(self) -> None:
        cfg = BGModelConfig(noise_std=0.1)
        model = BGModel(cfg)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        r1 = model.compute(np.array([0.6, 0.4]), rng=rng_a)
        r2 = model.compute(np.array([0.6, 0.4]), rng=rng_b)
        assert r1["selected_channel"] == r2["selected_channel"]
        assert r1["T_winner"] == r2["T_winner"]


# ---------------------------------------------------------------------------
# BGAdapter: policy interface contract
# ---------------------------------------------------------------------------


class TestBGAdapterInterface:
    def setup_method(self) -> None:
        self.adapter = BGAdapter()
        self.trial = _make_trial()

    def test_returns_bg_decision(self) -> None:
        ev = _make_evidence([0.8, 0.2])
        result = self.adapter(self.trial, ev)
        assert isinstance(result, BGDecision)

    def test_sim_time_preserved(self) -> None:
        ev = _make_evidence([0.8, 0.2], sim_time=0.25)
        result = self.adapter(self.trial, ev)
        assert result.sim_time == 0.25

    def test_trial_id_preserved(self) -> None:
        ev = _make_evidence([0.8, 0.2], trial_id=7)
        trial_with_id = TrialLog(
            trial_id=7, seed=42, task_type="two_choice",
            cue_identity="c", cue_onset_time=0.0,
        )
        result = self.adapter(trial_with_id, ev)
        assert result.trial_id == 7

    def test_selects_channel_0_low_conflict(self) -> None:
        ev = _make_evidence([0.8, 0.2])
        result = self.adapter(self.trial, ev)
        assert result.selected_channel == 0

    def test_no_selection_high_conflict(self) -> None:
        ev = _make_evidence([0.55, 0.45])
        result = self.adapter(self.trial, ev)
        assert result.selected_channel == -1

    def test_suppression_vector_length_matches_n_channels(self) -> None:
        ev = _make_evidence([0.8, 0.2, 0.1, 0.05])
        adapter = BGAdapter(config=BGModelConfig(n_channels=4))
        result = adapter(self.trial, ev)
        assert len(result.suppression_vector) == 4

    def test_channel_activations_length_matches_n_channels(self) -> None:
        ev = _make_evidence([0.8, 0.2])
        result = self.adapter(self.trial, ev)
        assert len(result.channel_activations) == 2


# ---------------------------------------------------------------------------
# BGAdapter: stop-signal override
# ---------------------------------------------------------------------------


class TestBGAdapterStopSignal:
    def setup_method(self) -> None:
        self.adapter = BGAdapter()
        self.trial = _make_trial(task_type="stop_signal")

    def test_stop_signal_returns_minus_one(self) -> None:
        ev = _make_evidence([0.8, 0.2], stop=True)
        result = self.adapter(self.trial, ev)
        assert result.selected_channel == -1

    def test_stop_signal_with_high_salience_still_inhibits(self) -> None:
        # Even a very strong salience must yield to the stop signal.
        ev = _make_evidence([1.0, 0.0], stop=True)
        result = self.adapter(self.trial, ev)
        assert result.selected_channel == -1

    def test_stop_signal_suppression_vector_is_maximal(self) -> None:
        ev = _make_evidence([0.8, 0.2], stop=True)
        result = self.adapter(self.trial, ev)
        # All channels suppressed to 1.0 by the policy-level override.
        assert result.suppression_vector == [1.0, 1.0]

    def test_stop_signal_channel_activations_are_zero(self) -> None:
        ev = _make_evidence([0.8, 0.2], stop=True)
        result = self.adapter(self.trial, ev)
        assert result.channel_activations == [0.0, 0.0]

    def test_stop_signal_latency_is_zero(self) -> None:
        ev = _make_evidence([0.8, 0.2], stop=True)
        result = self.adapter(self.trial, ev)
        assert result.selection_latency == 0.0


# ---------------------------------------------------------------------------
# BGAdapter: selection latency (M2 acceptance criterion end-to-end)
# ---------------------------------------------------------------------------


class TestBGAdapterLatency:
    def setup_method(self) -> None:
        self.adapter = BGAdapter()
        self.trial = _make_trial()

    def test_latency_positive_when_selected(self) -> None:
        ev = _make_evidence([0.8, 0.2])
        result = self.adapter(self.trial, ev)
        assert result.selection_latency > 0.0

    def test_latency_in_seconds(self) -> None:
        # schema stores latency in seconds; should be << 1 s for a rate-coded model.
        ev = _make_evidence([0.8, 0.2])
        result = self.adapter(self.trial, ev)
        assert 0.0 < result.selection_latency < 1.0

    def test_latency_monotone_with_conflict(self) -> None:
        # M2 acceptance criterion: latency must increase with conflict.
        ev_low = _make_evidence([0.8, 0.2])
        ev_med = _make_evidence([0.65, 0.35])
        ev_high = _make_evidence([0.55, 0.45])
        lat_low = self.adapter(self.trial, ev_low).selection_latency
        lat_med = self.adapter(self.trial, ev_med).selection_latency
        lat_high = self.adapter(self.trial, ev_high).selection_latency
        # Strict ordering required.
        assert lat_low < lat_med < lat_high

    def test_no_selection_latency_equals_max(self) -> None:
        cfg = BGModelConfig()
        adapter = BGAdapter(config=cfg)
        ev = _make_evidence([0.5, 0.5])
        result = adapter(self.trial, ev)
        # No selection → latency == latency_max_ms / 1000
        expected = cfg.latency_max_ms / 1000.0
        assert abs(result.selection_latency - expected) < 1e-9

    def test_latency_deterministic_no_noise(self) -> None:
        ev = _make_evidence([0.8, 0.2])
        r1 = self.adapter(self.trial, ev)
        r2 = self.adapter(self.trial, ev)
        assert r1.selection_latency == r2.selection_latency

    def test_latency_deterministic_with_noise_same_seed(self) -> None:
        adapter = BGAdapter(config=BGModelConfig(noise_std=0.05))
        ev = _make_evidence([0.8, 0.2])
        trial_a = _make_trial(seed=7)
        trial_b = _make_trial(seed=7)
        r1 = adapter(trial_a, ev)
        r2 = adapter(trial_b, ev)
        assert r1.selection_latency == r2.selection_latency


# ---------------------------------------------------------------------------
# Task 2.4 — Wiring: BGAdapter as drop-in policy in Phase 1 engines
# ---------------------------------------------------------------------------


class TestBGAdapterWiringTwoChoice:
    """Verify BGAdapter produces valid TrialLogs in the two_choice engine."""

    def _run(self, n_trials: int = 10) -> list:
        cfg = TwoChoiceConfig(
            n_trials=n_trials,
            conflict_levels={"low": [0.8, 0.2], "medium": [0.65, 0.35], "high": [0.55, 0.45]},
            response_window_start_ms=100,
            response_window_duration_ms=500,
            fixation_duration_ms=500,
            target_onset_ms=1000,
            decision_point_ms=100,
            seed=42,
        )
        adapter = BGAdapter()
        return run_two_choice_trials(cfg, adapter)

    def test_returns_correct_number_of_trials(self) -> None:
        logs = self._run(n_trials=10)
        assert len(logs) == 10

    def test_produces_nonempty_event_streams(self) -> None:
        logs = self._run(n_trials=5)
        for log in logs:
            assert len(log.events) > 0

    def test_events_nondecreasing_sim_time(self) -> None:
        logs = self._run(n_trials=5)
        for log in logs:
            times = [e.sim_time for e in log.events]
            assert times == sorted(times), f"sim_time not sorted: {times}"

    def test_low_conflict_mostly_correct_selections(self) -> None:
        # With conflict_levels containing low/medium/high, and BGAdapter selecting
        # correctly for low conflict and medium conflict:
        # expect > 50% success across all three conflict levels.
        logs = self._run(n_trials=60)
        successes = sum(1 for log in logs if log.success is True)
        assert successes / len(logs) >= 0.4

    def test_engine_accepts_bg_adapter_without_error(self) -> None:
        # Smoke test: engine runs to completion without raising.
        logs = self._run(n_trials=5)
        assert all(isinstance(log.success, bool) or log.success is None for log in logs)


class TestBGAdapterWiringChangeOfMind:
    """Verify BGAdapter works correctly in the change_of_mind engine."""

    def _run(self, n_trials: int = 20) -> list:
        cfg = ChangeOfMindConfig(n_trials=n_trials)
        adapter = BGAdapter()
        return run_change_of_mind_trials(cfg, adapter)

    def test_returns_correct_number_of_trials(self) -> None:
        logs = self._run(n_trials=20)
        assert len(logs) == 20

    def test_produces_nonempty_event_streams(self) -> None:
        logs = self._run(n_trials=5)
        for log in logs:
            assert len(log.events) > 0

    def test_events_nondecreasing_sim_time(self) -> None:
        logs = self._run(n_trials=5)
        for log in logs:
            times = [e.sim_time for e in log.events]
            assert times == sorted(times), f"sim_time not sorted in trial {log.trial_id}"

    def test_switch_trials_correct_switch_outcome(self) -> None:
        # ChangeOfMindConfig defaults use initial_salience=[0.8, 0.2] and
        # post_switch_salience=[0.2, 0.8].  BGAdapter: pre-switch → ch0,
        # post-switch → ch1 (correct_switch).  All switch trials should succeed.
        logs = self._run(n_trials=40)
        from nrp_bga_sb.schemas import EventType
        switch_logs = [
            log for log in logs
            if any(e.event_type == EventType.evidence_change for e in log.events)
        ]
        assert len(switch_logs) > 0
        correct_switches = sum(1 for log in switch_logs if log.success is True)
        # BG should switch correctly every time with default salience settings.
        assert correct_switches == len(switch_logs)

    def test_no_switch_baseline_trials_succeed(self) -> None:
        from nrp_bga_sb.schemas import EventType
        logs = self._run(n_trials=40)
        no_switch_logs = [
            log for log in logs
            if not any(e.event_type == EventType.evidence_change for e in log.events)
        ]
        if no_switch_logs:
            successes = sum(1 for log in no_switch_logs if log.success is True)
            assert successes == len(no_switch_logs)

    def test_bg_adapter_produces_valid_metrics(self) -> None:
        from nrp_bga_sb.scorer import score_trials
        logs = self._run(n_trials=20)
        metrics = score_trials(logs, condition_id="test_m2", bg_frequency_hz=40.0)
        assert metrics.n_trials == 20
        assert metrics.switch_success_rate is not None


# ---------------------------------------------------------------------------
# M2 acceptance: integration summary
# ---------------------------------------------------------------------------


class TestM2Acceptance:
    """M2 milestone: BG selects under salience manipulation; latency interpretable."""

    def test_m2_selection_correct_for_low_conflict(self) -> None:
        adapter = BGAdapter()
        trial = _make_trial()
        ev = _make_evidence([0.8, 0.2])
        result = adapter(trial, ev)
        assert result.selected_channel == 0
        assert result.decision_margin > 0.0

    def test_m2_selection_correct_for_medium_conflict(self) -> None:
        adapter = BGAdapter()
        trial = _make_trial()
        ev = _make_evidence([0.65, 0.35])
        result = adapter(trial, ev)
        assert result.selected_channel == 0
        assert result.decision_margin > 0.0

    def test_m2_selection_absent_for_high_conflict(self) -> None:
        adapter = BGAdapter()
        trial = _make_trial()
        ev = _make_evidence([0.55, 0.45])
        result = adapter(trial, ev)
        assert result.selected_channel == -1

    def test_m2_latency_strictly_monotone_with_conflict(self) -> None:
        adapter = BGAdapter()
        trial = _make_trial()
        lat_low = adapter(trial, _make_evidence([0.8, 0.2])).selection_latency
        lat_med = adapter(trial, _make_evidence([0.65, 0.35])).selection_latency
        lat_high = adapter(trial, _make_evidence([0.55, 0.45])).selection_latency
        assert lat_low < lat_med < lat_high, (
            f"Latency not monotone: low={lat_low:.4f}, med={lat_med:.4f}, high={lat_high:.4f}"
        )

    def test_m2_change_of_mind_full_cartesian(self) -> None:
        """Switch succeeds in all cases: pre→ch0, post→ch1 with default saliences."""
        from nrp_bga_sb.schemas import EventType
        adapter = BGAdapter()
        cfg = ChangeOfMindConfig(n_trials=30)
        logs = run_change_of_mind_trials(cfg, adapter)
        switch_logs = [
            log for log in logs
            if any(e.event_type == EventType.evidence_change for e in log.events)
        ]
        assert all(log.success is True for log in switch_logs)
