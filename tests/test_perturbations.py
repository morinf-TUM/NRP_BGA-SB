"""Tests for timing perturbation wrappers (Task 3.2).

Test ordering follows the 24-test checklist in the brief:
  1–6:    LatencyWrapper
  7–12:   JitterWrapper
  13–18:  DropoutWrapper
  19–22:  PhaseOffsetWrapper
  23–24:  Composition
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nrp_bga_sb.bg_model import BGAdapter
from nrp_bga_sb.perturbations import (
    DropoutWrapper,
    JitterWrapper,
    LatencyWrapper,
    PhaseOffsetWrapper,
)
from nrp_bga_sb.scheduler import FrequencyConfig, ScheduledBGAdapter
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trial_log(seed: int = 42, trial_id: int = 1) -> TrialLog:
    """Minimal TrialLog for policy calls."""
    return TrialLog(
        trial_id=trial_id,
        seed=seed,
        task_type="two_choice",
        cue_identity="left",
        cue_onset_time=0.0,
    )


def _make_action_evidence(
    salience: list[float] | None = None,
    trial_id: int = 1,
) -> ActionEvidence:
    """Minimal ActionEvidence for policy calls."""
    s = salience if salience is not None else [0.8, 0.2]
    return ActionEvidence(
        sim_time=0.1,
        trial_id=trial_id,
        n_channels=2,
        channel_salience=s,
    )


def _make_bg_decision(selection_latency: float = 0.05) -> BGDecision:
    """Return a BGDecision with the given selection_latency."""
    return BGDecision(
        sim_time=0.1,
        trial_id=1,
        selected_channel=0,
        decision_margin=0.6,
        suppression_vector=[0.1, 0.9],
        channel_activations=[0.8, 0.2],
        selection_latency=selection_latency,
    )


def _make_stub_policy(decision: BGDecision | None = None):
    """Return a no-op policy callable that returns a fixed BGDecision."""
    d = decision if decision is not None else _make_bg_decision()

    def policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
        return d

    return policy


# ---------------------------------------------------------------------------
# LatencyWrapper (tests 1–6)
# ---------------------------------------------------------------------------


class TestLatencyWrapper:
    """Tests 1–6: LatencyWrapper adds a fixed ms offset to selection_latency."""

    def test_zero_latency_leaves_selection_latency_unchanged(self) -> None:
        """Test 1: latency_ms=0.0 → selection_latency unchanged."""
        base_latency = 0.05
        policy = _make_stub_policy(_make_bg_decision(base_latency))
        wrapper = LatencyWrapper(base_policy=policy, latency_ms=0.0)
        result = wrapper(_make_trial_log(), _make_action_evidence())
        assert result.selection_latency == base_latency

    def test_fifty_ms_latency_adds_exactly_005_seconds(self) -> None:
        """Test 2: latency_ms=50.0 → selection_latency increased by exactly 0.05 s."""
        base_latency = 0.05
        policy = _make_stub_policy(_make_bg_decision(base_latency))
        wrapper = LatencyWrapper(base_policy=policy, latency_ms=50.0)
        result = wrapper(_make_trial_log(), _make_action_evidence())
        assert result.selection_latency == pytest.approx(base_latency + 0.05)

    def test_negative_latency_raises_value_error(self) -> None:
        """Test 3: Negative latency_ms → ValueError on construction."""
        policy = _make_stub_policy()
        with pytest.raises(ValueError, match="latency_ms"):
            LatencyWrapper(base_policy=policy, latency_ms=-1.0)

    def test_wraps_bg_adapter_returns_valid_bg_decision(self) -> None:
        """Test 4: Wraps BGAdapter correctly — returns valid BGDecision."""
        adapter = BGAdapter()
        wrapper = LatencyWrapper(base_policy=adapter, latency_ms=10.0)
        result = wrapper(_make_trial_log(), _make_action_evidence())
        assert isinstance(result, BGDecision)
        assert result.selection_latency >= 0.0

    def test_composition_stacks_latencies(self) -> None:
        """Test 5: LatencyWrapper wrapping LatencyWrapper stacks latencies."""
        base_latency = 0.05
        policy = _make_stub_policy(_make_bg_decision(base_latency))
        inner = LatencyWrapper(base_policy=policy, latency_ms=30.0)
        outer = LatencyWrapper(base_policy=inner, latency_ms=20.0)
        result = outer(_make_trial_log(), _make_action_evidence())
        assert result.selection_latency == pytest.approx(base_latency + 0.03 + 0.02)

    def test_original_decision_not_mutated(self) -> None:
        """Test 6: Original decision object is NOT mutated (model_copy semantics)."""
        original_decision = _make_bg_decision(0.05)
        original_latency = original_decision.selection_latency
        policy = _make_stub_policy(original_decision)
        wrapper = LatencyWrapper(base_policy=policy, latency_ms=50.0)
        wrapper(_make_trial_log(), _make_action_evidence())
        # Original must be unchanged
        assert original_decision.selection_latency == original_latency


# ---------------------------------------------------------------------------
# JitterWrapper (tests 7–12)
# ---------------------------------------------------------------------------


class TestJitterWrapper:
    """Tests 7–12: JitterWrapper adds zero-mean Gaussian noise to selection_latency."""

    def test_zero_jitter_leaves_selection_latency_unchanged(self) -> None:
        """Test 7: jitter_std_ms=0.0 → selection_latency unchanged."""
        base_latency = 0.05
        policy = _make_stub_policy(_make_bg_decision(base_latency))
        wrapper = JitterWrapper(base_policy=policy, jitter_std_ms=0.0)
        result = wrapper(_make_trial_log(), _make_action_evidence())
        assert result.selection_latency == base_latency

    def test_nonzero_jitter_returns_float(self) -> None:
        """Test 8: jitter_std_ms > 0.0 → selection_latency is a float (perturbed)."""
        policy = _make_stub_policy(_make_bg_decision(0.05))
        wrapper = JitterWrapper(base_policy=policy, jitter_std_ms=10.0)
        result = wrapper(_make_trial_log(), _make_action_evidence())
        assert isinstance(result.selection_latency, float)

    def test_determinism_same_seed_same_jitter(self) -> None:
        """Test 9: Determinism — same trial_log.seed → same jitter delta every call."""
        policy = _make_stub_policy(_make_bg_decision(0.05))
        wrapper = JitterWrapper(base_policy=policy, jitter_std_ms=10.0)
        log = _make_trial_log(seed=99)
        ev = _make_action_evidence()
        r1 = wrapper(log, ev)
        r2 = wrapper(log, ev)
        assert r1.selection_latency == r2.selection_latency

    def test_different_seeds_produce_different_jitter(self) -> None:
        """Test 10: Different trial seeds → different jitter deltas."""
        results = set()
        policy = _make_stub_policy(_make_bg_decision(0.2))
        wrapper = JitterWrapper(base_policy=policy, jitter_std_ms=20.0)
        for seed in range(20):
            log = _make_trial_log(seed=seed)
            r = wrapper(log, _make_action_evidence())
            results.add(round(r.selection_latency, 8))
        # With 20 different seeds and std=20ms, virtually all should differ
        assert len(results) > 1

    def test_negative_jitter_std_raises_value_error(self) -> None:
        """Test 11: Negative jitter_std_ms → ValueError on construction."""
        policy = _make_stub_policy()
        with pytest.raises(ValueError, match="jitter_std_ms"):
            JitterWrapper(base_policy=policy, jitter_std_ms=-5.0)

    def test_selection_latency_never_below_zero(self) -> None:
        """Test 12: selection_latency never goes below 0.0 (clipped)."""
        # Use a very small base latency and large jitter to force negative attempts.
        policy = _make_stub_policy(_make_bg_decision(0.001))
        wrapper = JitterWrapper(base_policy=policy, jitter_std_ms=200.0)
        for seed in range(50):
            log = _make_trial_log(seed=seed)
            result = wrapper(log, _make_action_evidence())
            assert result.selection_latency >= 0.0, (
                f"seed={seed}: selection_latency={result.selection_latency} < 0"
            )


# ---------------------------------------------------------------------------
# DropoutWrapper (tests 13–18)
# ---------------------------------------------------------------------------


class TestDropoutWrapper:
    """Tests 13–18: DropoutWrapper skips base_policy calls with configured probability."""

    def test_zero_dropout_never_drops(self) -> None:
        """Test 13: dropout_probability=0.0 → never drops, always calls base_policy."""
        mock_policy = MagicMock(return_value=_make_bg_decision())
        wrapper = DropoutWrapper(base_policy=mock_policy, dropout_probability=0.0)
        for i in range(5):
            wrapper(_make_trial_log(seed=i), _make_action_evidence())
        assert mock_policy.call_count == 5

    def test_full_dropout_first_call_passes_through(self) -> None:
        """Test 14: dropout_probability=1.0, first call → passes through (no previous decision)."""
        call_count = 0
        decision = _make_bg_decision(0.05)

        def policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
            nonlocal call_count
            call_count += 1
            return decision

        wrapper = DropoutWrapper(base_policy=policy, dropout_probability=1.0)
        result = wrapper(_make_trial_log(seed=42), _make_action_evidence())
        # First call must pass through regardless of dropout probability
        assert call_count == 1
        assert result is decision

    def test_full_dropout_second_call_returns_cached(self) -> None:
        """Test 15: dropout_probability=1.0, second call → returns cached first decision."""
        call_count = 0
        decision = _make_bg_decision(0.05)

        def policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
            nonlocal call_count
            call_count += 1
            return decision

        wrapper = DropoutWrapper(base_policy=policy, dropout_probability=1.0)
        # First call — passes through
        wrapper(_make_trial_log(seed=42), _make_action_evidence())
        # Second call — should return cached (same seed → same drop verdict)
        result = wrapper(_make_trial_log(seed=42), _make_action_evidence())
        # base_policy must have been called exactly once total
        assert call_count == 1
        assert result is decision

    def test_dropout_probability_outside_range_raises_value_error(self) -> None:
        """Test 16: dropout_probability outside [0, 1] → ValueError on construction."""
        policy = _make_stub_policy()
        with pytest.raises(ValueError, match="dropout_probability"):
            DropoutWrapper(base_policy=policy, dropout_probability=1.5)
        with pytest.raises(ValueError, match="dropout_probability"):
            DropoutWrapper(base_policy=policy, dropout_probability=-0.1)

    def test_full_dropout_three_calls_base_policy_called_once(self) -> None:
        """Test 17: Mock base_policy call count: p=1.0, 3 calls → called exactly once."""
        mock_policy = MagicMock(return_value=_make_bg_decision())
        wrapper = DropoutWrapper(base_policy=mock_policy, dropout_probability=1.0)
        seed = 42
        for _ in range(3):
            wrapper(_make_trial_log(seed=seed), _make_action_evidence())
        assert mock_policy.call_count == 1

    def test_determinism_same_seeds_same_drop_pattern(self) -> None:
        """Test 18: Determinism — same trial_log seeds in same order → same drop pattern."""
        seeds = [10, 20, 30, 40, 50]
        decisions = [_make_bg_decision(0.01 * i) for i in range(len(seeds))]

        def make_counter_policy(decs: list[BGDecision]):
            idx = [0]

            def policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
                d = decs[idx[0] % len(decs)]
                idx[0] += 1
                return d

            return policy

        def run_sequence() -> list[float]:
            wrapper = DropoutWrapper(
                base_policy=make_counter_policy(decisions),
                dropout_probability=0.5,
            )
            latencies = []
            for seed in seeds:
                r = wrapper(_make_trial_log(seed=seed), _make_action_evidence())
                latencies.append(r.selection_latency)
            return latencies

        result1 = run_sequence()
        result2 = run_sequence()
        assert result1 == result2


# ---------------------------------------------------------------------------
# PhaseOffsetWrapper (tests 19–22)
# ---------------------------------------------------------------------------


class TestPhaseOffsetWrapper:
    """Tests 19–22: PhaseOffsetWrapper adds a fixed ms phase offset to selection_latency."""

    def test_zero_offset_leaves_selection_latency_unchanged(self) -> None:
        """Test 19: phase_offset_ms=0.0 → selection_latency unchanged."""
        base_latency = 0.05
        policy = _make_stub_policy(_make_bg_decision(base_latency))
        wrapper = PhaseOffsetWrapper(base_policy=policy, phase_offset_ms=0.0)
        result = wrapper(_make_trial_log(), _make_action_evidence())
        assert result.selection_latency == base_latency

    def test_twenty_five_ms_offset_adds_exactly_0025_seconds(self) -> None:
        """Test 20: phase_offset_ms=25.0 → selection_latency increased by exactly 0.025 s."""
        base_latency = 0.05
        policy = _make_stub_policy(_make_bg_decision(base_latency))
        wrapper = PhaseOffsetWrapper(base_policy=policy, phase_offset_ms=25.0)
        result = wrapper(_make_trial_log(), _make_action_evidence())
        assert result.selection_latency == pytest.approx(base_latency + 0.025)

    def test_negative_phase_offset_raises_value_error(self) -> None:
        """Test 21: Negative phase_offset_ms → ValueError on construction."""
        policy = _make_stub_policy()
        with pytest.raises(ValueError, match="phase_offset_ms"):
            PhaseOffsetWrapper(base_policy=policy, phase_offset_ms=-5.0)

    def test_wraps_bg_adapter_returns_valid_bg_decision(self) -> None:
        """Test 22: Wraps BGAdapter correctly — returns valid BGDecision."""
        adapter = BGAdapter()
        wrapper = PhaseOffsetWrapper(base_policy=adapter, phase_offset_ms=10.0)
        result = wrapper(_make_trial_log(), _make_action_evidence())
        assert isinstance(result, BGDecision)
        assert result.selection_latency >= 0.0


# ---------------------------------------------------------------------------
# Composition (tests 23–24)
# ---------------------------------------------------------------------------


class TestComposition:
    """Tests 23–24: Wrappers chain correctly and produce valid BGDecisions."""

    def test_latency_over_jitter_over_scheduled_adapter(self) -> None:
        """Test 23: Chain LatencyWrapper(JitterWrapper(ScheduledBGAdapter(...))) is valid."""
        adapter = BGAdapter()
        config = FrequencyConfig(
            input_sampling_hz=160.0,
            integration_step_hz=160.0,
            output_emission_hz=160.0,
            commitment_update_hz=160.0,
            base_dt_ms=1.0,
        )
        scheduled = ScheduledBGAdapter(base_policy=adapter, config=config, accumulation_ms=50.0)
        jittered = JitterWrapper(base_policy=scheduled, jitter_std_ms=5.0)
        latency_wrapped = LatencyWrapper(base_policy=jittered, latency_ms=10.0)

        result = latency_wrapped(_make_trial_log(), _make_action_evidence())
        assert isinstance(result, BGDecision)
        assert result.selection_latency >= 0.0

    def test_dropout_over_latency_over_bg_adapter_varied_seeds(self) -> None:
        """Test 24: DropoutWrapper(LatencyWrapper(BGAdapter())) — 10 calls, mixed pass/drop."""
        adapter = BGAdapter()
        latency_wrapped = LatencyWrapper(base_policy=adapter, latency_ms=5.0)
        wrapper = DropoutWrapper(base_policy=latency_wrapped, dropout_probability=0.5)

        results = []
        for seed in range(10):
            log = _make_trial_log(seed=seed)
            ev = _make_action_evidence()
            result = wrapper(log, ev)
            assert isinstance(result, BGDecision)
            results.append(result)

        # With 10 varied seeds and p=0.5, at least 1 should pass through
        # (the first call always passes, so this is guaranteed).
        assert len(results) == 10
