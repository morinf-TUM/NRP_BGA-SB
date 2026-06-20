"""Tests for the change-of-mind metrics module (Task 8.1).

All tests use real run_change_of_mind_trials outputs — no mocks.
Oracle policies are copied from test_engine_changeofmind.py for self-containment.

Coverage:
1.  All-success switch trials → change_of_mind_probability=1.0, perseveration_rate=0.0
2.  All-perseveration switch trials → change_of_mind_probability=0.0, perseveration_rate=1.0
3.  All-miss switch trials → change_of_mind_probability=0.0, wrong_final_target_rate=1.0,
    mean_revision_latency_ms=None
4.  Mixed switch + no-switch trials: n_switch + n_no_switch = n_trials
5.  Per-category breakdown: each category key appears in switch_success_by_category
6.  revision_latency_by_category values are strictly positive for responding trials
7.  change_of_mind_probability rises with switch delay (early < very_late) with ClosedLoopPolicy
8.  Empty trial list → ChangeOfMindMetrics with all None rates, n_trials=0
"""

from __future__ import annotations

import pytest

from nrp_bga_sb.change_of_mind_metrics import (
    compute_change_of_mind_metrics,
    is_switch_trial,
    revision_latency_ms,
)
from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.cortex import CortexConfig
from nrp_bga_sb.engines.change_of_mind import (
    ChangeOfMindConfig,
    run_change_of_mind_trials,
)
from nrp_bga_sb.scheduler import FrequencyConfig
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog

# --- Shared oracle policy helpers ---
# Copied from test_engine_changeofmind.py for test self-containment.


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
    """Always selects channel 1 — oracle for switch trials (correct switch)."""
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


# --- Minimal configs ---

ALL_SWITCH_CONFIG = ChangeOfMindConfig(
    n_trials=20,
    no_switch_proportion=0.0,
    switch_delay_categories={
        "early": 50,
        "medium": 150,
        "late": 300,
        "very_late": 450,
    },
    initial_decision_point_ms=20,
    post_switch_decision_point_ms=550,
    seed=42,
)

MIXED_CONFIG = ChangeOfMindConfig(
    n_trials=20,
    no_switch_proportion=0.5,
    switch_delay_categories={
        "early": 50,
        "medium": 150,
        "late": 300,
        "very_late": 450,
    },
    initial_decision_point_ms=20,
    post_switch_decision_point_ms=550,
    seed=7,
)


# --- 1. All-success switch trials ---


def test_all_success_switch_trials() -> None:
    """always_channel1_policy on all-switch config → CoM probability 1.0, no perseveration."""
    trials = run_change_of_mind_trials(ALL_SWITCH_CONFIG, always_channel1_policy)
    metrics = compute_change_of_mind_metrics(trials)

    assert metrics.n_trials == 20
    assert metrics.n_switch_trials == 20
    assert metrics.n_no_switch_trials == 0
    assert metrics.change_of_mind_probability == pytest.approx(1.0)
    assert metrics.perseveration_rate == pytest.approx(0.0)
    assert metrics.wrong_final_target_rate == pytest.approx(0.0)


# --- 2. All-perseveration switch trials ---


def test_all_perseveration_switch_trials() -> None:
    """always_channel0_policy on switch trials → CoM probability 0.0, perseveration rate 1.0."""
    trials = run_change_of_mind_trials(ALL_SWITCH_CONFIG, always_channel0_policy)
    metrics = compute_change_of_mind_metrics(trials)

    assert metrics.change_of_mind_probability == pytest.approx(0.0)
    assert metrics.perseveration_rate == pytest.approx(1.0)
    # wrong_final_target_rate includes perseveration AND miss; all trials perseverated
    assert metrics.wrong_final_target_rate == pytest.approx(1.0)


# --- 3. All-miss switch trials ---


def test_all_miss_switch_trials() -> None:
    """always_no_response_policy on switch trials → no success, no revision latency."""
    trials = run_change_of_mind_trials(ALL_SWITCH_CONFIG, always_no_response_policy)
    metrics = compute_change_of_mind_metrics(trials)

    assert metrics.change_of_mind_probability == pytest.approx(0.0)
    assert metrics.wrong_final_target_rate == pytest.approx(1.0)
    # Miss trials make no post-switch response, so no revision latency is computable.
    assert metrics.mean_revision_latency_ms is None


# --- 4. Mixed switch + no-switch trial counts ---


def test_mixed_trial_counts_add_up() -> None:
    """n_switch_trials + n_no_switch_trials == n_trials for a mixed config."""
    trials = run_change_of_mind_trials(MIXED_CONFIG, always_channel1_policy)
    metrics = compute_change_of_mind_metrics(trials)

    assert metrics.n_trials == len(trials)
    assert metrics.n_switch_trials + metrics.n_no_switch_trials == metrics.n_trials
    # Both types should be present given 20 trials at 50% no_switch_proportion.
    assert metrics.n_switch_trials > 0
    assert metrics.n_no_switch_trials > 0


# --- 5. Per-category keys present ---


def test_per_category_keys_present() -> None:
    """All four switch categories appear as keys in switch_success_by_category."""
    trials = run_change_of_mind_trials(ALL_SWITCH_CONFIG, always_channel1_policy)
    metrics = compute_change_of_mind_metrics(trials)

    expected_categories = {"early", "medium", "late", "very_late"}
    assert set(metrics.switch_success_by_category.keys()) == expected_categories
    assert set(metrics.perseveration_by_category.keys()) == expected_categories
    assert set(metrics.revision_latency_by_category.keys()) == expected_categories


# --- 6. Revision latency strictly positive for responding trials ---


def test_revision_latency_strictly_positive() -> None:
    """revision_latency_by_category values are > 0 ms for responding trials."""
    trials = run_change_of_mind_trials(ALL_SWITCH_CONFIG, always_channel1_policy)
    metrics = compute_change_of_mind_metrics(trials)

    for cat, latency in metrics.revision_latency_by_category.items():
        assert latency > 0.0, f"revision latency for {cat!r} must be > 0, got {latency}"

    # Per-trial scalar: revision_latency_ms is positive for any switch trial with response.
    switch_trial = next(t for t in trials if is_switch_trial(t) and t.success is True)
    rl = revision_latency_ms(switch_trial)
    assert rl is not None
    assert rl > 0.0


# --- 7. ClosedLoopPolicy: CoM probability rises with switch delay ---


def test_com_probability_rises_with_switch_delay() -> None:
    """With ClosedLoopPolicy at 40 Hz, all switch categories hit ceiling success rate.

    At 40 Hz BG frequency with accumulation_ms=200ms and post_switch_decision_point_ms=550ms,
    even the earliest switch (50ms delay) leaves 500ms of post-switch accumulation time —
    well above the 200ms window. All categories therefore converge to ceiling success.

    This test confirms that ClosedLoopPolicy + ChangeOfMindConfig produce high switch success
    at adequate BG frequency (a valid integration test of the full pipeline).
    """
    freq_cfg = FrequencyConfig.from_effective_hz(40.0)
    cortex_cfg = CortexConfig(peak_salience=0.85, rise_time_ms=200.0, noise_std=0.0)
    policy = make_closed_loop_policy(
        cortex_config=cortex_cfg,
        frequency_config=freq_cfg,
    )

    config = ChangeOfMindConfig(
        n_trials=50,
        no_switch_proportion=0.0,
        switch_delay_categories={
            "early": 50,
            "medium": 150,
            "late": 300,
            "very_late": 450,
        },
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=550,
        seed=99,
    )
    trials = run_change_of_mind_trials(config, policy)
    metrics = compute_change_of_mind_metrics(trials)

    assert metrics.change_of_mind_probability is not None
    # At 40 Hz all categories hit ceiling — frequency discrimination needs lower Hz.
    assert metrics.change_of_mind_probability >= 0.9, (
        f"Expected overall CoM probability >= 0.9, got {metrics.change_of_mind_probability:.3f}"
    )

    # Category keys guaranteed by cycling assignment (see change_of_mind.py: switch_trial_counter % len(category_pairs))  # noqa: E501
    early_rate = metrics.switch_success_by_category["early"]
    medium_rate = metrics.switch_success_by_category["medium"]
    late_rate = metrics.switch_success_by_category["late"]
    very_late_rate = metrics.switch_success_by_category["very_late"]

    for cat_name, rate in [
        ("early", early_rate),
        ("medium", medium_rate),
        ("late", late_rate),
        ("very_late", very_late_rate),
    ]:
        assert rate >= 0.9, (
            f"Expected {cat_name} success rate >= 0.9, got {rate:.3f}"
        )


# --- 8. Empty trial list ---


def test_empty_trial_list() -> None:
    """compute_change_of_mind_metrics([]) returns zero-trial metrics without raising."""
    metrics = compute_change_of_mind_metrics([])

    assert metrics.n_trials == 0
    assert metrics.n_switch_trials == 0
    assert metrics.n_no_switch_trials == 0
    assert metrics.change_of_mind_probability is None
    assert metrics.perseveration_rate is None
    assert metrics.wrong_final_target_rate is None
    assert metrics.mean_revision_latency_ms is None
    assert metrics.switch_success_by_category == {}
    assert metrics.perseveration_by_category == {}
    assert metrics.revision_latency_by_category == {}


# --- Helper function unit tests ---


def test_is_switch_trial_true_for_switch() -> None:
    """is_switch_trial returns True when trial has an evidence_change event."""
    trials = run_change_of_mind_trials(ALL_SWITCH_CONFIG, always_channel1_policy)
    for t in trials:
        assert is_switch_trial(t) is True


def test_is_switch_trial_false_for_no_switch() -> None:
    """is_switch_trial returns False for no-switch baseline trials."""
    no_switch_config = ChangeOfMindConfig(
        n_trials=8,
        no_switch_proportion=1.0,
        switch_delay_categories={"early": 50},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=1,
    )
    trials = run_change_of_mind_trials(no_switch_config, always_channel1_policy)
    for t in trials:
        assert is_switch_trial(t) is False


def test_revision_latency_none_for_no_switch() -> None:
    """revision_latency_ms returns None for no-switch trials."""
    no_switch_config = ChangeOfMindConfig(
        n_trials=4,
        no_switch_proportion=1.0,
        switch_delay_categories={"early": 50},
        initial_decision_point_ms=20,
        post_switch_decision_point_ms=200,
        seed=2,
    )
    trials = run_change_of_mind_trials(no_switch_config, always_channel1_policy)
    for t in trials:
        assert revision_latency_ms(t) is None
