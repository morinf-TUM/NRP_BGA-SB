"""Change-of-mind metrics module (Task 8.1).

Computes behavioral metrics for change-of-mind trials from a list of TrialLog
objects. This module is pure (no engine calls); it works offline from logged data.

Behavioral definitions:
- A switch trial contains an evidence_change event (the switch cue).
- A no-switch trial has cue_identity="no_switch" and no evidence_change event.
- Revision latency: time (ms) from evidence_change to the post-switch decision_commit.
- change_of_mind_probability: fraction of switch trials that succeeded (channel 1 selected).
- perseveration_rate: fraction of switch trials where failure_mode="perseveration".
- wrong_final_target_rate: fraction of switch trials where success=False (perseveration + miss).
- mean_revision_latency_ms: mean revision latency over switch trials that made a response
  (success=True OR failure_mode="perseveration"). Miss trials are excluded (no response).
"""

from __future__ import annotations

from pydantic import BaseModel

from nrp_bga_sb.schemas import EventType, TrialLog

# --- Trial identification ---


def is_switch_trial(trial: TrialLog) -> bool:
    """Return True if trial contains an evidence_change event."""
    return any(e.event_type == EventType.evidence_change for e in trial.events)


# --- Per-trial scalar ---


def revision_latency_ms(trial: TrialLog) -> float | None:
    """Time (ms) from the evidence_change event to the post-switch decision_commit event.

    Returns None for no-switch trials (no evidence_change present).
    Uses event sim_time × 1000 to convert back to ms.
    The evidence_change event has payload {"switch_category": ..., "switch_delay_ms": ...}.
    The post-switch decision_commit event has payload {"phase": "post_switch"}.
    """
    # Trigger: no evidence_change → this is a no-switch trial.
    # Why: revision latency is undefined without a switch cue; returning None signals
    #      the caller not to include this trial in latency statistics.
    # Outcome: caller excludes this trial from latency aggregates.
    change_time: float | None = None
    commit_time: float | None = None

    for event in trial.events:
        if event.event_type == EventType.evidence_change:
            change_time = event.sim_time
        elif (
            event.event_type == EventType.decision_commit
            and event.payload.get("phase") == "post_switch"
        ):
            commit_time = event.sim_time

    if change_time is None:
        # No evidence_change event → no-switch trial.
        return None

    # Trigger: post_switch decision_commit absent despite evidence_change being present.
    # Why: this is a data integrity violation — every switch trial must have a post-switch
    #      decision point regardless of outcome (miss, perseveration, or success).
    # Outcome: raises ValueError to surface the upstream engine bug immediately.
    if commit_time is None:
        raise ValueError(
            f"Switch trial {trial.trial_id} has evidence_change but no post-switch "
            "decision_commit event — data integrity violation."
        )

    return (commit_time - change_time) * 1000.0


# --- Aggregate metrics Pydantic model ---


class ChangeOfMindMetrics(BaseModel):
    """Aggregated change-of-mind metrics for one experimental condition."""

    n_trials: int
    n_switch_trials: int
    n_no_switch_trials: int

    # Overall switch trial rates (None if no switch trials present)
    change_of_mind_probability: float | None  # fraction of switch trials where success=True
    perseveration_rate: float | None          # fraction of switch trials where failure_mode="perseveration"  # noqa: E501
    wrong_final_target_rate: float | None     # fraction of switch trials where success=False
    mean_revision_latency_ms: float | None    # mean revision latency over responding switch trials

    # Per-category breakdown (key = switch_category name, e.g. "early")
    switch_success_by_category: dict[str, float]   # category → success rate
    perseveration_by_category: dict[str, float]    # category → perseveration rate
    revision_latency_by_category: dict[str, float] # category → mean revision latency (ms)


# --- Compute function ---


def compute_change_of_mind_metrics(trials: list[TrialLog]) -> ChangeOfMindMetrics:
    """Compute all change-of-mind metrics from a list of TrialLog objects.

    Works on mixed lists (switch trials + no-switch baseline trials).
    Separates by is_switch_trial(); computes per-category stats from
    cue_identity (e.g. "switch_early" → category "early").

    Returns a zero-trial ChangeOfMindMetrics (all rates None, empty dicts) for
    an empty input list — does not raise.
    """
    # --- Empty input guard ---
    # Trigger: empty trials list.
    # Why: unlike stop_signal_metrics which raises on empty input, the spec explicitly
    #      requires returning a zero-value ChangeOfMindMetrics without raising.
    # Outcome: all scalar rates are None and category dicts are empty.
    if not trials:
        return ChangeOfMindMetrics(
            n_trials=0,
            n_switch_trials=0,
            n_no_switch_trials=0,
            change_of_mind_probability=None,
            perseveration_rate=None,
            wrong_final_target_rate=None,
            mean_revision_latency_ms=None,
            switch_success_by_category={},
            perseveration_by_category={},
            revision_latency_by_category={},
        )

    # --- Partition trials by type ---
    switch_trials = [t for t in trials if is_switch_trial(t)]
    no_switch_trials = [t for t in trials if not is_switch_trial(t)]

    n_switch = len(switch_trials)
    n_no_switch = len(no_switch_trials)

    # --- Overall switch rates (None when no switch trials) ---
    # Trigger: no switch trials in the input list.
    # Why: division by zero; None signals "not applicable" rather than 0.0
    #      to distinguish a missing condition from a zero-success condition.
    # Outcome: all per-switch-trial rate fields are None.
    if n_switch == 0:
        com_probability: float | None = None
        persev_rate: float | None = None
        wrong_rate: float | None = None
        mean_latency: float | None = None
    else:
        n_success = sum(1 for t in switch_trials if t.success is True)
        n_persev = sum(
            1 for t in switch_trials if t.failure_mode == "perseveration"
        )
        n_failure = sum(1 for t in switch_trials if t.success is False)

        com_probability = n_success / n_switch
        persev_rate = n_persev / n_switch
        wrong_rate = n_failure / n_switch

        # Revision latency is defined only for switch trials that made a post-switch
        # response: success=True OR failure_mode="perseveration".
        # Miss trials have no response → no revision latency.
        latencies = [
            rl
            for t in switch_trials
            if t.failure_mode != "miss"
            for rl in (revision_latency_ms(t),)
            # Guard is dead for switch trials (raises or returns float), but kept for safety
            # against future engine changes that might introduce a None-returning code path.
            if rl is not None
        ]
        mean_latency = (sum(latencies) / len(latencies)) if latencies else None

    # --- Per-category breakdown ---
    # Category name extracted from cue_identity: "switch_early" → "early".
    # Accumulate per-category counts for success, perseveration, and latency.
    cat_n: dict[str, int] = {}
    cat_success: dict[str, int] = {}
    cat_persev: dict[str, int] = {}
    cat_latencies: dict[str, list[float]] = {}

    for t in switch_trials:
        # Extract category from cue_identity by stripping the "switch_" prefix.
        # Trigger: cue_identity always starts with "switch_" for switch trials.
        # Why: is_switch_trial ensures the evidence_change event is present; the engine
        #      always sets cue_identity = f"switch_{cat_name}" for switch trials.
        # Outcome: category key is the bare name (e.g. "early", "very_late").
        cat = t.cue_identity.removeprefix("switch_")

        if cat not in cat_n:
            cat_n[cat] = 0
            cat_success[cat] = 0
            cat_persev[cat] = 0
            cat_latencies[cat] = []

        cat_n[cat] += 1
        if t.success is True:
            cat_success[cat] += 1
        if t.failure_mode == "perseveration":
            cat_persev[cat] += 1

        # Accumulate revision latency for responding trials (not miss).
        if t.failure_mode != "miss":
            rl = revision_latency_ms(t)
            # Guard is dead for switch trials (raises or returns float), but kept for safety
            # against future engine changes that might introduce a None-returning code path.
            if rl is not None:
                cat_latencies[cat].append(rl)

    switch_success_by_category: dict[str, float] = {
        cat: cat_success[cat] / cat_n[cat] for cat in cat_n
    }
    perseveration_by_category: dict[str, float] = {
        cat: cat_persev[cat] / cat_n[cat] for cat in cat_n
    }
    # Trigger: a category has no responding trials (all misses).
    # Why: mean of an empty list is undefined; 0.0 is returned to keep the dict
    #      fully populated while signalling no measurable latency for this category.
    # Outcome: category key present with value 0.0 (caller can cross-check with
    #          switch_success_by_category to detect this degenerate case).
    revision_latency_by_category: dict[str, float] = {
        cat: (sum(lats) / len(lats)) if cat_latencies[cat] else 0.0
        for cat, lats in cat_latencies.items()
    }

    return ChangeOfMindMetrics(
        n_trials=len(trials),
        n_switch_trials=n_switch,
        n_no_switch_trials=n_no_switch,
        change_of_mind_probability=com_probability,
        perseveration_rate=persev_rate,
        wrong_final_target_rate=wrong_rate,
        mean_revision_latency_ms=mean_latency,
        switch_success_by_category=switch_success_by_category,
        perseveration_by_category=perseveration_by_category,
        revision_latency_by_category=revision_latency_by_category,
    )
