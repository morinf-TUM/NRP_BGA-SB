"""Minimal scorer: aggregate trial logs into condition-level metrics."""

import numpy as np

from nrp_bga_sb.schemas import EventType, Metrics, TrialLog


def score_trials(
    trials: list[TrialLog],
    condition_id: str,
    bg_frequency_hz: float,
) -> Metrics:
    """Compute aggregate metrics for a batch of trials.

    Raises ValueError if trials is empty (caller must pass a non-empty batch).
    """
    if not trials:
        raise ValueError("score_trials requires at least one trial")

    n = len(trials)

    # --- Reaction Time ---
    rts = [
        t.movement_onset_time - t.cue_onset_time
        for t in trials
        if t.movement_onset_time is not None
    ]
    if len(rts) == 0:
        rt_mean = None
        rt_std = None
    elif len(rts) == 1:
        rt_mean = float(rts[0])
        rt_std = None
    else:
        rt_mean = float(np.mean(rts))
        rt_std = float(np.std(rts, ddof=1))

    # --- Wrong-action rate ---
    wrong_action_count = sum(1 for t in trials if t.failure_mode == "wrong_action")
    wrong_action_rate = wrong_action_count / n

    # --- False-alarm rate ---
    no_go_trials = [
        t for t in trials
        if any(e.event_type == EventType.no_go_cue for e in t.events)
    ]
    if no_go_trials:
        false_alarms = [t for t in no_go_trials if t.movement_onset_time is not None]
        false_alarm_rate = len(false_alarms) / len(no_go_trials)
    else:
        false_alarm_rate = None

    return Metrics(
        condition_id=condition_id,
        bg_frequency_hz=bg_frequency_hz,
        n_trials=n,
        reaction_time_mean=rt_mean,
        reaction_time_std=rt_std,
        wrong_action_rate=wrong_action_rate,
        false_alarm_rate=false_alarm_rate,
    )
