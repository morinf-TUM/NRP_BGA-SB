"""Two-choice conflict engine: parameterizable salience gap / target separation.

The two-choice task presents two targets (left and right) with manipulable salience
values that operationalize decision conflict. On each trial, the agent must select
the target (channel) with higher salience within the response window.

A trial is successful when:
- Agent selects the channel with higher salience (correct target) within response window
  → success

A trial fails when:
- Agent selects the channel with lower salience (wrong target) → wrong_target
- Agent selects neither channel (-1) or makes no decision within response window → timeout

The engine operates on a logical clock (integer ms). It uses a seeded random
generator to determine conflict levels and target assignments. Conflict is
operationalized as the salience gap: a small gap means high conflict (hard decision),
a large gap means low conflict (easy decision).

Conflict levels map to salience distributions. For example:
- Low conflict: [0.8, 0.2] (gap = 0.6)
- Medium conflict: [0.65, 0.35] (gap = 0.3)
- High conflict: [0.55, 0.45] (gap = 0.1)

Target assignment (which channel is correct) is randomly counterbalanced to avoid
systematic left/right biases.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, EventType, TaskEvent, TrialLog

# --- Configuration ---


@dataclass
class TwoChoiceConfig:
    """Configuration for the two-choice engine.

    Attributes:
        n_trials: total number of trials to run.
        conflict_levels: mapping of conflict level name to salience gap values.
            Example: {"low": [0.8, 0.2], "medium": [0.65, 0.35], "high": [0.55, 0.45]}
            Each pair (higher_salience, lower_salience) defines the two channels.
            By convention, channel 0 = left, channel 1 = right. The correct target
            (which salience is "correct") is randomized per trial.
        response_window_start_ms: delay from target onset to response window opening (ms).
        response_window_duration_ms: duration the response window is open (ms).
        fixation_duration_ms: time from trial start to fixation signal (ms).
        target_onset_ms: time from trial start to target onset (ms).
        decision_point_ms: offset from target onset (in ms) at which the policy is called
            and decision is made. All timing checks (e.g., response window validity)
            use target onset as the reference point (time 0).
        seed: random seed for trial generation.
    """
    n_trials: int
    conflict_levels: dict[str, list[float]]
    response_window_start_ms: int
    response_window_duration_ms: int
    fixation_duration_ms: int
    target_onset_ms: int
    decision_point_ms: int
    seed: int


@dataclass
class TwoChoiceState:
    """Mutable state for a single trial being executed.

    Attributes:
        trial_log: the TrialLog being assembled.
        conflict_level: the conflict level label for this trial (e.g., "low").
        channel_salience: the two salience values [higher, lower].
        correct_channel: which channel (0 or 1) has the higher salience.
        decision: the BGDecision returned by the policy.
        responded: True if an action was selected (channel >= 0).
    """
    trial_log: TrialLog
    conflict_level: str
    channel_salience: list[float]
    correct_channel: int
    decision: BGDecision | None = None
    responded: bool = False


# --- Main Engine ---


def run_two_choice_trials(
    config: TwoChoiceConfig,
    policy: Callable[[TrialLog, ActionEvidence], BGDecision],
    logger: TrialLogger | None = None,
) -> list[TrialLog]:
    """Run a sequence of two-choice trials, returning completed TrialLog objects.

    Args:
        config: TwoChoiceConfig with trial parameters, conflict levels, and timing.
        policy: callable(trial_log, action_evidence) -> BGDecision.
                Receives the trial log and evidence at decision point;
                returns a BGDecision with selected_channel (-1 for no action).
        logger: optional TrialLogger to persist trials to disk. If None,
                trials are not written.

    Returns:
        list of completed TrialLog objects, one per trial.
    """
    trials = []
    rng = random.Random(config.seed)

    # Build a list of conflict levels to cycle through (rounded-robin distribution)
    conflict_level_names = list(config.conflict_levels.keys())
    if not conflict_level_names:
        raise ValueError("conflict_levels dict must not be empty")

    for trial_idx in range(config.n_trials):
        trial_id = trial_idx + 1
        trial_seed = rng.randint(0, 2**31 - 1)
        trial_rng = random.Random(trial_seed)

        # --- Determine conflict level (round-robin across configured levels) ---
        conflict_level = conflict_level_names[trial_idx % len(conflict_level_names)]
        salience_pair = config.conflict_levels[conflict_level]

        # salience_pair is [higher_salience, lower_salience]
        # Randomly decide which channel (0 or 1) gets the higher salience
        correct_channel = trial_rng.randint(0, 1)
        if correct_channel == 0:
            channel_salience = [salience_pair[0], salience_pair[1]]  # ch0=higher, ch1=lower
        else:
            channel_salience = [salience_pair[1], salience_pair[0]]  # ch0=lower, ch1=higher

        # --- Determine target cue identity based on correct channel ---
        cue_identity = "left" if correct_channel == 0 else "right"

        # --- Open trial ---
        trial_log = logger.open_trial(
            trial_id=trial_id,
            seed=trial_seed,
            task_type="two_choice",
            cue_identity=cue_identity,
            cue_onset_time=config.target_onset_ms / 1000.0,
        ) if logger else TrialLog(
            trial_id=trial_id,
            seed=trial_seed,
            task_type="two_choice",
            cue_identity=cue_identity,
            cue_onset_time=config.target_onset_ms / 1000.0,
        )

        state = TwoChoiceState(
            trial_log=trial_log,
            conflict_level=conflict_level,
            channel_salience=channel_salience,
            correct_channel=correct_channel,
        )

        # --- Emit trial_start ---
        _record_event(trial_log, logger, EventType.trial_start, sim_time_ms=0)

        # --- Emit fixation_on ---
        _record_event(
            trial_log,
            logger,
            EventType.fixation_on,
            sim_time_ms=config.fixation_duration_ms,
        )

        # --- Emit target_on_left and/or target_on_right ---
        # In two-choice, both targets appear (each at different salience levels).
        # Emit both target events at target_onset_ms.
        _record_event(
            trial_log,
            logger,
            EventType.target_on_left,
            sim_time_ms=config.target_onset_ms,
        )
        _record_event(
            trial_log,
            logger,
            EventType.target_on_right,
            sim_time_ms=config.target_onset_ms,
        )

        # --- Call policy at decision point ---
        action_evidence = ActionEvidence(
            sim_time=config.decision_point_ms / 1000.0,
            trial_id=trial_id,
            n_channels=2,
            channel_salience=channel_salience,
        )

        decision = policy(trial_log, action_evidence)
        state.decision = decision

        # --- Record decision_commit ---
        _record_event(
            trial_log,
            logger,
            EventType.decision_commit,
            sim_time_ms=config.decision_point_ms,
        )

        # --- Classify response ---
        responded = decision.selected_channel >= 0
        state.responded = responded

        # --- Emit movement_onset if response was made ---
        if responded:
            movement_onset_ms = config.decision_point_ms
            _record_event(
                trial_log,
                logger,
                EventType.movement_onset,
                sim_time_ms=movement_onset_ms,
            )
            trial_log.movement_onset_time = movement_onset_ms / 1000.0

        # --- Classify trial outcome ---
        _classify_trial(config, state)

        # --- Emit movement_end only if response was made, and trial_end ---
        if responded:
            movement_end_ms = config.decision_point_ms + 100  # Dummy: 100 ms after decision
            _record_event(trial_log, logger, EventType.movement_end, sim_time_ms=movement_end_ms)
            trial_end_ms = movement_end_ms + 50  # Dummy: 50 ms after movement_end
        else:
            # No response: movement_end not emitted; trial_end follows decision_commit
            trial_end_ms = config.decision_point_ms + 50  # Dummy: 50 ms after decision

        _record_event(trial_log, logger, EventType.trial_end, sim_time_ms=trial_end_ms)

        # --- Save trial ---
        if logger:
            logger.save_trial(trial_log)

        trials.append(trial_log)

    return trials


# --- Helper Functions ---


def _record_event(
    trial_log: TrialLog,
    logger: TrialLogger | None,
    event_type: EventType,
    sim_time_ms: int,
) -> None:
    """Record a TaskEvent in the trial log.

    Directly appends to trial_log.events if logger is None; otherwise uses logger.
    """
    sim_time = sim_time_ms / 1000.0
    real_time = sim_time  # Phase 1: logical time equals real time

    if logger:
        logger.record_event(trial_log, event_type, sim_time, real_time)
    else:
        event = TaskEvent(
            event_type=event_type,
            sim_time=sim_time,
            real_time=real_time,
            trial_id=trial_log.trial_id,
            payload={},
        )
        trial_log.events.append(event)


def _classify_trial(config: TwoChoiceConfig, state: TwoChoiceState) -> None:
    """Classify trial outcome: correct / wrong_target / timeout.

    Sets state.trial_log.success and state.trial_log.failure_mode.
    """
    # Trigger: policy decision must be checked against target salience and response window.
    # Why: two-choice classification is the core metric of the task; incorrect
    #      classification breaks downstream metrics (wrong_target_rate, success_rate).
    # Outcome: trial_log.success and failure_mode are set deterministically.

    trial_log = state.trial_log
    responded = state.responded
    correct_channel = state.correct_channel
    selected_channel = state.decision.selected_channel if state.decision else -1

    # --- Response window is relative to target onset (in ms) ---
    response_window_start_ms = config.response_window_start_ms
    response_window_end_ms = (
        config.response_window_start_ms + config.response_window_duration_ms
    )

    # --- Decision point is relative to target onset (in ms) ---
    decision_time_relative_to_target_ms = config.decision_point_ms

    # --- Check if decision was within response window ---
    within_response_window = (
        response_window_start_ms <= decision_time_relative_to_target_ms < response_window_end_ms
    )

    # --- Classify outcome ---
    if not responded:
        # No action selected: timeout
        trial_log.success = False
        trial_log.failure_mode = "timeout"
    elif not within_response_window:
        # Decision outside window: always treated as timeout (no valid response)
        trial_log.success = False
        trial_log.failure_mode = "timeout"
    elif selected_channel == correct_channel:
        # Selected the correct (higher salience) target: success
        trial_log.success = True
        trial_log.failure_mode = None
    else:
        # Selected the wrong (lower salience) target: wrong_target error
        trial_log.success = False
        trial_log.failure_mode = "wrong_target"
