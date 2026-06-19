"""Go/no-go task engine: cue scheduler, response window, success/failure classifier.

The go/no-go task is a canonical response-inhibition paradigm. On go trials, the
agent must select an action within the response window. On no-go trials, the agent
must withhold action (select channel -1) throughout.

A trial is successful when:
- Go trial + action selected (channel >= 0) within response window → success
- No-go trial + no action (channel = -1) at decision time → correct_withhold

A trial fails when:
- Go trial + no action (channel = -1) within response window → miss
- No-go trial + action selected (channel >= 0) at any time → false_alarm

The engine operates on a logical clock (integer ms). It uses a seeded random
generator to determine go/no-go labels (controlled by the parameter go_probability).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, EventType, TaskEvent, TrialLog

# --- Configuration ---


@dataclass
class GoNoGoConfig:
    """Configuration for the go/no-go engine.

    Attributes:
        n_trials: total number of trials to run.
        go_probability: fraction of trials labeled as go (0.5 for balanced).
        response_window_start_ms: delay from cue onset to response window opening (ms).
        response_window_duration_ms: duration the response window is open (ms).
        fixation_duration_ms: time from trial start to fixation signal (ms).
        cue_onset_ms: time from fixation to cue onset (ms).
        decision_point_ms: offset from cue onset (in ms) at which the policy is called
            and decision is made. All timing checks (e.g., response window validity)
            use cue onset as the reference point (time 0).
        seed: random seed for trial generation.
    """
    n_trials: int
    go_probability: float
    response_window_start_ms: int
    response_window_duration_ms: int
    fixation_duration_ms: int
    cue_onset_ms: int
    decision_point_ms: int
    seed: int


@dataclass
class GoNoGoState:
    """Mutable state for a single trial being executed.

    Attributes:
        trial_log: the TrialLog being assembled.
        is_go_trial: True if this is a go trial (action required).
        decision: the BGDecision returned by the policy.
        responded: True if action was selected (channel >= 0).
    """
    trial_log: TrialLog
    is_go_trial: bool
    decision: BGDecision | None = None
    responded: bool = False


# --- Main Engine ---


def run_go_nogo_trials(
    config: GoNoGoConfig,
    policy: Callable[[TrialLog, ActionEvidence], BGDecision],
    logger: TrialLogger | None = None,
) -> list[TrialLog]:
    """Run a sequence of go/no-go trials, returning completed TrialLog objects.

    Args:
        config: GoNoGoConfig with trial parameters and timing.
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

    for trial_idx in range(config.n_trials):
        trial_id = trial_idx + 1
        trial_seed = rng.randint(0, 2**31 - 1)
        trial_rng = random.Random(trial_seed)

        # --- Determine trial type ---
        is_go_trial = trial_rng.random() < config.go_probability
        cue_identity = "go" if is_go_trial else "no_go"
        cue_event_type = EventType.go_cue if is_go_trial else EventType.no_go_cue

        # --- Open trial ---
        trial_log = logger.open_trial(
            trial_id=trial_id,
            seed=trial_seed,
            task_type="go_nogo",
            cue_identity=cue_identity,
            cue_onset_time=config.cue_onset_ms / 1000.0,
        ) if logger else TrialLog(
            trial_id=trial_id,
            seed=trial_seed,
            task_type="go_nogo",
            cue_identity=cue_identity,
            cue_onset_time=config.cue_onset_ms / 1000.0,
        )

        state = GoNoGoState(trial_log=trial_log, is_go_trial=is_go_trial)

        # --- Emit trial_start ---
        _record_event(trial_log, logger, EventType.trial_start, sim_time_ms=0)

        # --- Emit fixation_on ---
        _record_event(
            trial_log,
            logger,
            EventType.fixation_on,
            sim_time_ms=config.fixation_duration_ms,
        )

        # --- Emit cue (go_cue or no_go_cue) ---
        _record_event(trial_log, logger, cue_event_type, sim_time_ms=config.cue_onset_ms)

        # --- Call policy at decision point ---
        # Construct ActionEvidence. For Phase 1, we use a minimal evidence structure.
        # The policy will use this to decide whether to select an action.
        action_evidence = ActionEvidence(
            sim_time=config.decision_point_ms / 1000.0,
            trial_id=trial_id,
            n_channels=2,  # Phase 1: minimal 2-channel system
            channel_salience=[0.5, 0.5],  # Neutral evidence for both channels
        )

        decision = policy(trial_log, action_evidence)
        state.decision = decision

        # --- Record decision ---
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
            # Emit movement_onset at decision point (Phase 1: no latency modeled).
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


def _classify_trial(config: GoNoGoConfig, state: GoNoGoState) -> None:
    """Classify trial outcome: success, miss, false_alarm, or correct_withhold.

    Sets state.trial_log.success and state.trial_log.failure_mode.
    """
    # Trigger: policy decision must be checked against trial type and response window.
    # Why: go/no-go classification is the core metric of the task; incorrect
    #      classification breaks downstream metrics (false_alarm_rate, success_rate).
    # Outcome: trial_log.success and failure_mode are set deterministically.

    trial_log = state.trial_log
    is_go_trial = state.is_go_trial
    responded = state.responded

    # --- Response window is relative to cue onset (in ms) ---
    response_window_start_ms = config.response_window_start_ms
    response_window_end_ms = (
        config.response_window_start_ms + config.response_window_duration_ms
    )

    # --- Decision point is relative to cue onset (in ms) ---
    decision_time_relative_to_cue_ms = config.decision_point_ms

    # --- Check if decision was within response window ---
    within_response_window = (
        response_window_start_ms <= decision_time_relative_to_cue_ms < response_window_end_ms
    )

    if is_go_trial:
        if responded and within_response_window:
            trial_log.success = True
            trial_log.failure_mode = None
        else:
            trial_log.success = False
            if not responded:
                trial_log.failure_mode = "miss"
            else:
                trial_log.failure_mode = "wrong_action"  # Outside window
    else:  # no_go trial
        if not responded:
            trial_log.success = True
            trial_log.failure_mode = None
        else:
            trial_log.success = False
            trial_log.failure_mode = "false_alarm"
