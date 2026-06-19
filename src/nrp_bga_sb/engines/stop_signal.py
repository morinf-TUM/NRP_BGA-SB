"""Stop-signal task engine: variable SSD, staircase support, Verbruggen et al. 2019 validity.

The stop-signal task probes action cancellation. On go trials the agent must respond
within the response window. On stop trials a stop signal is delivered at a configurable
delay (SSD) after the go cue; the agent must inhibit its response.

A trial is successful when:
- Go trial + action selected within response window → success
- Stop trial + no action (channel -1) at decision point → success (inhibit)

A trial fails when:
- Go trial + no action selected within response window → miss
- Stop trial + action selected (channel ≥ 0) at decision point → stop_failure

Staircase algorithm (Verbruggen et al. 2019):
- After stop-success (inhibit): SSD increases by ssd_step_ms (harder — delay is longer,
  less time to cancel).
- After stop-failure (responded): SSD decreases by ssd_step_ms (easier — stop signal
  arrives earlier).
- This converges to ~50% inhibition probability as required by the independent race model.

SSD reference frame:
- SSD is an integer ms offset from go_cue onset.
- stop_signal is emitted at sim_time = go_cue_ms + SSD (in ms from trial start).

decision_point_ms reference frame:
- decision_point_ms is an integer ms offset from go_cue onset.
- The policy is called at sim_time = go_cue_ms + decision_point_ms.
- If SSD < decision_point_ms: stop signal arrived before the decision point →
  inhibition is possible (policy can see the stop_signal event in trial_log.events).
- If SSD >= decision_point_ms: stop signal arrives at or after the decision point →
  inhibition is effectively impossible.

Validity data for Verbruggen et al. 2019 (logged, not computed here):
- movement_onset_time is set to the sim_time of movement_onset for all responding trials.
- cue_onset_time records go_cue onset time.
- Together these allow downstream computation of:
    RT_go = movement_onset_time - cue_onset_time (go trials)
    RT_failed_stop = movement_onset_time - cue_onset_time (stop-failure trials)
  The validity assumption is that RT_failed_stop < RT_go on average.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, EventType, TaskEvent, TrialLog

# --- Configuration ---


@dataclass
class StopSignalConfig:
    """Configuration for the stop-signal engine.

    Attributes:
        n_trials: total number of trials to run.
        stop_proportion: fraction of trials labeled as stop (typically 0.25).
            Must be in [0.0, 1.0].
        initial_ssd_ms: starting SSD value in ms (staircase or fixed).
        ssd_step_ms: staircase step size in ms; SSD moves by this amount after each stop trial.
        ssd_min_ms: minimum allowable SSD in ms; hard lower bound for staircase.
        ssd_max_ms: maximum allowable SSD in ms; hard upper bound for staircase.
        use_staircase: if True, update SSD after each stop trial via the tracking staircase.
            If False, SSD stays at initial_ssd_ms for every stop trial (fixed-SSD mode).
        go_cue_onset_ms: sim_time (ms from trial start) at which go_cue is emitted.
            Also used as cue_onset_time recorded in TrialLog.
        decision_point_ms: offset from go_cue onset (ms) at which the policy is called.
            Reference frame: 0 = go_cue onset. Document this clearly: the engine adds
            go_cue_onset_ms to this offset to obtain the absolute sim_time.
        response_window_duration_ms: duration of the response window (ms), starting at
            go_cue onset. A response is only valid within [0, response_window_duration_ms).
        fixation_duration_ms: time from trial start to fixation signal (ms).
        seed: master random seed for the session; per-trial seeds are derived from it.
    """
    n_trials: int
    stop_proportion: float = 0.25
    initial_ssd_ms: int = 200
    ssd_step_ms: int = 50
    ssd_min_ms: int = 50
    ssd_max_ms: int = 600
    use_staircase: bool = True
    go_cue_onset_ms: int = 300          # go_cue emitted 300 ms after trial start
    decision_point_ms: int = 500        # policy called 500 ms after go_cue onset
    response_window_duration_ms: int = 700
    fixation_duration_ms: int = 200
    seed: int = 42

    def __post_init__(self) -> None:
        # Trigger: stop_proportion outside [0.0, 1.0] is a nonsensical configuration.
        # Why: a proportion must be a probability; values outside this range would
        #      silently corrupt trial-type assignment and staircase convergence.
        # Outcome: raises ValueError immediately at construction time (fail fast).
        if not (0.0 <= self.stop_proportion <= 1.0):
            raise ValueError(
                f"stop_proportion must be in [0.0, 1.0], got {self.stop_proportion}"
            )


# --- Staircase State ---


@dataclass
class StaircaseState:
    """Mutable staircase state that persists across stop trials.

    Attributes:
        current_ssd_ms: current SSD value in ms; updated after each stop trial.
    """
    current_ssd_ms: int

    def update_after_stop_success(self, step_ms: int, ssd_max_ms: int) -> None:
        """Increase SSD after inhibition success — makes next stop trial harder.

        Trigger: agent successfully inhibited on a stop trial.
        Why: tracking staircase targets ~50% inhibition; success means SSD was
             easy enough to allow inhibition, so we push it higher (later stop signal).
        Outcome: current_ssd_ms clamped to ssd_max_ms.
        """
        self.current_ssd_ms = min(self.current_ssd_ms + step_ms, ssd_max_ms)

    def update_after_stop_failure(self, step_ms: int, ssd_min_ms: int) -> None:
        """Decrease SSD after inhibition failure — makes next stop trial easier.

        Trigger: agent responded on a stop trial (failed to inhibit).
        Why: failure means SSD was too long for successful inhibition, so we pull
             it back (earlier stop signal gives more time to cancel).
        Outcome: current_ssd_ms clamped to ssd_min_ms.
        """
        self.current_ssd_ms = max(self.current_ssd_ms - step_ms, ssd_min_ms)


# --- Per-trial state ---


@dataclass
class StopSignalTrialState:
    """Mutable state for a single trial being executed.

    Attributes:
        trial_log: the TrialLog being assembled.
        is_stop_trial: True if this is a stop trial.
        ssd_ms: SSD value for this trial (set at trial start from staircase state).
        decision: the BGDecision returned by the policy.
        responded: True if the agent selected an action (channel >= 0).
    """
    trial_log: TrialLog
    is_stop_trial: bool
    ssd_ms: int
    decision: BGDecision | None = None
    responded: bool = False


# --- Main Engine ---


def run_stop_signal_trials(
    config: StopSignalConfig,
    policy: Callable[[TrialLog, ActionEvidence], BGDecision],
    logger: TrialLogger | None = None,
) -> list[TrialLog]:
    """Run a sequence of stop-signal trials, returning completed TrialLog objects.

    Implements the Verbruggen et al. 2019 consensus stop-signal paradigm:
    - Configurable stop_proportion of trials are stop trials.
    - SSD is updated via a 1-up/1-down tracking staircase (if use_staircase=True).
    - Stop signal is emitted at go_cue_onset_ms + SSD.
    - Policy is called at go_cue_onset_ms + decision_point_ms; if the stop signal
      arrived before the decision point, the policy can detect it via trial_log.events.
    - movement_onset_time is set for all responding trials, enabling downstream
      computation of RT_go and RT_failed_stop for validity verification.

    Args:
        config: StopSignalConfig with all timing and staircase parameters.
        policy: callable(trial_log, action_evidence) -> BGDecision.
                On stop trials where SSD < decision_point_ms, the trial_log.events
                will contain a stop_signal event before the decision point — the
                policy should use this to attempt inhibition.
                BGDecision.selected_channel == -1 means inhibition (no action).
        logger: optional TrialLogger for JSONL persistence. If None, trials are
                returned in memory only.

    Returns:
        list of completed TrialLog objects, one per trial.
    """
    trials: list[TrialLog] = []
    rng = random.Random(config.seed)

    # Staircase state is shared and mutated across all stop trials in the session.
    staircase = StaircaseState(current_ssd_ms=config.initial_ssd_ms)

    for trial_idx in range(config.n_trials):
        trial_id = trial_idx + 1
        trial_seed = rng.randint(0, 2**31 - 1)
        trial_rng = random.Random(trial_seed)

        # --- Determine trial type ---
        # Trigger: stop_proportion determines how often a stop signal will appear.
        # Why: pseudo-random interleaving based on trial seed ensures reproducibility
        #      while maintaining the target stop frequency over large trial counts.
        # Outcome: is_stop_trial governs the entire trial flow below.
        is_stop_trial = trial_rng.random() < config.stop_proportion
        cue_identity = "stop" if is_stop_trial else "go"

        # --- Snapshot SSD for this trial before any staircase update ---
        # The staircase state reflects the current SSD; it will only change after
        # this trial completes (if it is a stop trial).
        ssd_ms = staircase.current_ssd_ms if config.use_staircase else config.initial_ssd_ms

        # --- Open trial ---
        trial_log = logger.open_trial(
            trial_id=trial_id,
            seed=trial_seed,
            task_type="stop_signal",
            cue_identity=cue_identity,
            cue_onset_time=config.go_cue_onset_ms / 1000.0,
        ) if logger else TrialLog(
            trial_id=trial_id,
            seed=trial_seed,
            task_type="stop_signal",
            cue_identity=cue_identity,
            cue_onset_time=config.go_cue_onset_ms / 1000.0,
        )

        state = StopSignalTrialState(
            trial_log=trial_log,
            is_stop_trial=is_stop_trial,
            ssd_ms=ssd_ms,
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

        # --- Emit go_cue ---
        # The go_cue signals the agent to begin preparing a response.
        # All subsequent offsets (SSD, decision_point) are measured from this event.
        _record_event(trial_log, logger, EventType.go_cue, sim_time_ms=config.go_cue_onset_ms)

        # --- Emit stop_signal on stop trials, only when it arrives before the decision ---
        # Trigger: this is a stop trial AND ssd_ms < decision_point_ms.
        # Why: if the stop signal arrives at or after the decision point (ssd_ms >=
        #      decision_point_ms) it cannot influence the agent's response. Emitting it
        #      would create a stop_signal event whose sim_time is >= decision_commit's
        #      sim_time, breaking the sim_time ordering invariant for log replays that
        #      sort events by sim_time. Omitting it preserves the invariant cleanly.
        # Outcome: stop_signal event only appears in trial_log.events when ssd_ms <
        #          decision_point_ms. Late-stop trials (ssd_ms >= decision_point_ms) have
        #          no stop_signal event but are still classified as stop_failure (the agent
        #          responded), and the staircase still updates accordingly.
        stop_signal_ms: int | None = None
        if is_stop_trial and ssd_ms < config.decision_point_ms:
            stop_signal_ms = config.go_cue_onset_ms + ssd_ms
            _record_event(
                trial_log,
                logger,
                EventType.stop_signal,
                sim_time_ms=stop_signal_ms,
                payload={"ssd_ms": ssd_ms},
            )

        # --- Call policy at decision point ---
        # decision_point_ms is an offset from go_cue onset; absolute sim_time follows.
        decision_abs_ms = config.go_cue_onset_ms + config.decision_point_ms

        # stop_signal_present is True when a stop signal arrived before the decision.
        # This is the key flag for policies that implement an internal stopping process.
        # It mirrors the stop_signal event already in trial_log.events, but surfaces it
        # explicitly in ActionEvidence so the policy does not need to scan the event list.
        stop_signal_present = (
            is_stop_trial and ssd_ms < config.decision_point_ms
        )

        action_evidence = ActionEvidence(
            sim_time=decision_abs_ms / 1000.0,
            trial_id=trial_id,
            n_channels=2,
            channel_salience=[0.5, 0.5],  # neutral evidence; Phase 1
            stop_signal_present=stop_signal_present,
        )

        decision = policy(trial_log, action_evidence)
        state.decision = decision

        # --- Record decision_commit ---
        _record_event(
            trial_log,
            logger,
            EventType.decision_commit,
            sim_time_ms=decision_abs_ms,
        )

        # --- Classify response ---
        responded = decision.selected_channel >= 0
        state.responded = responded

        # --- Emit movement_onset if response was made ---
        # movement_onset_time is set so downstream code can compute:
        #   RT = movement_onset_time - cue_onset_time
        # This is needed for both go-RT and failed-stop-RT (Verbruggen 2019 validity).
        #
        # Phase 1 limitation: all movement onsets are pinned to decision_point_ms; RT
        # variance appears only in Phase 5+ when policies produce variable decision
        # timing. The Verbruggen failed-stop RT < go RT check is a structural tautology
        # here (both RTs are identical) and will become meaningful once real policies
        # have RT variance.
        if responded:
            movement_onset_ms = decision_abs_ms
            _record_event(
                trial_log,
                logger,
                EventType.movement_onset,
                sim_time_ms=movement_onset_ms,
            )
            # cue_onset_time = go_cue_onset_ms / 1000.0 (set at open_trial above)
            trial_log.movement_onset_time = movement_onset_ms / 1000.0

        # --- Classify trial outcome ---
        _classify_trial(config, state)

        # --- Update staircase after stop trial ---
        # Trigger: this was a stop trial; staircase must be updated before next trial.
        # Why: updating immediately after classification ensures the next trial's SSD
        #      snapshot reflects the outcome of this trial.
        # Outcome: staircase.current_ssd_ms moves up (inhibit) or down (fail) by step.
        if is_stop_trial and config.use_staircase:
            if state.trial_log.success:
                staircase.update_after_stop_success(config.ssd_step_ms, config.ssd_max_ms)
            else:
                staircase.update_after_stop_failure(config.ssd_step_ms, config.ssd_min_ms)

        # --- Emit stop_signal after decision point if it arrives late ---
        # Already emitted above for all stop trials (before or at decision point).
        # No additional emission needed here — the event was inserted at its true sim_time.

        # --- Emit movement_end and trial_end ---
        if responded:
            movement_end_ms = decision_abs_ms + 100   # dummy: 100 ms after decision
            _record_event(trial_log, logger, EventType.movement_end, sim_time_ms=movement_end_ms)
            trial_end_ms = movement_end_ms + 50        # dummy: 50 ms after movement_end
        else:
            # No response: movement_end not emitted; trial_end follows decision_commit
            trial_end_ms = decision_abs_ms + 50        # dummy: 50 ms after decision

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
    payload: dict | None = None,
) -> None:
    """Record a TaskEvent in the trial log.

    Directly appends to trial_log.events if logger is None; otherwise uses logger.
    payload is merged into the event payload dict (may carry ssd_ms for stop_signal).
    """
    sim_time = sim_time_ms / 1000.0
    real_time = sim_time  # Phase 1: logical time equals real time

    if logger:
        logger.record_event(trial_log, event_type, sim_time, real_time, payload or {})
    else:
        event = TaskEvent(
            event_type=event_type,
            sim_time=sim_time,
            real_time=real_time,
            trial_id=trial_log.trial_id,
            payload=payload or {},
        )
        trial_log.events.append(event)


def _classify_trial(config: StopSignalConfig, state: StopSignalTrialState) -> None:
    """Classify trial outcome and set trial_log.success / failure_mode.

    Go trials:
    - responded within response window → success
    - no response → miss

    Stop trials:
    - no response (inhibited) → success (stop_success)
    - responded → stop_failure

    Note: on stop trials, there is no "wrong action" — any response constitutes
    stop_failure regardless of which channel was selected.
    """
    # Trigger: policy decision must be checked against trial type and response window.
    # Why: stop-signal classification determines staircase updates and validity metrics;
    #      incorrect classification would corrupt the staircase and downstream SSRT.
    # Outcome: trial_log.success and failure_mode set deterministically.

    trial_log = state.trial_log
    is_stop_trial = state.is_stop_trial
    responded = state.responded

    # Response window: [0, response_window_duration_ms) from go_cue onset.
    # decision_point_ms is already relative to go_cue onset.
    within_response_window = (
        0 <= config.decision_point_ms < config.response_window_duration_ms
    )

    if is_stop_trial:
        if responded:
            # Responded despite stop signal — failed to inhibit.
            trial_log.success = False
            trial_log.failure_mode = "stop_failure"
        else:
            # No response — successfully inhibited.
            trial_log.success = True
            trial_log.failure_mode = None
    else:
        # Go trial.
        # Phase 1: policy is always called at exactly decision_point_ms, so
        # within_response_window is always True when a response exists. The else
        # branch (responded but outside window) is currently unreachable and is
        # retained as a guard for future phases where decision timing may vary.
        if responded and within_response_window:
            trial_log.success = True
            trial_log.failure_mode = None
        elif not responded:
            trial_log.success = False
            trial_log.failure_mode = "miss"
        else:
            # Responded outside response window — treated as miss.
            # Unreachable in Phase 1 (see comment above); kept for Phase 5+.
            trial_log.success = False
            trial_log.failure_mode = "miss"
