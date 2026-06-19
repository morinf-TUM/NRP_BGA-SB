"""Change-of-mind task engine: switch-cue timing, redirection scoring.

The change-of-mind task probes action switching after an initial commitment is
underway. A switch/redirection cue (evidence_change event) arrives at a
variable delay after the go cue. The agent must redirect from the original
target (channel 0) to the new target (channel 1).

On switch trials the policy is called TWICE:
1. Pre-switch call  — at initial_decision_point_ms (offset from go_cue onset).
   Reflects the original target having higher salience (initial_salience).
   Result is logged as decision_commit but does NOT determine the outcome.
2. evidence_change — emitted at switch_delay_ms (offset from go_cue onset).
3. Post-switch call — at post_switch_decision_point_ms (> switch_delay_ms).
   Reflects the new target having higher salience (post_switch_salience).
   Result of THIS call determines the trial outcome.

On no-switch (baseline) trials a single call is made at initial_decision_point_ms.
The agent selects an action just like a standard go trial.

Outcome classification (switch trials only):
- Correct switch   (success=True):                post-switch selects channel 1
- Perseveration    (failure_mode="perseveration"): post-switch selects channel 0
- Miss             (failure_mode="miss"):           post-switch returns -1

Reference frame:
- All *_ms fields that follow "offset from go_cue onset" naming are relative to
  go_cue onset. The engine adds go_cue_onset_ms to each offset to get absolute
  sim_time. This matches the convention established in stop_signal.py.

Switch delay categories:
- Parameterized as a dict[str, int] (category_name → ms from go_cue onset).
- Categories cycle deterministically over switch trials (no random sampling)
  so each switch trial belongs to exactly one category in round-robin order.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from nrp_bga_sb.logger import TrialLogger
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, EventType, TaskEvent, TrialLog

# --- Configuration ---


@dataclass
class ChangeOfMindConfig:
    """Configuration for the change-of-mind engine.

    Attributes:
        n_trials: total number of trials.
        no_switch_proportion: fraction that are baseline go trials (no switch cue).
            Must be in [0.0, 1.0].
        switch_delay_categories: dict mapping category name → switch delay in ms
            (offset from go_cue onset). E.g. {"early": 50, "medium": 150,
            "late": 300, "very_late": 450}. Cycles over switch trials in
            insertion order.
        initial_decision_point_ms: offset from go_cue onset (ms) at which the
            pre-switch policy call is made. Must be strictly less than the
            minimum switch delay across all categories.
        post_switch_decision_point_ms: offset from go_cue onset (ms) at which
            the post-switch policy call is made. Must be strictly greater than
            the maximum switch delay across all categories.
        response_window_duration_ms: total duration of the response window in
            ms, measured from go_cue onset.
        go_cue_onset_ms: absolute sim_time (ms from trial start) at which the
            go_cue is emitted. Also recorded as cue_onset_time in TrialLog.
        fixation_duration_ms: time from trial start to fixation_on (ms).
        initial_salience: per-channel salience before the switch cue.
            Default [0.8, 0.2] — channel 0 (original target) is more salient.
        post_switch_salience: per-channel salience after the switch cue.
            Default [0.2, 0.8] — channel 1 (new target) is more salient.
        seed: master random seed; per-trial seeds derived from it.
    """
    n_trials: int
    no_switch_proportion: float = 0.25
    switch_delay_categories: dict[str, int] = field(
        default_factory=lambda: {
            "early": 50,
            "medium": 150,
            "late": 300,
            "very_late": 450,
        }
    )
    initial_decision_point_ms: int = 20   # < earliest switch delay (50 ms)
    post_switch_decision_point_ms: int = 550  # > latest switch delay (450 ms)
    response_window_duration_ms: int = 700
    go_cue_onset_ms: int = 300
    fixation_duration_ms: int = 200
    initial_salience: list[float] = field(default_factory=lambda: [0.8, 0.2])
    post_switch_salience: list[float] = field(default_factory=lambda: [0.2, 0.8])
    seed: int = 42

    def __post_init__(self) -> None:
        # Trigger: no_switch_proportion outside [0.0, 1.0].
        # Why: a proportion is a probability; values outside this range would silently
        #      corrupt trial-type assignment.
        # Outcome: raises ValueError at construction time (fail fast).
        if not (0.0 <= self.no_switch_proportion <= 1.0):
            raise ValueError(
                f"no_switch_proportion must be in [0.0, 1.0], "
                f"got {self.no_switch_proportion}"
            )

        if not self.switch_delay_categories:
            raise ValueError("switch_delay_categories must not be empty")

        min_delay = min(self.switch_delay_categories.values())
        max_delay = max(self.switch_delay_categories.values())

        # Trigger: initial_decision_point_ms >= min switch delay.
        # Why: the pre-switch call must precede every possible switch cue so the
        #      evidence_change event always falls between the two policy calls.
        # Outcome: raises ValueError at construction time.
        if self.initial_decision_point_ms >= min_delay:
            raise ValueError(
                f"initial_decision_point_ms ({self.initial_decision_point_ms}) "
                f"must be < minimum switch delay ({min_delay})"
            )

        # Trigger: post_switch_decision_point_ms <= max switch delay.
        # Why: the post-switch call must follow every possible switch cue so the
        #      policy always has the opportunity to react to the evidence change.
        # Outcome: raises ValueError at construction time.
        if self.post_switch_decision_point_ms <= max_delay:
            raise ValueError(
                f"post_switch_decision_point_ms ({self.post_switch_decision_point_ms}) "
                f"must be > maximum switch delay ({max_delay})"
            )


# --- Per-trial state ---


@dataclass
class ChangeOfMindTrialState:
    """Mutable state for a single change-of-mind trial.

    Attributes:
        trial_log: the TrialLog being assembled.
        is_switch_trial: True if an evidence_change event will occur.
        switch_delay_ms: offset from go_cue onset at which evidence_change fires
            (None on no-switch trials).
        switch_category: human-readable delay category name (None on no-switch trials).
        pre_switch_decision: result of the pre-switch policy call (logged but not
            used for outcome classification).
        post_switch_decision: result of the post-switch policy call; determines
            outcome on switch trials.
        responded: True if post-switch (or single on no-switch) decision selected
            a non-negative channel.
    """
    trial_log: TrialLog
    is_switch_trial: bool
    switch_delay_ms: int | None
    switch_category: str | None
    pre_switch_decision: BGDecision | None = None
    post_switch_decision: BGDecision | None = None
    responded: bool = False


# --- Main Engine ---


def run_change_of_mind_trials(
    config: ChangeOfMindConfig,
    policy: Callable[[TrialLog, ActionEvidence], BGDecision],
    logger: TrialLogger | None = None,
) -> list[TrialLog]:
    """Run a sequence of change-of-mind trials, returning completed TrialLog objects.

    On switch trials the policy is called twice: before the switch cue (pre-switch)
    and after (post-switch). Only the post-switch decision determines the outcome.
    Switch delay categories cycle in insertion order over switch trials so each
    category appears equally often given a trial count that is a multiple of the
    category count.

    Args:
        config: ChangeOfMindConfig with all timing and salience parameters.
        policy: callable(trial_log, action_evidence) -> BGDecision.
            On switch trials the policy will receive two separate calls with
            different ActionEvidence (different channel_salience values).
            An oracle policy can inspect channel_salience to decide correctly.
        logger: optional TrialLogger for JSONL persistence. If None, trials are
            returned in memory only.

    Returns:
        list of completed TrialLog objects, one per trial.
    """
    trials: list[TrialLog] = []
    rng = random.Random(config.seed)

    # Pre-compute the ordered list of (category_name, delay_ms) pairs for cycling.
    # Insertion order is preserved in Python 3.7+ dicts; this makes cycling deterministic.
    category_pairs = list(config.switch_delay_categories.items())
    switch_trial_counter = 0  # counts switch trials seen so far for cycling

    for trial_idx in range(config.n_trials):
        trial_id = trial_idx + 1
        trial_seed = rng.randint(0, 2**31 - 1)
        trial_rng = random.Random(trial_seed)

        # --- Determine trial type ---
        # Trigger: no_switch_proportion controls how often a baseline (no-switch) trial appears.
        # Why: pseudo-random assignment per trial seed ensures reproducibility while
        #      maintaining the target proportion over large trial counts.
        # Outcome: is_switch_trial governs the entire trial flow below.
        is_switch_trial = trial_rng.random() >= config.no_switch_proportion
        cue_identity: str

        if is_switch_trial:
            # --- Assign switch delay category by cycling ---
            # Trigger: switch trial requires a delay category selection.
            # Why: round-robin cycling (not random) ensures uniform coverage of all
            #      delay categories, which is essential for statistical balance across
            #      the early/medium/late/very-late conditions.
            # Outcome: switch_category and switch_delay_ms set for this trial.
            cat_name, switch_delay_ms = category_pairs[
                switch_trial_counter % len(category_pairs)
            ]
            switch_trial_counter += 1
            cue_identity = f"switch_{cat_name}"
            state = ChangeOfMindTrialState(
                trial_log=_open_trial(
                    logger, trial_id, trial_seed, "change_of_mind", cue_identity,
                    config.go_cue_onset_ms / 1000.0,
                ),
                is_switch_trial=True,
                switch_delay_ms=switch_delay_ms,
                switch_category=cat_name,
            )
        else:
            cue_identity = "no_switch"
            state = ChangeOfMindTrialState(
                trial_log=_open_trial(
                    logger, trial_id, trial_seed, "change_of_mind", cue_identity,
                    config.go_cue_onset_ms / 1000.0,
                ),
                is_switch_trial=False,
                switch_delay_ms=None,
                switch_category=None,
            )

        # --- Emit trial_start ---
        _record_event(state.trial_log, logger, EventType.trial_start, sim_time_ms=0)

        # --- Emit fixation_on ---
        _record_event(
            state.trial_log, logger, EventType.fixation_on,
            sim_time_ms=config.fixation_duration_ms,
        )

        # --- Emit go_cue and initial target ---
        # go_cue onset is the reference point for all subsequent offsets.
        _record_event(
            state.trial_log, logger, EventType.go_cue,
            sim_time_ms=config.go_cue_onset_ms,
        )
        # Channel 0 is the initial target; emit target_on_left to represent it.
        # This event records the appearance of the original target at go_cue onset.
        _record_event(
            state.trial_log, logger, EventType.target_on_left,
            sim_time_ms=config.go_cue_onset_ms,
        )

        # --- Pre-switch policy call ---
        # Both switch and no-switch trials have a pre-switch call at
        # initial_decision_point_ms. On no-switch trials this is the only call.
        pre_switch_abs_ms = config.go_cue_onset_ms + config.initial_decision_point_ms
        pre_switch_evidence = ActionEvidence(
            sim_time=pre_switch_abs_ms / 1000.0,
            trial_id=trial_id,
            n_channels=2,
            channel_salience=list(config.initial_salience),
            stop_signal_present=False,
        )
        pre_switch_decision = policy(state.trial_log, pre_switch_evidence)
        state.pre_switch_decision = pre_switch_decision

        # Record pre-switch intent as decision_commit.
        # This logs what the agent intended before any switch cue arrives.
        # It is NOT the outcome-determining decision on switch trials.
        _record_event(
            state.trial_log, logger, EventType.decision_commit,
            sim_time_ms=pre_switch_abs_ms,
            payload={"phase": "pre_switch"},
        )

        if is_switch_trial:
            # --- Emit evidence_change (switch cue) ---
            # Trigger: switch trial requires an evidence_change event at switch_delay_ms.
            # Why: evidence_change is the signal to the agent that the target has
            #      changed; its sim_time must fall strictly between the two policy calls.
            # Outcome: evidence_change event logged; post-switch call follows with
            #          updated salience.
            assert state.switch_delay_ms is not None  # guaranteed by is_switch_trial branch
            switch_abs_ms = config.go_cue_onset_ms + state.switch_delay_ms
            _record_event(
                state.trial_log, logger, EventType.evidence_change,
                sim_time_ms=switch_abs_ms,
                payload={
                    "switch_category": state.switch_category,
                    "switch_delay_ms": state.switch_delay_ms,
                },
            )
            # Emit target_on_right to mark the new target appearing after the switch cue.
            _record_event(
                state.trial_log, logger, EventType.target_on_right,
                sim_time_ms=switch_abs_ms,
            )

            # --- Post-switch policy call ---
            # This call receives updated salience where channel 1 is now the dominant
            # target. Its result determines the trial outcome.
            post_switch_abs_ms = (
                config.go_cue_onset_ms + config.post_switch_decision_point_ms
            )
            post_switch_evidence = ActionEvidence(
                sim_time=post_switch_abs_ms / 1000.0,
                trial_id=trial_id,
                n_channels=2,
                channel_salience=list(config.post_switch_salience),
                stop_signal_present=False,
            )
            post_switch_decision = policy(state.trial_log, post_switch_evidence)
            state.post_switch_decision = post_switch_decision
            state.responded = post_switch_decision.selected_channel >= 0

            # Record post-switch commit at post_switch decision point.
            _record_event(
                state.trial_log, logger, EventType.decision_commit,
                sim_time_ms=post_switch_abs_ms,
                payload={"phase": "post_switch"},
            )

            # Emit movement_onset only if the agent responded after the switch.
            if state.responded:
                _record_event(
                    state.trial_log, logger, EventType.movement_onset,
                    sim_time_ms=post_switch_abs_ms,
                )
                state.trial_log.movement_onset_time = post_switch_abs_ms / 1000.0

        else:
            # --- No-switch trial: pre-switch decision is the outcome decision ---
            # The single policy call already happened; use its result for outcome.
            state.post_switch_decision = pre_switch_decision
            state.responded = pre_switch_decision.selected_channel >= 0

            if state.responded:
                _record_event(
                    state.trial_log, logger, EventType.movement_onset,
                    sim_time_ms=pre_switch_abs_ms,
                )
                state.trial_log.movement_onset_time = pre_switch_abs_ms / 1000.0

        # --- Classify trial outcome ---
        _classify_trial(config, state)

        # --- Emit movement_end and trial_end ---
        # movement_end only follows if the agent responded.
        if state.responded:
            if is_switch_trial:
                response_abs_ms = config.go_cue_onset_ms + config.post_switch_decision_point_ms
            else:
                response_abs_ms = pre_switch_abs_ms
            movement_end_ms = response_abs_ms + 100  # dummy: 100 ms after response
            _record_event(
                state.trial_log, logger, EventType.movement_end,
                sim_time_ms=movement_end_ms,
            )
            trial_end_ms = movement_end_ms + 50
        else:
            # No response: movement_end not emitted.
            if is_switch_trial:
                last_event_ms = config.go_cue_onset_ms + config.post_switch_decision_point_ms
            else:
                last_event_ms = pre_switch_abs_ms
            trial_end_ms = last_event_ms + 50

        _record_event(state.trial_log, logger, EventType.trial_end, sim_time_ms=trial_end_ms)

        # --- Save trial ---
        if logger:
            logger.save_trial(state.trial_log)

        trials.append(state.trial_log)

    return trials


# --- Helper Functions ---


def _open_trial(
    logger: TrialLogger | None,
    trial_id: int,
    seed: int,
    task_type: str,
    cue_identity: str,
    cue_onset_time: float,
) -> TrialLog:
    """Open a new TrialLog via logger if present, otherwise construct directly."""
    if logger:
        return logger.open_trial(
            trial_id=trial_id,
            seed=seed,
            task_type=task_type,  # type: ignore[arg-type]
            cue_identity=cue_identity,
            cue_onset_time=cue_onset_time,
        )
    return TrialLog(
        trial_id=trial_id,
        seed=seed,
        task_type=task_type,  # type: ignore[arg-type]
        cue_identity=cue_identity,
        cue_onset_time=cue_onset_time,
    )


def _record_event(
    trial_log: TrialLog,
    logger: TrialLogger | None,
    event_type: EventType,
    sim_time_ms: int,
    payload: dict | None = None,
) -> None:
    """Record a TaskEvent in the trial log.

    Appends directly to trial_log.events if logger is None; otherwise delegates
    to logger.record_event. payload carries optional per-event metadata
    (e.g., switch_category for evidence_change events).
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


def _classify_trial(
    config: ChangeOfMindConfig,
    state: ChangeOfMindTrialState,
) -> None:
    """Classify trial outcome and set trial_log.success / failure_mode.

    Switch trials:
    - post-switch selects channel 1 (new target) → correct switch (success)
    - post-switch selects channel 0 (original target) → perseveration (failure)
    - post-switch returns -1 (no response) → miss (failure)

    No-switch trials:
    - responded (channel >= 0) → success
    - no response → miss (failure)
    """
    # Trigger: outcome depends on trial type and the post-switch channel selection.
    # Why: correct classification is required for the switch_success_rate metric and
    #      for distinguishing perseveration from miss — two distinct failure modes with
    #      different implications for BG frequency sensitivity.
    # Outcome: trial_log.success and failure_mode set deterministically.
    trial_log = state.trial_log
    decision = state.post_switch_decision
    assert decision is not None, "post_switch_decision must be set before _classify_trial"

    if state.is_switch_trial:
        channel = decision.selected_channel
        if channel == 1:
            # Correct switch: agent redirected to the new (higher-salience) target.
            trial_log.success = True
            trial_log.failure_mode = None
        elif channel == 0:
            # Perseveration: agent persisted to the original target despite the cue.
            trial_log.success = False
            trial_log.failure_mode = "perseveration"
        else:
            # Miss: no response after the switch cue.
            trial_log.success = False
            trial_log.failure_mode = "miss"
    else:
        # No-switch baseline: any response is a success.
        if decision.selected_channel >= 0:
            trial_log.success = True
            trial_log.failure_mode = None
        else:
            trial_log.success = False
            trial_log.failure_mode = "miss"
