"""Stop-signal metrics module — Verbruggen et al. 2019 consensus methodology.

Computes all stop-signal-specific metrics from a list of TrialLog objects.
This module is pure (no engine calls); it works offline from logged trial data.

Key semantic note on RT:
  RT = movement_onset_time - cue_onset_time (both in seconds).
  In Phase 7 with ClosedLoopPolicy, movement_onset_time is set to
  (go_cue_onset_ms + int(selection_latency * 1000)) / 1000.0,
  so RT ≈ selection_latency (13–100 ms). This is a BG-internal latency
  proxy, not a reaction-time clock measured from the subject's perspective.
  Do not reinterpret it here — use it as-is.

Late-stop trials:
  When SSD >= decision_point_ms, the engine omits the stop_signal event
  (the signal arrived too late to influence the decision). These trials
  still appear as stop_failure in failure_mode. Downstream metrics must
  handle their absence from the stop_signal event list explicitly.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from nrp_bga_sb.schemas import EventType, TrialLog

# --- Trial identification ---


def _has_stop_signal_event(trial_log: TrialLog) -> bool:
    """True when trial_log.events contains an EventType.stop_signal event."""
    return any(e.event_type == EventType.stop_signal for e in trial_log.events)


def is_stop_trial(trial_log: TrialLog) -> bool:
    """Identify stop trials.

    A trial is a stop trial if ANY of:
    - cue_identity == "stop"
    - failure_mode == "stop_failure"
    - has a stop_signal event in events

    Trigger: three criteria are needed because stop trials can appear in two
    engine modes:
      (a) stop_trial_go_evidence=False → cue_identity="stop" marks the trial
      (b) stop_trial_go_evidence=True → cue_identity="go", but stop_signal event
          is present for early-stop trials and failure_mode="stop_failure" for
          both early and late-stop failures.
    Why: using only cue_identity would miss all stop-trial-go-evidence trials;
         using only failure_mode would miss successful early-stop trials whose
         cue_identity is "go"; using only the event would miss late-stop failures
         (no stop_signal event logged when SSD >= decision_point_ms).
    Outcome: all three modes of stop trial are captured regardless of engine
             configuration.
    """
    return (
        trial_log.cue_identity == "stop"
        or trial_log.failure_mode == "stop_failure"
        or _has_stop_signal_event(trial_log)
    )


def is_go_trial(trial_log: TrialLog) -> bool:
    return not is_stop_trial(trial_log)


# --- RT and latency helpers ---


def extract_ssd_ms(trial_log: TrialLog) -> int | None:
    """Extract SSD (ms from go_cue onset) from the stop_signal event payload.

    Returns None for late-stop trials (SSD >= decision_point_ms: stop signal not
    logged because it arrived after the decision point — see engine docs).
    """
    for event in trial_log.events:
        if event.event_type == EventType.stop_signal:
            return int(event.payload["ssd_ms"])
    return None


def _rt_s(trial_log: TrialLog) -> float | None:
    """RT in seconds: movement_onset_time - cue_onset_time.

    Returns None if movement_onset_time is None (no response).
    Negative values are clamped to 0.0 (should not occur with correct engine data).
    """
    if trial_log.movement_onset_time is None:
        return None
    raw = trial_log.movement_onset_time - trial_log.cue_onset_time
    # Trigger: raw RT is negative — movement registered before go cue.
    # Why: the engine should never produce this, but clamping prevents downstream
    #      metric corruption (e.g., negative mean RT) from any upstream data error.
    # Outcome: returns 0.0 instead of a negative value; caller is not notified
    #          (fail fast only on input contract violations, not data quirks).
    return max(0.0, raw)


# --- Core metric functions ---


def go_rt_stats(trials: list[TrialLog]) -> dict[str, float | None]:
    """Mean and std of RT for responding go trials.

    Returns {"mean_s": float | None, "std_s": float | None, "n": int}.
    None when fewer than 2 responding go trials.
    """
    rts = [
        rt
        for t in trials
        if is_go_trial(t)
        for rt in (_rt_s(t),)
        if rt is not None
    ]
    n = len(rts)
    if n < 2:
        return {"mean_s": None, "std_s": None, "n": n}
    mean = sum(rts) / n
    variance = sum((r - mean) ** 2 for r in rts) / n
    return {"mean_s": mean, "std_s": math.sqrt(variance), "n": n}


def failed_stop_rt_mean(trials: list[TrialLog]) -> float | None:
    """Mean RT for stop-failure trials. None if no stop failures."""
    rts = [
        rt
        for t in trials
        if is_stop_trial(t) and t.failure_mode == "stop_failure"
        for rt in (_rt_s(t),)
        if rt is not None
    ]
    if not rts:
        return None
    return sum(rts) / len(rts)


# --- Inhibition function ---


def inhibition_function(trials: list[TrialLog]) -> dict[int, dict]:
    """Stop failure rate at each observed SSD.

    Returns {ssd_ms: {"failure_rate": float, "n": int}} for all SSDs
    where at least 1 stop trial occurred AND ssd_ms is not None.

    Trigger: late-stop trials (ssd_ms=None) are excluded from the SSD bins.
    Why: their SSD is unknown (the stop_signal event was not logged because the
         signal arrived after the decision point). Assigning them to any SSD bin
         would corrupt failure rates at that SSD. They are conceptually different
         from inhibition failures — they are trigger failures where inhibition was
         mechanically impossible. Counting them under trigger_failure_rate instead
         keeps the inhibition function interpretable.
    Outcome: dict keys are only SSDs where a meaningful stop_signal event existed.
             Sorted ascending by SSD.
    """
    # ssd_ms → [n_total, n_failures]
    bins: dict[int, list[int]] = {}
    for t in trials:
        if not is_stop_trial(t):
            continue
        ssd = extract_ssd_ms(t)
        if ssd is None:
            # Late-stop: no SSD logged; excluded from inhibition function dict.
            continue
        if ssd not in bins:
            bins[ssd] = [0, 0]  # [n_total, n_failures]
        bins[ssd][0] += 1
        if t.failure_mode == "stop_failure":
            bins[ssd][1] += 1

    return {
        ssd: {
            "failure_rate": counts[1] / counts[0],
            "n": counts[0],
        }
        for ssd, counts in sorted(bins.items())
    }


# --- SSRT estimation ---


def estimate_ssrt(trials: list[TrialLog]) -> float | None:
    """SSRT estimate using the simplified mean-SSD method (Verbruggen 2019).

    Formula: SSRT = mean go RT - mean SSD across all stop trials.
    Rationale: with a 1-up/1-down staircase converging to ~50% inhibition,
    mean SSD approximates the SSD at which p(respond|stop) = 0.5.
    By the race model: SSRT = mean go RT - SSD_50.

    Returns None if fewer than 2 responding go trials OR no stop trials.

    Trigger: late-stop trials (ssd_ms=None) are excluded from the SSD mean.
    Why: their SSD is unknown (no stop_signal event logged). Including them with
         any imputed SSD value would bias the mean SSD estimate upward (they are
         late stops, so their true SSD is at or above the decision point).
         Excluding them keeps the SSD mean anchored to trials where the race-model
         assumption (SSRT < decision time) actually holds.
    Outcome: mean_ssd is computed only from stop trials with a known SSD.
             If no stop trials have a known SSD, returns None.
    """
    go_stats = go_rt_stats(trials)
    if go_stats["mean_s"] is None:
        return None

    # Collect SSDs from stop trials with known SSD (exclude late-stops).
    ssds = [
        ssd
        for t in trials
        if is_stop_trial(t)
        for ssd in (extract_ssd_ms(t),)
        if ssd is not None
    ]
    if not ssds:
        return None

    mean_ssd_s = (sum(ssds) / len(ssds)) / 1000.0
    return go_stats["mean_s"] - mean_ssd_s


def cancellation_latency_mean(trials: list[TrialLog]) -> float | None:
    """Mean time available for cancellation on successful early-stop trials (seconds).

    For each early-stop success (has stop_signal event, success=True):
      latency_s = time_of_decision_commit - time_of_stop_signal

    Extract decision_commit time from the decision_commit event in trial_log.events.
    Extract stop_signal time from the stop_signal event in trial_log.events.

    Returns None if no qualifying trials.
    """
    latencies: list[float] = []
    for t in trials:
        if not (is_stop_trial(t) and t.success is True and _has_stop_signal_event(t)):
            continue
        stop_time: float | None = None
        commit_time: float | None = None
        for event in t.events:
            if event.event_type == EventType.stop_signal:
                stop_time = event.sim_time
            elif event.event_type == EventType.decision_commit:
                commit_time = event.sim_time
        if stop_time is not None and commit_time is not None:
            latencies.append(commit_time - stop_time)

    if not latencies:
        return None
    return sum(latencies) / len(latencies)


# --- Validity metrics ---


def trigger_failure_rate(trials: list[TrialLog]) -> float | None:
    """Fraction of stop-failure trials where the stop signal arrived after the
    decision point (late-stop: SSD >= decision_point_ms, stop_signal not logged).

    These are trials where inhibition was mechanically impossible; they inflate
    the stop-failure rate without reflecting a genuine failure of the stop process.

    Formula: n_stop_failures_without_stop_signal_event / n_stop_failures_total
    Returns None if no stop-failure trials.
    """
    stop_failures = [
        t for t in trials
        if is_stop_trial(t) and t.failure_mode == "stop_failure"
    ]
    if not stop_failures:
        return None
    n_late = sum(1 for t in stop_failures if not _has_stop_signal_event(t))
    return n_late / len(stop_failures)


# --- Validity report ---

_INDEPENDENCE_NOTE = (
    "Race model independence assumed. In Phase 7, go and stop processes share"
    " the CortexEvidenceGenerator; true statistical independence is not guaranteed"
    " but is not empirically testable from behavioural data alone."
)


class StopSignalValidityReport(BaseModel):
    """Validity report for a stop-signal experiment condition.

    Follows Verbruggen et al. 2019 §Validity checks.
    All checks are soft (status flags + values); none raise errors.
    """

    # --- RT validity ---
    # Verbruggen 2019: failed-stop RT should be shorter than go RT (race model).
    go_rt_mean_s: float | None          # mean go RT (responding trials)
    failed_stop_rt_mean_s: float | None  # mean failed-stop RT
    rt_check_passed: bool               # True when failed_stop_rt_mean_s < go_rt_mean_s
    rt_check_note: str                  # human-readable explanation of result

    # --- Trial counts (exclusion tracking) ---
    n_total_trials: int
    n_stop_trials: int
    n_go_trials: int
    n_late_stop_trials: int             # stop trials with SSD >= decision_point_ms (not logged)
    n_excluded_for_ssrt: int            # late-stop trials excluded from SSRT computation

    # --- Independence assumption documentation ---
    # Verbruggen 2019: the race model assumes the go and stop processes are independent.
    # In Phase 7, the BG model uses the same cortical evidence generator for go and stop
    # processes; true independence is not guaranteed. This is documented, not checked.
    independence_assumption_note: str   # fixed text explaining the assumption and its status

    # --- Stop proportion check ---
    # Verbruggen 2019: empirical stop proportion should be close to the intended value.
    empirical_stop_proportion: float    # n_stop_trials / n_total_trials

    # --- Inhibition function monotonicity ---
    inhibition_function_monotone: bool | None
    # True if failure_rate rises with SSD (or None if <2 SSD levels)


def validate_stop_signal_data(
    trials: list[TrialLog],
    intended_stop_proportion: float = 0.25,
) -> StopSignalValidityReport:
    """Run Verbruggen et al. 2019 validity checks on a stop-signal trial list.

    Raises ValueError if trials is empty.

    All checks are soft: no exception is raised on check failure.
    The report documents results and notes for human interpretation.

    independence_assumption_note is always the fixed string in _INDEPENDENCE_NOTE.

    rt_check_note:
      - If rt_check_passed=True:  "Failed-stop RT < go RT: race model prediction satisfied."
      - If failed_stop_rt_mean_s is None:  "No stop failures: RT check not applicable."
      - If go_rt_mean_s is None:  "No responding go trials: RT check not applicable."
      - If rt_check_passed=False and values are equal within 1e-9:
          deterministic BG model note (deferred to Phase 9+).
      - If rt_check_passed=False and failed_stop_rt_mean_s > go_rt_mean_s:
          "unexpected" note directing investigation.
    """
    if not trials:
        raise ValueError("trials list is empty — cannot validate empty input")

    # --- Trial counts ---
    n_total = len(trials)
    n_stop = sum(1 for t in trials if is_stop_trial(t))
    n_go = sum(1 for t in trials if is_go_trial(t))

    # Late-stop trials: stop trials that did NOT get a stop_signal event logged
    # (SSD >= decision_point_ms — mechanically unavoidable failures).
    n_late = sum(
        1 for t in trials
        if is_stop_trial(t) and not _has_stop_signal_event(t)
    )

    # --- RT validity ---
    _go_stats = go_rt_stats(trials)
    go_rt_mean = _go_stats["mean_s"]
    failed_rt_mean = failed_stop_rt_mean(trials)

    # Trigger: determine which RT-check branch applies (priority order matters).
    # Why: each branch represents a distinct missing-data or comparison case;
    #      the order avoids evaluating comparisons when operands are None.
    # Outcome: rt_check_passed and rt_check_note are set deterministically.
    if failed_rt_mean is None:
        rt_check_passed = False
        rt_check_note = "No stop failures: RT check not applicable."
    elif go_rt_mean is None:
        rt_check_passed = False
        rt_check_note = "No responding go trials: RT check not applicable."
    elif failed_rt_mean < go_rt_mean:
        rt_check_passed = True
        rt_check_note = "Failed-stop RT < go RT: race model prediction satisfied."
    elif math.isclose(failed_rt_mean, go_rt_mean, abs_tol=1e-9):
        # Trigger: RTs are numerically equal — expected in deterministic BG models
        #          where both go RT and failed-stop RT derive from the same
        #          selection_latency (BG-internal, not subject-level reaction time).
        # Why: with a deterministic model there is no RT variability to separate
        #      the two distributions; this is a known limitation of Phase 7.
        # Outcome: failure documented with a deferred-check note; no error raised.
        rt_check_passed = False
        rt_check_note = (
            "Failed-stop RT == go RT: expected with deterministic BG model where both"
            " RTs derive from the same selection_latency. Race model check deferred to"
            " Phase 9+ with RT variability."
        )
    else:
        # failed_rt_mean > go_rt_mean — unexpected direction
        rt_check_passed = False
        rt_check_note = (
            "Failed-stop RT > go RT: unexpected. Investigate BG model or SSD schedule."
        )

    # --- Inhibition function monotonicity ---
    inh = inhibition_function(trials)
    ssds_sorted = sorted(inh.keys())
    if len(ssds_sorted) < 2:
        # Trigger: fewer than 2 SSD levels present in the data.
        # Why: monotonicity requires at least two consecutive points to compare.
        # Outcome: field is None to distinguish "not assessed" from "True/False".
        inh_monotone: bool | None = None
    else:
        rates = [inh[ssd]["failure_rate"] for ssd in ssds_sorted]
        # Non-decreasing across all consecutive pairs ↔ monotone inhibition function.
        inh_monotone = all(
            rates[i] <= rates[i + 1] for i in range(len(rates) - 1)
        )

    return StopSignalValidityReport(
        go_rt_mean_s=go_rt_mean,
        failed_stop_rt_mean_s=failed_rt_mean,
        rt_check_passed=rt_check_passed,
        rt_check_note=rt_check_note,
        n_total_trials=n_total,
        n_stop_trials=n_stop,
        n_go_trials=n_go,
        n_late_stop_trials=n_late,
        n_excluded_for_ssrt=n_late,
        independence_assumption_note=_INDEPENDENCE_NOTE,
        empirical_stop_proportion=n_stop / n_total,
        inhibition_function_monotone=inh_monotone,
    )


# --- Output model and aggregator ---


class StopSignalMetrics(BaseModel):
    """Aggregated stop-signal metrics for one experimental condition."""

    n_trials: int
    n_go_trials: int
    n_stop_trials: int

    go_rt_mean_s: float | None = None
    go_rt_std_s: float | None = None

    failed_stop_rt_mean_s: float | None = None

    stop_failure_rate: float | None = None
    # n_failures / n_stop_trials; None if 0 stop trials

    # inhibition_function: {ssd_ms: failure_rate} — early-stop trials only
    inhibition_function: dict[int, float] = {}
    inhibition_function_n: dict[int, int] = {}  # trial count per SSD

    ssrt_estimate_s: float | None = None
    mean_ssd_ms: float | None = None
    # mean SSD across all stop trials (excl. late-stops)

    cancellation_latency_mean_s: float | None = None

    trigger_failure_rate: float | None = None


def compute_stop_signal_metrics(trials: list[TrialLog]) -> StopSignalMetrics:
    """Compute all stop-signal metrics from a list of TrialLog objects.

    Raises ValueError if trials is empty.
    Handles mixed go/stop/late-stop trials correctly.
    """
    if not trials:
        raise ValueError("trials list is empty — cannot compute metrics on empty input")

    n_trials = len(trials)
    n_go = sum(1 for t in trials if is_go_trial(t))
    n_stop = sum(1 for t in trials if is_stop_trial(t))

    # --- Go RT ---
    _go_stats = go_rt_stats(trials)

    # --- Stop failure rate ---
    _stop_failure_rate: float | None = None
    if n_stop > 0:
        n_failures = sum(
            1 for t in trials
            if is_stop_trial(t) and t.failure_mode == "stop_failure"
        )
        _stop_failure_rate = n_failures / n_stop

    # --- Inhibition function ---
    _inh = inhibition_function(trials)
    _inh_rates = {ssd: v["failure_rate"] for ssd, v in _inh.items()}
    _inh_ns = {ssd: v["n"] for ssd, v in _inh.items()}

    # --- SSRT and mean SSD ---
    _ssrt = estimate_ssrt(trials)

    # Compute mean_ssd_ms separately (needed as a field even if SSRT is None).
    ssds = [
        ssd
        for t in trials
        if is_stop_trial(t)
        for ssd in (extract_ssd_ms(t),)
        if ssd is not None
    ]
    _mean_ssd_ms: float | None = (sum(ssds) / len(ssds)) if ssds else None

    return StopSignalMetrics(
        n_trials=n_trials,
        n_go_trials=n_go,
        n_stop_trials=n_stop,
        go_rt_mean_s=_go_stats["mean_s"],
        go_rt_std_s=_go_stats["std_s"],
        failed_stop_rt_mean_s=failed_stop_rt_mean(trials),
        stop_failure_rate=_stop_failure_rate,
        inhibition_function=_inh_rates,
        inhibition_function_n=_inh_ns,
        ssrt_estimate_s=_ssrt,
        mean_ssd_ms=_mean_ssd_ms,
        cancellation_latency_mean_s=cancellation_latency_mean(trials),
        trigger_failure_rate=trigger_failure_rate(trials),
    )
