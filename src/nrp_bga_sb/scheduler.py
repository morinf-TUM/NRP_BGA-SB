"""Logical-clock scheduler with four independent BG frequency knobs (Task 3.1).

Phase 3 of the NRP_BGA-SB project makes the four candidate "BG effective update
frequency" variables independently controllable so that ablations can determine
which is causally meaningful before collapsing them.

The four candidates (PROJECT_MEMORY §5):
  1. input_sampling_hz     — how often BG reads cortical state
  2. integration_step_hz   — BG internal ODE/policy solver step rate
  3. output_emission_hz    — how often BG publishes a gating output
  4. commitment_update_hz  — how often a committed-channel decision may change

ScheduledBGAdapter wraps any policy with the standard interface
  (TrialLog, ActionEvidence) → BGDecision
and runs an internal integer-tick simulation to apply the four gates before
returning a committed decision.

Integer-tick arithmetic is used throughout to avoid floating-point edge cases
that arise when using float modulo with sub-millisecond base_dt_ms values.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, model_validator

from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog

# --- FrequencyConfig ---


class FrequencyConfig(BaseModel):
    """Four independent BG update-frequency knobs plus the simulation base step.

    All Hz values are constrained to (0, 1000/base_dt_ms] so that no frequency
    can be faster than the simulation time step.

    Attributes:
        input_sampling_hz:    How often BG reads cortical state (per second).
        integration_step_hz:  How often the BG policy/solver is invoked (per second).
        output_emission_hz:   How often a new decision is promoted to the emission
                              layer (per second).
        commitment_update_hz: How often the committed-channel decision may change
                              (per second).
        base_dt_ms:           Simulation step size in milliseconds.  All period_ticks
                              values are derived from this.
    """

    input_sampling_hz: float = 160.0
    integration_step_hz: float = 1000.0
    output_emission_hz: float = 160.0
    commitment_update_hz: float = 160.0
    base_dt_ms: float = 1.0

    @model_validator(mode="after")
    def _validate_frequency_bounds(self) -> FrequencyConfig:
        # Trigger: any Hz value is ≤ 0 or exceeds the maximum rate imposed by the
        #          simulation step (1000.0 / base_dt_ms Hz).
        # Why: a frequency of 0 or below is nonsensical; a frequency faster than
        #      the simulation step means the gate would fire more than once per step,
        #      which is undefined behaviour in the integer-tick model.
        # Outcome: ValidationError raised immediately; caller must fix the config.
        if self.base_dt_ms <= 0:
            raise ValueError(f"base_dt_ms must be > 0, got {self.base_dt_ms}")

        max_hz = 1000.0 / self.base_dt_ms
        for name, value in [
            ("input_sampling_hz", self.input_sampling_hz),
            ("integration_step_hz", self.integration_step_hz),
            ("output_emission_hz", self.output_emission_hz),
            ("commitment_update_hz", self.commitment_update_hz),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
            if value > max_hz:
                raise ValueError(
                    f"{name}={value} Hz exceeds maximum allowed rate "
                    f"{max_hz} Hz (= 1000 / base_dt_ms={self.base_dt_ms})"
                )
        return self

    @classmethod
    def from_effective_hz(cls, effective_hz: float, **kwargs: object) -> FrequencyConfig:
        """Convenience constructor: sets all four frequency knobs to `effective_hz`.

        This represents the "unified effective BG frequency" used in Phase 5 sweep
        experiments, where all four timing variables are co-varied together.

        The primary variable, identified by the Phase 3 ablation (Task 3.3), is
        output_emission_hz: it maps directly to nrp-core's EngineTimestep and
        governs when downstream components (thalamus) see updated BG decisions.
        In the abstract constant-evidence model, all four knobs produce equal
        behavioral outcomes; they are co-varied here for simplicity until Phase 4
        introduces time-varying cortical evidence.

        Args:
            effective_hz: Target frequency in Hz for all four knobs.
            **kwargs:     Forwarded to FrequencyConfig (e.g., base_dt_ms).

        Returns:
            A FrequencyConfig with all four Hz knobs set to effective_hz.

        Raises:
            ValidationError: if effective_hz violates FrequencyConfig bounds.
        """
        return cls(
            input_sampling_hz=effective_hz,
            integration_step_hz=effective_hz,
            output_emission_hz=effective_hz,
            commitment_update_hz=effective_hz,
            **kwargs,
        )


# --- ScheduledBGAdapter ---


class ScheduledBGAdapter:
    """Wraps a BG policy with an integer-tick simulation of four frequency gates.

    Each __call__ runs a fresh simulation loop from tick 0 to n_steps-1.  No
    inter-call state is stored; the adapter is safe for repeated or concurrent
    use within the same process.

    The four gates are applied in pipeline order within each tick:
      1. Input sampling   — copies incoming evidence into last_sampled_evidence
      2. Integration      — calls base_policy to produce a raw decision
      3. Output emission  — promotes the raw decision to the emission layer
      4. Commitment gate  — promotes the emitted decision to the committed layer

    If the loop completes without establishing a committed decision (a pathological
    configuration where the accumulation window is shorter than all four periods),
    the adapter falls back to a direct call to base_policy.  This path preserves
    the fail-fast contract: the caller always receives a BGDecision.
    """

    def __init__(
        self,
        base_policy: Callable[[TrialLog, ActionEvidence], BGDecision],
        config: FrequencyConfig,
        accumulation_ms: float = 200.0,
        cortex_generator: Callable[[TrialLog, float], ActionEvidence] | None = None,
    ) -> None:
        """Construct a ScheduledBGAdapter.

        Args:
            base_policy:       Any callable matching the standard policy interface.
            config:            Four-knob frequency configuration.
            accumulation_ms:   Length of the pre-decision integration window (ms).
                               The loop runs for round(accumulation_ms / base_dt_ms)
                               ticks, with a minimum of 1.
            cortex_generator:  Optional callable (trial_log, elapsed_ms) → ActionEvidence.
                               When set, Gate 1 (input sampling) calls the generator at
                               each sampling tick with elapsed_ms = tick * base_dt_ms instead
                               of copying the static action_evidence.  This introduces
                               time-varying cortical evidence so that BG-frequency effects
                               become observable (Phase 4+).  When None, behaviour is
                               identical to Phase 3 (static evidence).
        """
        self._base_policy = base_policy
        self._config = config
        self._accumulation_ms = accumulation_ms
        self._cortex_generator = cortex_generator

        # Pre-compute integer period_ticks for each gate.
        # Trigger: convert Hz → ms period → ticks using the base simulation step.
        # Why: integer tick arithmetic avoids floating-point modulo drift that would
        #      produce spurious extra or missing gate fires at high step counts.
        # Outcome: each period is a positive integer number of base_dt_ms steps.
        dt = config.base_dt_ms
        self._n_steps: int = max(1, round(accumulation_ms / dt))
        self._input_period: int = max(1, round(1000.0 / (config.input_sampling_hz * dt)))
        self._integration_period: int = max(1, round(1000.0 / (config.integration_step_hz * dt)))
        self._emission_period: int = max(1, round(1000.0 / (config.output_emission_hz * dt)))
        self._commitment_period: int = max(1, round(1000.0 / (config.commitment_update_hz * dt)))

    def __call__(
        self,
        trial_log: TrialLog,
        action_evidence: ActionEvidence,
    ) -> BGDecision:
        """Run the four-gate tick loop and return the committed BGDecision.

        State is entirely local to this call; no instance fields are mutated.

        Returns:
            The committed BGDecision from the last commitment gate that fired,
            or — if no commitment was established — the result of a direct
            base_policy call (fallback path).
        """
        # --- Per-call state (fresh each invocation — statelessness invariant) ---
        last_sampled_evidence: ActionEvidence | None = None
        _last_raw_decision: BGDecision | None = None
        last_emitted_decision: BGDecision | None = None
        committed_decision: BGDecision | None = None

        # --- Tick loop ---
        for tick in range(self._n_steps):

            # Gate 1: Input sampling
            # Trigger: tick is a multiple of input_period.
            # Why: models the rate at which BG reads cortical salience; slower sampling
            #      means the BG operates on stale evidence between sampling events.
            # Outcome: last_sampled_evidence is updated from the cortex generator (if
            #   set) or from the static action_evidence (Phase 3 / no-generator path).
            if tick % self._input_period == 0:
                if self._cortex_generator is not None:
                    # Time-varying evidence: elapsed_ms increases with each tick.
                    # Trigger: cortex_generator is configured (Phase 4+).
                    # Why: makes frequency effects observable — slower input sampling
                    #      means BG reads earlier, lower-salience evidence; at very low
                    #      frequencies BG may never see enough evidence to commit.
                    # Outcome: last_sampled_evidence reflects cortical state at this tick.
                    elapsed_ms = tick * self._config.base_dt_ms
                    last_sampled_evidence = self._cortex_generator(trial_log, elapsed_ms)
                else:
                    last_sampled_evidence = action_evidence

            # Gate 2: Integration (policy call)
            # Trigger: tick is a multiple of integration_period AND evidence is available.
            # Why: models the BG internal computation rate (e.g., ODE solver step);
            #      the policy is only called when there is sampled cortical evidence
            #      to integrate.
            # Outcome: _last_raw_decision is updated with a fresh BGDecision.
            if tick % self._integration_period == 0 and last_sampled_evidence is not None:
                _last_raw_decision = self._base_policy(trial_log, last_sampled_evidence)

            # Gate 3: Output emission
            # Trigger: tick is a multiple of emission_period AND a raw decision exists.
            # Why: models the rate at which BG publishes its decision downstream;
            #      slower emission means the thalamus sees updates less frequently.
            # Outcome: last_emitted_decision is promoted from the integration layer.
            if tick % self._emission_period == 0 and _last_raw_decision is not None:
                last_emitted_decision = _last_raw_decision

            # Gate 4: Commitment update
            # Trigger: tick is a multiple of commitment_period AND an emitted decision
            #          exists.
            # Why: models the rate at which a committed action channel may change;
            #      slower commitment means the system is less reactive to new evidence
            #      once an action is underway.
            # Outcome: committed_decision is updated from the emission layer.
            if tick % self._commitment_period == 0 and last_emitted_decision is not None:
                committed_decision = last_emitted_decision

        # --- Return committed decision or fall back to direct policy call ---
        # Trigger: committed_decision is None after the full tick loop.
        # Why: a pathological configuration (all gate periods longer than the
        #      accumulation window, which cannot happen in practice because tick 0
        #      fires all gates simultaneously) could prevent any commitment.  The
        #      fallback guarantees the caller always receives a BGDecision.
        # Outcome: base_policy is called directly with the original evidence.
        if committed_decision is not None:
            return committed_decision

        return self._base_policy(trial_log, action_evidence)
