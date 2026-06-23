"""Stateful BG integration driver (knob 2). Carries GPR activations across
integration sub-steps within a trial and reads out the current (possibly
unsettled) state, so the integration RATE controls how far the BG has settled
by the decision deadline. This is the science-layer change that makes the
internal-integration-step knob functionally dissociable (PROJECT_MEMORY §15.7).

The nrp `bg` engine cannot run on the host (it imports nrp_core), so the
scheduling + stepping logic lives here, host-tested, and the engine delegates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nrp_bga_sb.bg_model import (
    BGIntegratorState,
    BGModel,
    BGModelConfig,
    selection_latency_s,
)
from nrp_bga_sb.schemas import ActionEvidence, BGDecision


@dataclass
class BGIntegratorDriver:
    """Schedules one carried GPR sweep per integration-tick boundary inside the
    half-open window [0, accumulation_ms). At the baseline rate the integrator
    converges and the readout equals BGModel.compute(); at low rates it is read
    out before settling, producing the miss-at-5-Hz / success-at->=10-Hz boundary.
    """

    integration_hz: float
    accumulation_ms: float = 200.0
    config: BGModelConfig = field(default_factory=BGModelConfig)

    def __post_init__(self) -> None:
        if self.integration_hz <= 0:
            raise ValueError(f"integration_hz must be > 0, got {self.integration_hz}")
        if self.accumulation_ms <= 0:
            raise ValueError(f"accumulation_ms must be > 0, got {self.accumulation_ms}")
        self._model = BGModel(self.config)
        self._period_ms = 1000.0 / self.integration_hz
        self._next_k = 0
        self._state: BGIntegratorState | None = None

    def advance(self, elapsed_ms: float, evidence: ActionEvidence) -> BGDecision:
        n = evidence.n_channels
        if self._state is None:
            self._state = BGIntegratorState.initial(n, self.config)

        saliences = evidence.channel_salience
        # Fire every integration tick whose boundary has been crossed by now and
        # lies strictly inside the accumulation window. Each fires one sweep on the
        # evidence current at this call. The strict `< accumulation_ms` bound
        # excludes the t=200 ms tick, which is what makes 5 Hz (period=200 ms) miss.
        while (
            self._next_k * self._period_ms <= elapsed_ms
            and self._next_k * self._period_ms < self.accumulation_ms
        ):
            self._state = self._model.step(self._state, saliences, n_sweeps=1)
            self._next_k += 1

        s = self._state
        return BGDecision(
            sim_time=evidence.sim_time,
            trial_id=evidence.trial_id,
            selected_channel=s.selected_channel,
            decision_margin=s.decision_margin,
            suppression_vector=s.suppression_vector,
            channel_activations=s.channel_activations,
            selection_latency=selection_latency_s(self.config, s.T_winner),
        )
