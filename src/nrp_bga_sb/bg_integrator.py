"""Stateful BG integration driver (knob 2). Carries GPR activations across
integration sub-steps within a trial and reads out the current (possibly
unsettled) state, so the integration RATE controls how far the BG has settled
at any moment. This is the science-layer change that makes the
internal-integration-step knob functionally dissociable (PROJECT_MEMORY §15.7).

There is no decision-deadline window: the integrator keeps sweeping at the
integration rate for the whole trial. A slower rate therefore settles LATER
(its observable effect in the nrp pipeline is decision latency, not a
go-success miss — see the design doc, Revision 2).

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
    """Schedules one carried GPR sweep per integration-tick boundary
    (k / integration_hz seconds) across the trial, reading out the current
    (possibly unsettled) state. More sub-steps -> closer to the BGModel.compute()
    fixed point; a slow integration rate reaches it later.
    """

    integration_hz: float
    config: BGModelConfig = field(default_factory=BGModelConfig)

    def __post_init__(self) -> None:
        if self.integration_hz <= 0:
            raise ValueError(f"integration_hz must be > 0, got {self.integration_hz}")
        self._model = BGModel(self.config)
        self._period_ms = 1000.0 / self.integration_hz
        self._next_k = 0
        self._state: BGIntegratorState | None = None

    def advance(self, elapsed_ms: float, evidence: ActionEvidence) -> BGDecision:
        n = evidence.n_channels
        if self._state is None:
            self._state = BGIntegratorState.initial(n, self.config)

        saliences = evidence.channel_salience
        # Fire every integration tick whose boundary has been crossed since the last
        # call; each fires one carried sweep on the evidence current at this call.
        # No upper bound: at a low rate the few sweeps land late, so the decision
        # settles late (latency signature) rather than never.
        while self._next_k * self._period_ms <= elapsed_ms:
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
