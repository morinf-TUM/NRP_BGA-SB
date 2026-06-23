"""BG engine: runs the GPR BG model on the most recent sampled evidence and emits
a `decision` datapack. Delegates to BGIntegratorDriver, a stateful integrator that
carries GPR activations across emission steps within a trial and reads out the
current (possibly unsettled) state -- so the internal-integration-step rate
(knob 2) is functionally dissociable (PROJECT_MEMORY §15.7), not idempotent. Its
observable effect in this pipeline is decision latency (a slow rate settles late).

Input-sampling, emission, and commitment remain EngineTimesteps (§15.4); the
integration rate rides on the params overlay (`integration_hz`)."""

import json
import os

from nrp_core.engines.python_json import EngineScript

from nrp.serde import decision_to_dict, evidence_from_dict
from nrp_bga_sb.bg_integrator import BGIntegratorDriver


class Script(EngineScript):
    def initialize(self):
        with open(os.environ["NRP_BGA_TRIAL_PARAMS"]) as fh:
            params = json.load(fh)
        # Knob 2: BG internal integration step, scheduled by the driver from the
        # integration rate. State is created here and carried across runLoop calls
        # (one NRPCoreSim run = one trial); a slower rate settles later.
        self._driver = BGIntegratorDriver(
            integration_hz=float(params["integration_hz"]),
        )
        self._registerDataPack("sampled_evidence")
        self._registerDataPack("decision")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("sampled_evidence")
        # Trigger: no evidence delivered yet (first ticks before the sampler TF fires).
        # Why: the driver needs a populated ActionEvidence; skip until present.
        # Outcome: `decision` keeps its previous value until evidence arrives.
        if not raw or "channel_salience" not in raw:
            return
        evidence = evidence_from_dict(raw)
        elapsed_ms = self._time_ns / 1.0e6
        decision = self._driver.advance(elapsed_ms, evidence)
        self._setDataPack("decision", decision_to_dict(decision))

    def shutdown(self):
        pass
