"""BG engine (single-rate, Phase 1): runs the GPR BG model on the most recent
sampled evidence and emits a `decision` datapack. Delegates to BGAdapter.

In Phase 1 input-sampling, integration, and emission all coincide with this
engine's EngineTimestep; later phases split sampling (Phase 3), commitment
(Phase 4), and internal integration (Phase 5) out into their own rates."""

import json
import os

from nrp_core.engines.python_json import EngineScript

from nrp.serde import decision_to_dict, evidence_from_dict
from nrp_bga_sb.bg_model import BGAdapter, BGModelConfig
from nrp_bga_sb.schemas import TrialLog


class Script(EngineScript):
    def initialize(self):
        with open(os.environ["NRP_BGA_TRIAL_PARAMS"]) as fh:
            params = json.load(fh)
        self._trial = TrialLog(
            trial_id=params["trial_id"], seed=params["seed"],
            task_type="go_nogo", cue_identity=params["cue_identity"],
            cue_onset_time=0.0,
        )
        # Knob 2: BG internal integration step. Modelled as N solver sub-steps per
        # emission step -- internal to this engine, NOT an EngineTimestep (§15.4).
        self._substeps = int(params.get("integration_substeps", 1))
        self._bg = BGAdapter(BGModelConfig())
        self._registerDataPack("sampled_evidence")
        self._registerDataPack("decision")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("sampled_evidence")
        # Trigger: no evidence delivered yet (first ticks before TF fires).
        # Why: BGAdapter needs a populated ActionEvidence; skip until present.
        # Outcome: `decision` keeps its previous value until evidence arrives.
        if not raw or "channel_salience" not in raw:
            return
        evidence = evidence_from_dict(raw)
        decision = None
        # Integrate the BG model `_substeps` times before emitting; the last
        # decision is the emitted one. With substeps=1 behaviour is unchanged.
        for _ in range(self._substeps):
            decision = self._bg(self._trial, evidence)
        self._setDataPack("decision", decision_to_dict(decision))

    def shutdown(self):
        pass
