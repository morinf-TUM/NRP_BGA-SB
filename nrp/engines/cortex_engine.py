"""Cortex engine: emits a time-varying cortical-salience ramp as the `evidence`
datapack. Delegates to the validated CortexEvidenceGenerator; reads per-trial
parameters from the env-pointed JSON file (NRP_BGA_TRIAL_PARAMS)."""

import json
import os

from nrp_core.engines.python_json import EngineScript

from nrp_bga_sb.cortex import CortexConfig, CortexEvidenceGenerator
from nrp_bga_sb.schemas import TrialLog
from nrp.serde import evidence_to_dict


class Script(EngineScript):
    def initialize(self):
        with open(os.environ["NRP_BGA_TRIAL_PARAMS"]) as fh:
            params = json.load(fh)
        # Build a minimal but complete TrialLog so the generator has every
        # required field regardless of which ones it reads.
        self._trial = TrialLog(
            trial_id=params["trial_id"],
            seed=params["seed"],
            task_type="go_nogo",
            cue_identity=params["cue_identity"],
            cue_onset_time=0.0,
        )
        self._cortex = CortexEvidenceGenerator(CortexConfig())
        self._registerDataPack("evidence")
        self._emit(0.0)

    def runLoop(self, timestep_ns):
        # _time_ns is engine logical time; the generator expects elapsed ms.
        self._emit(self._time_ns / 1.0e6)

    def _emit(self, elapsed_ms: float):
        ev = self._cortex(self._trial, elapsed_ms)
        self._setDataPack("evidence", evidence_to_dict(ev))

    def shutdown(self):
        pass
