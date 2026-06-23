"""Thalamus engine: gates the committed BG decision into a motor command.
Delegates to ThalamusGate. The incoming datapack is named `committed_decision`
so the same engine serves Phase 1-3 (fed straight from BG) and Phase 4+ (fed
from the commitment engine) without change."""

from nrp_core.engines.python_json import EngineScript

from nrp.serde import decision_from_dict, motor_to_dict
from nrp_bga_sb.thalamus import ThalamusConfig, ThalamusGate


class Script(EngineScript):
    def initialize(self):
        self._thalamus = ThalamusGate(ThalamusConfig())
        self._registerDataPack("committed_decision")
        self._registerDataPack("motor")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("committed_decision")
        if not raw or "selected_channel" not in raw:
            return
        motor = self._thalamus(decision_from_dict(raw))
        self._setDataPack("motor", motor_to_dict(motor))

    def shutdown(self):
        pass
