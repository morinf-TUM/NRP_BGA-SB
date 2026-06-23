"""Commitment engine: realises the BG DECISION-COMMITMENT update frequency
(§15.4 -- not a built-in nrp-core concept, so modelled as its own engine). It
latches the latest raw BG decision into `committed_decision` only at its own
(slow) EngineTimestep; between latches, downstream sees the previously committed
decision."""

from nrp_core.engines.python_json import EngineScript


class Script(EngineScript):
    def initialize(self):
        self._registerDataPack("raw_decision")
        self._registerDataPack("committed_decision")

    def runLoop(self, timestep_ns):
        raw = self._getDataPack("raw_decision")
        if raw and "selected_channel" in raw:
            self._setDataPack("committed_decision", dict(raw))

    def shutdown(self):
        pass
