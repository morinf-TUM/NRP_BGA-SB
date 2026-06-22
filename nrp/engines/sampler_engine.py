"""Sampler engine: realises the BG INPUT-SAMPLING frequency. It runs at its own
(slow) EngineTimestep and latches the most recent cortical evidence into
`sampled_evidence`. Between its steps the BG engine sees stale evidence -- this
is exactly the mechanism that makes low input-sampling rates miss the evidence
ramp peak."""

from nrp_core.engines.python_json import EngineScript


class Script(EngineScript):
    def initialize(self):
        self._registerDataPack("incoming_evidence")
        self._registerDataPack("sampled_evidence")

    def runLoop(self, timestep_ns):
        latest = self._getDataPack("incoming_evidence")
        if latest and "channel_salience" in latest:
            # Latch: copy the latest cortical evidence at the sampling rate.
            self._setDataPack("sampled_evidence", dict(latest))

    def shutdown(self):
        pass
