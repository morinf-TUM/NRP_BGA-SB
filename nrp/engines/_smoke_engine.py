"""Phase-0 smoke engine: proves NRPCoreSim launches a python_json engine and
that nrp_bga_sb is importable from inside the engine process."""

from nrp_core.engines.python_json import EngineScript

import nrp_bga_sb  # noqa: F401  -- import smoke: must succeed inside the engine process


class Script(EngineScript):
    def initialize(self):
        self._registerDataPack("tick")
        self._setDataPack("tick", {"t_ns": self._time_ns})

    def runLoop(self, timestep_ns):
        self._setDataPack("tick", {"t_ns": self._time_ns})

    def shutdown(self):
        pass
