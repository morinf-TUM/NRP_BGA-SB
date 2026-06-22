"""Phase-0 smoke TF: reads the tick datapack and appends it to NRP_BGA_LOG."""

import json
import os

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="tick", id=DataPackIdentifier("tick", "smoke"))
@TransceiverFunction("smoke")
def log_tick(tick):
    log_path = os.environ["NRP_BGA_LOG"]
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"t_ns": tick.data["t_ns"]}) + "\n")
    return []
