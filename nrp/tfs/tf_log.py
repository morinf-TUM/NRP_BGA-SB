"""Logger TF: append the current decision and motor command to NRP_BGA_LOG as
one JSON object per FTILoop step. Off-loop persistence — the trace is scored
offline by the existing scorer/task-engine logic."""

import json
import os

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="decision", id=DataPackIdentifier("decision", "bg"))
@EngineDataPack(keyword="motor", id=DataPackIdentifier("motor", "thalamus"))
@TransceiverFunction("thalamus")
def log_step(decision, motor):
    record = {
        "decision": dict(decision.data) if decision.data else None,
        "motor": dict(motor.data) if motor.data else None,
    }
    with open(os.environ["NRP_BGA_LOG"], "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return []
