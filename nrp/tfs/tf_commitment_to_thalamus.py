"""Phase 4 link: forward the commitment engine's latched decision to the thalamus.
This replaces the direct BG→thalamus path (tf_bg_to_thalamus); the thalamus now
consumes committed decisions at the commitment engine's (slow) rate."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="committed",
                id=DataPackIdentifier("committed_decision", "commitment"))
@TransceiverFunction("thalamus")
def commitment_to_thalamus(committed):
    # Trigger: commitment engine has not yet latched a decision (null on first steps).
    # Why: NlohmannJson.keys() raises on json_type 'null'; skip until data arrives.
    # Outcome: thalamus keeps its previous state until commitment engine has latched.
    if not committed.data:
        return []
    out = JsonDataPack("committed_decision", "thalamus")
    # NlohmannJson does not support .items(); use .keys() + subscript.
    for k in committed.data.keys():
        out.data[k] = committed.data[k]
    return [out]
