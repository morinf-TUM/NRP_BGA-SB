"""Phase 4 link: forward the BG engine's raw decision to the commitment engine.
This decouples the commitment rate from the BG output rate; the commitment
engine sees stale decisions between its own (slow) steps."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="decision", id=DataPackIdentifier("decision", "bg"))
@TransceiverFunction("commitment")
def bg_to_commitment(decision):
    # Trigger: BG has not yet produced a decision (null datapack on first steps).
    # Why: NlohmannJson.keys() raises on json_type 'null'; skip until data arrives.
    # Outcome: commitment engine keeps its previous state until BG has latched a decision.
    if not decision.data:
        return []
    out = JsonDataPack("raw_decision", "commitment")
    # NlohmannJson does not support .items(); use .keys() + subscript.
    for k in decision.data.keys():
        out.data[k] = decision.data[k]
    return [out]
