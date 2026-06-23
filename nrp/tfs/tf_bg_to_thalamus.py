"""Phase 1-3 link: forward the BG decision to the thalamus as the committed
decision. In Phase 4 this is replaced by tf_bg_to_commitment +
tf_commitment_to_thalamus."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="decision", id=DataPackIdentifier("decision", "bg"))
@TransceiverFunction("thalamus")
def bg_to_thalamus(decision):
    # Trigger: BG has not yet emitted a decision (null datapack on early steps).
    # Why: NlohmannJson.keys() raises on json_type 'null'; skip until data arrives.
    # Outcome: thalamus keeps its previous state until a decision is present.
    if not decision.data:
        return []
    out = JsonDataPack("committed_decision", "thalamus")
    # NlohmannJson does not support .items(); use .keys() + subscript.
    for k in decision.data.keys():
        out.data[k] = decision.data[k]
    return [out]
