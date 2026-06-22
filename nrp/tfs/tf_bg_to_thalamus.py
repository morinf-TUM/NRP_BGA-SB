"""Phase 1-3 link: forward the BG decision to the thalamus as the committed
decision. In Phase 4 this is replaced by tf_bg_to_commitment +
tf_commitment_to_thalamus."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="decision", id=DataPackIdentifier("decision", "bg"))
@TransceiverFunction("thalamus")
def bg_to_thalamus(decision):
    out = JsonDataPack("committed_decision", "thalamus")
    for k, v in decision.data.items():
        out.data[k] = v
    return [out]
