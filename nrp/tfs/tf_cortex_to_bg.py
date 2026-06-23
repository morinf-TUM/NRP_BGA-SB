"""Phase 1-2 link: forward cortical evidence to the BG engine each step. In
Phase 3 this is replaced by tf_cortex_to_sampler + tf_sampler_to_bg."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="evidence", id=DataPackIdentifier("evidence", "cortex"))
@TransceiverFunction("bg")
def cortex_to_bg(evidence):
    # Trigger: cortex has not yet emitted evidence (null datapack on first steps).
    # Why: NlohmannJson.keys() raises on json_type 'null'; skip until data arrives.
    # Outcome: BG engine keeps its previous state until evidence is present.
    if not evidence.data:
        return []
    out = JsonDataPack("sampled_evidence", "bg")
    # NlohmannJson does not support .items(); use .keys() + subscript.
    for k in evidence.data.keys():
        out.data[k] = evidence.data[k]
    return [out]
