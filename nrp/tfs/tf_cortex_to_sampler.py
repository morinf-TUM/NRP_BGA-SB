"""Phase 3 link: forward cortical evidence to the sampler engine each step.
The sampler then latches it at its own (slow) EngineTimestep before the BG
engine reads it via tf_sampler_to_bg."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="evidence", id=DataPackIdentifier("evidence", "cortex"))
@TransceiverFunction("sampler")
def cortex_to_sampler(evidence):
    # Trigger: cortex has not yet emitted evidence (null datapack on first steps).
    # Why: NlohmannJson.keys() raises on json_type 'null'; skip until data arrives.
    # Outcome: sampler engine keeps its previous state until evidence is present.
    if not evidence.data:
        return []
    out = JsonDataPack("incoming_evidence", "sampler")
    # NlohmannJson does not support .items(); use .keys() + subscript.
    for k in evidence.data.keys():
        out.data[k] = evidence.data[k]
    return [out]
