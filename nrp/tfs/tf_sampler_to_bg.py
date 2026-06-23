"""Phase 3 link: forward the sampler engine's latched evidence to the BG engine.
This decouples the BG input-sampling rate from the cortex emission rate; the BG
sees stale evidence between sampler steps."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="sampled", id=DataPackIdentifier("sampled_evidence", "sampler"))
@TransceiverFunction("bg")
def sampler_to_bg(sampled):
    # Trigger: sampler has not yet latched evidence (null datapack on first steps).
    # Why: NlohmannJson.keys() raises on json_type 'null'; skip until data arrives.
    # Outcome: BG engine keeps its previous state until sampler has latched evidence.
    if not sampled.data:
        return []
    out = JsonDataPack("sampled_evidence", "bg")
    # NlohmannJson does not support .items(); use .keys() + subscript.
    for k in sampled.data.keys():
        out.data[k] = sampled.data[k]
    return [out]
