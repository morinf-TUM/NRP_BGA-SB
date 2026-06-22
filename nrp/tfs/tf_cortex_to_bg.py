"""Phase 1-2 link: forward cortical evidence to the BG engine each step. In
Phase 3 this is replaced by tf_cortex_to_sampler + tf_sampler_to_bg."""

from nrp_core import *
from nrp_core.data.nrp_json import *


@EngineDataPack(keyword="evidence", id=DataPackIdentifier("evidence", "cortex"))
@TransceiverFunction("bg")
def cortex_to_bg(evidence):
    out = JsonDataPack("sampled_evidence", "bg")
    for k, v in evidence.data.items():
        out.data[k] = v
    return [out]
