import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "nrp" / "results"


def test_committed_snapshots_lock_key_divergence():
    """The committed nrp snapshots must preserve the headline divergence:
    sampling misses at 5 Hz, integration does NOT (idempotent sub-step)."""
    ablation = json.loads((RESULTS / "ablation.json").read_text())
    assert ablation["sampling"]["5.0"] == 0.0
    assert ablation["integration"]["5.0"] == 1.0
    sweep = json.loads((RESULTS / "gonogo_sweep.json").read_text())
    assert sweep["5.0"] == 0.0
    assert sweep["10.0"] == 1.0
