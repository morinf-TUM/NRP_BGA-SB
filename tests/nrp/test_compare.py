import json
from pathlib import Path

from nrp.compare import (
    KNOBS,
    load_nrp_ablation,
    load_nrp_gonogo_sweep,
    load_prototype_ablation,
    load_prototype_gonogo_sweep,
)

RESULTS = Path(__file__).resolve().parents[2] / "nrp" / "results"
PROTO = Path(__file__).resolve().parents[2] / "deprecated_toy_prototype_results"


def test_load_prototype_ablation_normalizes_and_maps(tmp_path):
    records = [
        {"condition": "baseline", "knob_name": "all", "freq_hz": 160.0, "miss_rate": 0.0},
        {"condition": "sweep", "knob_name": "input_sampling_hz", "freq_hz": 5.0, "miss_rate": 1.0},
        {"condition": "sweep", "knob_name": "integration_step_hz", "freq_hz": 5.0, "miss_rate": 0.0},
    ]
    p = tmp_path / "ab.json"
    p.write_text(json.dumps(records))
    out = load_prototype_ablation(p)
    # 'all' baseline excluded; miss_rate -> go_success_rate; names mapped.
    assert "all" not in out
    assert out["sampling"][5.0] == 0.0
    assert out["integration"][5.0] == 1.0


def test_load_nrp_ablation_coerces_str_hz_keys():
    out = load_nrp_ablation(RESULTS / "ablation.json")
    assert out["sampling"][5.0] == 0.0
    assert out["integration"][5.0] == 1.0
    assert set(out) == set(KNOBS)


def test_load_prototype_gonogo_sweep_filters_and_aggregates(tmp_path):
    records = [
        {"paradigm": "go_nogo", "frequency_hz": 10.0, "go_success_rate": 0.0},
        {"paradigm": "go_nogo", "frequency_hz": 10.0, "go_success_rate": 1.0},
        {"paradigm": "two_choice", "frequency_hz": 10.0, "go_success_rate": 0.0},
    ]
    p = tmp_path / "fs.json"
    p.write_text(json.dumps(records))
    out = load_prototype_gonogo_sweep(p)
    # two_choice excluded; go_nogo rates averaged: (0.0 + 1.0) / 2 = 0.5
    assert out == {10.0: 0.5}


def test_load_nrp_gonogo_sweep_coerces_str_hz_keys():
    out = load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json")
    assert out[5.0] == 0.0
    assert out[10.0] == 1.0


def test_committed_snapshots_lock_key_divergence():
    """The committed nrp snapshots must preserve the headline divergence:
    sampling misses at 5 Hz, integration does NOT (idempotent sub-step)."""
    ablation = json.loads((RESULTS / "ablation.json").read_text())
    assert ablation["sampling"]["5.0"] == 0.0
    assert ablation["integration"]["5.0"] == 1.0
    sweep = json.loads((RESULTS / "gonogo_sweep.json").read_text())
    assert sweep["5.0"] == 0.0
    assert sweep["10.0"] == 1.0


from nrp.compare import (
    align_series,
    classify_regime,
    compare_ablation,
)


def test_classify_regime_threshold():
    assert classify_regime(1.0) == "success"
    assert classify_regime(0.51) == "success"
    assert classify_regime(0.5) == "miss"
    assert classify_regime(0.0) == "miss"


def test_align_series_tags_membership():
    rows = align_series({5.0: 0.0, 10.0: 1.0}, {10.0: 1.0, 20.0: 1.0})
    by_hz = {r.freq_hz: r for r in rows}
    assert by_hz[5.0].tag == "proto_only"
    assert by_hz[10.0].tag == "common"
    assert by_hz[20.0].tag == "nrp_only"
    # sorted by frequency
    assert [r.freq_hz for r in rows] == [5.0, 10.0, 20.0]


def test_compare_ablation_flags_integration_divergence():
    proto = load_prototype_ablation(PROTO / "ablation_frequency_v2.json")
    nrp = load_nrp_ablation(RESULTS / "ablation.json")
    verdict = compare_ablation(proto, nrp)
    by_knob = {kv.knob: kv for kv in verdict.knobs}
    assert by_knob["sampling"].holds is True
    assert by_knob["emission"].holds is True
    assert by_knob["commitment"].holds is True
    assert by_knob["integration"].holds is False
    assert by_knob["integration"].divergent_freqs == [5.0]
    assert verdict.holds is False


from nrp.compare import compare_frequency_sweep


def test_compare_frequency_sweep_reports_onsets_with_caveat():
    proto = load_prototype_gonogo_sweep(PROTO / "frequency_sweep_results.json")
    nrp = load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json")
    verdict = compare_frequency_sweep(proto, nrp)
    # nrp onset is 10 Hz (5 Hz misses). Prototype onset is inflated to 20 Hz by
    # conflict-level aggregation (10 Hz aggregate = 0.333 -> miss regime).
    assert verdict.nrp_onset_hz == 10.0
    assert verdict.proto_onset_hz == 20.0
    assert "conflict" in verdict.caveat.lower()


def test_compare_frequency_sweep_onset_none_when_all_miss():
    verdict = compare_frequency_sweep({10.0: 0.0, 20.0: 0.0}, {10.0: 0.0})
    assert verdict.proto_onset_hz is None
    assert verdict.nrp_onset_hz is None
