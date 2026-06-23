"""Integration knob (knob 2) nrp signature: first-release time vs integration rate.

The go-success ablation cannot show this knob: nrp judges the motor gate as released
if it EVER opens during the 300 ms sim, so even a slow integrator settles and releases
-- just later. Release LATENCY is the dissociating signal. A slower integration rate
settles later (PROJECT_MEMORY §15.7 / design doc Revision 2)."""

from __future__ import annotations

import json
from pathlib import Path

from nrp.config_gen import build_config_four_knob
from nrp.run import run_trial

HIGH = 160.0
FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]


def first_release_index(trace: list[dict]) -> int | None:
    """Index of the first trace record whose motor gate is open/partial with a
    non-zero command (same release rule as nrp/score.py). The thalamus logs one
    record per ms, so the index is a millisecond proxy for release latency."""
    for i, rec in enumerate(trace):
        m = rec.get("motor")
        if m and m.get("gate_state") in ("open", "partial") and any(m.get("command", [])):
            return i
    return None


def measure(integration_hz: float, run_root: Path, seed: int = 0) -> int | None:
    """Run one go trial with only the integration rate varied (others pinned 160 Hz)
    and return the first-release record index (~ms), or None if never released."""
    cfg, overlay = build_config_four_knob(
        input_sampling_hz=HIGH, integration_hz=integration_hz,
        output_emission_hz=HIGH, commitment_hz=HIGH)
    params = {"trial_id": seed, "seed": seed, "cue_identity": "go", **overlay}
    trace = run_trial(cfg, params, run_root / f"int{integration_hz}hz_s{seed}")
    return first_release_index(trace)


if __name__ == "__main__":
    run_root = Path("nrp/run/integration_latency")
    results = {hz: measure(hz, run_root) for hz in FREQUENCIES_HZ}
    out_path = Path("nrp/results/integration_latency.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({str(hz): results[hz] for hz in FREQUENCIES_HZ}, indent=2))
    print("first-release record index (~ms) vs integration rate (others pinned 160 Hz):")
    for hz in FREQUENCIES_HZ:
        print(f"  {hz:6g} Hz -> {results[hz]}")
    print(f"saved -> {out_path}")
