"""Four-knob ablation through NRPCoreSim. For each knob, sweep its frequency
with the other three pinned high (160 Hz) and measure go-success rate. Reuses
the four-knob config builder and offline scoring; no science reimplemented."""

from __future__ import annotations

import json
from pathlib import Path

from nrp.config_gen import build_config_four_knob
from nrp.run import run_trial
from nrp.score import trace_to_outcome

KNOBS = ("sampling", "integration", "emission", "commitment")
FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
HIGH = 160.0


def _rates_for(knob: str, hz: float) -> dict:
    rates = {"input_sampling_hz": HIGH, "integration_hz": HIGH,
             "output_emission_hz": HIGH, "commitment_hz": HIGH}
    rates[{"sampling": "input_sampling_hz", "integration": "integration_hz",
           "emission": "output_emission_hz", "commitment": "commitment_hz"}[knob]] = hz
    return rates


def ablate_knob(knob: str, frequencies: list[float], run_root: Path,
                n_seeds: int = 3) -> dict[float, float]:
    out: dict[float, float] = {}
    for hz in frequencies:
        released = 0
        cfg, overlay = build_config_four_knob(**_rates_for(knob, hz))
        for seed in range(n_seeds):
            params = {"trial_id": seed, "seed": seed, "cue_identity": "go", **overlay}
            trace = run_trial(cfg, params, run_root / f"{knob}_{hz}hz_s{seed}")
            if trace_to_outcome(trace)["motor_released"]:
                released += 1
        out[hz] = released / n_seeds
    return out


if __name__ == "__main__":
    # Per-trial run dirs stay under the gitignored nrp/run/; the final snapshot
    # lands in the committed nrp/results/ so the offline comparison can consume it.
    run_root = Path("nrp/run/ablation")
    results = {k: ablate_knob(k, FREQUENCIES_HZ, run_root) for k in KNOBS}
    result_path = Path("nrp/results/ablation.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2))
    print("go-success rate vs frequency, per ablated knob (others pinned 160 Hz):")
    for k in KNOBS:
        row = "  ".join(f"{hz:g}:{results[k][hz]:.2f}" for hz in FREQUENCIES_HZ)
        print(f"  {k:11s} {row}")
    print(f"saved -> {result_path}")
