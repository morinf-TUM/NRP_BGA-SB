"""NRP go/no-go frequency sweep. Runs one NRPCoreSim trial per (frequency, seed),
scores each trace offline, and reports go-success rate vs BG frequency. Reuses
nrp.config_gen / nrp.run / nrp.score; no science is reimplemented here."""

from __future__ import annotations

import json
from pathlib import Path

from nrp.config_gen import build_config
from nrp.run import run_trial
from nrp.score import trace_to_outcome

FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]


def run_sweep(frequencies: list[float], n_seeds: int, run_root: Path) -> dict[float, float]:
    rates: dict[float, float] = {}
    for hz in frequencies:
        released = 0
        for seed in range(n_seeds):
            params = {"trial_id": seed, "seed": seed, "cue_identity": "go"}
            trace = run_trial(build_config(hz), params, run_root / f"{hz}hz_s{seed}")
            if trace_to_outcome(trace)["motor_released"]:
                released += 1
        rates[hz] = released / n_seeds
    return rates


if __name__ == "__main__":
    out_root = Path("nrp/run/gonogo_sweep")
    rates = run_sweep(FREQUENCIES_HZ, n_seeds=5, run_root=out_root)
    result_path = Path("nrp/run/nrp_gonogo_sweep.json")
    result_path.write_text(json.dumps(rates, indent=2))
    print("go-success rate vs BG frequency (Hz):")
    for hz in FREQUENCIES_HZ:
        print(f"  {hz:6.1f} Hz : {rates[hz]:.3f}")
    print(f"saved -> {result_path}")
