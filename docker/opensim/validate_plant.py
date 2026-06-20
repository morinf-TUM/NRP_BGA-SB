"""Independent Arm26 plant validation (Task 10.2). Runs canonical full reaches and
checks smoothness, duration, bounded endpoint error, and determinism.

Usage: python validate_plant.py --config config.json --out report.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _arm26_plant import Arm26Plant


def _checks(plant: Arm26Plant, cfg: dict, ch: int) -> dict:
    target_xy = np.array(plant.endpoint_for(cfg["q_target"][ch]))
    traj = plant.simulate(ch, onset_time_ms=0.0, gate_gain=1.0, gate_state="open")
    pos = np.array(traj["positions_xy"])
    times = np.array(traj["times_ms"])
    start = pos[0]

    # progress toward target along the start->target direction
    direction = target_xy - start
    dist = float(np.linalg.norm(direction))
    unit = direction / dist if dist > 1e-9 else direction
    proj = (pos - start) @ unit
    # endpoint error
    endpoint_error = float(np.linalg.norm(pos[-1] - target_xy))
    # monotonicity of progress (smooth reach): allow tiny numerical dips
    monotone = bool(np.all(np.diff(proj) > -1e-3))
    # movement duration: time to reach 99% of progress
    reached = np.where(proj >= 0.99 * proj[-1])[0]
    duration_ms = float(times[reached[0]]) if len(reached) else float(times[-1])
    return {
        "channel": ch,
        "endpoint_error": endpoint_error,
        "monotone_progress": monotone,
        "duration_ms": duration_ms,
        "final_progress_fraction": float(proj[-1] / dist) if dist > 1e-9 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    plant = Arm26Plant(cfg)
    per_channel = [_checks(plant, cfg, ch) for ch in range(len(cfg["q_target"]))]

    # determinism: identical repeated reach -> identical positions
    a = plant.simulate(0, 0.0, 1.0, "open")["positions_xy"]
    b = plant.simulate(0, 0.0, 1.0, "open")["positions_xy"]
    deterministic = bool(np.allclose(np.array(a), np.array(b), atol=0.0))

    tol = cfg.get("endpoint_error_tol", 0.02)
    passed = (
        deterministic
        and all(c["monotone_progress"] for c in per_channel)
        and all(c["endpoint_error"] <= tol for c in per_channel)
        and all(c["final_progress_fraction"] >= 0.95 for c in per_channel)
    )
    report = {"passed": passed, "deterministic": deterministic,
              "endpoint_error_tol": tol, "per_channel": per_channel}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
