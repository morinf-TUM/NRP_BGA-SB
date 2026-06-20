"""Batch reach simulator (in-container). Task 10.2.

Usage: python run_plant.py --config config.json --jobs jobs.json --out trajectories.json
"""
from __future__ import annotations

import argparse
import json

from _arm26_plant import Arm26Plant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.jobs) as f:
        jobs = json.load(f)["jobs"]

    plant = Arm26Plant(cfg)
    target_endpoints_xy = [plant.endpoint_for(q) for q in cfg["q_target"]]

    trajectories = []
    for job in jobs:
        traj = plant.simulate(
            selected_channel=job["selected_channel"],
            onset_time_ms=job["onset_time_ms"],
            gate_gain=job["gate_gain"],
            gate_state=job["gate_state"],
        )
        traj["trial_id"] = job["trial_id"]
        trajectories.append(traj)

    with open(args.out, "w") as f:
        json.dump({"target_endpoints_xy": target_endpoints_xy,
                   "trajectories": trajectories}, f)


if __name__ == "__main__":
    main()
