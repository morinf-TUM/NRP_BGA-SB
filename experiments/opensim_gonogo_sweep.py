"""Phase 10 / Task 10.4: go/no-go BG-frequency sweep on the OpenSim Arm26 plant.

Runs the existing closed-loop go/no-go pipeline on the host, ships each trial's
reduced ReachSpec to the OpenSim container, scores the returned hand trajectories
with compute_movement_metrics, and compares against the kinematic reacher run on
the SAME BG decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.movement_metrics import compute_movement_metrics
from nrp_bga_sb.opensim_plant import (
    OpenSimPlantClient,
    OpenSimPlantConfig,
    extract_reach_spec,
)
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig, ReacherTrajectory
from nrp_bga_sb.scheduler import FrequencyConfig

FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0]
_SEED = 12345
_TOTAL_DURATION_MS = 1300.0


def _gonogo_trials(freq_hz: float, n_trials: int):
    """Run the closed-loop go/no-go pipeline at one BG frequency (host side)."""
    freq_cfg = FrequencyConfig.from_effective_hz(freq_hz)
    # default accumulation_ms=200, cortical rise 100ms
    policy = make_closed_loop_policy(frequency_config=freq_cfg)
    config = GoNoGoConfig(
        n_trials=n_trials, go_probability=0.7,
        response_window_start_ms=0, response_window_duration_ms=600,
        fixation_duration_ms=200, cue_onset_ms=400, decision_point_ms=300, seed=_SEED,
    )
    return run_go_nogo_trials(config, policy)


def run_opensim_gonogo_condition(freq_hz: float, n_trials: int,
                                 client: OpenSimPlantClient) -> dict:
    """Run one BG-frequency condition end-to-end through the OpenSim plant and
    compare against the kinematic reacher on the SAME BG decisions."""
    trials = _gonogo_trials(freq_hz, n_trials)
    go_trials = [t for t in trials if t.cue_identity == "go"]

    def onset_ms_of(trial):
        return (trial.movement_onset_time * 1000.0
                if trial.movement_onset_time is not None else None)

    specs = [extract_reach_spec(t.motor_command_series, onset_ms_of(t), str(t.trial_id),
                                n_channels=len(client.config.q_target))
             for t in trials]
    trajs, target_endpoints_xy = client.run(specs)
    traj_by_id = {ot.trial_id: ot for ot in trajs}

    # OpenSim metrics over GO trials (endpoint_error in the arm's own frame via FK targets)
    rcfg = ReacherConfig(n_channels=2, target_positions=target_endpoints_xy)
    osim_onset = 0
    osim_errs = []
    for t in go_trials:
        ot = traj_by_id[str(t.trial_id)]
        rt = ReacherTrajectory(**ot.model_dump(exclude={"trial_id"}))
        m = compute_movement_metrics(rt, rcfg)
        if m.movement_onset_time_ms is not None:
            osim_onset += 1
            osim_errs.append(m.endpoint_error)

    # Kinematic reacher on the SAME trials (default 2D targets) for comparison
    kin = KinematicReacher(ReacherConfig())
    kin_cfg = ReacherConfig()
    kin_onset = 0
    for t in go_trials:
        ktraj = kin.simulate(t.motor_command_series, onset_ms_of(t), _TOTAL_DURATION_MS)
        km = compute_movement_metrics(ktraj, kin_cfg)
        if km.movement_onset_time_ms is not None:
            kin_onset += 1

    n_go = len(go_trials)
    return {
        "frequency_hz": freq_hz,
        "n_trials": len(trials),
        "n_go_trials": n_go,
        "opensim_movement_onset_rate": osim_onset / n_go if n_go else 0.0,
        "kinematic_movement_onset_rate": kin_onset / n_go if n_go else 0.0,
        "opensim_mean_endpoint_error": sum(osim_errs) / len(osim_errs) if osim_errs else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="nrp-bga-opensim:4.6")
    ap.add_argument("--io-dir", default="data/opensim_io")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()

    cfg = OpenSimPlantConfig(
        image=args.image, q0=[0.0, 0.35], q_target=[[1.2, 1.0], [0.8, 1.6]],
        kp=[120.0, 90.0], kd=[18.0, 14.0], dt_ms=2.0,
        movement_duration_ms=300.0, total_duration_ms=_TOTAL_DURATION_MS,
    )
    client = OpenSimPlantClient(cfg, io_dir=args.io_dir)
    results = [run_opensim_gonogo_condition(f, args.n_trials, client) for f in FREQUENCIES_HZ]

    out = Path("deprecated_toy_prototype_results/opensim_gonogo_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    _print_report(results)


def _print_report(results: list[dict]) -> None:
    print("\n=== Phase 10 OpenSim go/no-go frequency sweep (M8) ===")
    print(f"{'freq_hz':>8} {'osim_onset':>11} {'kin_onset':>10} {'osim_err':>9}")
    for r in results:
        print(f"{r['frequency_hz']:>8.1f} {r['opensim_movement_onset_rate']:>11.3f} "
              f"{r['kinematic_movement_onset_rate']:>10.3f} "
              f"{r['opensim_mean_endpoint_error']:>9.4f}")
    low = next(r for r in results if r["frequency_hz"] == 5.0)
    high = next(r for r in results if r["frequency_hz"] == 40.0)
    ok = low["opensim_movement_onset_rate"] < 0.5 < high["opensim_movement_onset_rate"]
    print(f"\nM8 qualitative effect preserved in OpenSim: {ok} "
          f"(5 Hz onset={low['opensim_movement_onset_rate']:.3f}, "
          f"40 Hz onset={high['opensim_movement_onset_rate']:.3f})")


if __name__ == "__main__":
    main()
