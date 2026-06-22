"""Phase 11b: OpenSim cerebellar correction confirmation (Milestone M9, embodied).

Mirrors opensim_gonogo_sweep.py (Phase 10) but adds a visuomotor-rotation
perturbation and cerebellar on/off comparison through the OpenSim Arm26 plant.

Integration design (host-side endpoint-space application):
  - The OpenSim container returns the physical trajectory (unperturbed dynamics).
  - VisuomotorRotation is applied to the returned endpoint to simulate what the
    visual system perceives after the angular distortion.
  - AdaptiveFilter.update() is called with the angular error after each executed
    movement, driving theta_hat -> theta across trials.
  - The learned counter-rotation (-theta_hat) is applied post-hoc to the visual
    endpoint to measure the cerebellar-corrected accuracy.
  - BG-effect guard: closed-gate (missed) trials never update the filter, so the
    movement-onset-rate-vs-frequency signature is structurally unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nrp_bga_sb.cerebellum import AdaptiveFilter
from nrp_bga_sb.closed_loop import make_closed_loop_policy
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig, run_go_nogo_trials
from nrp_bga_sb.opensim_plant import (
    OpenSimPlantClient,
    OpenSimPlantConfig,
    extract_reach_spec,
)
from nrp_bga_sb.perturbation_plant import VisuomotorRotation, rotate_xy, signed_angle
from nrp_bga_sb.reacher import KinematicReacher, ReacherConfig
from nrp_bga_sb.scheduler import FrequencyConfig

FREQUENCIES_HZ = [5.0, 10.0, 20.0, 40.0, 80.0]
_SEED = 12345
_TOTAL_DURATION_MS = 1300.0


def _gonogo_trials(freq_hz: float, n_trials: int):
    """Run the closed-loop go/no-go pipeline at one BG frequency (host side)."""
    freq_cfg = FrequencyConfig.from_effective_hz(freq_hz)
    policy = make_closed_loop_policy(frequency_config=freq_cfg)
    config = GoNoGoConfig(
        n_trials=n_trials, go_probability=0.7,
        response_window_start_ms=0, response_window_duration_ms=600,
        fixation_duration_ms=200, cue_onset_ms=400, decision_point_ms=300, seed=_SEED,
    )
    return run_go_nogo_trials(config, policy)


# --- Condition runner ---


def run_opensim_cerebellum_condition(
    freq_hz: float,
    n_trials: int,
    client: OpenSimPlantClient,
    cerebellum_enabled: bool = True,
    perturbation_deg: float = 30.0,
) -> dict:
    """Run one BG-frequency condition through the OpenSim plant under visuomotor
    perturbation, with or without cerebellar adaptation.

    Perturbation and correction are both applied in Cartesian endpoint space on the
    host, post-hoc on the trajectory returned by the container.  The OpenSim plant
    is unaware of the rotation: it drives joints to q_target as usual.
    """
    trials = _gonogo_trials(freq_hz, n_trials)

    def onset_ms_of(trial):
        return (trial.movement_onset_time * 1000.0
                if trial.movement_onset_time is not None else None)

    specs = [extract_reach_spec(t.motor_command_series, onset_ms_of(t), str(t.trial_id),
                                n_channels=len(client.config.q_target))
             for t in trials]
    trajs, target_endpoints_xy = client.run(specs)
    traj_by_id = {ot.trial_id: ot for ot in trajs}

    perturbation = VisuomotorRotation(rotation_deg=perturbation_deg)
    # Trigger: cerebellum_enabled=False.
    # Why: allow the same code path to run with and without adaptation for the
    #      on/off comparison.  None means no learning and no correction.
    # Outcome: filter stays at theta_hat=0, p_corrected=p_perturbed (no correction).
    adaptive_filter = AdaptiveFilter() if cerebellum_enabled else None

    go_trials = [t for t in trials if t.cue_identity == "go"]
    n_go = len(go_trials)

    deviations: list[float] = []
    n_move = 0

    for trial in go_trials:
        ot = traj_by_id[str(trial.trial_id)]

        # --- BG-effect guard ---
        # Trigger: closed gate (5 Hz miss, thalamus never opened).
        # Why: the cerebellar corrector must not update on missed trials; doing so
        #      would alter the onset-rate-vs-frequency signature and break the
        #      BG-effect invariant (§27.4).
        # Outcome: skip this trial; filter and onset counter unchanged.
        if ot.onset_time_ms is None or ot.selected_channel < 0:
            continue

        n_move += 1
        ch = ot.selected_channel
        desired = list(target_endpoints_xy[ch])
        p_physical = list(ot.positions_xy[-1])

        # Apply the visual distortion to the physical endpoint.
        p_perturbed = perturbation.apply(p_physical)

        if adaptive_filter is not None:
            # Apply the learned counter-rotation to measure corrected accuracy.
            # Learning uses the open-loop perturbed error (before counter-rotation)
            # so it reflects the true residual rotation, not the corrected one.
            p_corrected = rotate_xy(p_perturbed, -adaptive_filter.theta_hat)
            adaptive_filter.update(signed_angle(desired, p_perturbed))
        else:
            p_corrected = p_perturbed

        dev = float(np.linalg.norm(np.array(p_corrected) - np.array(desired)))
        deviations.append(dev)

    # Kinematic reacher on the SAME BG decisions for comparison (no perturbation).
    kin = KinematicReacher(ReacherConfig())
    kin_onset = 0
    for trial in go_trials:
        ktraj = kin.simulate(trial.motor_command_series, onset_ms_of(trial), _TOTAL_DURATION_MS)
        if ktraj.onset_time_ms is not None:
            kin_onset += 1

    return {
        "frequency_hz": freq_hz,
        "n_trials": n_trials,
        "n_go_trials": n_go,
        "opensim_movement_onset_rate": n_move / n_go if n_go else 0.0,
        "kinematic_movement_onset_rate": kin_onset / n_go if n_go else 0.0,
        "opensim_mean_endpoint_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "cerebellum_enabled": cerebellum_enabled,
        "perturbation_deg": perturbation_deg,
        "final_theta_hat": adaptive_filter.theta_hat if adaptive_filter is not None else 0.0,
    }


# --- Experiment entry point ---


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="nrp-bga-opensim:4.6")
    ap.add_argument("--io-dir", default="data/opensim_io")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--perturbation-deg", type=float, default=30.0)
    args = ap.parse_args()

    cfg = OpenSimPlantConfig(
        image=args.image, q0=[0.0, 0.35], q_target=[[1.2, 1.0], [0.8, 1.6]],
        kp=[120.0, 90.0], kd=[18.0, 14.0], dt_ms=2.0,
        movement_duration_ms=300.0, total_duration_ms=_TOTAL_DURATION_MS,
    )
    client = OpenSimPlantClient(cfg, io_dir=args.io_dir)
    results = []
    for freq in FREQUENCIES_HZ:
        for cb_on in (False, True):
            results.append(run_opensim_cerebellum_condition(
                freq, args.n_trials, client,
                cerebellum_enabled=cb_on,
                perturbation_deg=args.perturbation_deg,
            ))

    out = Path("deprecated_toy_prototype_results/opensim_cerebellum_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    _print_report(results)


def _print_report(results: list[dict]) -> None:
    print("\n=== Phase 11b OpenSim cerebellar correction (M9 embodied) ===")
    print(f"{'freq_hz':>8} {'cb':>4} {'osim_onset':>11} {'kin_onset':>10} "
          f"{'dev':>9} {'theta_hat':>10}")
    for r in results:
        print(f"{r['frequency_hz']:>8.1f} {str(r['cerebellum_enabled']):>4} "
              f"{r['opensim_movement_onset_rate']:>11.3f} "
              f"{r['kinematic_movement_onset_rate']:>10.3f} "
              f"{r['opensim_mean_endpoint_deviation']:>9.4f} "
              f"{r['final_theta_hat']:>10.4f}")
    # Qualitative checks (from paired off/on rows)
    pairs = [(results[i], results[i + 1]) for i in range(0, len(results), 2)]
    for off, on in pairs:
        freq = off["frequency_hz"]
        onset_ok = off["opensim_movement_onset_rate"] == on["opensim_movement_onset_rate"]
        dev_ok = (on["opensim_movement_onset_rate"] == 0.0
                  or on["opensim_mean_endpoint_deviation"] < off["opensim_mean_endpoint_deviation"])
        print(f"  {freq:.0f} Hz: onset unchanged={onset_ok}, dev↓={dev_ok}")


if __name__ == "__main__":
    main()
