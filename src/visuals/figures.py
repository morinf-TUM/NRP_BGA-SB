# visuals/figures.py
"""Static figure generators (light background, 1920×1080 @ 150 dpi)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend: no display required
import matplotlib.pyplot as plt
import numpy as np

from visuals.data_loader import (
    load_frequency_sweep,
    load_opensim_gonogo,
    load_perturbation_gonogo,
    load_perturbation_stopsignal,
)
from visuals.style import (
    FIG_SIZE_1080P,
    LIGHT_THEME,
    VERDICT_COLORS,
    apply_theme,
)

_FREQS = [5, 10, 20, 40, 80, 160]


# --- Helper ---

def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Figure 1: BG frequency threshold ---

def fig_frequency_threshold(output_dir: Path) -> Path:
    """Go-success rate vs. BG update frequency — 5→10 Hz threshold step."""
    apply_theme(LIGHT_THEME)
    data = load_frequency_sweep()
    opensim = load_opensim_gonogo()

    # Aggregate: go_nogo, low conflict, mean ± std across seeds
    by_freq: dict[float, list[float]] = defaultdict(list)
    for row in data:
        if row["paradigm"] == "go_nogo" and row["conflict_level"] == "low":
            by_freq[row["frequency_hz"]].append(row["go_success_rate"])

    freqs = sorted(by_freq)
    means = [np.mean(by_freq[f]) for f in freqs]
    stds  = [np.std(by_freq[f]) for f in freqs]

    fig, ax = plt.subplots(figsize=FIG_SIZE_1080P)

    # Shaded error band
    ax.fill_between(freqs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color="#3b82f6")

    # Main line
    ax.plot(freqs, means, "o-", color="#3b82f6", linewidth=2.5,
            markersize=8, label="Kinematic (go/no-go, low conflict)")

    # OpenSim overlay
    os_freqs = [r["frequency_hz"] for r in opensim]
    os_rates = [r["opensim_movement_onset_rate"] for r in opensim]
    ax.plot(os_freqs, os_rates, "s--", color="#f59e0b", linewidth=2,
            markersize=8, label="OpenSim Arm26")

    # Threshold annotation
    ax.axvline(x=10, color="#ef4444", linestyle=":", linewidth=1.5,
               label="Selection threshold (10 Hz)")

    ax.set_xscale("log")
    ax.set_xticks(freqs)
    ax.set_xticklabels([str(int(f)) for f in freqs])
    ax.set_xlabel("BG update frequency (Hz)", fontsize=14)
    ax.set_ylabel("Go-success rate", fontsize=14)
    ax.set_title("BG Update Frequency Governs Action Commitment", fontsize=16)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.4)

    return _save(fig, output_dir, "fig_frequency_threshold.png")


# --- Figure 2: Perturbation decomposition ---

def fig_perturbation_decomposition(output_dir: Path) -> Path:
    """4-panel figure: each perturbation type's effect on RT vs. channel selection."""
    apply_theme(LIGHT_THEME)
    gonogo = load_perturbation_gonogo()
    stopsig = load_perturbation_stopsignal()

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
    fig.suptitle("Timing Perturbations: RT vs. Channel Selection", fontsize=16, y=1.01)

    pert_types = [
        ("latency",      "Fixed Latency (ms)",       gonogo),
        ("jitter",       "Jitter Std (ms)",           gonogo),
        ("phase_offset", "Phase Offset (% period)",   gonogo),
        ("dropout",      "Dropout (%)",               stopsig),
    ]

    for (ptype, xlabel, dataset), ax in zip(pert_types, axes.flat):
        rows = [r for r in dataset
                if r["perturbation_type"] == ptype and r["frequency_hz"] == 40.0]
        rows.sort(key=lambda r: r["perturbation_value"])

        vals = [r["perturbation_value"] for r in rows]

        if ptype == "dropout":
            # stop_signal dataset: bg_commitment_latency_mean is None;
            # use go_rt_mean_s as the RT proxy for the dropout panel.
            # Trigger: perturbation_sweep_stopsignal.json does not populate
            #   bg_commitment_latency_mean for stop-signal paradigm rows.
            # Why: go_rt_mean_s carries the same conceptual information (mean
            #   go-trial RT) and is populated in the stop-signal dataset.
            # Outcome: RT normalisation succeeds without KeyError / None-division.
            rts_raw = [r["go_rt_mean_s"] for r in rows]
            rt0 = rts_raw[0] if (rts_raw and rts_raw[0] is not None and rts_raw[0] > 0) else 1.0
            rts_norm = [v / rt0 if v is not None else 0.0 for v in rts_raw]
            channel_metric = [r["stop_failure_rate"] for r in rows]
            channel_label = "Stop-failure rate"
        else:
            rts = [r["bg_commitment_latency_mean"] for r in rows]
            rt0 = rts[0] if (rts and rts[0] is not None and rts[0] > 0) else 1.0
            rts_norm = [v / rt0 if v is not None else 0.0 for v in rts]
            channel_metric = [r["go_success_rate"] for r in rows]
            channel_label = "Go-success rate"

        ax.plot(vals, rts_norm, "o-", color="#3b82f6", label="RT (normalised)", linewidth=2)
        ax.plot(vals, channel_metric, "s--", color="#f59e0b",
                label=channel_label, linewidth=2)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(ptype.replace("_", " ").title(), fontsize=12)
        ax.set_ylim(-0.05, 1.6)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    axes[0, 0].set_ylabel("Metric value", fontsize=11)
    axes[1, 0].set_ylabel("Metric value", fontsize=11)
    fig.tight_layout()

    return _save(fig, output_dir, "fig_perturbation_decomposition.png")


# --- Figure 3: Cerebellum learning curve ---

def fig_cerebellum_learning(cereb_trials: list[dict], output_dir: Path) -> Path:
    """Trial-by-trial endpoint deviation: filter-only adaptation under 30° rotation.

    Note on sign conventions: VisuomotorRotation uses rotation_deg=-30.0 (CCW
    convention, so -30° deflects the upward target to the upper-right quadrant).
    theta_hat decreases from 0 toward ~-0.52 rad (the counter-rotation angle)
    as the AdaptiveFilter learns. When converted to degrees, the values are
    negative; the axhline for the perturbation is drawn at -30°.
    """
    apply_theme(LIGHT_THEME)

    go_trials = [t for t in cereb_trials if t["is_go"]]
    trial_nums = list(range(1, len(go_trials) + 1))

    # Unperturbed reference endpoint is at (0, 1) — magnitude 1.0
    target = np.array([0.0, 1.0])

    deviations = []
    thetas = []
    for t in go_trials:
        ep = np.array(t["endpoint_xy"])
        deviations.append(float(np.linalg.norm(ep - target)))
        thetas.append(t["theta_hat"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_1080P)
    fig.suptitle("Cerebellar Adaptation Under 30° Visuomotor Rotation", fontsize=16)

    # Panel 1: endpoint deviation over trials
    ax1.plot(trial_nums, deviations, "o-", color="#ef4444", linewidth=2, markersize=5)
    ax1.axhline(0, color="#22c55e", linestyle="--", linewidth=1.5, label="Target (no deviation)")
    ax1.set_xlabel("Trial number (go trials only)", fontsize=13)
    ax1.set_ylabel("Endpoint deviation (a.u.)", fontsize=13)
    ax1.set_title("Endpoint deviation decreasing across trials", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.4)

    # Panel 2: theta_hat convergence toward -30° (the counter-rotation target).
    # rotation_deg=-30° so theta_hat converges toward -30° (not +30°).
    target_theta_deg = -30.0
    thetas_deg = [t * 180.0 / np.pi for t in thetas]
    ax2.plot(trial_nums, thetas_deg, "o-", color="#3b82f6", linewidth=2, markersize=5,
             label="θ̂ (learned counter-rotation)")
    ax2.axhline(target_theta_deg, color="#a855f7", linestyle="--", linewidth=1.5,
                label=f"Perturbation θ = {target_theta_deg:.0f}°")
    ax2.set_xlabel("Trial number (go trials only)", fontsize=13)
    ax2.set_ylabel("θ̂ (degrees)", fontsize=13)
    ax2.set_title("Adaptive filter converging toward −30°", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    return _save(fig, output_dir, "fig_cerebellum_learning.png")


# --- Figure 4: Three-interpretation verdict ---

def fig_three_interpretations(output_dir: Path) -> Path:
    """§11 verdict table as a visual summary card (light background)."""
    apply_theme(LIGHT_THEME)

    rows = [
        ("Selector bottleneck",
         "Wrong-channel choices\nrise at low frequency",
         "No wrong-channel selections\n(BG selects correctly or withholds)",
         "not_supported"),
        ("Urgency / commitment",
         "RT / vigor shift at low freq,\nchannel choice preserved",
         "✓ Latency/jitter shift RT\nwithout altering selected_channel",
         "supported"),
        ("Cancellation bottleneck",
         "Stop failures and SSRT\nworsen at low frequency",
         "✓ Flat inhibition (deterministic);\ndropout selectively impairs stopping",
         "supported"),
    ]
    col_headers = ["Account", "Prediction", "Observed", "Verdict"]

    fig, ax = plt.subplots(figsize=FIG_SIZE_1080P)
    ax.axis("off")
    fig.suptitle("Which Account Does the GPR Model Support?",
                 fontsize=18, fontweight="bold", y=0.96)

    col_widths = [0.22, 0.30, 0.35, 0.13]  # noqa: F841 — kept for readability
    x_positions = [0.02, 0.24, 0.54, 0.89]
    y_start = 0.80
    row_h = 0.16

    # Header
    for xi, header in zip(x_positions, col_headers):
        ax.text(xi, y_start + 0.04, header,
                fontsize=13, fontweight="bold", transform=ax.transAxes,
                va="bottom")

    # Divider line drawn in axes-coordinate space (transform=ax.transAxes).
    # axhline does not accept a transform kwarg; ax.plot with the transAxes
    # transform is the correct way to draw a line at a fixed axes-relative y.
    ax.plot([0.01, 0.99], [y_start + 0.02, y_start + 0.02],
            color="#aaaaaa", linewidth=1.0, transform=ax.transAxes)

    for i, (account, prediction, observed, verdict) in enumerate(rows):
        y = y_start - (i + 1) * row_h
        color = VERDICT_COLORS[verdict]
        verdict_sym = "✓" if verdict == "supported" else "✗"

        # Row background tint indicating verdict
        ax.add_patch(plt.Rectangle((0, y - 0.01), 1.0, row_h - 0.02,
                                    transform=ax.transAxes,
                                    color=color, alpha=0.06, linewidth=0))
        ax.text(x_positions[0], y + row_h / 2, account,
                fontsize=12, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(x_positions[1], y + row_h / 2, prediction,
                fontsize=11, transform=ax.transAxes, va="center")
        ax.text(x_positions[2], y + row_h / 2, observed,
                fontsize=11, transform=ax.transAxes, va="center")
        ax.text(x_positions[3], y + row_h / 2, verdict_sym,
                fontsize=22, color=color, fontweight="bold",
                transform=ax.transAxes, va="center", ha="center")

    return _save(fig, output_dir, "fig_three_interpretations.png")
