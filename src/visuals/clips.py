# visuals/clips.py
"""Clip frame-sequence generators.

Each writer generates numbered PNGs in a directory. dry_run=True returns
the expected frame count without writing any files (used for smoke checks).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from visuals.style import (
    DARK_THEME,
    FREQ_COLORS,
    FIG_SIZE_1080P,
    VERDICT_COLORS,
    apply_theme,
)

FPS = 24
_BG = "#0d1117"     # figure background colour
_FG = "#e6edf3"     # text and axis colour


# --- Internal helpers ---

def _new_dark_fig() -> tuple[plt.Figure, plt.Axes]:
    """Return a 1080p figure + single axes with the dark theme applied."""
    apply_theme(DARK_THEME)
    fig, ax = plt.subplots(figsize=FIG_SIZE_1080P)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    return fig, ax


def _save_frame(fig: plt.Figure, frames_dir: Path, index: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(frames_dir / f"{index:04d}.png", dpi=96, bbox_inches="tight",
                facecolor=_BG)
    plt.close(fig)


def _draw_glow_arc(ax: plt.Axes,
                   xs: list[float],
                   ys: list[float],
                   color: str,
                   alpha: float = 1.0) -> None:
    """Draw a trajectory arc with a soft bloom/glow effect.

    Three passes: wide+faint outer glow, medium mid-glow, sharp core line.
    """
    for lw, a in [(10, 0.06 * alpha), (5, 0.14 * alpha), (1.5, alpha)]:
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=a,
                solid_capstyle="round")


def _title_card_frames(title_text: str, frames_dir: Path | None,
                       n_frames: int, dry_run: bool, start_index: int) -> int:
    """Write a full-screen title-card block and return its frame count."""
    if dry_run:
        return n_frames
    for i in range(n_frames):
        fig, ax = _new_dark_fig()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        alpha = min(1.0, i / max(1, FPS * 0.5))   # 0.5 s fade-in
        ax.text(0.5, 0.5, title_text, color=_FG, fontsize=24,
                ha="center", va="center", alpha=alpha,
                transform=ax.transAxes, fontweight="bold", wrap=True)
        _save_frame(fig, frames_dir, start_index + i)
    return n_frames


# --- Clip 1: Frequency threshold ---

_THRESHOLD_FRAMES_PER_FREQ = 72    # 3 s per frequency
_THRESHOLD_TITLE_FRAMES    = 48    # 2 s title card
_THRESHOLD_CAPTION_FRAMES  = 48    # 2 s final caption


def write_threshold_frames(
    trials: list[dict],
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Animate arm attempts at 5→80 Hz; MISS (5 Hz) then HIT (≥10 Hz)."""
    total = (_THRESHOLD_TITLE_FRAMES
             + len(trials) * _THRESHOLD_FRAMES_PER_FREQ
             + _THRESHOLD_CAPTION_FRAMES)
    if dry_run:
        return total

    idx = 0
    # Title card
    idx += _title_card_frames(
        "BG update frequency governs action commitment",
        frames_dir, _THRESHOLD_TITLE_FRAMES, dry_run=False, start_index=idx,
    )

    target_x, target_y = 0.0, 1.0
    hold   = FPS // 2         # 0.5 s hold before arc
    arc_f  = FPS * 2          # 2 s arc draw
    result = FPS // 2         # 0.5 s result label hold
    # hold + arc_f + result = 12 + 48 + 12 = 72 = _THRESHOLD_FRAMES_PER_FREQ

    for trial in trials:
        freq     = trial["frequency_hz"]
        color    = FREQ_COLORS.get(int(freq), "#e6edf3")
        hit      = not trial["gate_closed"]
        positions = trial["positions_xy"]

        # Hold: show target + frequency label only
        for i in range(hold):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.2, 1.6)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=200, zorder=5)
            ax.text(0.05, 0.95, f"{int(freq)} Hz", color=color, fontsize=20,
                    fontweight="bold", transform=ax.transAxes, va="top")
            _save_frame(fig, frames_dir, idx)
            idx += 1

        # Arc draw: progressively reveal the trajectory
        for i in range(arc_f):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.2, 1.6)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=200, zorder=5)
            ax.text(0.05, 0.95, f"{int(freq)} Hz", color=color, fontsize=20,
                    fontweight="bold", transform=ax.transAxes, va="top")
            if hit and positions:
                reveal = max(2, int((i + 1) / arc_f * len(positions)))
                xs = [p[0] for p in positions[:reveal]]
                ys = [p[1] for p in positions[:reveal]]
                _draw_glow_arc(ax, xs, ys, color)
                # Hand dot
                ax.scatter([xs[-1]], [ys[-1]], color=color, s=120, zorder=6)
            _save_frame(fig, frames_dir, idx)
            idx += 1

        # Result label
        label = "✓ HIT" if hit else "✗ MISS"
        label_color = "#22c55e" if hit else "#ef4444"
        for i in range(result):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.2, 1.6)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=200, zorder=5)
            ax.text(0.05, 0.95, f"{int(freq)} Hz", color=color, fontsize=20,
                    fontweight="bold", transform=ax.transAxes, va="top")
            if hit and positions:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                _draw_glow_arc(ax, xs, ys, color, alpha=0.7)
            ax.text(0.5, 0.15, label, color=label_color, fontsize=22,
                    fontweight="bold", ha="center", transform=ax.transAxes)
            _save_frame(fig, frames_dir, idx)
            idx += 1

    # Caption
    for i in range(_THRESHOLD_CAPTION_FRAMES):
        fig, ax = _new_dark_fig()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.2, 1.6)
        ax.axis("off")
        alpha = min(1.0, i / (FPS * 0.5))
        ax.text(0.5, 0.5,
                "Below 10 Hz the BG samples only\nneutral cortical evidence",
                color=_FG, fontsize=18, ha="center", va="center",
                transform=ax.transAxes, alpha=alpha)
        _save_frame(fig, frames_dir, idx)
        idx += 1

    return total


# --- Clip 2: Cerebellar adaptation ---

_CEREB_TITLE_FRAMES     = 48    # 2 s
_CEREB_FRAMES_PER_TRIAL = 20    # ~8.3 ms wall-clock per trial at 24 fps


def write_cerebellum_frames(
    trials: list[dict],
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Animate 30 go-trials showing arc rotating toward target as theta_hat builds."""
    go_trials = [t for t in trials if t["is_go"]]
    total = _CEREB_TITLE_FRAMES + len(go_trials) * _CEREB_FRAMES_PER_TRIAL
    if dry_run:
        return total

    idx = 0
    idx += _title_card_frames(
        "Cerebellum corrects visuomotor rotation across trials",
        frames_dir, _CEREB_TITLE_FRAMES, dry_run=False, start_index=idx,
    )

    target_x, target_y = 0.0, 1.0
    # Colour gradient: red (early, deflected) → green (late, corrected)
    n = len(go_trials)

    for ti, trial in enumerate(go_trials):
        frac   = ti / max(1, n - 1)       # 0.0 → 1.0 across trials
        r = int(239 * (1 - frac) + 34 * frac)
        g = int(68  * (1 - frac) + 197 * frac)
        b = int(68  * (1 - frac) + 94  * frac)
        color  = f"#{r:02x}{g:02x}{b:02x}"
        positions = trial["positions_xy"]
        theta_deg = trial["theta_hat"] * 180.0 / np.pi

        arc_f  = int(_CEREB_FRAMES_PER_TRIAL * 0.7)    # 70% drawing arc
        hold_f = _CEREB_FRAMES_PER_TRIAL - arc_f       # 30% hold

        for i in range(arc_f):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-0.2, 1.4)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=180, zorder=5,
                       label="Target")
            reveal = max(2, int((i + 1) / arc_f * len(positions)))
            xs = [p[0] for p in positions[:reveal]]
            ys = [p[1] for p in positions[:reveal]]
            _draw_glow_arc(ax, xs, ys, color)
            ax.scatter([xs[-1]], [ys[-1]], color=color, s=100, zorder=6)
            ax.text(0.05, 0.95, f"Trial {ti + 1}", color=_FG, fontsize=16,
                    fontweight="bold", transform=ax.transAxes, va="top")
            ax.text(0.05, 0.88, f"θ̂ = {theta_deg:.1f}°", color="#a855f7",
                    fontsize=13, transform=ax.transAxes, va="top")
            _save_frame(fig, frames_dir, idx)
            idx += 1

        for i in range(hold_f):
            fig, ax = _new_dark_fig()
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-0.2, 1.4)
            ax.axis("off")
            ax.scatter([target_x], [target_y], color="#e6edf3", s=180, zorder=5)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            _draw_glow_arc(ax, xs, ys, color, alpha=0.6)
            ax.text(0.05, 0.95, f"Trial {ti + 1}", color=_FG, fontsize=16,
                    fontweight="bold", transform=ax.transAxes, va="top")
            ax.text(0.05, 0.88, f"θ̂ = {theta_deg:.1f}°", color="#a855f7",
                    fontsize=13, transform=ax.transAxes, va="top")
            _save_frame(fig, frames_dir, idx)
            idx += 1

    return total


# --- Clip 3: Perturbation decomposition ---

_PERT_TITLE_FRAMES  = 48
_PERT_DRAW_FRAMES   = 60    # per cell line draw
_PERT_LABEL_FRAMES  = 36    # label hold per cell
_PERT_HOLD_FRAMES   = 96    # final hold

_PERT_TYPES = [
    ("latency",      "Fixed latency (ms)",     "go_success_rate",   "Go-success rate",   False),
    ("jitter",       "Jitter std (ms)",         "go_success_rate",   "Go-success rate",   False),
    ("phase_offset", "Phase offset (% period)", "go_success_rate",   "Go-success rate",   False),
    ("dropout",      "Dropout (%)",             "stop_failure_rate", "Stop-failure rate", True),
]


def write_perturbation_frames(
    gonogo: list[dict],
    stopsig: list[dict],
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Animated 2×2 grid: each perturbation's RT vs. channel-selection metric."""
    n_cells = len(_PERT_TYPES)
    total = (_PERT_TITLE_FRAMES
             + n_cells * (_PERT_DRAW_FRAMES + _PERT_LABEL_FRAMES)
             + _PERT_HOLD_FRAMES)
    if dry_run:
        return total

    idx = 0
    idx += _title_card_frames(
        "Timing noise vs. signal integrity",
        frames_dir, _PERT_TITLE_FRAMES, dry_run=False, start_index=idx,
    )

    # Pre-compute cell data
    cells = []
    for ptype, xlabel, channel_key, channel_label, use_stopsig in _PERT_TYPES:
        dataset = stopsig if use_stopsig else gonogo
        rows = [r for r in dataset
                if r["perturbation_type"] == ptype and r["frequency_hz"] == 40.0]
        rows.sort(key=lambda r: r["perturbation_value"])
        vals = [r["perturbation_value"] for r in rows]
        rts  = [r["bg_commitment_latency_mean"] for r in rows]
        rt0  = rts[0] if rts[0] > 0 else 1e-6
        rts_norm = [v / rt0 for v in rts]
        ch = [r.get(channel_key, 0.0) or 0.0 for r in rows]
        label = ("urgency account" if ptype != "dropout"
                 else "cancellation bottleneck")
        cells.append((ptype, xlabel, vals, rts_norm, ch, channel_label, label, use_stopsig))

    def _draw_base(ax_list: list, reveal_cell: int, reveal_frac: float,
                   label_cells: set[int]) -> None:
        apply_theme(DARK_THEME)
        for ci, (ptype, xlabel, vals, rts_norm, ch, ch_label, verdict, _) in enumerate(cells):
            ax = ax_list[ci]
            ax.set_facecolor(_BG)
            ax.set_xlim(min(vals) - 0.5, max(vals) + 0.5)
            ax.set_ylim(-0.05, 1.7)
            ax.set_xlabel(xlabel, color=_FG, fontsize=10)
            ax.set_title(ptype.replace("_", " ").title(), color=_FG, fontsize=11)
            ax.tick_params(colors=_FG)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

            # Draw full lines for completed cells
            if ci < reveal_cell:
                ax.plot(vals, rts_norm, "o-", color="#3b82f6", linewidth=2,
                        label="RT (norm.)")
                ax.plot(vals, ch, "s--", color="#f59e0b", linewidth=2,
                        label=ch_label)
            elif ci == reveal_cell:
                n = max(2, int(reveal_frac * len(vals)))
                ax.plot(vals[:n], rts_norm[:n], "o-", color="#3b82f6", linewidth=2)
                ax.plot(vals[:n], ch[:n], "s--", color="#f59e0b", linewidth=2)

            if ci in label_cells:
                # Both urgency and cancellation accounts are "supported" per §11
                v_color = "#22c55e"
                ax.text(0.97, 0.97, verdict, color=v_color, fontsize=9,
                        fontweight="bold", transform=ax.transAxes,
                        ha="right", va="top")

    for ci in range(n_cells):
        for fi in range(_PERT_DRAW_FRAMES):
            apply_theme(DARK_THEME)
            fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
            fig.patch.set_facecolor(_BG)
            _draw_base(axes.flat, ci, (fi + 1) / _PERT_DRAW_FRAMES, set())
            _save_frame(fig, frames_dir, idx)
            idx += 1

        for fi in range(_PERT_LABEL_FRAMES):
            apply_theme(DARK_THEME)
            fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
            fig.patch.set_facecolor(_BG)
            _draw_base(axes.flat, ci + 1, 0.0, set(range(ci + 1)))
            _save_frame(fig, frames_dir, idx)
            idx += 1

    # Final hold
    for fi in range(_PERT_HOLD_FRAMES):
        apply_theme(DARK_THEME)
        fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
        fig.patch.set_facecolor(_BG)
        _draw_base(axes.flat, n_cells, 1.0, set(range(n_cells)))
        _save_frame(fig, frames_dir, idx)
        idx += 1

    return total


# --- Clip 4: Interpretation verdict ---

_INTERP_TITLE_FRAMES = 48
_INTERP_ROW_FRAMES   = 72     # per row reveal
_INTERP_HOLD_FRAMES  = 96

_INTERP_ROWS = [
    ("Selector bottleneck",
     "Wrong-channel choices rise at low freq",
     "No wrong-channel selections observed",
     "not_supported", "✗"),
    ("Urgency / commitment",
     "RT shifts at low freq, choices preserved",
     "✓ Latency/jitter shift RT; channel intact",
     "supported", "✓"),
    ("Cancellation bottleneck",
     "Stop failures worsen at low frequency",
     "✓ Dropout selectively impairs stopping",
     "supported", "✓"),
]


def write_interpretations_frames(
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Verdict table fading in row by row."""
    total = (_INTERP_TITLE_FRAMES
             + len(_INTERP_ROWS) * _INTERP_ROW_FRAMES
             + _INTERP_HOLD_FRAMES)
    if dry_run:
        return total

    idx = 0
    idx += _title_card_frames(
        "Which account does the GPR model support?",
        frames_dir, _INTERP_TITLE_FRAMES, dry_run=False, start_index=idx,
    )

    def _draw_table(n_visible: int, alpha_last: float) -> plt.Figure:
        apply_theme(DARK_THEME)
        fig, ax = plt.subplots(figsize=FIG_SIZE_1080P)
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)
        ax.axis("off")
        fig.suptitle("Which account does the GPR model support?",
                     color=_FG, fontsize=20, fontweight="bold", y=0.95)

        col_x = [0.03, 0.28, 0.58, 0.90]
        headers = ["Account", "Prediction", "Observed", "Verdict"]
        for cx, h in zip(col_x, headers):
            ax.text(cx, 0.80, h, color=_FG, fontsize=14, fontweight="bold",
                    transform=ax.transAxes, va="bottom")
        ax.axhline(0.78, xmin=0.01, xmax=0.99, color="#30363d", linewidth=1,
                   transform=ax.transAxes)

        row_h = 0.18
        for ri, (account, pred, obs, verdict, sym) in enumerate(_INTERP_ROWS):
            if ri >= n_visible:
                break
            a = alpha_last if ri == n_visible - 1 else 1.0
            y = 0.72 - ri * row_h
            color = VERDICT_COLORS[verdict]
            bg_alpha = 0.08 * a
            ax.add_patch(plt.Rectangle((0, y - 0.03), 1.0, row_h - 0.02,
                                        transform=ax.transAxes,
                                        color=color, alpha=bg_alpha))
            ax.text(col_x[0], y + 0.04, account, color=color,
                    fontsize=13, fontweight="bold", transform=ax.transAxes,
                    alpha=a, va="center")
            ax.text(col_x[1], y + 0.04, pred, color=_FG, fontsize=11,
                    transform=ax.transAxes, alpha=a, va="center")
            ax.text(col_x[2], y + 0.04, obs, color=_FG, fontsize=11,
                    transform=ax.transAxes, alpha=a, va="center")
            ax.text(col_x[3], y + 0.04, sym, color=color, fontsize=28,
                    fontweight="bold", transform=ax.transAxes,
                    ha="center", alpha=a, va="center")
        return fig

    for ri in range(len(_INTERP_ROWS)):
        for fi in range(_INTERP_ROW_FRAMES):
            alpha = min(1.0, (fi + 1) / (FPS * 0.75))
            fig = _draw_table(ri + 1, alpha)
            _save_frame(fig, frames_dir, idx)
            idx += 1

    for fi in range(_INTERP_HOLD_FRAMES):
        fig = _draw_table(len(_INTERP_ROWS), 1.0)
        _save_frame(fig, frames_dir, idx)
        idx += 1

    return total


# --- Bridge / title cards ---

def write_bridge_frames(
    text: str,
    frames_dir: Path | None,
    n_frames: int = 120,
    dry_run: bool = False,
) -> int:
    """Write a full-screen dark title card with centred text."""
    if dry_run:
        return n_frames
    return _title_card_frames(text, frames_dir, n_frames,
                               dry_run=False, start_index=0)
