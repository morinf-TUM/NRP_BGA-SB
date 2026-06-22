# visuals/clips.py
"""Clip frame-sequence generators.

Each writer generates numbered PNGs in a directory. dry_run=True returns
the expected frame count without writing any files (used for smoke checks).

Clips 1 (threshold) and 2 (cerebellum) use MuJoCo EGL for 3D arm rendering
with PIL text/glow-arc overlays.  Clips 3 (perturbation) and 4 (interpretations)
use matplotlib for chart-based content — no arm model required.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visuals.arm_renderer import ArmRenderer, REST_POSITION, sim_to_screen
from visuals.style import (
    DARK_THEME,
    FIG_SIZE_1080P,
    FREQ_COLORS,
    VERDICT_COLORS,
    apply_theme,
)

FPS  = 24
_BG  = "#0d1117"    # figure background colour (matplotlib clips)
_FG  = "#e6edf3"    # text and axis colour

# PIL helpers (used by arm-based clips)
_PIL_FONT = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
_W, _H = 1920, 1080


def _pil_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_PIL_FONT, size)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# --- Matplotlib helpers (used by chart-based clips) ---

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


def _title_card_frames(title_text: str, frames_dir: Path | None,
                       n_frames: int, dry_run: bool, start_index: int) -> int:
    """Write a full-screen matplotlib title-card block; return frame count."""
    if dry_run:
        return n_frames
    for i in range(n_frames):
        fig, ax = _new_dark_fig()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        alpha = min(1.0, i / max(1, FPS * 0.5))
        ax.text(0.5, 0.5, title_text, color=_FG, fontsize=24,
                ha="center", va="center", alpha=alpha,
                transform=ax.transAxes, fontweight="bold", wrap=True)
        _save_frame(fig, frames_dir, start_index + i)
    return n_frames


# --- PIL title card (used for arm-based clips) ---

def _pil_title_card_frames(
    title_text: str,
    frames_dir: Path,
    n_frames: int,
    start_index: int,
) -> int:
    """Write PIL dark title-card frames with fade-in; return frame count."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        alpha = min(1.0, i / max(1, FPS * 0.5))
        rv = int(230 * alpha); gv = int(237 * alpha); bv = int(243 * alpha)
        img  = Image.new("RGB", (_W, _H), (13, 17, 23))
        draw = ImageDraw.Draw(img)
        draw.multiline_text(
            (_W // 2, _H // 2), title_text,
            fill=(rv, gv, bv), font=_pil_font(44),
            anchor="mm", align="center",
        )
        img.save(frames_dir / f"{start_index + i:04d}.png")
    return n_frames


# --- Clip 1: Frequency threshold (MuJoCo arm + PIL) ---

_THRESHOLD_FRAMES_PER_FREQ = 72    # 3 s per frequency: 12 hold + 48 arc + 12 result
_THRESHOLD_TITLE_FRAMES    = 48    # 2 s intro title card
_THRESHOLD_CAPTION_FRAMES  = 48    # 2 s closing caption


def write_threshold_frames(
    trials: list[dict],
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Animate 3D arm at 5→80 Hz; MISS (5 Hz) then HIT (≥10 Hz)."""
    total = (_THRESHOLD_TITLE_FRAMES
             + len(trials) * _THRESHOLD_FRAMES_PER_FREQ
             + _THRESHOLD_CAPTION_FRAMES)
    if dry_run:
        return total

    hold   = FPS // 2    # 12 frames: arm at ready pose
    arc_f  = FPS * 2     # 48 frames: arm moves through trajectory
    result = FPS // 2    # 12 frames: result label displayed

    target_screen = sim_to_screen(0.0, 1.0)   # pixel coords of the target

    idx = 0
    idx += _pil_title_card_frames(
        "BG update frequency governs action commitment",
        frames_dir, _THRESHOLD_TITLE_FRAMES, idx,
    )

    renderer = ArmRenderer()
    try:
        for trial in trials:
            freq       = trial["frequency_hz"]
            color_hex  = FREQ_COLORS.get(int(freq), "#e6edf3")
            cr, cg, cb = _hex_to_rgb(color_hex)
            color_rgba = (cr / 255, cg / 255, cb / 255, 1.0)
            hit        = not trial["gate_closed"]
            positions  = trial["positions_xy"]
            start_pos  = positions[0]

            # Strip pre-onset stationary section so the arc shows only movement.
            moving   = [i for i, p in enumerate(positions)
                        if abs(p[0] - start_pos[0]) > 0.001
                        or abs(p[1] - start_pos[1]) > 0.001]
            anim_pos = positions[moving[0]:] if moving else positions

            freq_label = f"{int(freq)} Hz"
            lbl_pos    = (1100, 75)    # right-panel text anchor

            # --- Hold: natural ready pose ---
            for _ in range(hold):
                pixels = renderer.render_raw(REST_POSITION, color_rgba)
                renderer.save_frame(pixels, frames_dir, idx,
                                    labels=[(lbl_pos, freq_label, 80, color_hex)])
                idx += 1

            # --- Arc: arm animates through trajectory ---
            for fi in range(arc_f):
                if hit:
                    reveal      = max(2, int((fi + 1) / arc_f * len(anim_pos)))
                    current_pos = anim_pos[reveal - 1]
                    arc_pts     = [sim_to_screen(p[0], p[1])
                                   for p in anim_pos[:reveal]]
                else:
                    current_pos = REST_POSITION
                    arc_pts     = None

                pixels = renderer.render_raw(current_pos, color_rgba)
                labels = [(lbl_pos, freq_label, 80, color_hex)]
                if not hit:
                    # Blocked-gate marker: "X" at the target position
                    labels.append(
                        ((target_screen[0] - 18, target_screen[1] - 55),
                         "X", 72, "#ef4444"),
                    )
                renderer.save_frame(pixels, frames_dir, idx,
                                    labels=labels,
                                    arc_pts=arc_pts, arc_color=color_hex)
                idx += 1

            # --- Result label ---
            label_text  = "HIT"  if hit else "MISS"
            label_color = "#22c55e" if hit else "#ef4444"
            final_pos   = anim_pos[-1] if hit else REST_POSITION
            full_arc    = ([sim_to_screen(p[0], p[1]) for p in anim_pos]
                           if hit else None)
            for _ in range(result):
                pixels = renderer.render_raw(final_pos, color_rgba)
                renderer.save_frame(pixels, frames_dir, idx,
                                    labels=[
                                        (lbl_pos, freq_label, 80, color_hex),
                                        ((1100, 185), label_text, 64, label_color),
                                    ],
                                    arc_pts=full_arc, arc_color=color_hex,
                                    arc_alpha=0.7)
                idx += 1

        # --- Closing caption card ---
        caption = "Below 10 Hz the BG samples only\nneutral cortical evidence"
        neutral_rgba = (0.35, 0.65, 1.0, 1.0)
        for fi in range(_THRESHOLD_CAPTION_FRAMES):
            fade   = min(1.0, fi / (FPS * 0.5))
            cfade  = int(230 * fade)
            pixels = renderer.render_raw(REST_POSITION, neutral_rgba)
            img    = Image.fromarray(pixels, "RGB")
            draw   = ImageDraw.Draw(img)
            draw.multiline_text(
                (_W // 2, _H - 150), caption,
                fill=(cfade, cfade, int(243 * fade)),
                font=_pil_font(40), anchor="mm", align="center",
            )
            frames_dir.mkdir(parents=True, exist_ok=True)
            img.save(frames_dir / f"{idx:04d}.png")
            idx += 1
    finally:
        renderer.close()

    return total


# --- Clip 2: Cerebellar adaptation (MuJoCo arm + PIL) ---

_CEREB_TITLE_FRAMES     = 48    # 2 s
_CEREB_FRAMES_PER_TRIAL = 20    # ~0.83 s per trial at 24 fps


def write_cerebellum_frames(
    trials: list[dict],
    frames_dir: Path | None,
    dry_run: bool = False,
) -> int:
    """Animate 30 go-trials showing 3D arm arc rotating toward target as θ̂ builds."""
    go_trials = [t for t in trials if t["is_go"]]
    total = _CEREB_TITLE_FRAMES + len(go_trials) * _CEREB_FRAMES_PER_TRIAL
    if dry_run:
        return total

    arc_f  = int(_CEREB_FRAMES_PER_TRIAL * 0.7)    # 14 frames: arc draws
    hold_f = _CEREB_FRAMES_PER_TRIAL - arc_f         # 6 frames: hold at endpoint

    n   = len(go_trials)
    idx = 0
    idx += _pil_title_card_frames(
        "Cerebellum corrects visuomotor rotation across trials",
        frames_dir, _CEREB_TITLE_FRAMES, idx,
    )

    renderer = ArmRenderer()
    try:
        for ti, trial in enumerate(go_trials):
            frac  = ti / max(1, n - 1)
            # Colour fades red → green as learning progresses.
            rv = int(239 * (1 - frac) + 34 * frac)
            gv = int(68  * (1 - frac) + 197 * frac)
            bv = int(68  * (1 - frac) + 94  * frac)
            color_rgba = (rv / 255, gv / 255, bv / 255, 1.0)
            color_hex  = f"#{rv:02x}{gv:02x}{bv:02x}"

            positions  = trial["positions_xy"]
            theta_deg  = trial["theta_hat"] * 180.0 / np.pi
            start_pos  = positions[0]

            moving   = [i for i, p in enumerate(positions)
                        if abs(p[0] - start_pos[0]) > 0.001
                        or abs(p[1] - start_pos[1]) > 0.001]
            anim_pos = positions[moving[0]:] if moving else positions

            trial_lbl = f"Trial {ti + 1}"
            theta_lbl = f"adapt: {theta_deg:.1f} deg"

            # --- Arc draw ---
            for fi in range(arc_f):
                reveal      = max(2, int((fi + 1) / arc_f * len(anim_pos)))
                current_pos = anim_pos[reveal - 1]
                arc_pts     = [sim_to_screen(p[0], p[1])
                               for p in anim_pos[:reveal]]

                pixels = renderer.render_raw(current_pos, color_rgba)
                renderer.save_frame(pixels, frames_dir, idx,
                                    labels=[
                                        ((1100, 75),  trial_lbl, 60, "#e6edf3"),
                                        ((1100, 155), theta_lbl, 44, "#a855f7"),
                                    ],
                                    arc_pts=arc_pts, arc_color=color_hex)
                idx += 1

            # --- Hold at final endpoint ---
            final_pos = anim_pos[-1]
            full_arc  = [sim_to_screen(p[0], p[1]) for p in anim_pos]
            for _ in range(hold_f):
                pixels = renderer.render_raw(final_pos, color_rgba)
                renderer.save_frame(pixels, frames_dir, idx,
                                    labels=[
                                        ((1100, 75),  trial_lbl, 60, "#e6edf3"),
                                        ((1100, 155), theta_lbl, 44, "#a855f7"),
                                    ],
                                    arc_pts=full_arc, arc_color=color_hex,
                                    arc_alpha=0.7)
                idx += 1
    finally:
        renderer.close()

    return total


# --- Clip 3: Perturbation decomposition (matplotlib chart) ---

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
        rts  = [r["bg_commitment_latency_mean"] or 0.0 for r in rows]
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

    for fi in range(_PERT_HOLD_FRAMES):
        apply_theme(DARK_THEME)
        fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_1080P)
        fig.patch.set_facecolor(_BG)
        _draw_base(axes.flat, n_cells, 1.0, set(range(n_cells)))
        _save_frame(fig, frames_dir, idx)
        idx += 1

    return total


# --- Clip 4: Interpretation verdict (matplotlib) ---

_INTERP_TITLE_FRAMES = 48
_INTERP_ROW_FRAMES   = 72
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
        ax.plot([0.01, 0.99], [0.78, 0.78], color="#30363d", linewidth=1,
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


# --- Bridge / title cards (matplotlib) ---

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
