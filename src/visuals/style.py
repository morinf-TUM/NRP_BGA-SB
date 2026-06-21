"""Matplotlib themes and color palettes for Phase 13 visual deliverables."""
from __future__ import annotations

import matplotlib as mpl

DARK_THEME: dict = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "text.color":        "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.alpha":        0.4,
    "lines.linewidth":   2.0,
    "font.size":         14,
    "axes.titlesize":    18,
    "axes.titleweight":  "bold",
    "figure.dpi":        96,
}

LIGHT_THEME: dict = {
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8f9fa",
    "axes.edgecolor":    "#dee2e6",
    "axes.labelcolor":   "#212529",
    "text.color":        "#212529",
    "xtick.color":       "#495057",
    "ytick.color":       "#495057",
    "grid.color":        "#dee2e6",
    "grid.alpha":        0.7,
    "lines.linewidth":   2.0,
    "font.size":         13,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
}

# Per-frequency colours: red for miss (5 Hz), green at threshold, blues/purples above
FREQ_COLORS: dict[int, str] = {
    5:   "#ef4444",   # red   — miss frequency
    10:  "#22c55e",   # green — selection threshold
    20:  "#3b82f6",   # blue
    40:  "#a855f7",   # purple
    80:  "#f59e0b",   # amber
    160: "#06b6d4",   # cyan
}

VERDICT_COLORS: dict[str, str] = {
    "supported":     "#22c55e",   # green
    "not_supported": "#ef4444",   # red
}

FIG_SIZE_1080P = (1920 / 150, 1080 / 150)   # inches at 150 dpi → 1920×1080


def apply_theme(theme: dict) -> None:
    """Apply an rcParams theme dict globally."""
    mpl.rcParams.update(theme)
