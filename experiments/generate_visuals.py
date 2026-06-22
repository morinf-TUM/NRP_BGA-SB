#!/usr/bin/env python3
# experiments/generate_visuals.py
"""CLI for generating all NRP_BGA-SB visual deliverables.

Usage:
    python experiments/generate_visuals.py --all
    python experiments/generate_visuals.py --figures
    python experiments/generate_visuals.py --clips
    python experiments/generate_visuals.py --hero
    python experiments/generate_visuals.py --dry-run

Run from the project root with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src/ to path if running directly (not as package)
_ROOT = Path(__file__).parent.parent
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visuals.assemble import build_hero, encode_clip
from visuals.clips import (
    write_bridge_frames,
    write_cerebellum_frames,
    write_interpretations_frames,
    write_perturbation_frames,
    write_threshold_frames,
)
from visuals.data_loader import load_perturbation_gonogo, load_perturbation_stopsignal
from visuals.figures import (
    fig_cerebellum_learning,
    fig_frequency_threshold,
    fig_perturbation_decomposition,
    fig_three_interpretations,
)
from visuals.trajectory_gen import (
    generate_cerebellum_trajectories,
    generate_threshold_trajectories,
)

OUTPUT_DIR  = _ROOT / "deprecated_toy_prototype_visuals" / "output"
FRAMES_DIR  = OUTPUT_DIR / "frames"
FPS         = 24


def _log(msg: str) -> None:
    print(f"[generate_visuals] {msg}", flush=True)


def run_figures(dry_run: bool) -> None:
    _log("Generating static figures …")
    out = OUTPUT_DIR / "figures"
    if dry_run:
        _log("  [dry-run] would write fig_frequency_threshold.png")
        _log("  [dry-run] would write fig_perturbation_decomposition.png")
        _log("  [dry-run] would write fig_cerebellum_learning.png")
        _log("  [dry-run] would write fig_three_interpretations.png")
        return
    cereb_trials = generate_cerebellum_trajectories(n_trials=30)
    fig_frequency_threshold(out)
    _log("  ✓ fig_frequency_threshold.png")
    fig_perturbation_decomposition(out)
    _log("  ✓ fig_perturbation_decomposition.png")
    fig_cerebellum_learning(cereb_trials, out)
    _log("  ✓ fig_cerebellum_learning.png")
    fig_three_interpretations(out)
    _log("  ✓ fig_three_interpretations.png")


def run_clips(dry_run: bool) -> None:
    _log("Generating clip frame sequences and encoding MP4s …")

    threshold_trials = generate_threshold_trajectories()
    cereb_trials     = generate_cerebellum_trajectories(n_trials=30)
    gonogo           = load_perturbation_gonogo()
    stopsig          = load_perturbation_stopsignal()

    # Bridge card texts
    bridges = {
        "title":    "NRP_BGA-SB\nBasal Ganglia Frequency & Action Selection",
        "bridge1":  "Does the effect survive\nfull musculoskeletal embodiment?\n\n"
                    "OpenSim Arm26 onset rate:\n5 Hz → 0.000 | ≥10 Hz → 1.000",
        "bridge2":  "What does timing noise reveal?",
        "closing":  "Conclusion:\nUrgency and cancellation-bottleneck\n"
                    "accounts supported.\nSelector-bottleneck ruled out.",
    }

    clips_spec = [
        ("threshold",      lambda d: write_threshold_frames(threshold_trials,
                                         FRAMES_DIR / "threshold", d)),
        ("cerebellum",     lambda d: write_cerebellum_frames(cereb_trials,
                                         FRAMES_DIR / "cerebellum", d)),
        ("perturbation",   lambda d: write_perturbation_frames(gonogo, stopsig,
                                         FRAMES_DIR / "perturbation", d)),
        ("interpretations",lambda d: write_interpretations_frames(
                                         FRAMES_DIR / "interpretations", d)),
    ]
    bridge_spec = [
        ("bridge_title",   bridges["title"],   120),
        ("bridge_1",       bridges["bridge1"], 120),
        ("bridge_2",       bridges["bridge2"], 120),
        ("bridge_closing", bridges["closing"], 120),
    ]

    for name, writer in clips_spec:
        t0 = time.time()
        n = writer(dry_run)
        if not dry_run:
            mp4 = OUTPUT_DIR / f"clip_{name}.mp4"
            encode_clip(FRAMES_DIR / name, mp4, fps=FPS, fade_frames=FPS)
            _log(f"  ✓ clip_{name}.mp4 ({n} frames, {time.time()-t0:.1f}s)")
        else:
            _log(f"  [dry-run] clip_{name}: {n} frames")

    for name, text, n_frames in bridge_spec:
        t0 = time.time()
        n = write_bridge_frames(text, FRAMES_DIR / name, n_frames, dry_run)
        if not dry_run:
            mp4 = OUTPUT_DIR / f"{name}.mp4"
            encode_clip(FRAMES_DIR / name, mp4, fps=FPS, fade_frames=FPS // 2)
            _log(f"  ✓ {name}.mp4 ({n} frames, {time.time()-t0:.1f}s)")
        else:
            _log(f"  [dry-run] {name}: {n} frames")


def run_hero(dry_run: bool) -> None:
    _log("Assembling hero video …")
    if dry_run:
        _log("  [dry-run] would concat all clips → hero_video.mp4")
        return
    ordered = [
        OUTPUT_DIR / "bridge_title.mp4",
        OUTPUT_DIR / "clip_threshold.mp4",
        OUTPUT_DIR / "bridge_1.mp4",
        OUTPUT_DIR / "clip_cerebellum.mp4",
        OUTPUT_DIR / "bridge_2.mp4",
        OUTPUT_DIR / "clip_perturbation.mp4",
        OUTPUT_DIR / "clip_interpretations.mp4",
        OUTPUT_DIR / "bridge_closing.mp4",
    ]
    hero = OUTPUT_DIR / "hero_video.mp4"
    build_hero(ordered, hero)
    _log("  ✓ hero_video.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NRP_BGA-SB visual deliverables."
    )
    parser.add_argument("--figures",  action="store_true", help="Static PNGs only")
    parser.add_argument("--clips",    action="store_true", help="Animated clip MP4s")
    parser.add_argument("--hero",     action="store_true", help="Hero video only")
    parser.add_argument("--all",      action="store_true", help="Everything")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Verify pipeline runs; no file output")
    args = parser.parse_args()

    if not any([args.figures, args.clips, args.hero, args.all]):
        parser.print_help()
        sys.exit(1)

    dry = args.dry_run
    if dry:
        _log("DRY RUN — no files will be written")

    if args.figures or args.all:
        run_figures(dry)
    if args.clips or args.all:
        run_clips(dry)
    if args.hero or args.all:
        run_hero(dry)

    _log("Done.")


if __name__ == "__main__":
    main()
