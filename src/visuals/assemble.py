"""ffmpeg wrappers: PNG frame sequences → per-clip MP4s → hero concat."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _ffmpeg_encode_args(
    frames_dir: Path,
    output_path: Path,
    fps: int,
    fade_frames: int,
    n_frames: int,
) -> list[str]:
    """Build the ffmpeg argument list for encoding a frame sequence."""
    fade_out_start = max(0, n_frames - fade_frames)
    vf = f"fade=in:0:{fade_frames},fade=out:{fade_out_start}:{fade_frames}"
    return [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%04d.png"),
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]


def _concat_manifest(clip_paths: list[Path]) -> str:
    """Return the content of an ffmpeg concat manifest file."""
    return "\n".join(f"file '{p}'" for p in clip_paths)


def encode_clip(
    frames_dir: Path,
    output_path: Path,
    fps: int = 24,
    fade_frames: int = 24,
) -> Path:
    """Encode a numbered PNG frame directory into a single MP4.

    Args:
        frames_dir:   Directory containing %04d.png frames.
        output_path:  Destination MP4 path (created or overwritten).
        fps:          Output frame rate.
        fade_frames:  Number of frames for fade-in and fade-out.

    Returns:
        output_path on success.

    Raises:
        RuntimeError: if ffmpeg exits with a non-zero return code.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    args = _ffmpeg_encode_args(frames_dir, output_path, fps, fade_frames,
                                n_frames=len(pngs))
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_path


def build_hero(clip_paths: list[Path], output_path: Path) -> Path:
    """Concatenate multiple MP4 clips into a single hero video.

    Uses `ffmpeg -f concat -c copy` — no re-encoding, so all clips must
    share the same codec, resolution, and frame rate.

    Args:
        clip_paths:   Ordered list of existing MP4 files.
        output_path:  Destination hero MP4 path.

    Returns:
        output_path on success.

    Raises:
        FileNotFoundError: if any clip_path does not exist.
        RuntimeError: if ffmpeg exits with a non-zero return code.
    """
    for p in clip_paths:
        if not p.exists():
            raise FileNotFoundError(f"Clip not found: {p}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.parent / "concat_manifest.txt"
    manifest_path.write_text(_concat_manifest(clip_paths))

    args = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(manifest_path),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_path
