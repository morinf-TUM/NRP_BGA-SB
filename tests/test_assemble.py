import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from visuals.assemble import encode_clip, build_hero, _ffmpeg_encode_args, _concat_manifest


def test_ffmpeg_encode_args_contains_required_flags():
    args = _ffmpeg_encode_args(
        frames_dir=Path("/tmp/frames/threshold"),
        output_path=Path("/tmp/out/clip.mp4"),
        fps=24,
        fade_frames=24,
        n_frames=480,
    )
    cmd = " ".join(args)
    assert "-framerate" in cmd
    assert "24" in cmd
    assert "libx264" in cmd
    assert "yuv420p" in cmd
    assert "fade=in" in cmd
    assert "fade=out" in cmd
    assert "/tmp/out/clip.mp4" in cmd


def test_concat_manifest_lists_all_clips(tmp_path):
    clip_paths = [
        tmp_path / "clip_a.mp4",
        tmp_path / "clip_b.mp4",
        tmp_path / "clip_c.mp4",
    ]
    manifest = _concat_manifest(clip_paths)
    for p in clip_paths:
        assert str(p) in manifest
    assert manifest.count("file ") == 3


def test_encode_clip_calls_ffmpeg(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Create minimal dummy PNG files
    for i in range(5):
        (frames_dir / f"{i:04d}.png").write_bytes(b"PNG")

    out = tmp_path / "clip.mp4"
    with patch("visuals.assemble.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = encode_clip(frames_dir, out, fps=24, fade_frames=5)
    assert mock_run.called
    args_used = mock_run.call_args[0][0]
    assert "ffmpeg" in args_used[0]
    assert str(out) in args_used


def test_build_hero_calls_ffmpeg(tmp_path):
    clips = [tmp_path / f"clip_{i}.mp4" for i in range(3)]
    for c in clips:
        c.write_bytes(b"MP4")
    hero = tmp_path / "hero.mp4"
    with patch("visuals.assemble.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = build_hero(clips, hero)
    assert mock_run.called
