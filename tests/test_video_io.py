"""Tests for backend.tools.video_io.make_video_writer robustness."""

import cv2

from backend.tools.video_io import make_video_writer


def test_even_dims_prefer_ffmpeg_or_opencv(tmp_path):
    w = make_video_writer(str(tmp_path / "a.mp4"), 25, (160, 90))
    assert hasattr(w, "write") and hasattr(w, "release")
    w.release()


def test_odd_dims_route_to_opencv(tmp_path):
    # libx264+yuv420p can't do odd dims; must use cv2.VideoWriter, not crash.
    w = make_video_writer(str(tmp_path / "b.mp4"), 25, (161, 91))
    assert isinstance(w, cv2.VideoWriter)
    w.release()


def test_zero_fps_does_not_crash(tmp_path):
    # fps=0 (corrupt/variable-rate source) must be clamped, not produce a
    # 0-byte/unplayable writer construction failure.
    w = make_video_writer(str(tmp_path / "c.mp4"), 0, (160, 90))
    assert hasattr(w, "write") and hasattr(w, "release")
    w.release()


def test_nan_fps_does_not_crash(tmp_path):
    w = make_video_writer(str(tmp_path / "d.mp4"), float("nan"), (160, 90))
    assert hasattr(w, "write") and hasattr(w, "release")
    w.release()
