"""Regression tests for SubtitleRemover.run() resource cleanup (no models).

These pin the try/finally behaviour the round-4 fix introduced: resources are
released even when processing raises, and the prefetcher/temp file do not leak.
"""

import os

import pytest

from backend.main import SubtitleRemover


def _raise(_tbar):
    raise RuntimeError("boom")


def test_run_releases_resources_on_exception(synth_video, monkeypatch):
    path = synth_video(frames=5, w=160, h=90)
    sr = SubtitleRemover(path)
    sr.sub_areas = [(0, 90, 0, 160)]
    temp_name = sr.video_temp_file.name
    # Force processing to fail partway through.
    monkeypatch.setattr(sr, "_process_video", _raise)

    with pytest.raises(RuntimeError):
        sr.run()

    # finally must have cleaned up despite the exception.
    assert not sr.video_cap.isOpened()
    assert not os.path.exists(temp_name)
    assert sr.isFinished is False


class _Bar:
    """Minimal tqdm stand-in for driving the reader loops."""
    def __init__(self):
        self.n = 0
        self.total = 1000

    def update(self, inc):
        self.n += inc

    def write(self, *a):
        pass


def test_detection_loop_terminates_when_run_exceeds_real_frames(synth_video, monkeypatch):
    # Deadlock regression: a subtitle run whose end exceeds the REAL frame count
    # (over-counted CAP_PROP_FRAME_COUNT) must not hang the reader loop — the
    # FramePrefetcher emits only one EOF sentinel, so the inner read consuming it
    # used to wedge the outer read forever.
    import threading

    from backend.inpaint.opencv_inpaint import OpenCVInpaint

    path = synth_video(frames=6, w=160, h=90)
    sr = SubtitleRemover(path)
    sr.sub_areas = [(0, 90, 0, 160)]
    # Claim far more frames than actually decode, and a run that spans them.
    monkeypatch.setattr(sr, "frame_count", 100)
    box = (10, 150, 40, 60)   # xmin,xmax,ymin,ymax
    monkeypatch.setattr(sr, "_detect_subtitle_runs",
                        lambda **k: ({1: [box], 2: [box]}, [(1, 100)]))

    done = threading.Event()

    def run():
        try:
            sr._run_detection_inpaint(_Bar(), OpenCVInpaint())
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(timeout=20), "detection reader loop deadlocked on over-long run"
    sr.video_cap.release()
    sr.video_writer.release()


def test_run_does_not_mux_when_processing_fails(synth_video, monkeypatch):
    path = synth_video(frames=5, w=160, h=90)
    sr = SubtitleRemover(path)
    sr.sub_areas = [(0, 90, 0, 160)]
    merged = {"called": False}
    monkeypatch.setattr(sr, "_process_video", _raise)
    monkeypatch.setattr(sr, "merge_audio_to_video",
                        lambda: merged.__setitem__("called", True))

    with pytest.raises(RuntimeError):
        sr.run()
    assert merged["called"] is False     # no muxing of a never-written video
