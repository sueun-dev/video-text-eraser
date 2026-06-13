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
