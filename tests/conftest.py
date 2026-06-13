"""Shared fixtures for the video-text-eraser test suite.

These keep the suite fast and hermetic: pure-logic tests need no models,
GPU, or network. The repo root is put on sys.path so ``backend`` / ``webapp``
import the same way they do at runtime.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# config/ and model paths are resolved relative to the repo root at import time.
os.chdir(ROOT)


@pytest.fixture
def solid_frame():
    """A 120x200 BGR frame filled with a mid-gray so diffs are easy to reason about."""
    return np.full((120, 200, 3), 127, dtype=np.uint8)


@pytest.fixture
def make_frame():
    """Factory: make_frame(h, w, value) -> uint8 BGR frame."""
    def _make(h=120, w=200, value=127):
        return np.full((h, w, 3), value, dtype=np.uint8)
    return _make


@pytest.fixture
def synth_video(tmp_path):
    """Factory writing a tiny synthetic mp4 and returning its path.

    synth_video(frames=20, w=160, h=90, watermark=None) -> str path.
    When ``watermark=(y1,y2,x1,x2)`` a fixed opaque box is drawn every frame
    (a moving gradient background makes the box stand out temporally).
    """
    import cv2

    def _make(frames=20, w=160, h=90, fps=10, watermark=None):
        path = str(tmp_path / f"synth_{frames}_{w}x{h}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(frames):
            # A scrolling gradient so the background genuinely changes over time.
            base = np.tile(
                np.linspace((i * 13) % 256, (i * 13 + 200) % 256, w, dtype=np.uint8),
                (h, 1),
            )
            frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if watermark is not None:
                y1, y2, x1, x2 = watermark
                cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), thickness=-1)
            writer.write(frame)
        writer.release()
        return path

    return _make
