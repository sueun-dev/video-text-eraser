"""Tests for STTNAutoInpaint's EOF-driven chunk loop (no STTN model loaded)."""

import numpy as np

from backend.inpaint.sttn_auto_inpaint import STTNAutoInpaint
from backend.tools.inpaint_tools import create_mask, get_inpaint_area_by_mask


class _PassthroughSTTN:
    """Stub for the inner STTN model: returns the scaled crops unchanged."""
    model_input_width = 64
    model_input_height = 32

    def inpaint(self, frames):
        return list(frames)


class _Writer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)


def _prefetcher(n):
    """A fake FramePrefetcher yielding n frames, then EOF (False, None) forever."""
    state = {"i": 0}
    frame = np.full((90, 160, 3), 100, dtype=np.uint8)

    class _P:
        def read(self):
            if state["i"] < n:
                state["i"] += 1
                return True, frame.copy()
            return False, None

    return _P()


def _bare_auto():
    sa = object.__new__(STTNAutoInpaint)
    sa.sttn_inpaint = _PassthroughSTTN()
    return sa


def _bands_and_mask():
    mask = create_mask((90, 160), [(0, 160, 40, 60)])   # xmin,xmax,ymin,ymax
    bands = get_inpaint_area_by_mask(160, 90, int(160 * 3 / 16), mask)
    return mask[:, :, None], bands


def test_process_chunk_reports_eof_and_writes_all_real_frames():
    # Regression: when the chunk end exceeds the real frames (under-counted
    # metadata), _process_chunk must return reached_eof=True and still write
    # every frame it actually read (no drop).
    sa = _bare_auto()
    mask3, bands = _bands_and_mask()
    writer = _Writer()
    info = {"W_ori": 160, "H_ori": 90, "fps": 10, "len": 3}
    eof = sa._process_chunk(_prefetcher(3), writer, mask3, bands,
                            0, 10, info, None, None, None)
    assert eof is True
    assert len(writer.frames) == 3          # all real frames written, none dropped


def test_process_chunk_no_eof_when_chunk_within_frames():
    sa = _bare_auto()
    mask3, bands = _bands_and_mask()
    writer = _Writer()
    info = {"W_ori": 160, "H_ori": 90, "fps": 10, "len": 5}
    eof = sa._process_chunk(_prefetcher(5), writer, mask3, bands,
                            0, 3, info, None, None, None)
    assert eof is False                     # chunk fully satisfied, more may remain
    assert len(writer.frames) == 3
