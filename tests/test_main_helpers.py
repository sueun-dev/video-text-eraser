"""Tests for pure helpers on backend.main.SubtitleRemover (no media/models)."""

from backend.config import config
from backend.main import SubtitleRemover


def _bare_remover():
    return object.__new__(SubtitleRemover)


def test_collect_run_boxes_unions_and_dedups():
    sr = _bare_remover()
    a = (0, 100, 50, 70)   # wide box (xmin,xmax,ymin,ymax) -> a subtitle line
    b = (0, 120, 50, 70)
    frame_boxes = {1: [a], 2: [a, b], 3: [b]}
    result = sr._collect_run_boxes(frame_boxes, 1, 3)
    assert a in result and b in result
    assert len(result) == 2          # deduped despite appearing twice


def test_collect_run_boxes_includes_the_final_frame():
    # Regression: the run is inclusive of run_end, so a box that appears ONLY on
    # the last frame must still be in the mask (else it survives inpainting).
    sr = _bare_remover()
    a = (0, 100, 50, 70)
    last_only = (0, 120, 200, 220)
    frame_boxes = {1: [a], 2: [a], 3: [last_only]}
    result = sr._collect_run_boxes(frame_boxes, 1, 3)
    assert last_only in result


def test_collect_run_boxes_drops_tall_false_detections():
    sr = _bare_remover()
    limit = config.subtitleYXAxisDifferencePixel.value
    # A box much taller than wide: (ymax-ymin) - (xmax-xmin) > limit -> dropped.
    tall = (0, 5, 0, 5 + limit + 50)     # width 5, height 5+limit+50
    wide = (0, 100, 0, 20)
    frame_boxes = {1: [tall, wide]}
    result = sr._collect_run_boxes(frame_boxes, 1, 2)
    assert wide in result
    assert tall not in result


def test_collect_run_boxes_empty_when_no_boxes():
    sr = _bare_remover()
    assert sr._collect_run_boxes({}, 1, 5) == []


class _FakeBar:
    """Minimal stand-in for tqdm with a settable total."""
    def __init__(self, total):
        self.total = total
        self.n = 0

    def update(self, inc):
        self.n += inc


def test_update_progress_handles_zero_total():
    # frame_count==0 metadata -> tbar.total==0 must not ZeroDivisionError.
    sr = _bare_remover()
    sr.progress_listeners = []
    sr.isFinished = False
    sr.update_progress(_FakeBar(0), 1)
    assert sr.progress_total == 0


def test_update_progress_computes_percentage():
    sr = _bare_remover()
    sr.progress_listeners = []
    sr.isFinished = False
    bar = _FakeBar(10)
    sr.update_progress(bar, 5)
    assert sr.progress_total == 50
