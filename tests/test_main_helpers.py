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
    # range is [run_start, run_end); frame 3 excluded by design.
    result = sr._collect_run_boxes(frame_boxes, 1, 3)
    assert a in result and b in result
    assert len(result) == 2          # deduped despite appearing twice


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
