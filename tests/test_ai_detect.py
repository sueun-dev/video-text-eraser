"""Tests for webapp.ai_detect parsing/encoding (no network calls)."""

import base64
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from webapp.ai_detect import _encode_jpeg, _parse_boxes, detect


# --------------------------------------------------------------------------
# _parse_boxes
# --------------------------------------------------------------------------

def test_parse_clean_json():
    boxes = _parse_boxes('{"boxes":[{"label":"logo","kind":"logo",'
                         '"ymin":0.1,"ymax":0.2,"xmin":0.7,"xmax":0.95}]}')
    assert len(boxes) == 1
    assert boxes[0]["xmax"] == 0.95
    assert boxes[0]["kind"] == "logo"


def test_parse_json_wrapped_in_prose_and_fences():
    text = 'Sure! ```json\n{"boxes":[{"label":"x","kind":"text",' \
           '"ymin":0.1,"ymax":0.2,"xmin":0.0,"xmax":1.0}]}\n``` done'
    assert len(_parse_boxes(text)) == 1


def test_parse_empty_boxes():
    assert _parse_boxes('{"boxes": []}') == []


def test_parse_drops_inverted_box():
    # ymax <= ymin must be discarded.
    assert _parse_boxes('{"boxes":[{"ymin":0.5,"ymax":0.4,"xmin":0,"xmax":1}]}') == []


def test_parse_clamps_out_of_range():
    boxes = _parse_boxes('{"boxes":[{"ymin":-0.3,"ymax":1.4,"xmin":0.1,"xmax":0.9}]}')
    assert boxes[0]["ymin"] == 0.0
    assert boxes[0]["ymax"] == 1.0


def test_parse_skips_box_missing_fields():
    # One valid, one missing xmax -> only the valid one survives.
    text = '{"boxes":[{"ymin":0.1,"ymax":0.2,"xmin":0.1},' \
           '{"ymin":0.3,"ymax":0.4,"xmin":0.1,"xmax":0.5}]}'
    boxes = _parse_boxes(text)
    assert len(boxes) == 1
    assert boxes[0]["ymin"] == 0.3


def test_parse_raises_without_json():
    with pytest.raises(ValueError):
        _parse_boxes("no json here at all")


def test_parse_handles_trailing_second_brace_block():
    # Regression: a greedy {.*} match would span both blocks and fail to parse.
    text = '{"boxes":[{"label":"a","kind":"logo","ymin":0.1,"ymax":0.2,' \
           '"xmin":0.7,"xmax":0.9}]} note: {"unrelated": true}'
    boxes = _parse_boxes(text)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "a"


def test_parse_skips_leading_object_without_boxes():
    # A leading JSON object lacking "boxes" must not shadow the real one.
    text = '{"meta": 1} then the real answer: ' \
           '{"boxes":[{"ymin":0.1,"ymax":0.2,"xmin":0.1,"xmax":0.5}]}'
    boxes = _parse_boxes(text)
    assert len(boxes) == 1


def test_parse_label_truncated():
    long_label = "L" * 200
    boxes = _parse_boxes(
        '{"boxes":[{"label":"' + long_label + '","ymin":0,"ymax":0.1,"xmin":0,"xmax":0.1}]}'
    )
    assert len(boxes[0]["label"]) <= 80


# --------------------------------------------------------------------------
# _encode_jpeg
# --------------------------------------------------------------------------

def test_encode_jpeg_returns_decodable_base64():
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    encoded = _encode_jpeg(frame)
    raw = base64.b64decode(encoded)
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[2] == 3


def test_encode_jpeg_downscales_large_frame():
    frame = np.full((2000, 3000, 3), 80, dtype=np.uint8)
    raw = base64.b64decode(_encode_jpeg(frame, max_side=1280))
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 1280


def test_encode_jpeg_keeps_small_frame_size():
    frame = np.full((90, 160, 3), 80, dtype=np.uint8)
    raw = base64.b64decode(_encode_jpeg(frame, max_side=1280))
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (90, 160)


# --------------------------------------------------------------------------
# detect dispatch
# --------------------------------------------------------------------------

def test_detect_unknown_provider_raises():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect("gemini", frame, "key")


def test_detect_routes_to_correct_provider_with_target_mapping():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with patch("webapp.ai_detect.detect_claude", return_value=["C"]) as mc, \
         patch("webapp.ai_detect.detect_openai", return_value=["O"]) as mo:
        assert detect("claude", frame, "k", target="text") == ["C"]
        assert mc.call_args.args[3] == "text overlays only (subtitles, captions)"
        assert mo.call_count == 0

        assert detect("openai", frame, "k", target="logo") == ["O"]
        assert mo.call_args.args[3] == "graphic overlays only (logos, watermarks, badges)"

        detect("claude", frame, "k", target="anything-else")
        assert mc.call_args.args[3] == "both text and graphic overlays"


def test_parse_boxes_malformed_container_no_crash():
    # boxes as a dict, or a list of non-dicts, must degrade to [] not crash.
    assert _parse_boxes('{"boxes": {"ymin": 0.1}}') == []
    assert _parse_boxes('{"boxes": ["hello", 42]}') == []
