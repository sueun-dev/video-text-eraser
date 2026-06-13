"""Tests for backend.tools.ocr.get_coordinates."""

from backend.tools.ocr import get_coordinates


def test_axis_aligned_quad():
    # Perfect rectangle -> same rectangle as (xmin, xmax, ymin, ymax).
    quad = [[(10, 20), (110, 20), (110, 50), (10, 50)]]
    assert get_coordinates(quad) == [(10, 110, 20, 50)]


def test_slanted_quad_is_conservative():
    # Inner max/min keeps the box inside the slanted polygon edges.
    quad = [[(10, 22), (110, 18), (108, 50), (12, 48)]]
    # xmin = max(10, 12) = 12 ; xmax = min(110, 108) = 108
    # ymin = max(22, 18) = 22 ; ymax = min(50, 48) = 48
    assert get_coordinates(quad) == [(12, 108, 22, 48)]


def test_multiple_boxes_preserved_in_order():
    quads = [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(20, 20), (40, 20), (40, 30), (20, 30)],
    ]
    assert get_coordinates(quads) == [(0, 10, 0, 10), (20, 40, 20, 30)]


def test_float_coords_are_truncated_to_int():
    quad = [[(10.9, 20.2), (110.1, 20.8), (110.0, 50.9), (10.0, 50.1)]]
    out = get_coordinates(quad)
    assert all(isinstance(v, int) for v in out[0])


def test_non_list_input_returns_empty():
    assert get_coordinates("not a list") == []
    assert get_coordinates(None) == []


def test_empty_input_returns_empty():
    assert get_coordinates([]) == []
