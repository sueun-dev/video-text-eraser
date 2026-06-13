"""Helpers for converting OCR detection output into rectangular boxes."""

from typing import List, Sequence

from .constant import Box


def get_coordinates(detection_polys: Sequence) -> List[Box]:
    """Convert OCR quadrilaterals into conservative axis-aligned boxes.

    Each detection is a 4-point polygon ordered top-left, top-right,
    bottom-right, bottom-left. The inner max/min is taken on purpose so the
    resulting rectangle never extends past a slanted polygon edge.
    """
    if not isinstance(detection_polys, list):
        return []
    boxes: List[Box] = []
    for poly in detection_polys:
        (top_left, top_right, bottom_right, bottom_left) = (list(p) for p in poly)
        xmin = int(max(top_left[0], bottom_left[0]))
        xmax = int(min(top_right[0], bottom_right[0]))
        ymin = int(max(top_left[1], top_right[1]))
        ymax = int(min(bottom_right[1], bottom_left[1]))
        boxes.append((xmin, xmax, ymin, ymax))
    return boxes
