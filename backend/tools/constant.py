"""Enums and shared geometry type aliases used across the pipeline."""

from enum import Enum, unique
from typing import Tuple

# Geometry aliases. These are pure annotations (never constructed or
# isinstance-checked); the two coordinate orders are intentional and
# load-bearing, so the order is documented here once as the single source of
# truth instead of being re-declared per module.
Box = Tuple[int, int, int, int]         # (xmin, xmax, ymin, ymax) — OCR / detection
Area = Tuple[int, int, int, int]        # (ymin, ymax, xmin, xmax) — user area
Band = Tuple[int, int, int, int]        # (ymin, ymax, xmin, xmax) — inpaint band
FrameRange = Tuple[int, int]            # inclusive (start, end)


@unique
class InpaintMode(Enum):
    """Image inpainting algorithm."""

    STTN_AUTO = "sttn-auto"
    STTN_DET = "sttn-det"
    LAMA = "lama"
    PROPAINTER = "propainter"
    OPENCV = "opencv"


@unique
class SubtitleDetectMode(Enum):
    """Subtitle text-detection model."""

    PP_OCRv5_MOBILE = "PP_OCRv5_MOBILE"
    PP_OCRv5_SERVER = "PP_OCRv5_SERVER"
