"""Mask construction and frame-range utilities shared by every inpaint mode."""

from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.config import config
from backend.tools.constant import Band, Box, FrameRange

# Connected components smaller than this many pixels are treated as noise.
_MIN_ISLAND_AREA = 10


def batch_generator(data: Sequence, max_batch_size: int) -> Iterator[Sequence]:
    """Yield evenly sized batches no larger than ``max_batch_size``.

    The batch size is shrunk until the trailing batch holds at least half a
    batch, which keeps GPU utilisation steady across batches.
    """
    n_samples = len(data)
    batch_size = max_batch_size
    # Shrink only when there is a *nonzero* trailing batch smaller than half a
    # batch. A remainder of 0 means perfectly even batches (the best case) and
    # must not trigger shrinking, otherwise round counts like 12/4 collapse to
    # size-1 batches.
    while 0 < n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1

    num_batches = n_samples // batch_size
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    remainder_start = num_batches * batch_size
    if remainder_start < n_samples:
        yield data[remainder_start:]


def create_mask(size: Tuple[int, int], boxes: Iterable[Box]) -> np.ndarray:
    """Render text boxes as a filled binary mask of the given (H, W) size.

    Every box is grown by ``subtitleAreaDeviationPixel`` on all sides so that
    anti-aliased glyph edges do not survive the inpaint pass.
    """
    mask = np.zeros(size, dtype="uint8")
    padding = config.subtitleAreaDeviationPixel.value
    for xmin, xmax, ymin, ymax in boxes or []:
        x1 = max(xmin - padding, 0)
        y1 = max(ymin - padding, 0)
        x2 = xmax + padding
        y2 = ymax + padding
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return mask


def _find_islands(binary_mask: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return (top_y, bottom_y, center_y) for each blob in the mask."""
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    islands = []
    for i in range(1, num_labels):  # label 0 is the background
        if stats[i, cv2.CC_STAT_AREA] < _MIN_ISLAND_AREA:
            continue
        top = stats[i, cv2.CC_STAT_TOP]
        bottom = top + stats[i, cv2.CC_STAT_HEIGHT]
        islands.append((top, bottom, int(centroids[i][1])))
    islands.sort(key=lambda island: island[2])
    return islands


def _group_islands(
    islands: List[Tuple[int, int, int]], binary_mask: np.ndarray, max_height: int
) -> List[List[Tuple[int, int, int]]]:
    """Greedily merge vertically adjacent islands into bands of <= max_height.

    Two islands join the same group when the combined span fits the band
    height and there are mask pixels bridging the vertical gap between them
    (multi-line subtitles), or when they overlap outright.
    """
    groups: List[List[Tuple[int, int, int]]] = [[islands[0]]]
    for island in islands[1:]:
        group = groups[-1]
        group_top = min(member[0] for member in group)
        group_bottom = max(member[1] for member in group)
        top, bottom, _ = island

        if group_bottom < top:
            connected = bool(np.any(binary_mask[group_bottom:top, :] > 0))
        else:
            connected = True

        merged_height = max(group_bottom, bottom) - min(group_top, top)
        if merged_height <= max_height and connected:
            group.append(island)
        else:
            groups.append([island])
    return groups


def _band_for_group(
    group: List[Tuple[int, int, int]], band_height: int, frame_height: int
) -> Tuple[int, int]:
    """Place a band of exactly ``band_height`` covering the group of islands."""
    group_top = min(member[0] for member in group)
    group_bottom = max(member[1] for member in group)
    center_y = sum(member[2] for member in group) // len(group)

    ymin = max(0, center_y - band_height // 2)
    ymax = ymin + band_height
    if ymax > frame_height:
        ymax = frame_height
        ymin = max(0, frame_height - band_height)

    # Re-anchor when the centered band clips islands it could fully contain.
    if (ymin > group_top or ymax < group_bottom) and group_bottom - group_top <= band_height:
        ymin = group_top
        ymax = ymin + band_height
        if ymax > frame_height:
            ymax = frame_height
            ymin = max(0, frame_height - band_height)
    elif ymin > group_top or ymax < group_bottom:
        # The group is taller than the band; keep its center covered.
        island_center = (group_top + group_bottom) // 2
        ymin = max(0, island_center - band_height // 2)
        ymax = ymin + band_height
        if ymax > frame_height:
            ymax = frame_height
            ymin = max(0, frame_height - band_height)
    return ymin, ymax


def _align_to_multiple(
    ymin: int, ymax: int, xmin: int, xmax: int, multiple: int, frame_height: int
) -> Band:
    """Snap band dimensions to a model-required multiple (e.g. 8 for ProPainter)."""
    height = ymax - ymin
    remainder = height % multiple
    if remainder != 0:
        grow = multiple - remainder
        center_y = (ymin + ymax) / 2
        if ymin - grow / 2 >= 0 and ymax + grow / 2 <= frame_height:
            ymin = int(center_y - height / 2 - grow / 2)
            ymax = int(center_y + height / 2 + grow / 2)
        elif height > multiple:
            ymin = int(center_y - (height - remainder) / 2)
            ymax = int(center_y + (height - remainder) / 2)
        elif ymax + grow <= frame_height:
            ymax += grow
        elif ymin - grow >= 0:
            ymin -= grow
        elif height > multiple:
            ymax = ymin + height - remainder

    width = xmax - xmin
    remainder_w = width % multiple
    if remainder_w != 0:
        center_x = (xmin + xmax) / 2
        xmin = int(center_x - (width - remainder_w) / 2)
        xmax = int(center_x + (width - remainder_w) / 2)
    return int(ymin), int(ymax), int(xmin), int(xmax)


def get_inpaint_area_by_mask(
    W: int, H: int, h: int, mask: np.ndarray, multiple: int = 1
) -> List[Band]:
    """Convert a subtitle mask into full-width horizontal bands of height ``h``.

    Inpaint models only see these bands instead of whole frames, which keeps
    inference cheap. Returns bands as (ymin, ymax, xmin, xmax).
    """
    if np.all(mask == 0):
        return []

    binary_mask = (mask > 0).astype(np.uint8) * 255
    if binary_mask.ndim == 3:
        binary_mask = binary_mask[:, :, 0]
    islands = _find_islands(binary_mask)
    if not islands:
        return []

    bands: List[Band] = []
    for group in _group_islands(islands, binary_mask, h):
        ymin, ymax = _band_for_group(group, h, H)
        band = (ymin, ymax, 0, W)
        if multiple > 1:
            band = _align_to_multiple(*band[:2], *band[2:], multiple, H)
        if band not in bands:
            bands.append(band)
    return bands


def expand_frame_ranges(
    frame_ranges: List[FrameRange], backward_frame_count: int, forward_frame_count: int
) -> List[FrameRange]:
    """Pad each detected subtitle range by a few frames on both sides.

    Padding hides OCR misses at fade-in/fade-out boundaries. Expansion never
    crosses into a neighbouring range.
    """
    if not frame_ranges:
        return []

    sorted_ranges = sorted(frame_ranges)
    expanded: List[FrameRange] = []
    for i, (start, end) in enumerate(sorted_ranges):
        new_start = max(1, start - backward_frame_count)
        new_end = end + forward_frame_count

        if i < len(sorted_ranges) - 1:
            next_start = sorted_ranges[i + 1][0]
            if new_end >= next_start:
                if next_start - end == 1:
                    new_end = end
                else:
                    new_end = min(new_end, next_start - 1)

        if expanded and new_start <= expanded[-1][1]:
            new_start = expanded[-1][1] + 1

        expanded.append((new_start, new_end) if new_start <= new_end else (start, end))
    return expanded


def is_frame_number_in_ab_sections(
    frame_no: int, ab_sections: Optional[List[range]]
) -> bool:
    """Whether a frame falls inside the user-selected A/B sections.

    ``None`` or an empty list means "process the whole video".
    """
    if not ab_sections:
        return True
    return any(frame_no in section for section in ab_sections)
