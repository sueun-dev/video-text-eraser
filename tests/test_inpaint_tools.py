"""Tests for backend.tools.inpaint_tools pure utilities."""

import numpy as np

from backend.config import config
from backend.tools.inpaint_tools import (
    _align_to_multiple,
    batch_generator,
    create_mask,
    expand_frame_ranges,
    get_inpaint_area_by_mask,
    is_frame_number_in_ab_sections,
)


# --------------------------------------------------------------------------
# batch_generator
# --------------------------------------------------------------------------

def test_batch_generator_covers_all_items_once():
    data = list(range(10))
    batches = list(batch_generator(data, 4))
    flat = [x for b in batches for x in b]
    assert flat == data


def test_batch_generator_respects_max_size():
    batches = list(batch_generator(list(range(23)), 5))
    assert all(len(b) <= 5 for b in batches)
    assert sum(len(b) for b in batches) == 23


def test_batch_generator_avoids_tiny_trailing_batch():
    # 10 items, max 4 -> would leave a trailing batch of 2 (< half); the
    # generator shrinks batch size so the tail is at least half a batch.
    batches = list(batch_generator(list(range(10)), 4))
    assert batches[-1], "trailing batch must be non-empty"
    assert len(batches[-1]) >= 2


def test_batch_generator_single_item():
    assert list(batch_generator([42], 8)) == [[42]]


def test_batch_generator_exact_multiple():
    batches = list(batch_generator(list(range(12)), 4))
    assert [len(b) for b in batches] == [4, 4, 4]


# --------------------------------------------------------------------------
# is_frame_number_in_ab_sections
# --------------------------------------------------------------------------

def test_ab_sections_none_means_all():
    assert is_frame_number_in_ab_sections(5, None) is True


def test_ab_sections_empty_means_all():
    assert is_frame_number_in_ab_sections(5, []) is True


def test_ab_sections_inside_and_outside():
    sections = [range(0, 3), range(10, 12)]
    assert is_frame_number_in_ab_sections(2, sections) is True
    assert is_frame_number_in_ab_sections(11, sections) is True
    assert is_frame_number_in_ab_sections(3, sections) is False
    assert is_frame_number_in_ab_sections(5, sections) is False


# --------------------------------------------------------------------------
# create_mask
# --------------------------------------------------------------------------

def test_create_mask_shape_and_dtype():
    mask = create_mask((100, 200), [(40, 60, 50, 150)])
    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8


def test_create_mask_fills_box_with_padding():
    pad = config.subtitleAreaDeviationPixel.value
    # box (xmin,xmax,ymin,ymax) = (40,60,50,150)
    mask = create_mask((200, 100), [(40, 60, 50, 150)])
    assert mask[100, 50] == 255                  # inside
    assert mask[50 - pad + 1, 50] == 255         # within top padding
    assert mask[50 - pad - 5, 50] == 0           # beyond padding


def test_create_mask_clips_padding_at_border():
    # Box hugging the top-left corner must not raise and must clip to 0.
    mask = create_mask((50, 50), [(0, 5, 0, 5)])
    assert mask[0, 0] == 255


def test_create_mask_empty_boxes_is_blank():
    assert not create_mask((30, 30), []).any()
    assert not create_mask((30, 30), None).any()


# --------------------------------------------------------------------------
# get_inpaint_area_by_mask
# --------------------------------------------------------------------------

def test_band_is_full_width_and_exact_height():
    mask = create_mask((100, 200), [(40, 60, 50, 150)])
    bands = get_inpaint_area_by_mask(200, 100, 30, mask)
    assert len(bands) == 1
    ymin, ymax, xmin, xmax = bands[0]
    assert (xmin, xmax) == (0, 200)
    assert ymax - ymin == 30


def test_band_taller_than_island_fully_covers_it():
    mask = create_mask((100, 200), [(40, 60, 50, 150)])  # padded rows ~40..71
    ymin, ymax, _, _ = get_inpaint_area_by_mask(200, 100, 60, mask)[0]
    assert ymin <= 40 and ymax >= 71


def test_empty_mask_returns_no_bands():
    assert get_inpaint_area_by_mask(200, 100, 30, np.zeros((100, 200), np.uint8)) == []


def test_two_separated_islands_make_two_bands():
    # Two vertically separated text lines (boxes given as xmin,xmax,ymin,ymax).
    mask = create_mask((400, 200), [(50, 150, 20, 35), (50, 150, 300, 320)])
    bands = get_inpaint_area_by_mask(200, 400, 50, mask)
    assert len(bands) == 2


def test_multiple_alignment_divisible_by_8():
    mask = create_mask((100, 200), [(40, 60, 50, 150)])
    ymin, ymax, xmin, xmax = get_inpaint_area_by_mask(200, 100, 37, mask, multiple=8)[0]
    assert (ymax - ymin) % 8 == 0
    assert (xmax - xmin) % 8 == 0


def test_band_never_exceeds_frame_bounds():
    mask = create_mask((100, 200), [(85, 99, 50, 150)])  # near bottom edge
    ymin, ymax, _, _ = get_inpaint_area_by_mask(200, 100, 40, mask)[0]
    assert ymin >= 0 and ymax <= 100


def test_3d_mask_is_accepted():
    mask = create_mask((100, 200), [(40, 60, 50, 150)])[:, :, None]
    bands = get_inpaint_area_by_mask(200, 100, 30, mask)
    assert len(bands) == 1


def test_band_shorter_than_island_stays_centered():
    # Island spans rows ~20..150 (height ~130); band height 40 < island.
    mask = create_mask((300, 200), [(50, 150, 20, 150)])
    ymin, ymax, _, _ = get_inpaint_area_by_mask(200, 300, 40, mask)[0]
    assert ymax - ymin == 40
    island_center = (20 + 150) // 2
    assert ymin <= island_center <= ymax       # band kept over the island center


def test_alignment_near_bottom_edge_stays_in_bounds_and_divisible():
    # Island hugging the bottom edge with multiple=8 must not exceed H.
    mask = create_mask((100, 200), [(80, 99, 50, 150)])
    ymin, ymax, xmin, xmax = get_inpaint_area_by_mask(200, 100, 37, mask, multiple=8)[0]
    assert 0 <= ymin < ymax <= 100
    assert (ymax - ymin) % 8 == 0
    assert (xmax - xmin) % 8 == 0


def test_band_for_very_tall_island_clamped_to_frame():
    # Island taller than the frame-anchored band, near top.
    mask = create_mask((120, 200), [(5, 110, 40, 160)])
    ymin, ymax, _, _ = get_inpaint_area_by_mask(200, 120, 50, mask)[0]
    assert ymax - ymin == 50
    assert 0 <= ymin and ymax <= 120


def test_two_lines_merge_only_when_bridged():
    # Two stacked text lines with an empty gap -> two bands (split path).
    m = np.zeros((300, 200), np.uint8)
    m[20:40, 40:160] = 255
    m[60:80, 40:160] = 255
    assert len(get_inpaint_area_by_mask(200, 300, 100, m)) == 2
    # Add a vertical bridge between them -> they merge into one band.
    m[40:60, 95:105] = 255
    assert len(get_inpaint_area_by_mask(200, 300, 100, m)) == 1


def test_tiny_island_rejected_as_noise():
    # A blob smaller than _MIN_ISLAND_AREA (10 px) is ignored.
    n = np.zeros((100, 200), np.uint8)
    n[10:12, 10:13] = 255          # area = 2 * 3 = 6 < 10
    assert get_inpaint_area_by_mask(200, 100, 30, n) == []


def test_align_to_multiple_recenters_width():
    # Non-full-width band whose width is not a multiple of 8 is recentered.
    ymin, ymax, xmin, xmax = _align_to_multiple(0, 8, 5, 200, 8, frame_height=200)
    assert (ymax - ymin) % 8 == 0
    assert (xmax - xmin) % 8 == 0
    assert xmin == 6 and xmax == 198      # symmetric shrink around center


def test_align_to_multiple_symmetric_height_shrink():
    # Band touching both edges shrinks symmetrically to a multiple of 8.
    ymin, ymax, xmin, xmax = _align_to_multiple(0, 100, 0, 200, 8, frame_height=100)
    assert (ymax - ymin) % 8 == 0
    assert ymin == 2 and ymax == 98


# --------------------------------------------------------------------------
# expand_frame_ranges
# --------------------------------------------------------------------------

def test_expand_basic_padding():
    assert expand_frame_ranges([(10, 20), (40, 50)], 3, 3) == [(7, 23), (37, 53)]


def test_expand_clamps_start_to_one():
    assert expand_frame_ranges([(2, 5)], 10, 0) == [(1, 5)]


def test_expand_does_not_overlap_next_range():
    # Adjacent ranges (gap of 1) keep the earlier end where it is.
    assert expand_frame_ranges([(10, 20), (21, 30)], 3, 3) == [(7, 20), (21, 33)]


def test_expand_empty():
    assert expand_frame_ranges([], 3, 3) == []


def test_expand_single_range():
    assert expand_frame_ranges([(10, 20)], 5, 5) == [(5, 25)]


def test_expand_unsorted_input_is_sorted():
    out = expand_frame_ranges([(40, 50), (10, 20)], 2, 2)
    assert out == sorted(out)
