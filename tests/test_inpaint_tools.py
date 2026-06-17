"""Tests for backend.tools.inpaint_tools pure utilities."""

import cv2
import numpy as np

from backend.config import config
from backend.tools.inpaint_tools import (
    _align_to_multiple,
    batch_generator,
    composite_band,
    create_mask,
    expand_frame_ranges,
    get_inpaint_area_by_mask,
    guided_upsample,
    composite_run_pde,
    is_frame_number_in_ab_sections,
    pde_fill_strokes,
    refine_box_to_strokes,
    temporal_fuse_strokes,
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


def test_nonpositive_band_height_returns_no_bands():
    # A few-pixel-wide frame yields h<=0; must not crash (would zero-height crop).
    mask = create_mask((100, 200), [(40, 60, 50, 150)])
    assert get_inpaint_area_by_mask(200, 100, 0, mask) == []
    assert get_inpaint_area_by_mask(200, 100, -3, mask) == []


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


# --------------------------------------------------------------------------
# composite_band / refine_box_to_strokes
# --------------------------------------------------------------------------


def _band_with_text():
    """A flat-grey band with a bright horizontal bar standing in for a glyph."""
    band = np.full((60, 200, 3), 120, dtype=np.uint8)
    band[28:32, 40:160] = 240  # high-contrast "stroke"
    box = np.zeros((60, 200), dtype=np.uint8)
    box[20:40, 30:170] = 1  # filled detection box around the bar
    return band, box


def test_composite_band_keeps_pixels_outside_the_mask():
    orig = np.full((40, 50, 3), 100, dtype=np.uint8)
    restored = np.full((40, 50, 3), 0, dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    out = composite_band(orig, restored, mask, feather=0)
    # A corner far from the mask must be the untouched original.
    assert out[0, 0].tolist() == [100, 100, 100]
    # The mask centre takes the restored value.
    assert out[15, 15].tolist() == [0, 0, 0]


def test_composite_band_feather_is_a_soft_ramp():
    orig = np.zeros((40, 40, 3), dtype=np.uint8)
    restored = np.full((40, 40, 3), 255, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    out = composite_band(orig, restored, mask, feather=4)
    # Just outside the hard mask edge the alpha is partial, not 0 or 255.
    assert 0 < int(out[9, 20, 0]) < 255


def test_refine_box_to_strokes_shrinks_to_the_ink():
    band, box = _band_with_text()
    refined = refine_box_to_strokes(band, box)
    assert refined.sum() < box.sum()  # narrowed
    # Every true stroke pixel is still covered (no residual text).
    assert refined[28:32, 40:160].all()


def test_refine_strokes_preserves_background_between_letters():
    band, box = _band_with_text()
    restored = np.zeros_like(band)  # a "bad" fill, to expose any over-masking
    out = composite_band(band, restored, box, refine_strokes=True)
    # Background inside the box but clear of the stroke (+dilation+feather) is
    # kept from the original, not hallucinated.
    assert out[5, 5].tolist() == [120, 120, 120]      # well outside the box
    assert out[21, 100].tolist() == [120, 120, 120]   # 7px above the stroke
    assert out[30, 33].tolist() == [120, 120, 120]    # left of the stroke, in box
    # The stroke itself is replaced by the fill.
    assert int(out[30, 100, 0]) < 120


def test_refine_box_to_strokes_falls_back_when_no_strokes():
    flat = np.full((60, 200, 3), 120, dtype=np.uint8)  # no contrast at all
    box = np.zeros((60, 200), dtype=np.uint8)
    box[20:40, 30:170] = 1
    refined = refine_box_to_strokes(flat, box)
    # Degenerate segmentation must fall back to the full box, never erase it.
    assert refined.sum() == box.sum()


# --------------------------------------------------------------------------
# guided_upsample
# --------------------------------------------------------------------------


def test_guided_upsample_preserves_shape_and_dtype():
    restored = np.random.randint(0, 255, (40, 80, 3), dtype=np.uint8)
    guide = np.random.randint(0, 255, (40, 80, 3), dtype=np.uint8)
    out = guided_upsample(restored, guide)
    assert out.shape == restored.shape
    assert out.dtype == np.uint8


def test_guided_upsample_does_not_inject_guide_only_structure():
    # A flat fill guided by a frame whose strong edge is absent from the fill
    # must not grow that edge (the guide/fill covariance is ~0 there).
    restored = np.full((40, 80, 3), 120, dtype=np.uint8)
    guide = np.full((40, 80, 3), 120, dtype=np.uint8)
    guide[:, 40:] = 20  # strong vertical edge only in the guide
    out = guided_upsample(restored, guide)
    # The fill stays ~flat; no edge transferred from the guide.
    assert int(out.max()) - int(out.min()) <= 6


def test_guided_upsample_sharpens_shared_structure():
    # When the fill carries a (blurred) version of the guide's edge, guidance
    # sharpens it back toward the guide rather than leaving it smooth.
    guide = np.zeros((40, 80, 3), dtype=np.uint8)
    guide[:, 40:] = 200
    blurred = cv2.GaussianBlur(guide, (0, 0), 6)
    out = guided_upsample(blurred, guide)
    mid = 40
    # Edge contrast across the boundary is greater after guided sharpening.
    before = int(blurred[20, mid + 4, 0]) - int(blurred[20, mid - 4, 0])
    after = int(out[20, mid + 4, 0]) - int(out[20, mid - 4, 0])
    assert after >= before


# --------------------------------------------------------------------------
# pde_fill_strokes
# --------------------------------------------------------------------------


def test_pde_fill_removes_stroke_using_real_neighbours():
    # A flat band with a bright stroke: NS inpainting must replace the stroke
    # with the surrounding background value, not leave it.
    band = np.full((40, 80, 3), 100, dtype=np.uint8)
    band[18:22, 20:60] = 240
    ink = np.zeros((40, 80), dtype=np.uint8)
    ink[18:22, 20:60] = 1
    out = pde_fill_strokes(band, ink)
    assert abs(int(out[20, 40, 0]) - 100) <= 8   # stroke filled from neighbours
    assert out[0, 0].tolist() == [100, 100, 100]  # untouched elsewhere


def test_pde_fill_empty_mask_is_noop():
    band = np.random.randint(0, 255, (30, 50, 3), dtype=np.uint8)
    out = pde_fill_strokes(band, np.zeros((30, 50), dtype=np.uint8))
    assert np.array_equal(out, band)


def test_composite_band_accepts_precomputed_mask():
    orig = np.full((40, 50, 3), 100, dtype=np.uint8)
    restored = np.zeros((40, 50, 3), dtype=np.uint8)
    ink = np.zeros((40, 50), dtype=np.uint8)
    ink[10:20, 10:20] = 1
    out = composite_band(orig, restored, np.ones((40, 50), np.uint8),
                         precomputed_mask=ink)
    assert out[15, 15, 0] < 100          # restored inside the precomputed mask
    assert out[0, 0].tolist() == [100, 100, 100]  # original outside it


# --------------------------------------------------------------------------
# temporal_fuse_strokes / composite_run_pde
# --------------------------------------------------------------------------


def test_temporal_fuse_strokes_preserves_shape_and_background():
    # Three identical flat fills with one stroke region; fusion must keep shape
    # and leave the non-stroke background exactly as it was.
    fills = [np.full((40, 80, 3), 120, dtype=np.uint8) for _ in range(3)]
    origs = [np.full((40, 80, 3), 120, dtype=np.uint8) for _ in range(3)]
    masks = [np.zeros((40, 80), dtype=np.uint8) for _ in range(3)]
    for m in masks:
        m[18:22, 30:50] = 1
    out = temporal_fuse_strokes(fills, origs, masks, window=1)
    assert len(out) == 3
    assert out[0].shape == (40, 80, 3)
    assert out[1][0, 0].tolist() == [120, 120, 120]   # background untouched


def test_temporal_fuse_single_frame_is_noop():
    fills = [np.full((30, 50, 3), 100, dtype=np.uint8)]
    out = temporal_fuse_strokes(fills, list(fills), [np.zeros((30, 50), np.uint8)], window=2)
    assert np.array_equal(out[0], fills[0])


def test_composite_run_pde_removes_stroke_and_keeps_background():
    # A flat band with a bright stroke across three frames; the run composite
    # must fill the stroke (toward background) and keep pixels outside it.
    origs, models, box = [], [], np.ones((40, 80), np.uint8)
    for _ in range(3):
        band = np.full((40, 80, 3), 100, dtype=np.uint8)
        band[18:22, 20:60] = 240
        origs.append(band)
        models.append(np.full((40, 80, 3), 90, dtype=np.uint8))
    out = composite_run_pde(origs, models, box, ns_weight=0.5)
    assert len(out) == 3
    assert int(out[1][20, 40, 0]) < 200          # bright stroke removed
    assert out[1][0, 0].tolist() == [100, 100, 100]   # corner untouched


def test_refine_box_per_region_contrast():
    # A box with TWO regions: a crisp high-contrast bar and a faint low-contrast
    # bar. The crisp region must shrink to its stroke; the faint region must keep
    # its whole box (so it is fully removed, not under-segmented).
    band = np.full((60, 240, 3), 120, dtype=np.uint8)
    band[28:32, 20:80] = 245    # crisp/high-contrast stroke (region A)
    band[28:32, 160:220] = 132  # faint/low-contrast stroke (region B)
    box = np.zeros((60, 240), dtype=np.uint8)
    box[22:38, 10:90] = 1       # region A box
    box[22:38, 150:230] = 1     # region B box (separated -> distinct component)
    refined = refine_box_to_strokes(band, box)
    a = refined[22:38, 10:90]
    b = refined[22:38, 150:230]
    assert a.mean() < 0.6        # crisp region tightened to strokes
    assert b.mean() > 0.95       # faint region kept whole (full removal)
