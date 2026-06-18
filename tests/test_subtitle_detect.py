"""Tests for backend.tools.subtitle_detect range/region logic (no OCR model)."""

from backend.tools.subtitle_detect import SubtitleDetect


def _bare_detector(sample_step=3):
    """A SubtitleDetect instance without running __init__ (no video needed)."""
    det = object.__new__(SubtitleDetect)
    det.sample_step = sample_step
    det.sub_areas = []
    return det


# --------------------------------------------------------------------------
# find_continuous_ranges_with_same_mask
# --------------------------------------------------------------------------

def test_runs_break_on_gap_and_mask_change():
    a = [(0, 10, 0, 5)]
    b = [(0, 99, 0, 5)]
    frame_boxes = {1: a, 2: a, 3: b, 7: a}
    # 1-2 share mask a; 3 changes mask; 7 after a gap.
    assert SubtitleDetect.find_continuous_ranges_with_same_mask(frame_boxes) == [
        (1, 2), (3, 3), (7, 7)
    ]


def test_runs_single_contiguous_block():
    a = [(0, 10, 0, 5)]
    assert SubtitleDetect.find_continuous_ranges_with_same_mask(
        {1: a, 2: a, 3: a}
    ) == [(1, 3)]


def test_runs_single_frame():
    assert SubtitleDetect.find_continuous_ranges_with_same_mask(
        {5: [(0, 10, 0, 5)]}
    ) == [(5, 5)]


# --------------------------------------------------------------------------
# split_range_by_scene
# --------------------------------------------------------------------------

def test_split_one_cut():
    assert SubtitleDetect.split_range_by_scene([(1, 10)], [5]) == [(1, 4), (5, 10)]


def test_split_multiple_cuts():
    assert SubtitleDetect.split_range_by_scene([(1, 20)], [5, 12]) == [
        (1, 4), (5, 11), (12, 20)
    ]


def test_split_no_cuts_unchanged():
    assert SubtitleDetect.split_range_by_scene([(1, 10)], []) == [(1, 10)]


def test_split_cut_outside_interval_ignored():
    assert SubtitleDetect.split_range_by_scene([(1, 10)], [50]) == [(1, 10)]


# --------------------------------------------------------------------------
# filter_and_merge_intervals
# --------------------------------------------------------------------------

def test_merge_empty():
    assert SubtitleDetect.filter_and_merge_intervals([], 10) == []


def test_merge_expands_single_point():
    # (5,5) widened around its midpoint up to target length; (30,60) untouched.
    assert SubtitleDetect.filter_and_merge_intervals([(5, 5), (30, 60)], 10) == [
        (1, 9), (30, 60)
    ]


def test_merge_adjacent_short_intervals():
    # Two short adjacent runs merge to satisfy the minimum length.
    assert SubtitleDetect.filter_and_merge_intervals([(1, 3), (4, 6)], 10) == [(1, 6)]


def test_merge_keeps_long_intervals_apart():
    out = SubtitleDetect.filter_and_merge_intervals([(1, 50), (60, 120)], 10)
    assert out == [(1, 50), (60, 120)]


def test_merge_single_point_never_goes_below_frame_one():
    # Regression: half=(11-1)//2=5 would push (1,1) to start -4; must clamp to 1.
    out = SubtitleDetect.filter_and_merge_intervals([(1, 1)], 11)
    assert out == [(1, 6)]
    assert all(start >= 1 for start, _ in out)


def test_merge_single_point_clamped_by_prev_end():
    # The widened single point cannot start before the previous interval ends.
    out = SubtitleDetect.filter_and_merge_intervals([(3, 8), (10, 10)], 11)
    assert out == [(3, 15)]          # (10,10) widened to start 9 (prev_end+1), then merged


def test_merge_single_point_clamped_by_next_start():
    # The widened single point cannot extend into the next interval.
    out = SubtitleDetect.filter_and_merge_intervals([(5, 5), (8, 20)], 11)
    assert out == [(1, 20)]          # (5,5) end clamped to 7 (next_start-1), then merged


# --------------------------------------------------------------------------
# are_similar
# --------------------------------------------------------------------------

def test_similar_within_tolerance():
    assert SubtitleDetect.are_similar((10, 100, 20, 40), (15, 105, 25, 45)) is True


def test_not_similar_beyond_tolerance():
    assert SubtitleDetect.are_similar((10, 100, 20, 40), (10, 100, 20, 100)) is False


# --------------------------------------------------------------------------
# unify_regions (instance method, pure)
# --------------------------------------------------------------------------

def test_unify_snaps_similar_boxes_to_previous():
    det = _bare_detector()
    raw = {1: [(10, 100, 20, 40)], 2: [(12, 102, 22, 42)]}
    unified = det.unify_regions(raw)
    # Frame 2's jittered box is replaced by frame 1's box.
    assert unified[2] == unified[1] == [(10, 100, 20, 40)]


def test_unify_keeps_distinct_boxes():
    det = _bare_detector()
    raw = {1: [(10, 100, 20, 40)], 2: [(10, 100, 20, 200)]}
    unified = det.unify_regions(raw)
    assert unified[2] == [(10, 100, 20, 200)]


def test_unify_empty():
    det = _bare_detector()
    assert det.unify_regions({}) == {}


def test_unify_multibox_positional_matching():
    # Boxes are matched POSITIONALLY by index across frames.
    det = _bare_detector()
    a = (10, 100, 20, 40)
    a_jittered = (12, 102, 22, 42)
    b = (200, 300, 20, 40)
    # Frame 2 gains a second box: box 0 snaps to frame 1's box; box 1 has no
    # reference (idx >= len(previous)) and is kept verbatim.
    unified = det.unify_regions({1: [a], 2: [a_jittered, b]})
    assert unified[2][0] == a
    assert unified[2][1] == b


# --------------------------------------------------------------------------
# _interpolate_samples (instance method, pure)
# --------------------------------------------------------------------------

def test_interpolate_fills_small_gap():
    det = _bare_detector(sample_step=3)  # max_gap = 6
    a = [(0, 10, 0, 5)]
    filled = det._interpolate_samples({1: a, 4: a})
    assert sorted(filled) == [1, 2, 3, 4]
    assert all(filled[k] == a for k in filled)


def test_interpolate_skips_large_gap():
    det = _bare_detector(sample_step=3)  # max_gap = 6
    a = [(0, 10, 0, 5)]
    filled = det._interpolate_samples({1: a, 10: a})
    assert sorted(filled) == [1, 10]


def test_interpolate_empty():
    det = _bare_detector()
    assert det._interpolate_samples({}) == {}


# --------------------------------------------------------------------------
# _same_caption_block / _propagate_within_spans
# (boxes are (xmin, xmax, ymin, ymax))
# --------------------------------------------------------------------------

# Two centred lines stacked with a 7px gap — one caption.
_UPPER = (100, 400, 850, 907)
_LOWER = (120, 380, 914, 963)
# Text 100+px above the caption (e.g. a sign) — not part of the block.
_SCENE = (150, 300, 620, 735)


def test_same_caption_block_stacked_lines():
    assert SubtitleDetect._same_caption_block(_UPPER, _LOWER) is True


def test_same_caption_block_rejects_far_apart():
    assert SubtitleDetect._same_caption_block(_UPPER, _SCENE) is False


def test_same_caption_block_rejects_no_horizontal_overlap():
    left = (100, 200, 850, 907)
    right = (400, 500, 914, 963)  # vertically adjacent but x-disjoint
    assert SubtitleDetect._same_caption_block(left, right) is False


def test_propagate_fills_missing_caption_line():
    det = _bare_detector()
    # Frame 2 dropped the upper line; it must be restored from its neighbours.
    out = det._propagate_within_spans({1: [_UPPER, _LOWER], 2: [_LOWER], 3: [_UPPER, _LOWER]})
    assert _UPPER in out[2] and _LOWER in out[2]
    # Canonical (sorted) order keeps all three frames one identical-mask run.
    assert out[1] == out[2] == out[3] == sorted([_UPPER, _LOWER])


def test_propagate_leaves_scene_text_frame_local():
    det = _bare_detector()
    out = det._propagate_within_spans({1: [_UPPER], 2: [_UPPER, _SCENE], 3: [_UPPER]})
    assert _SCENE in out[2]          # kept where actually detected
    assert _SCENE not in out[1]      # not smeared onto clean neighbours
    assert _SCENE not in out[3]


def test_propagate_does_not_cross_gap():
    det = _bare_detector()
    # Frames 1 and 10 are separate spans; the upper line must not jump the gap.
    out = det._propagate_within_spans({1: [_LOWER], 10: [_UPPER, _LOWER]})
    assert out[1] == [_LOWER]


def test_propagate_empty():
    assert _bare_detector()._propagate_within_spans({}) == {}


# --------------------------------------------------------------------------
# _choose_sample_step (reads only fps from a real tiny video)
# --------------------------------------------------------------------------

def test_choose_sample_step_low_fps(synth_video):
    path = synth_video(frames=4, fps=10)
    assert SubtitleDetect._choose_sample_step(path) == 2


def test_choose_sample_step_30fps(synth_video):
    path = synth_video(frames=4, fps=30)
    assert SubtitleDetect._choose_sample_step(path) == 3


def test_choose_sample_step_60fps(synth_video):
    path = synth_video(frames=4, fps=60)
    assert SubtitleDetect._choose_sample_step(path) == 4
