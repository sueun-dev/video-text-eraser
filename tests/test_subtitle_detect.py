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
