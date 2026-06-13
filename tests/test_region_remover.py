"""Tests for backend.tools.region_remover pure helpers + motion analyzer."""

import numpy as np

from backend.tools.region_remover import (
    analyze_region_motion,
    build_mask,
    feather_alpha,
)


# --------------------------------------------------------------------------
# build_mask  (areas are absolute ymin, ymax, xmin, xmax)
# --------------------------------------------------------------------------

def test_build_mask_fills_padded_region():
    mask = build_mask(100, 200, [(30, 60, 50, 150)], padding=5)
    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8
    assert mask[45, 100] == 255          # inside box
    assert mask[27, 100] == 255          # inside padding (30-5=25)
    assert mask[20, 100] == 0            # beyond padding


def test_build_mask_clips_at_borders():
    mask = build_mask(50, 50, [(0, 10, 0, 10)], padding=8)
    assert mask[0, 0] == 255             # no negative-index wraparound


def test_build_mask_multiple_areas():
    mask = build_mask(200, 200, [(10, 20, 10, 20), (100, 120, 100, 120)], padding=2)
    assert mask[15, 15] == 255
    assert mask[110, 110] == 255
    assert mask[60, 60] == 0


def test_build_mask_no_areas_blank():
    assert not build_mask(40, 40, [], padding=4).any()


# --------------------------------------------------------------------------
# feather_alpha
# --------------------------------------------------------------------------

def test_feather_alpha_shape_and_range():
    mask = build_mask(100, 200, [(30, 60, 50, 150)], padding=5)
    alpha = feather_alpha(mask, feather=6)
    assert alpha.shape == (100, 200, 1)
    assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0


def test_feather_alpha_core_opaque_edge_ramps():
    mask = build_mask(100, 200, [(30, 60, 50, 150)], padding=5)
    alpha = feather_alpha(mask, feather=6)
    assert alpha[45, 100, 0] > 0.99      # core fully opaque
    assert alpha[5, 5, 0] < 0.01         # far outside fully transparent


def test_feather_zero_is_hard_mask():
    mask = build_mask(100, 200, [(30, 60, 50, 150)], padding=0)
    alpha = feather_alpha(mask, feather=0)
    assert set(np.unique(alpha)).issubset({0.0, 1.0})


# --------------------------------------------------------------------------
# analyze_region_motion  (needs a tiny synthetic video)
# --------------------------------------------------------------------------

def test_analyze_detects_static_overlay(synth_video):
    wm = (10, 40, 100, 150)  # y1,y2,x1,x2 fixed opaque box every frame
    path = synth_video(frames=24, w=160, h=90, watermark=wm)
    # area in (ymin,ymax,xmin,xmax) order
    result = analyze_region_motion(path, [(10, 40, 100, 150)])
    assert result["static"] is True
    assert result["recommended_mode"] == "lama-region"


def test_analyze_detects_moving_region(synth_video):
    # No watermark: the scrolling gradient makes every region change over time.
    path = synth_video(frames=24, w=160, h=90, watermark=None)
    result = analyze_region_motion(path, [(10, 40, 100, 150)])
    assert result["static"] is False
    assert result["recommended_mode"] == "sttn-auto"


def test_analyze_single_frame_video_is_static(synth_video):
    path = synth_video(frames=1, w=160, h=90)
    result = analyze_region_motion(path, [(10, 40, 100, 150)])
    assert result["static"] is True
