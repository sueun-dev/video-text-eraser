"""Tests for backend.config.getSttnMaxLoadNum and backend.tools.common_tools."""

import cv2
import numpy as np

from backend.config import config
from backend.tools.common_tools import (
    is_image_file,
    is_video_file,
    is_video_or_image,
    read_image,
)


# --------------------------------------------------------------------------
# config.getSttnMaxLoadNum
# --------------------------------------------------------------------------

def test_sttn_max_load_floor_is_stride_times_reference():
    snapshot = (
        config.sttnMaxLoadNum.value,
        config.sttnNeighborStride.value,
        config.sttnReferenceLength.value,
    )
    try:
        config.sttnMaxLoadNum.value = 10
        config.sttnNeighborStride.value = 5
        config.sttnReferenceLength.value = 10
        # max(10, 5*10) == 50
        assert config.getSttnMaxLoadNum() == 50
        config.sttnMaxLoadNum.value = 200
        assert config.getSttnMaxLoadNum() == 200
    finally:
        (
            config.sttnMaxLoadNum.value,
            config.sttnNeighborStride.value,
            config.sttnReferenceLength.value,
        ) = snapshot


# --------------------------------------------------------------------------
# file type predicates
# --------------------------------------------------------------------------

def test_is_video_file():
    assert is_video_file("clip.mp4")
    assert is_video_file("CLIP.MOV")          # case-insensitive
    assert not is_video_file("pic.png")


def test_is_image_file():
    assert is_image_file("pic.png")
    assert is_image_file("PHOTO.JPG")
    assert not is_image_file("clip.mp4")


def test_is_video_or_image():
    assert is_video_or_image("a.mkv")
    assert is_video_or_image("b.webp")
    assert not is_video_or_image("c.txt")
    assert not is_video_or_image("noext")


# --------------------------------------------------------------------------
# read_image
# --------------------------------------------------------------------------

def test_read_image_roundtrip(tmp_path):
    path = str(tmp_path / "x.png")
    img = np.full((20, 30, 3), 200, dtype=np.uint8)
    cv2.imwrite(path, img)
    out = read_image(path)
    assert out is not None
    assert out.shape == (20, 30, 3)


def test_read_image_strips_alpha(tmp_path):
    path = str(tmp_path / "rgba.png")
    rgba = np.full((10, 10, 4), 255, dtype=np.uint8)
    cv2.imwrite(path, rgba)
    out = read_image(path)
    assert out.shape[2] == 3            # BGRA -> BGR
