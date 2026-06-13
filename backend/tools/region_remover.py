"""Region-based watermark removal with LaMa, plus a static-region analyzer.

STTN reconstructs pixels from *other frames*, which fails for opaque
watermarks that cover the same spot in every frame. This module erases a
user-selected region in every frame with LaMa (pure spatial inpainting) and
feather-blends the result so no box seam survives.
"""

import os
import subprocess
import tempfile
import traceback
from typing import Callable, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.tools.video_io import FramePrefetcher, make_video_writer

Area = Tuple[int, int, int, int]  # (ymin, ymax, xmin, xmax) absolute pixels

# Extra pixels added around each area so anti-aliased edges are erased too.
DEFAULT_PADDING = 8
# Width of the soft alpha ramp used when compositing the inpainted patch.
DEFAULT_FEATHER = 12


def build_mask(
    height: int, width: int, areas: List[Area], padding: int = DEFAULT_PADDING
) -> np.ndarray:
    """Render areas as a binary uint8 mask, grown by ``padding`` pixels."""
    mask = np.zeros((height, width), np.uint8)
    for ymin, ymax, xmin, xmax in areas:
        y1 = max(0, ymin - padding)
        x1 = max(0, xmin - padding)
        y2 = min(height, ymax + padding)
        x2 = min(width, xmax + padding)
        mask[y1:y2, x1:x2] = 255
    return mask


def feather_alpha(mask: np.ndarray, feather: int = DEFAULT_FEATHER) -> np.ndarray:
    """Turn a hard mask into a soft float alpha map for seamless blending."""
    if feather <= 0:
        return (mask > 0).astype(np.float32)[:, :, None]
    kernel = feather * 2 + 1
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (kernel, kernel), 0)
    # Keep the core fully opaque; only the rim ramps down.
    alpha = np.clip(alpha * 1.5, 0.0, 1.0)
    return alpha[:, :, None]


class MotionAnalysis(TypedDict):
    """Result of :func:`analyze_region_motion` (uniform across all paths)."""

    static: bool
    region_change: float
    frame_change: float
    recommended_mode: str


def _motion_result(static: bool, region_change: float, frame_change: float) -> MotionAnalysis:
    return MotionAnalysis(
        static=bool(static),
        region_change=round(region_change, 2),
        frame_change=round(frame_change, 2),
        recommended_mode="lama-region" if static else "sttn-auto",
    )


def analyze_region_motion(
    video_path: str, areas: List[Area], samples: int = 12
) -> MotionAnalysis:
    """Estimate whether the selected region is a *fixed* overlay.

    Compares temporal change inside the region against the whole frame.
    A region that stays still while the frame moves is almost certainly an
    opaque watermark — STTN cannot help there, LaMa can. A too-short video is
    reported as static (single-frame inputs go to LaMa anyway).
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        cap.release()
        return _motion_result(True, 0.0, 0.0)

    indices = np.linspace(0, frame_count - 1, min(samples, frame_count), dtype=int)
    crops, frames = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(cv2.resize(gray, (160, 90)))
        region = [gray[y1:y2, x1:x2] for y1, y2, x1, x2 in areas]
        crops.append(region)
    cap.release()

    if len(frames) < 2:
        return _motion_result(True, 0.0, 0.0)

    region_change = float(np.mean([
        np.mean([np.abs(a - b).mean() for a, b in zip(crops[i], crops[i + 1])])
        for i in range(len(crops) - 1)
    ]))
    frame_change = float(np.mean([
        np.abs(frames[i] - frames[i + 1]).mean() for i in range(len(frames) - 1)
    ]))

    static = region_change < 2.0 or (
        frame_change > 0 and region_change < 0.35 * frame_change
    )
    return _motion_result(static, region_change, frame_change)


def merge_audio(source_video: str, silent_video: str, out_path: str) -> bool:
    """Copy the source audio track into the processed video. Returns success."""
    ffmpeg = FFmpegCLI.instance().ffmpeg_path
    audio_temp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
    audio_temp.close()
    try:
        subprocess.check_output(
            [ffmpeg, "-y", "-i", source_video, "-acodec", "copy", "-vn",
             "-loglevel", "error", audio_temp.name],
            stdin=subprocess.DEVNULL, timeout=600,
        )
        subprocess.check_output(
            [ffmpeg, "-y", "-i", silent_video, "-i", audio_temp.name,
             "-vcodec", "copy", "-acodec", "copy",
             "-loglevel", "error", out_path],
            stdin=subprocess.DEVNULL, timeout=600,
        )
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(audio_temp.name):
            try:
                os.remove(audio_temp.name)
            except Exception:
                pass


def remove_regions_video(
    video_path: str,
    out_path: str,
    areas: List[Area],
    lama_inpaint,
    padding: int = DEFAULT_PADDING,
    feather: int = DEFAULT_FEATHER,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    """Erase the given areas from every frame of a video using LaMa.

    The original audio track is preserved. ``lama_inpaint`` is a constructed
    :class:`backend.inpaint.lama_inpaint.LamaInpaint` instance.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    mask = build_mask(height, width, areas, padding)
    alpha = feather_alpha(mask, feather)

    silent_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    silent_temp.close()
    writer = make_video_writer(silent_temp.name, fps, (width, height))

    reader = FramePrefetcher(cap)
    done = 0
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            restored = lama_inpaint.inpaint(frame, mask)
            blended = (
                frame.astype(np.float32) * (1 - alpha)
                + restored.astype(np.float32) * alpha
            )
            writer.write(np.clip(blended, 0, 255).astype(np.uint8))
            done += 1
            if progress_cb:
                progress_cb(min(99, int(done / frame_count * 100)))
    finally:
        reader.release()
        writer.release()

    if not merge_audio(video_path, silent_temp.name, out_path):
        # No audio track or mux failure: ship the silent video.
        os.replace(silent_temp.name, out_path)
    elif os.path.exists(silent_temp.name):
        os.remove(silent_temp.name)

    if progress_cb:
        progress_cb(100)
    return out_path


def remove_regions_image(
    image: np.ndarray,
    areas: List[Area],
    lama_inpaint,
    padding: int = DEFAULT_PADDING,
    feather: int = DEFAULT_FEATHER,
) -> np.ndarray:
    """Erase the given areas from a single image using LaMa."""
    height, width = image.shape[:2]
    mask = build_mask(height, width, areas, padding)
    alpha = feather_alpha(mask, feather)
    restored = lama_inpaint.inpaint(image, mask)
    blended = (
        image.astype(np.float32) * (1 - alpha)
        + restored.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
