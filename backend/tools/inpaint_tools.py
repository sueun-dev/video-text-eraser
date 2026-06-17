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


def refine_box_to_strokes(
    orig_band: np.ndarray, box_mask: np.ndarray, dilate: int = 3
) -> np.ndarray:
    """Tighten a filled-box text mask down to the actual glyph strokes.

    A subtitle box is mostly background: the ink covers only ~10-15% of it, and
    the rest is clean pixels that must NOT be hallucinated. Estimating the local
    background with a median filter and Otsu-thresholding the residual isolates
    the high-contrast strokes, so only ink is replaced while the inter-letter
    background stays pristine (measured +~3 dB PSNR vs filling the whole box).

    Otsu adapts the threshold to each subtitle's contrast, so it generalises
    across fonts/sizes. This is only safe where the content is guaranteed
    high-contrast text — the OCR-detection path — because a low-contrast,
    area-filling overlay (a semi-transparent watermark) could fall below the
    threshold and survive; callers on the user-region path must not enable it.
    """
    box = (box_mask > 0).astype(np.uint8)
    if box.ndim == 3:
        box = box[:, :, 0]
    if not box.any():
        return box
    gray = cv2.cvtColor(orig_band, cv2.COLOR_BGR2GRAY) if orig_band.ndim == 3 else orig_band
    # Median kernel must be odd and comfortably wider than a stroke; clamp so it
    # never exceeds the band on tiny crops.
    k = min(21, gray.shape[0] - 1 if gray.shape[0] % 2 == 0 else gray.shape[0],
            gray.shape[1] - 1 if gray.shape[1] % 2 == 0 else gray.shape[1])
    k = max(3, k if k % 2 == 1 else k - 1)
    stroke = cv2.absdiff(gray, cv2.medianBlur(gray, k))
    inside = stroke[box > 0]
    thr, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu returns the split point T with foreground strictly above it (OpenCV's
    # THRESH_BINARY semantics); `>=` would swallow the whole background at T=0.
    ink = (stroke > thr).astype(np.uint8)
    if dilate:
        ink = cv2.dilate(ink, np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8))
    refined = np.logical_and(ink, box).astype(np.uint8)
    # If segmentation collapses (no clear strokes), fall back to the full box so
    # text is never left behind.
    if refined.sum() < 0.02 * box.sum():
        return box
    return refined


def guided_upsample(restored: np.ndarray, guide: np.ndarray, radius: int = 4,
                    eps: float = 0.02**2) -> np.ndarray:
    """Sharpen a blurry fill toward the original's edge structure (He et al. 2013).

    STTN inpaints at 432x240 and the band is bilinearly upscaled ~2x, which
    smears the real high-frequency detail the fill should carry. A guided filter
    with the original band as guide re-injects that structure via a local linear
    model ``q = a*I + b`` (``a,b`` from windowed covariance), restoring texture
    without a new dependency (pure ``cv2.boxFilter``). It cannot resurrect text:
    the guide's text edges live where the fill is smooth background, so the local
    guide/fill covariance there is ~0 and nothing transfers — only structure
    common to both (the real background) is sharpened.
    """
    ksize = (radius * 2 + 1, radius * 2 + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return np.asarray(cv2.boxFilter(x.astype(np.float32), -1, ksize))  # type: ignore[call-overload]

    guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean_i = box(guide_gray)
    var_i = box(guide_gray * guide_gray) - mean_i * mean_i
    out = np.empty_like(restored, dtype=np.float32)
    for c in range(restored.shape[2]):
        p = restored[:, :, c].astype(np.float32) / 255.0
        mean_p = box(p)
        cov_ip = box(guide_gray * p) - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        out[:, :, c] = box(a) * guide_gray + box(b)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def pde_fill_strokes(band: np.ndarray, ink_mask: np.ndarray) -> np.ndarray:
    """Fill thin glyph strokes by interpolating the real surrounding pixels.

    Navier-Stokes inpainting (Bertalmío, Bertozzi & Sapiro 2001) solves a
    fluid-dynamics analogue of the image where intensity plays the role of a
    stream function: it propagates the surrounding isophotes (level lines) into
    the hole, minimising a transport of the Laplacian along them. Once the mask
    is the *thin* glyph strokes, the true background is locally smooth across
    each stroke, so this real-pixel interpolation recovers it far better than a
    downscaled model hallucination — and needs no model. Blended with the
    temporal model fill it adds spatial fidelity while the model keeps frames
    coherent.
    """
    m = (np.asarray(ink_mask) > 0).astype(np.uint8)
    if m.ndim == 3:
        m = m[:, :, 0]
    if not m.any():
        return band
    return cv2.inpaint(band, m, 3, cv2.INPAINT_NS)


def composite_band(
    orig_band: np.ndarray,
    restored_band: np.ndarray,
    mask_band: np.ndarray,
    feather: int = 10,
    refine_strokes: bool = False,
    precomputed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Blend a restored band back over the original via a feathered text mask.

    The inpaint models return a whole-band reconstruction, but only the masked
    (subtitle) pixels actually need replacing. Compositing with a soft alpha (a)
    keeps the original background outside the mask intact — so the model cannot
    degrade clean pixels — and (b) ramps the boundary so there is no hard seam or
    colour step where the fill meets the untouched frame.

    With ``refine_strokes`` the box mask is first narrowed to the glyph strokes
    (see ``refine_box_to_strokes``), so background between and around letters is
    preserved exactly instead of hallucinated. Only enable it on the
    OCR-detection path, where the content is guaranteed high-contrast text.
    ``precomputed_mask`` supplies an already-refined stroke mask so a caller that
    needs it (e.g. for a PDE pre-fill) does not segment it twice.
    """
    if precomputed_mask is not None:
        m = (np.asarray(precomputed_mask) > 0).astype(np.uint8)
        feather = 2
    else:
        m = (mask_band > 0).astype(np.uint8)
        if m.ndim == 3:
            m = m[:, :, 0]
        if refine_strokes:
            m = refine_box_to_strokes(orig_band, m)
            # Strokes are thin; the wide box feather would re-expand the mask, so
            # use a narrow ramp that only softens the glyph edge.
            feather = 2
    if m.ndim == 3:
        m = m[:, :, 0]
    m = m.astype(np.float32)
    if feather > 0:
        k = feather * 2 + 1
        alpha = np.clip(cv2.GaussianBlur(m, (k, k), 0) * 1.5, 0.0, 1.0)
    else:
        alpha = m
    alpha = alpha[:, :, None]
    blended = (
        orig_band.astype(np.float32) * (1 - alpha)
        + restored_band.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def composite_with_pde(
    orig_band: np.ndarray,
    model_fill: np.ndarray,
    box_mask: np.ndarray,
    ns_weight: float,
) -> np.ndarray:
    """Composite a model fill blended with a Navier-Stokes real-background fill.

    Segments the glyph strokes once, then blends the model's fill with a PDE
    interpolation of the real surrounding pixels (``ns_weight`` toward the PDE).
    The PDE adds spatial fidelity, the model keeps frames temporally coherent,
    and averaging their uncorrelated per-frame errors lowers flicker below
    either alone. Shared by the OCR-text inpaint wrappers (STTN-det/LaMa/
    ProPainter); the watermark/user-region path keeps the plain composite.
    """
    ink = refine_box_to_strokes(orig_band, box_mask)
    ns = pde_fill_strokes(orig_band, ink)
    fill = (
        ns.astype(np.float32) * ns_weight
        + model_fill.astype(np.float32) * (1.0 - ns_weight)
    )
    return composite_band(orig_band, fill, box_mask, precomputed_mask=ink)


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
    # A non-positive band height (e.g. a frame only a few pixels wide) would
    # produce a zero-height crop that crashes cv2.resize downstream.
    if h <= 0 or np.all(mask == 0):
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
