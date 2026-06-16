"""Objective before/after quality metrics for subtitle/watermark removal.

Quantifies the dimensions that matter for inpaint quality so any change can be
compared numerically, not just by eye:

  [removal]    OCR text boxes remaining in the subtitle area (lower = cleaner).
  [background] mean abs pixel diff OUTSIDE the processing band vs the original,
               relative to a codec-only re-encode baseline (≈1.0 = untouched).
  [seam]       the strongest HORIZONTAL edge the processing introduced near the
               band boundary that was not in the original (lower = less visible
               seam / color step). This is the key blending-quality metric.
  [flicker]    frame-to-frame mean abs diff INSIDE the subtitle band (lower =
               more temporally stable fill, fewer flickering artifacts).

Usage:
    .venv/bin/python test/measure_quality.py ORIGINAL PROCESSED [YMIN YMAX XMIN XMAX]
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.tools.subtitle_detect import SubtitleDetect  # noqa: E402

SAMPLE_EVERY = 5


def _row_gradient(frame):
    """Per-row-boundary mean |vertical gradient| (length H-1)."""
    return np.abs(np.diff(frame.astype(np.int16), axis=0)).mean(axis=(1, 2))


def _codec_baseline(original, sub_top):
    """Mean abs diff above the band caused by re-encoding alone."""
    cap = cv2.VideoCapture(original)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    path = os.path.join(tempfile.gettempdir(), "vsr_quality_baseline.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(80):
        ok, fr = cap.read()
        if not ok:
            break
        writer.write(fr)
    writer.release()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    base = cv2.VideoCapture(path)
    diffs = []
    while True:
        ok1, a = cap.read()
        ok2, b = base.read()
        if not ok1 or not ok2:
            break
        diffs.append(np.abs(a[:sub_top].astype(np.int16) - b[:sub_top].astype(np.int16)).mean())
    cap.release()
    base.release()
    os.remove(path)
    return float(np.mean(diffs)) if diffs else 1.0


def measure(original, processed, sub_area):
    ymin, ymax, xmin, xmax = sub_area
    detector = SubtitleDetect(original, [sub_area])

    co = cv2.VideoCapture(original)
    cp = cv2.VideoCapture(processed)
    height = int(co.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Seam zone: the band can extend well above the subtitle area; scan from
    # there down to the subtitle bottom.
    zone_lo = max(1, ymin - 160)
    zone_hi = min(height - 1, ymax)

    boxes_before = boxes_after = 0
    background_diffs, seam_scores = [], []
    prev_proc_region = None
    flicker = []
    n = 0
    while True:
        ok1, a = co.read()
        ok2, b = cp.read()
        if not ok1 or not ok2:
            break
        n += 1
        if n % SAMPLE_EVERY:
            continue
        boxes_before += len(detector.detect_subtitle(a))
        boxes_after += len(detector.detect_subtitle(b))
        background_diffs.append(
            np.abs(a[:zone_lo].astype(np.int16) - b[:zone_lo].astype(np.int16)).mean()
        )
        # Seam: rows where the processed frame has MORE horizontal edge than the
        # original within the band zone (an introduced step/seam).
        excess = (_row_gradient(b) - _row_gradient(a))[zone_lo:zone_hi]
        seam_scores.append(float(excess.max()) if excess.size else 0.0)
        # Temporal flicker inside the band.
        region = cv2.cvtColor(b[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev_proc_region is not None and prev_proc_region.shape == region.shape:
            flicker.append(float(np.abs(region - prev_proc_region).mean()))
        prev_proc_region = region
    co.release()
    cp.release()

    base = _codec_baseline(original, zone_lo)
    removal = 1.0 - (boxes_after / boxes_before) if boxes_before else 1.0
    return {
        "removal_rate": round(removal, 3),
        "boxes_before": boxes_before,
        "boxes_after": boxes_after,
        "background_ratio_vs_codec": round(float(np.mean(background_diffs)) / max(base, 1e-6), 2),
        "seam_score": round(float(np.mean(seam_scores)), 2),
        "seam_score_max": round(float(np.max(seam_scores)) if seam_scores else 0.0, 2),
        "flicker": round(float(np.mean(flicker)) if flicker else 0.0, 2),
        "frames_sampled": len(background_diffs),
    }


def main():
    original = sys.argv[1] if len(sys.argv) > 1 else "test/test.mp4"
    processed = sys.argv[2] if len(sys.argv) > 2 else "/tmp/test_no_sub.mp4"
    sub_area = (
        tuple(int(a) for a in sys.argv[3:7]) if len(sys.argv) >= 7 else (340, 480, 0, 852)
    )
    m = measure(original, processed, sub_area)
    print(f"original : {original}")
    print(f"processed: {processed}")
    print(f"area     : {sub_area}\n")
    for k, v in m.items():
        print(f"  {k:28s}: {v}")
    print("\nLower seam_score & flicker = better blending/stability; "
          "removal_rate ~1.0 and background_ratio ~1.0 = clean.")


if __name__ == "__main__":
    main()
