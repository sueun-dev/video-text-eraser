"""E2E verification: did we erase exactly the subtitles and nothing else?

Checks, on sampled frames of (original, processed):
  1. Stream geometry survives (resolution / frame count / fps).
  2. OCR text detections inside the subtitle area drop to (near) zero.
  3. Pixels outside the inpaint band are untouched up to codec noise.
  4. Pixels inside the subtitle area actually changed on subtitled frames.
Saves side-by-side before/after images for visual inspection.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ORIGINAL = sys.argv[1] if len(sys.argv) > 1 else "test/test.mp4"
PROCESSED = sys.argv[2] if len(sys.argv) > 2 else "/tmp/test_no_sub.mp4"
# Subtitle area given to the remover: (ymin, ymax, xmin, xmax). Override for
# non-default clips:
#   verify_removal.py ORIG PROC YMIN YMAX XMIN XMAX [UNTOUCHED_BELOW_Y]
if len(sys.argv) >= 7:
    SUB_AREA = tuple(int(a) for a in sys.argv[3:7])
else:
    SUB_AREA = (340, 480, 0, 852)   # the bundled 852x480 test/test.mp4 clip
# Rows above this Y are expected codec-identical. The inpaint band extends
# ABOVE the subtitle area (STTN uses ~W*5/18), so this must sit safely above
# the band's top, not at the subtitle area's top. Caller can override (arg 8);
# default is tuned for the bundled clip + sttn-det.
UNTOUCHED_BELOW_Y = int(sys.argv[7]) if len(sys.argv) >= 8 else 240
SAMPLE_EVERY = 25
# Headroom over the measured codec-only baseline before we call it an edit.
CODEC_NOISE_HEADROOM = 1.25


def measure_codec_baseline(original: str, n_frames: int = 100) -> float:
    """Mean abs diff caused by re-encoding alone (no inpainting at all)."""
    cap = cv2.VideoCapture(original)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    baseline_path = "/tmp/vsr_codec_baseline.mp4"
    writer = cv2.VideoWriter(baseline_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_base = cv2.VideoCapture(baseline_path)
    diffs = []
    while True:
        ok1, f1 = cap.read()
        ok2, f2 = cap_base.read()
        if not ok1 or not ok2:
            break
        diffs.append(
            np.abs(
                f1[:UNTOUCHED_BELOW_Y].astype(np.int16)
                - f2[:UNTOUCHED_BELOW_Y].astype(np.int16)
            ).mean()
        )
    cap.release()
    cap_base.release()
    os.remove(baseline_path)
    return float(np.mean(diffs))


def open_or_die(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"FAIL: cannot open {path}")
    return cap


def geometry(cap):
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        round(cap.get(cv2.CAP_PROP_FPS), 2),
    )


def main():
    codec_noise_limit = measure_codec_baseline(ORIGINAL) * CODEC_NOISE_HEADROOM
    print(f"[0] codec-only noise baseline -> limit {codec_noise_limit:.3f}")

    cap_in = open_or_die(ORIGINAL)
    cap_out = open_or_die(PROCESSED)

    geo_in, geo_out = geometry(cap_in), geometry(cap_out)
    print(f"[1] geometry original={geo_in} processed={geo_out}")
    assert geo_in[:2] == geo_out[:2], "resolution changed!"
    assert abs(geo_in[2] - geo_out[2]) <= 2, "frame count changed!"
    print("    PASS: geometry preserved")

    # Fail loudly if SUB_AREA / UNTOUCHED_BELOW_Y do not fit this clip, instead
    # of silently producing meaningless results on a mismatched resolution.
    width, height = geo_in[0], geo_in[1]
    ymin, ymax, xmin, xmax = SUB_AREA
    assert 0 <= ymin < ymax <= height and 0 <= xmin < xmax <= width, (
        f"SUB_AREA {SUB_AREA} does not fit the {width}x{height} clip; pass "
        f"YMIN YMAX XMIN XMAX on the CLI for this video"
    )
    assert 0 < UNTOUCHED_BELOW_Y <= ymin, (
        f"UNTOUCHED_BELOW_Y={UNTOUCHED_BELOW_Y} must sit above the subtitle area "
        f"top ({ymin}); pass it as arg 8 for this video"
    )

    from backend.tools.subtitle_detect import SubtitleDetect

    detector = SubtitleDetect(ORIGINAL, [SUB_AREA])

    frame_no = 0
    texts_before = texts_after = 0
    subtitled_frames = changed_inside = 0
    outside_diffs = []
    pairs_for_save = []

    while True:
        ok_in, frame_in = cap_in.read()
        ok_out, frame_out = cap_out.read()
        if not ok_in or not ok_out:
            break
        frame_no += 1
        if frame_no % SAMPLE_EVERY != 0:
            continue

        boxes_in = detector.detect_subtitle(frame_in)
        boxes_out = detector.detect_subtitle(frame_out)
        texts_before += len(boxes_in)
        texts_after += len(boxes_out)

        top_in = frame_in[:UNTOUCHED_BELOW_Y].astype(np.int16)
        top_out = frame_out[:UNTOUCHED_BELOW_Y].astype(np.int16)
        outside_diffs.append(np.abs(top_in - top_out).mean())

        if boxes_in:
            subtitled_frames += 1
            ymin, ymax, xmin, xmax = SUB_AREA
            area_in = frame_in[ymin:ymax, xmin:xmax].astype(np.int16)
            area_out = frame_out[ymin:ymax, xmin:xmax].astype(np.int16)
            if np.abs(area_in - area_out).mean() > codec_noise_limit:
                changed_inside += 1
            if len(pairs_for_save) < 3:
                pairs_for_save.append((frame_no, frame_in.copy(), frame_out.copy()))

    print(f"[2] OCR text boxes in subtitle area (sampled): before={texts_before} after={texts_after}")
    assert texts_before > 0, "test invalid: no subtitles found in original samples"
    removal_rate = 1 - texts_after / texts_before
    print(f"    removal rate: {removal_rate:.1%}")
    assert removal_rate >= 0.95, f"FAIL: too much text survived ({texts_after}/{texts_before})"
    print("    PASS: subtitle text gone")

    mean_outside = float(np.mean(outside_diffs))
    max_outside = float(np.max(outside_diffs))
    print(f"[3] untouched region diff (rows 0..{UNTOUCHED_BELOW_Y}): mean={mean_outside:.3f} max={max_outside:.3f}")
    assert max_outside <= codec_noise_limit, "FAIL: pixels outside the subtitle band were modified"
    print("    PASS: everything above the subtitle band is codec-identical")

    print(f"[4] subtitled sample frames: {subtitled_frames}, with real change inside area: {changed_inside}")
    assert changed_inside == subtitled_frames, "FAIL: some subtitled frames were not inpainted"
    print("    PASS: every subtitled sample was actually inpainted")

    os.makedirs("/tmp/vsr_verify", exist_ok=True)
    for no, before, after in pairs_for_save:
        side_by_side = np.vstack([before, after])
        cv2.imwrite(f"/tmp/vsr_verify/frame_{no:04d}_before_after.png", side_by_side)
    print(f"saved {len(pairs_for_save)} before/after comparisons to /tmp/vsr_verify/")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
