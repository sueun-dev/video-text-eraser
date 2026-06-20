"""Frame-by-frame removal audit: did EVERY subtitled frame get its text erased?

For a (source, processed) pair, OCR every frame of both. A frame counts as a
removal FAILURE when the source had a wide text box in the subtitle zone (lower
part of the frame) and the processed frame still has a wide text box overlapping
that same region. Cross-checking against the source suppresses scene-content
false positives (e.g. high-contrast facial features) that appear in both and are
not subtitles the tool targeted.

Usage:
    .venv/bin/python test/frame_residual_audit.py SRC PROC [ZONE_TOP_FRAC]
ZONE_TOP_FRAC (default 0.55): only boxes whose vertical center is below this
fraction of the height count as subtitle-zone candidates.
"""

import sys

import cv2

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from backend.tools.subtitle_detect import SubtitleDetect  # noqa: E402


def wide_zone_boxes(detector, path, zone_top):
    cap = cv2.VideoCapture(path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fb = detector.find_subtitle_frame_no()
    out = {}
    for frame_no, boxes in fb.items():
        # A subtitle LINE is wide, short (one/two lines, < ~90px tall) and
        # horizontal (wider than tall). This excludes tall scene structures and
        # facial-feature false positives that are not subtitles.
        keep = [
            (x0, x1, y0, y1) for (x0, x1, y0, y1) in boxes
            if (x1 - x0) > 40 and (y1 - y0) < 90 and (x1 - x0) > (y1 - y0)
            and (y0 + y1) / 2 >= zone_top * height
        ]
        if keep:
            out[frame_no] = keep
    return out


def overlaps(a, boxes):
    ax0, ax1, ay0, ay1 = a
    for (x0, x1, y0, y1) in boxes:
        if min(ax1, x1) - max(ax0, x0) > 0 and min(ay1, y1) - max(ay0, y0) > 0:
            return True
    return False


def main():
    src, proc = sys.argv[1], sys.argv[2]
    zone_top = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55
    src_boxes = wide_zone_boxes(SubtitleDetect(src), src, zone_top)
    proc_boxes = wide_zone_boxes(SubtitleDetect(proc), proc, zone_top)
    failures = []
    for frame_no, sboxes in src_boxes.items():
        pboxes = proc_boxes.get(frame_no, [])
        residual = [b for b in pboxes if overlaps(b, sboxes)]
        if residual:
            failures.append((frame_no, residual))
    name = proc.rsplit("/", 1)[-1]
    print(f"AUDIT {name}: subtitled frames={len(src_boxes)} "
          f"residual-failures={len(failures)} "
          f"removal_rate={1 - len(failures) / max(1, len(src_boxes)):.4f}")
    for frame_no, residual in failures[:15]:
        print(f"  FAIL frame {frame_no}: {residual[:2]}")


if __name__ == "__main__":
    main()
