"""Integration check: run every inpaint mode and confirm it removes subtitles.

Runs each --inpaint-mode on a short clip and OCRs the subtitle region of the
output to confirm the text is gone and the geometry is preserved. This catches
regressions in any single mode's wrapper (e.g. a tensor-variable rename that
breaks ProPainter) that unit tests cannot see.

Usage:
    .venv/bin/python test/verify_all_modes.py [CLIP] [YMIN YMAX XMIN XMAX]

CLIP defaults to a 2.4s slice of test/test.mp4 (created on the fly). ProPainter
on CPU is slow (minutes); pass MODES env to limit, e.g. MODES=opencv,lama.
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from backend.tools.ffmpeg_cli import FFmpegCLI  # noqa: E402
from backend.tools.subtitle_detect import SubtitleDetect  # noqa: E402

ALL_MODES = ["opencv", "sttn-auto", "sttn-det", "lama", "propainter"]
MODES = os.environ.get("MODES", ",".join(ALL_MODES)).split(",")


def make_clip() -> str:
    """Slice a short subtitled segment from the bundled test clip."""
    out = os.path.join(tempfile.gettempdir(), "vsr_modes_clip.mp4")
    ffmpeg = FFmpegCLI.instance().ffmpeg_path
    subprocess.run(
        [ffmpeg, "-y", "-i", "test/test.mp4", "-ss", "1", "-t", "2.4",
         "-an", "-loglevel", "error", out],
        check=True,
    )
    return out


def count_boxes(detector: SubtitleDetect, path: str):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    boxes, frames, i = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        frames = i
        if i % 10 == 0:
            boxes += len(detector.detect_subtitle(frame))
    cap.release()
    return boxes, frames, width


def main() -> int:
    clip = sys.argv[1] if len(sys.argv) > 1 else make_clip()
    if len(sys.argv) >= 6:
        area = tuple(int(a) for a in sys.argv[2:6])
    else:
        area = (340, 480, 0, 852)

    detector = SubtitleDetect(clip, [area])
    orig_boxes, orig_frames, width = count_boxes(detector, clip)
    print(f"clip: {orig_frames} frames @ {width}px, {orig_boxes} subtitle boxes")
    if orig_boxes == 0:
        sys.exit("no subtitles detected in the clip; cannot verify removal")

    failures = []
    for mode in MODES:
        out = os.path.join(tempfile.gettempdir(), f"vsr_modes_{mode}.mp4")
        proc = subprocess.run(
            [sys.executable, "backend/main.py", "-i", clip, "-o", out,
             "--inpaint-mode", mode,
             "-c", *(str(v) for v in (area[0], area[1], area[2], area[3]))],
            env={**os.environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not os.path.exists(out):
            print(f"  {mode:10s}: FAIL (run error)\n{proc.stderr[-400:]}")
            failures.append(mode)
            continue
        boxes, frames, w = count_boxes(detector, out)
        removed = boxes <= orig_boxes * 0.2
        geom = w == width and frames >= orig_frames - 2
        ok = removed and geom
        print(f"  {mode:10s}: {boxes} boxes remain, {frames} frames, {w}px -> "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(mode)

    if failures:
        sys.exit(f"FAILED modes: {failures}")
    print("ALL MODES PASSED")
    return 0


if __name__ == "__main__":
    main()
