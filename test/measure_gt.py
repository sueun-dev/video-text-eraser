"""Ground-truth inpainting quality: PSNR/SSIM of the fill vs the true background.

The scene-content metrics in measure_quality.py cannot isolate inpaint artifacts
when the subtitle region itself contains motion (the frame-to-frame diff is
dominated by the scene, not the fill). This harness removes that confound by
synthesising the subtitle over a region of real video that is normally CLEAN, so
the untouched original frame IS the ground-truth answer for the fill.

    gen      : overlay synthetic subtitles on a clean band of a source video,
               writing the subtitled clip + saving the clean originals.
    score    : compare a processed clip's band against the saved clean originals
               (PSNR / SSIM / max error — higher PSNR & SSIM = better fill).
    temporal : Bonneel/Lai warping error — temporal flicker of the fill isolated
               from scene motion, vs the clean-vs-clean floor (lower = stabler).

Usage:
    .venv/bin/python test/measure_gt.py gen  [SRC]            -> /tmp/gt_subbed.mp4
    .venv/bin/python test/measure_gt.py score    PROCESSED
    .venv/bin/python test/measure_gt.py temporal PROCESSED
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# A clean band near the top of the frame (no real subtitles live here), so the
# untouched source pixels are exact ground truth for the synthetic fill.
ZONE = (90, 190, 40, 812)  # ymin, ymax, xmin, xmax
N_FRAMES = 48
SUBBED = "/tmp/gt_subbed.mp4"
CLEAN_NPY = "/tmp/gt_clean.npy"
ZONE_TXT = "/tmp/gt_zone.txt"
# Subtitle text changes every few frames, exposing different background over
# time (as real subtitles do) so temporal models have something to lean on.
LINES = [
    "The quick brown fox jumps",
    "over the lazy sleeping dog",
    "pack my box with five dozen",
    "liquor jugs the lazy way",
]


def _draw_subtitle(frame, text):
    """Burn a white, black-outlined subtitle (as hard subs look) centred in ZONE."""
    ymin, ymax, xmin, xmax = ZONE
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thick = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cx = xmin + (xmax - xmin - tw) // 2
    cy = ymin + (ymax - ymin + th) // 2
    cv2.putText(frame, text, (cx, cy), font, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (cx, cy), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def gen(src):
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(SUBBED, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    clean = []
    i = 0
    while i < N_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        clean.append(frame.copy())
        writer.write(_draw_subtitle(frame, LINES[(i // 6) % len(LINES)]))
        i += 1
    cap.release()
    writer.release()
    np.save(CLEAN_NPY, np.stack(clean))
    with open(ZONE_TXT, "w") as f:
        f.write(" ".join(str(v) for v in ZONE))
    print(f"wrote {SUBBED} ({i} frames @ {w}x{h}); clean GT -> {CLEAN_NPY}")
    print(f"subtitle zone (ymin ymax xmin xmax): {' '.join(str(v) for v in ZONE)}")
    print(f"\nNext: run removal on {SUBBED} with -c {' '.join(str(v) for v in ZONE)}")


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0**2 / mse)


def _ssim(a, b):
    """Global SSIM on grayscale (Wang et al. constants)."""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    )


def score(processed):
    clean = np.load(CLEAN_NPY)
    with open(ZONE_TXT) as f:
        ymin, ymax, xmin, xmax = (int(v) for v in f.read().split())
    cap = cv2.VideoCapture(processed)
    psnrs, ssims, maxerrs = [], [], []
    i = 0
    while i < len(clean):
        ok, frame = cap.read()
        if not ok:
            break
        gt = clean[i][ymin:ymax, xmin:xmax]
        out = frame[ymin:ymax, xmin:xmax]
        if out.shape != gt.shape:
            break
        psnrs.append(_psnr(gt, out))
        ssims.append(_ssim(gt, out))
        maxerrs.append(float(np.abs(gt.astype(np.int16) - out.astype(np.int16)).max()))
        i += 1
    cap.release()
    print(f"processed: {processed}  ({i} frames scored)")
    print(f"  PSNR (dB, higher=better) : {np.mean(psnrs):.2f}   min {np.min(psnrs):.2f}")
    print(f"  SSIM (0-1, higher=better): {np.mean(ssims):.4f}")
    print(f"  max abs error (0-255)    : {np.mean(maxerrs):.1f}")


def _occlusion_mask(flow_b, flow_f, gx, gy):
    """Forward-backward consistency mask (1 = reliable, no occlusion).

    ``flow_b`` is t -> t-1, ``flow_f`` is t-1 -> t. A pixel is reliable when the
    backward flow lands where the forward flow came from (Lai et al. 2018):
    ``|flow_b + flow_f(x + flow_b)| < 0.01*(|flow_b|^2 + |flow_f|^2) + 0.5``.
    """
    mapx = (gx + flow_b[..., 0]).astype(np.float32)
    mapy = (gy + flow_b[..., 1]).astype(np.float32)
    fwd_at = cv2.remap(flow_f, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    s = flow_b + fwd_at
    lhs = (s[..., 0] ** 2 + s[..., 1] ** 2)
    rhs = 0.01 * (flow_b[..., 0] ** 2 + flow_b[..., 1] ** 2
                  + fwd_at[..., 0] ** 2 + fwd_at[..., 1] ** 2) + 0.5
    return (lhs <= rhs).astype(np.float32)


def temporal(processed):
    """Bonneel/Lai warping error: temporal flicker isolated from scene motion.

    Naive frame-diff is dominated by scene motion. Warping each output frame to
    the previous using flow estimated on the CLEAN reference (saved by ``gen``)
    cancels real motion, leaving only fill flicker. Reported over the ZONE
    interior, alongside the clean-vs-clean self-warp floor (the noise level).
    """
    clean = np.load(CLEAN_NPY)
    with open(ZONE_TXT) as f:
        ymin, ymax, xmin, xmax = (int(v) for v in f.read().split())
    # Inset the zone so Farneback edge errors near the removed text do not leak in.
    y0, y1, x0, x1 = ymin + 6, ymax - 6, xmin + 6, xmax - 6
    cgray = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in clean]
    h, w = cgray[0].shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)

    cap = cv2.VideoCapture(processed)
    proc = []
    while len(proc) < len(clean):
        ok, fr = cap.read()
        if not ok:
            break
        proc.append(fr)
    cap.release()

    fb = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    proc_err, clean_err = [], []
    for t in range(1, min(len(proc), len(clean))):
        flow_b = cv2.calcOpticalFlowFarneback(cgray[t], cgray[t - 1], None, **fb)
        flow_f = cv2.calcOpticalFlowFarneback(cgray[t - 1], cgray[t], None, **fb)
        mask = _occlusion_mask(flow_b, flow_f, gx, gy)[y0:y1, x0:x1]
        denom = mask.sum() + 1e-6
        mapx = (gx + flow_b[..., 0]).astype(np.float32)
        mapy = (gy + flow_b[..., 1]).astype(np.float32)

        def warp_err(cur, prev):
            warped = cv2.remap(prev, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            d = (cur[y0:y1, x0:x1].astype(np.float32) - warped[y0:y1, x0:x1].astype(np.float32)) ** 2
            return float((d.mean(axis=2) * mask).sum() / denom)

        proc_err.append(warp_err(proc[t], proc[t - 1]))
        clean_err.append(warp_err(clean[t], clean[t - 1]))
    print(f"processed: {processed}  ({len(proc_err)} frame pairs)")
    print(f"  E_warp processed (lower=less flicker): {np.mean(proc_err):.2f}")
    print(f"  E_warp clean->clean floor (noise)    : {np.mean(clean_err):.2f}")
    print(f"  excess flicker over floor            : {np.mean(proc_err) - np.mean(clean_err):.2f}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("gen", "score", "temporal"):
        sys.exit(__doc__)
    if sys.argv[1] == "gen":
        gen(sys.argv[2] if len(sys.argv) > 2 else "test/test.mp4")
    elif sys.argv[1] == "temporal":
        temporal(sys.argv[2] if len(sys.argv) > 2 else "/tmp/gt_out.mp4")
    else:
        score(sys.argv[2] if len(sys.argv) > 2 else "/tmp/gt_out.mp4")


if __name__ == "__main__":
    main()
