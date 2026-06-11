# Video Text Eraser

AI-powered removal of hard-coded subtitles and text watermarks from videos and images — fully local, no third-party APIs.

> **Attribution** — This project is a refactored fork of
> [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
> (Apache License 2.0). All credit for the original architecture, models, and
> GUI goes to the upstream author. See [What's different](#whats-different)
> for the changes made in this fork.

## How it works

```
video ──▶ text detection (PaddleOCR, sampled + interpolated)
              │
        frame-run grouping (same mask, scene-cut aware)
              │
        mask → full-width band extraction (only the band is processed)
              │
        AI inpainting (STTN / LaMa / ProPainter)
              │
        x264 encode + original audio merged back
```

The key idea: hard subtitles change or disappear over time, so the background
behind them is visible in *other* frames. The temporal models (STTN,
ProPainter) attend to neighbouring and far-away reference frames to
reconstruct the true background instead of hallucinating it.

| Mode | Best for | Notes |
|------|----------|-------|
| `sttn-auto` (default) | live-action, speed | no OCR pass; erases the selected area in every frame |
| `sttn-det` | live-action, precision | OCR-driven; only touches frames that contain text |
| `lama` | animation, still images | single-frame model, no temporal context |
| `propainter` | violent camera motion | optical-flow based, heavy VRAM use |
| `opencv` | quick preview | classical Telea inpainting, lowest quality |

## What's different

Compared to upstream, this fork:

- **Refactored the entire core pipeline** (`backend/main.py`, detection,
  mask/band utilities, all five inpaint wrappers) — typed, documented,
  lint-clean, with the 200-line band-extraction routine decomposed into
  testable helpers.
- **Fixed six real bugs**, including two ProPainter-mode frame bugs (frames
  silently dropped from the output; wrong frame inpainted for single-frame
  batches), a CLI crash when `-o` was omitted, and a PaddleOCR failure when
  high-performance inference plugins are missing.
- **Runs natively on Apple Silicon** — torch MPS acceleration plus an FFmpeg
  resolution chain (bundled → PATH → `imageio-ffmpeg`) that fixes broken
  audio merging caused by the bundled x86-64-only binary.
- **Ships an end-to-end verification harness** (`test/verify_removal.py`)
  that proves a run removed *exactly* the subtitles: OCR re-detection must
  drop to zero, and pixels outside the inpaint band must be identical to the
  original up to a measured codec-noise baseline.

## Install

Requires Python 3.10+ (3.12 recommended).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt torch torchvision paddlepaddle
# NVIDIA GPU: install the CUDA builds of torch/paddlepaddle instead.
```

Model weights (~600 MB) are bundled in `backend/models/`.

## Usage

CLI:

```bash
.venv/bin/python backend/main.py \
  -i input.mp4 -o output.mp4 \
  --inpaint-mode sttn-det \
  -c 340 480 0 852          # subtitle area: ymin ymax xmin xmax (optional)
```

GUI:

```bash
.venv/bin/python gui.py
```

Verify a result:

```bash
.venv/bin/python test/verify_removal.py input.mp4 output.mp4
```

## License

[Apache License 2.0](LICENSE) — same as the upstream project it derives from.

---

## 한국어 안내

영상/이미지에 박힌(하드코딩된) 자막·텍스트 워터마크를 AI로 지우는 도구입니다.
서드파티 API 없이 전부 로컬에서 동작합니다.

이 프로젝트는 [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
(Apache 2.0)를 기반으로 한 포크이며, 다음이 다릅니다:

- 코어 파이프라인 전면 리팩토링 (타입힌트·문서화·린트 클린)
- 실제 버그 6건 수정 (ProPainter 프레임 유실 등)
- **Apple Silicon 네이티브 지원** (MPS 가속 + arm64 FFmpeg 자동 해결)
- "자막만 정확히 지웠는지" 증명하는 E2E 검증 스크립트 동봉

사용법은 위 [Usage](#usage) 섹션과 동일합니다.
