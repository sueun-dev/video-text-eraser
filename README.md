# Video Text Eraser

영상과 이미지에 박혀 있는(하드코딩된) 자막·텍스트 워터마크를 AI로 지우는 도구입니다.
서드파티 API 없이 **전부 로컬에서** 동작하며, Apple Silicon에 네이티브로 대응합니다.

> **출처 (Attribution)** — 이 프로젝트는
> [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
> (Apache License 2.0)를 리팩토링한 포크입니다. 원본의 아키텍처·모델·GUI에 대한
> 모든 공로는 원작자에게 있습니다. 이 포크에서 바뀐 점은
> [원본 대비 변경점](#원본-대비-변경점)을 참고하세요.

## 동작 원리

```
영상 ──▶ 텍스트 검출 (PaddleOCR, 샘플링 + 보간)
            │
       연속 구간 묶기 (동일 마스크 단위, 장면 전환 인식)
            │
       마스크 → 전체 너비 밴드 추출 (밴드 부분만 처리)
            │
       AI 인페인팅 (STTN / LaMa / ProPainter)
            │
       x264 인코딩 + 원본 오디오 재병합
```

핵심 아이디어는 이렇습니다. 하드 자막은 시간이 지나면 바뀌거나 사라지므로,
자막 뒤에 가려진 배경은 **다른 프레임에서는 드러나** 있습니다. 시간축 기반
모델(STTN, ProPainter)은 인접 프레임과 멀리 떨어진 참조 프레임을 함께 보고
배경을 *지어내는* 대신 실제 배경을 복원합니다.

| 모드 | 적합한 대상 | 특징 |
|------|-----------|------|
| `sttn-auto` (기본값) | 실사 영상, 속도 우선 | OCR 생략; 선택 영역을 모든 프레임에서 제거 |
| `sttn-det` | 실사 영상, 정밀도 우선 | OCR 기반; 텍스트가 있는 프레임만 건드림 |
| `lama` | 애니메이션, 정지 이미지 | 단일 프레임 모델, 시간축 정보 없음 |
| `propainter` | 카메라 움직임이 격렬한 영상 | 광학 흐름 기반, VRAM 사용량 큼 |
| `opencv` | 빠른 미리보기 | 고전적 Telea 인페인팅, 품질 최하 |

## 원본 대비 변경점

업스트림과 비교해 이 포크는 다음을 개선했습니다.

- **코어 파이프라인 전면 리팩토링** (`backend/main.py`, 자막 검출,
  마스크/밴드 유틸리티, 5종 인페인팅 래퍼 전부) — 타입 힌트·문서화·린트 클린
  처리했고, 200줄짜리 밴드 추출 루틴을 테스트 가능한 헬퍼들로 분해했습니다.
- **실제 버그 6건 수정** — ProPainter 모드의 프레임 버그 2건(출력에서
  프레임이 조용히 누락되던 문제, 단일 프레임 배치에서 엉뚱한 프레임이
  인페인트되던 문제), `-o` 미지정 시 CLI 크래시, 고성능 추론(HPI) 플러그인이
  없을 때 PaddleOCR가 전체 실패하던 문제 등이 포함됩니다.
- **Apple Silicon 네이티브 지원** — torch MPS 가속과 함께, 번들된 x86-64
  전용 바이너리 때문에 깨지던 오디오 병합을 고치는 FFmpeg 탐색 체인
  (번들 → PATH → `imageio-ffmpeg`)을 구현했습니다.
- **E2E 검증 하네스 동봉** (`test/verify_removal.py`) — "자막만 정확히
  지웠는지"를 증명합니다. OCR 재검출이 0으로 떨어져야 하고, 인페인트 밴드
  바깥의 픽셀은 측정된 코덱 노이즈 기준 내에서 원본과 동일해야 통과합니다.

## 설치

Python 3.10 이상이 필요합니다 (3.12 권장).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt torch torchvision paddlepaddle
# NVIDIA GPU 사용 시: torch/paddlepaddle의 CUDA 빌드를 대신 설치하세요.
```

모델 가중치(약 600MB)는 `backend/models/`에 함께 포함되어 있습니다.

## 사용법

**웹 UI (권장):**

```bash
.venv/bin/pip install fastapi uvicorn python-multipart   # 최초 1회
.venv/bin/python -m webapp                                # http://127.0.0.1:8765
```

브라우저에서 모든 기능을 쓸 수 있습니다:

- 영상/이미지 드래그&드롭 업로드, 프레임 탐색
- 미리보기 위에 **드래그로 제거 영역 지정** (여러 개 가능)
- **OCR 자동탐지** (텍스트) / **AI 자동탐지** (Claude·OpenAI 비전 — 그래픽 로고까지, API 키 필요)
- **영역 분석 → 모드 자동 추천**: 고정 워터마크인지 움직이는 자막인지 판별해
  LaMa/STTN을 자동 선택
- 6가지 제거 방식 + 전체 고급 설정, 실시간 진행률, 원본/결과 비교, 다운로드

> 고정·불투명 워터마크는 시간축 모델(STTN)로는 지울 수 없습니다(모든 프레임에서
> 가려져 있어 참조할 배경이 없음). 이런 경우를 위해 **LaMa 영역 모드**가
> 프레임마다 공간 복원으로 지우며, 분석 버튼이 자동으로 골라줍니다.

**CLI:**

```bash
.venv/bin/python backend/main.py \
  -i input.mp4 -o output.mp4 \
  --inpaint-mode sttn-det \
  -c 340 480 0 852          # 자막 영역: ymin ymax xmin xmax (생략 가능)
```

`-c` 옵션을 생략하면 화면 전체를 대상으로 처리합니다. 자막 위치를 알고 있다면
영역을 지정하는 편이 더 빠르고 정확합니다.

**GUI:**

```bash
.venv/bin/python gui.py
```

**결과 검증:**

```bash
.venv/bin/python test/verify_removal.py input.mp4 output.mp4
```

## 인자 옵션

| 인자 | 설명 |
|------|------|
| `-i`, `--input` | 입력 영상/이미지 경로 (필수) |
| `-o`, `--output` | 출력 경로 (생략 시 `<이름>_no_sub.mp4`) |
| `-c`, `--subtitle-area-coords` | 자막 영역 `ymin ymax xmin xmax`, 여러 번 지정 가능 |
| `--inpaint-mode` | `sttn-auto`(기본)·`sttn-det`·`lama`·`propainter`·`opencv` |

## 라이선스

[Apache License 2.0](LICENSE) — 이 프로젝트가 파생된 원본과 동일합니다.

---

## English

AI-powered removal of hard-coded subtitles and text watermarks from videos and
images — fully local, no third-party APIs, native on Apple Silicon.

This project is a refactored fork of
[YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
(Apache 2.0). Changes vs upstream:

- Full refactor of the core pipeline (typed, documented, lint-clean)
- Six real bug fixes (ProPainter frame loss, CLI crash, OCR fallback, …)
- Native Apple Silicon support (torch MPS + arm64 FFmpeg resolution)
- An end-to-end verification harness proving only the subtitles were erased

See the [Usage](#사용법) section above; the commands are identical.
