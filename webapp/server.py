"""FastAPI server exposing the full removal pipeline to the web frontend.

Run with:  .venv/bin/python -m webapp
Then open  http://127.0.0.1:8765
"""
# ruff: noqa: E402  -- sys.path/chdir must precede backend imports

import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config/ and model paths are resolved relative to the repo root

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import InpaintMode, SubtitleDetectMode, config
from backend.tools.common_tools import is_image_file, is_video_or_image, read_image
from backend.tools.region_remover import (
    analyze_region_motion,
    remove_regions_image,
    remove_regions_video,
)
from webapp import ai_detect

if TYPE_CHECKING:
    from backend.main import SubtitleRemover

DATA_DIR = ROOT / "webapp" / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Per-upload byte cap, enforced while streaming to disk.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def _sweep_stale_files(max_age_seconds: int = 24 * 3600) -> None:
    """Delete leftover uploads/results from previous runs so disk use stays bounded.

    Best-effort: media decoded from prior sessions should not linger forever on
    a long-lived local instance. Errors are ignored (files may be in use).
    """
    import time as _time

    now = _time.time()
    for folder in (UPLOAD_DIR, RESULT_DIR):
        for path in folder.glob("*"):
            try:
                if now - path.stat().st_mtime > max_age_seconds:
                    path.unlink(missing_ok=True)
            except OSError:
                pass


_sweep_stale_files()

MODE_MAP = {
    "sttn-auto": InpaintMode.STTN_AUTO,
    "sttn-det": InpaintMode.STTN_DET,
    "lama": InpaintMode.LAMA,
    "propainter": InpaintMode.PROPAINTER,
    "opencv": InpaintMode.OPENCV,
}

app = FastAPI(title="Video Text Eraser")

FILES: Dict[str, dict] = {}   # file_id -> {path, kind, meta}
JOBS: Dict[str, "Job"] = {}
_job_lock = threading.Lock()


class Job:
    def __init__(self, job_id: str, out_path: Path):
        self.id = job_id
        self.status = "queued"        # queued | running | done | error
        self.progress = 0
        self.log: List[str] = []
        self.out_path = out_path
        self.error: Optional[str] = None
        self.remover: Optional["SubtitleRemover"] = None  # set for non-region modes

    def current_progress(self) -> int:
        if self.remover is not None and self.status == "running":
            return max(self.progress, int(self.remover.progress_total))
        return self.progress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_lama_cache = {}


def _get_lama():
    """Construct LamaInpaint once and reuse it across jobs."""
    if "lama" not in _lama_cache:
        import torch
        from backend.inpaint.lama_inpaint import LamaInpaint
        from backend.tools.hardware_accelerator import HardwareAccelerator
        from backend.tools.model_config import ModelConfig

        accelerator = HardwareAccelerator.instance()
        device = (
            accelerator.device
            if accelerator.has_cuda() or accelerator.has_mps()
            else torch.device("cpu")
        )
        model_path = os.path.join(ModelConfig().LAMA_MODEL_DIR, "big-lama.pt")
        _lama_cache["lama"] = LamaInpaint(device, model_path)
    return _lama_cache["lama"]


def _file_or_404(file_id: str) -> dict:
    entry = FILES.get(file_id)
    if entry is None:
        raise HTTPException(404, "unknown file_id; upload first")
    return entry


def _read_frame(entry: dict, index: int) -> np.ndarray:
    if entry["kind"] == "image":
        frame = read_image(entry["path"])
        if frame is None:
            raise HTTPException(500, "failed to read image")
        return frame
    cap = cv2.VideoCapture(entry["path"])
    index = max(0, min(index, entry["meta"]["frames"] - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, f"failed to read frame {index}")
    return frame


def _image_media_type(suffix: str) -> str:
    """Map a file suffix to a standard image MIME type.

    Covers the JPEG family (jpg/jpe/jfif/jif/jfi -> jpeg), tif -> tiff, and
    dib -> bmp, which are all in the upload allowlist (common_tools) and would
    otherwise be served with a non-standard ``image/<ext>`` type.
    """
    ext = suffix.lower().lstrip(".") or "png"
    aliases = {
        "jpg": "jpeg", "jpe": "jpeg", "jfif": "jpeg", "jif": "jpeg", "jfi": "jpeg",
        "tif": "tiff", "dib": "bmp",
    }
    return f"image/{aliases.get(ext, ext)}"


def _to_abs_areas(areas_norm: List[List[float]], width: int, height: int):
    """Normalized [ymin,ymax,xmin,xmax] floats -> absolute pixel tuples."""
    result = []
    for ymin, ymax, xmin, xmax in areas_norm:
        result.append((
            int(round(ymin * height)), int(round(ymax * height)),
            int(round(xmin * width)), int(round(xmax * width)),
        ))
    return result


def _apply_settings(settings: dict) -> None:
    """Map frontend settings onto the global config items."""
    int_items = {
        "maskPadding": config.subtitleAreaDeviationPixel,
        "sttnNeighborStride": config.sttnNeighborStride,
        "sttnReferenceLength": config.sttnReferenceLength,
        "sttnMaxLoadNum": config.sttnMaxLoadNum,
        "propainterMaxLoadNum": config.propainterMaxLoadNum,
        "timelineBackward": config.subtitleTimelineBackwardFrameCount,
        "timelineForward": config.subtitleTimelineForwardFrameCount,
    }
    for key, item in int_items.items():
        if key in settings:
            item.value = int(settings[key])
    if "detectModel" in settings:
        config.subtitleDetectMode.value = (
            SubtitleDetectMode.PP_OCRv5_MOBILE
            if settings["detectModel"] == "mobile"
            else SubtitleDetectMode.PP_OCRv5_SERVER
        )
    if "hardwareAcceleration" in settings:
        config.hardwareAcceleration.value = bool(settings["hardwareAcceleration"])


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def _run_job(job: Job, entry: dict, mode: str, areas_abs, settings: dict) -> None:
    # status is set to "running" synchronously by process() under the lock.
    try:
        _apply_settings(settings)
        src = entry["path"]

        if mode == "lama-region":
            lama = _get_lama()
            padding = int(settings.get("maskPadding", 8))
            feather = int(settings.get("feather", 12))
            if entry["kind"] == "image":
                image = read_image(src)
                if image is None:
                    raise ValueError(f"cannot read image: {src}")
                result = remove_regions_image(image, areas_abs, lama, padding, feather)
                cv2.imencode(Path(src).suffix, result)[1].tofile(str(job.out_path))
                job.progress = 100
            else:
                job.log.append("LaMa region mode: erasing selected areas in every frame")
                remove_regions_video(
                    src, str(job.out_path), areas_abs, lama, padding, feather,
                    progress_cb=lambda p: setattr(job, "progress", p),
                )
        else:
            from backend.main import SubtitleRemover

            config.inpaintMode.value = MODE_MAP[mode]
            remover = SubtitleRemover(src)
            remover.sub_areas = list(areas_abs)
            remover.video_out_path = str(job.out_path)
            remover.append_output = lambda *args: job.log.append(" ".join(str(a) for a in args))
            job.remover = remover
            remover.run()
            job.progress = 100

        job.status = "done"
        job.log.append("completed")
    except Exception as e:
        traceback.print_exc()
        job.status = "error"
        job.error = str(e)
        job.log.append(f"ERROR: {e}")
    finally:
        # Self-heal the single-job guard: if the worker died without reaching a
        # terminal state (e.g. a BaseException), release it so the server does
        # not reject every future job with 409.
        if job.status == "running":
            job.status = "error"
            job.error = job.error or "job terminated unexpectedly"
            job.log.append("ERROR: job terminated unexpectedly")


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    file_id: str
    frame_index: int = 0


class AIDetectRequest(DetectRequest):
    provider: str            # "claude" | "openai"
    api_key: str
    model: Optional[str] = None
    target: str = "both"     # "text" | "logo" | "both"


class AnalyzeRequest(BaseModel):
    file_id: str
    areas: List[List[float]]


class ProcessRequest(BaseModel):
    file_id: str
    mode: str
    areas: List[List[float]] = []
    settings: dict = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if not is_video_or_image(f"x{suffix}"):
        raise HTTPException(400, f"unsupported file type: {suffix}")
    file_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{file_id}{suffix}"
    # Enforce a byte cap during streaming so a huge upload cannot fill the disk.
    # (The old image-only 100MB check ran only after the full write, too late.)
    total = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "file too large")
            out.write(chunk)

    if is_image_file(str(dest)):
        img = read_image(str(dest))
        if img is None:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "cannot decode image")
        meta = {"width": img.shape[1], "height": img.shape[0], "frames": 1, "fps": 0,
                "duration": 0}
        kind = "image"
    else:
        cap = cv2.VideoCapture(str(dest))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": frames,
            "fps": round(fps, 2),
            "duration": round(frames / fps, 1) if fps else 0,
        }
        cap.release()
        if meta["frames"] < 1 or meta["width"] < 1:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "cannot decode video")
        kind = "video"

    FILES[file_id] = {"path": str(dest), "kind": kind, "meta": meta,
                      "name": file.filename}
    return {"file_id": file_id, "kind": kind, **meta}


@app.get("/api/frame/{file_id}")
def frame(file_id: str, index: int = 0, width: int = 0):
    entry = _file_or_404(file_id)
    img = _read_frame(entry, index)
    if width and img.shape[1] > width:
        scale = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/api/detect/ocr")
def detect_ocr(req: DetectRequest):
    entry = _file_or_404(req.file_id)
    img = _read_frame(entry, req.frame_index)
    from backend.tools.subtitle_detect import SubtitleDetect

    detector = SubtitleDetect(entry["path"])
    h, w = img.shape[:2]
    boxes = []
    for xmin, xmax, ymin, ymax in detector.detect_subtitle(img):
        boxes.append({
            "label": "OCR text", "kind": "text",
            "ymin": ymin / h, "ymax": ymax / h,
            "xmin": xmin / w, "xmax": xmax / w,
        })
    return {"boxes": boxes}


@app.post("/api/detect/ai")
def detect_ai(req: AIDetectRequest):
    entry = _file_or_404(req.file_id)
    img = _read_frame(entry, req.frame_index)
    try:
        boxes = ai_detect.detect(req.provider, img, req.api_key, req.model, req.target)
    except Exception as e:
        raise HTTPException(502, str(e)) from e
    return {"boxes": boxes}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    entry = _file_or_404(req.file_id)
    if entry["kind"] == "image":
        return {"static": True, "recommended_mode": "lama-region",
                "reason": "이미지는 항상 LaMa로 처리됩니다"}
    if not req.areas:
        raise HTTPException(400, "areas required")
    meta = entry["meta"]
    areas_abs = _to_abs_areas(req.areas, meta["width"], meta["height"])
    result = analyze_region_motion(entry["path"], areas_abs)
    reason = (
        "영역이 모든 프레임에서 고정되어 있습니다 → 시간축 복원(STTN) 불가, LaMa 권장"
        if result["static"]
        else "영역 내용이 시간에 따라 변합니다 → 시간축 복원(STTN) 가능"
    )
    return {**result, "reason": reason}


@app.post("/api/process")
def process(req: ProcessRequest):
    entry = _file_or_404(req.file_id)
    if req.mode not in MODE_MAP and req.mode != "lama-region":
        raise HTTPException(400, f"unknown mode: {req.mode}")
    with _job_lock:
        # Count "queued" too: a newly created job only flips to "running" inside
        # the daemon thread, so checking only "running" would let a second
        # request slip through before the first thread starts (TOCTOU). The job
        # is marked "running" synchronously below, while still holding the lock.
        if any(j.status in ("queued", "running") for j in JOBS.values()):
            raise HTTPException(409, "다른 작업이 실행 중입니다. 완료 후 다시 시도하세요.")
        meta = entry["meta"]
        areas_abs = _to_abs_areas(req.areas, meta["width"], meta["height"])
        if not areas_abs:
            areas_abs = [(0, meta["height"], 0, meta["width"])]
        job_id = uuid.uuid4().hex[:12]
        suffix = Path(entry["path"]).suffix if entry["kind"] == "image" else ".mp4"
        job = Job(job_id, RESULT_DIR / f"{job_id}{suffix}")
        job.status = "running"
        JOBS[job_id] = job
        threading.Thread(
            target=_run_job, args=(job, entry, req.mode, areas_abs, req.settings),
            daemon=True,
        ).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {
        "status": job.status,
        "progress": job.current_progress(),
        "log": job.log[-8:],
        "error": job.error,
        "result_url": f"/api/result/{job_id}" if job.status == "done" else None,
    }


@app.get("/api/source/{file_id}")
def source(file_id: str):
    """Serve the uploaded original for side-by-side comparison."""
    entry = _file_or_404(file_id)
    suffix = Path(entry["path"]).suffix
    media = "video/mp4" if entry["kind"] == "video" else _image_media_type(suffix)
    return FileResponse(entry["path"], media_type=media)


@app.get("/api/result/{job_id}")
def result(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "result not ready")
    media = (
        "video/mp4" if job.out_path.suffix == ".mp4"
        else _image_media_type(job.out_path.suffix)
    )
    return FileResponse(str(job.out_path), media_type=media,
                        filename=f"erased{job.out_path.suffix}")


app.mount("/", StaticFiles(directory=str(ROOT / "webapp" / "static"), html=True))
