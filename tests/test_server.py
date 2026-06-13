"""Tests for webapp.server helpers and endpoint validation (no model jobs)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import InpaintMode, SubtitleDetectMode, config
from webapp import server


@pytest.fixture
def client():
    return TestClient(server.app)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def test_mode_map_covers_all_inpaint_modes():
    # Every non-lama-region mode maps to a real InpaintMode enum.
    for name, mode in server.MODE_MAP.items():
        assert isinstance(mode, InpaintMode)
    assert set(server.MODE_MAP) == {"sttn-auto", "sttn-det", "lama", "propainter", "opencv"}


def test_to_abs_areas_scales_normalized_coords():
    areas = server._to_abs_areas([[0.1, 0.5, 0.25, 0.75]], width=200, height=100)
    # (ymin,ymax,xmin,xmax) = (0.1*100, 0.5*100, 0.25*200, 0.75*200)
    assert areas == [(10, 50, 50, 150)]


def test_to_abs_areas_rounds():
    areas = server._to_abs_areas([[0.111, 0.889, 0.0, 1.0]], width=100, height=100)
    assert areas == [(11, 89, 0, 100)]


def test_apply_settings_maps_onto_config():
    snapshot = {
        "pad": config.subtitleAreaDeviationPixel.value,
        "stride": config.sttnNeighborStride.value,
        "detect": config.subtitleDetectMode.value,
    }
    try:
        server._apply_settings({
            "maskPadding": 17,
            "sttnNeighborStride": 9,
            "detectModel": "mobile",
        })
        assert config.subtitleAreaDeviationPixel.value == 17
        assert config.sttnNeighborStride.value == 9
        assert config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_MOBILE
    finally:
        config.subtitleAreaDeviationPixel.value = snapshot["pad"]
        config.sttnNeighborStride.value = snapshot["stride"]
        config.subtitleDetectMode.value = snapshot["detect"]


# --------------------------------------------------------------------------
# Endpoint validation (lightweight; no model inference)
# --------------------------------------------------------------------------

def test_unknown_job_returns_404(client):
    assert client.get("/api/job/doesnotexist").status_code == 404


def test_unknown_file_frame_returns_404(client):
    assert client.get("/api/frame/nope").status_code == 404


def test_upload_rejects_unsupported_type(client):
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    assert client.post("/api/upload", files=files).status_code == 400


def _upload_synth(client, synth_video):
    path = synth_video(frames=8, w=160, h=90, fps=10)
    with open(path, "rb") as fh:
        res = client.post("/api/upload", files={"file": ("v.mp4", fh.read(), "video/mp4")})
    assert res.status_code == 200, res.text
    return res.json()


def test_upload_returns_metadata(client, synth_video):
    meta = _upload_synth(client, synth_video)
    assert meta["kind"] == "video"
    assert meta["width"] == 160 and meta["height"] == 90
    assert meta["frames"] == 8


def test_frame_endpoint_returns_jpeg(client, synth_video):
    meta = _upload_synth(client, synth_video)
    res = client.get(f"/api/frame/{meta['file_id']}?index=2&width=80")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[1] == 80


def test_analyze_static_video(client, synth_video):
    path = synth_video(frames=12, w=160, h=90, watermark=(10, 40, 100, 150))
    with open(path, "rb") as fh:
        meta = client.post("/api/upload",
                           files={"file": ("v.mp4", fh.read(), "video/mp4")}).json()
    res = client.post("/api/analyze",
                      json={"file_id": meta["file_id"], "areas": [[0.11, 0.44, 0.625, 0.9375]]})
    assert res.status_code == 200
    assert res.json()["recommended_mode"] == "lama-region"


def test_process_unknown_file_returns_404(client):
    res = client.post("/api/process",
                      json={"file_id": "nope", "mode": "opencv", "areas": []})
    assert res.status_code == 404


def test_process_unknown_mode_returns_400(client, synth_video):
    meta = _upload_synth(client, synth_video)
    res = client.post("/api/process",
                      json={"file_id": meta["file_id"], "mode": "magic", "areas": []})
    assert res.status_code == 400


def test_source_endpoint_serves_original(client, synth_video):
    meta = _upload_synth(client, synth_video)
    res = client.get(f"/api/source/{meta['file_id']}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "video/mp4"


# --------------------------------------------------------------------------
# Bug-fix regressions
# --------------------------------------------------------------------------

def test_image_media_type_normalizes_jpeg():
    assert server._image_media_type(".png") == "image/png"
    assert server._image_media_type(".jpg") == "image/jpeg"
    assert server._image_media_type(".JPEG") == "image/jpeg"
    assert server._image_media_type(".webp") == "image/webp"
    assert server._image_media_type("") == "image/png"


def test_image_media_type_covers_full_allowlist():
    # Every JPEG-family / variant extension in the upload allowlist normalizes.
    assert server._image_media_type(".jfif") == "image/jpeg"
    assert server._image_media_type(".jif") == "image/jpeg"
    assert server._image_media_type(".jfi") == "image/jpeg"
    assert server._image_media_type(".tif") == "image/tiff"
    assert server._image_media_type(".dib") == "image/bmp"
    assert server._image_media_type(".bmp") == "image/bmp"


def test_source_jpg_served_as_jpeg(client, tmp_path):
    # Regression: a .jpg source must not be labelled image/png.
    img = np.full((30, 40, 3), 100, dtype=np.uint8)
    path = tmp_path / "pic.jpg"
    cv2.imwrite(str(path), img)
    with open(path, "rb") as fh:
        meta = client.post("/api/upload",
                           files={"file": ("pic.jpg", fh.read(), "image/jpeg")}).json()
    res = client.get(f"/api/source/{meta['file_id']}")
    assert res.headers["content-type"] == "image/jpeg"


def test_process_guard_rejects_running_job(client, synth_video):
    # Inject a fake running job; a new /process must be refused with 409.
    fake = server.Job("fakejob", server.RESULT_DIR / "fakejob.mp4")
    fake.status = "running"
    server.JOBS["fakejob"] = fake
    try:
        meta = _upload_synth(client, synth_video)
        res = client.post("/api/process",
                          json={"file_id": meta["file_id"], "mode": "opencv", "areas": []})
        assert res.status_code == 409
    finally:
        server.JOBS.pop("fakejob", None)


def test_process_guard_rejects_queued_job(client, synth_video):
    # TOCTOU regression: a job still in "queued" state must also block a new one.
    fake = server.Job("queuedjob", server.RESULT_DIR / "queuedjob.mp4")
    fake.status = "queued"
    server.JOBS["queuedjob"] = fake
    try:
        meta = _upload_synth(client, synth_video)
        res = client.post("/api/process",
                          json={"file_id": meta["file_id"], "mode": "opencv", "areas": []})
        assert res.status_code == 409
    finally:
        server.JOBS.pop("queuedjob", None)


def test_run_job_releases_guard_on_base_exception(synth_video):
    # A worker that dies via BaseException must not leave the job "running"
    # (which would block every future job with 409).
    job = server.Job("crashy", server.RESULT_DIR / "crashy.mp4")
    job.status = "running"
    entry = {"path": "x.mp4", "kind": "video", "meta": {}}

    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        def boom(_settings):
            raise KeyboardInterrupt("simulated hard stop")
        mp.setattr(server, "_apply_settings", boom)
        with _pytest.raises(KeyboardInterrupt):
            server._run_job(job, entry, "opencv", [(0, 1, 0, 1)], {})
    assert job.status == "error"          # guard released, not stuck "running"


def test_job_current_progress_uses_remover_only_while_running():
    job = server.Job("j", server.RESULT_DIR / "j.mp4")
    job.progress = 10

    class _Stub:
        progress_total = 55

    job.remover = _Stub()
    job.status = "running"
    assert job.current_progress() == 55          # max(10, 55)
    job.status = "done"
    assert job.current_progress() == 10          # ignores remover when not running
    job.remover = None
    job.status = "running"
    assert job.current_progress() == 10
