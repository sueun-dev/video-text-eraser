"""Vision-LLM region detection: find watermarks/logos/text via Claude or OpenAI.

PaddleOCR only finds *text*. Multimodal LLMs can also locate graphic logos,
channel badges, and semi-transparent overlays. This module sends one frame to
the chosen provider and returns normalized bounding boxes.

Authentication is by API key (Anthropic / OpenAI). Consumer-subscription
OAuth (Claude Pro, ChatGPT Plus) is not offered for third-party apps, so keys
are the supported path; they are kept client-side and only sent with the
request the user explicitly triggers.
"""

import base64
import json
from typing import List, Optional, TypedDict

import cv2
import numpy as np
import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
# Sonnet 4.6 is a deliberate default: locating a few overlay boxes in one frame
# is an easy vision task, so the cheaper tier suffices. Users can override the
# model per request (e.g. claude-opus-4-8) from the UI.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o"
REQUEST_TIMEOUT = 90

_PROMPT = """\
You are an overlay detector for a video watermark-removal tool.
Find ALL artificial overlays in this frame: hard-coded subtitles, captions,
watermarks, channel logos, badges, timestamps, and any superimposed text or
graphics. Do NOT include objects that are part of the actual scene.

Target filter: {target}

Reply with STRICT JSON only, no prose:
{{"boxes": [{{"label": "<short description>",
             "kind": "text" | "logo" | "watermark",
             "ymin": 0.0, "ymax": 0.0, "xmin": 0.0, "xmax": 0.0}}]}}
Coordinates are fractions of image height/width in [0, 1].
Return {{"boxes": []}} if nothing is found."""


class DetectedBox(TypedDict):
    label: str
    kind: str
    ymin: float
    ymax: float
    xmin: float
    xmax: float


def _encode_jpeg(frame_bgr: np.ndarray, max_side: int = 1280) -> str:
    """Downscale and JPEG-encode a frame as base64 for the API payload."""
    h, w = frame_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode()


def _parse_boxes(text: str) -> List[DetectedBox]:
    """Extract the boxes array from a (possibly chatty) model reply.

    Uses ``raw_decode`` from the first ``{`` so a trailing second brace block
    (e.g. ``{"boxes": []} note: {...}``) does not corrupt the match the way a
    greedy ``{.*}`` regex would.
    """
    payload = None
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "boxes" in candidate:
            payload = candidate
            break
        if payload is None and isinstance(candidate, dict):
            payload = candidate  # fall back to the first object if none has "boxes"
    if payload is None:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    boxes: List[DetectedBox] = []
    for box in payload.get("boxes", []):
        try:
            ymin = float(box["ymin"])
            ymax = float(box["ymax"])
            xmin = float(box["xmin"])
            xmax = float(box["xmax"])
        except (KeyError, TypeError, ValueError):
            continue
        if ymax <= ymin or xmax <= xmin:
            continue
        boxes.append(
            DetectedBox(
                label=str(box.get("label", "overlay"))[:80],
                kind=str(box.get("kind", "watermark")),
                ymin=max(0.0, min(1.0, ymin)),
                ymax=max(0.0, min(1.0, ymax)),
                xmin=max(0.0, min(1.0, xmin)),
                xmax=max(0.0, min(1.0, xmax)),
            )
        )
    return boxes


def detect_claude(
    frame_bgr: np.ndarray,
    api_key: str,
    model: Optional[str] = None,
    target: str = "both text and graphic overlays",
) -> List[DetectedBox]:
    """Detect overlay boxes with the Anthropic Messages API."""
    response = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model or DEFAULT_CLAUDE_MODEL,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": _encode_jpeg(frame_bgr),
                    }},
                    {"type": "text", "text": _PROMPT.format(target=target)},
                ],
            }],
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Claude API {response.status_code}: {response.text[:300]}")
    parts = response.json().get("content", [])
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    return _parse_boxes(text)


def detect_openai(
    frame_bgr: np.ndarray,
    api_key: str,
    model: Optional[str] = None,
    target: str = "both text and graphic overlays",
) -> List[DetectedBox]:
    """Detect overlay boxes with the OpenAI Chat Completions API."""
    response = requests.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_OPENAI_MODEL,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode_jpeg(frame_bgr)}",
                    }},
                    {"type": "text", "text": _PROMPT.format(target=target)},
                ],
            }],
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI API {response.status_code}: {response.text[:300]}")
    choices = response.json().get("choices", [])
    text = choices[0]["message"]["content"] if choices else ""
    return _parse_boxes(text)


def detect(provider: str, frame_bgr: np.ndarray, api_key: str,
           model: Optional[str] = None, target: str = "both") -> List[DetectedBox]:
    """Dispatch to the chosen provider. ``provider`` is 'claude' or 'openai'."""
    target_text = {
        "text": "text overlays only (subtitles, captions)",
        "logo": "graphic overlays only (logos, watermarks, badges)",
        "both": "both text and graphic overlays",
    }.get(target, "both text and graphic overlays")
    if provider == "claude":
        return detect_claude(frame_bgr, api_key, model, target_text)
    if provider == "openai":
        return detect_openai(frame_bgr, api_key, model, target_text)
    raise ValueError(f"unknown provider: {provider}")
