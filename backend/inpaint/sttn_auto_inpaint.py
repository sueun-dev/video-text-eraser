"""Whole-video inpainting with STTN, no per-frame subtitle detection (auto mode).

The user-selected area is treated as the hole for every frame; the
transformer regenerates the band from temporal context alone. Skipping OCR
makes this the fastest mode, at the risk of touching frames that never had
text inside the area.
"""

import gc
import os
from typing import Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from backend.config import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.inpaint_tools import (
    composite_band,
    get_inpaint_area_by_mask,
    is_frame_number_in_ab_sections,
)
from backend.tools.video_io import FramePrefetcher, make_video_writer

_compose = transforms.Compose([Stack(), ToTorchFormatTensor()])


def _to_tensors(frames) -> torch.Tensor:
    """Stack frames into a single normalized tensor (typed wrapper over Compose)."""
    return _compose(frames)

# Input resolution expected by the pretrained auto-mode checkpoint.
_MODEL_INPUT_WIDTH = 640
_MODEL_INPUT_HEIGHT = 120
# Height of the band handed to the model, as a fraction of frame width.
_BAND_HEIGHT_RATIO = 3 / 16
# Rough per-frame device memory cost (bytes) used to bound chunk size.
_BYTES_PER_PIXEL_BUDGET = 12


class STTNInpaint:
    """Core STTN forward pass shared by the auto-mode pipeline."""

    def __init__(self, device: torch.device, model_path: str) -> None:
        self.device = device
        self.model = InpaintGenerator().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu")["netG"])
        self.model.eval()
        self.model_input_width = _MODEL_INPUT_WIDTH
        self.model_input_height = _MODEL_INPUT_HEIGHT
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value

    def __call__(
        self, input_frames: List[np.ndarray], input_mask: np.ndarray
    ) -> List[np.ndarray]:
        """Regenerate the masked subtitle bands across a run of frames."""
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        frame_height, frame_width = mask.shape[:2]
        band_height = int(frame_width * _BAND_HEIGHT_RATIO)
        bands = get_inpaint_area_by_mask(frame_width, frame_height, band_height, mask)

        frames = [frame.copy() for frame in input_frames]
        if not bands:
            return frames

        restored_bands: Dict[int, List[np.ndarray]] = {}
        for k, (ymin, ymax, _, _) in enumerate(bands):
            scaled = [
                cv2.resize(
                    frame[ymin:ymax, :, :],
                    (self.model_input_width, self.model_input_height),
                )
                for frame in frames
            ]
            restored_bands[k] = self.inpaint(scaled)

        for j, frame in enumerate(frames):
            for k, (ymin, ymax, _, _) in enumerate(bands):
                restored = cv2.resize(restored_bands[k][j], (frame_width, ymax - ymin))
                restored = cv2.cvtColor(restored.astype(np.uint8), cv2.COLOR_BGR2RGB)
                frame[ymin:ymax, :, :] = composite_band(
                    frame[ymin:ymax, :, :], restored, mask[ymin:ymax, :]
                )
        return frames

    @staticmethod
    def read_mask(path: str) -> np.ndarray:
        img = cv2.imread(path, 0)
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        return img[:, :, None]

    def _reference_frame_ids(self, neighbor_ids: List[int], length: int) -> List[int]:
        """Sample far-away frames every ``ref_length`` as global context."""
        return [
            i for i in range(0, length, self.ref_length) if i not in neighbor_ids
        ]

    def inpaint(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Regenerate every frame's band using spatial-temporal attention.

        A window slides over the run in ``neighbor_stride`` steps; each window
        attends to its neighbours plus sampled reference frames. Overlapping
        predictions are averaged 50/50 for temporal smoothness.
        """
        frame_length = len(frames)
        feats = (_to_tensors(frames).unsqueeze(0) * 2 - 1).to(self.device)

        completed: List[Optional[np.ndarray]] = [None] * frame_length
        with torch.no_grad():
            feats = self.model.encoder(
                feats.view(frame_length, 3, self.model_input_height, self.model_input_width)
            )
            _, channels, feat_h, feat_w = feats.size()
            feats = feats.view(1, frame_length, channels, feat_h, feat_w)

            for f in range(0, frame_length, self.neighbor_stride):
                neighbor_ids = list(
                    range(max(0, f - self.neighbor_stride),
                          min(frame_length, f + self.neighbor_stride + 1))
                )
                ref_ids = self._reference_frame_ids(neighbor_ids, frame_length)
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                pred_img = torch.tanh(self.model.decoder(pred_feat[: len(neighbor_ids)]))
                pred_img = ((pred_img + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255

                for i, idx in enumerate(neighbor_ids):
                    img = pred_img[i].astype(np.uint8)
                    prev = completed[idx]
                    if prev is None:
                        completed[idx] = img
                    else:
                        completed[idx] = (
                            prev.astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        )
        # Every frame is covered by at least one neighbour window -> no None left.
        return cast(List[np.ndarray], completed)


class STTNAutoInpaint:
    """Streams a whole video through STTN in VRAM-bounded chunks."""

    def __init__(
        self,
        device: torch.device,
        model_path: str,
        video_path: str,
        mask_path: Optional[str] = None,
        clip_gap: Optional[int] = None,
    ) -> None:
        self.sttn_inpaint = STTNInpaint(device, model_path)
        self.video_path = video_path
        self.mask_path = mask_path
        stem = os.path.basename(video_path).rsplit(".", 1)[0]
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(video_path)), f"{stem}_no_sub.mp4"
        )
        self.clip_gap = clip_gap if clip_gap is not None else config.getSttnMaxLoadNum()

    def _read_video_info(self) -> Tuple[cv2.VideoCapture, dict]:
        reader = cv2.VideoCapture(self.video_path)
        info = {
            "W_ori": int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
            "H_ori": int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
            "fps": reader.get(cv2.CAP_PROP_FPS),
            "len": int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5),
        }
        return reader, info

    def _effective_clip_gap(self, width: int, height: int) -> int:
        """Shrink the chunk size when available VRAM cannot hold it."""
        vram_mb = HardwareAccelerator.instance().get_available_vram_mb()
        if vram_mb <= 0 or width <= 0 or height <= 0:
            # No GPU budget to enforce, or a source reporting 0x0 dimensions
            # (corrupt metadata) — the divide below would be a ZeroDivisionError.
            return self.clip_gap
        bytes_per_frame = width * height * _BYTES_PER_PIXEL_BUDGET
        max_frames = max(int(vram_mb * 1024 * 1024 / bytes_per_frame), 10)
        effective = min(self.clip_gap, max_frames)
        if effective < self.clip_gap:
            tqdm.write(
                f"GPU VRAM: {vram_mb:.0f}MB, adjusting clip_gap: {self.clip_gap} -> {effective}"
            )
        return effective

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None) -> None:
        reader, info = self._read_video_info()
        prefetcher = FramePrefetcher(reader)

        if input_sub_remover is not None:
            ab_sections = input_sub_remover.ab_sections
            writer = input_sub_remover.video_writer
        else:
            ab_sections = None
            writer = make_video_writer(
                self.video_out_path, info["fps"], (info["W_ori"], info["H_ori"])
            )

        try:
            if input_mask is None:
                if self.mask_path is None:
                    raise ValueError("input_mask or mask_path is required")
                mask = self.sttn_inpaint.read_mask(self.mask_path)
            else:
                _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
                mask = mask[:, :, None]

            band_height = int(info["W_ori"] * _BAND_HEIGHT_RATIO)
            bands = get_inpaint_area_by_mask(info["W_ori"], info["H_ori"], band_height, mask)
            clip_gap = self._effective_clip_gap(info["W_ori"], info["H_ori"])

            # Read fixed-size chunks until the source is genuinely exhausted
            # rather than a fixed chunk_count derived from info["len"]:
            # CAP_PROP_FRAME_COUNT is an estimate that can both under-count
            # (which would truncate the output and desync the audio) and
            # over-count (which the EOF break handles). info["len"] is only a
            # progress hint here.
            chunk = 0
            while True:
                start_f = chunk * clip_gap
                end_f = start_f + clip_gap
                tqdm.write(f"Processing: {start_f + 1} - {end_f} / ~{info['len']}")
                reached_eof = self._process_chunk(
                    prefetcher, writer, mask, bands, start_f, end_f,
                    info, ab_sections, input_sub_remover, tbar,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if reached_eof:
                    break
                chunk += 1
        finally:
            prefetcher.release()
            if writer is not None:
                writer.release()

    def _process_chunk(
        self, prefetcher, writer, mask, bands, start_f, end_f,
        info, ab_sections, sub_remover, tbar,
    ) -> bool:
        """Read one chunk of frames, restore its bands, and write results.

        Returns True if the source was exhausted mid-chunk (the caller must then
        stop, as further reads would block on the prefetcher).
        """
        frames_hr: List[np.ndarray] = []
        scaled: Dict[int, List[np.ndarray]] = {k: [] for k in range(len(bands))}
        # Maps chunk-relative frame index -> index into the inpainted band list
        # (only frames inside the A/B sections are inpainted).
        inpainted_index: Dict[int, int] = {}
        reached_eof = False

        for j in range(start_f, end_f):
            success, frame = prefetcher.read()
            if not success:
                print(f"Warning: Failed to read frame {j}.")
                reached_eof = True
                break
            frames_hr.append(frame)
            if is_frame_number_in_ab_sections(j, ab_sections):
                inpainted_index[j - start_f] = len(inpainted_index)
                for k, (ymin, ymax, _, _) in enumerate(bands):
                    crop = frame[ymin:ymax, :, :]
                    scaled[k].append(
                        cv2.resize(
                            crop,
                            (self.sttn_inpaint.model_input_width,
                             self.sttn_inpaint.model_input_height),
                        )
                    )

        if not frames_hr:
            print(f"Warning: No valid frames found in range {start_f + 1}-{end_f}.")
            return reached_eof

        restored_bands = {
            k: self.sttn_inpaint.inpaint(scaled[k]) if scaled[k] else []
            for k in range(len(bands))
        }

        for j, frame in enumerate(frames_hr):
            preview_original = (
                frame.copy()
                if sub_remover is not None and sub_remover.gui_mode
                else None
            )
            if bands and j in inpainted_index:
                comp_idx = inpainted_index[j]
                for k, (ymin, ymax, _, _) in enumerate(bands):
                    if comp_idx >= len(restored_bands[k]):
                        continue
                    restored = cv2.resize(
                        restored_bands[k][comp_idx], (info["W_ori"], ymax - ymin)
                    )
                    restored = cv2.cvtColor(restored.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    frame[ymin:ymax, :, :] = composite_band(
                        frame[ymin:ymax, :, :], restored, mask[ymin:ymax, :]
                    )
            writer.write(frame)
            if sub_remover is not None:
                if tbar is not None:
                    sub_remover.update_progress(tbar, increment=1)
                if preview_original is not None:
                    sub_remover.update_preview_with_comp(preview_original, frame)
        return reached_eof
