"""Mask-guided video inpainting with STTN (detection-driven mode).

Frames are cropped to the subtitle band, downscaled, and restored by the
spatial-temporal transformer using both neighbouring and far-away reference
frames. Unlike the auto mode, the mask is fed to the network so pixels outside
the mask are preserved exactly.
"""

from typing import List, Optional, cast

import cv2
import numpy as np
import torch
from torchvision import transforms

from backend.config import config
from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import composite_band, get_inpaint_area_by_mask, guided_upsample

_compose = transforms.Compose([Stack(), ToTorchFormatTensor()])


def _to_tensors(frames) -> torch.Tensor:
    """Stack frames into a single normalized tensor (typed wrapper over Compose)."""
    return _compose(frames)

# Input resolution expected by the pretrained detection-mode checkpoint.
_MODEL_INPUT_WIDTH = 432
_MODEL_INPUT_HEIGHT = 240


class STTNDetInpaint:
    """Mask-guided STTN: keeps non-mask pixels, regenerates only the holes."""

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
        """Restore the masked subtitle bands across a run of frames."""
        mask = input_mask[:, :, None]
        frame_height, frame_width = mask.shape[:2]
        # Portrait videos need a taller band relative to their width.
        if frame_height > frame_width:
            band_height = int(frame_height * 5 / 9)
        else:
            band_height = int(frame_width * 5 / 18)
        bands = get_inpaint_area_by_mask(frame_width, frame_height, band_height, mask)

        frames = [frame.copy() for frame in input_frames]
        if not bands:
            return frames

        restored_bands = {}
        for k, (ymin, ymax, _, _) in enumerate(bands):
            scaled_frames, scaled_masks = [], []
            for frame in frames:
                frame_crop = frame[ymin:ymax, :, :]
                mask_crop = mask[ymin:ymax, :, :]
                scaled_frames.append(
                    cv2.resize(frame_crop, (self.model_input_width, self.model_input_height))
                )
                scaled_masks.append(
                    cv2.resize(mask_crop, (self.model_input_width, self.model_input_height))
                )
            restored_bands[k] = self.inpaint(scaled_frames, scaled_masks)

        for j, frame in enumerate(frames):
            for k, (ymin, ymax, _, _) in enumerate(bands):
                # Lanczos preserves more edge energy than bilinear on the ~2x
                # upscale from the model's 432-wide band to full width.
                restored = cv2.resize(
                    restored_bands[k][j], (frame_width, ymax - ymin),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                restored = cv2.cvtColor(restored.astype(np.uint8), cv2.COLOR_BGR2RGB)
                # Re-inject the original band's high-frequency detail lost in the
                # downscale->upscale round-trip (guide is still the original here).
                restored = guided_upsample(restored, frame[ymin:ymax, :, :])
                frame[ymin:ymax, :, :] = composite_band(
                    frame[ymin:ymax, :, :], restored, mask[ymin:ymax, :, :],
                    refine_strokes=True,
                )
        return frames

    def _reference_frame_ids(self, neighbor_ids: List[int], length: int) -> List[int]:
        """Sample far-away frames every ``ref_length`` as global context."""
        return [
            i for i in range(0, length, self.ref_length) if i not in neighbor_ids
        ]

    def inpaint(
        self, frames: List[np.ndarray], masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Fill the masked holes using spatial-temporal attention.

        A window slides over the run in ``neighbor_stride`` steps; each window
        attends to its neighbours plus sampled reference frames. Overlapping
        predictions are averaged 50/50 for temporal smoothness.
        """
        frame_length = len(frames)
        feats = (_to_tensors(frames).unsqueeze(0) * 2 - 1).to(self.device)
        binary_masks = [
            np.expand_dims((np.array(m) > 0.5).astype(np.uint8), 2) for m in masks
        ]
        masks_tensor = (_to_tensors(masks).unsqueeze(0) > 0.5).float().to(self.device)

        completed: List[Optional[np.ndarray]] = [None] * frame_length
        with torch.no_grad():
            masked_input = (feats * (1 - masks_tensor)).view(
                frame_length, 3, self.model_input_height, self.model_input_width
            )
            feats = self.model.encoder(masked_input)
            _, channels, feat_h, feat_w = feats.size()
            feats = feats.view(1, frame_length, channels, feat_h, feat_w)

            for f in range(0, frame_length, self.neighbor_stride):
                neighbor_ids = list(
                    range(max(0, f - self.neighbor_stride),
                          min(frame_length, f + self.neighbor_stride + 1))
                )
                ref_ids = self._reference_frame_ids(neighbor_ids, frame_length)
                window = neighbor_ids + ref_ids
                pred_feat = self.model.infer(
                    feats[0, window, :, :, :], masks_tensor[0, window, :, :, :]
                )
                pred_img = torch.tanh(self.model.decoder(pred_feat[: len(neighbor_ids)]))
                pred_img = ((pred_img + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255

                for i, idx in enumerate(neighbor_ids):
                    img = (
                        pred_img[i].astype(np.uint8) * binary_masks[idx]
                        + frames[idx] * (1 - binary_masks[idx])
                    )
                    prev = completed[idx]
                    if prev is None:
                        completed[idx] = img
                    else:
                        completed[idx] = (
                            prev.astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        )
        # Every frame is covered by at least one neighbour window -> no None left.
        return cast(List[np.ndarray], completed)
