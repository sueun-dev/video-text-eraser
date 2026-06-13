"""Single-frame inpainting with LaMa (TorchScript big-lama checkpoint).

LaMa has no temporal awareness: each frame is restored independently. That
makes it the best choice for still images and animation, and the fallback for
runs too short to give STTN/ProPainter temporal context.
"""

import gc
from typing import List, Optional, Union, cast

import numpy as np
import torch
from PIL import Image

from backend.inpaint.utils.lama_util import get_image, pad_img_to_modulo, prepare_img_and_mask
from backend.tools.inpaint_tools import get_inpaint_area_by_mask

# LaMa's convolutional stack requires inputs padded to a multiple of 8.
_PAD_MODULO = 8
# Frames per forward pass when batching; keeps peak memory bounded.
_MINI_BATCH_SIZE = 4
# Height of the band handed to the model, as a fraction of frame width.
_BAND_HEIGHT_RATIO = 3 / 16


class LamaInpaint:
    """Single-frame spatial inpainting with the TorchScript big-lama model."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_path: str = "big-lama.pt",
    ) -> None:
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
    ) -> np.ndarray:
        """Inpaint a single image; returns an array cropped to the input size."""
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_height, orig_width = np.array(image).shape[:2]
        image_t, mask_t = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image_t, mask_t)
            result = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            result = np.clip(result * 255, 0, 255).astype("uint8")
            return result[:orig_height, :orig_width]

    def _inpaint_batch(
        self, images: List[np.ndarray], masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Inpaint many frames in fixed-size mini batches."""
        if len(images) == 1:
            return [self.inpaint(images[0], masks[0])]

        orig_height, orig_width = images[0].shape[:2]
        results: List[Optional[np.ndarray]] = [None] * len(images)
        for start in range(0, len(images), _MINI_BATCH_SIZE):
            end = min(start + _MINI_BATCH_SIZE, len(images))

            padded_imgs = np.stack(
                [pad_img_to_modulo(get_image(images[i]), _PAD_MODULO) for i in range(start, end)]
            )
            padded_masks = np.stack(
                [pad_img_to_modulo(get_image(masks[i]), _PAD_MODULO) for i in range(start, end)]
            )

            img_tensor = torch.from_numpy(padded_imgs).to(self.device)
            mask_tensor = (torch.from_numpy(padded_masks).to(self.device) > 0) * 1

            with torch.inference_mode():
                inpainted = self.model(img_tensor, mask_tensor)
                batch = inpainted.permute(0, 2, 3, 1).detach().cpu().numpy()
                batch = np.clip(batch * 255, 0, 255).astype("uint8")

            for i in range(end - start):
                results[start + i] = batch[i][:orig_height, :orig_width]

            del img_tensor, mask_tensor, padded_imgs, padded_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Every slot is filled by the loop above.
        return cast(List[np.ndarray], results)

    def __call__(
        self, input_frames: List[np.ndarray], input_mask: np.ndarray
    ) -> List[np.ndarray]:
        """Inpaint the masked subtitle bands of every frame.

        Only the horizontal band(s) covering the mask are processed; the
        restored band is written back into a copy of each original frame.
        """
        mask = input_mask[:, :, None]
        frame_height, frame_width = mask.shape[:2]
        band_height = int(frame_width * _BAND_HEIGHT_RATIO)
        bands = get_inpaint_area_by_mask(frame_width, frame_height, band_height, mask)

        frames = [frame.copy() for frame in input_frames]
        if not bands:
            return frames

        restored_bands = {}
        for k, (ymin, ymax, _, _) in enumerate(bands):
            cropped_frames = [frame[ymin:ymax, :, :] for frame in frames]
            cropped_masks = [mask[ymin:ymax, :, :]] * len(frames)
            restored_bands[k] = self._inpaint_batch(cropped_frames, cropped_masks)
            del cropped_frames, cropped_masks
            gc.collect()

        for j, frame in enumerate(frames):
            for k, (ymin, ymax, _, _) in enumerate(bands):
                frame[ymin:ymax, :, :] = restored_bands[k][j]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return frames
