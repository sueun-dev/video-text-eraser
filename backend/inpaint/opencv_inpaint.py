"""Classical (non-learning) inpainting fallback based on cv2.inpaint."""

from typing import List

import cv2
import numpy as np


class OpenCVInpaint:
    """Telea inpainting; fast and dependency-free but lowest quality."""

    INPAINT_RADIUS = 3

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(frame, mask, self.INPAINT_RADIUS, cv2.INTER_LINEAR)

    def __call__(self, frames: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        return [self.inpaint(frame, mask) for frame in frames]
