"""Classical (non-learning) inpainting fallback based on cv2.inpaint."""

from typing import List

import cv2
import numpy as np

from backend.tools.inpaint_tools import composite_run_pde


class OpenCVInpaint:
    """Glyph-precise Navier-Stokes inpainting: fast, dependency-free, no model.

    Fills only the detected glyph strokes by interpolating the real surrounding
    pixels (Navier-Stokes), temporally fused across the run for stability — the
    same real-background fill the model modes blend in, here used on its own.
    Far better than the old whole-box Telea fill while staying purely classical.
    """

    INPAINT_RADIUS = 3

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Single-frame Telea fill of the given mask (used for previews/tests)."""
        return cv2.inpaint(frame, mask, self.INPAINT_RADIUS, cv2.INPAINT_TELEA)

    def __call__(self, frames: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        box = mask[:, :, 0] if mask.ndim == 3 else mask
        # ns_weight=1.0: pure real-pixel PDE fill (there is no model to blend).
        return composite_run_pde(frames, frames, box, ns_weight=1.0)
