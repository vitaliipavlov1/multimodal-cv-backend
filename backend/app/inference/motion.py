"""
Farneback dense optical flow module.
Computes per-frame flow on CPU — synchronous, no Triton dependency.
"""

from typing import Optional

import cv2
import numpy as np


class FarnebackMotion:
    """
    Wraps cv2.calcOpticalFlowFarneback for frame-by-frame dense flow.
    Returns None on the first frame or after a resolution change.
    """

    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None

    def step(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute optical flow between the previous and current frame.

        Args:
            frame_bgr: Current frame in BGR uint8 format.

        Returns:
            Flow array [H, W, 2] or None if no previous frame is available.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        flow: Optional[np.ndarray] = None
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

        self._prev_gray = gray
        return flow

    def reset(self) -> None:
        """Discard the stored previous frame (e.g. on stream restart)."""
        self._prev_gray = None
