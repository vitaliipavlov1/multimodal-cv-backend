import cv2
import numpy as np
from typing import Optional


class FarnebackMotion:
    def __init__(self, stride: int = 2):
        self.prev_gray: Optional[np.ndarray] = None
        self.last_flow: Optional[np.ndarray] = None
        self.stride = stride
        self.frame_idx = 0


    def step(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        flow = None

        if (
                self.prev_gray is not None
                and self.prev_gray.shape == gray.shape
        ):
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray,
                gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            self.last_flow = flow
        else:
            # 🔁 новый источник / новое разрешение
            self.last_flow = None
            flow = None

        self.prev_gray = gray
        self.frame_idx += 1
        return flow
