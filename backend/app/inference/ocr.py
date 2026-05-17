"""
PaddleOCR inference module — async, CPU-based, throttled.

Runs text detection + recognition on a per-frame basis.
PaddleOCR is not thread-safe; serialised via asyncio.Lock + asyncio.to_thread.
Throttled to avoid saturating the CPU during real-time inference.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class PaddleOCRInferenceModule:
    """
    Async OCR module backed by PaddleOCR.

    Results format per frame:
        [{"box": [x1, y1, x2, y2], "text": "..."}, ...]
    """

    def __init__(
        self,
        lang:         str   = "latin",
        use_gpu:      bool  = False,
        min_interval: float = 1.0,
    ) -> None:
        self.lang         = lang
        self.use_gpu      = use_gpu
        self.min_interval = min_interval

        self._ocr:        Optional[PaddleOCR] = None
        self._setup_lock  = asyncio.Lock()
        self._infer_lock  = asyncio.Lock()
        self._ready       = False
        self._last_ts     = 0.0
        self._logged_once = False

    async def setup(self) -> None:
        async with self._setup_lock:
            if self._ready:
                return
            logger.info("Loading PaddleOCR models (lang=%s  gpu=%s)...", self.lang, self.use_gpu)
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=True,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            self._ready = True
            logger.info("PaddleOCR ready")

    async def infer_batch(
        self, frames: Sequence[np.ndarray]
    ) -> List[List[Dict]]:
        """
        Run OCR on a batch of frames.

        Throttled: returns empty lists if called faster than min_interval seconds.
        Returns a list of per-frame result lists.
        """
        if not frames:
            return []

        now = time.time()
        if now - self._last_ts < self.min_interval:
            return [[] for _ in frames]
        self._last_ts = now

        if not self._ready:
            await self.setup()

        raw_results = []
        async with self._infer_lock:
            for frame in frames:
                result = await asyncio.to_thread(self._ocr.ocr, frame, cls=True)
                raw_results.append(result)

                if not self._logged_once:
                    logger.debug("PaddleOCR raw sample: %r", result)
                    self._logged_once = True

        return [self._parse(raw, i) for i, raw in enumerate(raw_results)]

    @staticmethod
    def _parse(raw: Optional[List], frame_idx: int) -> List[Dict]:
        """Convert raw PaddleOCR output to [{box, text}] format."""
        if not raw:
            return []

        out: List[Dict] = []
        for line in raw:
            if (
                not isinstance(line, (list, tuple))
                or len(line) < 2
                or not isinstance(line[1], (list, tuple))
                or not line[1]
            ):
                continue

            text = line[1][0]
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            poly = np.array(line[0], dtype=np.float32)
            if poly.ndim != 2 or poly.shape[0] < 4:
                continue

            out.append({
                "box":  [
                    int(np.min(poly[:, 0])), int(np.min(poly[:, 1])),
                    int(np.max(poly[:, 0])), int(np.max(poly[:, 1])),
                ],
                "text": text,
            })

        if out:
            logger.debug("OCR frame %d: %d text region(s)", frame_idx, len(out))

        return out
