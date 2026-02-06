"""
Async OCR module (NO TRITON, NO MMOCR).

- Text detector + recognizer: PaddleOCR
- Languages: English + Spanish
- Event-only (NOT realtime)
- Fully compatible with:
  - final_inference_pipeline_multimodal.py
  - FastAPI WS
  - frontend drawOCR
"""

from __future__ import annotations
import asyncio
import logging
from typing import List, Dict, Sequence

import numpy as np
import cv2
import time

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class PaddleOCRInferenceModule:
    def __init__(
        self,
        lang: str = "latin",         
        use_gpu: bool = False,
    ):

        self.lang = lang
        self.use_gpu = use_gpu

        self.ocr: PaddleOCR | None = None
        self._setup_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()
        self._ready = False

        self._last_ocr_ts = 0.0
        self.min_interval = 1.0  # секунды

    # ---------------- setup ----------------
    async def setup(self):
        async with self._setup_lock:
            if self._ready:
                return

            logger.info("Loading PaddleOCR models...")

            # PaddleOCR сам скачает модели в кеш (см. ниже)
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=True,
                use_gpu=self.use_gpu,
                show_log=False,
            )

            self._ready = True
            logger.info("PaddleOCR loaded successfully")

    # ---------------- inference ----------------
    async def infer_batch(
        self,
        frames: Sequence[np.ndarray],
    ) -> List[List[Dict]]:
        """
        Returns:
        [
          [
            {"box": [x1,y1,x2,y2], "text": "..."},
            ...
          ],
          ...
        ]
        """

        now = time.time()
        if now - self._last_ocr_ts < self.min_interval:
            logger.debug(
                "OCR skipped (throttled): dt=%.2fs < min_interval=%.2fs",
                now - self._last_ocr_ts,
                self.min_interval
            )

            return [[] for _ in frames]

        self._last_ocr_ts = now


        if not frames:
            return []

        if not self._ready:
            await self.setup()

        results: List[List[Dict]] = [[] for _ in frames]

        # PaddleOCR НЕ thread-safe → сериализуем

        ocr_results = []

        async with self._infer_lock:

            for frame in frames:
                result = await asyncio.to_thread(
                    self.ocr.ocr,
                    frame,
                    cls=True
                )
                ocr_results.append(result)

                if not hasattr(self, "_raw_logged"):
                    logger.info(
                        "RAW PaddleOCR result example: %s",
                        repr(result)
                    )
                    self._raw_logged = True

        for img_idx, det in enumerate(ocr_results):
            if not det:
                logger.debug("OCR result empty: frame=%d (no detections)", img_idx)
                continue

            for line in det:
                if (
                    not isinstance(line, (list, tuple)) or
                    len(line) < 2 or
                    not isinstance(line[1], (list, tuple)) or
                    len(line[1]) == 0
                ):
                    continue

                text = line[1][0]
                if not isinstance(text, str):
                    continue

                text = text.strip()
                if not text:
                    continue

                poly = line[0]
                poly = np.array(poly, dtype=np.float32)
                if poly.ndim != 2 or poly.shape[0] < 4:
                    continue

                x1 = int(np.min(poly[:, 0]))
                y1 = int(np.min(poly[:, 1]))
                x2 = int(np.max(poly[:, 0]))
                y2 = int(np.max(poly[:, 1]))

                results[img_idx].append({
                    "box": [x1, y1, x2, y2],
                    "text": text,
                })

            logger.debug(
                "OCR normalized result (frame %d): %s",
                img_idx,
                results[img_idx]
            )

            
        total_boxes = sum(len(r) for r in results)

        logger.info(
            "OCR batch summary: frames=%d, total_boxes=%d",
            len(results),
            total_boxes
        )

        return results

