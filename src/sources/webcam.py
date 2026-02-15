from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.sources.base import VideoSource

logger = logging.getLogger(__name__)


class WebcamSource(VideoSource):
    def __init__(self, device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self._cap = cv2.VideoCapture(device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device {device}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or fps
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (w, h)
        logger.info("Webcam opened: %dx%d @ %.1f fps", w, h, self._fps)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size
