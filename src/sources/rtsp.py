from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2
import numpy as np

from src.sources.base import VideoSource

logger = logging.getLogger(__name__)


class RTSPSource(VideoSource):
    def __init__(self, url: str, fps: float = 30.0) -> None:
        self._url = url
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {url}")

        self._fps_val = self._cap.get(cv2.CAP_PROP_FPS) or fps
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (w, h)

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info("RTSP source started: %s (%dx%d)", url, w, h)

    def _reader(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("RTSP read failed, reconnecting...")
                self._cap.release()
                self._cap = cv2.VideoCapture(self._url)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size
