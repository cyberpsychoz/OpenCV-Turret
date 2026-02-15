from __future__ import annotations

import collections
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ClipRecorder:
    def __init__(
        self,
        directory: str = "data/events",
        pre_seconds: int = 5,
        post_seconds: int = 10,
        fps: float = 30.0,
    ) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pre_frames = int(pre_seconds * fps)
        self._post_frames = int(post_seconds * fps)
        self._fps = fps
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=self._pre_frames)
        self._recording = False
        self._frames_left = 0
        self._writer: cv2.VideoWriter | None = None
        self._clip_path: str | None = None
        self._lock = threading.Lock()

    def feed(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._recording and self._writer is not None:
                self._writer.write(frame)
                self._frames_left -= 1
                if self._frames_left <= 0:
                    self._stop_recording()
            else:
                self._buffer.append(frame.copy())

    def trigger(self, label: str) -> str | None:
        with self._lock:
            if self._recording:
                self._frames_left = self._post_frames
                return self._clip_path

            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{label}.mp4"
            filepath = self._dir / filename
            self._clip_path = str(filepath)

            if not self._buffer:
                return None

            h, w = self._buffer[0].shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(filepath), fourcc, self._fps, (w, h))

            for buffered_frame in self._buffer:
                self._writer.write(buffered_frame)

            self._recording = True
            self._frames_left = self._post_frames
            logger.info("Clip recording started: %s", filepath)
            return self._clip_path

    def _stop_recording(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording = False
        logger.info("Clip recording finished: %s", self._clip_path)

    def close(self) -> None:
        with self._lock:
            if self._recording:
                self._stop_recording()
