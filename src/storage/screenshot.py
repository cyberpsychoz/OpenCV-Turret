from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.core.models import Event

logger = logging.getLogger(__name__)


class ScreenshotSaver:
    def __init__(self, directory: str = "data/events") -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, event: Event, frame: np.ndarray) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.timestamp))
        filename = f"{ts}_{event.species}_{event.zone_name}_{event.track_id}.jpg"
        filepath = self._dir / filename

        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.info("Screenshot saved: %s", filepath)
        return str(filepath)
