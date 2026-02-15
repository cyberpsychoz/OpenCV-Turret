from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from src.core.detector import COCO_ANIMAL_NAMES
from src.core.models import BBox, Track

if TYPE_CHECKING:
    from src.config import DetectorConfig

logger = logging.getLogger(__name__)


class AnimalTracker:
    def __init__(self, detector_cfg: DetectorConfig) -> None:
        from src.core.detector import AnimalDetector

        self._detector = AnimalDetector(detector_cfg)
        self._prev_centers: dict[int, tuple[int, int]] = {}

    def track(self, frame: np.ndarray) -> list[Track]:
        results = self._detector.track(frame, persist=True)

        tracks: list[Track] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            for box, track_id, cls_id, conf in zip(
                r.boxes.xyxy,
                r.boxes.id,
                r.boxes.cls,
                r.boxes.conf,
            ):
                x1, y1, x2, y2 = map(int, box.tolist())
                tid = int(track_id)
                cid = int(cls_id)
                bbox = BBox(x1, y1, x2, y2)
                center = bbox.center

                speed = 0.0
                prev = self._prev_centers.get(tid)
                if prev is not None:
                    dx = center[0] - prev[0]
                    dy = center[1] - prev[1]
                    speed = math.sqrt(dx * dx + dy * dy)

                self._prev_centers[tid] = center

                tracks.append(
                    Track(
                        track_id=tid,
                        bbox=bbox,
                        class_id=cid,
                        class_name=COCO_ANIMAL_NAMES.get(cid, f"class_{cid}"),
                        confidence=float(conf),
                        speed=speed,
                        prev_center=prev,
                    )
                )

        self._cleanup_stale(set(t.track_id for t in tracks))
        return tracks

    def _cleanup_stale(self, active_ids: set[int]) -> None:
        stale = [k for k in self._prev_centers if k not in active_ids]
        for k in stale:
            del self._prev_centers[k]
