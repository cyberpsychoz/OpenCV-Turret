from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from ultralytics import YOLO

from src.core.models import BBox, Detection

if TYPE_CHECKING:
    from src.config import DetectorConfig

logger = logging.getLogger(__name__)

COCO_ANIMAL_NAMES: dict[int, str] = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


class AnimalDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self._model = YOLO(cfg.model)
        self._model.to(cfg.device)
        self._imgsz = cfg.imgsz
        self._conf = cfg.confidence
        self._classes = cfg.animal_classes
        self._device = cfg.device
        logger.info(
            "AnimalDetector loaded model=%s device=%s classes=%s",
            cfg.model, cfg.device, self._classes,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(
            frame,
            imgsz=self._imgsz,
            conf=self._conf,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = COCO_ANIMAL_NAMES.get(cls_id, f"class_{cls_id}")
                detections.append(
                    Detection(
                        bbox=BBox(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        return detections

    def track(self, frame: np.ndarray, persist: bool = True) -> list:
        return self._model.track(
            frame,
            imgsz=self._imgsz,
            conf=self._conf,
            classes=self._classes,
            device=self._device,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False,
        )
