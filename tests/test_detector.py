from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import DetectorConfig
from src.core.models import BBox, Detection


class TestAnimalDetector:
    @patch("src.core.detector.YOLO")
    def test_detect_returns_detections(self, mock_yolo_cls, sample_frame):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock(tolist=MagicMock(return_value=[100, 100, 200, 200]))]
        mock_box.conf = [MagicMock(__float__=lambda s: 0.9)]
        mock_box.cls = [MagicMock(__int__=lambda s: 15)]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        from src.core.detector import AnimalDetector

        cfg = DetectorConfig()
        detector = AnimalDetector(cfg)
        detections = detector.detect(sample_frame)

        assert len(detections) == 1
        assert detections[0].class_name == "cat"
        assert detections[0].bbox.x1 == 100

    @patch("src.core.detector.YOLO")
    def test_detect_empty_frame(self, mock_yolo_cls, sample_frame):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]

        from src.core.detector import AnimalDetector

        cfg = DetectorConfig()
        detector = AnimalDetector(cfg)
        detections = detector.detect(sample_frame)

        assert detections == []

    @patch("src.core.detector.YOLO")
    def test_track_calls_model_track(self, mock_yolo_cls, sample_frame):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.track.return_value = []

        from src.core.detector import AnimalDetector

        cfg = DetectorConfig()
        detector = AnimalDetector(cfg)
        detector.track(sample_frame)

        mock_model.track.assert_called_once()
