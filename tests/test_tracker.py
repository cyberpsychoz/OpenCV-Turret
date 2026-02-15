from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config import DetectorConfig


class TestAnimalTracker:
    @patch("src.core.detector.YOLO")
    def test_track_returns_tracks(self, mock_yolo_cls, sample_frame):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.id = torch.tensor([1.0])
        mock_result.boxes.xyxy = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        mock_result.boxes.cls = torch.tensor([15.0])
        mock_result.boxes.conf = torch.tensor([0.9])
        mock_model.track.return_value = [mock_result]

        from src.core.tracker import AnimalTracker

        cfg = DetectorConfig()
        tracker = AnimalTracker(cfg)
        tracks = tracker.track(sample_frame)

        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert tracks[0].class_name == "cat"

    @patch("src.core.detector.YOLO")
    def test_track_computes_speed(self, mock_yolo_cls, sample_frame):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        def make_result(x1):
            mock_result = MagicMock()
            mock_result.boxes = MagicMock()
            mock_result.boxes.id = torch.tensor([1.0])
            mock_result.boxes.xyxy = torch.tensor([[float(x1), 100.0, float(x1 + 100), 200.0]])
            mock_result.boxes.cls = torch.tensor([15.0])
            mock_result.boxes.conf = torch.tensor([0.9])
            return mock_result

        mock_model.track.side_effect = [[make_result(100)], [make_result(150)]]

        from src.core.tracker import AnimalTracker

        cfg = DetectorConfig()
        tracker = AnimalTracker(cfg)

        tracks1 = tracker.track(sample_frame)
        assert tracks1[0].speed == 0.0

        tracks2 = tracker.track(sample_frame)
        assert tracks2[0].speed == 50.0

    @patch("src.core.detector.YOLO")
    def test_no_tracks_when_no_ids(self, mock_yolo_cls, sample_frame):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.id = None
        mock_model.track.return_value = [mock_result]

        from src.core.tracker import AnimalTracker

        cfg = DetectorConfig()
        tracker = AnimalTracker(cfg)
        tracks = tracker.track(sample_frame)

        assert tracks == []
