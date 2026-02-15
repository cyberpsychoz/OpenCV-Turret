import numpy as np
import pytest

from src.core.models import BBox, Detection, Event, EventType, Track, Zone


@pytest.fixture
def sample_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    return BBox(100, 100, 200, 200)


@pytest.fixture
def sample_detection(sample_bbox):
    return Detection(bbox=sample_bbox, confidence=0.85, class_id=15, class_name="cat")


@pytest.fixture
def sample_track(sample_bbox):
    return Track(
        track_id=1,
        bbox=sample_bbox,
        class_id=15,
        class_name="cat",
        confidence=0.85,
        speed=5.0,
    )


@pytest.fixture
def sample_zone():
    return Zone(
        name="garden",
        points=[(50, 50), (300, 50), (300, 300), (50, 300)],
        priority=1,
        target_species=["cat", "dog"],
    )


@pytest.fixture
def sample_event():
    return Event(
        event_type=EventType.ZONE_ENTRY,
        zone_name="garden",
        track_id=1,
        species="cat",
        confidence=0.85,
    )
