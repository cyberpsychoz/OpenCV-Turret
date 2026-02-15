import json
import time
from pathlib import Path

import pytest

from src.core.models import BBox, EventType, Track, Zone
from src.core.zone_manager import ZoneManager, load_zones, save_zones


class TestZoneManager:
    def test_zone_entry_detected(self, sample_zone):
        zm = ZoneManager([sample_zone])

        track = Track(
            track_id=1,
            bbox=BBox(140, 140, 160, 160),
            class_id=15,
            class_name="cat",
            confidence=0.9,
        )

        events = zm.check([track])
        assert len(events) == 1
        assert events[0].event_type == EventType.ZONE_ENTRY
        assert events[0].species == "cat"
        assert events[0].zone_name == "garden"

    def test_no_duplicate_entry(self, sample_zone):
        zm = ZoneManager([sample_zone])
        track = Track(
            track_id=1,
            bbox=BBox(140, 140, 160, 160),
            class_id=15,
            class_name="cat",
            confidence=0.9,
        )

        events1 = zm.check([track])
        assert len(events1) == 1

        events2 = zm.check([track])
        assert len(events2) == 0

    def test_zone_exit_detected(self, sample_zone):
        zm = ZoneManager([sample_zone])

        inside = Track(track_id=1, bbox=BBox(140, 140, 160, 160), class_id=15, class_name="cat", confidence=0.9)
        outside = Track(track_id=1, bbox=BBox(400, 400, 420, 420), class_id=15, class_name="cat", confidence=0.9)

        zm.check([inside])
        events = zm.check([outside])

        assert len(events) == 1
        assert events[0].event_type == EventType.ZONE_EXIT

    def test_species_filter(self, sample_zone):
        zm = ZoneManager([sample_zone])

        track = Track(
            track_id=1,
            bbox=BBox(140, 140, 160, 160),
            class_id=21,
            class_name="bear",
            confidence=0.9,
        )

        events = zm.check([track])
        assert len(events) == 0

    def test_track_disappears_generates_exit(self, sample_zone):
        zm = ZoneManager([sample_zone])

        track = Track(track_id=1, bbox=BBox(140, 140, 160, 160), class_id=15, class_name="cat", confidence=0.9)

        zm.check([track])
        events = zm.check([])

        assert len(events) == 1
        assert events[0].event_type == EventType.ZONE_EXIT


class TestZonePersistence:
    def test_save_and_load(self, tmp_path):
        zones = [
            Zone(name="test", points=[(0, 0), (100, 0), (100, 100)], priority=1, target_species=["cat"]),
        ]

        path = tmp_path / "zones.json"
        save_zones(zones, path)

        loaded = load_zones(path)
        assert len(loaded) == 1
        assert loaded[0].name == "test"
        assert loaded[0].points == [(0, 0), (100, 0), (100, 100)]
        assert loaded[0].target_species == ["cat"]

    def test_load_missing_file(self, tmp_path):
        zones = load_zones(tmp_path / "nonexistent.json")
        assert zones == []
