import time

import pytest

from src.core.models import Event, EventType
from src.storage.event_logger import EventLogger


class TestEventLogger:
    def test_log_and_query(self, tmp_path):
        db = tmp_path / "test.db"
        logger = EventLogger(str(db))

        event = Event(
            event_type=EventType.ZONE_ENTRY,
            zone_name="garden",
            track_id=1,
            species="cat",
            confidence=0.9,
        )

        row_id = logger.log(event)
        assert row_id == 1

        results = logger.query()
        assert len(results) == 1
        assert results[0]["species"] == "cat"
        assert results[0]["zone_name"] == "garden"
        assert results[0]["event_type"] == "zone_entry"

        logger.close()

    def test_query_filters(self, tmp_path):
        db = tmp_path / "test.db"
        logger = EventLogger(str(db))

        for species in ["cat", "cat", "dog", "bird"]:
            event = Event(
                event_type=EventType.ZONE_ENTRY,
                zone_name="garden",
                track_id=1,
                species=species,
            )
            logger.log(event)

        cats = logger.query(species="cat")
        assert len(cats) == 2

        dogs = logger.query(species="dog")
        assert len(dogs) == 1

        logger.close()

    def test_query_limit(self, tmp_path):
        db = tmp_path / "test.db"
        logger = EventLogger(str(db))

        for i in range(10):
            event = Event(
                event_type=EventType.ZONE_ENTRY,
                zone_name="garden",
                track_id=i,
                species="cat",
            )
            logger.log(event)

        results = logger.query(limit=3)
        assert len(results) == 3

        logger.close()

    def test_query_since(self, tmp_path):
        db = tmp_path / "test.db"
        logger = EventLogger(str(db))

        old = Event(
            event_type=EventType.ZONE_ENTRY,
            zone_name="garden",
            track_id=1,
            species="cat",
            timestamp=1000.0,
        )
        new = Event(
            event_type=EventType.ZONE_ENTRY,
            zone_name="garden",
            track_id=2,
            species="dog",
            timestamp=2000.0,
        )

        logger.log(old)
        logger.log(new)

        results = logger.query(since=1500.0)
        assert len(results) == 1
        assert results[0]["species"] == "dog"

        logger.close()
