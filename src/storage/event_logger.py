from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src.core.models import Event

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL    NOT NULL,
    event_type  TEXT    NOT NULL,
    zone_name   TEXT    NOT NULL,
    track_id    INTEGER NOT NULL,
    species     TEXT    NOT NULL,
    confidence  REAL    DEFAULT 0.0,
    screenshot  TEXT,
    clip        TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_species   ON events(species);
CREATE INDEX IF NOT EXISTS idx_events_zone      ON events(zone_name);
"""


class EventLogger:
    def __init__(self, db_path: str = "data/db/events.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info("EventLogger: database at %s", path)

    def log(self, event: Event) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO events (timestamp, event_type, zone_name, track_id, species, confidence, screenshot, clip)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.timestamp,
                event.event_type.value,
                event.zone_name,
                event.track_id,
                event.species,
                event.confidence,
                event.screenshot_path,
                event.clip_path,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def query(
        self,
        species: str | None = None,
        zone: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        sql = "SELECT * FROM events WHERE 1=1"
        params: list = []

        if species:
            sql += " AND species = ?"
            params.append(species)
        if zone:
            sql += " AND zone_name = ?"
            params.append(zone)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()
