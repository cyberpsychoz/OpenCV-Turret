from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.core.models import EventType, Event, Track, Zone

logger = logging.getLogger(__name__)


class ZoneManager:
    def __init__(self, zones: list[Zone], loiter_seconds: float = 30.0) -> None:
        self._zones = zones
        self._loiter_seconds = loiter_seconds
        self._contours: list[np.ndarray] = [
            np.array(z.points, dtype=np.int32) for z in zones
        ]
        # track_id -> {zone_name -> entry_timestamp}
        self._occupancy: dict[int, dict[str, float]] = {}
        logger.info("ZoneManager initialized with %d zones", len(zones))

    @property
    def zones(self) -> list[Zone]:
        return self._zones

    def check(self, tracks: list[Track]) -> list[Event]:
        events: list[Event] = []
        now = time.time()
        active_ids = {t.track_id for t in tracks}

        for track in tracks:
            cx, cy = track.center
            if track.track_id not in self._occupancy:
                self._occupancy[track.track_id] = {}

            track_zones = self._occupancy[track.track_id]

            for zone, contour in zip(self._zones, self._contours):
                if zone.target_species and track.class_name not in zone.target_species:
                    continue

                inside = cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0

                if inside and zone.name not in track_zones:
                    track_zones[zone.name] = now
                    events.append(
                        Event(
                            event_type=EventType.ZONE_ENTRY,
                            zone_name=zone.name,
                            track_id=track.track_id,
                            species=track.class_name,
                            timestamp=now,
                            confidence=track.confidence,
                        )
                    )
                elif inside and zone.name in track_zones:
                    duration = now - track_zones[zone.name]
                    if duration >= self._loiter_seconds:
                        events.append(
                            Event(
                                event_type=EventType.LOITERING,
                                zone_name=zone.name,
                                track_id=track.track_id,
                                species=track.class_name,
                                timestamp=now,
                                confidence=track.confidence,
                            )
                        )
                        track_zones[zone.name] = now
                elif not inside and zone.name in track_zones:
                    del track_zones[zone.name]
                    events.append(
                        Event(
                            event_type=EventType.ZONE_EXIT,
                            zone_name=zone.name,
                            track_id=track.track_id,
                            species=track.class_name,
                            timestamp=now,
                            confidence=track.confidence,
                        )
                    )

        gone = [tid for tid in self._occupancy if tid not in active_ids]
        for tid in gone:
            for zone_name in self._occupancy[tid]:
                events.append(
                    Event(
                        event_type=EventType.ZONE_EXIT,
                        zone_name=zone_name,
                        track_id=tid,
                        species="unknown",
                        timestamp=now,
                    )
                )
            del self._occupancy[tid]

        return events


def load_zones(path: str | Path) -> list[Zone]:
    path = Path(path)
    if not path.exists():
        logger.warning("Zones file not found: %s — running without zones", path)
        return []

    with open(path) as f:
        data = json.load(f)

    zones: list[Zone] = []
    for z in data:
        zones.append(
            Zone(
                name=z["name"],
                points=[(p[0], p[1]) for p in z["points"]],
                priority=z.get("priority", 0),
                target_species=z.get("target_species", []),
            )
        )

    logger.info("Loaded %d zones from %s", len(zones), path)
    return zones


def save_zones(zones: list[Zone], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "name": z.name,
            "points": z.points,
            "priority": z.priority,
            "target_species": z.target_species,
        }
        for z in zones
    ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved %d zones to %s", len(zones), path)
