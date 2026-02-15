from __future__ import annotations

import time
from collections import Counter
from datetime import datetime

from src.storage.event_logger import EventLogger


class StatsCalculator:
    def __init__(self, logger: EventLogger) -> None:
        self._logger = logger

    def summary(self, since: float | None = None) -> dict:
        events = self._logger.query(since=since, limit=10000)

        species_counts: Counter[str] = Counter()
        zone_counts: Counter[str] = Counter()
        hourly: Counter[int] = Counter()
        type_counts: Counter[str] = Counter()

        for e in events:
            species_counts[e["species"]] += 1
            zone_counts[e["zone_name"]] += 1
            hour = datetime.fromtimestamp(e["timestamp"]).hour
            hourly[hour] += 1
            type_counts[e["event_type"]] += 1

        return {
            "total_events": len(events),
            "by_species": dict(species_counts.most_common()),
            "by_zone": dict(zone_counts.most_common()),
            "by_hour": dict(sorted(hourly.items())),
            "by_type": dict(type_counts.most_common()),
        }

    def today(self) -> dict:
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.summary(since=midnight.timestamp())
