from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


@dataclass(frozen=True, slots=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.width, self.height


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    speed: float = 0.0
    prev_center: Optional[tuple[int, int]] = None

    @property
    def center(self) -> tuple[int, int]:
        return self.bbox.center


@dataclass(frozen=True, slots=True)
class Zone:
    name: str
    points: list[tuple[int, int]]
    priority: int = 0
    target_species: list[str] = field(default_factory=list)


class EventType(Enum):
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    LOITERING = "loitering"


@dataclass(frozen=True, slots=True)
class Event:
    event_type: EventType
    zone_name: str
    track_id: int
    species: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0
    screenshot_path: Optional[str] = None
    clip_path: Optional[str] = None
