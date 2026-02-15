from __future__ import annotations

import logging

import cv2
import numpy as np

from src.core.models import Zone
from src.core.zone_manager import save_zones

logger = logging.getLogger(__name__)

WINDOW = "Zone Editor — click to add points, ENTER to finish zone, ESC to save & exit"


class ZoneEditor:
    def __init__(self, frame: np.ndarray, existing_zones: list[Zone] | None = None) -> None:
        self._frame = frame.copy()
        self._base = frame.copy()
        self._zones: list[Zone] = list(existing_zones or [])
        self._current_points: list[tuple[int, int]] = []
        self._zone_count = len(self._zones)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_points.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and self._current_points:
            self._current_points.pop()
            self._redraw()

    def _redraw(self) -> None:
        self._frame = self._base.copy()

        for zone in self._zones:
            pts = np.array(zone.points, dtype=np.int32)
            cv2.polylines(self._frame, [pts], True, (0, 255, 0), 2)
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(self._frame, zone.name, tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self._current_points:
            for pt in self._current_points:
                cv2.circle(self._frame, pt, 4, (0, 0, 255), -1)
            if len(self._current_points) > 1:
                pts = np.array(self._current_points, dtype=np.int32)
                cv2.polylines(self._frame, [pts], False, (0, 0, 255), 2)

        cv2.imshow(WINDOW, self._frame)

    def _finish_zone(self) -> None:
        if len(self._current_points) < 3:
            logger.warning("Need at least 3 points for a zone")
            return

        self._zone_count += 1
        name = f"zone_{self._zone_count}"
        zone = Zone(name=name, points=list(self._current_points))
        self._zones.append(zone)
        self._current_points = []
        logger.info("Zone '%s' added with %d points", name, len(zone.points))
        self._redraw()

    def run(self) -> list[Zone]:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self._redraw()

        print("Zone Editor Controls:")
        print("  Left-click  — add point")
        print("  Right-click — undo last point")
        print("  ENTER       — finish current zone")
        print("  ESC         — save all zones and exit")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 13:  # Enter
                self._finish_zone()

        cv2.destroyWindow(WINDOW)
        return self._zones


def setup_zones(source, zones_path: str) -> list[Zone]:
    from src.core.zone_manager import load_zones

    ret, frame = source.read()
    if not ret or frame is None:
        raise RuntimeError("Cannot read frame from source for zone setup")

    existing = load_zones(zones_path)
    editor = ZoneEditor(frame, existing)
    zones = editor.run()

    if zones:
        save_zones(zones, zones_path)
        print(f"Saved {len(zones)} zones to {zones_path}")

    return zones
