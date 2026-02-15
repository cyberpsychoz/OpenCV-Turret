from __future__ import annotations

import cv2
import numpy as np

from src.core.models import Track, Zone

COLORS = [
    (0, 255, 0),    # green
    (255, 165, 0),   # orange
    (0, 0, 255),     # red
    (255, 255, 0),   # cyan
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (128, 0, 255),   # purple
    (255, 128, 0),   # light blue
]


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    return COLORS[track_id % len(COLORS)]


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    for track in tracks:
        color = _color_for_id(track.track_id)
        b = track.bbox
        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 2)

        label = f"#{track.track_id} {track.class_name} {track.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (b.x1, b.y1 - th - 6), (b.x1 + tw, b.y1), color, -1)
        cv2.putText(
            frame, label, (b.x1, b.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

        if track.speed > 2.0:
            speed_label = f"{track.speed:.0f} px/f"
            cv2.putText(
                frame, speed_label, (b.x1, b.y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    return frame


def draw_zones(frame: np.ndarray, zones: list[Zone]) -> np.ndarray:
    overlay = frame.copy()
    for i, zone in enumerate(zones):
        color = COLORS[i % len(COLORS)]
        pts = np.array(zone.points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (*color[:3],))
        cv2.polylines(frame, [pts], True, color, 2)

        centroid = pts.mean(axis=0).astype(int)
        cv2.putText(
            frame, zone.name, tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame


def draw_hud(frame: np.ndarray, fps: float, track_count: int, event_count: int) -> np.ndarray:
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Tracks: {track_count}",
        f"Events: {event_count}",
    ]
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    return frame
