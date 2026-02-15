from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from src.config import AppConfig
from src.core.models import Event, Track
from src.core.tracker import AnimalTracker
from src.core.zone_manager import ZoneManager, load_zones
from src.sources.base import VideoSource
from src.ui.overlay import draw_hud, draw_tracks, draw_zones

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._tracker = AnimalTracker(config.detector)
        zones = load_zones(config.zones_file)
        self._zone_manager = ZoneManager(zones)
        self._event_handlers: list = []
        self._total_events = 0

    def add_event_handler(self, handler) -> None:
        self._event_handlers.append(handler)

    def process_frame(self, frame: np.ndarray) -> tuple[list[Track], list[Event], np.ndarray]:
        tracks = self._tracker.track(frame)
        events = self._zone_manager.check(tracks)

        for event in events:
            self._total_events += 1
            for handler in self._event_handlers:
                try:
                    handler(event, frame)
                except Exception:
                    logger.exception("Event handler failed")

        return tracks, events, frame

    def run(self, source: VideoSource) -> None:
        cfg = self._config.display
        frame_count = 0
        fps_timer = time.time()
        fps_display = 0.0

        logger.info("Pipeline started")

        try:
            while True:
                ret, frame = source.read()
                if not ret or frame is None:
                    break

                tracks, events, annotated = self.process_frame(frame)

                if cfg.show_window:
                    if cfg.show_zones:
                        draw_zones(annotated, self._zone_manager.zones)
                    if cfg.show_tracks:
                        draw_tracks(annotated, tracks)
                    if cfg.show_fps:
                        draw_hud(annotated, fps_display, len(tracks), self._total_events)

                    cv2.imshow("Wildlife Deterrent", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break

                frame_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps_display = frame_count / elapsed
                    frame_count = 0
                    fps_timer = time.time()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            source.release()
            cv2.destroyAllWindows()
            logger.info("Pipeline stopped")
