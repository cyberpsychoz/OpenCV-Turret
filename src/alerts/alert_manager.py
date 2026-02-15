from __future__ import annotations

import logging
import time

import numpy as np

from src.alerts.base import AlertHandler
from src.config import AlertsConfig
from src.core.models import Event, EventType

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, cfg: AlertsConfig) -> None:
        self._cooldown = cfg.cooldown_seconds
        self._handlers: list[AlertHandler] = []
        self._last_alert: dict[str, float] = {}

        if cfg.sound.enabled:
            from src.alerts.sound import SoundAlert
            self._handlers.append(SoundAlert(cfg.sound.directory))

        if cfg.desktop.enabled:
            from src.alerts.desktop import DesktopAlert
            self._handlers.append(DesktopAlert())

        if cfg.telegram.enabled:
            from src.alerts.telegram import TelegramAlert
            self._handlers.append(TelegramAlert(cfg.telegram))

        logger.info("AlertManager: %d handlers, cooldown=%ds", len(self._handlers), self._cooldown)

    def _cooldown_key(self, event: Event) -> str:
        return f"{event.zone_name}:{event.species}"

    def _should_alert(self, event: Event) -> bool:
        if event.event_type == EventType.ZONE_EXIT:
            return False

        key = self._cooldown_key(event)
        now = time.time()
        last = self._last_alert.get(key, 0.0)

        if now - last < self._cooldown:
            return False

        self._last_alert[key] = now
        return True

    def handle(self, event: Event, frame: np.ndarray) -> None:
        if not self._should_alert(event):
            return

        logger.info(
            "ALERT: %s %s in zone '%s' (track #%d)",
            event.species, event.event_type.value, event.zone_name, event.track_id,
        )

        for handler in self._handlers:
            try:
                handler.send(event)
            except Exception:
                logger.exception("Alert handler %s failed", type(handler).__name__)

    def close(self) -> None:
        for handler in self._handlers:
            handler.close()
