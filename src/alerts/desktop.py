from __future__ import annotations

import logging

from src.alerts.base import AlertHandler
from src.core.models import Event

logger = logging.getLogger(__name__)


class DesktopAlert(AlertHandler):
    def __init__(self) -> None:
        self._available = False
        try:
            from plyer import notification
            self._notification = notification
            self._available = True
            logger.info("Desktop notifications enabled")
        except ImportError:
            logger.warning("plyer not available — desktop notifications disabled")

    def send(self, event: Event) -> None:
        if not self._available:
            return

        title = f"Wildlife Alert: {event.species}"
        message = f"{event.event_type.value} in zone '{event.zone_name}' (track #{event.track_id})"

        try:
            self._notification.notify(
                title=title,
                message=message,
                app_name="Wildlife Deterrent",
                timeout=5,
            )
        except Exception:
            logger.exception("Desktop notification failed")
