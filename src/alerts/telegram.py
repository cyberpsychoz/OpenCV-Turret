from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.alerts.base import AlertHandler
from src.core.models import Event

if TYPE_CHECKING:
    from src.config import TelegramConfig

logger = logging.getLogger(__name__)


class TelegramAlert(AlertHandler):
    def __init__(self, cfg: TelegramConfig) -> None:
        self._token = cfg.bot_token
        self._chat_id = cfg.chat_id

        if not self._token or not self._chat_id:
            logger.warning("Telegram bot_token/chat_id not configured")
            self._enabled = False
            return

        self._enabled = True
        self._url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        logger.info("Telegram alerts enabled for chat_id=%s", self._chat_id)

    def send(self, event: Event) -> None:
        if not self._enabled:
            return

        text = (
            f"🦊 *Wildlife Alert*\n"
            f"Species: {event.species}\n"
            f"Event: {event.event_type.value}\n"
            f"Zone: {event.zone_name}\n"
            f"Track: #{event.track_id}"
        )

        try:
            import urllib.request
            import urllib.parse
            import json

            data = urllib.parse.urlencode({
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode()

            req = urllib.request.Request(self._url, data=data)
            urllib.request.urlopen(req, timeout=5)
            logger.info("Telegram alert sent for %s in %s", event.species, event.zone_name)
        except Exception:
            logger.exception("Telegram send failed")
