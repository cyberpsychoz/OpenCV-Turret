import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import AlertsConfig
from src.core.models import Event, EventType


class TestAlertManager:
    @patch("src.alerts.sound.SoundAlert")
    @patch("src.alerts.desktop.DesktopAlert")
    def test_cooldown_prevents_spam(self, mock_desktop_cls, mock_sound_cls):
        mock_sound = MagicMock()
        mock_desktop = MagicMock()
        mock_sound_cls.return_value = mock_sound
        mock_desktop_cls.return_value = mock_desktop

        cfg = AlertsConfig(cooldown_seconds=60)
        cfg.telegram.enabled = False

        from src.alerts.alert_manager import AlertManager

        am = AlertManager(cfg)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        event = Event(
            event_type=EventType.ZONE_ENTRY,
            zone_name="garden",
            track_id=1,
            species="cat",
        )

        am.handle(event, frame)
        am.handle(event, frame)

        total_calls = mock_sound.send.call_count + mock_desktop.send.call_count
        assert total_calls == 2  # once per handler, only first event

    @patch("src.alerts.sound.SoundAlert")
    @patch("src.alerts.desktop.DesktopAlert")
    def test_exit_events_not_alerted(self, mock_desktop_cls, mock_sound_cls):
        mock_sound_cls.return_value = MagicMock()
        mock_desktop_cls.return_value = MagicMock()

        cfg = AlertsConfig(cooldown_seconds=0)
        cfg.telegram.enabled = False

        from src.alerts.alert_manager import AlertManager

        am = AlertManager(cfg)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        event = Event(
            event_type=EventType.ZONE_EXIT,
            zone_name="garden",
            track_id=1,
            species="cat",
        )

        am.handle(event, frame)
        assert mock_sound_cls.return_value.send.call_count == 0

    @patch("src.alerts.sound.SoundAlert")
    @patch("src.alerts.desktop.DesktopAlert")
    def test_different_keys_both_alerted(self, mock_desktop_cls, mock_sound_cls):
        mock_sound = MagicMock()
        mock_desktop = MagicMock()
        mock_sound_cls.return_value = mock_sound
        mock_desktop_cls.return_value = mock_desktop

        cfg = AlertsConfig(cooldown_seconds=60)
        cfg.telegram.enabled = False

        from src.alerts.alert_manager import AlertManager

        am = AlertManager(cfg)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        event1 = Event(event_type=EventType.ZONE_ENTRY, zone_name="garden", track_id=1, species="cat")
        event2 = Event(event_type=EventType.ZONE_ENTRY, zone_name="garden", track_id=2, species="dog")

        am.handle(event1, frame)
        am.handle(event2, frame)

        assert mock_sound.send.call_count == 2
