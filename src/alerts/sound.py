from __future__ import annotations

import logging
import random
from pathlib import Path

from src.alerts.base import AlertHandler
from src.core.models import Event

logger = logging.getLogger(__name__)


class SoundAlert(AlertHandler):
    def __init__(self, sounds_dir: str = "data/sounds") -> None:
        self._sounds_dir = Path(sounds_dir)
        self._sounds_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        try:
            import pygame.mixer
            pygame.mixer.init()
            self._mixer = pygame.mixer
            self._initialized = True
            logger.info("Sound alerts enabled, dir=%s", self._sounds_dir)
        except Exception:
            logger.warning("pygame not available — sound alerts disabled")

    def send(self, event: Event) -> None:
        if not self._initialized:
            return

        wav_files = list(self._sounds_dir.glob("*.wav"))
        if not wav_files:
            logger.debug("No .wav files in %s", self._sounds_dir)
            return

        sound_file = random.choice(wav_files)
        try:
            self._mixer.Sound(str(sound_file)).play()
            logger.info("Playing deterrent sound: %s", sound_file.name)
        except Exception:
            logger.exception("Failed to play sound")

    def close(self) -> None:
        if self._initialized:
            self._mixer.quit()
