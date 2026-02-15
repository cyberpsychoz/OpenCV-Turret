from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import Event


class AlertHandler(ABC):
    @abstractmethod
    def send(self, event: Event) -> None:
        ...

    def close(self) -> None:
        pass
