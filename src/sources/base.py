from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VideoSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        ...

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
