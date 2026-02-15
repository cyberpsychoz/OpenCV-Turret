from __future__ import annotations

import cv2
import numpy as np

from src.core.models import Track


class HeatmapAccumulator:
    def __init__(self, width: int, height: int) -> None:
        self._heatmap = np.zeros((height, width), dtype=np.float32)

    def update(self, tracks: list[Track]) -> None:
        for track in tracks:
            cx, cy = track.center
            if 0 <= cy < self._heatmap.shape[0] and 0 <= cx < self._heatmap.shape[1]:
                cv2.circle(self._heatmap, (cx, cy), 20, 1.0, -1)

    def render(self, alpha: float = 0.5) -> np.ndarray:
        normalized = cv2.normalize(self._heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        return colored

    def overlay(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        heatmap_vis = self.render()
        if heatmap_vis.shape[:2] != frame.shape[:2]:
            heatmap_vis = cv2.resize(heatmap_vis, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(heatmap_vis, alpha, frame, 1 - alpha, 0)

    def reset(self) -> None:
        self._heatmap[:] = 0
