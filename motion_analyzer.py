import numpy as np
from collections import deque

class MotionAnalyzer:
    def __init__(self, history_size=10):
        self.history = {}
        self.history_size = history_size

    def update(self, target_id, bbox, frame_shape):
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        area = w * h

        if target_id not in self.history:
            self.history[target_id] = deque(maxlen=self.history_size)

        self.history[target_id].append({
            'center': (cx, cy),
            'area': area,
            'bbox': bbox
        })

    def analyze(self, target_id, frame_shape):
        if target_id not in self.history:
            return {'toward': 0.0, 'speed': 0.0, 'direction': (0, 0)}

        hist = self.history[target_id]
        if len(hist) < 3:
            return {'toward': 0.0, 'speed': 0.0, 'direction': (0, 0)}

        old = hist[0]
        new = hist[-1]

        dx = new['center'][0] - old['center'][0]
        dy = new['center'][1] - old['center'][1]

        frame_h, frame_w = frame_shape[:2]
        norm_speed = np.sqrt(dx**2 + dy**2) / max(frame_w, frame_h)
        speed_score = min(1.0, norm_speed * 10)

        area_change = (new['area'] - old['area']) / max(old['area'], 1)
        toward_score = 0.0

        if area_change > 0.05:
            toward_score = min(1.0, area_change * 5)

        frame_cx = frame_w / 2
        frame_cy = frame_h / 2

        old_dist = np.sqrt((old['center'][0] - frame_cx)**2 + (old['center'][1] - frame_cy)**2)
        new_dist = np.sqrt((new['center'][0] - frame_cx)**2 + (new['center'][1] - frame_cy)**2)

        if new_dist < old_dist:
            toward_score = max(toward_score, (old_dist - new_dist) / max(old_dist, 1) * 3)

        return {
            'toward': min(1.0, toward_score),
            'speed': speed_score,
            'direction': (dx, dy)
        }

    def cleanup(self, active_ids):
        dead = [tid for tid in self.history if tid not in active_ids]
        for tid in dead:
            del self.history[tid]
