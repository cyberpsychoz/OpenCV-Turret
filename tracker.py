import numpy as np
import time

class SimpleTargetTracker:
    def __init__(self):
        self.targets = {}
        self.next_id = 0
        self.fire_delay = 3
        self.lost_threshold = 10

    def update(self, detected_boxes):
        current_time = time.time()
        matched = {}

        for new_box in detected_boxes:
            best_match = None
            min_distance = float('inf')
            
            for tid, target in self.targets.items():
                dx = (new_box[0] - target['bbox'][0])**2
                dy = (new_box[1] - target['bbox'][1])**2
                distance = (dx + dy)**0.5
                if distance < min_distance and distance < 50:
                    min_distance = distance
                    best_match = tid
            
            if best_match:
                self.targets[best_match]['bbox'] = new_box
                self.targets[best_match]['last_seen'] = current_time
                matched[best_match] = new_box
            else:
                self.targets[self.next_id] = {
                    'bbox': new_box,
                    'last_seen': current_time,
                    'color': 'unknown',
                    'fire_timer': None
                }
                matched[self.next_id] = new_box
                self.next_id += 1

        lost = []
        for tid in list(self.targets.keys()):
            if tid not in matched and current_time - self.targets[tid]['last_seen'] > self.lost_threshold:
                lost.append(tid)
        
        for tid in lost:
            del self.targets[tid]
        
        return self.targets