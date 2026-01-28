"""
Detection filtering to reduce false positives.
Filters based on aspect ratio, size, temporal consistency, and motion.
"""
import numpy as np
from collections import defaultdict
import time


class DetectionFilter:
    """
    Filters detections to reduce false positives.

    Criteria:
    - Aspect ratio: People are typically taller than wide (0.3 < w/h < 0.9)
    - Size bounds: Filter very small or very large detections
    - Temporal consistency: Require detection in N consecutive frames
    - Motion: Stationary objects over long time are likely not people
    """

    def __init__(self,
                 min_aspect_ratio=0.2,
                 max_aspect_ratio=0.95,
                 min_area_ratio=0.001,
                 max_area_ratio=0.8,
                 temporal_frames=2,
                 stationary_threshold=30,
                 iou_threshold=0.3):
        """
        Args:
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            min_area_ratio: Minimum bbox area / frame area
            max_area_ratio: Maximum bbox area / frame area
            temporal_frames: Require detection in N frames before confirming
            stationary_threshold: Frames stationary before filtering
            iou_threshold: IoU threshold for matching detections across frames
        """
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.temporal_frames = temporal_frames
        self.stationary_threshold = stationary_threshold
        self.iou_threshold = iou_threshold

        # Tracking for temporal filtering
        self.pending_detections = {}  # id -> {bbox, count, last_seen, positions}
        self.next_pending_id = 0
        self.frame_count = 0

    def filter(self, detections, frame_shape):
        """
        Filter detections.

        Args:
            detections: List of {'bbox': (x,y,w,h), 'conf': float, ...}
            frame_shape: (height, width, channels)

        Returns:
            Filtered list of detections
        """
        self.frame_count += 1
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w

        filtered = []

        for det in detections:
            bbox = det['bbox']
            x, y, w, h = bbox

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            # Aspect ratio filter
            aspect = w / h
            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                continue

            # Size filter
            area = w * h
            area_ratio = area / frame_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            # Temporal filter - check if this detection matches pending
            matched_pending = self._match_pending(bbox)

            if matched_pending is not None:
                # Update existing pending detection
                pid = matched_pending
                self.pending_detections[pid]['count'] += 1
                self.pending_detections[pid]['last_seen'] = self.frame_count
                self.pending_detections[pid]['positions'].append((x + w//2, y + h//2))

                # Keep only last N positions
                if len(self.pending_detections[pid]['positions']) > self.stationary_threshold:
                    self.pending_detections[pid]['positions'].pop(0)

                # Check if confirmed (seen enough times)
                if self.pending_detections[pid]['count'] >= self.temporal_frames:
                    # Check if not stationary for too long
                    if not self._is_stationary(pid):
                        det_copy = det.copy()
                        det_copy['filtered'] = True
                        filtered.append(det_copy)
            else:
                # New pending detection
                self.pending_detections[self.next_pending_id] = {
                    'bbox': bbox,
                    'count': 1,
                    'last_seen': self.frame_count,
                    'positions': [(x + w//2, y + h//2)]
                }
                self.next_pending_id += 1

        # Cleanup old pending detections
        self._cleanup_pending()

        return filtered

    def _match_pending(self, bbox):
        """Find matching pending detection by IoU."""
        best_iou = 0
        best_id = None

        for pid, pdata in self.pending_detections.items():
            iou = self._calc_iou(bbox, pdata['bbox'])
            if iou > best_iou and iou > self.iou_threshold:
                best_iou = iou
                best_id = pid

        if best_id is not None:
            # Update bbox to current
            self.pending_detections[best_id]['bbox'] = bbox

        return best_id

    def _calc_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to x1,y1,x2,y2
        ax1, ay1, ax2, ay2 = x1, y1, x1+w1, y1+h1
        bx1, by1, bx2, by2 = x2, y2, x2+w2, y2+h2

        # Intersection
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _is_stationary(self, pid):
        """Check if detection has been stationary too long."""
        positions = self.pending_detections[pid]['positions']

        if len(positions) < self.stationary_threshold:
            return False

        # Calculate total movement
        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_movement += np.sqrt(dx*dx + dy*dy)

        # If very little movement over many frames, likely stationary object
        avg_movement = total_movement / len(positions)
        return avg_movement < 2.0  # pixels per frame threshold

    def _cleanup_pending(self):
        """Remove old pending detections."""
        to_remove = []
        for pid, pdata in self.pending_detections.items():
            if self.frame_count - pdata['last_seen'] > 10:
                to_remove.append(pid)

        for pid in to_remove:
            del self.pending_detections[pid]

    def reset(self):
        """Reset filter state."""
        self.pending_detections.clear()
        self.next_pending_id = 0
        self.frame_count = 0


class ConfidenceBooster:
    """
    Boosts confidence of detections based on context.

    - Higher confidence if detection is consistent over time
    - Higher confidence if detection is moving
    - Lower confidence if detection matches known false positive patterns
    """

    def __init__(self):
        self.history = defaultdict(list)  # track_id -> confidence history

    def boost(self, track_id, base_confidence, is_moving=False, frames_tracked=0):
        """
        Calculate boosted confidence.

        Args:
            track_id: Unique track identifier
            base_confidence: Original detection confidence
            is_moving: Whether target is moving
            frames_tracked: How many frames this target has been tracked

        Returns:
            Boosted confidence value
        """
        boost = 0.0

        # Boost for movement (people move)
        if is_moving:
            boost += 0.1

        # Boost for temporal consistency
        if frames_tracked > 5:
            boost += 0.05
        if frames_tracked > 15:
            boost += 0.05
        if frames_tracked > 30:
            boost += 0.05

        # Track confidence history
        self.history[track_id].append(base_confidence)
        if len(self.history[track_id]) > 30:
            self.history[track_id].pop(0)

        # Boost for consistent high confidence
        if len(self.history[track_id]) >= 5:
            avg_conf = np.mean(self.history[track_id][-5:])
            if avg_conf > 0.6:
                boost += 0.05

        return min(1.0, base_confidence + boost)

    def cleanup(self, active_ids):
        """Remove history for inactive tracks."""
        to_remove = [tid for tid in self.history if tid not in active_ids]
        for tid in to_remove:
            del self.history[tid]
