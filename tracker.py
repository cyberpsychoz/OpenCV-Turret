import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import config


class Tracker:
    """
    Enhanced tracker with Kalman filter and appearance features.
    Uses color histogram matching for better re-identification.
    """

    def __init__(self, use_appearance=True):
        self.targets = {}
        self.next_id = 0
        self.max_disappeared = config.MAX_DISAPPEARED_FRAMES
        self.use_appearance = use_appearance
        self.last_frame = None

    def update(self, detections, frame=None):
        """
        Update tracker with new detections.

        Args:
            detections: List of {'bbox': (x,y,w,h), 'conf': float}
            frame: Optional frame for appearance features

        Returns:
            Dict of active targets
        """
        self.last_frame = frame

        if not detections:
            self._mark_disappeared()
            return self.targets

        if not self.targets:
            for det in detections:
                self._create_target(det, frame)
            return self.targets

        pred_boxes = []
        target_ids = list(self.targets.keys())

        for tid in target_ids:
            t = self.targets[tid]
            t['kf'].predict()
            px, py, pw, ph = t['kf'].x[:4].flatten()
            pred_boxes.append((int(px), int(py), int(pw), int(ph)))

        det_boxes = [d['bbox'] for d in detections]
        cost = self._calc_cost_matrix(pred_boxes, det_boxes, target_ids, frame)

        row_idx, col_idx = linear_sum_assignment(cost)

        matched_det = set()
        matched_tar = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < 2.0:  # Increased threshold with appearance
                tid = target_ids[r]
                det = detections[c]
                self._update_target(tid, det, frame)
                matched_det.add(c)
                matched_tar.add(tid)

        for tid in target_ids:
            if tid not in matched_tar:
                self.targets[tid]['disappeared'] += 1
                if self.targets[tid]['disappeared'] > self.max_disappeared:
                    del self.targets[tid]

        for i, det in enumerate(detections):
            if i not in matched_det:
                self._create_target(det, frame)

        return self.targets

    def _create_target(self, detection, frame=None):
        bbox = detection['bbox']
        kf = self._init_kalman(bbox)

        hist = None
        if self.use_appearance and frame is not None:
            hist = self._calc_histogram(frame, bbox)

        self.targets[self.next_id] = {
            'id': self.next_id,
            'bbox': bbox,
            'kf': kf,
            'disappeared': 0,
            'age': 0,
            'threat': {'total': 0, 'level': 'LOW', 'components': {}},
            'histogram': hist,
            'conf': detection.get('conf', 0.5)
        }
        self.next_id += 1

    def _update_target(self, tid, detection, frame=None):
        bbox = detection['bbox']
        t = self.targets[tid]
        t['kf'].update(np.array(bbox).reshape((4, 1)))
        t['bbox'] = bbox
        t['disappeared'] = 0
        t['age'] += 1
        t['conf'] = detection.get('conf', t['conf'])

        # Update appearance with exponential moving average
        if self.use_appearance and frame is not None:
            new_hist = self._calc_histogram(frame, bbox)
            if new_hist is not None and t['histogram'] is not None:
                alpha = 0.3  # Learning rate
                t['histogram'] = alpha * new_hist + (1 - alpha) * t['histogram']
            elif new_hist is not None:
                t['histogram'] = new_hist

    def _init_kalman(self, bbox):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.x = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0]).reshape((8, 1))

        kf.F = np.eye(8)
        kf.F[0, 4] = kf.F[1, 5] = kf.F[2, 6] = kf.F[3, 7] = 1

        kf.H = np.eye(4, 8)
        kf.Q = np.eye(8) * 0.1
        kf.Q[4:, 4:] *= 0.01
        kf.R = np.eye(4) * 1.0
        kf.P *= 10.0

        return kf

    def _calc_histogram(self, frame, bbox):
        """Calculate color histogram for appearance matching."""
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        if x2 <= x or y2 <= y:
            return None

        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return None

        # Convert to HSV for better color matching
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Calculate histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def _compare_histograms(self, hist1, hist2):
        """Compare two histograms using correlation."""
        if hist1 is None or hist2 is None:
            return 0.5  # Neutral score if no histogram

        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def _calc_cost_matrix(self, pred_boxes, det_boxes, target_ids, frame=None):
        cost = np.zeros((len(pred_boxes), len(det_boxes)))

        for i, pb in enumerate(pred_boxes):
            tid = target_ids[i]
            target_hist = self.targets[tid].get('histogram')

            for j, db in enumerate(det_boxes):
                # IoU cost
                iou = self._calc_iou(pb, db)
                iou_cost = 1 - iou

                # Distance cost
                dist = self._calc_distance(pb, db)
                dist_cost = dist * 0.005

                # Appearance cost
                appearance_cost = 0
                if self.use_appearance and frame is not None and target_hist is not None:
                    det_hist = self._calc_histogram(frame, db)
                    similarity = self._compare_histograms(target_hist, det_hist)
                    appearance_cost = (1 - similarity) * 0.5

                cost[i, j] = iou_cost + dist_cost + appearance_cost

        return cost

    def _calc_iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])

        if x2 <= x1 or y2 <= y1:
            return 0

        inter = (x2 - x1) * (y2 - y1)
        area1 = b1[2] * b1[3]
        area2 = b2[2] * b2[3]
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _calc_distance(self, b1, b2):
        c1 = (b1[0] + b1[2] / 2, b1[1] + b1[3] / 2)
        c2 = (b2[0] + b2[2] / 2, b2[1] + b2[3] / 2)
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def _mark_disappeared(self):
        for tid in list(self.targets.keys()):
            self.targets[tid]['disappeared'] += 1
            if self.targets[tid]['disappeared'] > self.max_disappeared:
                del self.targets[tid]
