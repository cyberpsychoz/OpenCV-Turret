import config

class ThreatScorer:
    def __init__(self):
        self.weights = config.THREAT_WEIGHTS

    def calculate(self, pose_result, motion_result, bbox, frame_shape):
        scores = {
            'aggressive_pose': pose_result.get('aggressive', 0.0),
            'moving_toward': motion_result.get('toward', 0.0),
            'rapid_movement': motion_result.get('speed', 0.0),
            'proximity': self._calc_proximity(bbox, frame_shape)
        }

        total = sum(scores[k] * self.weights[k] for k in self.weights)

        if scores['aggressive_pose'] > 0.6 and scores['moving_toward'] > 0.4:
            total = min(1.0, total * 1.3)

        level = 'LOW'
        if total >= config.THREAT_THRESHOLD_ENGAGE:
            level = 'HIGH'
        elif total >= config.THREAT_THRESHOLD_WARN:
            level = 'MEDIUM'

        return {
            'total': total,
            'level': level,
            'components': scores
        }

    def _calc_proximity(self, bbox, frame_shape):
        x, y, w, h = bbox
        area = w * h
        frame_area = frame_shape[0] * frame_shape[1]
        ratio = area / frame_area

        return min(1.0, ratio * 20)
