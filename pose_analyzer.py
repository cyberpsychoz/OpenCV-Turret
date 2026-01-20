import cv2
import mediapipe as mp
import numpy as np
import config

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=config.POSE_MIN_DETECTION_CONF,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONF
        )

        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24

    def analyze(self, frame, bbox):
        x, y, w, h = bbox

        pad = int(w * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {'aggressive': 0.0, 'landmarks': None}

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return {'aggressive': 0.0, 'landmarks': None}

        lm = results.pose_landmarks.landmark
        score = self._calculate_aggression(lm)

        return {
            'aggressive': score,
            'landmarks': lm,
            'roi_offset': (x1, y1)
        }

    def _calculate_aggression(self, lm):
        score = 0.0

        arms_forward = self._check_arms_forward(lm)
        if arms_forward > 0.5:
            score += 0.4 * arms_forward

        arms_raised = self._check_arms_raised(lm)
        if arms_raised > 0.5:
            score += 0.3 * arms_raised

        combat_stance = self._check_combat_stance(lm)
        if combat_stance > 0.5:
            score += 0.3 * combat_stance

        return min(1.0, score)

    def _check_arms_forward(self, lm):
        l_shoulder = lm[self.LEFT_SHOULDER]
        r_shoulder = lm[self.RIGHT_SHOULDER]
        l_wrist = lm[self.LEFT_WRIST]
        r_wrist = lm[self.RIGHT_WRIST]
        nose = lm[self.NOSE]

        if l_wrist.visibility < 0.5 or r_wrist.visibility < 0.5:
            return 0.0

        shoulder_z = (l_shoulder.z + r_shoulder.z) / 2
        wrist_z = (l_wrist.z + r_wrist.z) / 2

        forward = shoulder_z - wrist_z
        if forward > 0.1:
            wrist_y = (l_wrist.y + r_wrist.y) / 2
            shoulder_y = (l_shoulder.y + r_shoulder.y) / 2

            if abs(wrist_y - shoulder_y) < 0.15:
                return min(1.0, forward * 3)

        return 0.0

    def _check_arms_raised(self, lm):
        l_shoulder = lm[self.LEFT_SHOULDER]
        r_shoulder = lm[self.RIGHT_SHOULDER]
        l_wrist = lm[self.LEFT_WRIST]
        r_wrist = lm[self.RIGHT_WRIST]

        if l_wrist.visibility < 0.5 and r_wrist.visibility < 0.5:
            return 0.0

        score = 0.0

        if l_wrist.visibility > 0.5 and l_wrist.y < l_shoulder.y:
            score += 0.5

        if r_wrist.visibility > 0.5 and r_wrist.y < r_shoulder.y:
            score += 0.5

        return score

    def _check_combat_stance(self, lm):
        l_shoulder = lm[self.LEFT_SHOULDER]
        r_shoulder = lm[self.RIGHT_SHOULDER]
        l_hip = lm[self.LEFT_HIP]
        r_hip = lm[self.RIGHT_HIP]

        if l_hip.visibility < 0.5 or r_hip.visibility < 0.5:
            return 0.0

        shoulder_width = abs(l_shoulder.x - r_shoulder.x)
        hip_width = abs(l_hip.x - r_hip.x)

        if hip_width > shoulder_width * 1.2:
            return min(1.0, (hip_width / shoulder_width - 1.0) * 2)

        return 0.0
