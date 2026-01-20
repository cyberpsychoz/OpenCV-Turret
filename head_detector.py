import cv2
import mediapipe as mp

class BodyPartDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=0)
        self.cached_results = {}
        self.cache_frame_id = -1

    def _get_pose_results(self, frame, box, frame_id=None):
        cache_key = (box[0], box[1], box[2], box[3])

        if frame_id is not None and frame_id == self.cache_frame_id:
            if cache_key in self.cached_results:
                return self.cached_results[cache_key]

        if frame_id is not None and frame_id != self.cache_frame_id:
            self.cached_results.clear()
            self.cache_frame_id = frame_id

        x, y, w, h = box
        person_roi = frame[y:y+h, x:x+w]

        if person_roi.size == 0:
            return None, person_roi

        results = self.pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
        self.cached_results[cache_key] = (results, person_roi)
        return results, person_roi

    def get_head_region(self, frame, box, frame_id=None):
        x, y, w, h = box
        results, person_roi = self._get_pose_results(frame, box, frame_id)

        if person_roi.size == 0:
            return frame[y:y+h//4, x:x+w]

        if results and results.pose_landmarks:
            landmark = results.pose_landmarks.landmark[0]
            head_x = int(landmark.x * person_roi.shape[1])
            head_y = int(landmark.y * person_roi.shape[0])
            return person_roi[max(0, head_y-30):head_y+10, max(0, head_x-30):head_x+30]

        return frame[y:y+h//4, x:x+w]

    def get_arm_regions(self, frame, box, frame_id=None):
        x, y, w, h = box
        results, person_roi = self._get_pose_results(frame, box, frame_id)

        if person_roi.size == 0:
            return []

        if results and results.pose_landmarks:
            arm_regions = []
            landmarks = results.pose_landmarks.landmark

            for idx in [11, 13, 15, 12, 14, 16]:
                lm = landmarks[idx]
                px = int(lm.x * person_roi.shape[1])
                py = int(lm.y * person_roi.shape[0])
                region = person_roi[max(0, py-20):py+20, max(0, px-20):px+20]
                if region.size > 0:
                    arm_regions.append(region)

            return arm_regions

        return []

    def get_all_regions(self, frame, box, frame_id=None):
        x, y, w, h = box
        results, person_roi = self._get_pose_results(frame, box, frame_id)

        head_roi = None
        arm_rois = []

        if person_roi.size == 0:
            head_roi = frame[y:y+h//4, x:x+w]
            return head_roi, arm_rois

        if results and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            landmark = landmarks[0]
            head_x = int(landmark.x * person_roi.shape[1])
            head_y = int(landmark.y * person_roi.shape[0])
            head_roi = person_roi[max(0, head_y-30):head_y+10, max(0, head_x-30):head_x+30]

            for idx in [11, 13, 15, 12, 14, 16]:
                lm = landmarks[idx]
                px = int(lm.x * person_roi.shape[1])
                py = int(lm.y * person_roi.shape[0])
                region = person_roi[max(0, py-20):py+20, max(0, px-20):px+20]
                if region.size > 0:
                    arm_rois.append(region)
        else:
            head_roi = frame[y:y+h//4, x:x+w]

        return head_roi, arm_rois
