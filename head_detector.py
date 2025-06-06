import cv2
import mediapipe as mp

class BodyPartDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=0)

    def get_head_region(self, frame, box):
        x, y, w, h = box
        person_roi = frame[y:y+h, x:x+w]
        
        if person_roi.size == 0:
            return frame[y:y+h//4, x:x+w]
            
        results = self.pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            landmark = results.pose_landmarks.landmark[0]
            head_x = int(landmark.x * person_roi.shape[1])
            head_y = int(landmark.y * person_roi.shape[0])
            return person_roi[max(0, head_y-30):head_y+10, max(0, head_x-30):head_x+30]
        
        return frame[y:y+h//4, x:x+w]

    def get_arm_regions(self, frame, box):
        x, y, w, h = box
        person_roi = frame[y:y+h, x:x+w]
        
        if person_roi.size == 0:
            return []
            
        results = self.pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            arm_regions = []
            landmarks = results.pose_landmarks.landmark
            
            for idx in [11, 13, 15, 12, 14, 16]:
                lm = landmarks[idx]
                px = int(lm.x * person_roi.shape[1])
                py = int(lm.y * person_roi.shape[0])
                arm_regions.append(person_roi[max(0, py-20):py+20, max(0, px-20):px+20])
            
            return arm_regions
            
        return []