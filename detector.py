from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self):
        self.model = YOLO("yolov8s.pt")
        self.confidence_threshold = 0.5
        self.person_class_id = 0

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        boxes = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                if cls == self.person_class_id and conf > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append((x1, y1, w, h))
        return boxes