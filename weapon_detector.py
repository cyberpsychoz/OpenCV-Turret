from ultralytics import YOLO
import os

class WeaponDetector:
    def __init__(self, model_path="models/weights/best.pt", conf=0.4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weapon model not found: {model_path}")

        self.model = YOLO(model_path)
        self.model.to('cpu')
        self.conf = conf

    def detect(self, frame, person_bbox=None):
        if person_bbox:
            x, y, w, h = person_bbox
            pad = int(w * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            roi = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            roi = frame
            offset = (0, 0)

        if roi.size == 0:
            return []

        results = self.model(roi, conf=self.conf, device='cpu', verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls)
                cls_name = self.model.names.get(cls_id, 'weapon')

                detections.append({
                    'bbox': (bx1 + offset[0], by1 + offset[1], bx2 - bx1, by2 - by1),
                    'conf': conf,
                    'class': cls_name,
                    'class_id': cls_id
                })

        return detections

    def detect_in_frame(self, frame):
        return self.detect(frame, person_bbox=None)

    def get_classes(self):
        return self.model.names
