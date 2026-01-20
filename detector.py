from ultralytics import YOLO
import config

class Detector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.model.to('cpu')

    def detect(self, frame):
        results = self.model(
            frame,
            imgsz=config.YOLO_IMGSZ,
            conf=config.YOLO_CONF,
            classes=[0],
            device='cpu',
            verbose=False
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                detections.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'conf': conf
                })

        return detections
