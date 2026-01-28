"""
Fast YOLO detector using OpenCV DNN with ONNX models.
2-4x faster than ultralytics on CPU.
"""
import cv2
import numpy as np


class FastDetector:
    """YOLO detector using OpenCV DNN backend."""

    def __init__(self, model_path="yolov8n.onnx", conf=0.4, input_size=320, classes=None):
        """
        Args:
            model_path: Path to ONNX model
            conf: Confidence threshold
            input_size: Model input size (must match export size)
            classes: List of class IDs to detect (None = all, [0] = person only)
        """
        self.conf = conf
        self.input_size = input_size
        self.classes = classes

        # Load ONNX model with OpenCV DNN
        self.net = cv2.dnn.readNetFromONNX(model_path)

        # Use CPU backend optimized for speed
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # COCO class names for person detection
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'
        }

    def detect(self, frame):
        """
        Detect objects in frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of detections: [{'bbox': (x,y,w,h), 'conf': float, 'class_id': int}, ...]
        """
        h, w = frame.shape[:2]

        # Preprocess: resize + normalize + swap channels
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1/255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)
        outputs = self.net.forward()

        # YOLOv8 output shape: (1, 84, 2100) for 320x320
        # 84 = 4 (bbox) + 80 (classes)
        # Transpose to (2100, 84)
        outputs = outputs[0].T

        detections = []

        for detection in outputs:
            # First 4 values: cx, cy, w, h (normalized to input_size)
            cx, cy, bw, bh = detection[:4]

            # Remaining values: class scores
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence < self.conf:
                continue

            if self.classes is not None and class_id not in self.classes:
                continue

            # Convert to original image coordinates
            x = int((cx - bw/2) * w / self.input_size)
            y = int((cy - bh/2) * h / self.input_size)
            box_w = int(bw * w / self.input_size)
            box_h = int(bh * h / self.input_size)

            # Clamp to image bounds
            x = max(0, x)
            y = max(0, y)
            box_w = min(box_w, w - x)
            box_h = min(box_h, h - y)

            if box_w > 0 and box_h > 0:
                detections.append({
                    'bbox': (x, y, box_w, box_h),
                    'conf': float(confidence),
                    'class_id': int(class_id)
                })

        # Apply NMS
        if detections:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections, iou_threshold=0.45):
        """Non-maximum suppression."""
        if not detections:
            return []

        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['conf'] for d in detections])

        # Convert to x1,y1,x2,y2
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf,
            iou_threshold
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]


class FastWeaponDetector(FastDetector):
    """Weapon detector using ONNX model."""

    def __init__(self, model_path="models/weights/best.onnx", conf=0.3, input_size=320):
        super().__init__(model_path, conf, input_size, classes=None)

        # Weapon class names from the model
        self.class_names = {
            0: 'Gun',
            1: 'explosion',
            2: 'grenade',
            3: 'knife'
        }

    def detect(self, frame, person_bbox=None):
        """
        Detect weapons in frame or within person bbox.

        Args:
            frame: BGR image
            person_bbox: Optional (x,y,w,h) to crop search area

        Returns:
            List of weapon detections
        """
        if person_bbox is not None:
            x, y, w, h = person_bbox
            # Expand bbox slightly
            pad = int(w * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return []

            offset = (x1, y1)
        else:
            roi = frame
            offset = (0, 0)

        detections = super().detect(roi)

        # Adjust coordinates for offset
        result = []
        for d in detections:
            bx, by, bw, bh = d['bbox']
            result.append({
                'bbox': (bx + offset[0], by + offset[1], bw, bh),
                'conf': d['conf'],
                'class': self.class_names.get(d['class_id'], 'weapon'),
                'class_id': d['class_id']
            })

        return result


class FastPersonDetector(FastDetector):
    """Person-only detector for maximum speed."""

    def __init__(self, model_path="yolov8n.onnx", conf=0.4, input_size=320):
        # Only detect class 0 (person)
        super().__init__(model_path, conf, input_size, classes=[0])
