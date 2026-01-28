import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Use yolov8s for better accuracy (slower) or yolov8n for speed
YOLO_MODEL = "yolov8s.onnx"
YOLO_MODEL_FAST = "yolov8n.onnx"
YOLO_IMGSZ = 416  # 416 for yolov8s, 320 for yolov8n
YOLO_IMGSZ_FAST = 320
YOLO_CONF = 0.4

# Detection filtering
FILTER_MIN_ASPECT = 0.2   # min width/height
FILTER_MAX_ASPECT = 0.95  # max width/height
FILTER_MIN_AREA = 0.002   # min bbox area / frame area
FILTER_MAX_AREA = 0.7     # max bbox area / frame area
FILTER_TEMPORAL_FRAMES = 2  # require N frames before confirming

POSE_MODEL_COMPLEXITY = 0
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5

FRAME_SKIP = 2

WEAPON_MODEL = "models/weights/best.onnx"
WEAPON_MODEL_PT = "models/weights/best.pt"
WEAPON_CONF = 0.25

DEPTH_MODEL = "MiDaS_small"
DEPTH_ENABLED = True

THREAT_WEIGHTS = {
    'weapon_detected': 0.45,
    'aggressive_pose': 0.25,
    'moving_toward': 0.15,
    'rapid_movement': 0.10,
    'proximity': 0.05
}

THREAT_THRESHOLD_ENGAGE = 0.7
THREAT_THRESHOLD_WARN = 0.4

MAX_DISAPPEARED_FRAMES = 30
