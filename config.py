import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Use yolov8s for better accuracy (slower) or yolov8n for speed
YOLO_MODEL = "yolov8s.onnx"
YOLO_MODEL_FAST = "yolov8n.onnx"
YOLO_IMGSZ = 416  # 416 for yolov8s, 320 for yolov8n
YOLO_IMGSZ_FAST = 320
YOLO_CONF = 0.3  # Lower to not miss people

# Detection filtering (relaxed to not lose people)
FILTER_MIN_ASPECT = 0.15  # min width/height (people can be wide when crouching)
FILTER_MAX_ASPECT = 1.2   # max width/height (allow wider poses)
FILTER_MIN_AREA = 0.001   # min bbox area / frame area
FILTER_MAX_AREA = 0.85    # max bbox area / frame area
FILTER_TEMPORAL_FRAMES = 1  # immediate detection (no delay)

POSE_MODEL_COMPLEXITY = 0
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5

FRAME_SKIP = 2

WEAPON_MODEL = "models/weights/best.onnx"
WEAPON_MODEL_PT = "models/weights/best.pt"
WEAPON_CONF = 0.15  # Low threshold - better to detect than miss

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
