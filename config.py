import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

YOLO_MODEL = "yolov8n.onnx"
YOLO_MODEL_PT = "yolov8n.pt"
YOLO_IMGSZ = 320
YOLO_CONF = 0.35

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
