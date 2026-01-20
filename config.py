import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

YOLO_MODEL = "yolov8n.pt"
YOLO_IMGSZ = 320
YOLO_CONF = 0.4

POSE_MODEL_COMPLEXITY = 0
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5

FRAME_SKIP = 2

THREAT_WEIGHTS = {
    'aggressive_pose': 0.35,
    'moving_toward': 0.30,
    'rapid_movement': 0.20,
    'proximity': 0.15
}

THREAT_THRESHOLD_ENGAGE = 0.7
THREAT_THRESHOLD_WARN = 0.4

MAX_DISAPPEARED_FRAMES = 30
