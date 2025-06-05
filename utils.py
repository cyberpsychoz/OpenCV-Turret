# utils.py
import os
import cv2

def get_video_files(folder="test_videos"):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return [f for f in os.listdir(folder) if f.lower().endswith(video_extensions)]

def create_output_dir():
    os.makedirs("output", exist_ok=True)

def get_output_path(input_path):
    filename = os.path.basename(input_path)
    return os.path.join("output", f"processed_{filename}")