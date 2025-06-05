import cv2
import os
import time
from detector import PersonDetector
from color_classifier import ColorClassifier
from head_detector import HeadDetector
from utils import get_video_files, create_output_dir, get_output_path

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def draw_progress_bar(progress, total_length=50):
    filled = int(progress * total_length)
    return '[' + '#' * filled + '-' * (total_length - filled) + ']'

def process_video(video_path, detector, classifier, head_detector):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть видео {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = get_output_path(video_path).replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"\n[INFO] Обрабатываю: {os.path.basename(video_path)}")
    print(f"Формат: {frame_width}x{frame_height}, {fps} FPS, {total_frames} кадров")

    start_time = time.time()
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people_boxes = detector.detect(frame)
        
        for box in people_boxes:
            x, y, w, h = box
            
            # ограничение по размеру поиска людей в видео
            #aspect_ratio = w / h
            #if aspect_ratio < 0.2 or aspect_ratio > 1.5 or w < 50 or h < 100:
            #    continue

            try:
                head_roi = head_detector.get_head_region(frame, box)
            except Exception as e:
                print(f"[WARNING] Не удалось определить голову: {e}")
                continue
            
            color = classifier.classify(head_roi)
            
            # Визуализация
            color_map = {"red": (0,0,255), "blue": (255,0,0), "unknown": (128,128,128)}
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[color], 2)
            cv2.putText(frame, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_map[color], 2)

        out.write(frame)
        processed_frames += 1

        # Прогресс
        if total_frames > 0:
            progress = processed_frames / total_frames
            bar = draw_progress_bar(progress)
            elapsed = time.time() - start_time
            eta = (elapsed / max(1, progress)) * (1 - progress) if progress > 0 else 0
            print(f"\r{bar} {progress*100:.1f}% | Пройдено: {format_time(elapsed)} | Осталось: {format_time(eta)}", end="", flush=True)
        else:
            elapsed = time.time() - start_time
            print(f"\rОбработано кадров: {processed_frames} | Прошло: {format_time(elapsed)}", end="", flush=True)

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"\n[INFO] Готово: {os.path.basename(output_path)} | Затрачено: {format_time(total_time)}")

def main():
    create_output_dir()
    detector = PersonDetector()
    classifier = ColorClassifier()
    head_detector = HeadDetector()
    
    video_files = get_video_files()
    
    if not video_files:
        print("[ERROR] Не найдено видеофайлов в папке test_videos")
        return
    
    print(f"\n[INFO] Найдено {len(video_files)} видео для обработки:")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i}. {video_file}")
    
    for video_file in video_files:
        video_path = os.path.join("test_videos", video_file)
        process_video(video_path, detector, classifier, head_detector)

if __name__ == "__main__":
    main()