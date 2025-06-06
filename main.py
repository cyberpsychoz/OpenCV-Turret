import cv2
import os
import time
from detector import PersonDetector
from color_classifier import ColorClassifier
from head_detector import BodyPartDetector
from tracker import SimpleTargetTracker
from utils import get_video_files, create_output_dir, get_output_path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn

console = Console()
layout = Layout()
layout.split_column(
    Layout(name="progress", size=3),
    Layout(name="logs")
)

# Панель логов
log_table = Table.grid()
log_table.add_column("Логи целей", no_wrap=True)
logs = []

def update_logs(log_entry):
    logs.insert(0, log_entry)
    if len(logs) > 5:
        logs.pop()
    
    log_table = Table.grid()
    log_table.add_column("Логи целей", no_wrap=True)
    for log in logs:
        log_table.add_row(log)
    layout["logs"].update(Panel(log_table, title="Логи целей", border_style="cyan"))

# Прогресс-бар
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:.0f}%"),
    TextColumn("[bold blue]{task.fields[eta]}"),
    transient=True
)

progress_task = progress.add_task("[green]Обработка", total=100, eta="")

layout["progress"].update(Panel(progress, title="Прогресс", border_style="green"))

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def process_video(video_path, detector, classifier, body_detector, tracker):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        console.print(f"[red][ERROR] Не удалось открыть видео {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    output_path = get_output_path(video_path).replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    console.print(f"\n[bold cyan][INFO] Обрабатываю: {os.path.basename(video_path)}")
    console.print(f"Формат: {frame_width}x{frame_height}, {fps} FPS, {total_frames} кадров")

    start_time = time.time()
    processed_frames = 0

    with Live(layout, refresh_per_second=4):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detected_boxes = detector.detect(frame)
            tracked_targets = tracker.update(detected_boxes)
            
            for tid, target in tracked_targets.items():
                x, y, w, h = target['bbox']
                full_body_roi = frame[y:y+h, x:x+w]
                head_roi = body_detector.get_head_region(frame, (x, y, w, h))
                arm_rois = body_detector.get_arm_regions(frame, (x, y, w, h))
                
                current_color = classifier.classify(full_body_roi, [head_roi] + arm_rois)
                tracker.targets[tid]['color'] = current_color
                
                # Логируем события
                if current_color == 'red':
                    tracker.targets[tid]['fire_timer'] = None
                    update_logs(f"[green]Цель {tid}: Красная повязка")
                else:
                    if tracker.targets[tid]['fire_timer'] is None:
                        tracker.targets[tid]['fire_timer'] = time.time()
                    elif time.time() - tracker.targets[tid]['fire_timer'] > tracker.fire_delay:
                        tracker.targets[tid]['fire_timer'] = None
                        update_logs(f"[red]Цель {tid}: Нет красной повязки >{tracker.fire_delay}с")

                color_map = {"red": (0,0,255), "blue": (255,0,0), "unknown": (128,128,128)}
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[current_color], 2)
                cv2.putText(frame, f"{current_color} ID:{tid}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_map[current_color], 2)
                
                if tracker.targets[tid]['fire_timer']:
                    elapsed = time.time() - tracker.targets[tid]['fire_timer']
                    cv2.putText(frame, f"Fire in {int(tracker.fire_delay - elapsed)}s", 
                               (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            out.write(frame)
            processed_frames += 1

            # Обновляем прогресс
            if total_frames > 0:
                progress.update(progress_task, completed=int((processed_frames / total_frames) * 100), 
                              eta=f"Осталось: {format_time((time.time() - start_time) / max(1e-5, processed_frames) * (total_frames - processed_frames))}")

        cap.release()
        out.release()

        total_time = time.time() - start_time
        update_logs(f"[bold green]ГОТОВО: {os.path.basename(output_path)} | Затрачено: {format_time(total_time)}")
        progress.update(progress_task, completed=100, eta="ГОТОВО")

def main():
    create_output_dir()
    detector = PersonDetector()
    classifier = ColorClassifier()
    body_detector = BodyPartDetector()
    tracker = SimpleTargetTracker()
    
    video_files = get_video_files()
    
    if not video_files:
        console.print("[red][ERROR] Не найдено видеофайлов в папке test_videos")
        return
    
    console.print(f"[cyan][INFO] Найдено {len(video_files)} видео для обработки:")
    for i, video_file in enumerate(video_files, 1):
        console.print(f"{i}. {video_file}")
    
    for video_file in video_files:
        progress.reset(progress_task)
        update_logs(f"[bold blue]Начинаю обработку: {video_file}")
        video_path = os.path.join("test_videos", video_file)
        process_video(video_path, detector, classifier, body_detector, tracker)

if __name__ == "__main__":
    main()