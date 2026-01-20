import cv2
import os
import time
import logging
from detector import PersonDetector
from head_detector import BodyPartDetector
from color_classifier import ImprovedColorClassifier
from tracker import ImprovedTargetTracker
from safety_system import SafetySystem, EngagementDecision, ThreatLevel
from utils import get_video_files, create_output_dir, get_output_path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn
from rich.text import Text
import numpy as np
import traceback



# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('turret_system.log'),
        logging.StreamHandler()
    ]
)

console = Console()
logger = logging.getLogger(__name__)

class TurretSystem:
    def __init__(self):
        # Инициализация всех компонентов
        self.detector = PersonDetector()
        self.classifier = ImprovedColorClassifier()
        self.body_detector = BodyPartDetector()
        self.tracker = ImprovedTargetTracker()
        self.safety_system = SafetySystem()
        
        # Статистика системы
        self.system_stats = {
            'processed_frames': 0,
            'detected_persons': 0,
            'red_bandana_detections': 0,
            'engagement_decisions': 0,
            'safety_violations': 0,
            'system_start_time': time.time(),
            'targets_engaged': 0,
            'false_positives_prevented': 0
        }
        
        # Настройки обработки
        self.fps_calculator = FPSCalculator()
        self.frame_skip = 1  # Обрабатывать каждый кадр для лучшего обнаружения
        self.current_frame_count = 0
        
        # Настройка интерфейса
        self.setup_interface()

        logger.info("Turret system initialized successfully")
    
    def setup_interface(self):
        """Настройка Rich интерфейса"""
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main").split_row(
                Layout(name="stats", ratio=1),
                Layout(name="logs", ratio=2)
                ),
            Layout(name="progress", size=3)
        )
        
        # Заголовок
        header_text = Text("🎯 ADVANCED TURRET SYSTEM v2.0", style="bold red")
        self.layout["header"].update(Panel(header_text, style="red"))
        
        # Прогресс-бар
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:.1f}%"),
            TextColumn("•"),
            TextColumn("[bold blue]{task.fields[fps]} FPS"),
            TextColumn("•"),
            TextColumn("[bold green]{task.fields[targets]} targets"),
            expand=True
        )
        
        self.progress_task = self.progress.add_task(
            "[green]Processing...", 
            total=100, 
            fps="0", 
            targets="0"
        )
        
        self.layout["progress"].update(Panel(self.progress, title="Processing Status", border_style="green"))
        
        # Логи и статистика
        self.logs = []
        self.update_interface()
    
    def update_interface(self):
        """Обновление интерфейса"""

        if not hasattr(self, 'layout') or not self.layout._children: return

        # Статистика
        stats_table = Table(title="System Statistics", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        uptime = time.time() - self.system_stats['system_start_time']
        stats_table.add_row("Uptime", f"{uptime:.1f}s")
        stats_table.add_row("Frames Processed", str(self.system_stats['processed_frames']))
        stats_table.add_row("Persons Detected", str(self.system_stats['detected_persons']))
        stats_table.add_row("Red Bandanas", str(self.system_stats['red_bandana_detections']))
        stats_table.add_row("Engagement Decisions", str(self.system_stats['engagement_decisions']))
        stats_table.add_row("Targets Engaged", str(self.system_stats['targets_engaged']))
        stats_table.add_row("False Positives Prevented", str(self.system_stats['false_positives_prevented']))
        stats_table.add_row("Safety Violations", str(self.system_stats['safety_violations']))
        
        # Статистика трекинга
        tracking_stats = self.tracker.get_tracking_statistics()
        stats_table.add_row("Active Targets", str(tracking_stats['active_targets']))
        stats_table.add_row("Average Stability", f"{tracking_stats['average_stability']:.2f}")

        self.layout["stats"].update(Panel(stats_table, border_style="cyan"))
        
        # Логи
        logs_table = Table(title="System Logs", show_header=False)
        logs_table.add_column("Time", style="dim")
        logs_table.add_column("Event", style="white")
        
        for log_entry in self.logs[-10:]:  # Показываем последние 10 записей
            logs_table.add_row(log_entry['time'], log_entry['message'])
        
        self.layout["logs"].update(Panel(logs_table, border_style="yellow"))
    
    def add_log(self, message, level="INFO"):
        """Добавление записи в лог"""
        timestamp = time.strftime("%H:%M:%S")
        
        # Цветовое кодирование по уровню
        if level == "DANGER":
            styled_message = f"[bold red]⚠️  {message}[/]"
        elif level == "WARNING":
            styled_message = f"[yellow]⚡ {message}[/]"
        elif level == "ENGAGE":
            styled_message = f"[bold red]🎯 {message}[/]"
        elif level == "SAFE":
            styled_message = f"[green]✅ {message}[/]"
        elif level == "RED_BANDANA":
            styled_message = f"[bold cyan]🔴 {message}[/]"
        else:
            styled_message = f"[white]ℹ️  {message}[/]"
        
        self.logs.append({
            'time': timestamp,
            'message': styled_message
        })
        
        # Ограничиваем размер лога
        if len(self.logs) > 100:
            self.logs.pop(0)
        
        logger.info(f"{level}: {message}")
    
    def process_frame(self, frame, frame_number, total_frames):
        """Обработка одного кадра"""
        self.current_frame_count += 1
        
        # Пропуск кадров для производительности
        if self.current_frame_count % self.frame_skip != 0:
            return frame, []
        
        self.system_stats['processed_frames'] += 1
        frame_start_time = time.time()
        
        # 1. Детекция людей
        detected_boxes = self.detector.detect(frame)
        self.system_stats['detected_persons'] += len(detected_boxes)
        
        if detected_boxes:
            self.add_log(f"Detected {len(detected_boxes)} person(s)")
        
        # 2. Обновление трекера
        tracked_targets = self.tracker.update(detected_boxes)
        
        # 3. Обработка каждой цели
        engagement_decisions = []
        
        for target_id, target_data in tracked_targets.items():
            # Получение bbox цели
            bbox = target_data['bbox']
            x, y, w, h = bbox
            
            # Проверка валидности bbox
            if x < 0 or y < 0 or x + w >= frame.shape[1] or y + h >= frame.shape[0]:
                continue
            
            # Извлечение ROI для классификации
            full_body_roi = frame[y:y+h, x:x+w]
            
            if full_body_roi.size == 0:
                continue
            
            # Получение дополнительных регионов (голова, руки)
            additional_rois = []
            try:
                head_roi = self.body_detector.get_head_region(frame, bbox)
                if head_roi.size > 0:
                    additional_rois.append(head_roi)
                
                arm_rois = self.body_detector.get_arm_regions(frame, bbox)
                additional_rois.extend([roi for roi in arm_rois if roi.size > 0])
            except Exception as e:
                self.add_log(f"Error extracting body parts for target {target_id}: {str(e)}", "WARNING")
            
            # Классификация цвета
            try:
                color_result = self.classifier.classify_with_confidence(
                    full_body_roi, 
                    additional_rois=additional_rois,
                    target_id=target_id
                )
                
                color, confidence = color_result
                
                # Обновление информации о цели
                target_data['color'] = color
                target_data['color_confidence'] = confidence
                
                # Логирование результатов классификации
                if color == "red" and confidence > 0.6:
                    self.system_stats['red_bandana_detections'] += 1
                    self.add_log(f"Target {target_id}: RED BANDANA detected (conf: {confidence:.2f})", "RED_BANDANA")
                elif color != "unknown":
                    self.add_log(f"Target {target_id}: {color} classification (conf: {confidence:.2f})")
                
            except Exception as e:
                self.add_log(f"Classification error for target {target_id}: {str(e)}", "WARNING")
                color, confidence = "unknown", 0.0
                target_data['color'] = color
                target_data['color_confidence'] = confidence
            
            # Оценка безопасности поражения цели
            try:
                safety_report = self.safety_system.evaluate_target_safety(
                    target_id, target_data, (color, confidence)
                )
                
                # Обработка решения системы безопасности
                decision = safety_report['decision']
                reasons = safety_report['reasons']
                
                if decision == EngagementDecision.ENGAGE:
                    self.system_stats['engagement_decisions'] += 1
                    self.system_stats['targets_engaged'] += 1
                    self.add_log(f"TARGET {target_id} ENGAGEMENT AUTHORIZED", "ENGAGE")
                    engagement_decisions.append({
                        'target_id': target_id,
                        'action': 'ENGAGE',
                        'bbox': bbox,
                        'color': color,
                        'confidence': confidence
                    })
                
                elif decision == EngagementDecision.ABORT:
                    if 'RED_BANDANA_DETECTED' in reasons:
                        self.system_stats['false_positives_prevented'] += 1
                        self.add_log(f"Target {target_id}: ENGAGEMENT ABORTED - Friendly detected", "SAFE")
                    else:
                        self.add_log(f"Target {target_id}: ENGAGEMENT ABORTED - {', '.join(reasons)}", "WARNING")
                
                elif decision == EngagementDecision.PREPARE:
                    self.add_log(f"Target {target_id}: PREPARING for engagement", "WARNING")
                    engagement_decisions.append({
                        'target_id': target_id,
                        'action': 'PREPARE',
                        'bbox': bbox,
                        'color': color,
                        'confidence': confidence
                    })
                
                else:  # HOLD_FIRE
                    if reasons:
                        self.add_log(f"Target {target_id}: HOLDING FIRE - {', '.join(reasons)}")
                
                # Подсчет нарушений безопасности
                if safety_report['overall_safety'].value >= ThreatLevel.WARNING.value:
                    self.system_stats['safety_violations'] += 1
                
            except Exception as e:
                self.add_log(f"Safety evaluation error for target {target_id}: {str(e)}", "DANGER")
                self.system_stats['safety_violations'] += 1
        
        # 4. Отрисовка результатов на кадре
        annotated_frame = self.annotate_frame(frame, tracked_targets, engagement_decisions)
        
        # 5. Обновление FPS и интерфейса
        frame_process_time = time.time() - frame_start_time
        fps = self.fps_calculator.update(frame_process_time)
        
        # Обновление прогресс-бара
        progress_percentage = (frame_number / total_frames) * 100
        self.progress.update(
            self.progress_task,
            completed=progress_percentage,
            fps=f"{fps:.1f}",
            targets=str(len(tracked_targets))
        )
        
        return annotated_frame, engagement_decisions
    
    def annotate_frame(self, frame, targets, engagement_decisions):
        """Отрисовка аннотаций на кадре"""
        annotated_frame = frame.copy()
        
        # Создание словаря решений для быстрого поиска
        engagement_dict = {ed['target_id']: ed for ed in engagement_decisions}
        
        for target_id, target_data in targets.items():
            bbox = target_data['bbox']
            x, y, w, h = bbox
            
            color = target_data.get('color', 'unknown')
            confidence = target_data.get('color_confidence', 0.0)
            stability = target_data.get('stability_score', 0.0)
            
            # Определение цвета рамки на основе решения
            if target_id in engagement_dict:
                action = engagement_dict[target_id]['action']
                if action == 'ENGAGE':
                    box_color = (0, 0, 255)  # Красный для поражения
                    thickness = 3
                elif action == 'PREPARE':
                    box_color = (0, 165, 255)  # Оранжевый для подготовки
                    thickness = 2
                else:
                    box_color = (0, 255, 0)  # Зеленый для удержания
                    thickness = 2
            else:
                # Цвет на основе классификации
                if color == "red" and confidence > 0.6:
                    box_color = (255, 0, 0)  # Синий для дружественных
                    thickness = 2
                elif color == "blue":
                    box_color = (0, 255, 255)  # Желтый для потенциальных целей
                    thickness = 2
                else:
                    box_color = (128, 128, 128)  # Серый для неопределенных
                    thickness = 1
            
            # Отрисовка рамки
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, thickness)
            
            # Подготовка текста
            texts = [
                f"ID: {target_id}",
                f"Color: {color} ({confidence:.2f})",
                f"Stability: {stability:.2f}"
            ]
            
            if target_id in engagement_dict:
                texts.append(f"Action: {engagement_dict[target_id]['action']}")
            
            # Отрисовка текста
            text_y = y - 10
            for i, text in enumerate(texts):
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Фон для текста
                cv2.rectangle(annotated_frame, 
                            (x, text_y - text_size[1] - 5), 
                            (x + text_size[0] + 5, text_y + 5), 
                            (0, 0, 0), -1)
                
                # Текст
                cv2.putText(annotated_frame, text, (x + 2, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                text_y -= 25
        
        # Отрисовка общей информации
        info_texts = [
            f"Active Targets: {len(targets)}",
            f"FPS: {self.fps_calculator.get_fps():.1f}",
            f"Frame: {self.system_stats['processed_frames']}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
    
    def process_video(self, video_path):
        """Обработка видео файла"""
        self.add_log(f"Starting video processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.add_log(f"Error opening video: {video_path}", "DANGER")
            return False
        
        # Получение информации о видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.add_log(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Настройка выходного видео
        output_path = get_output_path(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        
        try:
            with Live(self.layout, refresh_per_second=4) as live:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_number += 1
                    
                    # Обработка кадра
                    annotated_frame, engagement_decisions = self.process_frame(
                        frame, frame_number, total_frames
                    )
                    
                    # Запись обработанного кадра
                    out.write(annotated_frame)
                    
                    # Обновление интерфейса
                    if frame_number % 10 == 0:  # Обновляем каждые 10 кадров
                        self.update_interface()
                        live.update(self.layout)
                    
                    # Проверка прерывания (ESC для остановки)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.add_log("Processing interrupted by user", "WARNING")
                        break
        
        except KeyboardInterrupt:
            self.add_log("Processing interrupted by user", "WARNING")
        
        except Exception as e:
            self.add_log(f"Error during processing: {str(e)}", "DANGER")
            return False
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        self.add_log(f"Video processing completed: {output_path}", "SAFE")
        return True
    
    def generate_report(self):
        """Генерация итогового отчета"""
        uptime = time.time() - self.system_stats['system_start_time']
        
        report = f"""
TURRET SYSTEM OPERATION REPORT
{'='*50}

System Statistics:
- Total Uptime: {uptime:.1f} seconds
- Frames Processed: {self.system_stats['processed_frames']}
- Persons Detected: {self.system_stats['detected_persons']}
- Red Bandana Detections: {self.system_stats['red_bandana_detections']}
- Engagement Decisions: {self.system_stats['engagement_decisions']}
- Targets Engaged: {self.system_stats['targets_engaged']}
- False Positives Prevented: {self.system_stats['false_positives_prevented']}
- Safety Violations: {self.system_stats['safety_violations']}

Tracking Statistics:
{self.tracker.get_tracking_statistics()}

Safety Statistics:
{self.safety_system.get_safety_statistics()}

Average Processing Rate: {self.system_stats['processed_frames'] / uptime:.2f} FPS
"""
        
        # Сохранение отчета
        with open('turret_system_report.txt', 'w') as f:
            f.write(report)
        
        self.add_log("System report generated", "SAFE")
        return report


class FPSCalculator:
    """Класс для расчета FPS"""
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.frame_times = []
    
    def update(self, frame_time):
        """Обновление с временем обработки кадра"""
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        
        return 0
    
    def get_fps(self):
        """Получение текущего FPS"""
        if len(self.frame_times) > 1:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0


def main():
    """Основная функция"""
    console.print(Panel.fit(
        "[bold red]🎯 ADVANCED TURRET SYSTEM v2.0[/bold red]\n"
        "[yellow]Initializing combat systems...[/yellow]",
        border_style="red"
    ))
    
    # Создание выходной директории
    create_output_dir()
    
    # Инициализация системы
    try:
        turret_system = TurretSystem()
    except Exception as e:
        console.print(f"[bold red]Error initializing turret system: {e}[/bold red]")
        traceback.print_exc()
        return

    
    # Получение списка видео файлов
    video_files = get_video_files()
    
    if not video_files:
        console.print("[yellow]No video files found in 'test_videos' directory[/yellow]")
        console.print("Please place video files (.mp4, .avi, .mov, .mkv) in the 'test_videos' folder")
        return
    
    console.print(f"[green]Found {len(video_files)} video file(s)[/green]")
    
    # Обработка каждого видео
    for video_file in video_files:
        video_path = os.path.join("test_videos", video_file)
        
        console.print(f"\n[bold blue]Processing: {video_file}[/bold blue]")
        
        success = turret_system.process_video(video_path)
        
        if success:
            console.print(f"[green]✅ Successfully processed: {video_file}[/green]")
        else:
            console.print(f"[red]❌ Failed to process: {video_file}[/red]")
    
    # Генерация итогового отчета
    report = turret_system.generate_report()
    console.print("\n[bold green]FINAL SYSTEM REPORT[/bold green]")
    console.print(report)
    
    console.print("\n[bold yellow]System shutdown complete[/bold yellow]")


if __name__ == "__main__":
    main()
