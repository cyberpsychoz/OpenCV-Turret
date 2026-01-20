import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import warnings
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*NVIDIA.*')

import cv2
import time
import logging
import sys
from detector import PersonDetector
from head_detector import BodyPartDetector
from color_classifier import ImprovedColorClassifier
from tracker import ImprovedTargetTracker
from safety_system import SafetySystem, EngagementDecision, ThreatLevel
from utils import get_video_files, create_output_dir, get_output_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('turret_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TurretSystem:
    def __init__(self, use_pose=True):
        logger.info("Initializing turret system...")
        self.detector = PersonDetector()
        self.classifier = ImprovedColorClassifier()
        self.use_pose = use_pose
        if use_pose:
            self.body_detector = BodyPartDetector()
        else:
            self.body_detector = None
        self.tracker = ImprovedTargetTracker()
        self.safety_system = SafetySystem()

        self.stats = {
            'frames': 0,
            'persons': 0,
            'red_bandanas': 0,
            'engagements': 0,
            'start_time': time.time()
        }

        self.frame_skip = 3
        self.frame_count = 0
        self.last_logged_states = {}
        self.fps_times = []

        logger.info("Turret system ready")

    def process_frame(self, frame, frame_num, total_frames):
        self.frame_count += 1

        if self.frame_count % self.frame_skip != 0:
            return frame, []

        self.stats['frames'] += 1
        t0 = time.time()

        boxes = self.detector.detect(frame)
        self.stats['persons'] += len(boxes)

        targets = self.tracker.update(boxes)
        decisions = []

        for tid, tdata in targets.items():
            bbox = tdata['bbox']
            x, y, w, h = bbox

            if x < 0 or y < 0 or x + w >= frame.shape[1] or y + h >= frame.shape[0]:
                continue

            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            add_rois = []
            if self.use_pose and self.body_detector:
                try:
                    head, arms = self.body_detector.get_all_regions(frame, bbox, frame_num)
                    if head is not None and head.size > 0:
                        add_rois.append(head)
                    add_rois.extend(arms)
                except:
                    pass

            try:
                color, conf = self.classifier.classify_with_confidence(roi, add_rois, tid)
                tdata['color'] = color
                tdata['color_confidence'] = conf

                if color == "red" and conf > 0.6:
                    self.stats['red_bandanas'] += 1
            except:
                color, conf = "unknown", 0.0
                tdata['color'] = color
                tdata['color_confidence'] = conf

            try:
                report = self.safety_system.evaluate_target_safety(tid, tdata, (color, conf))
                decision = report['decision']

                if decision == EngagementDecision.ENGAGE:
                    self.stats['engagements'] += 1
                    decisions.append({'id': tid, 'action': 'ENGAGE', 'bbox': bbox})
                    prev = self.last_logged_states.get(f'd_{tid}')
                    if prev != 'ENGAGE':
                        logger.warning(f"TARGET {tid} ENGAGEMENT AUTHORIZED")
                        self.last_logged_states[f'd_{tid}'] = 'ENGAGE'

                elif decision == EngagementDecision.ABORT and 'RED_BANDANA_DETECTED' in report['reasons']:
                    prev = self.last_logged_states.get(f'd_{tid}')
                    if prev != 'ABORT':
                        logger.info(f"Target {tid}: ABORT - Friendly (red bandana)")
                        self.last_logged_states[f'd_{tid}'] = 'ABORT'

                elif decision == EngagementDecision.PREPARE:
                    decisions.append({'id': tid, 'action': 'PREPARE', 'bbox': bbox})
            except:
                pass

        if self.frame_count % 100 == 0:
            active = set(targets.keys())
            self.classifier.cleanup_dead_targets(active)
            self.safety_system.cleanup_dead_targets(active)

        self._annotate(frame, targets, decisions)

        dt = time.time() - t0
        self.fps_times.append(dt)
        if len(self.fps_times) > 30:
            self.fps_times.pop(0)

        return frame, decisions

    def _annotate(self, frame, targets, decisions):
        dec_map = {d['id']: d['action'] for d in decisions}

        for tid, t in targets.items():
            x, y, w, h = t['bbox']
            color = t.get('color', '?')
            conf = t.get('color_confidence', 0)

            if tid in dec_map:
                if dec_map[tid] == 'ENGAGE':
                    c = (0, 0, 255)
                else:
                    c = (0, 165, 255)
            elif color == 'red' and conf > 0.6:
                c = (255, 0, 0)
            else:
                c = (128, 128, 128)

            cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)
            cv2.putText(frame, f"{tid}:{color[0]}({conf:.1f})", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        fps = self._get_fps()
        cv2.putText(frame, f"FPS:{fps:.0f} T:{len(targets)}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _get_fps(self):
        if len(self.fps_times) < 2:
            return 0
        return 1.0 / (sum(self.fps_times) / len(self.fps_times))

    def process_video(self, path):
        logger.info(f"Processing: {path}")
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            logger.error(f"Cannot open: {path}")
            return False

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

        out_path = get_output_path(path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_num = 0
        last_log = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                processed, _ = self.process_frame(frame, frame_num, total)
                out.write(processed)

                if time.time() - last_log > 5:
                    pct = frame_num / total * 100
                    fps = self._get_fps()
                    logger.info(f"Progress: {pct:.1f}% ({frame_num}/{total}) FPS:{fps:.1f}")
                    last_log = time.time()

                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("Interrupted by user")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            cap.release()
            out.release()

        logger.info(f"Output: {out_path}")
        return True

    def report(self):
        elapsed = time.time() - self.stats['start_time']
        avg_fps = self.stats['frames'] / elapsed if elapsed > 0 else 0

        txt = f"""
=== TURRET SYSTEM REPORT ===
Runtime: {elapsed:.1f}s
Frames processed: {self.stats['frames']}
Average FPS: {avg_fps:.1f}
Persons detected: {self.stats['persons']}
Red bandanas: {self.stats['red_bandanas']}
Engagements: {self.stats['engagements']}
Tracking: {self.tracker.get_tracking_statistics()}
"""
        logger.info(txt)

        with open('turret_system_report.txt', 'w') as f:
            f.write(txt)

        return txt


def main():
    logger.info("=== TURRET SYSTEM v2.0 (Lightweight) ===")
    create_output_dir()

    try:
        system = TurretSystem()
    except Exception as e:
        logger.error(f"Init failed: {e}")
        return

    videos = get_video_files()
    if not videos:
        logger.warning("No videos in test_videos/")
        return

    logger.info(f"Found {len(videos)} video(s)")

    for v in videos:
        path = os.path.join("test_videos", v)
        system.process_video(path)

    system.report()
    logger.info("Done")


if __name__ == "__main__":
    main()
