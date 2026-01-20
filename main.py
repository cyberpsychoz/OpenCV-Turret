import cv2
import time
import argparse
import logging

import config
from pipeline import ThreadedPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)


class TurretSystem:
    def __init__(self, use_pose=True, use_weapon=True, threaded=True):
        log.info("Initializing...")
        self.threaded = threaded

        if threaded:
            self.pipeline = ThreadedPipeline(use_pose=use_pose, use_weapon=use_weapon)
            self.pipeline.start()
        else:
            from detector import Detector
            from pose_analyzer import PoseAnalyzer
            from motion_analyzer import MotionAnalyzer
            from threat_scorer import ThreatScorer
            from tracker import Tracker

            self.detector = Detector()
            self.tracker = Tracker()
            self.pose_analyzer = PoseAnalyzer() if use_pose else None
            self.motion_analyzer = MotionAnalyzer()
            self.threat_scorer = ThreatScorer()
            self.weapon_detector = None
            if use_weapon:
                try:
                    from weapon_detector import WeaponDetector
                    self.weapon_detector = WeaponDetector()
                except Exception:
                    pass
            self.frame_count = 0
            self.fps_times = []

        self.use_pose = use_pose
        self.use_weapon = use_weapon
        log.info(f"Ready (threaded={threaded}, pose={use_pose}, weapon={use_weapon})")

    def process(self, frame):
        if self.threaded:
            return self.pipeline.process(frame)

        self.frame_count += 1
        if self.frame_count % config.FRAME_SKIP != 0:
            return getattr(self, '_last_targets', {})

        t0 = time.time()

        detections = self.detector.detect(frame)
        targets = self.tracker.update(detections)

        for tid, t in targets.items():
            bbox = t['bbox']

            pose_result = {'aggressive': 0.0}
            if self.use_pose and self.pose_analyzer:
                pose_result = self.pose_analyzer.analyze(frame, bbox)

            weapon_result = []
            if self.use_weapon and self.weapon_detector:
                weapon_result = self.weapon_detector.detect(frame, bbox)
                if weapon_result:
                    t['weapons'] = weapon_result

            self.motion_analyzer.update(tid, bbox, frame.shape)
            motion_result = self.motion_analyzer.analyze(tid, frame.shape)

            threat = self.threat_scorer.calculate(
                pose_result, motion_result, bbox, frame.shape,
                weapon_result=weapon_result
            )
            t['threat'] = threat

        self._last_targets = targets

        dt = time.time() - t0
        self.fps_times.append(dt)
        if len(self.fps_times) > 30:
            self.fps_times.pop(0)

        return targets

    def get_fps(self):
        if self.threaded:
            return self.pipeline.get_fps()
        if len(self.fps_times) < 2:
            return 0
        return 1.0 / (sum(self.fps_times) / len(self.fps_times))

    def stop(self):
        if self.threaded:
            self.pipeline.stop()

    def draw(self, frame, targets):
        for tid, t in targets.items():
            x, y, w, h = t['bbox']
            threat = t.get('threat', {'total': 0, 'level': 'LOW', 'components': {}})
            level = threat['level']

            if level == 'HIGH':
                color = (0, 0, 255)
                thick = 3
            elif level == 'MEDIUM':
                color = (0, 165, 255)
                thick = 2
            else:
                color = (0, 255, 0)
                thick = 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)

            weapons = t.get('weapons', [])
            for wp in weapons:
                wx, wy, ww, wh = wp['bbox']
                cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), (0, 0, 255), 2)
                wlabel = f"{wp['class']} {wp['conf']:.2f}"
                cv2.putText(frame, wlabel, (wx, wy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            label = f"{tid} {level} {threat['total']:.2f}"
            if weapons:
                label += " [ARMED]"
            cv2.rectangle(frame, (x, y - 20), (x + len(label) * 10, y), (0, 0, 0), -1)
            cv2.putText(frame, label, (x + 2, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        fps = self.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f} | Targets: {len(targets)}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame


def run_webcam(system, camera_id=0):
    log.info(f"Starting webcam {camera_id}")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        log.error("Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    log.info("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            targets = system.process(frame)
            frame = system.draw(frame, targets)

            cv2.imshow("Turret System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        system.stop()
        cap.release()
        cv2.destroyAllWindows()


def run_video(system, video_path, output_path=None):
    log.info(f"Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        log.error(f"Cannot open: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.info(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_num = 0
    last_log = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            targets = system.process(frame)
            frame = system.draw(frame, targets)

            if out:
                out.write(frame)

            cv2.imshow("Turret System", frame)

            if time.time() - last_log > 3:
                pct = frame_num / total * 100
                log.info(f"Progress: {pct:.1f}% FPS: {system.get_fps():.1f}")
                last_log = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        system.stop()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    log.info("Done")


def main():
    parser = argparse.ArgumentParser(description="Turret Threat Detection System")
    parser.add_argument('--mode', choices=['webcam', 'video'], default='webcam')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-pose', action='store_true', help='Disable pose analysis')
    parser.add_argument('--no-weapon', action='store_true', help='Disable weapon detection')
    parser.add_argument('--no-thread', action='store_true', help='Disable threading')

    args = parser.parse_args()

    system = TurretSystem(
        use_pose=not args.no_pose,
        use_weapon=not args.no_weapon,
        threaded=not args.no_thread
    )

    if args.mode == 'webcam':
        run_webcam(system, args.camera)
    else:
        if not args.video:
            log.error("--video required for video mode")
            return
        run_video(system, args.video, args.output)


if __name__ == "__main__":
    main()
