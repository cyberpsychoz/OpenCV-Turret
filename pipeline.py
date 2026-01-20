import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import config
from detector import Detector
from pose_analyzer import PoseAnalyzer
from motion_analyzer import MotionAnalyzer
from threat_scorer import ThreatScorer
from tracker import Tracker


@dataclass
class FrameData:
    frame_id: int
    frame: Any
    timestamp: float


@dataclass
class DetectionResult:
    frame_id: int
    detections: List[Dict]


@dataclass
class AnalysisResult:
    frame_id: int
    targets: Dict


class ThreadedPipeline:
    def __init__(self, use_pose=True, queue_size=4):
        self.use_pose = use_pose
        self.running = False

        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.detection_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)

        self.detector = Detector()
        self.tracker = Tracker()
        self.pose_analyzer = PoseAnalyzer() if use_pose else None
        self.motion_analyzer = MotionAnalyzer()
        self.threat_scorer = ThreatScorer()

        self.detector_thread = None
        self.analyzer_thread = None

        self.latest_targets = {}
        self.frame_count = 0
        self.process_times = []
        self.last_process_time = time.time()

        self._lock = threading.Lock()

    def start(self):
        self.running = True

        self.detector_thread = threading.Thread(target=self._detector_loop, daemon=True)
        self.analyzer_thread = threading.Thread(target=self._analyzer_loop, daemon=True)

        self.detector_thread.start()
        self.analyzer_thread.start()

    def stop(self):
        self.running = False

        for q in [self.frame_queue, self.detection_queue]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

        if self.detector_thread:
            self.detector_thread.join(timeout=1)
        if self.analyzer_thread:
            self.analyzer_thread.join(timeout=1)

    def process(self, frame):
        self.frame_count += 1
        t0 = time.time()

        if self.frame_count % config.FRAME_SKIP == 0:
            try:
                self.frame_queue.put_nowait(FrameData(
                    frame_id=self.frame_count,
                    frame=frame.copy(),
                    timestamp=t0
                ))
            except queue.Full:
                pass

        try:
            while True:
                result = self.result_queue.get_nowait()
                with self._lock:
                    self.latest_targets = result.targets

                    now = time.time()
                    dt = now - self.last_process_time
                    self.last_process_time = now
                    if dt > 0.001:
                        self.process_times.append(dt)
                        if len(self.process_times) > 30:
                            self.process_times.pop(0)
        except queue.Empty:
            pass

        with self._lock:
            return self.latest_targets.copy()

    def _detector_loop(self):
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            detections = self.detector.detect(frame_data.frame)

            try:
                self.detection_queue.put_nowait(DetectionResult(
                    frame_id=frame_data.frame_id,
                    detections=detections
                ))
            except queue.Full:
                pass

            self.detection_queue.frame_data = frame_data

    def _analyzer_loop(self):
        frame_cache = {}

        while self.running:
            try:
                det_result = self.detection_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_data = getattr(self.detection_queue, 'frame_data', None)
            if frame_data:
                frame_cache[det_result.frame_id] = frame_data.frame

            targets = self.tracker.update(det_result.detections)

            frame = frame_cache.get(det_result.frame_id)
            if frame is not None:
                for tid, t in targets.items():
                    bbox = t['bbox']

                    pose_result = {'aggressive': 0.0}
                    if self.use_pose and self.pose_analyzer and frame is not None:
                        pose_result = self.pose_analyzer.analyze(frame, bbox)

                    self.motion_analyzer.update(tid, bbox, frame.shape if frame is not None else (480, 640, 3))
                    motion_result = self.motion_analyzer.analyze(tid, frame.shape if frame is not None else (480, 640, 3))

                    threat = self.threat_scorer.calculate(
                        pose_result, motion_result, bbox,
                        frame.shape if frame is not None else (480, 640, 3)
                    )
                    t['threat'] = threat

            for fid in list(frame_cache.keys()):
                if fid < det_result.frame_id - 5:
                    del frame_cache[fid]

            try:
                self.result_queue.put_nowait(AnalysisResult(
                    frame_id=det_result.frame_id,
                    targets=targets.copy()
                ))
            except queue.Full:
                pass

            active = set(targets.keys())
            self.motion_analyzer.cleanup(active)

    def get_fps(self):
        with self._lock:
            if len(self.process_times) < 2:
                return 0
            avg = sum(self.process_times) / len(self.process_times)
            return 1.0 / avg if avg > 0 else 0
