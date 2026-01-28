"""
Fast processing pipeline using multiprocessing and ONNX.
Bypasses Python GIL for true parallel processing.
"""
import multiprocessing as mp
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import queue

import config


def _detector_process(input_queue, output_queue, model_path, conf, input_size, use_filter=True):
    """Detector process - runs YOLO inference."""
    from fast_detector import FastPersonDetector

    detector = FastPersonDetector(model_path, conf, input_size, use_filter=use_filter)

    while True:
        try:
            item = input_queue.get(timeout=0.5)
            if item is None:  # Poison pill
                break

            frame_id, frame = item
            detections = detector.detect(frame)

            output_queue.put((frame_id, detections, frame))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Detector error: {e}")
            continue


def _weapon_process(input_queue, output_queue, model_path, conf):
    """Weapon detector process."""
    from fast_detector import FastWeaponDetector

    detector = FastWeaponDetector(model_path, conf)

    while True:
        try:
            item = input_queue.get(timeout=0.5)
            if item is None:
                break

            frame_id, frame, bbox = item
            weapons = detector.detect(frame, bbox)

            output_queue.put((frame_id, bbox, weapons))
        except queue.Empty:
            continue
        except Exception:
            continue


class FastPipeline:
    """
    High-performance pipeline using multiprocessing.
    Uses separate processes for detection and analysis to bypass GIL.
    """

    def __init__(self, use_pose=True, use_weapon=True, use_depth=False):
        self.use_pose = use_pose
        self.use_weapon = use_weapon
        self.use_depth = use_depth

        # Queues for inter-process communication
        self.frame_queue = mp.Queue(maxsize=4)
        self.detection_queue = mp.Queue(maxsize=4)
        self.weapon_queue = mp.Queue(maxsize=8)
        self.weapon_result_queue = mp.Queue(maxsize=8)

        # Start detector process
        self.detector_proc = mp.Process(
            target=_detector_process,
            args=(
                self.frame_queue,
                self.detection_queue,
                config.YOLO_MODEL,
                config.YOLO_CONF,
                config.YOLO_IMGSZ,
                True  # use_filter
            ),
            daemon=True
        )

        # Start weapon detector process
        self.weapon_proc = None
        if use_weapon:
            self.weapon_proc = mp.Process(
                target=_weapon_process,
                args=(
                    self.weapon_queue,
                    self.weapon_result_queue,
                    config.WEAPON_MODEL,
                    config.WEAPON_CONF
                ),
                daemon=True
            )

        # Main thread components (lightweight)
        from tracker import Tracker
        from motion_analyzer import MotionAnalyzer
        from threat_scorer import ThreatScorer

        self.tracker = Tracker()
        self.motion_analyzer = MotionAnalyzer()
        self.threat_scorer = ThreatScorer()

        self.pose_analyzer = None
        if use_pose:
            from pose_analyzer import PoseAnalyzer
            self.pose_analyzer = PoseAnalyzer()

        self.depth_estimator = None
        if use_depth:
            try:
                from fast_depth import FastDepthEstimator
                self.depth_estimator = FastDepthEstimator()
            except ImportError:
                from depth_estimator import DepthEstimator
                self.depth_estimator = DepthEstimator()

        # State
        self.running = False
        self.frame_count = 0
        self.latest_targets = {}
        self.pending_weapons = {}  # frame_id -> {bbox -> weapons}
        self.fps_times = []
        self.last_time = time.time()

    def start(self):
        """Start processing workers."""
        self.running = True
        self.detector_proc.start()
        if self.weapon_proc:
            self.weapon_proc.start()

    def stop(self):
        """Stop all workers."""
        self.running = False

        # Send poison pills
        try:
            self.frame_queue.put(None)
            if self.weapon_proc:
                self.weapon_queue.put(None)
        except:
            pass

        # Wait for processes
        self.detector_proc.join(timeout=1)
        if self.weapon_proc:
            self.weapon_proc.join(timeout=1)

        # Terminate if still alive
        if self.detector_proc.is_alive():
            self.detector_proc.terminate()
        if self.weapon_proc and self.weapon_proc.is_alive():
            self.weapon_proc.terminate()

    def process(self, frame):
        """
        Process a frame.

        Args:
            frame: BGR image

        Returns:
            dict of targets with threat levels
        """
        self.frame_count += 1
        t0 = time.time()

        # Submit frame for detection (skip some frames for speed)
        if self.frame_count % config.FRAME_SKIP == 0:
            try:
                self.frame_queue.put_nowait((self.frame_count, frame.copy()))
            except queue.Full:
                pass

        # Collect weapon results
        self._collect_weapon_results()

        # Collect detection results and process
        try:
            while True:
                frame_id, detections, det_frame = self.detection_queue.get_nowait()

                # Update tracker with frame for appearance matching
                targets = self.tracker.update(detections, det_frame)

                # Depth estimation (in main thread, heavy)
                depth_map = None
                if self.use_depth and self.depth_estimator:
                    try:
                        depth_map, _ = self.depth_estimator.estimate(det_frame)
                    except:
                        pass

                # Process each target
                for tid, t in targets.items():
                    bbox = t['bbox']

                    # Submit weapon detection request
                    if self.use_weapon:
                        try:
                            self.weapon_queue.put_nowait((frame_id, det_frame, bbox))
                        except queue.Full:
                            pass

                    # Pose analysis (main thread, uses MediaPipe)
                    pose_result = {'aggressive': 0.0}
                    if self.use_pose and self.pose_analyzer:
                        pose_result = self.pose_analyzer.analyze(det_frame, bbox)

                    # Motion analysis
                    self.motion_analyzer.update(tid, bbox, det_frame.shape)
                    motion_result = self.motion_analyzer.analyze(tid, det_frame.shape)

                    # Get cached weapon results
                    weapon_result = self._get_weapon_result(frame_id, bbox)
                    if weapon_result:
                        t['weapons'] = weapon_result

                    # Depth value
                    depth_value = None
                    if depth_map is not None:
                        depth_value = self.depth_estimator.get_relative_depth(
                            depth_map, bbox, det_frame.shape
                        )
                        t['depth'] = depth_value

                    # Calculate threat
                    threat = self.threat_scorer.calculate(
                        pose_result, motion_result, bbox, det_frame.shape,
                        weapon_result=weapon_result,
                        depth_value=depth_value
                    )
                    t['threat'] = threat

                self.latest_targets = targets

                # Cleanup
                self.motion_analyzer.cleanup(set(targets.keys()))
                self._cleanup_pending_weapons(frame_id)

                # FPS calculation
                now = time.time()
                dt = now - self.last_time
                self.last_time = now
                if dt > 0.001:
                    self.fps_times.append(dt)
                    if len(self.fps_times) > 30:
                        self.fps_times.pop(0)

        except queue.Empty:
            pass

        return self.latest_targets.copy()

    def _collect_weapon_results(self):
        """Collect pending weapon detection results."""
        try:
            while True:
                frame_id, bbox, weapons = self.weapon_result_queue.get_nowait()
                if frame_id not in self.pending_weapons:
                    self.pending_weapons[frame_id] = {}
                self.pending_weapons[frame_id][bbox] = weapons
        except queue.Empty:
            pass

    def _get_weapon_result(self, frame_id, bbox):
        """Get weapon result for a specific frame/bbox."""
        if frame_id in self.pending_weapons:
            # Try exact match first
            if bbox in self.pending_weapons[frame_id]:
                return self.pending_weapons[frame_id][bbox]
            # Try close match (bbox might differ slightly)
            for stored_bbox, weapons in self.pending_weapons[frame_id].items():
                if self._bbox_similar(bbox, stored_bbox):
                    return weapons
        return []

    def _bbox_similar(self, b1, b2, threshold=50):
        """Check if two bboxes are similar."""
        return (abs(b1[0] - b2[0]) < threshold and
                abs(b1[1] - b2[1]) < threshold)

    def _cleanup_pending_weapons(self, current_frame_id):
        """Remove old weapon results."""
        old_frames = [fid for fid in self.pending_weapons if fid < current_frame_id - 10]
        for fid in old_frames:
            del self.pending_weapons[fid]

    def get_fps(self):
        """Get current FPS."""
        if len(self.fps_times) < 2:
            return 0
        avg = sum(self.fps_times) / len(self.fps_times)
        return 1.0 / avg if avg > 0 else 0
