"""
Fast depth estimation using lightweight methods.
Much faster than MiDaS for real-time applications.
"""
import cv2
import numpy as np


class FastDepthEstimator:
    """
    Fast depth estimation using monocular cues.
    Uses combination of:
    - Object size (larger = closer)
    - Vertical position (lower in frame = closer)
    - Focus/blur analysis (sharper = closer, for some cameras)
    """

    def __init__(self):
        # Reference person height in pixels at 2m distance (for 480p)
        self.ref_height = 200
        self.ref_distance = 2.0  # meters

    def estimate(self, frame):
        """
        Estimate depth map from frame.

        Returns:
            depth_map: Normalized depth (0-255, higher = closer)
            depth_colored: Colorized visualization
        """
        h, w = frame.shape[:2]

        # Create depth map based on vertical position
        # Lower in frame = closer (typical ground-level camera)
        y_coords = np.arange(h).reshape(-1, 1)
        depth_map = (y_coords / h * 255).astype(np.uint8)
        depth_map = np.tile(depth_map, (1, w))

        # Add edge-based depth cues
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Blur edges to create gradient
        edge_depth = cv2.GaussianBlur(edges, (21, 21), 0)

        # Combine with position-based depth
        depth_map = cv2.addWeighted(depth_map, 0.7, edge_depth, 0.3, 0)

        # Colorize
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        return depth_map, depth_colored

    def estimate_from_bbox(self, bbox, frame_shape):
        """
        Estimate relative depth from bounding box.
        Uses size and position as depth cues.

        Args:
            bbox: (x, y, w, h)
            frame_shape: (height, width, channels)

        Returns:
            depth: 0-1 (1 = closest)
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        # Size-based depth (larger = closer)
        area = w * h
        frame_area = frame_w * frame_h
        size_ratio = area / frame_area
        size_depth = min(1.0, size_ratio * 15)  # Scale factor

        # Position-based depth (lower in frame = closer)
        center_y = y + h / 2
        position_depth = center_y / frame_h

        # Combine (weighted average)
        depth = size_depth * 0.6 + position_depth * 0.4

        return min(1.0, depth)

    def get_depth_at_bbox(self, depth_map, bbox):
        """Get average depth value within bbox."""
        x, y, w, h = bbox
        roi = depth_map[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0
        return float(np.mean(roi))

    def get_relative_depth(self, depth_map, bbox, frame_shape):
        """Get relative depth (0-1) where 1 is closest."""
        # Use bbox-based estimation (faster and more accurate for persons)
        return self.estimate_from_bbox(bbox, frame_shape)

    def estimate_distance_meters(self, bbox, frame_shape, focal_length=None):
        """
        Estimate approximate distance in meters.
        Uses pinhole camera model with assumed person height.

        Args:
            bbox: Person bounding box
            frame_shape: Frame dimensions
            focal_length: Camera focal length (pixels). If None, estimated from frame.

        Returns:
            distance in meters (approximate)
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        # Estimate focal length if not provided
        if focal_length is None:
            # Assume ~60 degree FOV
            focal_length = frame_w / (2 * np.tan(np.radians(30)))

        # Assume average person height ~1.7m
        real_height = 1.7

        # Distance = (focal_length * real_height) / pixel_height
        if h > 0:
            distance = (focal_length * real_height) / h
            return min(100.0, max(0.5, distance))

        return 10.0  # Default
