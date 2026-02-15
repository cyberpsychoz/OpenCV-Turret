import cv2
import numpy as np

class DepthEstimator:
    """MiDaS-based monocular depth estimation."""

    def __init__(self, model_type="MiDaS_small"):
        """
        Initialize depth estimator.

        Args:
            model_type: "MiDaS_small" (fast) or "DPT_Large" (accurate)
        """
        import torch

        self.device = torch.device('cpu')
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.torch = torch

    def estimate(self, frame):
        """
        Estimate depth map from RGB frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            depth_map: Normalized depth map (0-255, uint8)
            depth_colored: Colorized depth map for visualization
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img_rgb).to(self.device)

        with self.torch.no_grad():
            prediction = self.model(input_batch)
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_normalized.astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        return depth_map, depth_colored

    def get_depth_at_bbox(self, depth_map, bbox):
        """
        Get average depth value within a bounding box.

        Args:
            depth_map: Depth map from estimate()
            bbox: (x, y, w, h) bounding box

        Returns:
            average_depth: Float 0-255 (higher = closer to camera)
        """
        x, y, w, h = bbox
        roi = depth_map[y:y+h, x:x+w]

        if roi.size == 0:
            return 0.0

        return float(np.mean(roi))

    def get_relative_depth(self, depth_map, bbox, frame_shape):
        """
        Get relative depth (0-1) where 1 is closest.

        Args:
            depth_map: Depth map from estimate()
            bbox: (x, y, w, h) bounding box
            frame_shape: Original frame shape

        Returns:
            relative_depth: Float 0-1
        """
        avg_depth = self.get_depth_at_bbox(depth_map, bbox)
        return avg_depth / 255.0
