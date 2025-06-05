import cv2
import numpy as np

class ColorClassifier:
    def classify(self, roi):
        if roi.size == 0:
            return "unknown"
            
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        hsv = cv2.GaussianBlur(hsv, (5,5), 0)

        red_mask = self._create_red_mask(hsv)
        blue_mask = self._create_blue_mask(hsv)

        red_ratio = self._calculate_color_ratio(red_mask)
        blue_ratio = self._calculate_color_ratio(blue_mask)

        if red_ratio > blue_ratio and red_ratio > 0.1:
            return "red"
        elif blue_ratio > 0.1:
            return "blue"
        return "unknown"

    def _create_red_mask(self, hsv):
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        return cv2.bitwise_or(mask1, mask2)

    def _create_blue_mask(self, hsv):
        return cv2.inRange(hsv, (100, 150, 50), (140, 255, 255))

    def _calculate_color_ratio(self, mask):
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum(cv2.contourArea(cnt) for cnt in contours)
        return total_area / (mask.shape[0] * mask.shape[1])