import cv2

class ColorClassifier:
    def classify(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        red_lower1 = (0, 100, 100)
        red_upper1 = (10, 255, 255)
        red_lower2 = (160, 100, 100)
        red_upper2 = (180, 255, 255)
        
        blue_lower = (100, 150, 50)
        blue_upper = (140, 255, 255)

        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        red_ratio = cv2.countNonZero(red_mask) / (roi.shape[0] * roi.shape[1])
        blue_ratio = cv2.countNonZero(blue_mask) / (roi.shape[0] * roi.shape[1])

        if red_ratio > blue_ratio and red_ratio > 0.1:
            return "red"
        elif blue_ratio > 0.1:
            return "blue"
        return "unknown"