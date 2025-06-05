# detector.py
import cv2
import os

class PersonDetector:
    def __init__(self):
        # Принудительно используем CPU на Windows
        self.use_cuda = False  # Отключаем CUDA на Windows
        print("[INFO] Используем CPU (HOG) для детекции людей")
        
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        if self.use_cuda:
            print("[INFO] Используем CUDA для детекции людей")
            self.net = cv2.dnn.readNet("yolov8s.onnx")  # Загрузите свою модель YOLO
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.confidence_threshold = 0.5
            self.person_class_id = 0
        else:
            print("[INFO] Используем CPU (HOG) для детекции людей")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        if self.use_cuda:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            boxes = []
            for detection in detections:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]
                
                if class_id == self.person_class_id and confidence > self.confidence_threshold:
                    x, y, w, h = (detection[:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(int)
                    boxes.append((x - w//2, y - h//2, w, h))
            return boxes
        else:
            # классик HOG
            boxes, _ = self.hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)
            return boxes