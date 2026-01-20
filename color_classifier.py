import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import colorsys

class ImprovedColorClassifier:
    def __init__(self):
        self.color_history = defaultdict(list)
        self.history_length = 10
        self.confidence_threshold = 0.6
        
        # Адаптивные цветовые диапазоны - расширенные для лучшего обнаружения красных повязок
        self.red_ranges = [
            # Основной красный
            {'lower': np.array([0, 100, 50]), 'upper': np.array([15, 255, 255])},
            # Темно-красный
            {'lower': np.array([165, 100, 50]), 'upper': np.array([180, 255, 255])}
        ]
        
        self.blue_ranges = [
            # Основной синий
            {'lower': np.array([100, 150, 50]), 'upper': np.array([130, 255, 255])},
            # Темно-синий
            {'lower': np.array([110, 100, 50]), 'upper': np.array([140, 255, 255])}
        ]
    
    def classify_with_confidence(self, full_body_roi, additional_rois=None, target_id=None):
        """
        Классификация с учетом уверенности и временной стабилизации
        """
        if additional_rois is None:
            additional_rois = []
        
        # Обработка всех регионов
        all_regions = [full_body_roi] + additional_rois
        region_results = []
        
        for roi in all_regions:
            if roi.size == 0:
                continue
            
            # Предварительная обработка
            processed_roi = self._preprocess_roi(roi)
            
            # Извлечение доминантных цветов
            dominant_colors = self._extract_dominant_colors(processed_roi)
            
            # Классификация каждого доминантного цвета
            color_scores = self._classify_dominant_colors(dominant_colors)
            region_results.append(color_scores)
        
        # Агрегация результатов по всем регионам
        final_scores = self._aggregate_region_results(region_results)
        
        # Временная стабилизация
        if target_id is not None:
            final_scores = self._apply_temporal_smoothing(final_scores, target_id)
        
        # Принятие решения
        return self._make_final_decision(final_scores)
    
    def _preprocess_roi(self, roi):
        """Улучшенная предобработка изображения"""
        # Увеличение контраста
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Конвертация в HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def _extract_dominant_colors(self, hsv_roi, n_colors=5):
        """Извлечение доминантных цветов через кластеризацию"""
        # Изменение формы для кластеризации
        pixels = hsv_roi.reshape((-1, 3))
        
        # Фильтрация слишком темных/светлых пикселей
        mask = (pixels[:, 2] > 30) & (pixels[:, 2] < 240)  # V канал
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 10:
            return []
        
        # Кластеризация
        kmeans = KMeans(n_clusters=min(n_colors, len(filtered_pixels)), 
                       random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Получение центров кластеров и их весов
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Вычисление весов каждого цвета
        color_weights = []
        for i in range(len(colors)):
            weight = np.sum(labels == i) / len(labels)
            color_weights.append((colors[i], weight))
        
        # Сортировка по весу
        color_weights.sort(key=lambda x: x[1], reverse=True)
        
        return color_weights
    
    def _classify_dominant_colors(self, dominant_colors):
        """Классификация доминантных цветов"""
        red_score = 0
        blue_score = 0
        total_weight = 0
        
        for color_hsv, weight in dominant_colors:
            total_weight += weight
            
            # Проверка на красный
            red_confidence = self._calculate_red_confidence(color_hsv)
            red_score += red_confidence * weight
            
            # Проверка на синий
            blue_confidence = self._calculate_blue_confidence(color_hsv)
            blue_score += blue_confidence * weight
        
        if total_weight > 0:
            red_score /= total_weight
            blue_score /= total_weight
        
        return {'red': red_score, 'blue': blue_score}
    
    def _calculate_red_confidence(self, hsv_color):
        """Вычисление уверенности в красном цвете"""
        h, s, v = hsv_color
        confidence = 0
        
        for red_range in self.red_ranges:
            # Проверка попадания в диапазон
            if (red_range['lower'][0] <= h <= red_range['upper'][0] and 
                red_range['lower'][1] <= s <= red_range['upper'][1] and 
                red_range['lower'][2] <= v <= red_range['upper'][2]):
                
                # Вычисление уверенности на основе близости к центру диапазона
                h_center = (red_range['lower'][0] + red_range['upper'][0]) / 2
                s_center = (red_range['lower'][1] + red_range['upper'][1]) / 2
                v_center = (red_range['lower'][2] + red_range['upper'][2]) / 2
                
                h_dist = min(abs(h - h_center), 180 - abs(h - h_center))  # Циклическое расстояние для H
                s_dist = abs(s - s_center)
                v_dist = abs(v - v_center)
                
                # Нормализация расстояний
                h_norm = 1 - (h_dist / 90)
                s_norm = 1 - (s_dist / 127.5)
                v_norm = 1 - (v_dist / 127.5)
                
                range_confidence = (h_norm + s_norm + v_norm) / 3
                confidence = max(confidence, range_confidence)
        
        return confidence
    
    def _calculate_blue_confidence(self, hsv_color):
        """Вычисление уверенности в синем цвете"""
        h, s, v = hsv_color
        confidence = 0
        
        for blue_range in self.blue_ranges:
            if (blue_range['lower'][0] <= h <= blue_range['upper'][0] and 
                blue_range['lower'][1] <= s <= blue_range['upper'][1] and 
                blue_range['lower'][2] <= v <= blue_range['upper'][2]):
                
                # Аналогично красному
                h_center = (blue_range['lower'][0] + blue_range['upper'][0]) / 2
                s_center = (blue_range['lower'][1] + blue_range['upper'][1]) / 2
                v_center = (blue_range['lower'][2] + blue_range['upper'][2]) / 2
                
                h_dist = abs(h - h_center)
                s_dist = abs(s - s_center)
                v_dist = abs(v - v_center)
                
                h_norm = 1 - (h_dist / 15)  # Синий имеет более узкий диапазон H
                s_norm = 1 - (s_dist / 52.5)
                v_norm = 1 - (v_dist / 102.5)
                
                range_confidence = (h_norm + s_norm + v_norm) / 3
                confidence = max(confidence, range_confidence)
        
        return confidence
    
    def _aggregate_region_results(self, region_results):
        """Агрегация результатов по всем регионам"""
        if not region_results:
            return {'red': 0, 'blue': 0}
        
        # Взвешенное усреднение (голова важнее тела)
        weights = [0.4] + [0.6 / (len(region_results) - 1)] * (len(region_results) - 1)
        
        final_red = 0
        final_blue = 0
        
        for i, result in enumerate(region_results):
            weight = weights[i] if i < len(weights) else weights[-1]
            final_red += result['red'] * weight
            final_blue += result['blue'] * weight
        
        return {'red': final_red, 'blue': final_blue}
    
    def _apply_temporal_smoothing(self, current_scores, target_id):
        """Применение временной стабилизации"""
        self.color_history[target_id].append(current_scores)
        
        # Ограничение размера истории
        if len(self.color_history[target_id]) > self.history_length:
            self.color_history[target_id].pop(0)
        
        # Взвешенное усреднение с убывающими весами
        history = self.color_history[target_id]
        weights = np.exp(np.linspace(-2, 0, len(history)))  # Экспоненциальные веса
        weights /= np.sum(weights)
        
        smoothed_red = 0
        smoothed_blue = 0
        
        for i, scores in enumerate(history):
            smoothed_red += scores['red'] * weights[i]
            smoothed_blue += scores['blue'] * weights[i]
        
        return {'red': smoothed_red, 'blue': smoothed_blue}
    
    def _make_final_decision(self, scores):
        """Принятие финального решения с учетом уверенности"""
        red_score = scores['red']
        blue_score = scores['blue']
        
        # Требуем минимальную уверенность
        if red_score < self.confidence_threshold and blue_score < self.confidence_threshold:
            return "unknown", max(red_score, blue_score)
        
        # Требуем значительную разницу между цветами
        if abs(red_score - blue_score) < 0.2:
            return "unknown", max(red_score, blue_score)
        
        if red_score > blue_score:
            return "red", red_score
        else:
            return "blue", blue_score
    
    def adapt_thresholds(self, feedback_data):
        """Адаптация порогов на основе обратной связи"""
        # Анализ ошибок и корректировка порогов
        for ground_truth, prediction, confidence in feedback_data:
            if ground_truth == "red" and prediction != "red" and confidence > 0.3:
                # Уменьшаем порог для красного
                self.confidence_threshold *= 0.95
            elif ground_truth != "red" and prediction == "red" and confidence > 0.7:
                # Увеличиваем порог для красного
                self.confidence_threshold *= 1.05
        
        # Ограничиваем диапазон порогов
        self.confidence_threshold = np.clip(self.confidence_threshold, 0.3, 0.8)
