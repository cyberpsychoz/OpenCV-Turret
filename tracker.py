import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import defaultdict

class ImprovedTargetTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.targets = {}
        self.disappeared = defaultdict(int)
        self.next_id = 0
        self.fire_delay = 3.0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Статистика для адаптации
        self.tracking_stats = {
            'total_detections': 0,
            'successful_matches': 0,
            'lost_targets': 0,
            'false_positives': 0
        }
    
    def create_kalman_filter(self, initial_bbox):
        """Создание Kalman фильтра для трекинга"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # Матрица состояния: [x, y, w, h, vx, vy, vw, vh]
        kf.x = np.array([
            initial_bbox[0], initial_bbox[1], 
            initial_bbox[2], initial_bbox[3],
            0, 0, 0, 0
        ]).reshape((8, 1))
        
        # Матрица перехода состояния (модель постоянной скорости)
        dt = 1.0
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Матрица наблюдения
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Ковариационная матрица шума процесса
        kf.Q = np.eye(8) * 0.1
        kf.Q[4:, 4:] *= 0.01  # Меньший шум для скоростей
        
        # Ковариационная матрица шума измерений
        kf.R = np.eye(4) * 1.0
        
        # Начальная ковариационная матрица
        kf.P *= 10.0
        
        return kf
    
    def predict_positions(self):
        """Предсказание следующих позиций всех целей"""
        predictions = {}
        for target_id, target in self.targets.items():
            kf = target['kalman_filter']
            kf.predict()
            # Извлекаем предсказанную позицию
            pred_x, pred_y, pred_w, pred_h = kf.x[:4].flatten()
            predictions[target_id] = (int(pred_x), int(pred_y), int(pred_w), int(pred_h))
        return predictions
    
    def calculate_iou(self, box1, box2):
        """Вычисление IoU (Intersection over Union)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Координаты пересечения
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        # Площади
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def calculate_distance_cost(self, predictions, detections):
        """Вычисление матрицы стоимости для сопоставления"""
        if not predictions or not detections:
            return np.array([])
        
        pred_ids = list(predictions.keys())
        cost_matrix = np.zeros((len(pred_ids), len(detections)))
        
        for i, pred_id in enumerate(pred_ids):
            pred_box = predictions[pred_id]
            for j, det_box in enumerate(detections):
                # Комбинированная метрика: IoU + евклидово расстояние
                iou = self.calculate_iou(pred_box, det_box)
                
                # Евклидово расстояние между центрами
                pred_center = (pred_box[0] + pred_box[2]/2, pred_box[1] + pred_box[3]/2)
                det_center = (det_box[0] + det_box[2]/2, det_box[1] + det_box[3]/2)
                euclidean_dist = np.sqrt(
                    (pred_center[0] - det_center[0])**2 + 
                    (pred_center[1] - det_center[1])**2
                )
                
                # Нормализация расстояния
                normalized_dist = euclidean_dist / self.max_distance
                
                # Комбинированная стоимость (меньше = лучше)
                cost = (1 - iou) + normalized_dist
                cost_matrix[i, j] = cost
        
        return cost_matrix, pred_ids
    
    def update(self, detected_boxes):
        """Основной метод обновления трекера"""
        current_time = time.time()
        self.tracking_stats['total_detections'] += len(detected_boxes)
        
        # Предсказание позиций
        predictions = self.predict_positions()
        
        # Вычисление матрицы стоимости
        if predictions and detected_boxes:
            cost_matrix, pred_ids = self.calculate_distance_cost(predictions, detected_boxes)
            
            # Венгерский алгоритм для оптимального сопоставления
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Обработка сопоставлений
            matched_pairs = []
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 1.5:  # Порог принятия сопоставления
                    target_id = pred_ids[row]
                    detection = detected_boxes[col]
                    matched_pairs.append((target_id, detection))
                    self.tracking_stats['successful_matches'] += 1
            
            # Обновление сопоставленных целей
            matched_detection_indices = set()
            for target_id, detection in matched_pairs:
                # Обновление Kalman фильтра
                kf = self.targets[target_id]['kalman_filter']
                kf.update(np.array(detection).reshape((4, 1)))
                
                # Обновление информации о цели
                self.targets[target_id]['bbox'] = detection
                self.targets[target_id]['last_seen'] = current_time
                self.targets[target_id]['prediction_error'] = self._calculate_prediction_error(
                    predictions[target_id], detection
                )
                
                # Сброс счетчика исчезновения
                if target_id in self.disappeared:
                    del self.disappeared[target_id]
                
                matched_detection_indices.add(detected_boxes.index(detection))
        else:
            matched_pairs = []
            matched_detection_indices = set()
        
        # Создание новых целей из неспопоставленных обнаружений
        for i, detection in enumerate(detected_boxes):
            if i not in matched_detection_indices:
                self._create_new_target(detection, current_time)
        
        # Обработка исчезнувших целей
        self._handle_disappeared_targets(current_time)
        
        # Адаптивная настройка параметров
        self._adapt_parameters()
        
        return self.targets
    
    def _create_new_target(self, detection, current_time):
        """Создание новой цели"""
        kalman_filter = self.create_kalman_filter(detection)
        
        self.targets[self.next_id] = {
            'bbox': detection,
            'kalman_filter': kalman_filter,
            'last_seen': current_time,
            'created_at': current_time,
            'color': 'unknown',
            'color_confidence': 0.0,
            'fire_timer': None,
            'prediction_error': 0.0,
            'stability_score': 0.0,
            'update_count': 1
        }
        
        self.next_id += 1
    
    def _calculate_prediction_error(self, prediction, actual):
        """Вычисление ошибки предсказания"""
        pred_center = (prediction[0] + prediction[2]/2, prediction[1] + prediction[3]/2)
        actual_center = (actual[0] + actual[2]/2, actual[1] + actual[3]/2)
        
        error = np.sqrt(
            (pred_center[0] - actual_center[0])**2 + 
            (pred_center[1] - actual_center[1])**2
        )
        
        return error
    
    def _handle_disappeared_targets(self, current_time):
        """Обработка исчезнувших целей"""
        targets_to_remove = []
        
        for target_id in list(self.targets.keys()):
            target = self.targets[target_id]
            
            # Проверка времени последнего обнаружения
            time_since_last_seen = current_time - target['last_seen']
            
            if time_since_last_seen > 1.0:  # 1 секунда без обнаружения
                self.disappeared[target_id] += 1
                
                # Продолжаем предсказание для исчезнувших целей
                kf = target['kalman_filter']
                kf.predict()
                pred_x, pred_y, pred_w, pred_h = kf.x[:4].flatten()
                target['bbox'] = (int(pred_x), int(pred_y), int(pred_w), int(pred_h))
                
                # Удаление цели после длительного отсутствия
                if self.disappeared[target_id] > self.max_disappeared:
                    targets_to_remove.append(target_id)
                    self.tracking_stats['lost_targets'] += 1
        
        # Удаление целей
        for target_id in targets_to_remove:
            del self.targets[target_id]
            if target_id in self.disappeared:
                del self.disappeared[target_id]
    
    def _adapt_parameters(self):
        """Адаптивная настройка параметров"""
        if self.tracking_stats['total_detections'] > 100:
            # Вычисление метрик производительности
            match_rate = (self.tracking_stats['successful_matches'] / 
                         max(1, self.tracking_stats['total_detections']))
            
            # Адаптация максимального расстояния
            if match_rate < 0.7:  # Слишком много потерянных целей
                self.max_distance = min(150, self.max_distance * 1.1)
            elif match_rate > 0.9:  # Слишком много ложных сопоставлений
                self.max_distance = max(50, self.max_distance * 0.9)
            
            # Адаптация времени исчезновения
            avg_prediction_error = np.mean([
                target.get('prediction_error', 0) 
                for target in self.targets.values()
            ])
            
            if avg_prediction_error > 20:  # Высокая ошибка предсказания
                self.max_disappeared = min(50, self.max_disappeared + 5)
            elif avg_prediction_error < 5:  # Низкая ошибка предсказания
                self.max_disappeared = max(15, self.max_disappeared - 2)
    
    def get_target_stability(self, target_id):
        """Оценка стабильности цели для принятия решения о стрельбе"""
        if target_id not in self.targets:
            return 0.0
        
        target = self.targets[target_id]
        
        # Факторы стабильности
        age_factor = min(1.0, (time.time() - target['created_at']) / 5.0)  # Стабильность со временем
        error_factor = max(0.0, 1.0 - target.get('prediction_error', 100) / 50.0)  # Точность предсказания
        update_factor = min(1.0, target.get('update_count', 0) / 10.0)  # Количество обновлений
        
        stability = (age_factor + error_factor + update_factor) / 3.0
        target['stability_score'] = stability
        
        return stability
    
    def should_engage_target(self, target_id, color_classification, color_confidence):
        """Принятие решения о поражении цели"""
        if target_id not in self.targets:
            return False, "Target not found"
        
        target = self.targets[target_id]
        
        # Проверка красной повязки
        if color_classification == "red" and color_confidence > 0.6:
            target['fire_timer'] = None  # Сброс таймера
            return False, "Red bandana detected - friendly"
        
        # Проверка стабильности цели
        stability = self.get_target_stability(target_id)
        if stability < 0.5:
            return False, f"Target unstable (stability: {stability:.2f})"
        
        # Проверка времени задержки
        current_time = time.time()
        if target['fire_timer'] is None:
            target['fire_timer'] = current_time
            return False, f"Fire delay started ({self.fire_delay}s)"
        
        elapsed = current_time - target['fire_timer']
        if elapsed < self.fire_delay:
            return False, f"Fire delay ({self.fire_delay - elapsed:.1f}s remaining)"
        
        # Дополнительная проверка на неопределенность
        if color_classification == "unknown" and color_confidence < 0.3:
            return False, "Color classification uncertain"
        
        # Разрешение на поражение
        return True, "Target engagement authorized"
    
    def get_tracking_statistics(self):
        """Получение статистики трекинга"""
        stats = self.tracking_stats.copy()
        
        if stats['total_detections'] > 0:
            stats['match_rate'] = stats['successful_matches'] / stats['total_detections']
        else:
            stats['match_rate'] = 0.0
        
        stats['active_targets'] = len(self.targets)
        stats['disappeared_targets'] = len(self.disappeared)
        
        # Средняя стабильность целей
        if self.targets:
            avg_stability = np.mean([
                target.get('stability_score', 0) 
                for target in self.targets.values()
            ])
            stats['average_stability'] = avg_stability
        else:
            stats['average_stability'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Сброс статистики"""
        self.tracking_stats = {
            'total_detections': 0,
            'successful_matches': 0,
            'lost_targets': 0,
            'false_positives': 0
        }