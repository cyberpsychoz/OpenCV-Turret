import time
import numpy as np
from collections import defaultdict, deque
from enum import Enum
import logging

class ThreatLevel(Enum):
    SAFE = 0
    CAUTION = 1
    WARNING = 2
    DANGER = 3

class EngagementDecision(Enum):
    HOLD_FIRE = 0
    PREPARE = 1
    ENGAGE = 2
    ABORT = 3

class SafetySystem:
    def __init__(self):
        self.engagement_log = deque(maxlen=1000)
        self.safety_violations = defaultdict(int)
        self.false_positive_history = deque(maxlen=100)
        
        # Настройки безопасности
        self.min_confidence_threshold = 0.7
        self.min_stability_threshold = 0.6
        self.required_consensus_frames = 5
        self.max_engagement_rate = 2  # максимум 2 поражения в минуту
        self.cooldown_period = 30  # 30 секунд между поражениями
        
        # История принятых решений для каждой цели
        self.target_decision_history = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_target_safety(self, target_id, target_data, classification_result):
        """
        Комплексная оценка безопасности поражения цели
        """
        safety_report = {
            'target_id': target_id,
            'timestamp': time.time(),
            'checks': {},
            'overall_safety': ThreatLevel.SAFE,
            'decision': EngagementDecision.HOLD_FIRE,
            'reasons': []
        }
        
        # 1. Проверка классификации красной повязки
        red_check = self._check_red_bandana_classification(classification_result)
        safety_report['checks']['red_bandana'] = red_check
        
        if red_check['is_red_bandana']:
            safety_report['decision'] = EngagementDecision.ABORT
            safety_report['reasons'].append("RED_BANDANA_DETECTED")
            safety_report['overall_safety'] = ThreatLevel.SAFE
            return safety_report
        
        # 2. Проверка стабильности цели
        stability_check = self._check_target_stability(target_data)
        safety_report['checks']['stability'] = stability_check
        
        if not stability_check['is_stable']:
            safety_report['reasons'].append("TARGET_UNSTABLE")
        
        # 3. Проверка консенсуса решений
        consensus_check = self._check_decision_consensus(target_id, classification_result)
        safety_report['checks']['consensus'] = consensus_check
        
        if not consensus_check['has_consensus']:
            safety_report['reasons'].append("NO_CONSENSUS")
        
        # 4. Проверка частоты поражений
        rate_check = self._check_engagement_rate()
        safety_report['checks']['engagement_rate'] = rate_check
        
        if not rate_check['within_limits']:
            safety_report['reasons'].append("ENGAGEMENT_RATE_EXCEEDED")
        
        # 5. Проверка времени задержки
        delay_check = self._check_fire_delay(target_data)
        safety_report['checks']['fire_delay'] = delay_check
        
        if not delay_check['delay_satisfied']:
            safety_report['reasons'].append("FIRE_DELAY_ACTIVE")
        
        # 6. Проверка зоны поражения
        zone_check = self._check_engagement_zone(target_data)
        safety_report['checks']['engagement_zone'] = zone_check
        
        if not zone_check['in_valid_zone']:
            safety_report['reasons'].append("INVALID_ENGAGEMENT_ZONE")
        
        # 7. Анализ исторических ложных срабатываний
        history_check = self._check_false_positive_history(target_id)
        safety_report['checks']['history'] = history_check
        
        if history_check['high_false_positive_risk']:
            safety_report['reasons'].append("HIGH_FALSE_POSITIVE_RISK")
        
        # Принятие финального решения
        safety_report['decision'], safety_report['overall_safety'] = self._make_final_decision(
            safety_report['checks']
        )
        
        # Логирование решения
        self._log_decision(safety_report)
        
        return safety_report
    
    def _check_red_bandana_classification(self, classification_result):
        """Проверка надежности классификации красной повязки"""
        color, confidence = classification_result
        
        return {
            'is_red_bandana': color == "red" and confidence > self.min_confidence_threshold,
            'color': color,
            'confidence': confidence,
            'threshold_met': confidence > self.min_confidence_threshold,
            'risk_level': ThreatLevel.SAFE if color == "red" else ThreatLevel.CAUTION
        }
    
    def _check_target_stability(self, target_data):
        """Проверка стабильности отслеживания цели"""
        stability_score = target_data.get('stability_score', 0)
        prediction_error = target_data.get('prediction_error', float('inf'))
        age = time.time() - target_data.get('created_at', time.time())
        
        is_stable = (
            stability_score > self.min_stability_threshold and
            prediction_error < 30 and  # Ошибка предсказания меньше 30 пикселей
            age > 2.0  # Цель отслеживается более 2 секунд
        )
        
        return {
            'is_stable': is_stable,
            'stability_score': stability_score,
            'prediction_error': prediction_error,
            'tracking_age': age,
            'risk_level': ThreatLevel.CAUTION if not is_stable else ThreatLevel.SAFE
        }
    
    def _check_decision_consensus(self, target_id, current_classification):
        """Проверка консенсуса решений за последние кадры"""
        # Добавляем текущее решение в историю
        self.target_decision_history[target_id].append({
            'timestamp': time.time(),
            'classification': current_classification
        })
        
        # Ограничиваем размер истории
        if len(self.target_decision_history[target_id]) > self.required_consensus_frames:
            self.target_decision_history[target_id].pop(0)
        
        recent_decisions = self.target_decision_history[target_id]
        
        if len(recent_decisions) < self.required_consensus_frames:
            return {
                'has_consensus': False,
                'decision_count': len(recent_decisions),
                'required_count': self.required_consensus_frames,
                'risk_level': ThreatLevel.CAUTION
            }
        
        # Проверяем консенсус
        non_red_count = sum(1 for d in recent_decisions if d['classification'][0] != 'red')
        consensus_ratio = non_red_count / len(recent_decisions)
        
        has_consensus = consensus_ratio >= 0.8  # 80% решений должны быть согласованы
        
        return {
            'has_consensus': has_consensus,
            'consensus_ratio': consensus_ratio,
            'non_red_decisions': non_red_count,
            'total_decisions': len(recent_decisions),
            'risk_level': ThreatLevel.SAFE if has_consensus else ThreatLevel.WARNING
        }
    
    def _check_engagement_rate(self):
        """Проверка частоты поражений"""
        current_time = time.time()
        
        # Считаем поражения за последнюю минуту
        recent_engagements = [
            log for log in self.engagement_log
            if log['decision'] == EngagementDecision.ENGAGE and
               current_time - log['timestamp'] <= 60
        ]
        
        within_limits = len(recent_engagements) < self.max_engagement_rate
        
        # Проверяем период охлаждения
        if recent_engagements:
            time_since_last = current_time - max(log['timestamp'] for log in recent_engagements)
            cooldown_satisfied = time_since_last >= self.cooldown_period
        else:
            cooldown_satisfied = True
        
        return {
            'within_limits': within_limits and cooldown_satisfied,
            'recent_engagements': len(recent_engagements),
            'max_allowed': self.max_engagement_rate,
            'cooldown_satisfied': cooldown_satisfied,
            'risk_level': ThreatLevel.DANGER if not within_limits else ThreatLevel.SAFE
        }
    
    def _check_fire_delay(self, target_data):
        """Проверка задержки перед стрельбой"""
        fire_timer = target_data.get('fire_timer')
        
        if fire_timer is None:
            return {
                'delay_satisfied': False,
                'remaining_delay': float('inf'),
                'risk_level': ThreatLevel.CAUTION
            }
        
        elapsed = time.time() - fire_timer
        fire_delay = 3.0  # 3 секунды задержка
        delay_satisfied = elapsed >= fire_delay
        
        return {
            'delay_satisfied': delay_satisfied,
            'elapsed_time': elapsed,
            'required_delay': fire_delay,
            'remaining_delay': max(0, fire_delay - elapsed),
            'risk_level': ThreatLevel.SAFE if delay_satisfied else ThreatLevel.CAUTION
        }
    
    def _check_engagement_zone(self, target_data):
        """Проверка нахождения цели в допустимой зоне поражения"""
        bbox = target_data.get('bbox', (0, 0, 0, 0))
        x, y, w, h = bbox
        
        # Центр цели
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Определяем допустимые зоны (например, исключаем края кадра)
        frame_margin = 50  # Отступ от краев кадра
        min_target_size = 2000  # Минимальная площадь цели
        
        # Предполагаем размер кадра 1280x720 (нужно получать динамически)
        frame_width, frame_height = 1280, 720
        
        in_valid_zone = (
            center_x > frame_margin and
            center_x < frame_width - frame_margin and
            center_y > frame_margin and
            center_y < frame_height - frame_margin and
            w * h > min_target_size
        )
        
        return {
            'in_valid_zone': in_valid_zone,
            'target_center': (center_x, center_y),
            'target_size': w * h,
            'min_required_size': min_target_size,
            'risk_level': ThreatLevel.WARNING if not in_valid_zone else ThreatLevel.SAFE
        }
    
    def _check_false_positive_history(self, target_id):
        """Анализ истории ложных срабатываний"""
        # Простая проверка на основе истории решений
        recent_decisions = self.target_decision_history.get(target_id, [])
        
        if len(recent_decisions) < 10:
            return {
                'high_false_positive_risk': False,
                'confidence_variance': 0,
                'risk_level': ThreatLevel.SAFE
            }
        
        # Анализ изменчивости классификации
        confidences = [d['classification'][1] for d in recent_decisions[-10:]]
        confidence_variance = np.var(confidences)
        
        # Высокая вариативность может указывать на неопределенность
        high_risk = confidence_variance > 0.1
        
        return {
            'high_false_positive_risk': high_risk,
            'confidence_variance': confidence_variance,
            'recent_decision_count': len(recent_decisions),
            'risk_level': ThreatLevel.WARNING if high_risk else ThreatLevel.SAFE
        }
    
    def _make_final_decision(self, checks):
        """Принятие финального решения на основе всех проверок"""
        # Проверяем критические блокировки
        if checks['red_bandana']['is_red_bandana']:
            return EngagementDecision.ABORT, ThreatLevel.SAFE
        
        if not checks['engagement_rate']['within_limits']:
            return EngagementDecision.HOLD_FIRE, ThreatLevel.DANGER
        
        # Подсчитываем количество пройденных проверок
        passed_checks = 0
        total_checks = 0
        max_risk_level = ThreatLevel.SAFE
        
        critical_checks = ['stability', 'consensus', 'fire_delay', 'engagement_zone']
        
        for check_name in critical_checks:
            if check_name in checks:
                total_checks += 1
                check_data = checks[check_name]
                
                # Обновляем максимальный уровень риска
                risk_level = check_data.get('risk_level', ThreatLevel.SAFE)
                if risk_level.value > max_risk_level.value:
                    max_risk_level = risk_level
                
                # Определяем, пройдена ли проверка
                if check_name == 'stability' and check_data.get('is_stable', False):
                    passed_checks += 1
                elif check_name == 'consensus' and check_data.get('has_consensus', False):
                    passed_checks += 1
                elif check_name == 'fire_delay' and check_data.get('delay_satisfied', False):
                    passed_checks += 1
                elif check_name == 'engagement_zone' and check_data.get('in_valid_zone', False):
                    passed_checks += 1
        
        # Принятие решения на основе количества пройденных проверок
        if passed_checks == total_checks and total_checks >= 3:
            return EngagementDecision.ENGAGE, max_risk_level
        elif passed_checks >= total_checks * 0.75:
            return EngagementDecision.PREPARE, max_risk_level
        else:
            return EngagementDecision.HOLD_FIRE, max_risk_level
    
    def _log_decision(self, safety_report):
        """Логирование принятого решения"""
        self.engagement_log.append({
            'timestamp': safety_report['timestamp'],
            'target_id': safety_report['target_id'],
            'decision': safety_report['decision'],
            'safety_level': safety_report['overall_safety'],
            'reasons': safety_report['reasons']
        })
        
        # Логирование в файл
        self.logger.info(f"Target {safety_report['target_id']}: "
                        f"{safety_report['decision'].name} - "
                        f"Safety: {safety_report['overall_safety'].name} - "
                        f"Reasons: {', '.join(safety_report['reasons'])}")
    
    def get_safety_statistics(self):
        """Получение статистики безопасности"""
        if not self.engagement_log:
            return {
                'total_decisions': 0,
                'engagement_rate': 0,
                'safety_violations': 0,
                'average_risk_level': 0
            }
        
        total_decisions = len(self.engagement_log)
        engagements = sum(1 for log in self.engagement_log 
                         if log['decision'] == EngagementDecision.ENGAGE)
        
        safety_violations = sum(1 for log in self.engagement_log 
                               if log['safety_level'].value >= ThreatLevel.WARNING.value)
        
        return {
            'total_decisions': total_decisions,
            'engagements': engagements,
            'engagement_rate': engagements / total_decisions if total_decisions > 0 else 0,
            'safety_violations': safety_violations,
            'violation_rate': safety_violations / total_decisions if total_decisions > 0 else 0,
            'recent_engagements': len([log for log in self.engagement_log 
                                     if time.time() - log['timestamp'] <= 300])  # За последние 5 минут
        }