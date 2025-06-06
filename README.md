# 🎯 Турель на базе OpenCV

> **Версия 0.3** — *Улучшенная версия с трекером и расширенным поиском повязок*

Это экспериментальный проект для тестовой турели, которая распознаёт людей с повязками на видео и может использоваться как основа для систем идентификации целей. Код написан на Python с использованием OpenCV.

## 🧪 Возможности
- **YOLOv8 через Ultralytics**: Точная детекция людей
- **MediaPipe**: Определение ключевых точек тела для точного выделения головы и рук
- **Цветовая классификация**: Поиск повязок на всей фигуре
- **Фильтр Калмана**: Предсказание движения целей
- **GPU-ускорение**: Поддержка CUDA (только для NVIDIA)
- **Прогресс-бар**: Отображение процента обработки

## ⚠️ Предупреждение
- **Сырая реализация** 🧪  
  Многие функции требуют доработки и оптимизации

## 📦 Установка

```bash
# Установка зависимостей
pip install opencv-python numpy mediapipe ultralytics filterp rich

# Для Windows (без GUI)
pip install opencv-python-headless numpy mediapipe ultralytics filterp rich
```

## 🚀 Использование

1. Поместите тестовые видео в папку `test_videos`
2. Запустите:
   ```bash
   python main.py
   ```
3. Обработанные видео сохранятся в папке `output`

## 📷 Пример вывода
```
[INFO] Найдено 2 видео для обработки:
1. test1.mp4
2. test2.mp4

[INFO] Обрабатываю: test1.mp4
Формат: 1280x720, 30 FPS, 900 кадров
[##################################################--------------------------] 67.2% | Пройдено: 00:00:18 | Осталось: 00:00:09
[INFO] Готово: processed_test1.avi | Затрачено: 00:00:27
```
[gif](https://github.com/cyberpsychoz/OpenCV-Turret/template.gif](https://github.com/cyberpsychoz/OpenCV-Turret/blob/main/template.gif)

## 🧰 Roadmap (План развития)
- [~] Подключение различных модулей для улучшения обнаружения целей 
- [ ] Интеграция с реальной турелью через Arduino/Raspberry Pi
- [~] Улучшение алгоритма классификации цвета
- [ ] Добавление трекинга целей
- [ ] Создание графического интерфейса (GUI)
- [ ] Поддержка RTSP-потоков с камер
- [ ] Тестирование на мобильных устройствах

## 🤝 Вклад
Любые улучшения приветствуются! Чтобы внести изменения:
1. Fork проекта
2. Создайте свою ветку (`git checkout -b feature/god-help`)
3. Зафиксируйте изменения (`git commit -m 'Add some feature'`)
4. Запушите (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия
MIT License
Проект создан в образовательных целях. Не используйте его в боевых системах.

## 📬 Связь
- Telegram: @terminisle
- Discord: terminisle

### 📌 Примечание для Windows-пользователей с AMD GPU
OpenCV не поддерживает CUDA на видеокартах AMD. Код автоматически переключается на CPU-режим. Для максимальной производительности используйте Windows с NVIDIA GPU или перейдите на Linux.

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)](https://github.com/cyberpsychoz/OpenCV-Turret)
