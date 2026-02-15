# Wildlife Deterrent Vision System

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

Real-time animal detection and deterrent system built on YOLOv8 and OpenCV. Detects wildlife entering user-defined zones, logs events to SQLite, triggers sound/notification alerts, and records video clips.

## Features

- **YOLOv8n + ByteTrack** -- detection and tracking of 10 animal species (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
- **Zone monitoring** -- user-drawn polygon zones with per-species filtering and loitering detection
- **Alert pipeline** -- sound deterrents (pygame), desktop notifications (plyer), Telegram bot; cooldown-based deduplication
- **Event logging** -- SQLite database with WAL mode, indexed queries by species/zone/time
- **Clip recording** -- rolling pre-buffer + post-event MP4 recording
- **Screenshots** -- annotated JPEG snapshots on each event
- **Analytics** -- per-day/hour/species statistics, position heatmaps, text reports
- **Multiple sources** -- webcam, video file, RTSP stream (threaded reader)
- **Interactive zone editor** -- draw protection zones with mouse, saved as JSON
- **YAML configuration** -- all parameters externalized, no hardcoded values

## Installation

```bash
pip install ultralytics opencv-python numpy PyYAML pygame plyer
```

For development:

```bash
pip install pytest pytest-cov
```

## Usage

```bash
# Run with webcam (default)
python -m src

# Run on a video file
python -m src --source path/to/video.mp4

# Run on RTSP stream
python -m src --source rtsp://192.168.1.100:554/stream

# Set up protection zones interactively
python -m src --setup-zones

# Generate event report
python -m src --report today
python -m src --report week

# Use custom config
python -m src --config my_config.yaml

# Verbose logging
python -m src -v
```

Press `q` or `ESC` to quit the live view.

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit as needed:

```yaml
detector:
  model: yolov8n.pt
  imgsz: 640
  confidence: 0.4
  device: cpu
  animal_classes: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

alerts:
  cooldown_seconds: 60
  sound:
    enabled: true
  desktop:
    enabled: true
  telegram:
    enabled: false
    bot_token: ""
    chat_id: ""

camera:
  source: 0
  width: 1280
  height: 720
```

## Project Structure

```
src/
  core/
    models.py          -- Detection, Track, Zone, Event dataclasses
    detector.py        -- YOLOv8 wrapper, animal class filter
    tracker.py         -- ByteTrack wrapper, speed calculation
    zone_manager.py    -- Polygon zones, point-in-polygon, entry/exit/loiter events
  sources/
    base.py            -- VideoSource ABC
    webcam.py, file.py, rtsp.py
  alerts/
    alert_manager.py   -- Cooldown, deduplication, routing
    sound.py, desktop.py, telegram.py
  storage/
    event_logger.py    -- SQLite event log
    clip_recorder.py   -- Rolling buffer to MP4
    screenshot.py      -- JPEG snapshots
  analytics/
    stats.py, heatmap.py, report.py
  ui/
    overlay.py         -- Bounding boxes, zones, HUD rendering
    zone_editor.py     -- Interactive polygon editor
  pipeline.py          -- Main processing loop
  app.py               -- CLI entry point
  __main__.py          -- python -m src
tests/
  test_detector.py, test_tracker.py, test_zone_manager.py,
  test_alert_manager.py, test_event_logger.py
```

## Testing

```bash
pytest tests/ -v --cov=src
```

## License

MIT License. This project is intended for wildlife management and educational purposes.

## Contact

Telegram: @gotogrub
