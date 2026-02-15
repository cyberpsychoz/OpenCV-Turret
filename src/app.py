from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import AppConfig, load_config


def _create_source(cfg: AppConfig):
    source = cfg.camera.source
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        from src.sources.webcam import WebcamSource
        return WebcamSource(
            device=int(source),
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
        )
    elif isinstance(source, str) and source.startswith("rtsp://"):
        from src.sources.rtsp import RTSPSource
        return RTSPSource(url=source, fps=cfg.camera.fps)
    else:
        from src.sources.file import FileSource
        return FileSource(path=source)


def run_pipeline(cfg: AppConfig) -> None:
    from src.alerts.alert_manager import AlertManager
    from src.pipeline import Pipeline
    from src.storage.clip_recorder import ClipRecorder
    from src.storage.event_logger import EventLogger
    from src.storage.screenshot import ScreenshotSaver

    pipeline = Pipeline(cfg)

    event_logger = EventLogger(cfg.storage.database)
    alert_manager = AlertManager(cfg.alerts)
    screenshot_saver = ScreenshotSaver(cfg.storage.screenshots.directory) if cfg.storage.screenshots.enabled else None
    clip_recorder = ClipRecorder(
        directory=cfg.storage.clips.directory,
        pre_seconds=cfg.storage.clips.pre_seconds,
        post_seconds=cfg.storage.clips.post_seconds,
    ) if cfg.storage.clips.enabled else None

    def on_event(event, frame):
        event_logger.log(event)
        alert_manager.handle(event, frame)
        if screenshot_saver:
            screenshot_saver.save(event, frame)
        if clip_recorder:
            clip_recorder.trigger(f"{event.species}_{event.zone_name}")

    pipeline.add_event_handler(on_event)

    source = _create_source(cfg)

    if clip_recorder:
        original_process = pipeline.process_frame

        def process_with_clip(frame):
            result = original_process(frame)
            clip_recorder.feed(frame)
            return result

        pipeline.process_frame = process_with_clip

    try:
        pipeline.run(source)
    finally:
        alert_manager.close()
        event_logger.close()
        if clip_recorder:
            clip_recorder.close()


def run_zone_setup(cfg: AppConfig) -> None:
    from src.ui.zone_editor import setup_zones

    source = _create_source(cfg)
    setup_zones(source, cfg.zones_file)
    source.release()


def run_report(cfg: AppConfig, period: str) -> None:
    from datetime import datetime, timedelta

    from src.analytics.report import generate_text_report

    since = None
    if period == "today":
        since = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    elif period == "week":
        since = (datetime.now() - timedelta(days=7)).timestamp()
    elif period == "month":
        since = (datetime.now() - timedelta(days=30)).timestamp()

    report = generate_text_report(cfg.storage.database, since=since)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wildlife Deterrent Vision System",
        prog="wildlife-deterrent",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    parser.add_argument("--source", "-s", default=None, help="Video source (webcam index, file path, or RTSP URL)")
    parser.add_argument("--setup-zones", action="store_true", help="Launch interactive zone editor")
    parser.add_argument("--report", choices=["today", "week", "month", "all"], help="Generate event report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    if args.source is not None:
        if args.source.isdigit():
            cfg.camera.source = int(args.source)
        else:
            cfg.camera.source = args.source

    if args.setup_zones:
        run_zone_setup(cfg)
    elif args.report:
        run_report(cfg, args.report)
    else:
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
