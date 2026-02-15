from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml


@dataclass
class DetectorConfig:
    model: str = "yolov8n.pt"
    imgsz: int = 640
    confidence: float = 0.4
    device: str = "cpu"
    animal_classes: list[int] = field(
        default_factory=lambda: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    )


@dataclass
class TrackerConfig:
    type: str = "bytetrack"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8


@dataclass
class SoundConfig:
    enabled: bool = True
    directory: str = "data/sounds"


@dataclass
class DesktopConfig:
    enabled: bool = True


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class AlertsConfig:
    cooldown_seconds: int = 60
    sound: SoundConfig = field(default_factory=SoundConfig)
    desktop: DesktopConfig = field(default_factory=DesktopConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


@dataclass
class ScreenshotConfig:
    enabled: bool = True
    directory: str = "data/events"


@dataclass
class ClipConfig:
    enabled: bool = True
    directory: str = "data/events"
    pre_seconds: int = 5
    post_seconds: int = 10


@dataclass
class StorageConfig:
    database: str = "data/db/events.db"
    screenshots: ScreenshotConfig = field(default_factory=ScreenshotConfig)
    clips: ClipConfig = field(default_factory=ClipConfig)


@dataclass
class CameraConfig:
    source: Union[int, str] = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class DisplayConfig:
    show_window: bool = True
    show_fps: bool = True
    show_zones: bool = True
    show_tracks: bool = True


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    zones_file: str = "data/zones/zones.json"
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


def _merge(target: dict, source: dict) -> dict:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def _dict_to_dataclass(cls, data: dict):
    import dataclasses

    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, ftype in fieldtypes.items():
        if name not in data:
            continue
        val = data[name]
        if isinstance(val, dict):
            resolved = eval(ftype) if isinstance(ftype, str) else ftype
            if dataclasses.is_dataclass(resolved):
                val = _dict_to_dataclass(resolved, val)
        kwargs[name] = val
    return cls(**kwargs)


def load_config(path: Union[str, Path, None] = None) -> AppConfig:
    if path is None:
        candidates = [Path("config.yaml"), Path("config.example.yaml")]
        for c in candidates:
            if c.exists():
                path = c
                break

    if path is not None:
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    flat = {}
    if "detector" in raw:
        flat["detector"] = raw["detector"]
    if "tracker" in raw:
        flat["tracker"] = raw["tracker"]
    if "zones" in raw:
        flat["zones_file"] = raw["zones"].get("file", "data/zones/zones.json")
    if "alerts" in raw:
        flat["alerts"] = raw["alerts"]
    if "storage" in raw:
        flat["storage"] = raw["storage"]
    if "camera" in raw:
        flat["camera"] = raw["camera"]
    if "display" in raw:
        flat["display"] = raw["display"]

    return _dict_to_dataclass(AppConfig, flat)
