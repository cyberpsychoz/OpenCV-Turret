from __future__ import annotations

from src.analytics.stats import StatsCalculator
from src.storage.event_logger import EventLogger


def generate_text_report(db_path: str, since: float | None = None) -> str:
    logger = EventLogger(db_path)
    stats = StatsCalculator(logger)

    data = stats.summary(since=since) if since else stats.today()
    logger.close()

    lines = [
        "=" * 50,
        "  Wildlife Deterrent — Event Report",
        "=" * 50,
        "",
        f"Total events: {data['total_events']}",
        "",
        "By species:",
    ]

    for species, count in data["by_species"].items():
        lines.append(f"  {species:>12}: {count}")

    lines.append("")
    lines.append("By zone:")
    for zone, count in data["by_zone"].items():
        lines.append(f"  {zone:>12}: {count}")

    lines.append("")
    lines.append("By hour:")
    for hour, count in data["by_hour"].items():
        bar = "#" * min(count, 40)
        lines.append(f"  {hour:02d}:00  {bar} ({count})")

    lines.append("")
    lines.append("By event type:")
    for etype, count in data["by_type"].items():
        lines.append(f"  {etype:>12}: {count}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)
