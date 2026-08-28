"""Time utilities"""

from datetime import datetime, timezone


def time_now() -> datetime:
    """Return datetime of call without microseconds"""
    # Naive UTC on purpose: these values are printed and subtracted as-is, and an
    # aware datetime would add a "+00:00" suffix to every timestamp we log.
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def time_now_str() -> str:
    """Return datetime of call as a string in the format YYYY-MM-DD_HH-MM-SS"""
    return time_now().strftime("%Y-%m-%d_%H-%M-%S")


def seconds_to_str(seconds: int) -> str:
    """Convert a duration in seconds to a string in the format HH:MM:SS

    Hours are not capped at 24: a 25h duration is "25:00:00", not "01:00:00".
    """
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
