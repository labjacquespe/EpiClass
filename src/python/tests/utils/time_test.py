"""Format contract for epiclass.utils.time.

Nothing in the repo parses these strings back, so a format change breaks no
code -- it just silently makes new output filenames inconsistent with the ones
in past run directories (SHAP pickles, log dirs, bigwig metric chunks all
embed time_now_str()). These tests pin the format so that change cannot happen
unnoticed; they were written alongside the datetime.utcnow() deprecation fix
(Python 3.12), which had to preserve every property asserted here.
"""
from datetime import datetime, timedelta, timezone

import pytest

from epiclass.utils.time import seconds_to_str, time_now, time_now_str


class TestTimeNow:
    """time_now() / time_now_str()"""

    def test_is_naive_utc(self):
        """Value is naive (no tzinfo) so it prints without a +00:00 suffix."""
        now = time_now()
        assert now.tzinfo is None
        assert "+" not in str(now)

    def test_matches_utc_wall_clock(self):
        """Naive, but expressed in UTC rather than local time."""
        expected = datetime.now(timezone.utc).replace(tzinfo=None)
        assert abs(expected - time_now()) < timedelta(seconds=5)

    def test_microseconds_dropped(self):
        """Microseconds are truncated, not rounded."""
        assert time_now().microsecond == 0

    def test_subtraction_yields_timedelta(self):
        """Callers in mains/ print `end - begin`; both operands must be naive."""
        assert isinstance(time_now() - time_now(), timedelta)

    def test_str_format(self):
        """Used in directory names: YYYY-MM-DD_HH-MM-SS, no separators beyond those."""
        stamp = time_now_str()
        assert datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S")
        assert len(stamp) == 19


class TestSecondsToStr:
    """seconds_to_str()"""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "00:00:00"),
            (1, "00:00:01"),
            (59, "00:00:59"),
            (60, "00:01:00"),
            (3599, "00:59:59"),
            (3600, "01:00:00"),
            (86399, "23:59:59"),
            # A duration, not a time of day: hours are uncapped, never wrapping to 0.
            (86400, "24:00:00"),
            (90000, "25:00:00"),
            (90061, "25:01:01"),
            (356400, "99:00:00"),
        ],
    )
    def test_known_values(self, seconds: int, expected: str):
        """HH:MM:SS, zero-padded."""
        assert seconds_to_str(seconds) == expected
