"""
TimeContext — lightweight temporal context flags for ZADOS pipelines.

Stamps each processing turn with structured time metadata so that journals,
memory packets, and the LLM context window all carry consistent temporal
awareness.  The module is intentionally simple: it reads the system clock,
classifies the moment into human-readable categories, and produces a flat
list of context flags.  No external dependencies; no heavy processing.

Circadian Phases (aligned with ZADOS sleep-mode boundaries)
------------------------------------------------------------
  waking      05:00 – 07:00  transitioning from sleep
  active      07:00 – 18:00  normal waking processing
  wind_down   18:00 – 22:00  evening; sleep modes may activate
  sleep       22:00 – 05:00  deep-sleep / REM modes expected

Time-of-Day Bands
-----------------
  morning     06:00 – 12:00
  afternoon   12:00 – 18:00
  evening     18:00 – 22:00
  night       22:00 – 06:00

Usage
-----
    from zados.core.time_context import get_time_context

    tc = get_time_context(session_start=session.session_start_time)
    # tc.flags  → ["time:afternoon", "day:tuesday", "circadian:active"]
    # tc.to_dict()  → plain dict for bundle / journal / memory packet
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class TimeContextSnapshot:
    """Structured temporal context for one pipeline turn.

    All fields are plain Python types so the snapshot serialises without
    special handling.
    """

    # Raw timestamps
    timestamp: float = 0.0          # Unix epoch (float seconds)
    iso_timestamp: str = ""         # ISO 8601 UTC string, e.g. "2026-03-18T14:32:07Z"

    # Human-readable classifications
    hour: int = 0                   # 0-23 (local hour)
    time_of_day: str = ""           # morning / afternoon / evening / night
    day_of_week: str = ""           # monday .. sunday
    circadian_phase: str = ""       # waking / active / wind_down / sleep

    # Session-relative timing
    session_elapsed_s: float = 0.0  # seconds since session opened (0.0 if unknown)

    # Flat context flags — appended to journal notes, bundle context
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for MemoryPacket / JournalEntry."""
        return {
            "timestamp":         self.timestamp,
            "iso_timestamp":     self.iso_timestamp,
            "hour":              self.hour,
            "time_of_day":       self.time_of_day,
            "day_of_week":       self.day_of_week,
            "circadian_phase":   self.circadian_phase,
            "session_elapsed_s": self.session_elapsed_s,
            "flags":             list(self.flags),
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_time_context(
    session_start: Optional[float] = None,
    now: Optional[float] = None,
) -> TimeContextSnapshot:
    """Build a TimeContextSnapshot for the current moment.

    Parameters
    ----------
    session_start : float, optional
        Unix timestamp when the session was opened.  Used to compute
        ``session_elapsed_s``.  Pass ``session.session_start_time``.
    now : float, optional
        Override the current time (useful for tests).  Defaults to
        ``time.time()``.

    Returns
    -------
    TimeContextSnapshot
    """
    ts = now if now is not None else time.time()
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_local = datetime.fromtimestamp(ts)  # local time for circadian/TOD

    hour = dt_local.hour
    time_of_day = _classify_time_of_day(hour)
    day_of_week = dt_local.strftime("%A").lower()          # "monday" .. "sunday"
    circadian_phase = _classify_circadian_phase(hour)

    session_elapsed_s = 0.0
    if session_start is not None and session_start > 0:
        session_elapsed_s = max(0.0, ts - session_start)

    flags = [
        f"time:{time_of_day}",
        f"day:{day_of_week}",
        f"circadian:{circadian_phase}",
    ]
    if session_elapsed_s > 0:
        flags.append(f"elapsed:{int(session_elapsed_s)}s")

    return TimeContextSnapshot(
        timestamp=ts,
        iso_timestamp=dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        hour=hour,
        time_of_day=time_of_day,
        day_of_week=day_of_week,
        circadian_phase=circadian_phase,
        session_elapsed_s=session_elapsed_s,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify_time_of_day(hour: int) -> str:
    """Map a 0-23 hour to a named band."""
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    return "night"


def _classify_circadian_phase(hour: int) -> str:
    """Map a 0-23 hour to a ZADOS circadian phase label."""
    if 5 <= hour < 7:
        return "waking"
    if 7 <= hour < 18:
        return "active"
    if 18 <= hour < 22:
        return "wind_down"
    return "sleep"
