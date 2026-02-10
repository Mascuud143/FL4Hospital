from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator


@dataclass(frozen=True)
class SimClock:
    """
    Deterministic simulated-time iterator.

    Example:
        clock = SimClock(start, end, step_s=60)
        for t in clock:
            ...
    """
    start_time: datetime
    end_time: datetime
    step_s: int = 60  # simulated seconds per tick

    def __post_init__(self):
        if self.step_s <= 0:
            raise ValueError("step_s must be > 0")

    def __iter__(self) -> Iterator[datetime]:
        t = self.start_time.astimezone(timezone.utc)
        end = self.end_time.astimezone(timezone.utc)
        step = timedelta(seconds=int(self.step_s))

        while t <= end:
            yield t
            t = t + step
