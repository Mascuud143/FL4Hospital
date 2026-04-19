from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from simulation_batch.utility_usage_writer import insert_utility_usage, flush_utility_usage_writes


@dataclass
class ToiletUsagePolicy:
    """
    Generates aggregated water usage per patient/room assignment.

    Per occupied day:
      - Aggregate all toilet visit water usage into ONE row.
      - Optional shower water is added to the same daily total.
    """

    # Visits per day-part (inclusive randint ranges)
    visits_night_range: Tuple[int, int] = (0, 1)      # 00-06 (rare)
    visits_morning_range: Tuple[int, int] = (1, 2)    # 06-12
    visits_afternoon_range: Tuple[int, int] = (1, 3)  # 12-18
    visits_evening_range: Tuple[int, int] = (1, 2)    # 18-24

    # Water per toilet visit (liters)
    flush_l_range: Tuple[float, float] = (4.0, 9.0)
    sink_l_range: Tuple[float, float] = (0.3, 3.0)
    night_sink_multiplier: float = 0.8

    # Shower behavior
    shower_probability: float = 0.20
    shower_window_hours: Tuple[int, int] = (6, 10)  # morning only
    shower_duration_s_range: Tuple[int, int] = (4 * 60, 8 * 60)  # 4-8 min
    shower_flow_l_per_min_range: Tuple[float, float] = (5.0, 8.0)  # liters/min


def _day_start(t: datetime) -> datetime:
    t = t.astimezone(timezone.utc)
    return t.replace(hour=0, minute=0, second=0, microsecond=0)


def _random_times_in_window(rng: random.Random, w0: datetime, w1: datetime, k: int) -> List[datetime]:
    """Sample k random instants inside [w0, w1)."""
    if k <= 0 or w0 >= w1:
        return []
    span = int((w1 - w0).total_seconds())
    if span <= 0:
        return []
    times = [w0 + timedelta(seconds=rng.randint(0, span - 1)) for _ in range(k)]
    times.sort()
    return times


def _window_fraction(w0: datetime, w1: datetime, full0: datetime, full1: datetime) -> float:
    overlap_start = max(w0, full0)
    overlap_end = min(w1, full1)
    overlap_s = max(0.0, (overlap_end - overlap_start).total_seconds())
    full_s = max(1.0, (full1 - full0).total_seconds())
    return min(1.0, overlap_s / full_s)


class ToiletUsageGenerator:
    """
    Inserts aggregated UtilityUsage rows (water only) per occupied day
    for all RoomAssignments overlapping [start_time, end_time].

    Assumption: one patient per room at a time.
    """

    def __init__(self, *, seed: int = 42, policy: ToiletUsagePolicy | None = None):
        self.rng = random.Random(seed)
        self.policy = policy or ToiletUsagePolicy()

    def generate_for_horizon(self, start_time: datetime, end_time: datetime) -> int:
        start_time = start_time.astimezone(timezone.utc)
        end_time = end_time.astimezone(timezone.utc)

        inserted = 0

        with session_scope() as session:
            assigns = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.end_time > start_time)
                .filter(RoomAssignment.start_time < end_time)
                .all()
            )

            for a in assigns:
                a_start = max(a.start_time.astimezone(timezone.utc), start_time)
                a_end = min(a.end_time.astimezone(timezone.utc), end_time)

                day_cursor = _day_start(a_start)
                while day_cursor < a_end:
                    day0 = day_cursor
                    day1 = day0 + timedelta(days=1)

                    # Clip this day to assignment window
                    w0 = max(day0, a_start)
                    w1 = min(day1, a_end)
                    if w0 >= w1:
                        day_cursor += timedelta(days=1)
                        continue

                    total_liters = 0.0

                    # -------------------------
                    # Morning shower (optional)
                    # -------------------------
                    if self.rng.random() < self.policy.shower_probability:
                        h0, h1 = self.policy.shower_window_hours
                        s0 = day0.replace(hour=h0, minute=0, second=0, microsecond=0)
                        s1 = day0.replace(hour=h1, minute=0, second=0, microsecond=0)

                        ss0 = max(s0, w0)
                        ss1 = min(s1, w1)

                        if ss0 < ss1 and self.rng.random() < _window_fraction(w0, w1, s0, s1):
                            t_shower = _random_times_in_window(self.rng, ss0, ss1, k=1)[0]
                            dur_s = self.rng.randint(*self.policy.shower_duration_s_range)
                            t_shower_end = min(t_shower + timedelta(seconds=dur_s), w1)

                            minutes = max(0.0, (t_shower_end - t_shower).total_seconds() / 60.0)
                            lpm = self.rng.uniform(*self.policy.shower_flow_l_per_min_range)
                            liters = float(minutes * lpm)
                            total_liters += liters

                    # -------------------------
                    # Toilet visits by day-part
                    # -------------------------
                    parts = [
                        ("night",     day0.replace(hour=0),  day0.replace(hour=6),  self.policy.visits_night_range),
                        ("morning",   day0.replace(hour=6),  day0.replace(hour=12), self.policy.visits_morning_range),
                        ("afternoon", day0.replace(hour=12), day0.replace(hour=18), self.policy.visits_afternoon_range),
                        ("evening",   day0.replace(hour=18), day1,                  self.policy.visits_evening_range),
                    ]

                    for part_name, p0, p1, k_range in parts:
                        pp0 = max(p0, w0)
                        pp1 = min(p1, w1)
                        if pp0 >= pp1:
                            continue

                        sampled_k = self.rng.randint(k_range[0], k_range[1])
                        k = min(sampled_k, max(0, round(sampled_k * _window_fraction(w0, w1, p0, p1))))
                        visit_times = _random_times_in_window(self.rng, pp0, pp1, k)

                        for _t in visit_times:
                            # ---- Water usage for the visit ----
                            flush_l = self.rng.uniform(*self.policy.flush_l_range)
                            sink_l = self.rng.uniform(*self.policy.sink_l_range)
                            if part_name == "night":
                                sink_l *= self.policy.night_sink_multiplier

                            liters = float(flush_l + sink_l)
                            total_liters += liters

                    if total_liters > 0.0:
                        insert_utility_usage(
                            room_id=a.room_id,
                            category="water",
                            start_time=w0,
                            end_time=w1,
                            power_kwh=None,
                            water_liters=round(total_liters, 2),
                        )
                        inserted += 1

                    day_cursor += timedelta(days=1)

        flush_utility_usage_writes()
        return inserted
