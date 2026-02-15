from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from persistence.models.utility_usage import UtilityUsage

from persistence.models.device import Device as DeviceModel
from persistence.models.toilet_light import ToiletLight


@dataclass
class ToiletUsagePolicy:
    """
    Generates water + toilet-light utility usage per patient/room assignment.

    Per toilet visit:
      1) UtilityUsage(category="toilet_light") session with power_consumption (kWh)
      2) UtilityUsage(category="water") session with water_consumption (liters)
         ✅ same [start_time, end_time] as the light session

    Showers:
      - only in the morning window
      - 50% chance per occupied day (per assignment/day)
      - UtilityUsage(category="water") session with liters = duration * flow
    """

    # Visits per day-part (inclusive randint ranges)
    visits_night_range: Tuple[int, int] = (0, 1)      # 00-06 (rare)
    visits_morning_range: Tuple[int, int] = (1, 3)    # 06-12
    visits_afternoon_range: Tuple[int, int] = (2, 5)  # 12-18 (most)
    visits_evening_range: Tuple[int, int] = (1, 4)    # 18-24

    # Toilet light energy assumptions
    toilet_light_power_w: float = 12.0
    light_duration_s_range: Tuple[int, int] = (60, 6 * 60)  # 1–6 minutes

    # Water per toilet visit (liters)
    flush_l_range: Tuple[float, float] = (4.0, 9.0)
    sink_l_range: Tuple[float, float] = (0.3, 3.0)
    night_sink_multiplier: float = 0.8

    # Shower behavior
    shower_probability: float = 0.50
    shower_window_hours: Tuple[int, int] = (6, 10)  # morning only
    shower_duration_s_range: Tuple[int, int] = (5 * 60, 12 * 60)  # 5–12 min
    shower_flow_l_per_min_range: Tuple[float, float] = (6.0, 12.0)  # liters/min


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


def _get_toilet_light_device_id(session, *, room_id: int) -> int | None:
    """
    Resolve toilet light device_id for a room via Device table.
    Assumes seeding created Device rows with device_type="toilet_light" and room_id set.
    """
    d = (
        session.query(DeviceModel)
        .filter(DeviceModel.room_id == room_id)
        .filter(DeviceModel.device_type == "toilet_light")
        .one_or_none()
    )
    return int(d.device_id) if d else None


class ToiletUsageGenerator:
    """
    Inserts UtilityUsage rows for toilet visits and showers
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

                # Resolve toilet light device once per assignment/room
                toilet_light_device_id = _get_toilet_light_device_id(session, room_id=a.room_id)

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

                    # -------------------------
                    # Morning shower (50% chance)
                    # -------------------------
                    if self.rng.random() < self.policy.shower_probability:
                        h0, h1 = self.policy.shower_window_hours
                        s0 = day0.replace(hour=h0, minute=0, second=0, microsecond=0)
                        s1 = day0.replace(hour=h1, minute=0, second=0, microsecond=0)

                        ss0 = max(s0, w0)
                        ss1 = min(s1, w1)

                        if ss0 < ss1:
                            t_shower = _random_times_in_window(self.rng, ss0, ss1, k=1)[0]
                            dur_s = self.rng.randint(*self.policy.shower_duration_s_range)
                            t_shower_end = min(t_shower + timedelta(seconds=dur_s), w1)

                            minutes = max(0.0, (t_shower_end - t_shower).total_seconds() / 60.0)
                            lpm = self.rng.uniform(*self.policy.shower_flow_l_per_min_range)
                            liters = float(minutes * lpm)

                            session.add(
                                UtilityUsage(
                                    room_id=a.room_id,
                                    category="water",
                                    start_time=t_shower,
                                    end_time=t_shower_end,
                                    power_consumption=None,
                                    water_consumption=liters,
                                )
                            )
                            inserted += 1

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

                        k = self.rng.randint(k_range[0], k_range[1])
                        visit_times = _random_times_in_window(self.rng, pp0, pp1, k)

                        for t in visit_times:
                            # ---- Toilet light session ----
                            dur_s = self.rng.randint(*self.policy.light_duration_s_range)
                            t_end = min(t + timedelta(seconds=dur_s), w1)

                            hours = max(0.0, (t_end - t).total_seconds() / 3600.0)
                            power_kwh = (self.policy.toilet_light_power_w * hours) / 1000.0

                            session.add(
                                UtilityUsage(
                                    room_id=a.room_id,
                                    category="toilet_light",
                                    start_time=t,
                                    end_time=t_end,
                                    power_consumption=power_kwh,
                                    water_consumption=None,
                                )
                            )
                            inserted += 1

                            # Mirror light ON/OFF in toilet_lights table
                            if toilet_light_device_id is not None:
                                session.add(ToiletLight(device_id=toilet_light_device_id, state=True, timestamp=t))
                                session.add(ToiletLight(device_id=toilet_light_device_id, state=False, timestamp=t_end))

                            # ---- Water usage for the visit ----
                            # ✅ same [t, t_end] as the light session
                            flush_l = self.rng.uniform(*self.policy.flush_l_range)
                            sink_l = self.rng.uniform(*self.policy.sink_l_range)
                            if part_name == "night":
                                sink_l *= self.policy.night_sink_multiplier

                            liters = float(flush_l + sink_l)

                            session.add(
                                UtilityUsage(
                                    room_id=a.room_id,
                                    category="water",
                                    start_time=t,
                                    end_time=t_end,  # ✅ MATCH LIGHT DURATION
                                    power_consumption=None,
                                    water_consumption=liters,
                                )
                            )
                            inserted += 1

                    day_cursor += timedelta(days=1)

        return inserted
