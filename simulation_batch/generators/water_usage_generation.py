from __future__ import annotations

# Build water-usage rows.
# - Stores water rules - ToiletUsagePolicy
# - Uses toilet visits and shower events
# - Sums daily water use
# - Writes utility rows - ToiletUsageGenerator

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from persistence.database import session_scope
from persistence.models.device import Device as DeviceModel
from persistence.models.room_assignment import RoomAssignment
from persistence.models.toilet_light import ToiletLight
from simulation_batch.csv_storage import flush_utility_usage_writes, insert_utility_usage, write_model_rows


@dataclass
class ToiletUsagePolicy:
    visits_night_range: Tuple[int, int] = (0, 1)
    visits_morning_range: Tuple[int, int] = (1, 2)
    visits_afternoon_range: Tuple[int, int] = (1, 3)
    visits_evening_range: Tuple[int, int] = (1, 2)
    flush_l_range: Tuple[float, float] = (4.0, 9.0)
    sink_l_range: Tuple[float, float] = (0.3, 3.0)
    wc_duration_s_range: Tuple[int, int] = (1 * 60, 5 * 60)
    night_sink_multiplier: float = 0.8
    shower_probability: float = 0.20
    shower_window_hours: Tuple[int, int] = (6, 10)
    shower_duration_s_range: Tuple[int, int] = (4 * 60, 8 * 60)
    shower_flow_l_per_min_range: Tuple[float, float] = (5.0, 8.0)


def _day_start(t: datetime) -> datetime:
    t = t.astimezone(timezone.utc)
    return t.replace(hour=0, minute=0, second=0, microsecond=0)


def _random_times_in_window(rng: random.Random, w0: datetime, w1: datetime, k: int) -> List[datetime]:
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


def _merge_intervals(intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    if not intervals:
        return []
    ordered = sorted((start, end) for start, end in intervals if start < end)
    if not ordered:
        return []
    merged: List[Tuple[datetime, datetime]] = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


class ToiletUsageGenerator:
    def __init__(self, *, seed: int = 42, policy: ToiletUsagePolicy | None = None):
        self.rng = random.Random(seed)
        self.policy = policy or ToiletUsagePolicy()

    def generate_for_horizon(self, start_time: datetime, end_time: datetime) -> int:
        start_time = start_time.astimezone(timezone.utc)
        end_time = end_time.astimezone(timezone.utc)
        inserted = 0
        toilet_light_rows: List[dict[str, object]] = []
        with session_scope() as session:
            toilet_light_device_ids = {
                int(room_id): int(device_id)
                for room_id, device_id in session.query(DeviceModel.room_id, DeviceModel.device_id)
                .filter(DeviceModel.device_type == "toilet_light")
                .all()
                if room_id is not None and device_id is not None
            }
            assigns = session.query(RoomAssignment).filter(RoomAssignment.end_time > start_time).filter(RoomAssignment.start_time < end_time).all()
            for assignment in assigns:
                a_start = max(assignment.start_time.astimezone(timezone.utc), start_time)
                a_end = min(assignment.end_time.astimezone(timezone.utc), end_time)
                day_cursor = _day_start(a_start)
                while day_cursor < a_end:
                    day0 = day_cursor
                    day1 = day0 + timedelta(days=1)
                    w0 = max(day0, a_start)
                    w1 = min(day1, a_end)
                    if w0 >= w1:
                        day_cursor += timedelta(days=1)
                        continue
                    total_liters = 0.0
                    toilet_light_intervals: List[Tuple[datetime, datetime]] = []
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
                            liters = float(minutes * self.rng.uniform(*self.policy.shower_flow_l_per_min_range))
                            total_liters += liters
                            toilet_light_intervals.append((t_shower, t_shower_end))
                    parts = [
                        ("night", day0.replace(hour=0), day0.replace(hour=6), self.policy.visits_night_range),
                        ("morning", day0.replace(hour=6), day0.replace(hour=12), self.policy.visits_morning_range),
                        ("afternoon", day0.replace(hour=12), day0.replace(hour=18), self.policy.visits_afternoon_range),
                        ("evening", day0.replace(hour=18), day1, self.policy.visits_evening_range),
                    ]
                    for part_name, p0, p1, k_range in parts:
                        pp0 = max(p0, w0)
                        pp1 = min(p1, w1)
                        if pp0 >= pp1:
                            continue
                        sampled_k = self.rng.randint(k_range[0], k_range[1])
                        k = min(sampled_k, max(0, round(sampled_k * _window_fraction(w0, w1, p0, p1))))
                        visit_times = _random_times_in_window(self.rng, pp0, pp1, k)
                        for visit_time in visit_times:
                            wc_dur_s = self.rng.randint(*self.policy.wc_duration_s_range)
                            visit_end = min(visit_time + timedelta(seconds=wc_dur_s), pp1)
                            duration_fraction = max(
                                0.0,
                                min(1.0, (visit_end - visit_time).total_seconds() / max(float(wc_dur_s), 1.0)),
                            )
                            flush_l = self.rng.uniform(*self.policy.flush_l_range)
                            sink_l = self.rng.uniform(*self.policy.sink_l_range) * duration_fraction
                            if part_name == "night":
                                sink_l *= self.policy.night_sink_multiplier
                            total_liters += float(flush_l + sink_l)
                            toilet_light_intervals.append((visit_time, visit_end))
                    device_id = toilet_light_device_ids.get(int(assignment.room_id))
                    if device_id is not None:
                        for interval_start, interval_end in _merge_intervals(toilet_light_intervals):
                            toilet_light_rows.append({"device_id": device_id, "state": True, "timestamp": interval_start})
                            toilet_light_rows.append({"device_id": device_id, "state": False, "timestamp": interval_end})
                    if total_liters > 0.0:
                        insert_utility_usage(room_id=assignment.room_id, category="water", start_time=w0, end_time=w1, power_kwh=None, water_liters=round(total_liters, 2))
                        inserted += 1
                    day_cursor += timedelta(days=1)
            if toilet_light_rows:
                write_model_rows(ToiletLight, toilet_light_rows)
                session.bulk_insert_mappings(ToiletLight, toilet_light_rows)
        flush_utility_usage_writes()
        return inserted


__all__ = ["ToiletUsageGenerator", "ToiletUsagePolicy"]
