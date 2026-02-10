from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference


@dataclass
class ComfortPolicy:
    """
    Controls how many preference changes a patient can do per day,
    and what ranges are used.
    """
    max_changes_per_day: int = 3
    # probability that toilet temp is set on a change
    p_set_toilet_temp: float = 0.6
    # probability patient requests airflow on a change
    p_airflow: float = 0.2


def _day_bounds(t: datetime) -> Tuple[datetime, datetime]:
    t = t.astimezone(timezone.utc)
    start = t.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def _random_times_in_day(
    rng: random.Random,
    day_start: datetime,
    *,
    k: int,
) -> List[datetime]:
    """
    Sample k random instants inside [day_start, day_start+1day).
    """
    if k <= 0:
        return []
    times = []
    for _ in range(k):
        # random second within the day
        sec = rng.randint(0, 24 * 60 * 60 - 1)
        times.append(day_start + timedelta(seconds=sec))
    times.sort()
    return times


def _pick_targets_for_time(rng: random.Random, t: datetime, policy: ComfortPolicy) -> Dict:
    """
    Generate human intent targets at time t.
    Uses rough day-part behavior (night/morning/afternoon/evening).
    """
    hour = t.astimezone(timezone.utc).hour

    # defaults
    airflow = rng.random() < policy.p_airflow

    if 0 <= hour < 6:       # night
        t_main = round(rng.uniform(20.0, 22.0), 2)
        light = 0.0
        sound = round(rng.uniform(0, 20), 2)
    elif 6 <= hour < 12:    # morning
        t_main = round(rng.uniform(21.0, 23.0), 2)
        light = round(rng.uniform(10, 40), 2)
        sound = round(rng.uniform(10, 35), 2)
    elif 12 <= hour < 18:   # afternoon
        t_main = round(rng.uniform(22.0, 24.0), 2)
        light = round(rng.uniform(20, 60), 2)
        sound = round(rng.uniform(15, 45), 2)
    else:                   # evening
        t_main = round(rng.uniform(20.0, 22.5), 2)
        light = round(rng.uniform(5, 35), 2)
        sound = round(rng.uniform(0, 25), 2)

    t_toilet: Optional[float] = None
    if rng.random() < policy.p_set_toilet_temp:
        t_toilet = round(rng.uniform(19.0, 23.0), 2)

    return dict(
        temperature_main=t_main,
        temperature_toilet=t_toilet,
        light_intensity=light,
        sound_level=sound,
        airflow=airflow,
    )


class ComfortGenerator:
    """
    Creates ComfortPreference rows at random times, bounded per patient per day,
    and only during active RoomAssignments.
    """

    def __init__(self, *, seed: int = 42, policy: Optional[ComfortPolicy] = None):
        self.rng = random.Random(seed)
        self.policy = policy or ComfortPolicy()

    def generate_for_horizon(self, start_time: datetime, end_time: datetime) -> int:
        """
        Generate ComfortPreference rows for all assignments overlapping [start_time, end_time].
        Returns number of rows inserted.
        """
        start_time = start_time.astimezone(timezone.utc)
        end_time = end_time.astimezone(timezone.utc)

        inserted = 0

        with session_scope() as session:
            # assignments that overlap horizon
            assigns = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.end_time > start_time)
                .filter(RoomAssignment.start_time < end_time)
                .all()
            )

            for a in assigns:
                a_start = max(a.start_time.astimezone(timezone.utc), start_time)
                a_end = min(a.end_time.astimezone(timezone.utc), end_time)

                # iterate each day in the assignment window
                day_cursor = a_start.replace(hour=0, minute=0, second=0, microsecond=0)
                while day_cursor < a_end:
                    day_start, day_end = _day_bounds(day_cursor)

                    # effective window for this day within assignment
                    w0 = max(day_start, a_start)
                    w1 = min(day_end, a_end)

                    if w0 >= w1:
                        day_cursor += timedelta(days=1)
                        continue

                    # bounded number of changes for this day
                    k = self.rng.randint(0, self.policy.max_changes_per_day)
                    times = _random_times_in_day(self.rng, day_start, k=k)

                    # keep only times in [w0, w1)
                    times = [t for t in times if w0 <= t < w1]

                    for t in times:
                        targets = _pick_targets_for_time(self.rng, t, self.policy)

                        row = ComfortPreference(
                            patient_id=a.patient_id,
                            room_id=a.room_id,
                            timestamp=t,
                            temperature_main=targets["temperature_main"],
                            temperature_toilet=targets["temperature_toilet"],
                            light_intensity=targets["light_intensity"],
                            sound_level=targets["sound_level"],
                            airflow=targets["airflow"],
                            source="simulation",
                        )
                        session.add(row)
                        inserted += 1

                    day_cursor += timedelta(days=1)

        return inserted
