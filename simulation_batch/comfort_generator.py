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

    # base probability patient requests airflow on a change
    p_airflow: float = 0.2

    # --- Added "CO2 feature" knobs (implemented via airflow requests) ---
    # extra probability of requesting airflow during day parts when CO2 is likely to feel worse
    p_airflow_day_extra: float = 0.10  # daytime bump
    p_airflow_evening_extra: float = 0.05  # evening bump

    # probability to request airflow if the generator thinks it's "stuffy"
    # (we simulate "stuffy moments" as random events during occupied hours)
    p_stuffy_event: float = 0.15
    p_airflow_if_stuffy: float = 0.75

    # --- Added temperature behavior knobs ---
    # patient temp-change sensitivity: larger means they make bigger changes (in °C)
    temp_adjust_sigma_c: float = 0.25

    # clamp to realistic preference ranges
    min_main_temp_c: float = 10
    max_main_temp_c: float = 30.0


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
        sec = rng.randint(0, 24 * 60 * 60 - 1)
        times.append(day_start + timedelta(seconds=sec))
    times.sort()
    return times


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pick_targets_for_time(rng: random.Random, t: datetime, policy: ComfortPolicy) -> Dict:
    """
    Generate human intent targets at time t.
    Uses rough day-part behavior (night/morning/afternoon/evening),
    plus:
      - CO2 feature: "stuffy" events -> airflow request
      - temperature feature: small personal adjustment noise + clamping
    """
    hour = t.astimezone(timezone.utc).hour

    # --- CO2-ish behavior proxy: airflow decisions ---
    # baseline airflow probability, with day-part bumps
    p_airflow = policy.p_airflow
    if 6 <= hour < 18:
        p_airflow += policy.p_airflow_day_extra
    elif 18 <= hour < 24:
        p_airflow += policy.p_airflow_evening_extra

    # simulate occasional "stuffy" moments during typical occupied hours
    stuffy = False
    if 8 <= hour < 22:
        stuffy = rng.random() < policy.p_stuffy_event

    if stuffy:
        airflow = rng.random() < policy.p_airflow_if_stuffy
    else:
        airflow = rng.random() < p_airflow

    # --- day-part defaults ---
    if 0 <= hour < 6:       # night
        t_main_base = rng.uniform(20.0, 22.0)
        light = 0.0
        sound = rng.uniform(0, 20)
    elif 6 <= hour < 12:    # morning
        t_main_base = rng.uniform(21.0, 23.0)
        light = rng.uniform(10, 40)
        sound = rng.uniform(10, 35)
    elif 12 <= hour < 18:   # afternoon
        t_main_base = rng.uniform(22.0, 24.0)
        light = rng.uniform(20, 60)
        sound = rng.uniform(15, 45)
    else:                   # evening
        t_main_base = rng.uniform(20.0, 22.5)
        light = rng.uniform(5, 35)
        sound = rng.uniform(0, 25)

    # --- Temperature feature: personal adjustment ---
    # add a small random "preference tweak" so it’s not purely bucketed
    t_main = t_main_base + rng.gauss(0.0, policy.temp_adjust_sigma_c)
    t_main = _clamp(t_main, policy.min_main_temp_c, policy.max_main_temp_c)
    t_main = round(t_main, 2)

    light = round(light, 2)
    sound = round(sound, 2)

    t_toilet: Optional[float] = None
    if rng.random() < policy.p_set_toilet_temp:
        t_toilet = rng.uniform(19.0, 23.0)
        # also give toilet temp a small preference tweak
        t_toilet += rng.gauss(0.0, 0.2)
        t_toilet = _clamp(t_toilet, 18.0, 24.0)
        t_toilet = round(t_toilet, 2)

    return dict(
        temperature_main=t_main,
        temperature_toilet=t_toilet,
        light_intensity=light,
        sound_level=sound,
        airflow=airflow,
        # Optional: if your ComfortPreference model has a co2_target/co2_sensitivity column,
        # you can add it here and store it below.
        # co2_target=...,
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
            assigns = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.end_time > start_time)
                .filter(RoomAssignment.start_time < end_time)
                .all()
            )

            for a in assigns:
                a_start = max(a.start_time.astimezone(timezone.utc), start_time)
                a_end = min(a.end_time.astimezone(timezone.utc), end_time)

                day_cursor = a_start.replace(hour=0, minute=0, second=0, microsecond=0)
                while day_cursor < a_end:
                    day_start, day_end = _day_bounds(day_cursor)

                    w0 = max(day_start, a_start)
                    w1 = min(day_end, a_end)

                    if w0 >= w1:
                        day_cursor += timedelta(days=1)
                        continue

                    k = self.rng.randint(0, self.policy.max_changes_per_day)
                    times = _random_times_in_day(self.rng, day_start, k=k)
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

                        # If you add real CO2 columns to ComfortPreference, set them here:
                        # row.co2_target = targets["co2_target"]

                        session.add(row)
                        inserted += 1

                    day_cursor += timedelta(days=1)

        return inserted
