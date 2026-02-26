from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.medication import Medication

from simulation_batch.room_engine import _as_utc


# ==========================================================
# POLICY
# ==========================================================

@dataclass
class ComfortPolicy:
    max_changes_per_day: int = 3

    p_set_toilet_temp: float = 0.6
    p_airflow: float = 0.2

    p_airflow_day_extra: float = 0.10
    p_airflow_evening_extra: float = 0.05

    p_stuffy_event: float = 0.15
    p_airflow_if_stuffy: float = 0.75

    temp_adjust_sigma_c: float = 0.25

    min_main_temp_c: float = 10
    max_main_temp_c: float = 30.0

    # ✅ NEW: probability of biasing comfort toward medication
    p_bias_to_medication: float = 0.6


# ==========================================================
# HELPERS
# ==========================================================

def _day_bounds(t: datetime) -> Tuple[datetime, datetime]:
    t = _as_utc(t)
    start = t.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def _random_times_in_day(
    rng: random.Random,
    day_start: datetime,
    *,
    k: int,
) -> List[datetime]:

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


# ==========================================================
# TARGET GENERATION
# ==========================================================

def _pick_targets_for_time(
    rng: random.Random,
    t: datetime,
    policy: ComfortPolicy,
) -> Dict:

    t = _as_utc(t)
    hour = t.hour

    p_airflow = policy.p_airflow

    if 6 <= hour < 18:
        p_airflow += policy.p_airflow_day_extra
    elif 18 <= hour < 24:
        p_airflow += policy.p_airflow_evening_extra

    stuffy = False
    if 8 <= hour < 22:
        stuffy = rng.random() < policy.p_stuffy_event

    if stuffy:
        airflow = rng.random() < policy.p_airflow_if_stuffy
    else:
        airflow = rng.random() < p_airflow

    # Day-part defaults
    if 0 <= hour < 6:
        t_main_base = rng.uniform(20.0, 22.0)
        light = 0.0
        sound = rng.randint(0, 20)

    elif 6 <= hour < 12:
        t_main_base = rng.uniform(21.0, 23.0)
        light = rng.randint(10, 40)
        sound = rng.randint(10, 35)

    elif 12 <= hour < 18:
        t_main_base = rng.uniform(22.0, 24.0)
        light = rng.randint(20, 60)
        sound = rng.randint(15, 45)

    else:
        t_main_base = rng.uniform(20.0, 22.5)
        light = rng.randint(5, 35)
        sound = rng.randint(0, 25)

    t_main = t_main_base + rng.gauss(0.0, policy.temp_adjust_sigma_c)
    t_main = round(_clamp(t_main, policy.min_main_temp_c, policy.max_main_temp_c), 2)

    light = round(light, 2)
    sound = round(sound, 2)

    t_toilet: Optional[float] = None
    if rng.random() < policy.p_set_toilet_temp:
        t_toilet = rng.uniform(19.0, 23.0)
        t_toilet += rng.gauss(0.0, 0.2)
        t_toilet = round(_clamp(t_toilet, 18.0, 24.0), 2)

    return dict(
        temperature_main=t_main,
        temperature_toilet=t_toilet,
        light_intensity=light,
        sound_level=sound,
        airflow=airflow,
    )


# ==========================================================
# GENERATOR
# ==========================================================

class ComfortGenerator:

    def __init__(self, *, seed: int = 42, policy: Optional[ComfortPolicy] = None):
        self.rng = random.Random(seed)
        self.policy = policy or ComfortPolicy()

    def generate_for_horizon(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> int:

        start_time = _as_utc(start_time)
        end_time = _as_utc(end_time)

        inserted = 0

        with session_scope() as session:

            assigns = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.end_time > start_time)
                .filter(RoomAssignment.start_time < end_time)
                .all()
            )

            for a in assigns:

                a_start = _as_utc(a.start_time)
                a_end = _as_utc(a.end_time)

                w_start = max(a_start, start_time)
                w_end = min(a_end, end_time)

                day_cursor = w_start.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

                while day_cursor < w_end:

                    day_start, day_end = _day_bounds(day_cursor)

                    # Load medications once per day
                    meds = (
                        session.query(Medication)
                        .filter(Medication.patient_id == a.patient_id)
                        .filter(Medication.medication_time >= day_start)
                        .filter(Medication.medication_time < day_end)
                        .all()
                    )

                    med_times = [_as_utc(m.medication_time) for m in meds]

                    k = self.rng.randint(0, self.policy.max_changes_per_day)
                    times = _random_times_in_day(self.rng, day_start, k=k)
                    times = [t for t in times if w_start <= t < w_end]

                    for t in times:

                        targets = None

                        if med_times:

                            closest_med_time = min(
                                med_times,
                                key=lambda mt: abs(mt - t),
                            )

                            min_allowed_time = closest_med_time + timedelta(minutes=10)

                            if (
                                self.rng.random() < self.policy.p_bias_to_medication
                                and t >= min_allowed_time
                            ):
                                targets = _pick_targets_for_time(
                                    self.rng,
                                    closest_med_time,
                                    self.policy,
                                )

                        # fallback
                        if targets is None:
                            targets = _pick_targets_for_time(
                                self.rng,
                                t,
                                self.policy,
                            )

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