from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.medication import Medication
from persistence.models.visit import Visit  # ✅ FIX: real visits table
from persistence.models import Speaker, Device  # ✅ NEW: import Speaker and Device models

from simulation_batch.room_engine import _as_utc
from simulation_batch.csv_filestorage import write_model_row


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

    # probability of biasing comfort toward medication
    p_bias_to_medication: float = 0.6

    # ✅ NEW: symptom influence strength
    fever_cool_target_range: Tuple[float, float] = (18.0, 20.0)
    chills_warm_target_range: Tuple[float, float] = (23.0, 25.5)
    p_apply_symptom_bias: float = 0.85  # probability to apply symptom bias when visit exists


# ==========================================================
# HELPERS
# ==========================================================

def _day_bounds(t: datetime) -> Tuple[datetime, datetime]:
    t = _as_utc(t)
    start = t.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def _random_times_in_day(rng: random.Random, day_start: datetime, *, k: int) -> List[datetime]:
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


def _round_temp(value: float) -> float:
    return round(float(value), 1)


def _symptom_flags(symptoms: Optional[str]) -> Dict[str, bool]:
    """
    Very simple NLP: look for keywords in Visit.symptoms.
    You can expand this list anytime.
    """
    s = (symptoms or "").lower()
    return {
        "fever": any(k in s for k in ["fever", "high temperature", "pyrexia"]),
        "chills": any(k in s for k in ["chills", "shivering", "rigors"]),
        "cold": any(k in s for k in ["cold", "feeling cold"]),
        "sweating": any(k in s for k in ["sweat", "sweating"]),
        "cough": "cough" in s,
        "resp": any(k in s for k in ["shortness of breath", "dyspnea", "breathless"]),
    }


def _apply_visit_symptom_bias(
    rng: random.Random,
    *,
    base_targets: Dict,
    visit: Optional[Visit],
    policy: ComfortPolicy,
) -> Dict:
    """
    Modify temperature_main (and optionally airflow) based on visit symptoms/vitals.
    - fever or body_temperature >= 38 -> cooler target
    - chills/cold -> warmer target
    """
    if visit is None:
        return base_targets

    # probabilistic: sometimes patient doesn't change settings even with symptoms
    if rng.random() > policy.p_apply_symptom_bias:
        return base_targets

    flags = _symptom_flags(getattr(visit, "symptoms", None))
    body_temp = getattr(visit, "body_temperature", None)

    fever = flags["fever"] or (isinstance(body_temp, (int, float)) and body_temp >= 38.0)
    chills = flags["chills"] or flags["cold"]

    # Start with base
    out = dict(base_targets)

    if fever:
        out["temperature_main"] = _round_temp(rng.uniform(*policy.fever_cool_target_range))
        # Often want airflow when feverish
        out["airflow"] = True

    elif chills:
        out["temperature_main"] = _round_temp(rng.uniform(*policy.chills_warm_target_range))
        # airflow typically OFF when cold
        out["airflow"] = False

    # Always clamp
    out["temperature_main"] = _round_temp(
        _clamp(out["temperature_main"], policy.min_main_temp_c, policy.max_main_temp_c)
    )

    return out


# ==========================================================
# TARGET GENERATION
# ==========================================================

def _pick_targets_for_time(rng: random.Random, t: datetime, policy: ComfortPolicy) -> Dict:
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
    t_main = _round_temp(_clamp(t_main, policy.min_main_temp_c, policy.max_main_temp_c))

    light = round(light, 2)
    sound = round(sound, 2)

    t_toilet: Optional[float] = None
    if rng.random() < policy.p_set_toilet_temp:
        t_toilet = rng.uniform(19.0, 23.0)
        t_toilet += rng.gauss(0.0, 0.2)
        t_toilet = _round_temp(_clamp(t_toilet, 18.0, 24.0))

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

    def generate_for_horizon(self, start_time: datetime, end_time: datetime) -> int:
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

                day_cursor = w_start.replace(hour=0, minute=0, second=0, microsecond=0)

                while day_cursor < w_end:
                    day_start, day_end = _day_bounds(day_cursor)

                    # meds for this patient in this day
                    meds = (
                        session.query(Medication)
                        .filter(Medication.patient_id == a.patient_id)
                        .filter(Medication.medication_time >= day_start)
                        .filter(Medication.medication_time < day_end)
                        .all()
                    )
                    med_times = [_as_utc(m.medication_time) for m in meds]

                    # ✅ FIX: visits are from Visit model, not ComfortPreference
                    visits = (
                        session.query(Visit)
                        .filter(Visit.patient_id == a.patient_id)
                        .filter(Visit.visit_time >= day_start)
                        .filter(Visit.visit_time < day_end)
                        .all()
                    )
                    visit_times = [_as_utc(v.visit_time) for v in visits]

                    k = self.rng.randint(0, self.policy.max_changes_per_day)
                    times = _random_times_in_day(self.rng, day_start, k=k)
                    times = [t for t in times if w_start <= t < w_end]

                    for t in times:
                        targets: Optional[Dict] = None

                        # -------------------------
                        # 1) Base targets (normal)
                        # -------------------------
                        base_targets = _pick_targets_for_time(self.rng, t, self.policy)

                        # -------------------------
                        # 2) Symptom-based bias using closest visit
                        # -------------------------
                        closest_visit: Optional[Visit] = None
                        if visits:
                            # closest visit by absolute time distance
                            closest_visit = min(
                                visits,
                                key=lambda v: abs(_as_utc(v.visit_time) - t),
                            )

                            # optional: enforce "only apply after 10 minutes of visit"
                            visit_time = _as_utc(closest_visit.visit_time)

                            earliest_bias_time = visit_time - timedelta(minutes=20)
                            latest_bias_time   = visit_time - timedelta(minutes=10)

                            if earliest_bias_time <= t <= latest_bias_time:
                                targets = _apply_visit_symptom_bias(
                                    self.rng,
                                    base_targets=base_targets,
                                    visit=closest_visit,
                                    policy=self.policy,
                                )


                        # -------------------------
                        # 3) Medication time bias (your old logic)
                        # -------------------------
                        if med_times:
                            closest_med_time = min(med_times, key=lambda mt: abs(mt - t))
                            min_allowed_time_med = closest_med_time + timedelta(minutes=10)

                            if (
                                self.rng.random() < self.policy.p_bias_to_medication
                                and t >= min_allowed_time_med
                            ):
                                targets = _pick_targets_for_time(self.rng, closest_med_time, self.policy)

                        # -------------------------
                        # 4) Final: pick medication targets OR symptom-adjusted base targets
                        # -------------------------
                        if targets is None:
                            targets = base_targets

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

                        write_model_row(row)
                        session.add(row)
                        #session.add(speaker_row)
                        inserted += 1

                    day_cursor += timedelta(days=1)

        return inserted
