from __future__ import annotations

# Build comfort rows.
# - Stores comfort rules - ComfortPolicy
# - Picks base comfort targets - _pick_targets_for_time()
# - Changes targets with visit symptoms - _apply_visit_symptom_bias()
# - Builds comfort rows - ComfortGenerator

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from persistence.database import session_scope
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.medication import Medication
from persistence.models.room_assignment import RoomAssignment
from persistence.models.visit import Visit
from simulation_batch.csv_storage import write_model_rows
from simulation_batch.simulation_steps.room_simulation import as_utc


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
    p_bias_to_medication: float = 0.6
    fever_cool_target_range: Tuple[float, float] = (10.0, 20.0)
    chills_warm_target_range: Tuple[float, float] = (23.0, 30.0)
    p_apply_symptom_bias: float = 0.85
    symptom_bias_before_visit_minutes: int = 60
    symptom_bias_after_visit_minutes: int = 60


def _day_bounds(t: datetime) -> Tuple[datetime, datetime]:
    t = as_utc(t)
    start = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _random_times_in_day(rng: random.Random, day_start: datetime, *, k: int) -> List[datetime]:
    if k <= 0:
        return []
    times = [day_start + timedelta(seconds=rng.randint(0, 24 * 60 * 60 - 1)) for _ in range(k)]
    times.sort()
    return times


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _round_temp(value: float) -> float:
    return round(float(value), 1)


def _symptom_flags(symptoms: Optional[str]) -> Dict[str, bool]:
    symptoms = (symptoms or "").lower()
    return {
        "fever": any(key in symptoms for key in ["fever", "high temperature", "pyrexia"]),
        "chills": any(key in symptoms for key in ["chills", "shivering", "rigors"]),
        "cold": any(key in symptoms for key in ["cold", "feeling cold"]),
        "sweating": any(key in symptoms for key in ["sweat", "sweating"]),
        "cough": "cough" in symptoms,
        "resp": any(key in symptoms for key in ["shortness of breath", "dyspnea", "breathless"]),
    }


def _apply_visit_symptom_bias(rng: random.Random, *, base_targets: Dict, visit: Optional[Visit], policy: ComfortPolicy) -> Dict:
    if visit is None or rng.random() > policy.p_apply_symptom_bias:
        return base_targets
    flags = _symptom_flags(getattr(visit, "symptoms", None))
    body_temp = getattr(visit, "body_temperature", None)
    fever = flags["fever"] or (isinstance(body_temp, (int, float)) and body_temp >= 38.0)
    chills = flags["chills"] or flags["cold"]
    out = dict(base_targets)
    if fever:
        out["temperature_main"] = _round_temp(rng.uniform(*policy.fever_cool_target_range))
        out["airflow"] = True
    elif chills:
        out["temperature_main"] = _round_temp(rng.uniform(*policy.chills_warm_target_range))
        out["airflow"] = False
    out["temperature_main"] = _round_temp(_clamp(out["temperature_main"], policy.min_main_temp_c, policy.max_main_temp_c))
    return out


def _pick_targets_for_time(rng: random.Random, t: datetime, policy: ComfortPolicy) -> Dict:
    t = as_utc(t)
    hour = t.hour
    p_airflow = policy.p_airflow
    if 6 <= hour < 18:
        p_airflow += policy.p_airflow_day_extra
    elif 18 <= hour < 24:
        p_airflow += policy.p_airflow_evening_extra
    stuffy = 8 <= hour < 22 and rng.random() < policy.p_stuffy_event
    airflow = rng.random() < (policy.p_airflow_if_stuffy if stuffy else p_airflow)
    if 0 <= hour < 6:
        t_main_base, light, sound = rng.uniform(20.0, 22.0), 0.0, rng.randint(0, 20)
    elif 6 <= hour < 12:
        t_main_base, light, sound = rng.uniform(21.0, 23.0), rng.randint(10, 40), rng.randint(10, 35)
    elif 12 <= hour < 18:
        t_main_base, light, sound = rng.uniform(22.0, 24.0), rng.randint(20, 60), rng.randint(15, 45)
    else:
        t_main_base, light, sound = rng.uniform(20.0, 22.5), rng.randint(5, 35), rng.randint(0, 25)
    t_main = _round_temp(_clamp(t_main_base + rng.gauss(0.0, policy.temp_adjust_sigma_c), policy.min_main_temp_c, policy.max_main_temp_c))
    t_toilet: Optional[float] = None
    if rng.random() < policy.p_set_toilet_temp:
        t_toilet = _round_temp(_clamp(rng.uniform(19.0, 23.0) + rng.gauss(0.0, 0.2), 18.0, 24.0))
    return {"temperature_main": t_main, "temperature_toilet": t_toilet, "light_intensity": round(light, 2), "sound_level": round(sound, 2), "airflow": airflow}


class ComfortGenerator:
    def __init__(self, *, seed: int = 42, policy: Optional[ComfortPolicy] = None):
        self.rng = random.Random(seed)
        self.policy = policy or ComfortPolicy()
        self.write_batch_size = 50000

    def generate_for_horizon(self, start_time: datetime, end_time: datetime) -> int:
        start_time = as_utc(start_time)
        end_time = as_utc(end_time)
        inserted = 0
        pending_rows: List[Tuple[int, int, datetime, float, Optional[float], float, float, bool, str]] = []

        def flush_pending(session) -> None:
            nonlocal inserted, pending_rows
            if not pending_rows:
                return
            serialized_rows = [
                {"patient_id": patient_id, "room_id": room_id, "timestamp": timestamp, "temperature_main": temperature_main, "temperature_toilet": temperature_toilet, "light_intensity": light_intensity, "sound_level": sound_level, "airflow": airflow, "source": source}
                for patient_id, room_id, timestamp, temperature_main, temperature_toilet, light_intensity, sound_level, airflow, source in pending_rows
            ]
            write_model_rows(ComfortPreference, serialized_rows)
            session.bulk_insert_mappings(ComfortPreference, serialized_rows)
            inserted += len(pending_rows)
            pending_rows = []

        with session_scope() as session:
            assigns = session.query(RoomAssignment).filter(RoomAssignment.end_time > start_time).filter(RoomAssignment.start_time < end_time).all()
            for assignment in assigns:
                w_start = max(as_utc(assignment.start_time), start_time)
                w_end = min(as_utc(assignment.end_time), end_time)
                day_cursor = w_start.replace(hour=0, minute=0, second=0, microsecond=0)
                while day_cursor < w_end:
                    day_start, day_end = _day_bounds(day_cursor)
                    meds = session.query(Medication).filter(Medication.patient_id == assignment.patient_id).filter(Medication.medication_time >= day_start).filter(Medication.medication_time < day_end).all()
                    med_times = [as_utc(med.medication_time) for med in meds]
                    visits = session.query(Visit).filter(Visit.patient_id == assignment.patient_id).filter(Visit.visit_time >= day_start).filter(Visit.visit_time < day_end).all()
                    k = self.rng.randint(0, self.policy.max_changes_per_day)
                    times = [t for t in _random_times_in_day(self.rng, day_start, k=k) if w_start <= t < w_end]
                    for t in times:
                        targets: Optional[Dict] = None
                        base_targets = _pick_targets_for_time(self.rng, t, self.policy)
                        if visits:
                            closest_visit = min(visits, key=lambda visit: abs(as_utc(visit.visit_time) - t))
                            visit_time = as_utc(closest_visit.visit_time)
                            earliest_bias_time = visit_time - timedelta(minutes=self.policy.symptom_bias_before_visit_minutes)
                            latest_bias_time = visit_time + timedelta(minutes=self.policy.symptom_bias_after_visit_minutes)
                            if earliest_bias_time <= t <= latest_bias_time:
                                targets = _apply_visit_symptom_bias(self.rng, base_targets=base_targets, visit=closest_visit, policy=self.policy)
                        if med_times:
                            closest_med_time = min(med_times, key=lambda med_time: abs(med_time - t))
                            if self.rng.random() < self.policy.p_bias_to_medication and t >= closest_med_time + timedelta(minutes=10):
                                targets = _pick_targets_for_time(self.rng, closest_med_time, self.policy)
                        if targets is None:
                            targets = base_targets
                        pending_rows.append((int(assignment.patient_id), int(assignment.room_id), t, float(targets["temperature_main"]), targets["temperature_toilet"], float(targets["light_intensity"]), float(targets["sound_level"]), bool(targets["airflow"]), "simulation"))
                        if len(pending_rows) >= self.write_batch_size:
                            flush_pending(session)
                    day_cursor += timedelta(days=1)
            flush_pending(session)
        return inserted


__all__ = ["ComfortGenerator", "ComfortPolicy"]
