import csv
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import timedelta
from typing import Any

from next_hour_schema import (
    DIAGNOSIS_COLUMNS,
    DIAGNOSIS_TO_INDEX,
    MAX_MEDICATION_SLOTS,
    MEDICATION_NAMES,
    MEDICATION_SCHEDULE_COLUMNS,
    MEDICATION_TO_INDEX,
    MEDICATION_TYPE_COLUMNS,
    SYMPTOM_COLUMNS,
    SYMPTOM_TO_INDEX,
    TIME_COLUMNS,
    make_one_hot,
    make_time_vector,
    medication_slots_for_diagnosis,
    normalize_schedule,
)


def parse_ts(value: str | None):
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = __import__("datetime").datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=__import__("datetime").timezone.utc)
    return parsed.astimezone(__import__("datetime").timezone.utc)


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    return None


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_at_or_before(times, rows, t):
    idx = bisect_right(times, t) - 1
    if idx < 0:
        return None
    return rows[idx]


def active_symptom_for_time(times, rows, t, window_minutes: int) -> str:
    if not rows:
        return ""

    delta = timedelta(minutes=window_minutes)
    start = t - delta
    end = t + delta
    lo = bisect_left(times, start)
    hi = bisect_right(times, end)
    if lo >= hi:
        return ""

    best_symptom = ""
    best_abs_seconds: float | None = None
    best_ts = None
    for idx in range(lo, hi):
        candidate = rows[idx]
        symptom = (candidate.get("symptoms") or "").strip()
        if symptom == "":
            continue
        abs_seconds = abs((candidate["ts"] - t).total_seconds())
        candidate_ts = candidate["ts"]
        if (
            best_abs_seconds is None
            or abs_seconds < best_abs_seconds
            or (abs_seconds == best_abs_seconds and (best_ts is None or candidate_ts > best_ts))
        ):
            best_abs_seconds = abs_seconds
            best_ts = candidate_ts
            best_symptom = symptom
    return best_symptom


def choose_admission(admissions_by_patient, admitted_times_by_patient, patient_id: int, t):
    admissions = admissions_by_patient.get(patient_id, [])
    if not admissions:
        return None
    admitted_times = admitted_times_by_patient[patient_id]
    idx = bisect_right(admitted_times, t) - 1
    if idx < 0:
        return admissions[0]
    candidate = admissions[idx]
    discharged_at = candidate.get("discharged_at")
    if discharged_at is None or t <= discharged_at:
        return candidate
    return candidate


def build_time_feature_cache() -> dict[tuple[int, int], dict[str, int]]:
    cache: dict[tuple[int, int], dict[str, int]] = {}
    for hour in range(24):
        for minute in (0, 30):
            cache[(hour, minute)] = dict(zip(TIME_COLUMNS, make_time_vector(hour, minute)))
    return cache


def build_symptom_feature_cache() -> dict[str, dict[str, int]]:
    cache: dict[str, dict[str, int]] = {}
    fallback_index = SYMPTOM_TO_INDEX[""]
    for symptom in SYMPTOM_TO_INDEX:
        encoded = make_one_hot(SYMPTOM_TO_INDEX.get(symptom, fallback_index), len(SYMPTOM_COLUMNS))
        cache[symptom] = dict(zip(SYMPTOM_COLUMNS, encoded))
    return cache


def build_diagnosis_feature_cache() -> dict[str, dict[str, Any]]:
    empty_type_vector = [0] * len(MEDICATION_NAMES)
    empty_schedule_vector = [0] * len(TIME_COLUMNS)
    cache: dict[str, dict[str, Any]] = {}
    for diagnosis in DIAGNOSIS_TO_INDEX:
        diagnosis_vector = make_one_hot(DIAGNOSIS_TO_INDEX.get(diagnosis), len(DIAGNOSIS_COLUMNS))
        med_slots = medication_slots_for_diagnosis(diagnosis)
        medication_features: dict[str, int] = {}
        for slot in range(1, MAX_MEDICATION_SLOTS + 1):
            if slot <= len(med_slots):
                med_name, med_hours = med_slots[slot - 1]
                type_vector = make_one_hot(MEDICATION_TO_INDEX.get(med_name), len(MEDICATION_NAMES))
                schedule_vector = normalize_schedule(med_hours)
            else:
                type_vector = empty_type_vector
                schedule_vector = empty_schedule_vector
            medication_features.update(zip(MEDICATION_TYPE_COLUMNS[slot], type_vector))
            medication_features.update(zip(MEDICATION_SCHEDULE_COLUMNS[slot], schedule_vector))
        cache[diagnosis] = {
            "diagnosis": dict(zip(DIAGNOSIS_COLUMNS, diagnosis_vector)),
            "medication": medication_features,
            "med_slots": med_slots,
        }
    return cache


def build_indices(data_dir: str) -> dict[str, Any]:
    patients_raw = read_csv(os.path.join(data_dir, "patients.csv"))
    admissions_raw = read_csv(os.path.join(data_dir, "admissions.csv"))
    assignments_raw = read_csv(os.path.join(data_dir, "room_assignments.csv"))
    visits_raw = read_csv(os.path.join(data_dir, "visits.csv"))
    comfort_raw = read_csv(os.path.join(data_dir, "comfort_preferences.csv"))

    patients_by_id: dict[int, dict[str, Any]] = {}
    for row in patients_raw:
        pid = to_int(row.get("patient_id"))
        if pid is None:
            continue
        patients_by_id[pid] = {
            "height": to_float(row.get("height")),
            "gender": (row.get("gender") or "").strip(),
        }

    admissions_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in admissions_raw:
        pid = to_int(row.get("patient_id"))
        admitted_at = parse_ts(row.get("admitted_at"))
        if pid is None or admitted_at is None:
            continue
        admissions_by_patient[pid].append(
            {
                "admitted_at": admitted_at,
                "discharged_at": parse_ts(row.get("discharged_at")),
                "age": to_int(row.get("age")),
                "weight": to_float(row.get("weight")),
                "diagnosis": (row.get("current_diagnosis") or "").strip(),
            }
        )

    admitted_times_by_patient: dict[int, list] = {}
    for pid, rows in admissions_by_patient.items():
        rows.sort(key=lambda x: x["admitted_at"])
        admitted_times_by_patient[pid] = [r["admitted_at"] for r in rows]

    assignments: list[dict[str, Any]] = []
    for row in assignments_raw:
        pid = to_int(row.get("patient_id"))
        room_id = to_int(row.get("room_id"))
        start = parse_ts(row.get("start_time"))
        end = parse_ts(row.get("end_time"))
        if pid is None or room_id is None or start is None or end is None:
            continue
        if end <= start:
            continue
        assignments.append({"patient_id": pid, "room_id": room_id, "start": start, "end": end})
    assignments.sort(key=lambda x: (x["room_id"], x["patient_id"], x["start"]))

    visits_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in visits_raw:
        pid = to_int(row.get("patient_id"))
        ts = parse_ts(row.get("visit_time"))
        if pid is None or ts is None:
            continue
        visits_by_patient[pid].append(
            {
                "ts": ts,
                "symptoms": (row.get("symptoms") or "").strip(),
            }
        )

    visit_times_by_patient: dict[int, list] = {}
    for pid, rows in visits_by_patient.items():
        rows.sort(key=lambda x: x["ts"])
        visit_times_by_patient[pid] = [r["ts"] for r in rows]

    comfort_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in comfort_raw:
        room_id = to_int(row.get("room_id"))
        ts = parse_ts(row.get("timestamp"))
        if room_id is None or ts is None:
            continue
        comfort_by_room[room_id].append(
            {
                "ts": ts,
                "temperature_main": to_float(row.get("temperature_main")),
                "temperature_toilet": to_float(row.get("temperature_toilet")),
                "light_intensity": to_int(row.get("light_intensity")),
                "sound_level": to_int(row.get("sound_level")),
                "airflow": to_bool(row.get("airflow")),
            }
        )

    comfort_times_by_room: dict[int, list] = {}
    for room_id, rows in comfort_by_room.items():
        rows.sort(key=lambda x: x["ts"])
        last_toilet_temp: float | None = None
        for row in rows:
            if row["temperature_toilet"] is None:
                row["temperature_toilet"] = last_toilet_temp
            else:
                last_toilet_temp = row["temperature_toilet"]
        comfort_times_by_room[room_id] = [r["ts"] for r in rows]

    return {
        "patients_by_id": patients_by_id,
        "assignments": assignments,
        "admissions_by_patient": admissions_by_patient,
        "admitted_times_by_patient": admitted_times_by_patient,
        "visits_by_patient": visits_by_patient,
        "visit_times_by_patient": visit_times_by_patient,
        "comfort_by_room": comfort_by_room,
        "comfort_times_by_room": comfort_times_by_room,
    }


def resolve_input_path(path_value: str, project_root: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    candidate_cwd = os.path.abspath(path_value)
    if os.path.exists(candidate_cwd):
        return candidate_cwd
    return os.path.abspath(os.path.join(project_root, path_value))
