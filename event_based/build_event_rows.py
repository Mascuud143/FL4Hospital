import argparse
import csv
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from schema import (
    ACTIVE_MEDICATION_COLUMNS,
    DIAGNOSIS_COLUMNS,
    DIAGNOSIS_TO_INDEX,
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    EVENT_TYPE_COLUMNS,
    MEDICATION_NAMES,
    MEDICATION_TO_INDEX,
    SYMPTOM_COLUMNS,
    SYMPTOM_TO_INDEX,
    TARGET_COLUMNS,
    TRIGGER_MEDICATION_COLUMNS,
    gender_to_binary,
    make_one_hot,
)

DEFAULT_DATA_DIR = "filestorage"
DEFAULT_OUT_DIR = "rows"
WRITE_BATCH_SIZE = 500


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
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
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_at_or_before(times: list[datetime], rows: list[dict[str, Any]], t: datetime) -> dict[str, Any] | None:
    idx = bisect_right(times, t) - 1
    if idx < 0:
        return None
    return rows[idx]


def first_in_window(
    times: list[datetime],
    rows: list[dict[str, Any]],
    start_inclusive: datetime,
    end_exclusive: datetime,
) -> dict[str, Any] | None:
    lo = bisect_left(times, start_inclusive)
    hi = bisect_left(times, end_exclusive)
    if lo >= hi:
        return None
    return rows[lo]


def parse_bp(value: str | None) -> tuple[float | None, float | None]:
    raw = str(value or "").strip()
    if "/" not in raw:
        return None, None
    left, right = raw.split("/", 1)
    return to_float(left), to_float(right)


def _admission_for_time(admissions: list[dict[str, Any]], t: datetime) -> dict[str, Any] | None:
    for admission in admissions:
        admitted_at = admission["admitted_at"]
        discharged_at = admission["discharged_at"]
        if admitted_at <= t and (discharged_at is None or t < discharged_at):
            return admission
    return None


def _assignment_for_time(assignments: list[dict[str, Any]], t: datetime) -> dict[str, Any] | None:
    for assignment in assignments:
        if assignment["start"] <= t < assignment["end"]:
            return assignment
    return None


def _active_medication_vector(med_rows: list[dict[str, Any]], t: datetime, active_hours: int) -> list[int]:
    start = t - timedelta(hours=active_hours)
    values = [0] * len(MEDICATION_NAMES)
    for row in med_rows:
        if row["ts"] > t:
            break
        if row["ts"] < start:
            continue
        idx = MEDICATION_TO_INDEX.get(row["drug_name"])
        if idx is not None:
            values[idx] = 1
    return values


def build_indices(data_dir: str) -> dict[str, Any]:
    patients_raw = read_csv(os.path.join(data_dir, "patients.csv"))
    admissions_raw = read_csv(os.path.join(data_dir, "admissions.csv"))
    assignments_raw = read_csv(os.path.join(data_dir, "room_assignments.csv"))
    medications_raw = read_csv(os.path.join(data_dir, "medications.csv"))
    visits_raw = read_csv(os.path.join(data_dir, "visits.csv"))
    comfort_raw = read_csv(os.path.join(data_dir, "comfort_preferences.csv"))

    patients_by_id: dict[int, dict[str, Any]] = {}
    for row in patients_raw:
        patient_id = to_int(row.get("patient_id"))
        if patient_id is None:
            continue
        patients_by_id[patient_id] = {
            "height": to_float(row.get("height")),
            "gender": (row.get("gender") or "").strip(),
        }

    admissions_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    admissions_by_id: dict[int, dict[str, Any]] = {}
    for row in admissions_raw:
        admission_id = to_int(row.get("admission_id"))
        patient_id = to_int(row.get("patient_id"))
        admitted_at = parse_ts(row.get("admitted_at"))
        if admission_id is None or patient_id is None or admitted_at is None:
            continue
        admission = {
            "admission_id": admission_id,
            "patient_id": patient_id,
            "admitted_at": admitted_at,
            "discharged_at": parse_ts(row.get("discharged_at")),
            "age": to_float(row.get("age")),
            "weight": to_float(row.get("weight")),
            "diagnosis": (row.get("current_diagnosis") or "").strip(),
        }
        admissions_by_patient[patient_id].append(admission)
        admissions_by_id[admission_id] = admission
    for admissions in admissions_by_patient.values():
        admissions.sort(key=lambda item: item["admitted_at"])

    assignments_by_admission: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments_raw:
        admission_id = to_int(row.get("admission_id"))
        patient_id = to_int(row.get("patient_id"))
        room_id = to_int(row.get("room_id"))
        start = parse_ts(row.get("start_time"))
        end = parse_ts(row.get("end_time"))
        if admission_id is None or patient_id is None or room_id is None or start is None or end is None:
            continue
        assignments_by_admission[admission_id].append(
            {
                "admission_id": admission_id,
                "patient_id": patient_id,
                "room_id": room_id,
                "start": start,
                "end": end,
            }
        )
    for assignments in assignments_by_admission.values():
        assignments.sort(key=lambda item: item["start"])

    eligible_admission_ids = set(assignments_by_admission.keys())

    meds_by_admission: dict[int, list[dict[str, Any]]] = defaultdict(list)
    med_times_by_admission: dict[int, list[datetime]] = {}
    for row in medications_raw:
        patient_id = to_int(row.get("patient_id"))
        ts = parse_ts(row.get("medication_time"))
        if patient_id is None or ts is None:
            continue
        admission = _admission_for_time(admissions_by_patient.get(patient_id, []), ts)
        if admission is None or admission["admission_id"] not in eligible_admission_ids:
            continue
        meds_by_admission[admission["admission_id"]].append(
            {
                "ts": ts,
                "drug_name": (row.get("drug_name") or "").strip(),
                "status": (row.get("status") or "").strip(),
                "route": (row.get("route") or "").strip(),
                "dose": (row.get("dose") or "").strip(),
            }
        )
    for admission_id, rows in meds_by_admission.items():
        rows.sort(key=lambda item: item["ts"])
        med_times_by_admission[admission_id] = [row["ts"] for row in rows]

    visits_by_admission: dict[int, list[dict[str, Any]]] = defaultdict(list)
    visit_times_by_admission: dict[int, list[datetime]] = {}
    visit_events_by_admission: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in visits_raw:
        patient_id = to_int(row.get("patient_id"))
        ts = parse_ts(row.get("visit_time"))
        if patient_id is None or ts is None:
            continue
        admission = _admission_for_time(admissions_by_patient.get(patient_id, []), ts)
        if admission is None or admission["admission_id"] not in eligible_admission_ids:
            continue
        visits_by_admission[admission["admission_id"]].append(
            {
                "ts": ts,
                "symptoms": (row.get("symptoms") or "").strip(),
                "body_temperature": to_float(row.get("body_temperature")),
                "blood_pressure": (row.get("blood_pressure") or "").strip(),
            }
        )
    for admission_id, rows in visits_by_admission.items():
        rows.sort(key=lambda item: item["ts"])
        visit_times_by_admission[admission_id] = [row["ts"] for row in rows]
        for row in rows:
            visit_events_by_admission[admission_id].append(
                {"ts": row["ts"], "event_type": "visit", "detail": (row["symptoms"] or "").strip()}
            )

    comfort_by_admission: dict[int, list[dict[str, Any]]] = defaultdict(list)
    comfort_times_by_admission: dict[int, list[datetime]] = {}
    for row in comfort_raw:
        patient_id = to_int(row.get("patient_id"))
        room_id = to_int(row.get("room_id"))
        ts = parse_ts(row.get("timestamp"))
        if patient_id is None or room_id is None or ts is None:
            continue
        admission = _admission_for_time(admissions_by_patient.get(patient_id, []), ts)
        if admission is None or admission["admission_id"] not in eligible_admission_ids:
            continue
        comfort_by_admission[admission["admission_id"]].append(
            {
                "ts": ts,
                "patient_id": patient_id,
                "room_id": room_id,
                "temperature_main": to_float(row.get("temperature_main")),
                "temperature_toilet": to_float(row.get("temperature_toilet")),
                "light_intensity": to_float(row.get("light_intensity")),
                "sound_level": to_float(row.get("sound_level")),
                "airflow": to_bool(row.get("airflow")),
            }
        )
    for admission_id, rows in comfort_by_admission.items():
        rows.sort(key=lambda item: item["ts"])
        comfort_times_by_admission[admission_id] = [row["ts"] for row in rows]

    events_by_admission: dict[int, list[dict[str, Any]]] = defaultdict(list)
    event_times_by_admission: dict[int, list[datetime]] = {}
    for admission_id in eligible_admission_ids:
        for med_row in meds_by_admission.get(admission_id, []):
            events_by_admission[admission_id].append(
                {"ts": med_row["ts"], "event_type": "medication", "detail": med_row["drug_name"]}
            )
        events_by_admission[admission_id].extend(visit_events_by_admission.get(admission_id, []))
        events_by_admission[admission_id].sort(key=lambda item: (item["ts"], item["event_type"], item["detail"]))
        event_times_by_admission[admission_id] = [row["ts"] for row in events_by_admission[admission_id]]

    return {
        "patients_by_id": patients_by_id,
        "admissions_by_id": admissions_by_id,
        "assignments_by_admission": assignments_by_admission,
        "eligible_admission_ids": eligible_admission_ids,
        "meds_by_admission": meds_by_admission,
        "med_times_by_admission": med_times_by_admission,
        "visits_by_admission": visits_by_admission,
        "visit_times_by_admission": visit_times_by_admission,
        "comfort_by_admission": comfort_by_admission,
        "comfort_times_by_admission": comfort_times_by_admission,
        "events_by_admission": events_by_admission,
        "event_times_by_admission": event_times_by_admission,
    }


def build_rows(
    indices: dict[str, Any],
    out_dir: str,
    medication_effect_minutes: int,
    medication_active_hours: int,
) -> tuple[int, int]:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "event_based_rows.csv")
    fieldnames = [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, *TARGET_COLUMNS]

    row_count = 0
    filtered_events = 0

    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        pending_rows: list[dict[str, Any]] = []

        for admission_id in sorted(indices["eligible_admission_ids"]):
            admission = indices["admissions_by_id"].get(admission_id)
            assignments = indices["assignments_by_admission"].get(admission_id, [])
            if admission is None or not assignments:
                continue
            patient_id = admission["patient_id"]
            patient = indices["patients_by_id"].get(patient_id, {})
            med_rows = indices["meds_by_admission"].get(admission_id, [])
            med_times = indices["med_times_by_admission"].get(admission_id, [])
            visit_rows = indices["visits_by_admission"].get(admission_id, [])
            visit_times = indices["visit_times_by_admission"].get(admission_id, [])
            comfort_rows = indices["comfort_by_admission"].get(admission_id, [])
            comfort_times = indices["comfort_times_by_admission"].get(admission_id, [])
            events = indices["events_by_admission"].get(admission_id, [])
            event_times = indices["event_times_by_admission"].get(admission_id, [])

            if not comfort_rows or not events:
                continue

            for idx, event in enumerate(events):
                event_time = event["ts"]
                assignment = _assignment_for_time(assignments, event_time)
                if assignment is None:
                    continue
                room_id = assignment["room_id"]

                next_change_time = assignment["end"]
                if idx + 1 < len(event_times):
                    next_change_time = min(next_change_time, event_times[idx + 1])

                effect_minutes = medication_effect_minutes if event["event_type"] == "medication" else 0
                effect_window_start = event_time + timedelta(minutes=effect_minutes)
                if effect_window_start >= next_change_time:
                    filtered_events += 1
                    continue

                prev_target = latest_at_or_before(comfort_times, comfort_rows, event_time)
                target_row = first_in_window(comfort_times, comfort_rows, effect_window_start, next_change_time)
                if target_row is None:
                    target_row = prev_target
                if target_row is None:
                    filtered_events += 1
                    continue

                latest_visit = latest_at_or_before(visit_times, visit_rows, event_time)
                latest_med = latest_at_or_before(med_times, med_rows, event_time)
                current_symptom = latest_visit["symptoms"] if latest_visit else ""
                bp_systolic, bp_diastolic = parse_bp(latest_visit["blood_pressure"] if latest_visit else None)

                hours_since_last_medication = None
                if latest_med is not None:
                    hours_since_last_medication = (event_time - latest_med["ts"]).total_seconds() / 3600.0

                hours_since_last_symptom_change = None
                if latest_visit is not None:
                    hours_since_last_symptom_change = (event_time - latest_visit["ts"]).total_seconds() / 3600.0

                hours_since_last_comfort = None
                if prev_target is not None:
                    hours_since_last_comfort = (event_time - prev_target["ts"]).total_seconds() / 3600.0

                row: dict[str, Any] = {
                    "admission_id": admission_id,
                    "patient_id": patient_id,
                    "room_id": room_id,
                    "event_time": event_time.isoformat(),
                    "event_type": event["event_type"],
                    "event_detail": event["detail"],
                    "effect_window_start": effect_window_start.isoformat(),
                    "next_change_time": next_change_time.isoformat(),
                    "target_time": target_row["ts"].isoformat(),
                    "age": admission["age"],
                    "height": patient.get("height"),
                    "weight": admission["weight"],
                    "gender_binary": gender_to_binary(patient.get("gender")),
                    "body_temperature": latest_visit["body_temperature"] if latest_visit else None,
                    "bp_systolic": bp_systolic,
                    "bp_diastolic": bp_diastolic,
                    "hours_since_last_medication": hours_since_last_medication,
                    "hours_since_last_symptom_change": hours_since_last_symptom_change,
                    "hours_since_last_comfort": hours_since_last_comfort,
                    "prev_target_temp_main": prev_target["temperature_main"] if prev_target else None,
                    "prev_target_temp_toilet": prev_target["temperature_toilet"] if prev_target else None,
                    "prev_target_light": prev_target["light_intensity"] if prev_target else None,
                    "prev_target_sound": prev_target["sound_level"] if prev_target else None,
                    "prev_target_airflow": prev_target["airflow"] if prev_target else None,
                    "event_is_medication": 1 if event["event_type"] == "medication" else 0,
                    "event_is_visit": 1 if event["event_type"] == "visit" else 0,
                    "y_target_temp_main": target_row["temperature_main"],
                    "y_target_temp_toilet": target_row["temperature_toilet"],
                    "y_target_light": target_row["light_intensity"],
                    "y_target_sound": target_row["sound_level"],
                    "y_target_airflow": target_row["airflow"],
                }

                diagnosis_vector = make_one_hot(DIAGNOSIS_TO_INDEX.get(admission["diagnosis"]), len(DIAGNOSIS_COLUMNS))
                for col, value in zip(DIAGNOSIS_COLUMNS, diagnosis_vector):
                    row[col] = value

                symptom_vector = make_one_hot(SYMPTOM_TO_INDEX.get(current_symptom, SYMPTOM_TO_INDEX[""]), len(SYMPTOM_COLUMNS))
                for col, value in zip(SYMPTOM_COLUMNS, symptom_vector):
                    row[col] = value

                active_med_vector = _active_medication_vector(med_rows, event_time, medication_active_hours)
                for col, value in zip(ACTIVE_MEDICATION_COLUMNS, active_med_vector):
                    row[col] = value

                trigger_vector = [0] * len(MEDICATION_NAMES)
                if event["event_type"] == "medication":
                    med_idx = MEDICATION_TO_INDEX.get(event["detail"])
                    if med_idx is not None:
                        trigger_vector[med_idx] = 1
                for col, value in zip(TRIGGER_MEDICATION_COLUMNS, trigger_vector):
                    row[col] = value

                pending_rows.append(row)
                if len(pending_rows) >= WRITE_BATCH_SIZE:
                    writer.writerows(pending_rows)
                    pending_rows = []
                row_count += 1

        if pending_rows:
            writer.writerows(pending_rows)

    return row_count, filtered_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build state-to-outcome event rows from meaningful medication and symptom changes.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing source CSV files.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for Task #2 event rows.")
    parser.add_argument("--medication-effect-minutes", type=int, default=5, help="Delay before medication events can receive a label.")
    parser.add_argument("--medication-active-hours", type=int, default=24, help="Lookback window used to mark active medications.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(project_root, args.data_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(script_dir, args.out_dir)
    indices = build_indices(data_dir)
    row_count, filtered_events = build_rows(
        indices=indices,
        out_dir=out_dir,
        medication_effect_minutes=args.medication_effect_minutes,
        medication_active_hours=args.medication_active_hours,
    )
    print("build_event_rows.py complete")
    print(f"eligible_admissions={len(indices['eligible_admission_ids'])}")
    print(f"rows={row_count}")
    print(f"filtered_events={filtered_events}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
