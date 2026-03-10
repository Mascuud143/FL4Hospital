import argparse
import csv
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_DATA_DIR = "filestorage"
DEFAULT_OUT_DIR = "outputs"


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def latest_at_or_before(times: list[datetime], rows: list[dict[str, Any]], t: datetime) -> dict[str, Any] | None:
    idx = bisect_right(times, t) - 1
    if idx < 0:
        return None
    return rows[idx]


def first_in_window(
    times: list[datetime],
    rows: list[dict[str, Any]],
    start_exclusive: datetime,
    end_inclusive: datetime,
) -> dict[str, Any] | None:
    lo = bisect_right(times, start_exclusive)
    hi = bisect_right(times, end_inclusive)
    if lo >= hi:
        return None
    return rows[lo]


def active_symptom_for_time(
    times: list[datetime],
    rows: list[dict[str, Any]],
    t: datetime,
    window_minutes: int,
) -> str:
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
    best_ts: datetime | None = None
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


def choose_admission(
    admissions_by_patient: dict[int, list[dict[str, Any]]],
    admitted_times_by_patient: dict[int, list[datetime]],
    patient_id: int,
    t: datetime,
) -> dict[str, Any] | None:
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


def build_indices(data_dir: str) -> dict[str, Any]:
    patients_raw = read_csv(os.path.join(data_dir, "patients.csv"))
    admissions_raw = read_csv(os.path.join(data_dir, "admissions.csv"))
    assignments_raw = read_csv(os.path.join(data_dir, "room_assignments.csv"))
    meds_raw = read_csv(os.path.join(data_dir, "medications.csv"))
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
            "ethnicity": (row.get("ethnicity") or "").strip(),
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
                "diagnosis": (row.get("current_diagnosis") or "").strip(),
            }
        )

    admitted_times_by_patient: dict[int, list[datetime]] = {}
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

    meds_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in meds_raw:
        pid = to_int(row.get("patient_id"))
        ts = parse_ts(row.get("medication_time"))
        if pid is None or ts is None:
            continue
        meds_by_patient[pid].append(
            {
                "ts": ts,
                "drug_name": (row.get("drug_name") or "").strip(),
                "status": (row.get("status") or "").strip(),
                "route": (row.get("route") or "").strip(),
                "dose": (row.get("dose") or "").strip(),
            }
        )

    med_times_by_patient: dict[int, list[datetime]] = {}
    for pid, rows in meds_by_patient.items():
        rows.sort(key=lambda x: x["ts"])
        med_times_by_patient[pid] = [r["ts"] for r in rows]

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
                "body_temperature": to_float(row.get("body_temperature")),
                "blood_pressure": (row.get("blood_pressure") or "").strip(),
            }
        )

    visit_times_by_patient: dict[int, list[datetime]] = {}
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

    comfort_times_by_room: dict[int, list[datetime]] = {}
    change_events_by_room: dict[int, list[dict[str, Any]]] = {}
    change_event_times_by_room: dict[int, list[datetime]] = {}

    for room_id, rows in comfort_by_room.items():
        rows.sort(key=lambda x: x["ts"])
        # Forward-fill toilet temperature within each room timeline.
        # If a row has null toilet temp, keep previous known toilet temp.
        last_toilet_temp: float | None = None
        for row in rows:
            if row["temperature_toilet"] is None:
                row["temperature_toilet"] = last_toilet_temp
            else:
                last_toilet_temp = row["temperature_toilet"]
        comfort_times_by_room[room_id] = [r["ts"] for r in rows]
        events: list[dict[str, Any]] = []
        previous_state: tuple[Any, ...] | None = None
        for row in rows:
            state = (
                row["temperature_main"],
                row["temperature_toilet"],
                row["light_intensity"],
                row["sound_level"],
                row["airflow"],
            )
            if previous_state is None or state != previous_state:
                events.append(row)
            previous_state = state
        change_events_by_room[room_id] = events
        change_event_times_by_room[room_id] = [r["ts"] for r in events]

    return {
        "patients_by_id": patients_by_id,
        "assignments": assignments,
        "admissions_by_patient": admissions_by_patient,
        "admitted_times_by_patient": admitted_times_by_patient,
        "meds_by_patient": meds_by_patient,
        "med_times_by_patient": med_times_by_patient,
        "visits_by_patient": visits_by_patient,
        "visit_times_by_patient": visit_times_by_patient,
        "comfort_by_room": comfort_by_room,
        "comfort_times_by_room": comfort_times_by_room,
        "change_events_by_room": change_events_by_room,
        "change_event_times_by_room": change_event_times_by_room,
    }


def build_rows(
    indices: dict[str, Any],
    out_dir: str,
    step_minutes: int,
    horizon_minutes: int,
    max_assignments: int | None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_a_path = os.path.join(out_dir, "model_a_rows.csv")
    model_b_path = os.path.join(out_dir, "model_b_rows.csv")

    model_a_fields = [
        "client_id",
        "room_id",
        "patient_id",
        "t",
        "hour",
        "weekday",
        "day_of_stay",
        "age",
        "height",
        "gender",
        "ethnicity",
        "diagnosis",
        "latest_symptom",
        "minutes_since_last_med",
        "last_medication",
        "last_med_status",
        "curr_temp_main",
        "curr_temp_toilet",
        "curr_light",
        "curr_sound",
        "curr_airflow",
        "minutes_since_last_change",
        "y_event",
    ]

    model_b_fields = model_a_fields[:-1] + [
        "y_temp_main",
        "y_temp_toilet",
        "y_light",
        "y_sound",
        "y_airflow",
        "y_when_minutes",
    ]

    patients_by_id = indices["patients_by_id"]
    assignments = indices["assignments"]
    admissions_by_patient = indices["admissions_by_patient"]
    admitted_times_by_patient = indices["admitted_times_by_patient"]
    meds_by_patient = indices["meds_by_patient"]
    med_times_by_patient = indices["med_times_by_patient"]
    visits_by_patient = indices["visits_by_patient"]
    visit_times_by_patient = indices["visit_times_by_patient"]
    comfort_by_room = indices["comfort_by_room"]
    comfort_times_by_room = indices["comfort_times_by_room"]
    change_events_by_room = indices["change_events_by_room"]
    change_event_times_by_room = indices["change_event_times_by_room"]

    step = timedelta(minutes=step_minutes)
    horizon = timedelta(minutes=horizon_minutes)

    assignment_count = 0
    a_rows = 0
    b_rows = 0

    with open(model_a_path, "w", encoding="utf-8", newline="") as fa, open(
        model_b_path, "w", encoding="utf-8", newline=""
    ) as fb:
        writer_a = csv.DictWriter(fa, fieldnames=model_a_fields)
        writer_b = csv.DictWriter(fb, fieldnames=model_b_fields)
        writer_a.writeheader()
        writer_b.writeheader()

        for assignment in assignments:
            if max_assignments is not None and assignment_count >= max_assignments:
                break
            assignment_count += 1

            pid = assignment["patient_id"]
            room_id = assignment["room_id"]
            t = assignment["start"]
            end = assignment["end"]

            patient_profile = patients_by_id.get(pid, {})
            med_rows = meds_by_patient.get(pid, [])
            med_times = med_times_by_patient.get(pid, [])
            visit_rows = visits_by_patient.get(pid, [])
            visit_times = visit_times_by_patient.get(pid, [])
            comfort_rows = comfort_by_room.get(room_id, [])
            comfort_times = comfort_times_by_room.get(room_id, [])
            change_rows = change_events_by_room.get(room_id, [])
            change_times = change_event_times_by_room.get(room_id, [])

            while t < end:
                admission = choose_admission(admissions_by_patient, admitted_times_by_patient, pid, t)
                last_med = latest_at_or_before(med_times, med_rows, t) if med_rows else None
                active_symptom = active_symptom_for_time(
                    visit_times,
                    visit_rows,
                    t,
                    window_minutes=60,
                )
                current_comfort = latest_at_or_before(comfort_times, comfort_rows, t) if comfort_rows else None
                next_change = (
                    first_in_window(change_times, change_rows, t, t + horizon) if change_rows else None
                )

                if current_comfort is not None:
                    current_change = latest_at_or_before(change_times, change_rows, t)
                    if current_change is not None:
                        mins_last_change = round((t - current_change["ts"]).total_seconds() / 60.0, 2)
                    else:
                        mins_last_change = None
                else:
                    mins_last_change = None

                row_base: dict[str, Any] = {
                    "client_id": room_id,
                    "room_id": room_id,
                    "patient_id": pid,
                    "t": t.isoformat(),
                    "hour": t.hour,
                    "weekday": t.weekday(),
                    "day_of_stay": ((t - assignment["start"]).days + 1),
                    "age": admission.get("age") if admission else None,
                    "height": patient_profile.get("height"),
                    "gender": patient_profile.get("gender", ""),
                    "ethnicity": patient_profile.get("ethnicity", ""),
                    "diagnosis": admission.get("diagnosis") if admission else "",
                    "latest_symptom": active_symptom,
                    "minutes_since_last_med": (
                        round((t - last_med["ts"]).total_seconds() / 60.0, 2) if last_med else None
                    ),
                    "last_medication": last_med.get("drug_name") if last_med else "",
                    "last_med_status": last_med.get("status") if last_med else "",
                    "curr_temp_main": current_comfort.get("temperature_main") if current_comfort else None,
                    "curr_temp_toilet": current_comfort.get("temperature_toilet") if current_comfort else None,
                    "curr_light": current_comfort.get("light_intensity") if current_comfort else None,
                    "curr_sound": current_comfort.get("sound_level") if current_comfort else None,
                    "curr_airflow": current_comfort.get("airflow") if current_comfort else None,
                    "minutes_since_last_change": mins_last_change,
                }

                if next_change is None:
                    row_a = dict(row_base)
                    row_a["y_event"] = 0
                    writer_a.writerow(row_a)
                    a_rows += 1
                else:
                    row_a = dict(row_base)
                    row_a["y_event"] = 1
                    writer_a.writerow(row_a)
                    a_rows += 1

                    row_b = dict(row_base)
                    row_b["y_temp_main"] = next_change.get("temperature_main")
                    row_b["y_temp_toilet"] = next_change.get("temperature_toilet")
                    row_b["y_light"] = next_change.get("light_intensity")
                    row_b["y_sound"] = next_change.get("sound_level")
                    row_b["y_airflow"] = next_change.get("airflow")
                    row_b["y_when_minutes"] = round((next_change["ts"] - t).total_seconds() / 60.0, 2)
                    writer_b.writerow(row_b)
                    b_rows += 1

                t += step

    print("build_row.py complete")
    print(f"assignments_processed={assignment_count}")
    print(f"model_a_rows={a_rows}")
    print(f"model_b_rows={b_rows}")
    print(f"model_a_path={model_a_path}")
    print(f"model_b_path={model_b_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build federated-ready comfort rows from filestorage CSVs.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to source CSV directory.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Path to output directory.")
    parser.add_argument("--step-minutes", type=int, default=30, help="Decision interval in minutes.")
    parser.add_argument("--horizon-minutes", type=int, default=30, help="Prediction look-ahead window in minutes.")
    parser.add_argument(
        "--max-assignments",
        type=int,
        default=None,
        help="Optional cap for faster dry-runs (e.g., 50).",
    )
    return parser.parse_args()


def resolve_input_path(path_value: str, project_root: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    candidate_cwd = os.path.abspath(path_value)
    if os.path.exists(candidate_cwd):
        return candidate_cwd
    return os.path.abspath(os.path.join(project_root, path_value))


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = resolve_input_path(args.data_dir, project_root)
    if os.path.isabs(args.out_dir):
        out_dir = args.out_dir
    elif args.out_dir == DEFAULT_OUT_DIR:
        out_dir = os.path.join(script_dir, args.out_dir)
    else:
        out_dir = os.path.abspath(args.out_dir)

    indices = build_indices(data_dir)
    build_rows(
        indices=indices,
        out_dir=out_dir,
        step_minutes=args.step_minutes,
        horizon_minutes=args.horizon_minutes,
        max_assignments=args.max_assignments,
    )


if __name__ == "__main__":
    main()
