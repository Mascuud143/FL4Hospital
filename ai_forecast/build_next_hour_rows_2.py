import argparse
import csv
import os
from bisect import bisect_left
from datetime import datetime, timedelta
from typing import Any

from build_row import (
    DEFAULT_DATA_DIR,
    active_symptom_for_time,
    build_indices,
    choose_admission,
    latest_at_or_before,
    resolve_input_path,
)
from next_hour_schema import (
    CHANGE_METADATA_COLUMNS,
    DIAGNOSIS_COLUMNS,
    DIAGNOSIS_TO_INDEX,
    INPUT_COLUMNS,
    MAX_MEDICATION_SLOTS,
    MEDICATION_NAMES,
    MEDICATION_SCHEDULE_COLUMNS,
    MEDICATION_TO_INDEX,
    MEDICATION_TYPE_COLUMNS,
    SYMPTOM_COLUMNS,
    SYMPTOM_TO_INDEX,
    TARGET_COLUMNS,
    TIME_COLUMNS,
    gender_to_binary,
    make_one_hot,
    make_time_vector,
    medication_slots_for_diagnosis,
    normalize_schedule,
)

DEFAULT_OUT_DIR = "outputs_next_hour_2"


def times_in_window(times: list[datetime], start: datetime, end: datetime) -> list[datetime]:
    lo = bisect_left(times, start)
    hi = bisect_left(times, end)
    return times[lo:hi]


def ceil_to_step(ts: datetime, step: timedelta) -> datetime:
    step_seconds = int(step.total_seconds())
    if step_seconds <= 0:
        return ts
    epoch_seconds = int(ts.timestamp())
    remainder = epoch_seconds % step_seconds
    if remainder == 0:
        return ts.replace(second=0, microsecond=0) if step_seconds >= 60 else ts.replace(microsecond=0)
    rounded = epoch_seconds + (step_seconds - remainder)
    return datetime.fromtimestamp(rounded, tz=ts.tzinfo)


def stepped_range(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    if end < start:
        return []
    current = ceil_to_step(start, step)
    values: list[datetime] = []
    while current <= end:
        values.append(current)
        current += step
    return values


def candidate_times_for_assignment(
    start: datetime,
    end: datetime,
    visit_times: list[datetime],
    medication_times: list[datetime],
    before_minutes: int,
    after_minutes: int,
    sample_minutes: int,
) -> list[datetime]:
    candidates = {start}
    before = timedelta(minutes=before_minutes)
    after = timedelta(minutes=after_minutes)
    step = timedelta(minutes=sample_minutes)
    for ts in times_in_window(visit_times, start - after, end + before):
        candidates.update(stepped_range(ts - before, ts + after, step))
    for ts in times_in_window(medication_times, start - after, end + before):
        candidates.update(stepped_range(ts - before, ts + after, step))
    return sorted(t for t in candidates if start <= t < end)


def build_rows(
    indices: dict[str, Any],
    out_dir: str,
    horizon_minutes: int,
    max_assignments: int | None,
    before_minutes: int,
    after_minutes: int,
    sample_minutes: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "next_hour_rows.csv")
    fieldnames = [
        "client_id",
        "room_id",
        "patient_id",
        "t",
        "diagnosis",
        "symptom",
        "primary_medication_count",
        *CHANGE_METADATA_COLUMNS,
        *INPUT_COLUMNS,
        *TARGET_COLUMNS,
    ]

    patients_by_id = indices["patients_by_id"]
    assignments = indices["assignments"]
    admissions_by_patient = indices["admissions_by_patient"]
    admitted_times_by_patient = indices["admitted_times_by_patient"]
    med_times_by_patient = indices["med_times_by_patient"]
    visits_by_patient = indices["visits_by_patient"]
    visit_times_by_patient = indices["visit_times_by_patient"]
    comfort_by_room = indices["comfort_by_room"]
    comfort_times_by_room = indices["comfort_times_by_room"]

    horizon = timedelta(minutes=horizon_minutes)
    assignment_count = 0
    row_count = 0

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for assignment in assignments:
            if max_assignments is not None and assignment_count >= max_assignments:
                break
            assignment_count += 1

            pid = assignment["patient_id"]
            room_id = assignment["room_id"]
            start = assignment["start"]
            end = assignment["end"]

            patient_profile = patients_by_id.get(pid, {})
            visit_rows = visits_by_patient.get(pid, [])
            visit_times = visit_times_by_patient.get(pid, [])
            medication_times = med_times_by_patient.get(pid, [])
            comfort_rows = comfort_by_room.get(room_id, [])
            comfort_times = comfort_times_by_room.get(room_id, [])

            if not comfort_rows:
                continue

            for t in candidate_times_for_assignment(
                start,
                end,
                visit_times,
                medication_times,
                before_minutes,
                after_minutes,
                sample_minutes,
            ):
                target_time = t + horizon
                next_hour_comfort = latest_at_or_before(comfort_times, comfort_rows, target_time)
                if next_hour_comfort is None:
                    continue
                current_comfort = latest_at_or_before(comfort_times, comfort_rows, t)
                if current_comfort is None:
                    continue

                admission = choose_admission(admissions_by_patient, admitted_times_by_patient, pid, t)
                if admission is None:
                    continue

                diagnosis_value = admission.get("diagnosis") or ""
                symptom_value = active_symptom_for_time(visit_times, visit_rows, t, window_minutes=60)
                med_slots = medication_slots_for_diagnosis(diagnosis_value)

                row: dict[str, Any] = {
                    "client_id": room_id,
                    "room_id": room_id,
                    "patient_id": pid,
                    "t": t.isoformat(),
                    "diagnosis": diagnosis_value,
                    "symptom": symptom_value,
                    "primary_medication_count": len(med_slots),
                    "age": admission.get("age"),
                    "height": patient_profile.get("height"),
                    "weight": admission.get("weight"),
                    "gender_binary": gender_to_binary(patient_profile.get("gender", "")),
                    "curr_temp_main_eval": current_comfort.get("temperature_main"),
                    "curr_temp_toilet_eval": current_comfort.get("temperature_toilet"),
                    "curr_light_eval": current_comfort.get("light_intensity"),
                    "curr_sound_eval": current_comfort.get("sound_level"),
                    "curr_airflow_eval": current_comfort.get("airflow"),
                    "y_temp_main": next_hour_comfort.get("temperature_main"),
                    "y_temp_toilet": next_hour_comfort.get("temperature_toilet"),
                    "y_light": next_hour_comfort.get("light_intensity"),
                    "y_sound": next_hour_comfort.get("sound_level"),
                    "y_airflow": next_hour_comfort.get("airflow"),
                }
                row["y_any_change"] = int(
                    row["curr_temp_main_eval"] != row["y_temp_main"]
                    or row["curr_temp_toilet_eval"] != row["y_temp_toilet"]
                    or row["curr_light_eval"] != row["y_light"]
                    or row["curr_sound_eval"] != row["y_sound"]
                    or int(row["curr_airflow_eval"]) != int(row["y_airflow"])
                )

                for col, value in zip(TIME_COLUMNS, make_time_vector(t.hour, t.minute)):
                    row[col] = value
                for col, value in zip(
                    DIAGNOSIS_COLUMNS,
                    make_one_hot(DIAGNOSIS_TO_INDEX.get(diagnosis_value), len(DIAGNOSIS_COLUMNS)),
                ):
                    row[col] = value
                for col, value in zip(
                    SYMPTOM_COLUMNS,
                    make_one_hot(SYMPTOM_TO_INDEX.get(symptom_value, SYMPTOM_TO_INDEX[""]), len(SYMPTOM_COLUMNS)),
                ):
                    row[col] = value

                for slot in range(1, MAX_MEDICATION_SLOTS + 1):
                    if slot <= len(med_slots):
                        med_name, med_hours = med_slots[slot - 1]
                        type_vector = make_one_hot(MEDICATION_TO_INDEX.get(med_name), len(MEDICATION_NAMES))
                        schedule_vector = normalize_schedule(med_hours)
                    else:
                        type_vector = [0] * len(MEDICATION_NAMES)
                        schedule_vector = [0] * len(TIME_COLUMNS)
                    for col, value in zip(MEDICATION_TYPE_COLUMNS[slot], type_vector):
                        row[col] = value
                    for col, value in zip(MEDICATION_SCHEDULE_COLUMNS[slot], schedule_vector):
                        row[col] = value

                writer.writerow(row)
                row_count += 1

    print("build_next_hour_rows_2.py complete")
    print(f"assignments_processed={assignment_count}")
    print(f"next_hour_rows={row_count}")
    print(f"next_hour_rows_path={out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build next-hour rows from symptom and medication event windows.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to source CSV directory.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Path to output directory.")
    parser.add_argument("--horizon-minutes", type=int, default=60, help="Prediction look-ahead window in minutes.")
    parser.add_argument("--before-minutes", type=int, default=60, help="Minutes before each symptom/medication event to sample.")
    parser.add_argument("--after-minutes", type=int, default=60, help="Minutes after each symptom/medication event to sample.")
    parser.add_argument("--sample-minutes", type=int, default=30, help="Sampling interval inside each event window.")
    parser.add_argument("--max-assignments", type=int, default=None, help="Optional cap for faster dry-runs.")
    return parser.parse_args()


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
        indices,
        out_dir,
        args.horizon_minutes,
        args.max_assignments,
        args.before_minutes,
        args.after_minutes,
        args.sample_minutes,
    )


if __name__ == "__main__":
    main()
