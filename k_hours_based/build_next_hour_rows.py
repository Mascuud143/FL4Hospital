import argparse
import csv
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import timedelta
from typing import Any

from build_next_hour_rows_base import (
    active_symptom_for_time,
    build_diagnosis_feature_cache,
    build_indices,
    build_symptom_feature_cache,
    build_time_feature_cache,
    choose_admission,
    latest_at_or_before,
    parse_bp,
    resolve_input_path,
)
from next_hour_schema import (
    CHANGE_METADATA_COLUMNS,
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    gender_to_binary,
)
from runtime_defaults import (
    BUILD_CSV_WRITE_BATCH_SIZE,
    BUILD_DEFAULT_CHUNK_SIZE,
    BUILD_DEFAULT_WORKERS,
    default_build_workers,
)

DEFAULT_DATA_DIR = "filestorage"
DEFAULT_OUT_DIR = "outputs_next_hour_parallel"
DEFAULT_TMP_DIR = ".tmp_next_hour_rows_parallel"
DEFAULT_WORKERS = BUILD_DEFAULT_WORKERS
DEFAULT_CHUNK_SIZE = BUILD_DEFAULT_CHUNK_SIZE
CSV_WRITE_BATCH_SIZE = BUILD_CSV_WRITE_BATCH_SIZE


def chunk_assignments(assignments: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not assignments:
        return []

    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_room_id: Any = None
    current_room_group: list[dict[str, Any]] = []

    def flush_room_group() -> None:
        nonlocal current_chunk, current_room_group
        if not current_room_group:
            return
        if current_chunk and len(current_chunk) + len(current_room_group) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.extend(current_room_group)
        current_room_group = []

    for assignment in assignments:
        room_id = assignment["room_id"]
        if current_room_id is None:
            current_room_id = room_id
        if room_id != current_room_id:
            flush_room_group()
            current_room_id = room_id
        current_room_group.append(assignment)

    flush_room_group()
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def build_subset_indices(indices: dict[str, Any], assignments: list[dict[str, Any]]) -> dict[str, Any]:
    patient_ids = {assignment["patient_id"] for assignment in assignments}
    room_ids = {assignment["room_id"] for assignment in assignments}
    return {
        "patients_by_id": {pid: indices["patients_by_id"].get(pid, {}) for pid in patient_ids},
        "admissions_by_patient": {pid: indices["admissions_by_patient"].get(pid, []) for pid in patient_ids},
        "admitted_times_by_patient": {pid: indices["admitted_times_by_patient"].get(pid, []) for pid in patient_ids},
        "visits_by_patient": {pid: indices["visits_by_patient"].get(pid, []) for pid in patient_ids},
        "visit_times_by_patient": {pid: indices["visit_times_by_patient"].get(pid, []) for pid in patient_ids},
        "comfort_by_room": {room_id: indices["comfort_by_room"].get(room_id, []) for room_id in room_ids},
        "comfort_times_by_room": {room_id: indices["comfort_times_by_room"].get(room_id, []) for room_id in room_ids},
    }


def build_fieldnames() -> list[str]:
    return [
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


def write_assignment_chunk(
    chunk_id: int,
    assignments: list[dict[str, Any]],
    indices: dict[str, Any],
    temp_dir: str,
    step_minutes: int,  
    horizon_minutes: int,
) -> tuple[str, int]:
    fieldnames = build_fieldnames()
    out_path = os.path.join(temp_dir, f"part_{chunk_id:05d}.csv")
    time_feature_cache = build_time_feature_cache()
    symptom_feature_cache = build_symptom_feature_cache()
    diagnosis_feature_cache = build_diagnosis_feature_cache()
    step = timedelta(minutes=step_minutes)
    horizon = timedelta(minutes=horizon_minutes)
    row_count = 0

    patients_by_id = indices["patients_by_id"]
    admissions_by_patient = indices["admissions_by_patient"]
    admitted_times_by_patient = indices["admitted_times_by_patient"]
    visits_by_patient = indices["visits_by_patient"]
    visit_times_by_patient = indices["visit_times_by_patient"]
    comfort_by_room = indices["comfort_by_room"]
    comfort_times_by_room = indices["comfort_times_by_room"]

    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        pending_rows: list[dict[str, Any]] = []

        for assignment in assignments:
            pid = assignment["patient_id"]
            room_id = assignment["room_id"]
            t = assignment["start"]
            end = assignment["end"]

            patient_profile = patients_by_id.get(pid, {})
            visit_rows = visits_by_patient.get(pid, [])
            visit_times = visit_times_by_patient.get(pid, [])
            comfort_rows = comfort_by_room.get(room_id, [])
            comfort_times = comfort_times_by_room.get(room_id, [])

            while t < end:
                target_time = t + horizon
                next_hour_comfort = latest_at_or_before(comfort_times, comfort_rows, target_time) if comfort_rows else None
                if next_hour_comfort is None:
                    t += step
                    continue

                current_comfort = latest_at_or_before(comfort_times, comfort_rows, t) if comfort_rows else None
                if current_comfort is None:
                    t += step
                    continue

                admission = choose_admission(admissions_by_patient, admitted_times_by_patient, pid, t)
                if admission is None:
                    t += step
                    continue

                diagnosis_value = admission.get("diagnosis") or ""
                diagnosis_features = diagnosis_feature_cache.get(diagnosis_value)
                if diagnosis_features is None:
                    diagnosis_features = {"diagnosis": {}, "medication": {}, "med_slots": []}
                symptom_value = active_symptom_for_time(visit_times, visit_rows, t, window_minutes=60)
                latest_visit = latest_at_or_before(visit_times, visit_rows, t) if visit_rows else None
                bp_systolic, bp_diastolic = parse_bp(latest_visit.get("blood_pressure") if latest_visit else None)
                med_slots = diagnosis_features["med_slots"]

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
                    "body_temperature": latest_visit.get("body_temperature") if latest_visit else None,
                    "bp_systolic": bp_systolic,
                    "bp_diastolic": bp_diastolic,
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

                row.update(time_feature_cache[(t.hour, 30 if t.minute >= 30 else 0)])
                row.update(diagnosis_features["diagnosis"])
                row.update(symptom_feature_cache.get(symptom_value, symptom_feature_cache[""]))
                row.update(diagnosis_features["medication"])
                pending_rows.append(row)
                if len(pending_rows) >= CSV_WRITE_BATCH_SIZE:
                    writer.writerows(pending_rows)
                    pending_rows = []
                row_count += 1
                t += step

        if pending_rows:
            writer.writerows(pending_rows)

    return out_path, row_count


def merge_part_files(part_paths: list[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header_written = False
    with open(out_path, "w", encoding="utf-8", newline="") as out_handle:
        for part_path in part_paths:
            with open(part_path, "r", encoding="utf-8", newline="") as in_handle:
                for line_number, line in enumerate(in_handle):
                    if line_number == 0:
                        if header_written:
                            continue
                        header_written = True
                    out_handle.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build next-hour rows in parallel by assignment chunks.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to source CSV directory.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Path to output directory.")
    parser.add_argument("--step-minutes", type=int, default=30, help="Decision interval in minutes.")
    parser.add_argument("--horizon-minutes", type=int, default=60, help="Prediction look-ahead window in minutes.")
    parser.add_argument("--max-assignments", type=int, default=None, help="Optional cap for faster dry-runs.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of worker processes.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Assignments per worker task.")
    parser.add_argument("--csv-write-batch-size", type=int, default=CSV_WRITE_BATCH_SIZE, help="Rows buffered per CSV write.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary part CSV files.")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    global CSV_WRITE_BATCH_SIZE
    CSV_WRITE_BATCH_SIZE = max(1, int(args.csv_write_batch_size))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = resolve_input_path(args.data_dir, project_root)
    if os.path.isabs(args.out_dir):
        out_dir = args.out_dir
    elif args.out_dir == DEFAULT_OUT_DIR:
        out_dir = os.path.join(script_dir, args.out_dir)
    else:
        out_dir = os.path.abspath(args.out_dir)

    temp_dir = os.path.join(out_dir, DEFAULT_TMP_DIR)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    indices = build_indices(data_dir)
    assignments = indices["assignments"]
    if args.max_assignments is not None:
        assignments = assignments[: max(0, args.max_assignments)]
    if not assignments:
        raise RuntimeError("No assignments available to build rows.")

    assignment_chunks = chunk_assignments(assignments, args.chunk_size)
    total_chunks = len(assignment_chunks)
    width = max(1, len(str(total_chunks)))
    worker_limit = default_build_workers()
    worker_count = max(1, min(args.workers, worker_limit, total_chunks))

    results: list[tuple[int, str, int]] = []
    max_in_flight = max(1, min(total_chunks, worker_count * 2))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map: dict[Any, int] = {}
        next_chunk_id = 0

        while next_chunk_id < total_chunks and len(future_map) < max_in_flight:
            chunk = assignment_chunks[next_chunk_id]
            subset_indices = build_subset_indices(indices, chunk)
            future = executor.submit(
                write_assignment_chunk,
                next_chunk_id,
                chunk,
                subset_indices,
                temp_dir,
                args.step_minutes,
                args.horizon_minutes,
            )
            future_map[future] = next_chunk_id
            next_chunk_id += 1

        while future_map:
            done, _ = wait(tuple(future_map), return_when=FIRST_COMPLETED)
            for future in done:
                chunk_id = future_map.pop(future)
                part_path, row_count = future.result()
                print(
                    f"[chunk {chunk_id + 1:0{width}d}/{total_chunks}] rows={row_count} path={part_path}",
                    flush=True,
                )
                results.append((chunk_id, part_path, row_count))

                if next_chunk_id < total_chunks:
                    chunk = assignment_chunks[next_chunk_id]
                    subset_indices = build_subset_indices(indices, chunk)
                    next_future = executor.submit(
                        write_assignment_chunk,
                        next_chunk_id,
                        chunk,
                        subset_indices,
                        temp_dir,
                        args.step_minutes,
                        args.horizon_minutes,
                    )
                    future_map[next_future] = next_chunk_id
                    next_chunk_id += 1

    results.sort(key=lambda item: item[0])
    out_path = os.path.join(out_dir, "next_hour_rows.csv")
    merge_part_files([part_path for _, part_path, _ in results], out_path)

    total_rows = sum(row_count for _, _, row_count in results)
    print("build_next_hour_rows_parallel.py complete")
    print(f"assignments_processed={len(assignments)}")
    print(f"chunks={total_chunks}")
    print(f"workers_used={worker_count}")
    print(f"csv_write_batch_size={CSV_WRITE_BATCH_SIZE}")
    print(f"next_hour_rows={total_rows}")
    print(f"next_hour_rows_path={out_path}")
    elapsed_seconds = time.perf_counter() - started
    print(f"elapsed_seconds={elapsed_seconds:.1f}")
    print(f"elapsed_minutes={elapsed_seconds / 60.0:.2f}")

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"temp_dir={temp_dir}")


if __name__ == "__main__":
    main()
