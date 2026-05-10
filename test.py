from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FILESTORAGE_DIR = Path(__file__).resolve().parent / "filestorage"
ADMISSIONS_PATH = FILESTORAGE_DIR / "admissions.csv"
ASSIGNMENTS_PATH = FILESTORAGE_DIR / "room_assignments.csv"
DEVICES_PATH = FILESTORAGE_DIR / "devices.csv"
SENSORS_PATH = FILESTORAGE_DIR / "sensors.csv"
DATA_PATH = FILESTORAGE_DIR / "data.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_ts(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _patients_with_three_admissions(admissions: list[dict[str, str]]) -> set[str]:
    counts = Counter(row["patient_id"] for row in admissions if row.get("patient_id"))
    return {patient_id for patient_id, count in counts.items() if count == 3}


def _assignment_windows_for_patients(
    assignments: list[dict[str, str]],
    kept_patient_ids: set[str],
) -> dict[str, list[tuple[datetime, datetime]]]:
    windows_by_room: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    for row in assignments:
        patient_id = (row.get("patient_id") or "").strip()
        room_id = (row.get("room_id") or "").strip()
        start_raw = (row.get("start_time") or "").strip()
        end_raw = (row.get("end_time") or "").strip()
        if patient_id not in kept_patient_ids or not room_id or not start_raw or not end_raw:
            continue
        windows_by_room[room_id].append((_parse_ts(start_raw), _parse_ts(end_raw)))
    return windows_by_room


def _sensor_ids_for_rooms(devices: list[dict[str, str]], sensors: list[dict[str, str]], room_ids: set[str]) -> dict[str, str]:
    device_to_room = {
        (row.get("device_id") or "").strip(): (row.get("room_id") or "").strip()
        for row in devices
        if (row.get("device_id") or "").strip() and (row.get("room_id") or "").strip() in room_ids
    }
    sensor_to_room: dict[str, str] = {}
    for row in sensors:
        sensor_id = (row.get("sensor_id") or "").strip()
        device_id = (row.get("device_id") or "").strip()
        room_id = device_to_room.get(device_id)
        if sensor_id and room_id:
            sensor_to_room[sensor_id] = room_id
    return sensor_to_room


def _keep_data_rows(
    sensor_to_room: dict[str, str],
    windows_by_room: dict[str, list[tuple[datetime, datetime]]],
) -> int:
    kept_count = 0
    temp_path = DATA_PATH.with_suffix(".csv.tmp")
    with DATA_PATH.open("r", encoding="utf-8", newline="") as src, temp_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or ["data_id", "sensor_id", "value", "timestamp"])
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            sensor_id = (row.get("sensor_id") or "").strip()
            timestamp_raw = (row.get("timestamp") or "").strip()
            room_id = sensor_to_room.get(sensor_id)
            if not room_id or not timestamp_raw:
                continue
            timestamp = _parse_ts(timestamp_raw)
            windows = windows_by_room.get(room_id, [])
            if any(start <= timestamp <= end for start, end in windows):
                writer.writerow(row)
                kept_count += 1
    temp_path.replace(DATA_PATH)
    return kept_count


def main() -> None:
    admissions = _read_csv(ADMISSIONS_PATH)
    assignments = _read_csv(ASSIGNMENTS_PATH)
    devices = _read_csv(DEVICES_PATH)
    sensors = _read_csv(SENSORS_PATH)

    kept_patient_ids = _patients_with_three_admissions(admissions)
    windows_by_room = _assignment_windows_for_patients(assignments, kept_patient_ids)
    sensor_to_room = _sensor_ids_for_rooms(devices, sensors, set(windows_by_room))
    kept_count = _keep_data_rows(sensor_to_room, windows_by_room)

    print(f"Kept {kept_count} data rows in {DATA_PATH.name}")
    print("Patient IDs left:")
    for patient_id in sorted(kept_patient_ids, key=lambda value: int(value)):
        print(patient_id)


if __name__ == "__main__":
    main()
