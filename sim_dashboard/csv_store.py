from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FILES = (
    "rooms.csv",
    "devices.csv",
    "sensors.csv",
    "patients.csv",
    "admissions.csv",
    "room_assignments.csv",
    "medications.csv",
    "visits.csv",
    "comfort_preferences.csv",
    "data.csv",
    "utility_usages.csv",
    "ventilations.csv",
    "toilet_heaters.csv",
    "toilet_lights.csv",
)


_CACHE_SIGNATURE: tuple[Any, ...] | None = None
_CACHE_DATA: dict[str, Any] | None = None


def _data_dir() -> Path:
    default_dir = Path(__file__).resolve().parents[1] / "filestorage"
    override = os.getenv("FL4HOSPITAL_FILESTORAGE")
    return Path(override) if override else default_dir


def _parse_ts(value: str | None) -> datetime | None:
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


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _read_csv(data_dir: Path, file_name: str) -> list[dict[str, str]]:
    path = data_dir / file_name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _signature(data_dir: Path) -> tuple[Any, ...]:
    sig: list[Any] = [str(data_dir.resolve())]
    for file_name in FILES:
        path = data_dir / file_name
        if not path.exists():
            sig.append((file_name, None, None))
            continue
        stat = path.stat()
        sig.append((file_name, int(stat.st_mtime_ns), stat.st_size))
    return tuple(sig)


def load_data() -> dict[str, Any]:
    global _CACHE_SIGNATURE, _CACHE_DATA

    data_dir = _data_dir()
    current_signature = _signature(data_dir)
    if _CACHE_SIGNATURE == current_signature and _CACHE_DATA is not None:
        return _CACHE_DATA

    rooms_raw = _read_csv(data_dir, "rooms.csv")
    devices_raw = _read_csv(data_dir, "devices.csv")
    sensors_raw = _read_csv(data_dir, "sensors.csv")
    patients_raw = _read_csv(data_dir, "patients.csv")
    admissions_raw = _read_csv(data_dir, "admissions.csv")
    assignments_raw = _read_csv(data_dir, "room_assignments.csv")
    medications_raw = _read_csv(data_dir, "medications.csv")
    visits_raw = _read_csv(data_dir, "visits.csv")
    comfort_raw = _read_csv(data_dir, "comfort_preferences.csv")
    data_raw = _read_csv(data_dir, "data.csv")
    utility_raw = _read_csv(data_dir, "utility_usages.csv")
    ventilations_raw = _read_csv(data_dir, "ventilations.csv")
    toilet_heaters_raw = _read_csv(data_dir, "toilet_heaters.csv")
    toilet_lights_raw = _read_csv(data_dir, "toilet_lights.csv")

    rooms: list[dict[str, Any]] = []
    room_by_id: dict[int, dict[str, Any]] = {}
    for row in rooms_raw:
        room_id = _to_int(row.get("room_id"))
        if room_id is None:
            continue
        room = {
            "room_id": room_id,
            "room_number": (row.get("room_number") or "").strip(),
        }
        rooms.append(room)
        room_by_id[room_id] = room
    rooms.sort(key=lambda r: r["room_id"])

    patients: list[dict[str, Any]] = []
    patients_by_id: dict[int, dict[str, Any]] = {}
    for row in patients_raw:
        patient_id = _to_int(row.get("patient_id"))
        if patient_id is None:
            continue
        patient = {
            "patient_id": patient_id,
            "name": (row.get("name") or "").strip(),
            "gender": (row.get("gender") or "").strip(),
            "ethnicity": (row.get("ethnicity") or "").strip(),
            "height": _to_float(row.get("height")),
        }
        patients.append(patient)
        patients_by_id[patient_id] = patient
    patients.sort(key=lambda p: p["patient_id"])

    admissions: list[dict[str, Any]] = []
    admissions_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in admissions_raw:
        admission_id = _to_int(row.get("admission_id"))
        patient_id = _to_int(row.get("patient_id"))
        if admission_id is None or patient_id is None:
            continue
        item = {
            "admission_id": admission_id,
            "patient_id": patient_id,
            "initial_room_id": _to_int(row.get("initial_room_id")),
            "admitted_at": _parse_ts(row.get("admitted_at")),
            "discharged_at": _parse_ts(row.get("discharged_at")),
            "age": _to_int(row.get("age")),
            "weight": _to_float(row.get("weight")),
            "current_diagnosis": (row.get("current_diagnosis") or "").strip(),
        }
        admissions.append(item)
        admissions_by_patient[patient_id].append(item)
    for group in admissions_by_patient.values():
        group.sort(key=lambda a: (a["admitted_at"] is None, a["admitted_at"]))
    admissions.sort(key=lambda a: (a["admitted_at"] is None, a["admitted_at"]))

    assignments: list[dict[str, Any]] = []
    assignments_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    assignments_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments_raw:
        assignment_id = _to_int(row.get("assignment_id"))
        admission_id = _to_int(row.get("admission_id"))
        patient_id = _to_int(row.get("patient_id"))
        room_id = _to_int(row.get("room_id"))
        if assignment_id is None or patient_id is None or room_id is None:
            continue
        item = {
            "assignment_id": assignment_id,
            "admission_id": admission_id,
            "patient_id": patient_id,
            "room_id": room_id,
            "start_time": _parse_ts(row.get("start_time")),
            "end_time": _parse_ts(row.get("end_time")),
        }
        assignments.append(item)
        assignments_by_patient[patient_id].append(item)
        assignments_by_room[room_id].append(item)
    for group in assignments_by_patient.values():
        group.sort(key=lambda a: (a["start_time"] is None, a["start_time"]))
    for group in assignments_by_room.values():
        group.sort(key=lambda a: (a["start_time"] is None, a["start_time"]))

    medications_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in medications_raw:
        medication_id = _to_int(row.get("medication_id"))
        patient_id = _to_int(row.get("patient_id"))
        if medication_id is None or patient_id is None:
            continue
        item = {
            "medication_id": medication_id,
            "patient_id": patient_id,
            "medication_time": _parse_ts(row.get("medication_time")),
            "route": (row.get("route") or "").strip(),
            "drug_name": (row.get("drug_name") or "").strip(),
            "dose": (row.get("dose") or "").strip(),
            "status": (row.get("status") or "").strip(),
        }
        medications_by_patient[patient_id].append(item)
    for group in medications_by_patient.values():
        group.sort(
            key=lambda m: (m["medication_time"] is None, m["medication_time"]),
            reverse=True,
        )

    visits_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in visits_raw:
        visit_id = _to_int(row.get("visit_id"))
        patient_id = _to_int(row.get("patient_id"))
        if visit_id is None or patient_id is None:
            continue
        item = {
            "visit_id": visit_id,
            "patient_id": patient_id,
            "visit_time": _parse_ts(row.get("visit_time")),
            "body_temperature": _to_float(row.get("body_temperature")),
            "blood_pressure": (row.get("blood_pressure") or "").strip(),
            "symptoms": (row.get("symptoms") or "").strip(),
        }
        visits_by_patient[patient_id].append(item)
    for group in visits_by_patient.values():
        group.sort(key=lambda v: (v["visit_time"] is None, v["visit_time"]), reverse=True)

    comfort_by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    comfort_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in comfort_raw:
        comfort_pref_id = _to_int(row.get("comfort_pref_id"))
        patient_id = _to_int(row.get("patient_id"))
        room_id = _to_int(row.get("room_id"))
        if comfort_pref_id is None or patient_id is None or room_id is None:
            continue
        item = {
            "comfort_pref_id": comfort_pref_id,
            "patient_id": patient_id,
            "room_id": room_id,
            "timestamp": _parse_ts(row.get("timestamp")),
            "temperature_main": _to_float(row.get("temperature_main")),
            "temperature_toilet": _to_float(row.get("temperature_toilet")),
            "light_intensity": _to_float(row.get("light_intensity")),
            "sound_level": _to_float(row.get("sound_level")),
            "airflow": _to_bool(row.get("airflow")),
            "source": (row.get("source") or "").strip(),
        }
        comfort_by_patient[patient_id].append(item)
        comfort_by_room[room_id].append(item)
    for group in comfort_by_patient.values():
        group.sort(key=lambda c: (c["timestamp"] is None, c["timestamp"]), reverse=True)
    for group in comfort_by_room.values():
        group.sort(key=lambda c: (c["timestamp"] is None, c["timestamp"]), reverse=True)

    device_to_room: dict[int, int] = {}
    device_to_location: dict[int, str] = {}
    for row in devices_raw:
        device_id = _to_int(row.get("device_id"))
        room_id = _to_int(row.get("room_id"))
        if device_id is None:
            continue
        if room_id is not None:
            device_to_room[device_id] = room_id
        device_to_location[device_id] = (row.get("location") or "").strip()

    sensors: list[dict[str, Any]] = []
    sensors_by_id: dict[int, dict[str, Any]] = {}
    sensors_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sensors_raw:
        sensor_id = _to_int(row.get("sensor_id"))
        device_id = _to_int(row.get("device_id"))
        if sensor_id is None or device_id is None:
            continue
        room_id = device_to_room.get(device_id)
        item = {
            "sensor_id": sensor_id,
            "device_id": device_id,
            "room_id": room_id,
            "location": device_to_location.get(device_id, ""),
            "sensor_type": (row.get("sensor_type") or "").strip(),
            "uuid": (row.get("uuid") or "").strip(),
            "unit": (row.get("unit") or "").strip(),
        }
        sensors.append(item)
        sensors_by_id[sensor_id] = item
        if room_id is not None:
            sensors_by_room[room_id].append(item)
    for group in sensors_by_room.values():
        group.sort(key=lambda sensor: (sensor["sensor_type"], sensor["sensor_id"]))

    data_points: list[dict[str, Any]] = []
    data_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in data_raw:
        data_id = _to_int(row.get("data_id"))
        sensor_id = _to_int(row.get("sensor_id"))
        if data_id is None or sensor_id is None:
            continue
        sensor = sensors_by_id.get(sensor_id)
        if sensor is None:
            continue
        item = {
            "data_id": data_id,
            "sensor_id": sensor_id,
            "device_id": sensor["device_id"],
            "room_id": sensor["room_id"],
            "location": sensor["location"],
            "sensor_type": sensor["sensor_type"],
            "unit": sensor["unit"],
            "value": _to_float(row.get("value")),
            "timestamp": _parse_ts(row.get("timestamp")),
        }
        data_points.append(item)
        room_id = sensor["room_id"]
        if room_id is not None:
            data_by_room[room_id].append(item)
    for group in data_by_room.values():
        group.sort(key=lambda d: (d["timestamp"] is None, d["timestamp"]))

    utility_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    devices_by_room: dict[int, set[int]] = defaultdict(set)
    for row in utility_raw:
        usage_id = _to_int(row.get("usage_id"))
        room_id = _to_int(row.get("room_id"))
        if usage_id is None or room_id is None:
            continue
        device_id = _to_int(row.get("device_id"))
        item = {
            "usage_id": usage_id,
            "category": (row.get("category") or "").strip(),
            "water_consumption": _to_float(row.get("water_consumption")),
            "power_consumption": _to_float(row.get("power_consumption")),
            "start_time": _parse_ts(row.get("start_time")),
            "end_time": _parse_ts(row.get("end_time")),
            "room_id": room_id,
            "device_id": device_id,
        }
        utility_by_room[room_id].append(item)
        if device_id is not None:
            devices_by_room[room_id].add(device_id)
    for group in utility_by_room.values():
        group.sort(key=lambda u: (u["start_time"] is None, u["start_time"]), reverse=True)

    ventilations: list[dict[str, Any]] = []
    vents_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in ventilations_raw:
        ventilation_id = _to_int(row.get("ventilation_id"))
        device_id = _to_int(row.get("device_id"))
        if ventilation_id is None:
            continue
        item = {
            "ventilation_id": ventilation_id,
            "mode": (row.get("mode") or "").strip(),
            "level": _to_float(row.get("level")),
            "device_id": device_id,
            "timestamp": _parse_ts(row.get("timestamp")),
        }
        ventilations.append(item)
        if device_id is not None and device_id in device_to_room:
            vents_by_room[device_to_room[device_id]].append(item)
    for group in vents_by_room.values():
        group.sort(key=lambda v: (v["timestamp"] is None, v["timestamp"]), reverse=True)

    toilet_heaters: list[dict[str, Any]] = []
    toilet_heaters_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in toilet_heaters_raw:
        toilet_heater_id = _to_int(row.get("toilet_heater_id"))
        device_id = _to_int(row.get("device_id"))
        if toilet_heater_id is None:
            continue
        item = {
            "toilet_heater_id": toilet_heater_id,
            "device_id": device_id,
            "state": _to_bool(row.get("state")),
            "timestamp": _parse_ts(row.get("timestamp")),
        }
        toilet_heaters.append(item)
        if device_id is not None and device_id in device_to_room:
            toilet_heaters_by_room[device_to_room[device_id]].append(item)
    for group in toilet_heaters_by_room.values():
        group.sort(key=lambda value: (value["timestamp"] is None, value["timestamp"]), reverse=True)

    toilet_lights: list[dict[str, Any]] = []
    toilet_lights_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in toilet_lights_raw:
        toilet_light_id = _to_int(row.get("toilet_light_id"))
        device_id = _to_int(row.get("device_id"))
        if toilet_light_id is None:
            continue
        item = {
            "toilet_light_id": toilet_light_id,
            "device_id": device_id,
            "state": _to_bool(row.get("state")),
            "timestamp": _parse_ts(row.get("timestamp")),
        }
        toilet_lights.append(item)
        if device_id is not None and device_id in device_to_room:
            toilet_lights_by_room[device_to_room[device_id]].append(item)
    for group in toilet_lights_by_room.values():
        group.sort(key=lambda value: (value["timestamp"] is None, value["timestamp"]), reverse=True)

    female_heights = [
        p["height"]
        for p in patients
        if (p["gender"] or "").lower() == "female" and p["height"] is not None
    ]
    male_heights = [
        p["height"]
        for p in patients
        if (p["gender"] or "").lower() == "male" and p["height"] is not None
    ]

    female_ages: list[int] = []
    male_ages: list[int] = []
    for admission in admissions:
        age = admission["age"]
        patient = patients_by_id.get(admission["patient_id"])
        if age is None or patient is None:
            continue
        gender = (patient["gender"] or "").lower()
        if gender == "female":
            female_ages.append(age)
        elif gender == "male":
            male_ages.append(age)

    result = {
        "data_dir": str(data_dir),
        "rooms": rooms,
        "room_by_id": room_by_id,
        "sensors": sensors,
        "sensors_by_id": sensors_by_id,
        "sensors_by_room": sensors_by_room,
        "data_points": data_points,
        "data_by_room": data_by_room,
        "patients": patients,
        "patients_by_id": patients_by_id,
        "admissions": admissions,
        "admissions_by_patient": admissions_by_patient,
        "assignments": assignments,
        "assignments_by_patient": assignments_by_patient,
        "assignments_by_room": assignments_by_room,
        "medications_by_patient": medications_by_patient,
        "visits_by_patient": visits_by_patient,
        "comfort_by_patient": comfort_by_patient,
        "comfort_by_room": comfort_by_room,
        "utility_by_room": utility_by_room,
        "devices_by_room": {k: sorted(v) for k, v in devices_by_room.items()},
        "ventilations": ventilations,
        "ventilations_by_room": vents_by_room,
        "toilet_heaters": toilet_heaters,
        "toilet_heaters_by_room": toilet_heaters_by_room,
        "toilet_lights": toilet_lights,
        "toilet_lights_by_room": toilet_lights_by_room,
        "female_heights": female_heights,
        "male_heights": male_heights,
        "female_ages": female_ages,
        "male_ages": male_ages,
        "total_rooms": len(rooms),
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_medications": len(medications_raw),
        "total_visits": len(visits_raw),
        "readmissions": sum(1 for vals in admissions_by_patient.values() if len(vals) > 1),
    }

    _CACHE_SIGNATURE = current_signature
    _CACHE_DATA = result
    return result
