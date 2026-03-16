from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from persistence.database import session_scope
from persistence.models import Admission, Device, Medication, Patient, Room, RoomAssignment, Sensor, Visit
from persistence.models.speaker import Speaker
from persistence.models.toilet_heater import ToiletHeater
from persistence.models.toilet_light import ToiletLight
from persistence.models.ventilation import Ventilation


DEFAULT_ROOM_MAP = {1: 1, 2: 2}
DEFAULT_ROOMS = {
    1: "H-01",
    2: "H-02",
}
SIMULATED_DEVICE_SPECS = (
    ("light", "main"),
    ("ventilation", "main"),
    ("speaker", "main"),
    ("toilet_light", "toilet"),
    ("toilet_heater", "toilet"),
)


@dataclass
class InitSummary:
    rooms: int = 0
    patients: int = 0
    admissions: int = 0
    assignments: int = 0
    medications: int = 0
    visits: int = 0
    simulated_devices: int = 0


def _read_csv_rows(csv_dir: Path, filename: str) -> list[dict[str, str]]:
    path = csv_dir / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def ensure_simulated_devices(room_id: int) -> int:
    created = 0
    with session_scope() as session:
        for device_type, location in SIMULATED_DEVICE_SPECS:
            device = (
                session.query(Device)
                .filter(Device.room_id == room_id, Device.device_type == device_type, Device.location == location)
                .one_or_none()
            )
            if device is None:
                device = Device(device_type=device_type, room_id=room_id, location=location)
                session.add(device)
                session.flush()
                created += 1

            if device_type == "ventilation" and device.ventilation is None:
                session.add(Ventilation(device_id=device.device_id, mode="off", level=0.0))
            elif device_type == "speaker" and device.speaker is None:
                session.add(Speaker(device_id=device.device_id, level=0.0))
            elif device_type == "toilet_light" and device.toilet_light is None:
                session.add(ToiletLight(device_id=device.device_id, state=False))
            elif device_type == "toilet_heater" and device.toilet_heater is None:
                session.add(ToiletHeater(device_id=device.device_id, state=False))
    return created


def initialize_hybrid_hospital(csv_dir: str | Path = "hybrid_filestorage") -> InitSummary:
    csv_path = Path(csv_dir)
    summary = InitSummary()
    room_rows = {int(row["room_id"]): row for row in _read_csv_rows(csv_path, "rooms.csv") if row.get("room_id")}
    patient_rows = {int(row["patient_id"]): row for row in _read_csv_rows(csv_path, "patients.csv") if row.get("patient_id")}
    admission_rows = [row for row in _read_csv_rows(csv_path, "admissions.csv") if row.get("patient_id")]
    medication_rows = [row for row in _read_csv_rows(csv_path, "medications.csv") if row.get("patient_id")]
    visit_rows = [row for row in _read_csv_rows(csv_path, "visits.csv") if row.get("patient_id")]

    with session_scope() as session:
        for room_id in DEFAULT_ROOM_MAP.values():
            room = session.get(Room, room_id)
            room_csv = room_rows.get(room_id, {})
            room_number = room_csv.get("room_number") or DEFAULT_ROOMS[room_id]
            if room is None:
                session.add(Room(room_id=room_id, room_number=room_number))
                summary.rooms += 1
            else:
                room.room_number = room_number

        for patient_id, room_id in DEFAULT_ROOM_MAP.items():
            patient_csv = patient_rows.get(patient_id, {})
            patient = session.get(Patient, patient_id)
            if patient is None:
                patient = Patient(patient_id=patient_id)
                session.add(patient)
                summary.patients += 1

            patient.name = patient_csv.get("name") or f"Patient {patient_id}"
            patient.gender = patient_csv.get("gender")
            patient.ethnicity = patient_csv.get("ethnicity")
            patient.height = _parse_int(patient_csv.get("height"))

            patient_admissions = sorted(
                [row for row in admission_rows if _parse_int(row.get("patient_id")) == patient_id],
                key=lambda row: _parse_dt(row.get("admitted_at")) or datetime.now(timezone.utc),
            )
            if patient_admissions:
                src = patient_admissions[0]
                admission = (
                    session.query(Admission)
                    .filter(Admission.patient_id == patient_id)
                    .order_by(Admission.admitted_at.asc())
                    .first()
                )
                if admission is None:
                    admission = Admission(patient_id=patient_id)
                    session.add(admission)
                    summary.admissions += 1
                admission.initial_room_id = room_id
                admission.admitted_at = _parse_dt(src.get("admitted_at")) or datetime.now(timezone.utc)
                admission.discharged_at = _parse_dt(src.get("discharged_at"))
                admission.age = _parse_int(src.get("age"))
                admission.weight = _parse_float(src.get("weight"))
                admission.current_diagnosis = src.get("current_diagnosis")
            else:
                admission = (
                    session.query(Admission)
                    .filter(Admission.patient_id == patient_id)
                    .order_by(Admission.admitted_at.asc())
                    .first()
                )
                if admission is None:
                    admission = Admission(
                        patient_id=patient_id,
                        initial_room_id=room_id,
                        admitted_at=datetime.now(timezone.utc),
                    )
                    session.add(admission)
                    summary.admissions += 1

            session.flush()
            assignment = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.patient_id == patient_id, RoomAssignment.room_id == room_id, RoomAssignment.end_time.is_(None))
                .one_or_none()
            )
            if assignment is None:
                assignment = RoomAssignment(
                    admission_id=admission.admission_id,
                    patient_id=patient_id,
                    room_id=room_id,
                    start_time=admission.admitted_at or datetime.now(timezone.utc),
                    end_time=None,
                )
                session.add(assignment)
                summary.assignments += 1

            for med_row in medication_rows:
                if _parse_int(med_row.get("patient_id")) != patient_id:
                    continue
                medication_id = _parse_int(med_row.get("medication_id"))
                medication = session.get(Medication, medication_id) if medication_id is not None else None
                if medication is None:
                    medication = Medication(
                        medication_id=medication_id,
                        patient_id=patient_id,
                        drug_name=med_row.get("drug_name") or "Medication",
                    )
                    session.add(medication)
                    summary.medications += 1
                medication.medication_time = _parse_dt(med_row.get("medication_time")) or datetime.now(timezone.utc)
                medication.route = med_row.get("route")
                medication.dose = med_row.get("dose")
                medication.status = med_row.get("status")

            for visit_row in visit_rows:
                if _parse_int(visit_row.get("patient_id")) != patient_id:
                    continue
                visit_id = _parse_int(visit_row.get("visit_id"))
                visit = session.get(Visit, visit_id) if visit_id is not None else None
                if visit is None:
                    visit = Visit(visit_id=visit_id, patient_id=patient_id)
                    session.add(visit)
                    summary.visits += 1
                visit.visit_time = _parse_dt(visit_row.get("visit_time")) or datetime.now(timezone.utc)
                visit.body_temperature = _parse_float(visit_row.get("body_temperature"))
                visit.blood_pressure = visit_row.get("blood_pressure")
                visit.symptoms = visit_row.get("symptoms")

    for room_id in DEFAULT_ROOM_MAP.values():
        summary.simulated_devices += ensure_simulated_devices(room_id)
    return summary


def register_nordic_device(*, room_id: int, location: str, mac_address: str) -> Device:
    normalized_mac = mac_address.strip().upper()
    with session_scope() as session:
        device = (
            session.query(Device)
            .filter(Device.mac_address == normalized_mac)
            .one_or_none()
        )
        if device is not None:
            raise ValueError(f"MAC address {normalized_mac} is already registered.")

        room_location_sensor = (
            session.query(Device)
            .filter(
                Device.room_id == room_id,
                Device.location == location,
                Device.device_type == "sensor",
            )
            .one_or_none()
        )
        if room_location_sensor is not None:
            raise ValueError(f"Room {room_id} already has a Nordic sensor registered for {location}.")

        device = Device(mac_address=normalized_mac, device_type="sensor", room_id=room_id, location=location)
        session.add(device)
        session.flush()
        session.refresh(device)
        return device


def update_nordic_device_mac(*, device_id: int, mac_address: str) -> Device:
    normalized_mac = mac_address.strip().upper()
    with session_scope() as session:
        existing = (
            session.query(Device)
            .filter(Device.mac_address == normalized_mac, Device.device_id != device_id)
            .one_or_none()
        )
        if existing is not None:
            raise ValueError(f"MAC address {normalized_mac} is already registered.")

        device = session.get(Device, device_id)
        if device is None:
            raise ValueError("Device not found.")
        device.mac_address = normalized_mac
        session.flush()
        session.refresh(device)
        return device


def add_sensor_to_nordic_device(*, device_id: int, sensor_type: str, uuid: str, unit: str | None = None) -> Sensor:
    normalized_uuid = uuid.strip()
    normalized_type = sensor_type.strip().lower()
    normalized_unit = unit.strip() if unit else None
    with session_scope() as session:
        device = session.get(Device, device_id)
        if device is None:
            raise ValueError("Device not found.")
        sensor = (
            session.query(Sensor)
            .filter(Sensor.device_id == device.device_id, Sensor.sensor_type == normalized_type)
            .one_or_none()
        )
        if sensor is not None:
            raise ValueError(f"Device already has a sensor registered for type {normalized_type}.")

        sensor = Sensor(
            device_id=device.device_id,
            sensor_type=normalized_type,
            uuid=normalized_uuid,
            unit=normalized_unit,
        )
        session.add(sensor)
        session.flush()
        session.refresh(sensor)
        return sensor


def update_nordic_sensor(*, sensor_id: int, sensor_type: str, uuid: str, unit: str | None = None) -> Sensor:
    normalized_uuid = uuid.strip()
    normalized_type = sensor_type.strip().lower()
    normalized_unit = unit.strip() if unit else None
    with session_scope() as session:
        sensor = session.get(Sensor, sensor_id)
        if sensor is None:
            raise ValueError("Sensor not found.")

        existing_type = (
            session.query(Sensor)
            .filter(
                Sensor.device_id == sensor.device_id,
                Sensor.sensor_type == normalized_type,
                Sensor.sensor_id != sensor_id,
            )
            .one_or_none()
        )
        if existing_type is not None:
            raise ValueError(f"This device already has a sensor registered for type {normalized_type}.")

        sensor.sensor_type = normalized_type
        sensor.uuid = normalized_uuid
        sensor.unit = normalized_unit
        session.flush()
        session.refresh(sensor)
        return sensor
