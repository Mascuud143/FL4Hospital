from __future__ import annotations
print(">>> RUN SCRIPT 1 <<<")

import math
import random
from datetime import date, datetime, timedelta
from typing import List, Optional

from persistence.database import session_scope
from persistence.models.room import Room
from persistence.models.patient import Patient
from persistence.models.admission import Admission
from persistence.models.room_assignment import RoomAssignment

from simulation_batch.config import (
    PATIENT_COUNT,
    DAYS,
    START_DATE,
)

from simulation_batch.generators.patients import generate_patients
from simulation_batch.generators.name_gen import generate_name
from simulation_batch.generators.age_height_weight_gen import age_height_weight_generator

# Runtime (not DB) device/sensor objects
from ble import Device as BLEDevice, Sensor as BLESensor


# -------------------------
# Helpers
# -------------------------

def _ensure_datetime_midnight(d: date | datetime) -> datetime:
    if isinstance(d, date) and not isinstance(d, datetime):
        return datetime.combine(d, datetime.min.time())
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _random_mac(rng: random.Random) -> str:
    return "02:00:00:%02x:%02x:%02x" % (
        rng.randint(0, 255),
        rng.randint(0, 255),
        rng.randint(0, 255),
    )


def _unit_for(st: str) -> str:
    return {
        "temperature": "°C",
        "humidity": "%",
        "co2": "ppm",
        "light": "lux",
        "sound": "dB",
    }.get(st, "")


# -------------------------
# Seeding
# -------------------------
from simulation_batch.config import (
    PATIENT_COUNT,
    DAYS,
    START_DATE,
    CHANGE_ROOM_PROB,
    MIN_DAYS_BEFORE_TRANSFER,
    MIN_DAYS_AFTER_TRANSFER,
)


def seed_simulated_world(
    *,
    seed: int = 42,
    patient_count: int = PATIENT_COUNT,
    days: int = DAYS,
    start_date: date | datetime = START_DATE,
    include_speaker: bool = True,
) -> List[BLEDevice]:
    """
    Seeds DB:
      - Rooms (dynamic count, created as needed)
      - Patients (identity-only)
      - Admissions (per-stay attributes, admitted/discharged timestamps)
      - RoomAssignments (linked to an Admission via admission_id)

    Returns runtime devices (BLEDevice list) for:
      - main sensor device per room (temp, humidity, co2, light, sound)
      - toilet sensor device per room (temp)
      - ventilation device per room (no sensors)
      - speaker device per room (optional, no sensors)
      - toilet heater device per room (no sensors)
      - toilet light device per room (no sensors)
    """
    rng = random.Random(seed)
    start_dt = _ensure_datetime_midnight(start_date)

    main_sensor_types = ("temperature", "humidity", "co2", "light", "sound")
    toilet_sensor_types = ("temperature",)

    # --------- Build patients (identity-only) ----------
    patients: list[Patient] = []
    stay_days_list: list[int] = []

    # Keep per-patient per-stay attributes here, since they now belong to Admission
    per_stay_attrs: dict[int, dict] = {}  # temp index -> attrs

    generated = generate_patients(patient_count)
    for idx, p in enumerate(generated):
        age_hw = age_height_weight_generator()

        patient = Patient(
            name=generate_name(),
            ethnicity=p["ethnicity"],
            gender=rng.choice(["Male", "Female"]),
        )

        # Per-stay data -> Admission (store now so we can use it later)
        per_stay_attrs[idx] = {
            "age": age_hw["age"],
            "height": age_hw["height"],
            "weight": age_hw["weight"],
            "current_diagnosis": p["diagnosis"],
        }

        stay_days = rng.randint(5, 20)
        patients.append(patient)
        stay_days_list.append(stay_days)

    ble_devices: List[BLEDevice] = []

    with session_scope() as session:
        # insert patients first so they get IDs
        for patient in patients:
            session.add(patient)
        session.flush()

        # --------- Dynamic room allocation ----------
        rooms: list[Room] = []
        current_room: Optional[Room] = None
        current_offset_days = 0
        room_index = 0

        room_occupancy: dict[int, list[tuple[datetime, datetime]]] = {}  # room_id -> list of (start_time, end_time)

        # Track each patient's single admission window for transfer phase
        # patient_id -> (admit_dt, discharge_dt, admission_id)
        patient_stay: dict[int, tuple[datetime, datetime, int]] = {}

        for idx, (patient, stay) in enumerate(zip(patients, stay_days_list)):
            # If no room yet or patient doesn't fit in remaining horizon for this room, open new room
            if current_room is None or (current_offset_days + stay > days):
                room_index += 1
                current_offset_days = 0

                current_room = Room(room_number=100 + room_index)
                session.add(current_room)
                session.flush()
                rooms.append(current_room)

            # admission & discharge within the room timeline
            admitted_at = start_dt + timedelta(days=current_offset_days)
            discharged_at = admitted_at + timedelta(days=stay)

            # ---- Create Admission (per-stay row) ----
            attrs = per_stay_attrs[idx]
            adm = Admission(
                patient_id=patient.patient_id,
                initial_room_id=current_room.room_id,
                admitted_at=admitted_at,
                discharged_at=discharged_at,
                age=attrs["age"],
                weight=attrs["weight"],
                height=attrs["height"],
                current_diagnosis=attrs["current_diagnosis"],
            )
            session.add(adm)
            session.flush()  # ensures adm.admission_id

            patient_stay[patient.patient_id] = (admitted_at, discharged_at, adm.admission_id)

            # ---- Create initial RoomAssignment linked to Admission ----
            session.add(
                RoomAssignment(
                    admission_id=adm.admission_id,
                    patient_id=patient.patient_id,
                    room_id=current_room.room_id,
                    start_time=admitted_at,
                    end_time=discharged_at,
                )
            )

            room_occupancy.setdefault(current_room.room_id, []).append((admitted_at, discharged_at))
            current_offset_days += stay

        session.flush()

        # ============================================================
        # EXTRA PHASE: ROOM CHANGE WHILE ADMITTED
        # ============================================================

        def overlaps(a_start, a_end, b_start, b_end):
            return not (a_end <= b_start or a_start >= b_end)

        original_last_room = max(rooms, key=lambda r: r.room_number)
        transfer_rooms = [original_last_room]
        max_room_number = original_last_room.room_number

        print(f">>> TRANSFER PHASE START (prob={CHANGE_ROOM_PROB}) <<<")

        for patient in patients:
            if rng.random() > CHANGE_ROOM_PROB:
                continue

            stay_info = patient_stay.get(patient.patient_id)
            if not stay_info:
                continue

            admit, discharge, admission_id = stay_info
            total_days = (discharge - admit).days

            if total_days < MIN_DAYS_BEFORE_TRANSFER + MIN_DAYS_AFTER_TRANSFER + 1:
                continue

            transfer_day = rng.randint(
                MIN_DAYS_BEFORE_TRANSFER,
                total_days - MIN_DAYS_AFTER_TRANSFER,
            )
            transfer_time = admit + timedelta(days=transfer_day)

            # Find the first assignment for this admission (initial room)
            orig = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.admission_id == admission_id)
                .order_by(RoomAssignment.start_time.asc())
                .first()
            )
            if not orig:
                print(f"SKIP patient {patient.patient_id}: no orig assignment found")
                continue

            orig_room_id = orig.room_id
            orig.end_time = transfer_time

            # Update occupancy for original room
            room_occupancy[orig_room_id] = [
                (s, e)
                for (s, e) in room_occupancy[orig_room_id]
                if not (s == admit and e == discharge)
            ]
            room_occupancy[orig_room_id].append((admit, transfer_time))

            # Pick destination room without overlap
            destination = None
            for r in transfer_rooms:
                if r.room_id == orig_room_id:
                    continue

                if all(
                    not overlaps(transfer_time, discharge, s, e)
                    for (s, e) in room_occupancy.get(r.room_id, [])
                ):
                    destination = r
                    break

            if destination is None:
                max_room_number += 1
                destination = Room(room_number=max_room_number)
                session.add(destination)
                session.flush()

                rooms.append(destination)
                transfer_rooms.append(destination)
                room_occupancy[destination.room_id] = []

            # Create second assignment segment (same admission_id)
            session.add(
                RoomAssignment(
                    admission_id=admission_id,
                    patient_id=patient.patient_id,
                    room_id=destination.room_id,
                    start_time=transfer_time,
                    end_time=discharge,
                )
            )

            print(
                f"Patient {patient.patient_id} moved "
                f"to room {destination.room_number} "
                f"[{transfer_time} → {discharge}]"
            )

            room_occupancy[destination.room_id].append((transfer_time, discharge))
            session.flush()

        # --------- Create runtime devices per room ----------
        for room in rooms:
            # MAIN sensor device
            mac_main = _random_mac(rng)
            dev_main = BLEDevice(mac_address=mac_main, label=f"Room {room.room_number} Main Sensor")
            dev_main.room_id = room.room_id
            dev_main.device_type = "sensor"
            dev_main.location = "main"

            for st in main_sensor_types:
                uuid = f"{mac_main}-{st}"
                dev_main.add_sensor(BLESensor(uuid=uuid, sensor_type=st, unit=_unit_for(st), parser=lambda b: None))
            ble_devices.append(dev_main)

            # TOILET sensor device
            mac_toilet = _random_mac(rng)
            dev_toilet = BLEDevice(mac_address=mac_toilet, label=f"Room {room.room_number} Toilet Sensor")
            dev_toilet.room_id = room.room_id
            dev_toilet.device_type = "sensor"
            dev_toilet.location = "toilet"

            for st in toilet_sensor_types:
                uuid = f"{mac_toilet}-{st}"
                dev_toilet.add_sensor(BLESensor(uuid=uuid, sensor_type=st, unit=_unit_for(st), parser=lambda b: None))
            ble_devices.append(dev_toilet)

            # Ventilation device (actuator, no sensors)
            vent = BLEDevice(mac_address=None, label=f"Room {room.room_number} Ventilation")
            vent.room_id = room.room_id
            vent.device_type = "ventilation"
            vent.location = "main"
            ble_devices.append(vent)

            # Speaker device (actuator, no sensors)
            if include_speaker:
                spk = BLEDevice(mac_address=None, label=f"Room {room.room_number} Speaker")
                spk.room_id = room.room_id
                spk.device_type = "speaker"
                spk.location = "main"
                ble_devices.append(spk)

            # Toilet heater device (actuator, no sensors)
            th = BLEDevice(mac_address=None, label=f"Room {room.room_number} Toilet Heater")
            th.room_id = room.room_id
            th.device_type = "toilet_heater"
            th.location = "toilet"
            ble_devices.append(th)

            # Toilet light device (actuator, no sensors)
            tl = BLEDevice(mac_address=None, label=f"Room {room.room_number} Toilet Light")
            tl.room_id = room.room_id
            tl.device_type = "toilet_light"
            tl.location = "toilet"
            ble_devices.append(tl)

    return ble_devices