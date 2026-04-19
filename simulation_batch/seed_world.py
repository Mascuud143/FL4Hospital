from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from persistence.database import session_scope
from persistence.models.room import Room
from persistence.models.patient import Patient
from persistence.models.admission import Admission
from persistence.models.room_assignment import RoomAssignment
from simulation_batch.csv_filestorage import write_model_row

from simulation_batch.config import (
    PATIENT_COUNT,
    DAYS,
    START_DATE,
    CHANGE_ROOM_PROB,
    MIN_DAYS_BEFORE_TRANSFER,
    MIN_DAYS_AFTER_TRANSFER,
)

from simulation_batch.generators.patients import generate_patients
from simulation_batch.generators.name_gen import generate_name
from simulation_batch.generators.age_height_weight_gender_gen import age_height_weight__gender_generator

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
        "temperature": "C",
        "humidity": "%",
        "co2": "ppm",
        "light": "lux",
        "sound": "dB",
    }.get(st, "")


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def _find_or_create_room_for_window(
    *,
    session,
    rooms: list[Room],
    room_occupancy: dict[int, list[Tuple[datetime, datetime]]],
    window_start: datetime,
    window_end: datetime,
    room_number_seed: int,
) -> Room:
    """
    Pick an existing room that has no overlap with [window_start, window_end),
    else create a new room.
    """
    # try existing rooms
    for r in rooms:
        occ = room_occupancy.get(r.room_id, [])
        if all(not _overlaps(window_start, window_end, s, e) for (s, e) in occ):
            return r

    # create new room
    next_number = room_number_seed
    if rooms:
        next_number = max(int(x.room_number) for x in rooms) + 1

    new_room = Room(room_number=next_number)
    write_model_row(new_room)
    session.add(new_room)
    session.flush()
    rooms.append(new_room)
    room_occupancy[new_room.room_id] = []
    return new_room


# -------------------------
# Seeding
# -------------------------

def seed_simulated_world(
    *,
    seed: int = 42,
    patient_count: int = PATIENT_COUNT,
    days: int = DAYS,
    start_date: date | datetime = START_DATE,
    change_room_prob: float = CHANGE_ROOM_PROB,
    min_days_before_transfer: int = MIN_DAYS_BEFORE_TRANSFER,
    min_days_after_transfer: int = MIN_DAYS_AFTER_TRANSFER,
    create_devices: bool = True,
    include_speaker: bool = True,
) -> List[BLEDevice]:
    """
    Seeds DB:
      - Rooms (dynamic count, created as needed)
      - Patients (identity-only)
      - Admissions (per-stay attributes, admitted/discharged timestamps)
      - RoomAssignments (linked to an Admission via admission_id)

    Readmission logic:
      - 20% of unique patients are "recurrent"
      - among recurrent:
          * 40% have 3 admissions total
          * 5% have 5 admissions total (over 4 times)
          * the rest have 2 admissions total
      - readmission start time constraint:
          * next_admit >= prev_discharge + 30 days
          * after that, random extra gap is added

    IMPORTANT realism fixes:
      - Age is not re-sampled for readmissions; it progresses slightly with time.
      - Weight is anchored to patient baseline and only drifts slightly per admission.
      - Diagnosis defaults to stable across admissions (optionally can change with small probability).

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
    end_dt = start_dt + timedelta(days=days)

    main_sensor_types = ("temperature", "humidity", "co2", "light", "sound")
    toilet_sensor_types = ("temperature",)

    # ---- Readmission gap settings ----
    MIN_READMIT_GAP_DAYS = 30
    EXTRA_GAP_MAX_DAYS = 90  # after 1 month, random 0..90 days

    # ---- Patient realism knobs ----
    WEIGHT_DRIFT_SIGMA_KG = 1.5     # small per-admission noise
    MAX_WEIGHT_DELTA_KG = 6.0       # clamp extreme changes
    DIAGNOSIS_CHANGE_PROB = 0.5     # keep stable by default; set e.g. 0.2 if you want changes

    # --------- Build patients (identity-only) ----------
    patients: list[Patient] = []
    stay_days_list: list[int] = []

    # Store baseline patient attributes used for all admissions
    per_patient_baseline: dict[int, dict] = {}  # idx -> {age0, weight0, diagnosis}

    generated = generate_patients(patient_count)
    for idx, p in enumerate(generated):
        age_hw = age_height_weight__gender_generator()

        patient = Patient(
            name=generate_name(),
            ethnicity=p.get("ethnicity"),
            gender=age_hw["gender"],
            height=age_hw["height"],
        )

        per_patient_baseline[idx] = {
            "age0": float(age_hw["age"]),
            "weight0": float(age_hw["weight"]),
            "current_diagnosis": p.get("diagnosis"),
        }

        stay_days = rng.randint(5, 20)
        patients.append(patient)
        stay_days_list.append(stay_days)

    # --------- Decide readmission counts ----------
    N = patient_count
    recurrent_count = int(round(0.20 * N))

    patient_indices = list(range(N))
    rng.shuffle(patient_indices)
    recurrent_indices = set(patient_indices[:recurrent_count])

    recurrent_list = list(recurrent_indices)
    rng.shuffle(recurrent_list)

    r3_count = int(round(0.40 * recurrent_count))
    r4p_count = int(round(0.05 * recurrent_count))
    # remainder are 2 admissions
    r3_set = set(recurrent_list[:r3_count])
    r4p_set = set(recurrent_list[r3_count:r3_count + r4p_count])

    def admission_total_for_idx(i: int) -> int:
        if i not in recurrent_indices:
            return 1
        if i in r3_set:
            return 3
        if i in r4p_set:
            return 5  # "over 4 times"
        return 2

    def age_at_time(base_age_years: float, at: datetime) -> float:
        # Progress age with elapsed simulation time (in years).
        # Over 10 simulated days this barely changes, but it stays logically correct.
        days_since_start = (at - start_dt).total_seconds() / 86400.0
        return base_age_years + (days_since_start / 365.25)

    def weight_at_admission(base_weight: float) -> float:
        w = base_weight + rng.gauss(0.0, WEIGHT_DRIFT_SIGMA_KG)
        # clamp so it doesn't jump unrealistically
        w = max(base_weight - MAX_WEIGHT_DELTA_KG, min(base_weight + MAX_WEIGHT_DELTA_KG, w))
        return w

    ble_devices: List[BLEDevice] = []

    with session_scope() as session:
        # insert patients first so they get IDs
        for patient in patients:
            write_model_row(patient)
            session.add(patient)
        session.flush()

        # --------- Rooms + occupancy ----------
        rooms: list[Room] = []
        room_occupancy: dict[int, list[tuple[datetime, datetime]]] = {}

        # Keep original packing logic for initial admissions
        current_room: Optional[Room] = None
        current_offset_days = 0
        room_index = 0

        # Track created admissions for transfer logic later
        admissions_created: list[dict] = []
        # For scheduling readmissions
        last_discharge_by_patient_id: dict[int, datetime] = {}

        # --------- Create initial admission for every patient ----------
        for idx, (patient, stay) in enumerate(zip(patients, stay_days_list)):
            if current_room is None or (current_offset_days + stay > days):
                room_index += 1
                current_offset_days = 0
                current_room = Room(room_number=100 + room_index)
                write_model_row(current_room)
                session.add(current_room)
                session.flush()
                rooms.append(current_room)
                room_occupancy.setdefault(current_room.room_id, [])

            admitted_at = start_dt + timedelta(days=current_offset_days)
            discharged_at = admitted_at + timedelta(days=stay)

            if admitted_at >= end_dt:
                break
            if discharged_at > end_dt:
                discharged_at = end_dt

            base = per_patient_baseline[idx]
            adm = Admission(
                patient_id=patient.patient_id,
                initial_room_id=current_room.room_id,
                admitted_at=admitted_at,
                discharged_at=discharged_at,
                age=int(round(age_at_time(base["age0"], admitted_at))),
                weight=round(weight_at_admission(base["weight0"]), 2),
                current_diagnosis=base["current_diagnosis"],
            )
            write_model_row(adm)
            session.add(adm)
            session.flush()

            row = RoomAssignment(
                admission_id=adm.admission_id,
                patient_id=patient.patient_id,
                room_id=current_room.room_id,
                start_time=admitted_at,
                end_time=discharged_at,
            )
            write_model_row(row)
            session.add(row)

            room_occupancy[current_room.room_id].append((admitted_at, discharged_at))
            admissions_created.append(
                dict(
                    patient_id=patient.patient_id,
                    admission_id=adm.admission_id,
                    admitted_at=admitted_at,
                    discharged_at=discharged_at,
                )
            )
            last_discharge_by_patient_id[patient.patient_id] = discharged_at

            current_offset_days += stay

        session.flush()

        # --------- Create readmissions (additional admissions per recurrent patient) ----------
        for idx, patient in enumerate(patients):
            total_adm = admission_total_for_idx(idx)
            if total_adm <= 1:
                continue

            pid = patient.patient_id
            if pid not in last_discharge_by_patient_id:
                continue

            base = per_patient_baseline[idx]
            base_age = float(base["age0"])
            base_weight = float(base["weight0"])
            base_dx = base["current_diagnosis"]

            for _k in range(2, total_adm + 1):
                prev_discharge = last_discharge_by_patient_id[pid]
                earliest = prev_discharge + timedelta(days=MIN_READMIT_GAP_DAYS)
                extra_gap = rng.randint(0, EXTRA_GAP_MAX_DAYS)
                admitted_at = earliest + timedelta(days=extra_gap)

                if admitted_at >= end_dt:
                    break

                stay = rng.randint(5, 20)
                discharged_at = admitted_at + timedelta(days=stay)
                if discharged_at > end_dt:
                    discharged_at = end_dt

                room_for_adm = _find_or_create_room_for_window(
                    session=session,
                    rooms=rooms,
                    room_occupancy=room_occupancy,
                    window_start=admitted_at,
                    window_end=discharged_at,
                    room_number_seed=100 + (room_index + 1),
                )

                # Diagnosis behavior (default stable)
                dx = base_dx
                if DIAGNOSIS_CHANGE_PROB > 0 and rng.random() < DIAGNOSIS_CHANGE_PROB:
                    dx = generate_patients(1)[0].get("diagnosis")

                adm = Admission(
                    patient_id=pid,
                    initial_room_id=room_for_adm.room_id,
                    admitted_at=admitted_at,
                    discharged_at=discharged_at,
                    age=int(round(age_at_time(base_age, admitted_at))),
                    weight=round(weight_at_admission(base_weight), 2),
                    current_diagnosis=dx,
                )
                write_model_row(adm)
                session.add(adm)
                session.flush()

                row = RoomAssignment(
                    admission_id=adm.admission_id,
                    patient_id=pid,
                    room_id=room_for_adm.room_id,
                    start_time=admitted_at,
                    end_time=discharged_at,
                )
                write_model_row(row)
                session.add(row)

                room_occupancy.setdefault(room_for_adm.room_id, [])
                room_occupancy[room_for_adm.room_id].append((admitted_at, discharged_at))

                admissions_created.append(
                    dict(
                        patient_id=pid,
                        admission_id=adm.admission_id,
                        admitted_at=admitted_at,
                        discharged_at=discharged_at,
                    )
                )

                last_discharge_by_patient_id[pid] = discharged_at

        session.flush()

        # ============================================================
        # TRANSFER PHASE: ROOM CHANGE WHILE ADMITTED (per admission)
        # ============================================================
        print(f">>> TRANSFER PHASE START (prob={change_room_prob}) <<<")

        transfer_rooms = list(rooms)
        max_room_number = max(int(r.room_number) for r in rooms) if rooms else 100

        for adm_info in admissions_created:
            if rng.random() > change_room_prob:
                continue

            admit = adm_info["admitted_at"]
            discharge = adm_info["discharged_at"]
            admission_id = adm_info["admission_id"]
            pid = adm_info["patient_id"]

            total_days = (discharge - admit).days
            if total_days < min_days_before_transfer + min_days_after_transfer + 1:
                continue

            transfer_day = rng.randint(
                min_days_before_transfer,
                max(min_days_before_transfer, total_days - min_days_after_transfer),
            )
            transfer_time = admit + timedelta(days=transfer_day)
            if transfer_time <= admit or transfer_time >= discharge:
                continue

            orig = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.admission_id == admission_id)
                .order_by(RoomAssignment.start_time.asc())
                .first()
            )
            if not orig:
                continue

            orig_room_id = orig.room_id

            old_end = orig.end_time
            orig.end_time = transfer_time

            # Update occupancy for original room segment
            occ_list = room_occupancy.get(orig_room_id, [])
            updated = []
            replaced = False
            for (s, e) in occ_list:
                if not replaced and s == admit and e == old_end:
                    updated.append((admit, transfer_time))
                    replaced = True
                else:
                    updated.append((s, e))
            room_occupancy[orig_room_id] = updated

            # Find destination room (no overlap)
            destination = None
            for r in transfer_rooms:
                if r.room_id == orig_room_id:
                    continue
                occ = room_occupancy.get(r.room_id, [])
                if all(not _overlaps(transfer_time, discharge, s, e) for (s, e) in occ):
                    destination = r
                    break

            if destination is None:
                max_room_number += 1
                destination = Room(room_number=max_room_number)
                write_model_row(destination)
                session.add(destination)
                session.flush()
                rooms.append(destination)
                transfer_rooms.append(destination)
                room_occupancy.setdefault(destination.room_id, [])

            row = RoomAssignment(
                admission_id=admission_id,
                patient_id=pid,
                room_id=destination.room_id,
                start_time=transfer_time,
                end_time=discharge,
            )
            write_model_row(row)
            session.add(row)

            room_occupancy.setdefault(destination.room_id, [])
            room_occupancy[destination.room_id].append((transfer_time, discharge))

        session.flush()

        if create_devices:
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
                    dev_main.add_sensor(
                        BLESensor(uuid=uuid, sensor_type=st, unit=_unit_for(st), parser=lambda b: None)
                    )
                ble_devices.append(dev_main)

                # TOILET sensor device
                mac_toilet = _random_mac(rng)
                dev_toilet = BLEDevice(mac_address=mac_toilet, label=f"Room {room.room_number} Toilet Sensor")
                dev_toilet.room_id = room.room_id
                dev_toilet.device_type = "sensor"
                dev_toilet.location = "toilet"

                for st in toilet_sensor_types:
                    uuid = f"{mac_toilet}-{st}"
                    dev_toilet.add_sensor(
                        BLESensor(uuid=uuid, sensor_type=st, unit=_unit_for(st), parser=lambda b: None)
                    )
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
