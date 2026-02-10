from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence

from persistence.database import session_scope
from persistence.models.room import Room
from persistence.models.patient import Patient
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference

from simulation_batch.config import (
    ROOM_COUNT,
    PATIENT_COUNT,
    DAYS,
    START_DATE,
)

from simulation_batch.generators.patients import generate_patients
from simulation_batch.generators.comfort import generate_comfort
from simulation_batch.generators.name_gen import generate_name
from simulation_batch.generators.age_height_weight_gen import age_height_weight_generator

# Runtime BLE objects (not DB models)
from ble import Device as BLEDevice, Sensor as BLESensor


def _ensure_datetime_midnight(d: date | datetime) -> datetime:
    """Convert date/datetime to datetime at 00:00:00."""
    if isinstance(d, date) and not isinstance(d, datetime):
        return datetime.combine(d, datetime.min.time())
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _random_mac(rng: random.Random) -> str:
    """Generate a locally-administered-ish fake MAC in your '02:00:00:..' style."""
    return "02:00:00:%02x:%02x:%02x" % (
        rng.randint(0, 255),
        rng.randint(0, 255),
        rng.randint(0, 255),
    )


def seed_simulated_world(
    *,
    seed: int = 42,
    room_count: int = ROOM_COUNT,
    patient_count: int = PATIENT_COUNT,
    days: int = DAYS,
    start_date: date | datetime = START_DATE,
    # how many comfort "change points" per day (your old logic was fixed [0,6,12,18])
    comfort_hours: Sequence[int] = (0, 6, 12, 18),
    # how many BLE devices per room (like your old "3 nordic per room")
    devices_per_room: int = 3,
    # which sensors each BLE device exposes in SIM mode
    sensor_types: Sequence[str] = ("temperature", "humidity", "co2", "light", "sound"),
) -> List[BLEDevice]:
    """
    Seeds Rooms + Patients + RoomAssignments + ComfortPreference into the DB.

    Returns runtime BLE Device objects (ble.Device) with:
      - mac_address
      - room_id
      - sensors (ble.Sensor) for sensor_types

    NOTE:
      - This function intentionally does NOT create DB Device/Sensor rows.
        Your main should call seed_devices_and_sensors(devices, ...) after this,
        so db_sink can resolve devices/sensors.
    """
    rng = random.Random(seed)
    start_date_dt = _ensure_datetime_midnight(start_date)

    # ---------- Build patients ----------
    patients: list[Patient] = []
    patient_stay_days: list[int] = []

    for p in generate_patients(patient_count):
        age_hw = age_height_weight_generator()
        patient = Patient(
            age=age_hw["age"],
            name=generate_name(),
            current_diagnosis=p["diagnosis"],
            ethnicity=p["ethnicity"],
            gender=rng.choice(["Male", "Female"]),
            height=age_hw["height"],
            weight=age_hw["weight"],
        )
        stay_days = rng.randint(5, 20)
        patient.stay_days = stay_days  # if your model has this column
        patients.append(patient)
        patient_stay_days.append(stay_days)

    # ---------- Seed into DB ----------
    ble_devices: List[BLEDevice] = []

    with session_scope() as session:
        # 1) Rooms
        rooms: list[Room] = []
        for i in range(room_count):
            room = Room(room_number=100 + i)
            session.add(room)
            rooms.append(room)
        session.flush()  # ensures room_id populated

        # 2) Patients
        for patient in patients:
            session.add(patient)
        session.flush()  # ensures patient_id populated

        # 3) Assign patients to rooms over the timeline (similar to your current logic)
        current_room_index = 0
        current_room_stay_sum = 0

        for i, stay in enumerate(patient_stay_days):
            patient = patients[i]

            # move to next room if this stay would exceed total simulation DAYS
            if current_room_stay_sum + stay > days:
                current_room_index += 1
                current_room_stay_sum = 0
                if current_room_index >= len(rooms):
                    # If we run out of rooms, wrap or clamp; here we clamp to last room.
                    current_room_index = len(rooms) - 1

            room = rooms[current_room_index]

            # admission & release
            admission = start_date_dt + timedelta(days=current_room_stay_sum)
            release = admission + timedelta(days=stay)

            patient.admission_date = admission
            patient.release_date = release

            assignment = RoomAssignment(
                patient_id=patient.patient_id,
                room_id=room.room_id,
                start_time=admission,
                end_time=release,
            )
            session.add(assignment)

            # 4) Comfort preferences (targets) at given hours each day (for 1-day profile)
            base_date = _ensure_datetime_midnight(admission)

            for hour in comfort_hours:
                comfort_pref = generate_comfort()

                # Optional: your hour-based tweaks (kept from your code)
                if hour == 0:  # midnight
                    comfort_pref["temperature"] = round(rng.uniform(20.0, 22.0), 2)
                    comfort_pref["light_intensity"] = 0.0
                    comfort_pref["sound_level"] = round(rng.uniform(0, 20), 2)
                elif hour == 6:  # morning
                    comfort_pref["temperature"] = round(rng.uniform(21.0, 23.0), 2)
                elif hour == 12:  # afternoon
                    comfort_pref["temperature"] = round(rng.uniform(22.0, 24.0), 2)
                    comfort_pref["light_intensity"] = round(rng.uniform(20, 50), 2)
                elif hour == 18:  # evening
                    comfort_pref["temperature"] = round(rng.uniform(20.0, 21.0), 2)
                    comfort_pref["sound_level"] = round(rng.uniform(0, 20), 2)
                    comfort_pref["light_intensity"] = round(rng.uniform(20, 50), 2)

                comfort = ComfortPreference(
                    patient_id=patient.patient_id,
                    room_id=room.room_id,
                    timestamp=base_date + timedelta(hours=int(hour)),
                    temperature=comfort_pref["temperature"],
                    light_intensity=comfort_pref["light_intensity"],
                    sound_level=comfort_pref["sound_level"],
                    ventilation=comfort_pref["ventilation"],
                    source=comfort_pref["source"],
                )
                session.add(comfort)

            # advance timeline within this room
            current_room_stay_sum += stay

        session.flush()

        # ---------- Create runtime BLE devices for each room ----------
        # (DB seeding of these devices happens later in seed_devices_and_sensors)
        for room in rooms:
            for j in range(devices_per_room):
                mac = _random_mac(rng)
                d = BLEDevice(mac_address=mac, label=f"Sim Room {room.room_number} Dev {j+1}")
                d.room_id = room.room_id  # important for comfort-aware simulation

                # Attach runtime sensors.
                # UUID choice: keep it unique per (device, sensor_type) to avoid ambiguity.
                for st in sensor_types:
                    unit = {
                        "temperature": "°C",
                        "humidity": "%",
                        "co2": "ppm",
                        "light": "lux",
                        "sound": "dB",
                    }.get(st, "")

                    # Unique uuid per device+sensor_type (works great with db_sink resolution)
                    uuid = f"{mac}-{st}"

                    # parser usually not used in simulated mode; keep a harmless default
                    d.add_sensor(BLESensor(uuid=uuid, sensor_type=st, unit=unit, parser=lambda b: None))

                ble_devices.append(d)

    return ble_devices
