from __future__ import annotations

# Set up the hospital data.
# - Builds patient records
# - Builds first admissions
# - Builds readmissions
# - Adds room transfers
# - Adds room devices

import random
from datetime import date, datetime, timedelta
from typing import List

from ble import Device as BLEDevice
from persistence.database import session_scope
from simulation_batch.config import (
    CHANGE_ROOM_PROB,
    DAYS,
    MIN_DAYS_AFTER_TRANSFER,
    MIN_DAYS_BEFORE_TRANSFER,
    PATIENT_COUNT,
    START_DATE,
)
from simulation_batch.generators.device_setup import create_room_devices
from simulation_batch.generators.first_admissions import create_initial_admissions
from simulation_batch.generators.patient_records import build_patient_records, insert_patients
from simulation_batch.generators.readmissions import create_readmissions, generate_readmission_plan
from simulation_batch.generators.room_transfers import apply_room_transfers


def _ensure_datetime_midnight(d: date | datetime) -> datetime:
    if isinstance(d, date) and not isinstance(d, datetime):
        return datetime.combine(d, datetime.min.time())
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


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
    rng = random.Random(seed)
    start_dt = _ensure_datetime_midnight(start_date)
    end_dt = start_dt + timedelta(days=days)
    min_readmit_gap_days = 30
    extra_gap_max_days = 90
    diagnosis_change_prob = 0.5
    patients, stay_days_list, per_patient_baseline = build_patient_records(patient_count=patient_count, rng=rng)
    admission_totals = generate_readmission_plan(patient_count=patient_count, rng=rng)

    ble_devices: List[BLEDevice] = []

    with session_scope() as session:
        insert_patients(session=session, patients=patients)
        rooms, room_occupancy, admissions_created, last_discharge_by_patient_id, room_index = create_initial_admissions(
            session=session,
            patients=patients,
            stay_days_list=stay_days_list,
            per_patient_baseline=per_patient_baseline,
            start_dt=start_dt,
            end_dt=end_dt,
            days=days,
            rng=rng,
        )
        session.flush()
        create_readmissions(
            session=session,
            patients=patients,
            admission_totals=admission_totals,
            per_patient_baseline=per_patient_baseline,
            last_discharge_by_patient_id=last_discharge_by_patient_id,
            rooms=rooms,
            room_occupancy=room_occupancy,
            admissions_created=admissions_created,
            start_dt=start_dt,
            end_dt=end_dt,
            rng=rng,
            room_index=room_index,
            min_readmit_gap_days=min_readmit_gap_days,
            extra_gap_max_days=extra_gap_max_days,
            diagnosis_change_prob=diagnosis_change_prob,
        )
        session.flush()
        apply_room_transfers(
            session=session,
            admissions_created=admissions_created,
            change_room_prob=change_room_prob,
            min_days_before_transfer=min_days_before_transfer,
            min_days_after_transfer=min_days_after_transfer,
            rooms=rooms,
            room_occupancy=room_occupancy,
            rng=rng,
        )
        session.flush()

        if create_devices:
            ble_devices.extend(create_room_devices(rooms=rooms, rng=rng, include_speaker=include_speaker))

    return ble_devices


__all__ = ["seed_simulated_world"]
