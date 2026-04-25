from __future__ import annotations

# Build the first admissions.
# - Creates first admission rows - create_initial_admissions()
# - Creates first room rows - create_room()
# - Creates first room assignments - create_assignment()
# - Tracks room occupancy for later stays

from datetime import datetime, timedelta
from typing import Optional

from persistence.models.room import Room
from simulation_batch.generators.admission_records import age_at_time, create_admission, weight_at_admission
from simulation_batch.generators.room_assignments import create_assignment, create_room


def create_initial_admissions(
    *,
    session,
    patients,
    stay_days_list: list[int],
    per_patient_baseline: dict[int, dict],
    start_dt: datetime,
    end_dt: datetime,
    days: int,
    rng,
) -> tuple[list[Room], dict[int, list[tuple[datetime, datetime]]], list[dict], dict[int, datetime], int]:
    rooms: list[Room] = []
    room_occupancy: dict[int, list[tuple[datetime, datetime]]] = {}
    current_room: Optional[Room] = None
    current_offset_days = 0
    room_index = 0
    admissions_created: list[dict] = []
    last_discharge_by_patient_id: dict[int, datetime] = {}

    for idx, (patient, stay) in enumerate(zip(patients, stay_days_list)):
        if current_room is None or (current_offset_days + stay > days):
            room_index += 1
            current_offset_days = 0
            current_room = create_room(session=session, room_number=100 + room_index)
            rooms.append(current_room)
            room_occupancy.setdefault(current_room.room_id, [])

        admitted_at = start_dt + timedelta(days=current_offset_days)
        discharged_at = admitted_at + timedelta(days=stay)
        if admitted_at >= end_dt:
            break
        if discharged_at > end_dt:
            discharged_at = end_dt

        base = per_patient_baseline[idx]
        admission = create_admission(
            session=session,
            patient_id=patient.patient_id,
            room_id=current_room.room_id,
            admitted_at=admitted_at,
            discharged_at=discharged_at,
            age=int(round(age_at_time(base_age_years=base["age0"], at=admitted_at, start_dt=start_dt))),
            weight=weight_at_admission(base_weight=base["weight0"], rng=rng),
            diagnosis=base["current_diagnosis"],
        )
        create_assignment(
            session=session,
            admission_id=admission.admission_id,
            patient_id=patient.patient_id,
            room_id=current_room.room_id,
            start_time=admitted_at,
            end_time=discharged_at,
        )
        room_occupancy[current_room.room_id].append((admitted_at, discharged_at))
        admissions_created.append(
            {
                "patient_id": patient.patient_id,
                "admission_id": admission.admission_id,
                "admitted_at": admitted_at,
                "discharged_at": discharged_at,
            }
        )
        last_discharge_by_patient_id[patient.patient_id] = discharged_at
        current_offset_days += stay

    return rooms, room_occupancy, admissions_created, last_discharge_by_patient_id, room_index
