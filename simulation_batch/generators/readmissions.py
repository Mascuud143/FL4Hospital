from __future__ import annotations

# Build readmissions.
# - Decides later stays to add - generate_readmission_plan()
# - Creates readmission rows - create_readmissions()
# - Creates new room assignments
# - Updates room occupancy

from datetime import datetime, timedelta

from simulation_batch.generators.admission_records import age_at_time, create_admission, weight_at_admission
from simulation_batch.generators.diagnosis_profiles import generate_patients
from simulation_batch.generators.room_assignments import create_assignment, find_or_create_room_for_window


def generate_readmission_plan(*, patient_count: int, rng) -> dict[int, int]:
    recurrent_count = int(round(0.20 * patient_count))
    patient_indices = list(range(patient_count))
    rng.shuffle(patient_indices)
    recurrent_indices = set(patient_indices[:recurrent_count])
    recurrent_list = list(recurrent_indices)
    rng.shuffle(recurrent_list)
    r3_count = int(round(0.40 * recurrent_count))
    r5_count = int(round(0.05 * recurrent_count))
    r3_set = set(recurrent_list[:r3_count])
    r5_set = set(recurrent_list[r3_count:r3_count + r5_count])
    totals: dict[int, int] = {}
    for index in range(patient_count):
        if index not in recurrent_indices:
            totals[index] = 1
        elif index in r3_set:
            totals[index] = 3
        elif index in r5_set:
            totals[index] = 5
        else:
            totals[index] = 2
    return totals


def create_readmissions(
    *,
    session,
    patients,
    admission_totals: dict[int, int],
    per_patient_baseline: dict[int, dict],
    last_discharge_by_patient_id: dict[int, datetime],
    rooms,
    room_occupancy,
    admissions_created: list[dict],
    start_dt: datetime,
    end_dt: datetime,
    rng,
    room_index: int,
    min_readmit_gap_days: int,
    extra_gap_max_days: int,
    diagnosis_change_prob: float,
) -> None:
    for idx, patient in enumerate(patients):
        total_admissions = admission_totals[idx]
        if total_admissions <= 1:
            continue
        patient_id = patient.patient_id
        if patient_id not in last_discharge_by_patient_id:
            continue
        base = per_patient_baseline[idx]
        base_age = float(base["age0"])
        base_weight = float(base["weight0"])
        base_diagnosis = base["current_diagnosis"]
        for _ in range(2, total_admissions + 1):
            prev_discharge = last_discharge_by_patient_id[patient_id]
            earliest = prev_discharge + timedelta(days=min_readmit_gap_days)
            admitted_at = earliest + timedelta(days=rng.randint(0, extra_gap_max_days))
            if admitted_at >= end_dt:
                break
            stay = rng.randint(5, 20)
            discharged_at = admitted_at + timedelta(days=stay)
            if discharged_at > end_dt:
                discharged_at = end_dt
            room_for_admission = find_or_create_room_for_window(
                session=session,
                rooms=rooms,
                room_occupancy=room_occupancy,
                window_start=admitted_at,
                window_end=discharged_at,
                room_number_seed=100 + (room_index + 1),
            )
            diagnosis = base_diagnosis
            if diagnosis_change_prob > 0 and rng.random() < diagnosis_change_prob:
                diagnosis = generate_patients(1)[0].get("diagnosis")
            admission = create_admission(
                session=session,
                patient_id=patient_id,
                room_id=room_for_admission.room_id,
                admitted_at=admitted_at,
                discharged_at=discharged_at,
                age=int(round(age_at_time(base_age_years=base_age, at=admitted_at, start_dt=start_dt))),
                weight=weight_at_admission(base_weight=base_weight, rng=rng),
                diagnosis=diagnosis,
            )
            create_assignment(
                session=session,
                admission_id=admission.admission_id,
                patient_id=patient_id,
                room_id=room_for_admission.room_id,
                start_time=admitted_at,
                end_time=discharged_at,
            )
            room_occupancy.setdefault(room_for_admission.room_id, [])
            room_occupancy[room_for_admission.room_id].append((admitted_at, discharged_at))
            admissions_created.append(
                {
                    "patient_id": patient_id,
                    "admission_id": admission.admission_id,
                    "admitted_at": admitted_at,
                    "discharged_at": discharged_at,
                }
            )
            last_discharge_by_patient_id[patient_id] = discharged_at
