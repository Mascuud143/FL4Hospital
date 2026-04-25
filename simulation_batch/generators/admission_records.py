from __future__ import annotations

# Build admission rows.
# - Computes age at admission time - age_at_time()
# - Computes weight at admission time - weight_at_admission()
# - Creates one admission row - create_admission()

import random
from datetime import datetime

from persistence.models.admission import Admission
from simulation_batch.csv_storage import write_model_row


def age_at_time(*, base_age_years: float, at: datetime, start_dt: datetime) -> float:
    days_since_start = (at - start_dt).total_seconds() / 86400.0
    return base_age_years + (days_since_start / 365.25)


def weight_at_admission(*, base_weight: float, rng: random.Random, sigma_kg: float = 1.5, max_delta_kg: float = 6.0) -> float:
    weight = base_weight + rng.gauss(0.0, sigma_kg)
    return max(base_weight - max_delta_kg, min(base_weight + max_delta_kg, weight))


def create_admission(
    *,
    session,
    patient_id: int,
    room_id: int,
    admitted_at: datetime,
    discharged_at: datetime,
    age: int,
    weight: float,
    diagnosis: str | None,
) -> Admission:
    admission = Admission(
        patient_id=patient_id,
        initial_room_id=room_id,
        admitted_at=admitted_at,
        discharged_at=discharged_at,
        age=age,
        weight=round(weight, 2),
        current_diagnosis=diagnosis,
    )
    write_model_row(admission)
    session.add(admission)
    session.flush()
    return admission
