from __future__ import annotations

# Build patient records.
# - Creates patient diagnosis - generate_patients()
# - Creates base age, height, weight - age_height_weight__gender_generator()
# - Creates patient names - generate_name()
# - Creates stay lengths - random int between 5 and 20
# - Records a patient in memory - build_patient_records()
# - Records patients in db - insert_patients()
import random

from persistence.models.patient import Patient
from simulation_batch.csv_storage import write_model_row
from simulation_batch.generators.patient_attributes import age_height_weight__gender_generator
from simulation_batch.generators.diagnosis_profiles import generate_patients
from simulation_batch.generators.patient_names import generate_name


def build_patient_records(*, patient_count: int, rng: random.Random) -> tuple[list[Patient], list[int], dict[int, dict]]:
    patients: list[Patient] = []
    stay_days_list: list[int] = []
    per_patient_baseline: dict[int, dict] = {}
    generated = generate_patients(patient_count)
    for idx, profile in enumerate(generated):
        age_hw = age_height_weight__gender_generator()
        patient = Patient(
            name=generate_name(),
            ethnicity=profile.get("ethnicity"),
            gender=age_hw["gender"],
            height=age_hw["height"],
        )
        per_patient_baseline[idx] = {
            "age0": float(age_hw["age"]),
            "weight0": float(age_hw["weight"]),
            "current_diagnosis": profile.get("diagnosis"),
        }
        patients.append(patient)
        stay_days_list.append(rng.randint(5, 20))
    return patients, stay_days_list, per_patient_baseline


def insert_patients(*, session, patients: list[Patient]) -> None:
    for patient in patients:
        write_model_row(patient)
        session.add(patient)
    session.flush()
