import os
import sys

import numpy as np
import pandas as pd

_AI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AI_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from simulation_batch.generators.diagnosis_profiles import DIAGNOSES

DIAGNOSIS_NAMES = sorted(DIAGNOSES.keys())
DIAGNOSIS_TO_INDEX = {name: idx for idx, name in enumerate(DIAGNOSIS_NAMES)}

SYMPTOM_NAMES = sorted({symptom for item in DIAGNOSES.values() for symptom in item["symptoms"]} | {""})
SYMPTOM_TO_INDEX = {name: idx for idx, name in enumerate(SYMPTOM_NAMES)}

MEDICATION_NAMES = sorted({med for item in DIAGNOSES.values() for med in item["medications"].keys()})
MEDICATION_TO_INDEX = {name: idx for idx, name in enumerate(MEDICATION_NAMES)}

MAX_MEDICATION_SLOTS = max(len(item["medications"]) for item in DIAGNOSES.values())
TIME_VECTOR_SLOTS = 48

BASE_NUMERIC_COLUMNS = [
    "age",
    "height",
    "weight",
    "gender_binary",
    "body_temperature",
    "bp_systolic",
    "bp_diastolic",
]
TIME_COLUMNS = [f"time_slot_{slot:02d}" for slot in range(TIME_VECTOR_SLOTS)]
DIAGNOSIS_COLUMNS = [f"diagnosis_{idx}" for idx, _ in enumerate(DIAGNOSIS_NAMES)]
SYMPTOM_COLUMNS = [f"symptom_{idx}" for idx, _ in enumerate(SYMPTOM_NAMES)]


def medication_type_columns(slot: int) -> list[str]:
    return [f"med_{slot}_type_{idx}" for idx, _ in enumerate(MEDICATION_NAMES)]


def medication_schedule_columns(slot: int) -> list[str]:
    return [f"med_{slot}_sched_slot_{time_slot:02d}" for time_slot in range(TIME_VECTOR_SLOTS)]


MEDICATION_TYPE_COLUMNS = {
    slot: medication_type_columns(slot)
    for slot in range(1, MAX_MEDICATION_SLOTS + 1)
}
MEDICATION_SCHEDULE_COLUMNS = {
    slot: medication_schedule_columns(slot)
    for slot in range(1, MAX_MEDICATION_SLOTS + 1)
}

INPUT_COLUMNS = [
    *BASE_NUMERIC_COLUMNS,
    *TIME_COLUMNS,
    *DIAGNOSIS_COLUMNS,
    *SYMPTOM_COLUMNS,
    *[col for slot in range(1, MAX_MEDICATION_SLOTS + 1) for col in MEDICATION_TYPE_COLUMNS[slot]],
    *[col for slot in range(1, MAX_MEDICATION_SLOTS + 1) for col in MEDICATION_SCHEDULE_COLUMNS[slot]],
]

TARGET_COLUMNS = ["y_temp_main", "y_temp_toilet", "y_light", "y_sound", "y_airflow"]
AIRFLOW_INDEX = 4
CHANGE_METADATA_COLUMNS = [
    "curr_temp_main_eval",
    "curr_temp_toilet_eval",
    "curr_light_eval",
    "curr_sound_eval",
    "curr_airflow_eval",
    "y_any_change",
]
CHANGE_BASELINE_COLUMNS = CHANGE_METADATA_COLUMNS[:5]


def gender_to_binary(value: str | None) -> int:
    return 1 if str(value or "").strip().lower() == "male" else 0


def make_time_vector(hour: int, minute: int = 0) -> list[int]:
    vector = [0] * TIME_VECTOR_SLOTS
    half_hour_offset = 1 if int(minute) >= 30 else 0
    slot = int(hour) * 2 + half_hour_offset
    if 0 <= slot < TIME_VECTOR_SLOTS:
        vector[slot] = 1
    return vector


def make_one_hot(index: int | None, size: int) -> list[int]:
    vector = [0] * size
    if index is not None and 0 <= index < size:
        vector[index] = 1
    return vector


def normalize_schedule(hours: list[int]) -> list[int]:
    vector = [0] * TIME_VECTOR_SLOTS
    for hour in hours:
        if hour is None or int(hour) < 0:
            continue
        slot = int(hour) * 2
        if 0 <= slot < TIME_VECTOR_SLOTS:
            vector[slot] = 1
    return vector


def medication_slots_for_diagnosis(diagnosis: str) -> list[tuple[str, list[int]]]:
    medications = DIAGNOSES.get(diagnosis, {}).get("medications", {})
    return list(medications.items())[:MAX_MEDICATION_SLOTS]


def row_to_input_vector(df: pd.DataFrame) -> np.ndarray:
    return df[INPUT_COLUMNS].to_numpy(dtype=np.float64)


def next_hour_change_flags(current_values: np.ndarray, next_values: np.ndarray) -> np.ndarray:
    current_reg = current_values[:, :AIRFLOW_INDEX]
    next_reg = next_values[:, :AIRFLOW_INDEX]
    current_airflow = np.rint(current_values[:, AIRFLOW_INDEX]).astype(int)
    next_airflow = np.rint(next_values[:, AIRFLOW_INDEX]).astype(int)
    regression_changed = np.any(current_reg != next_reg, axis=1)
    airflow_changed = current_airflow != next_airflow
    return (regression_changed | airflow_changed).astype(int)
