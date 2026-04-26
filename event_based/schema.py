import os
import sys

import numpy as np
import pandas as pd

_AI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AI_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from simulation_batch.generators.patients import DIAGNOSES

DIAGNOSIS_NAMES = sorted(DIAGNOSES.keys())
DIAGNOSIS_TO_INDEX = {name: idx for idx, name in enumerate(DIAGNOSIS_NAMES)}

SYMPTOM_NAMES = sorted({""} | set(DIAGNOSES.keys()) | {symptom for item in DIAGNOSES.values() for symptom in item["symptoms"]})
SYMPTOM_TO_INDEX = {name: idx for idx, name in enumerate(SYMPTOM_NAMES)}

MEDICATION_NAMES = sorted({med for item in DIAGNOSES.values() for med in item["medications"].keys()})
MEDICATION_TO_INDEX = {name: idx for idx, name in enumerate(MEDICATION_NAMES)}

BASE_NUMERIC_COLUMNS = [
    "age",
    "height",
    "weight",
    "gender_binary",
    "body_temperature",
    "bp_systolic",
    "bp_diastolic",
    "hours_since_last_medication",
    "hours_since_last_symptom_change",
    "hours_since_last_comfort",
    "prev_target_temp_main",
    "prev_target_temp_toilet",
    "prev_target_light",
    "prev_target_sound",
    "prev_target_airflow",
]

EVENT_TYPE_COLUMNS = ["event_is_medication", "event_is_visit"]
DIAGNOSIS_COLUMNS = [f"diagnosis_{idx}" for idx, _ in enumerate(DIAGNOSIS_NAMES)]
SYMPTOM_COLUMNS = [f"symptom_{idx}" for idx, _ in enumerate(SYMPTOM_NAMES)]
ACTIVE_MEDICATION_COLUMNS = [f"active_med_{idx}" for idx, _ in enumerate(MEDICATION_NAMES)]
TRIGGER_MEDICATION_COLUMNS = [f"trigger_med_{idx}" for idx, _ in enumerate(MEDICATION_NAMES)]

FEATURE_COLUMNS = [
    *BASE_NUMERIC_COLUMNS,
    *EVENT_TYPE_COLUMNS,
    *DIAGNOSIS_COLUMNS,
    *SYMPTOM_COLUMNS,
    *ACTIVE_MEDICATION_COLUMNS,
    *TRIGGER_MEDICATION_COLUMNS,
]

FEATURE_GROUPS = {
    "baseline_numeric": BASE_NUMERIC_COLUMNS,
    "event_type": EVENT_TYPE_COLUMNS,
    "diagnosis": DIAGNOSIS_COLUMNS,
    "symptoms": SYMPTOM_COLUMNS,
    "active_medications": ACTIVE_MEDICATION_COLUMNS,
    "trigger_medication": TRIGGER_MEDICATION_COLUMNS,
}

TARGET_COLUMNS = [
    "y_target_temp_main",
    "y_target_temp_toilet",
    "y_target_light",
    "y_target_sound",
    "y_target_airflow",
]
REGRESSION_TARGET_COLUMNS = TARGET_COLUMNS[:4]
IDENTIFIER_COLUMNS = [
    "admission_id",
    "patient_id",
    "room_id",
    "event_time",
    "event_type",
    "event_detail",
    "effect_window_start",
    "next_change_time",
    "target_time",
]


def gender_to_binary(value: str | None) -> int:
    return 1 if str(value or "").strip().lower() == "male" else 0


def make_one_hot(index: int | None, size: int) -> list[int]:
    vector = [0] * size
    if index is not None and 0 <= index < size:
        vector[index] = 1
    return vector


def row_to_input_vector(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
