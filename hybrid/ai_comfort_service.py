from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import sys

import numpy as np
import pandas as pd

from hybrid.ai_config import load_ai_config
from persistence.models import Admission, ComfortPreference, Data, Device, Medication, Patient, RoomAssignment, Sensor, Visit


_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from event_based.schema import (
    DIAGNOSIS_NAMES as _EVENT_DIAGNOSIS_NAMES,
    DIAGNOSIS_TO_INDEX as _EVENT_DIAGNOSIS_TO_INDEX,
    FEATURE_COLUMNS as _EVENT_FEATURE_COLUMNS,
    MEDICATION_NAMES as _EVENT_MEDICATION_NAMES,
    MEDICATION_TO_INDEX as _EVENT_MEDICATION_TO_INDEX,
    SYMPTOM_NAMES as _EVENT_SYMPTOM_NAMES,
    SYMPTOM_TO_INDEX as _EVENT_SYMPTOM_TO_INDEX,
    row_to_input_vector as _event_row_to_input_vector,
)
from k_hours_based.next_hour_schema import (
    AIRFLOW_INDEX as _K_HOURS_AIRFLOW_INDEX,
    DIAGNOSIS_NAMES as _K_HOURS_DIAGNOSIS_NAMES,
    DIAGNOSIS_TO_INDEX as _K_HOURS_DIAGNOSIS_TO_INDEX,
    INPUT_COLUMNS as _K_HOURS_INPUT_COLUMNS,
    MAX_MEDICATION_SLOTS as _K_HOURS_MAX_MEDICATION_SLOTS,
    MEDICATION_NAMES as _K_HOURS_MEDICATION_NAMES,
    MEDICATION_SCHEDULE_COLUMNS as _K_HOURS_MEDICATION_SCHEDULE_COLUMNS,
    MEDICATION_TO_INDEX as _K_HOURS_MEDICATION_TO_INDEX,
    MEDICATION_TYPE_COLUMNS as _K_HOURS_MEDICATION_TYPE_COLUMNS,
    SYMPTOM_NAMES as _K_HOURS_SYMPTOM_NAMES,
    SYMPTOM_TO_INDEX as _K_HOURS_SYMPTOM_TO_INDEX,
    make_one_hot as _make_one_hot,
    make_time_vector as _k_hours_make_time_vector,
    medication_slots_for_diagnosis as _k_hours_medication_slots_for_diagnosis,
    normalize_schedule as _k_hours_normalize_schedule,
    row_to_input_vector as _k_hours_row_to_input_vector,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolve_weights_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((_ROOT_DIR / path).resolve())


def _hours_since(value: datetime | None, now: datetime) -> float:
    ts = _as_utc(value)
    if ts is None:
        return 0.0
    return max((now - ts).total_seconds() / 3600.0, 0.0)


def _parse_blood_pressure(raw: str | None) -> tuple[float, float]:
    if not raw:
        return 120.0, 80.0
    match = re.search(r"(\d+)\s*/\s*(\d+)", raw)
    if not match:
        return 120.0, 80.0
    return float(match.group(1)), float(match.group(2))


def _pick_known_value(raw: str | None, allowed: list[str]) -> str:
    text = str(raw or "").strip()
    if text in allowed:
        return text
    lowered = text.lower()
    for item in allowed:
        if item and item.lower() in lowered:
            return item
    return ""


def _active_assignment(session, room_id: int) -> RoomAssignment | None:
    now = _utc_now()
    return (
        session.query(RoomAssignment)
        .filter(
            RoomAssignment.room_id == room_id,
            RoomAssignment.start_time <= now,
            ((RoomAssignment.end_time.is_(None)) | (RoomAssignment.end_time >= now)),
        )
        .order_by(RoomAssignment.start_time.desc())
        .first()
    )


def _active_admission(session, patient_id: int | None) -> Admission | None:
    if patient_id is None:
        return None
    now = _utc_now()
    return (
        session.query(Admission)
        .filter(
            Admission.patient_id == patient_id,
            Admission.admitted_at <= now,
            ((Admission.discharged_at.is_(None)) | (Admission.discharged_at >= now)),
        )
        .order_by(Admission.admitted_at.desc())
        .first()
    )


def _latest_visit(session, patient_id: int | None) -> Visit | None:
    if patient_id is None:
        return None
    return (
        session.query(Visit)
        .filter(Visit.patient_id == patient_id)
        .order_by(Visit.visit_time.desc())
        .first()
    )


def _latest_medications(session, patient_id: int | None) -> list[Medication]:
    if patient_id is None:
        return []
    return (
        session.query(Medication)
        .filter(Medication.patient_id == patient_id)
        .order_by(Medication.medication_time.desc())
        .all()
    )


def _latest_comfort(session, room_id: int) -> ComfortPreference | None:
    return (
        session.query(ComfortPreference)
        .filter(ComfortPreference.room_id == room_id)
        .order_by(ComfortPreference.timestamp.desc())
        .first()
    )


def _latest_sensor_value(session, room_id: int, sensor_type: str, location: str | None = None) -> float | None:
    query = (
        session.query(Data.value)
        .join(Data.sensor)
        .join(Sensor.device)
        .filter(Device.room_id == room_id, Sensor.sensor_type == sensor_type)
    )
    if location in {"main", "toilet"}:
        query = query.filter(Device.location == location)
    value = query.order_by(Data.timestamp.desc(), Data.data_id.desc()).first()
    return float(value[0]) if value and value[0] is not None else None


def _build_live_context(session, room_id: int, patient_id: int | None) -> dict:
    assignment = _active_assignment(session, room_id)
    effective_patient_id = patient_id or (assignment.patient_id if assignment else None)
    patient = session.get(Patient, effective_patient_id) if effective_patient_id else None
    admission = _active_admission(session, effective_patient_id)
    visit = _latest_visit(session, effective_patient_id)
    medications = _latest_medications(session, effective_patient_id)
    comfort = _latest_comfort(session, room_id)
    now = _utc_now()
    body_temp = float(visit.body_temperature) if visit and visit.body_temperature is not None else 37.0
    bp_systolic, bp_diastolic = _parse_blood_pressure(visit.blood_pressure if visit else None)
    diagnosis = _pick_known_value(admission.current_diagnosis if admission else "", _K_HOURS_DIAGNOSIS_NAMES)
    symptom = _pick_known_value(visit.symptoms if visit else "", _K_HOURS_SYMPTOM_NAMES)
    current_main = (
        float(comfort.temperature_main)
        if comfort and comfort.temperature_main is not None
        else (_latest_sensor_value(session, room_id, "temperature", "main") or 22.0)
    )
    current_toilet = (
        float(comfort.temperature_toilet)
        if comfort and comfort.temperature_toilet is not None
        else (_latest_sensor_value(session, room_id, "temperature", "toilet") or current_main)
    )
    current_light = (
        float(comfort.light_intensity)
        if comfort and comfort.light_intensity is not None
        else (_latest_sensor_value(session, room_id, "light", "main") or 50.0)
    )
    current_sound = (
        float(comfort.sound_level)
        if comfort and comfort.sound_level is not None
        else (_latest_sensor_value(session, room_id, "sound", "main") or 35.0)
    )
    return {
        "now": now,
        "patient_id": effective_patient_id,
        "patient": patient,
        "admission": admission,
        "visit": visit,
        "medications": medications,
        "comfort": comfort,
        "diagnosis": diagnosis,
        "symptom": symptom,
        "body_temperature": body_temp,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "current_main": current_main,
        "current_toilet": current_toilet,
        "current_light": current_light,
        "current_sound": current_sound,
        "current_airflow": bool(comfort.airflow) if comfort else False,
        "age": float(admission.age) if admission and admission.age is not None else 70.0,
        "height": float(patient.height) if patient and patient.height is not None else 170.0,
        "weight": float(admission.weight) if admission and admission.weight is not None else 75.0,
        "gender": str(patient.gender or "male") if patient else "male",
    }


def _load_k_hours_model(weights_path: str):
    from k_hours_based.fl_mlp_client import get_input_dim as _k_hours_get_input_dim
    from k_hours_based.fl_mlp_client import make_model as _k_hours_make_model
    from k_hours_based.fl_mlp_client import set_params as _k_hours_set_params

    model = _k_hours_make_model(_k_hours_get_input_dim())
    data = np.load(weights_path)
    param_keys = sorted(data.files, key=lambda key: int(key.split("_")[1]))
    params = [np.asarray(data[key], dtype=np.float64) for key in param_keys]
    _k_hours_set_params(model, params)
    return model


def _load_k_hours_lstm_model(weights_path: str):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for k_hours LSTM AI mode.")
    from k_hours_based.fl_lstm_client import get_input_dim as _k_hours_get_input_dim
    from k_hours_based.fl_lstm_client import make_model as _k_hours_make_model
    from k_hours_based.fl_lstm_client import set_params as _k_hours_set_params

    model = _k_hours_make_model(_k_hours_get_input_dim())
    state_dict = model.state_dict()
    raw = np.load(weights_path)
    params = [raw[key] for key in sorted(raw.files, key=lambda item: int(item.split("_")[1]))]
    if len(params) != len(state_dict):
        raise ValueError(f"Parameter count mismatch: expected {len(state_dict)}, got {len(params)}")
    new_state = OrderedDict()
    for (name, tensor), value in zip(state_dict.items(), params):
        new_state[name] = torch.tensor(value, dtype=tensor.dtype)
    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model


def _load_event_model(weights_path: str):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for event_based AI mode.")
    from event_based.fl_client import get_input_dim as _event_get_input_dim
    from event_based.fl_client import make_model as _event_make_model

    model = _event_make_model(_event_get_input_dim())
    state_dict = model.state_dict()
    raw = np.load(weights_path)
    params = [raw[key] for key in sorted(raw.files, key=lambda item: int(item.split("_")[1]))]
    if len(params) != len(state_dict):
        raise ValueError(f"Parameter count mismatch: expected {len(state_dict)}, got {len(params)}")
    new_state = OrderedDict()
    for (name, tensor), value in zip(state_dict.items(), params):
        new_state[name] = torch.tensor(value, dtype=tensor.dtype)
    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model


def _build_k_hours_frame(context: dict) -> pd.DataFrame:
    now = context["now"]
    row = {col: 0.0 for col in _K_HOURS_INPUT_COLUMNS}
    row["age"] = context["age"]
    row["height"] = context["height"]
    row["weight"] = context["weight"]
    row["gender_binary"] = 1.0 if str(context["gender"]).strip().lower() == "male" else 0.0
    row["body_temperature"] = context["body_temperature"]
    row["bp_systolic"] = context["bp_systolic"]
    row["bp_diastolic"] = context["bp_diastolic"]

    time_vector = _k_hours_make_time_vector(now.hour, now.minute)
    for idx, col in enumerate([c for c in _K_HOURS_INPUT_COLUMNS if c.startswith("time_slot_")]):
        row[col] = float(time_vector[idx])

    diagnosis_one_hot = _make_one_hot(_K_HOURS_DIAGNOSIS_TO_INDEX.get(context["diagnosis"]), len(_K_HOURS_DIAGNOSIS_NAMES))
    for idx in range(len(_K_HOURS_DIAGNOSIS_NAMES)):
        row[f"diagnosis_{idx}"] = float(diagnosis_one_hot[idx])

    symptom_one_hot = _make_one_hot(_K_HOURS_SYMPTOM_TO_INDEX.get(context["symptom"], _K_HOURS_SYMPTOM_TO_INDEX.get("")), len(_K_HOURS_SYMPTOM_NAMES))
    for idx in range(len(_K_HOURS_SYMPTOM_NAMES)):
        row[f"symptom_{idx}"] = float(symptom_one_hot[idx])

    med_name = ""
    if context["medications"]:
        med_name = _pick_known_value(context["medications"][0].drug_name, _K_HOURS_MEDICATION_NAMES)
    med_slots = list(_k_hours_medication_slots_for_diagnosis(context["diagnosis"]))
    if med_name in _K_HOURS_MEDICATION_TO_INDEX:
        med_slots = [(med_name, [now.hour])] + med_slots
    med_slots = med_slots[:_K_HOURS_MAX_MEDICATION_SLOTS]

    for slot in range(1, _K_HOURS_MAX_MEDICATION_SLOTS + 1):
        if slot <= len(med_slots):
            med_slot_name, med_hours = med_slots[slot - 1]
            type_vector = _make_one_hot(_K_HOURS_MEDICATION_TO_INDEX.get(med_slot_name), len(_K_HOURS_MEDICATION_NAMES))
            sched_vector = _k_hours_normalize_schedule(med_hours)
        else:
            type_vector = [0] * len(_K_HOURS_MEDICATION_NAMES)
            sched_vector = [0] * len(_K_HOURS_MEDICATION_SCHEDULE_COLUMNS[1])
        for col, value in zip(_K_HOURS_MEDICATION_TYPE_COLUMNS[slot], type_vector):
            row[col] = float(value)
        for col, value in zip(_K_HOURS_MEDICATION_SCHEDULE_COLUMNS[slot], sched_vector):
            row[col] = float(value)

    return pd.DataFrame([row])


def _predict_k_hours(weights_path: str, context: dict) -> dict:
    model = _load_k_hours_model(weights_path)
    frame = _build_k_hours_frame(context)
    x = _k_hours_row_to_input_vector(frame)
    y_pred = model.predict(x)[0]
    airflow_score = float(np.clip(y_pred[_K_HOURS_AIRFLOW_INDEX], 0.0, 1.0))
    return {
        "temperature_main": float(np.clip(y_pred[0], 16.0, 30.0)),
        "temperature_toilet": float(np.clip(y_pred[1], 16.0, 32.0)),
        "light_intensity": float(max(y_pred[2], 0.0)),
        "sound_level": float(max(y_pred[3], 0.0)),
        "airflow": airflow_score >= 0.5,
    }


def _predict_k_hours_lstm(weights_path: str, context: dict, sequence_length: int = 4) -> dict:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for k_hours LSTM AI mode.")
    model = _load_k_hours_lstm_model(weights_path)
    frame = _build_k_hours_frame(context)
    x_row = _k_hours_row_to_input_vector(frame).astype(np.float32)[0]
    x_seq = np.repeat(x_row[np.newaxis, :], sequence_length, axis=0)[np.newaxis, :, :]
    with torch.no_grad():
        y_pred = model(torch.tensor(x_seq, dtype=torch.float32)).cpu().numpy()[0]
    airflow_score = float(np.clip(y_pred[_K_HOURS_AIRFLOW_INDEX], 0.0, 1.0))
    return {
        "temperature_main": float(np.clip(y_pred[0], 16.0, 30.0)),
        "temperature_toilet": float(np.clip(y_pred[1], 16.0, 32.0)),
        "light_intensity": float(max(y_pred[2], 0.0)),
        "sound_level": float(max(y_pred[3], 0.0)),
        "airflow": airflow_score >= 0.5,
    }


def _build_event_frame(context: dict) -> pd.DataFrame:
    row = {col: 0.0 for col in _EVENT_FEATURE_COLUMNS}
    row["age"] = context["age"]
    row["height"] = context["height"]
    row["weight"] = context["weight"]
    row["gender_binary"] = 1.0 if str(context["gender"]).strip().lower() == "male" else 0.0
    row["body_temperature"] = context["body_temperature"]
    row["bp_systolic"] = context["bp_systolic"]
    row["bp_diastolic"] = context["bp_diastolic"]
    row["hours_since_last_medication"] = _hours_since(
        context["medications"][0].medication_time if context["medications"] else None,
        context["now"],
    )
    row["hours_since_last_symptom_change"] = _hours_since(context["visit"].visit_time if context["visit"] else None, context["now"])
    row["hours_since_last_comfort"] = _hours_since(context["comfort"].timestamp if context["comfort"] else None, context["now"])
    row["prev_target_temp_main"] = context["current_main"]
    row["prev_target_temp_toilet"] = context["current_toilet"]
    row["prev_target_light"] = context["current_light"]
    row["prev_target_sound"] = context["current_sound"]
    row["prev_target_airflow"] = 1.0 if context["current_airflow"] else 0.0
    row["event_is_medication"] = 1.0 if context["medications"] else 0.0
    row["event_is_visit"] = 1.0

    diagnosis_idx = _EVENT_DIAGNOSIS_TO_INDEX.get(_pick_known_value(context["diagnosis"], _EVENT_DIAGNOSIS_NAMES))
    for idx, value in enumerate(_make_one_hot(diagnosis_idx, len(_EVENT_DIAGNOSIS_NAMES))):
        row[f"diagnosis_{idx}"] = float(value)

    symptom_idx = _EVENT_SYMPTOM_TO_INDEX.get(_pick_known_value(context["symptom"], _EVENT_SYMPTOM_NAMES))
    for idx, value in enumerate(_make_one_hot(symptom_idx, len(_EVENT_SYMPTOM_NAMES))):
        row[f"symptom_{idx}"] = float(value)

    active_names = {_pick_known_value(med.drug_name, _EVENT_MEDICATION_NAMES) for med in context["medications"]}
    trigger_name = _pick_known_value(context["medications"][0].drug_name, _EVENT_MEDICATION_NAMES) if context["medications"] else ""
    for med_name in active_names:
        if med_name in _EVENT_MEDICATION_TO_INDEX:
            row[f"active_med_{_EVENT_MEDICATION_TO_INDEX[med_name]}"] = 1.0
    if trigger_name in _EVENT_MEDICATION_TO_INDEX:
        row[f"trigger_med_{_EVENT_MEDICATION_TO_INDEX[trigger_name]}"] = 1.0

    return pd.DataFrame([row])


def _predict_event_based(weights_path: str, context: dict) -> dict:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for event_based AI mode.")
    model = _load_event_model(weights_path)
    frame = _build_event_frame(context)
    x = _event_row_to_input_vector(frame).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()[0]
    airflow_prob = 1.0 / (1.0 + np.exp(-logits[4]))
    return {
        "temperature_main": float(np.clip(logits[0], 16.0, 30.0)),
        "temperature_toilet": float(np.clip(logits[1], 16.0, 32.0)),
        "light_intensity": float(max(logits[2], 0.0)),
        "sound_level": float(max(logits[3], 0.0)),
        "airflow": float(airflow_prob) >= 0.5,
    }


def predict_live_comfort(session, room_id: int, patient_id: int | None = None) -> tuple[dict, dict]:
    config = load_ai_config()
    source = config["selected_source"]
    model_type = config["k_hours_model_type"] if source == "k_hours" else config["event_based_model_type"]
    weights_key = "k_hours_weights_path" if source == "k_hours" else "event_based_weights_path"
    weights_path = _resolve_weights_path(config[weights_key])
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing AI weights file: {weights_path}")
    context = _build_live_context(session, room_id, patient_id)
    if context["patient_id"] is None:
        raise ValueError("AI mode requires an active patient assignment.")
    if source == "k_hours":
        if model_type == "mlp":
            prediction = _predict_k_hours(weights_path, context)
        elif model_type == "lstm":
            prediction = _predict_k_hours_lstm(weights_path, context)
        else:
            raise ValueError(f"Unsupported k_hours model type: {model_type}")
    else:
        prediction = _predict_event_based(weights_path, context)
    return prediction, {"source": source, "model_type": model_type, "weights_path": weights_path, "patient_id": context["patient_id"]}
