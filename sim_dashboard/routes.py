from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from statistics import mean, pstdev
import math
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, render_template, request

from sim_dashboard.csv_store import load_data
from simulation_batch.config import DAYS, START_DATE

sim_bp = Blueprint("sim_bp", __name__)
_ROOT_DIR = Path(__file__).resolve().parents[1]
_AI_JOBS: dict[str, dict] = {}
_AI_JOBS_LOCK = threading.Lock()
_MAX_LOG_LINES = 2000

try:
    _AI_DIR = str((_ROOT_DIR / "ai").resolve())
    if _AI_DIR not in sys.path:
        sys.path.insert(0, _AI_DIR)

    from fl_client import get_input_dim as _mlp_get_input_dim
    from fl_client import make_model as _mlp_make_model
    from fl_client import set_params as _mlp_set_params
    from next_hour_schema import (
        AIRFLOW_INDEX as _AIRFLOW_INDEX,
        DIAGNOSIS_NAMES as _DIAGNOSIS_NAMES,
        DIAGNOSIS_TO_INDEX as _DIAGNOSIS_TO_INDEX,
        INPUT_COLUMNS as _INPUT_COLUMNS,
        MAX_MEDICATION_SLOTS as _MAX_MEDICATION_SLOTS,
        MEDICATION_NAMES as _MEDICATION_NAMES,
        MEDICATION_TO_INDEX as _MEDICATION_TO_INDEX,
        MEDICATION_SCHEDULE_COLUMNS as _MEDICATION_SCHEDULE_COLUMNS,
        MEDICATION_TYPE_COLUMNS as _MEDICATION_TYPE_COLUMNS,
        SYMPTOM_NAMES as _SYMPTOM_NAMES,
        SYMPTOM_TO_INDEX as _SYMPTOM_TO_INDEX,
        TARGET_COLUMNS as _TARGET_COLUMNS,
        make_one_hot as _make_one_hot,
        make_time_vector as _make_time_vector,
        medication_slots_for_diagnosis as _medication_slots_for_diagnosis,
        normalize_schedule as _normalize_schedule,
        row_to_input_vector as _row_to_input_vector,
    )
    _PREDICT_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    _PREDICT_IMPORT_ERROR = str(exc)


def _new_job(kind: str, command: list[str]) -> str:
    job_id = uuid.uuid4().hex
    with _AI_JOBS_LOCK:
        _AI_JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "progress": 0,
            "logs": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "command": command,
            "return_code": None,
            "stop_requested": False,
            "pid": None,
            "current_process": None,
        }
    return job_id


def _append_log(job_id: str, line: str) -> None:
    clean = line.rstrip("\r\n")
    if not clean:
        return
    clean = re.sub(r"\x1b\[[0-9;]*m", "", clean)
    with _AI_JOBS_LOCK:
        job = _AI_JOBS.get(job_id)
        if not job:
            return
        job["logs"].append(clean)
        if len(job["logs"]) > _MAX_LOG_LINES:
            job["logs"] = job["logs"][-_MAX_LOG_LINES:]

        if job["kind"] == "split":
            if "Step 1/2" in clean:
                job["progress"] = max(job["progress"], 15)
            if "Step 2/2" in clean:
                job["progress"] = max(job["progress"], 65)
            if "complete" in clean.lower():
                job["progress"] = max(job["progress"], 95)
        elif job["kind"] == "federated":
            if "Step 1/2" in clean:
                job["progress"] = max(job["progress"], 10)
            if "Step 2/2" in clean:
                job["progress"] = max(job["progress"], 92)
            match = re.search(r"round[^0-9]*(\d+)\s*/\s*(\d+)", clean, flags=re.IGNORECASE)
            if match:
                current_round = int(match.group(1))
                total_rounds = max(1, int(match.group(2)))
                pct = 10 + int((current_round / total_rounds) * 80)
                job["progress"] = max(job["progress"], min(90, pct))


def _finish_job(job_id: str, ok: bool, return_code: int | None) -> None:
    with _AI_JOBS_LOCK:
        job = _AI_JOBS.get(job_id)
        if not job:
            return
        job["status"] = "completed" if ok else "failed"
        job["progress"] = 100 if ok else min(99, max(job["progress"], 1))
        job["return_code"] = return_code
        job["ended_at"] = datetime.now(timezone.utc).isoformat()
        job["pid"] = None
        job["current_process"] = None


def _mark_stopped(job_id: str, return_code: int | None) -> None:
    with _AI_JOBS_LOCK:
        job = _AI_JOBS.get(job_id)
        if not job:
            return
        job["status"] = "stopped"
        job["return_code"] = return_code
        job["ended_at"] = datetime.now(timezone.utc).isoformat()
        job["pid"] = None
        job["current_process"] = None


def _run_ai_commands(job_id: str, commands: list[list[str]]) -> None:
    for idx, command in enumerate(commands, start=1):
        with _AI_JOBS_LOCK:
            job = _AI_JOBS.get(job_id)
            if not job:
                return
            if job.get("stop_requested"):
                _append_log(job_id, "[dashboard] stop requested before next step")
                _mark_stopped(job_id, return_code=None)
                return

        _append_log(job_id, f"[dashboard] step {idx}/{len(commands)} command: {' '.join(command)}")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(_ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            _append_log(job_id, f"[dashboard] failed to start command: {exc}")
            _finish_job(job_id, ok=False, return_code=None)
            return

        with _AI_JOBS_LOCK:
            job = _AI_JOBS.get(job_id)
            if job:
                job["pid"] = process.pid
                job["current_process"] = process

        if process.stdout is not None:
            for line in process.stdout:
                _append_log(job_id, line)

        return_code = process.wait()
        with _AI_JOBS_LOCK:
            job = _AI_JOBS.get(job_id)
            stop_requested = bool(job and job.get("stop_requested"))
            if job:
                job["pid"] = None
                job["current_process"] = None

        if stop_requested:
            _append_log(job_id, "[dashboard] job stopped by user")
            _mark_stopped(job_id, return_code=return_code)
            return

        if return_code != 0:
            _append_log(job_id, f"[dashboard] job failed with exit code {return_code}")
            _finish_job(job_id, ok=False, return_code=return_code)
            return

    _append_log(job_id, "[dashboard] job completed successfully")
    _finish_job(job_id, ok=True, return_code=0)


def _start_ai_job(kind: str, commands: list[list[str]]) -> str:
    job_id = _new_job(kind, commands[0] if commands else [])
    thread = threading.Thread(target=_run_ai_commands, args=(job_id, commands), daemon=True)
    thread.start()
    return job_id


def _parse_kv_lines(lines: list[str]) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    for line in lines:
        for key, value in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)", line):
            token = value.rstrip(",")
            try:
                if re.fullmatch(r"-?\d+", token):
                    metrics[key] = int(token)
                else:
                    metrics[key] = float(token)
            except ValueError:
                metrics[key] = token
    return metrics


def _extract_job_metrics(job: dict) -> dict:
    raw = _parse_kv_lines(job.get("logs", []))
    if job.get("kind") == "federated":
        items = [
            {
                "key": "regression_correct_rate",
                "label": "Regression Correct Rate",
                "value": raw.get("regression_correct_rate"),
                "explain": "Fraction of evaluated samples where regression output met correctness criterion.",
            },
            {
                "key": "evaluated_examples",
                "label": "Evaluated Samples",
                "value": raw.get("evaluated_examples"),
                "explain": "Number of test examples included in federated evaluation.",
            },
            {
                "key": "mae_y_temp_main",
                "label": "MAE Temp Main",
                "value": raw.get("mae_y_temp_main"),
                "explain": "Average absolute error for main room temperature target. Lower is better.",
            },
            {
                "key": "mae_y_temp_toilet",
                "label": "MAE Temp Toilet",
                "value": raw.get("mae_y_temp_toilet"),
                "explain": "Average absolute error for toilet temperature target. Lower is better.",
            },
            {
                "key": "mae_y_light",
                "label": "MAE Light",
                "value": raw.get("mae_y_light"),
                "explain": "Average absolute error for light intensity prediction. Lower is better.",
            },
            {
                "key": "mae_y_sound",
                "label": "MAE Sound",
                "value": raw.get("mae_y_sound"),
                "explain": "Average absolute error for sound level prediction. Lower is better.",
            },
            {
                "key": "airflow_accuracy",
                "label": "Airflow Accuracy",
                "value": raw.get("airflow_accuracy"),
                "explain": "Overall correctness of airflow on/off classification.",
            },
            {
                "key": "airflow_f1",
                "label": "Airflow F1",
                "value": raw.get("airflow_f1"),
                "explain": "Balance between airflow precision and recall.",
            },
            {
                "key": "change_accuracy",
                "label": "Change Accuracy",
                "value": raw.get("change_accuracy"),
                "explain": "Correctness of predicting whether any comfort setting changes.",
            },
            {
                "key": "change_f1",
                "label": "Change F1",
                "value": raw.get("change_f1"),
                "explain": "Balanced score for change/no-change detection.",
            },
        ]
        return {"raw": raw, "items": items}

    if job.get("kind") == "split":
        items = [
            {
                "key": "next_hour_total",
                "label": "Next-hour Total Rows",
                "value": raw.get("next_hour_total"),
                "explain": "Total rows available before train/test split.",
            },
            {
                "key": "model_a_total",
                "label": "Model A Total Rows",
                "value": raw.get("model_a_total"),
                "explain": "Total rows for Model A before split.",
            },
            {
                "key": "model_b_total",
                "label": "Model B Total Rows",
                "value": raw.get("model_b_total"),
                "explain": "Total rows for Model B before split.",
            },
        ]
        return {"raw": raw, "items": items}

    return {"raw": raw, "items": []}


def _fmt_time(value: datetime | None) -> str:
    if not value:
        return "--:--"
    return value.strftime("%H:%M")


def _fmt_dt(value: datetime | None) -> str:
    if not value:
        return "--"
    return value.strftime("%Y-%m-%d %H:%M")


def _in_window(ts: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    if not ts or not start:
        return False
    if ts < start:
        return False
    if end and ts > end:
        return False
    return True


def _assignment_for_time(assignments: list[dict], ts: datetime | None) -> dict | None:
    if not ts:
        return None
    for assignment in assignments:
        a_start = assignment.get("start_time")
        a_end = assignment.get("end_time")
        if a_start and ts < a_start:
            continue
        if a_end and ts > a_end:
            continue
        return assignment
    return None


def _build_dist_svg(values: list[int | float], *, title: str, x_label: str) -> str | None:
    if not values:
        return None

    clean_values = [v for v in values if v is not None]
    if len(clean_values) < 2:
        return None

    width, height = 700, 240
    pad = 30
    bins = 20
    min_value = min(clean_values)
    max_value = max(clean_values)
    if min_value == max_value:
        min_value -= 1
        max_value += 1

    bin_width = (max_value - min_value) / bins
    counts = [0] * bins
    for value in clean_values:
        idx = int((value - min_value) / bin_width)
        if idx == bins:
            idx -= 1
        counts[idx] += 1

    max_count = max(counts) if counts else 1
    plot_width = width - 2 * pad
    plot_height = height - 2 * pad

    def x_for(val: float) -> float:
        return pad + (val - min_value) / (max_value - min_value) * plot_width

    def y_for_count(count: float) -> float:
        return pad + (1 - count / max_count) * plot_height

    mu = mean(clean_values)
    sigma = pstdev(clean_values) or 1.0
    curve_points = []
    steps = 80
    for i in range(steps + 1):
        x = min_value + (max_value - min_value) * i / steps
        pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
        curve_points.append((x_for(x), y_for_count(pdf * max_count * sigma)))

    bars = []
    for i, count in enumerate(counts):
        x0 = x_for(min_value + i * bin_width)
        x1 = x_for(min_value + (i + 1) * bin_width)
        y = y_for_count(count)
        bars.append(
            f'<rect x="{x0:.1f}" y="{y:.1f}" width="{(x1 - x0 - 1):.1f}" '
            f'height="{(pad + plot_height - y):.1f}" fill="#8fbcd4" />'
        )

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in curve_points)
    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
  <rect x="{pad}" y="{pad}" width="{plot_width}" height="{plot_height}" fill="#f8f8f8" stroke="#ddd" />
  {''.join(bars)}
  <polyline fill="none" stroke="#cc2f2f" stroke-width="2" points="{polyline}" />
  <text x="{pad}" y="18" font-size="12" fill="#333">{title}</text>
  <text x="{pad}" y="{height - 6}" font-size="12" fill="#333">{x_label}</text>
  <text x="{width - pad - 140}" y="{pad - 8}" font-size="12" fill="#333">Mean={mu:.1f}, SD={sigma:.1f}</text>
</svg>
""".strip()


@sim_bp.route("/")
def rooms():
    data = load_data()
    room_data = [
        {"room_id": room["room_id"], "room_number": room["room_number"]}
        for room in data["rooms"]
    ]

    simulation_period = f"{START_DATE} to {(START_DATE + timedelta(days=DAYS)).strftime('%Y-%m-%d')}"
    simulation_info = {
        "total_rooms": data["total_rooms"],
        "total_patients": data["total_patients"],
        "total_admissions": data["total_admissions"],
        "readmissions": data["readmissions"],
        "total_medications": data["total_medications"],
        "total_visits": data["total_visits"],
        "simulation_period": simulation_period,
    }

    height_svg_female = _build_dist_svg(
        data["female_heights"], title="Height Distribution - Female", x_label="Heights (cm)"
    )
    height_svg_male = _build_dist_svg(
        data["male_heights"], title="Height Distribution - Male", x_label="Heights (cm)"
    )
    age_svg_female = _build_dist_svg(
        data["female_ages"], title="Age Distribution - Female", x_label="Age (years)"
    )
    age_svg_male = _build_dist_svg(
        data["male_ages"], title="Age Distribution - Male", x_label="Age (years)"
    )

    return render_template(
        "rooms.html",
        rooms=room_data,
        simulation_info=simulation_info,
        height_svg_female=height_svg_female,
        height_svg_male=height_svg_male,
        age_svg_female=age_svg_female,
        age_svg_male=age_svg_male,
    )


@sim_bp.route("/rooms/<int:room_id>")
def room_detail(room_id: int):
    data = load_data()
    room = data["room_by_id"].get(room_id)
    if not room:
        return "Room not found", 404

    known_device_ids = data["devices_by_room"].get(room_id, [])
    device_data = [{"device_id": device_id, "device_type": "Device"} for device_id in known_device_ids]

    assignments = []
    for assignment in data["assignments_by_room"].get(room_id, []):
        patient = data["patients_by_id"].get(assignment["patient_id"])
        assignments.append(
            {
                "assignment_id": assignment["assignment_id"],
                "patient": {
                    "patient_id": assignment["patient_id"],
                    "name": patient["name"] if patient else f"Patient {assignment['patient_id']}",
                },
                "start_time": assignment["start_time"],
                "end_time": assignment["end_time"],
            }
        )

    comfort_preferences = []
    for pref in data["comfort_by_room"].get(room_id, []):
        patient = data["patients_by_id"].get(pref["patient_id"])
        comfort_preferences.append(
            {
                "comfort_pref_id": pref["comfort_pref_id"],
                "timestamp": pref["timestamp"],
                "temperature_main": pref["temperature_main"],
                "temperature_toilet": pref["temperature_toilet"],
                "light_intensity": pref["light_intensity"],
                "sound_level": pref["sound_level"],
                "airflow": pref["airflow"],
                "source": pref["source"],
                "patient_name": patient["name"] if patient else None,
                "sensor_windows": [],
            }
        )

    utility_usages = data["utility_by_room"].get(room_id, [])[:10]
    ventilation_data = data["ventilations_by_room"].get(room_id, [])
    room_data = {"room_id": room["room_id"], "room_number": room["room_number"], "devices": device_data}

    return render_template(
        "room_detail.html",
        room=room_data,
        assignments=assignments,
        comfort_preferences=comfort_preferences,
        utility_usages=utility_usages,
        ventilation_data=ventilation_data,
    )


@sim_bp.route("/patients")
def patients():
    data = load_data()
    patient_data = []
    for patient in data["patients"]:
        admissions = data["admissions_by_patient"].get(patient["patient_id"], [])
        latest_admission = max(admissions, key=lambda a: a["admitted_at"] or datetime.min.replace(tzinfo=timezone.utc)) if admissions else None
        patient_data.append(
            {
                "patient_id": patient["patient_id"],
                "name": patient["name"],
                "age": latest_admission["age"] if latest_admission else None,
                "admission_date": latest_admission["admitted_at"] if latest_admission else None,
                "release_date": latest_admission["discharged_at"] if latest_admission else None,
            }
        )

    return render_template("patients.html", patients=patient_data)


@sim_bp.route("/ai-suggestion")
def ai_suggestion():
    diagnosis_names = _DIAGNOSIS_NAMES if _PREDICT_IMPORT_ERROR is None else []
    symptom_names = _SYMPTOM_NAMES if _PREDICT_IMPORT_ERROR is None else []
    medication_names = _MEDICATION_NAMES if _PREDICT_IMPORT_ERROR is None else []
    return render_template(
        "ai_suggestion.html",
        diagnosis_names=diagnosis_names,
        symptom_names=symptom_names,
        medication_names=medication_names,
        predict_import_error=_PREDICT_IMPORT_ERROR,
    )


@sim_bp.route("/api/ai/split/start", methods=["POST"])
def api_ai_split_start():
    payload = request.get_json(silent=True) or {}
    split_method = str(payload.get("split_method", "split_by_room"))
    input_dir = str(payload.get("input_dir", "ai/outputs")).strip() or "ai/outputs"
    output_dir = str(payload.get("output_dir", "ai/splits")).strip() or "ai/splits"
    train_ratio = float(payload.get("train_ratio", 0.8))
    min_train_rows = int(payload.get("min_train_rows", 1))
    build_rows_first = bool(payload.get("build_rows_first", True))
    build_variant = str(payload.get("build_variant", "v1"))
    data_dir = str(payload.get("data_dir", "filestorage")).strip() or "filestorage"
    step_minutes = int(payload.get("step_minutes", 30))
    horizon_minutes = int(payload.get("horizon_minutes", 30))
    before_minutes = int(payload.get("before_minutes", 60))
    after_minutes = int(payload.get("after_minutes", 60))
    sample_minutes = int(payload.get("sample_minutes", 30))

    script_map = {
        "split_by_room": os.path.join("ai", "split_by_room.py"),
        "split_by_y": os.path.join("ai", "split_by_y.py"),
        "split_next_hour_by_room": os.path.join("ai", "split_next_hour_by_room.py"),
    }
    script_path = script_map.get(split_method)
    if not script_path:
        return jsonify({"error": f"Unsupported split method: {split_method}"}), 400

    split_command = [
        sys.executable,
        "-u",
        script_path,
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--train-ratio",
        str(train_ratio),
        "--min-train-rows",
        str(min_train_rows),
    ]

    commands: list[list[str]] = []
    if build_rows_first:
        if split_method == "split_next_hour_by_room":
            variant_map = {
                "v1": os.path.join("ai", "build_next_hour_rows.py"),
                "v2": os.path.join("ai", "build_next_hour_rows_2.py"),
                "v3": os.path.join("ai", "build_next_hour_rows_3.py"),
            }
            build_script = variant_map.get(build_variant)
            if not build_script:
                return jsonify({"error": f"Unsupported build variant: {build_variant}"}), 400
        else:
            build_script = os.path.join("ai", "build_row.py")

        build_command = [
            sys.executable,
            "-u",
            build_script,
            "--data-dir",
            data_dir,
            "--out-dir",
            input_dir,
            "--horizon-minutes",
            str(horizon_minutes),
        ]
        if split_method == "split_next_hour_by_room" and build_variant == "v1":
            build_command.extend(
                [
                    "--step-minutes",
                    str(step_minutes),
                ]
            )
        if split_method == "split_next_hour_by_room" and build_variant in {"v2", "v3"}:
            build_command.extend(
                [
                    "--before-minutes",
                    str(before_minutes),
                    "--after-minutes",
                    str(after_minutes),
                    "--sample-minutes",
                    str(sample_minutes),
                ]
            )
        elif split_method != "split_next_hour_by_room":
            build_command.extend(
                [
                    "--step-minutes",
                    str(step_minutes),
                ]
            )
        commands.append(build_command)

    commands.append(split_command)
    job_id = _start_ai_job("split", commands)
    return jsonify({"job_id": job_id, "status": "running"})


@sim_bp.route("/api/ai/federated/start", methods=["POST"])
def api_ai_federated_start():
    payload = request.get_json(silent=True) or {}
    model_type = str(payload.get("model_type", "mlp"))
    split_dir = str(payload.get("split_dir", "ai/splits_next_hour")).strip() or "ai/splits_next_hour"
    rounds = int(payload.get("rounds", 3))
    local_epochs = int(payload.get("local_epochs", 1))
    lstm_sequence_length = int(payload.get("lstm_sequence_length", 4))
    lstm_batch_size = int(payload.get("lstm_batch_size", 32))
    max_rooms = int(payload.get("max_rooms", 20))
    fraction_fit = float(payload.get("fraction_fit", 0.2))
    fraction_evaluate = float(payload.get("fraction_evaluate", 0.2))
    min_fit_clients = int(payload.get("min_fit_clients", 2))
    min_evaluate_clients = int(payload.get("min_evaluate_clients", 2))
    min_available_clients = int(payload.get("min_available_clients", 2))
    client_cpu = float(payload.get("client_cpu", 0.5))
    chunksize = int(payload.get("chunksize", 50000))

    mode_map = {
        "mlp": "ai_fl",
        "lstm": "ai_fl_lstm",
        "mlp_lstm": "ai_fl_lstm_MLP",
    }
    mode = mode_map.get(model_type)
    if not mode:
        return jsonify({"error": f"Unsupported model type: {model_type}"}), 400

    command = [
        sys.executable,
        "-u",
        "main.py",
        "--mode",
        mode,
        "--ai-next-hour-split-dir",
        split_dir,
        "--ai-rounds",
        str(rounds),
        "--ai-local-epochs",
        str(local_epochs),
        "--ai-lstm-sequence-length",
        str(lstm_sequence_length),
        "--ai-lstm-batch-size",
        str(lstm_batch_size),
        "--ai-max-rooms",
        str(max_rooms),
        "--ai-fraction-fit",
        str(fraction_fit),
        "--ai-fraction-evaluate",
        str(fraction_evaluate),
        "--ai-min-fit-clients",
        str(min_fit_clients),
        "--ai-min-evaluate-clients",
        str(min_evaluate_clients),
        "--ai-min-available-clients",
        str(min_available_clients),
        "--ai-client-cpu",
        str(client_cpu),
        "--ai-chunksize",
        str(chunksize),
    ]
    job_id = _start_ai_job("federated", [command])
    return jsonify({"job_id": job_id, "status": "running"})


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return default


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return default


def _build_single_input_row(payload: dict) -> pd.DataFrame:
    hour = max(0, min(23, _safe_int(payload.get("hour", 12), 12)))
    minute = max(0, min(59, _safe_int(payload.get("minute", 0), 0)))
    diagnosis = str(payload.get("diagnosis", "") or "").strip()
    symptom = str(payload.get("symptom", "") or "").strip()
    medication = str(payload.get("medication", "") or "").strip()

    row = {col: 0.0 for col in _INPUT_COLUMNS}
    row["age"] = _safe_float(payload.get("age", 70), 70.0)
    row["height"] = _safe_float(payload.get("height", 170), 170.0)
    row["weight"] = _safe_float(payload.get("weight", 75), 75.0)
    row["gender_binary"] = 1.0 if str(payload.get("gender", "male")).strip().lower() == "male" else 0.0

    time_vector = _make_time_vector(hour, minute)
    for idx, col in enumerate([c for c in _INPUT_COLUMNS if c.startswith("time_slot_")]):
        row[col] = float(time_vector[idx])

    diagnosis_one_hot = _make_one_hot(_DIAGNOSIS_TO_INDEX.get(diagnosis), len(_DIAGNOSIS_NAMES))
    for idx in range(len(_DIAGNOSIS_NAMES)):
        row[f"diagnosis_{idx}"] = float(diagnosis_one_hot[idx])

    symptom_one_hot = _make_one_hot(_SYMPTOM_TO_INDEX.get(symptom, _SYMPTOM_TO_INDEX.get("")), len(_SYMPTOM_NAMES))
    for idx in range(len(_SYMPTOM_NAMES)):
        row[f"symptom_{idx}"] = float(symptom_one_hot[idx])

    med_slots = list(_medication_slots_for_diagnosis(diagnosis))
    if medication in _MEDICATION_TO_INDEX:
        med_slots = [(medication, [hour])] + med_slots
    med_slots = med_slots[:_MAX_MEDICATION_SLOTS]

    for slot in range(1, _MAX_MEDICATION_SLOTS + 1):
        if slot <= len(med_slots):
            med_name, med_hours = med_slots[slot - 1]
            type_vector = _make_one_hot(_MEDICATION_TO_INDEX.get(med_name), len(_MEDICATION_NAMES))
            sched_vector = _normalize_schedule(med_hours)
        else:
            type_vector = [0] * len(_MEDICATION_NAMES)
            sched_vector = [0] * len([col for col in _MEDICATION_SCHEDULE_COLUMNS[1]])

        for col, value in zip(_MEDICATION_TYPE_COLUMNS[slot], type_vector):
            row[col] = float(value)
        for col, value in zip(_MEDICATION_SCHEDULE_COLUMNS[slot], sched_vector):
            row[col] = float(value)

    return pd.DataFrame([row])


@sim_bp.route("/api/ai/predict/mlp", methods=["POST"])
def api_ai_predict_mlp():
    if _PREDICT_IMPORT_ERROR is not None:
        return jsonify({"error": f"Prediction modules unavailable: {_PREDICT_IMPORT_ERROR}"}), 500

    payload = request.get_json(silent=True) or {}
    weights_path = str(payload.get("weights_path", "ai/fl_weights_sim/latest_global_weights.npz")).strip() or "ai/fl_weights_sim/latest_global_weights.npz"
    if not os.path.isabs(weights_path):
        weights_path = str((_ROOT_DIR / weights_path).resolve())
    if not os.path.exists(weights_path):
        return jsonify({"error": f"Saved weights not found: {weights_path}"}), 404

    df = _build_single_input_row(payload)
    x = _row_to_input_vector(df)
    model = _mlp_make_model(_mlp_get_input_dim())

    data = np.load(weights_path)
    param_keys = sorted(data.files, key=lambda key: int(key.split("_")[1]))
    params = [np.asarray(data[key], dtype=np.float64) for key in param_keys]
    _mlp_set_params(model, params)

    y_pred = model.predict(x)[0]
    airflow_score = float(np.clip(y_pred[_AIRFLOW_INDEX], 0.0, 1.0))
    airflow_flag = int(airflow_score >= 0.5)

    curr_temp_main = _safe_float(payload.get("curr_temp_main", 22.0), 22.0)
    curr_temp_toilet = _safe_float(payload.get("curr_temp_toilet", 23.0), 23.0)
    curr_light = _safe_float(payload.get("curr_light", 58.0), 58.0)
    curr_sound = _safe_float(payload.get("curr_sound", 34.0), 34.0)
    curr_airflow = 1 if bool(payload.get("curr_airflow", False)) else 0

    change_pred = int(
        (round(float(y_pred[0]), 2) != round(curr_temp_main, 2))
        or (round(float(y_pred[1]), 2) != round(curr_temp_toilet, 2))
        or (round(float(y_pred[2]), 0) != round(curr_light, 0))
        or (round(float(y_pred[3]), 0) != round(curr_sound, 0))
        or (airflow_flag != curr_airflow)
    )

    return jsonify(
        {
            "weights_path": weights_path,
            "model_type": "mlp_saved_weights",
            "prediction": {
                "y_temp_main": round(float(y_pred[0]), 3),
                "y_temp_toilet": round(float(y_pred[1]), 3),
                "y_light": round(float(y_pred[2]), 3),
                "y_sound": round(float(y_pred[3]), 3),
                "y_airflow_score": round(airflow_score, 4),
                "y_airflow": airflow_flag,
                "y_any_change": change_pred,
            },
        }
    )


@sim_bp.route("/api/ai/jobs/<job_id>", methods=["GET"])
def api_ai_job_status(job_id: str):
    cursor = request.args.get("cursor", default=0, type=int)
    if cursor < 0:
        cursor = 0

    with _AI_JOBS_LOCK:
        job = _AI_JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        logs = job["logs"][cursor:]
        next_cursor = cursor + len(logs)
        metrics = _extract_job_metrics(job)
        return jsonify(
            {
                "id": job["id"],
                "kind": job["kind"],
                "status": job["status"],
                "progress": job["progress"],
                "started_at": job["started_at"],
                "ended_at": job["ended_at"],
                "return_code": job["return_code"],
                "stop_requested": job["stop_requested"],
                "pid": job["pid"],
                "metrics": metrics,
                "logs": logs,
                "next_cursor": next_cursor,
            }
        )


@sim_bp.route("/api/ai/jobs/<job_id>/stop", methods=["POST"])
def api_ai_job_stop(job_id: str):
    with _AI_JOBS_LOCK:
        job = _AI_JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        if job["status"] in {"completed", "failed", "stopped"}:
            return jsonify({"id": job_id, "status": job["status"], "message": "Job already finished"})

        job["stop_requested"] = True
        process = job.get("current_process")

    if process is not None:
        try:
            process.terminate()
        except Exception:  # noqa: BLE001
            pass

    return jsonify({"id": job_id, "status": "stopping"})


@sim_bp.route("/patients/<int:patient_id>")
def patient_detail(patient_id: int):
    data = load_data()
    patient = data["patients_by_id"].get(patient_id)
    if not patient:
        return "Patient not found", 404

    admissions_sorted = sorted(
        data["admissions_by_patient"].get(patient_id, []),
        key=lambda a: (a["admitted_at"] is None, a["admitted_at"]),
        reverse=True,
    )
    latest_admission = admissions_sorted[0] if admissions_sorted else None

    patient_data = {
        "patient_id": patient["patient_id"],
        "name": patient["name"],
        "gender": patient["gender"],
        "height": patient["height"],
        "ethnicity": patient["ethnicity"],
        "age": latest_admission["age"] if latest_admission else None,
        "weight": latest_admission["weight"] if latest_admission else None,
        "current_diagnosis": latest_admission["current_diagnosis"] if latest_admission else None,
        "admission_date": latest_admission["admitted_at"] if latest_admission else None,
        "release_date": latest_admission["discharged_at"] if latest_admission else None,
    }

    room_lookup = {room["room_id"]: room["room_number"] for room in data["rooms"]}
    admissions = [
        {
            "admission_id": adm["admission_id"],
            "admitted_at": adm["admitted_at"],
            "discharged_at": adm["discharged_at"],
            "age": adm["age"],
            "weight": adm["weight"],
            "current_diagnosis": adm["current_diagnosis"],
        }
        for adm in admissions_sorted
    ]

    medications = data["medications_by_patient"].get(patient_id, [])
    visits = data["visits_by_patient"].get(patient_id, [])
    comforts = data["comfort_by_patient"].get(patient_id, [])
    room_assignments = data["assignments_by_patient"].get(patient_id, [])

    admission_views = []
    for admission in admissions_sorted:
        admission_assignments = sorted(
            [a for a in room_assignments if a.get("admission_id") == admission["admission_id"]],
            key=lambda assignment: (assignment["start_time"] is None, assignment["start_time"]),
        )
        admission_start = admission["admitted_at"]
        admission_end = admission["discharged_at"]

        utility_events = []
        for assignment in admission_assignments:
            for usage in data["utility_by_room"].get(assignment["room_id"], []):
                start_time = usage["start_time"]
                if not start_time:
                    continue
                assign_start = max(filter(None, [assignment["start_time"], admission_start]))
                assign_end_candidates = [x for x in [assignment["end_time"], admission_end] if x is not None]
                assign_end = min(assign_end_candidates) if assign_end_candidates else None
                if assign_start and start_time < assign_start:
                    continue
                if assign_end and start_time > assign_end:
                    continue
                utility_events.append(usage)
        utility_events.sort(key=lambda usage: (usage["start_time"] is None, usage["start_time"]))

        event_times = [t for t in [admission_start, admission_end] if t is not None]
        event_times += [m["medication_time"] for m in medications if _in_window(m["medication_time"], admission_start, admission_end)]
        event_times += [v["visit_time"] for v in visits if _in_window(v["visit_time"], admission_start, admission_end)]
        event_times += [c["timestamp"] for c in comforts if _in_window(c["timestamp"], admission_start, admission_end)]
        event_times += [a["start_time"] for a in admission_assignments if a["start_time"]]
        event_times += [a["end_time"] for a in admission_assignments if a["end_time"]]
        event_times += [u["start_time"] for u in utility_events if u["start_time"]]
        event_times = [ts for ts in event_times if ts is not None]
        if not event_times:
            continue

        span_start = min(event_times).date()
        span_end = max(event_times).date()
        day_map = {}
        cursor = span_start
        while cursor <= span_end:
            day_map[cursor] = {
                "date_key": cursor.isoformat(),
                "label": cursor.strftime("%A, %d %b %Y"),
                "events": [],
                "comfort_events": [],
                "environment_events": [],
                "clinical_events": [],
                "room_events": [],
            }
            cursor += timedelta(days=1)

        def add_event(ts, category, title, details, target_day=None, badge_class=None, badge_label=None):
            day_key = target_day if target_day else (ts.date() if ts else span_start)
            if day_key not in day_map:
                return
            event = {
                "time": _fmt_time(ts),
                "timestamp_str": _fmt_dt(ts),
                "timestamp": ts,
                "category": category,
                "badge_class": badge_class if badge_class else category,
                "badge_label": badge_label if badge_label else category,
                "title": title,
                "details": details,
            }
            day_map[day_key]["events"].append(event)
            bucket = f"{category}_events"
            if bucket in day_map[day_key]:
                day_map[day_key][bucket].append(event)

        add_event(admission_start, "room", "Admission started", _fmt_dt(admission_start))
        if admission_end:
            add_event(admission_end, "room", "Discharged", _fmt_dt(admission_end))

        for assignment in admission_assignments:
            room_number = room_lookup.get(assignment["room_id"], f"Room {assignment['room_id']}")
            add_event(
                assignment["start_time"],
                "room",
                f"Moved to {room_number}",
                f"Assignment #{assignment['assignment_id']} (start)",
            )
            if assignment["end_time"]:
                add_event(
                    assignment["end_time"],
                    "room",
                    f"Left {room_number}",
                    f"Assignment #{assignment['assignment_id']} (end)",
                )

        for comfort in comforts:
            if not _in_window(comfort["timestamp"], admission_start, admission_end):
                continue
            active_assignment = _assignment_for_time(admission_assignments, comfort["timestamp"])
            room_hint = ""
            if active_assignment:
                room_id = active_assignment["room_id"]
                room_name = room_lookup.get(room_id, f"Room {room_id}")
                room_hint = f" in {room_name}"
            comfort_text = (
                f"Main {comfort['temperature_main']} C, "
                f"Toilet {comfort['temperature_toilet'] if comfort['temperature_toilet'] is not None else '--'} C, "
                f"Light {comfort['light_intensity'] if comfort['light_intensity'] is not None else '--'}, "
                f"Sound {comfort['sound_level'] if comfort['sound_level'] is not None else '--'}, "
                f"Airflow {'On' if comfort['airflow'] else 'Off'}{room_hint}"
            )
            add_event(comfort["timestamp"], "comfort", "Comfort setting updated", comfort_text)

        for medication in medications:
            if not _in_window(medication["medication_time"], admission_start, admission_end):
                continue
            med_text = (
                f"{medication['drug_name']} | Dose {medication['dose'] if medication['dose'] else '--'} | "
                f"Route {medication['route'] if medication['route'] else '--'} | "
                f"Status {medication['status'] if medication['status'] else '--'}"
            )
            add_event(
                medication["medication_time"],
                "clinical",
                "Medication",
                med_text,
                badge_class="clinical-medication",
                badge_label="medication",
            )

        for visit in visits:
            if not _in_window(visit["visit_time"], admission_start, admission_end):
                continue
            visit_text = (
                f"Temp {visit['body_temperature'] if visit['body_temperature'] is not None else '--'} C | "
                f"BP {visit['blood_pressure'] if visit['blood_pressure'] else '--'} | "
                f"Symptoms {visit['symptoms'] if visit['symptoms'] else '--'}"
            )
            add_event(
                visit["visit_time"],
                "clinical",
                "Visit",
                visit_text,
                badge_class="clinical-visit",
                badge_label="visit",
            )

        for usage in utility_events:
            room_number = room_lookup.get(usage["room_id"], f"Room {usage['room_id']}")
            env_text = (
                f"{room_number} | Power {usage['power_consumption'] if usage['power_consumption'] is not None else '--'} kWh | "
                f"Water {usage['water_consumption'] if usage['water_consumption'] is not None else '--'} L"
            )
            add_event(
                usage["start_time"],
                "environment",
                f"Environment usage: {usage['category']}",
                env_text,
            )

        days = []
        for day in sorted(day_map.keys()):
            current_day = day_map[day]
            current_day["events"] = sorted(
                current_day["events"],
                key=lambda event: (event["timestamp"] is None, event["timestamp"]),
            )
            # CSV exports do not contain a direct sensor_id -> room_id mapping.
            current_day["sensor_rows"] = []
            current_day["sensor_count"] = 0
            current_day["sensor_types"] = []
            current_day["sensor_rooms"] = []
            days.append(current_day)

        admission_views.append(
            {
                "admission_id": admission["admission_id"],
                "admitted_at": admission["admitted_at"],
                "discharged_at": admission["discharged_at"],
                "age": admission["age"],
                "weight": admission["weight"],
                "current_diagnosis": admission["current_diagnosis"],
                "days": days,
                "assignment_count": len(admission_assignments),
            }
        )

    active_admission_id = admission_views[0]["admission_id"] if admission_views else None

    return render_template(
        "patient_detail.html",
        patient=patient_data,
        comforts=comforts,
        admissions=admissions,
        medications=medications,
        visits=visits,
        room_assignments=room_assignments,
        admission_views=admission_views,
        active_admission_id=active_admission_id,
    )
