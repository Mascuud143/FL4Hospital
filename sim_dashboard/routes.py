from __future__ import annotations

import csv
from datetime import datetime, time, timedelta, timezone
from statistics import mean, pstdev
import math
import json
import os
import platform
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, redirect, render_template, request

from sim_dashboard.csv_store import load_data
from simulation_batch.config import DAYS, START_DATE
import simulation_batch.config as sim_config
from k_hours_based.runtime_defaults import (
    BUILD_CSV_WRITE_BATCH_SIZE as _DEFAULT_BUILD_CSV_WRITE_BATCH_SIZE,
    BUILD_DEFAULT_CHUNK_SIZE as _DEFAULT_BUILD_CHUNK_SIZE,
    FEDERATED_DEFAULT_CHUNKSIZE as _DEFAULT_FEDERATED_CHUNKSIZE,
    SPLIT_CSV_WRITE_BATCH_SIZE as _DEFAULT_SPLIT_CSV_WRITE_BATCH_SIZE,
    SPLIT_DEFAULT_CHUNK_SIZE as _DEFAULT_SPLIT_CHUNK_SIZE,
    default_build_workers,
    default_federated_client_cpu,
    default_federated_workers,
)

sim_bp = Blueprint("sim_bp", __name__)
_ROOT_DIR = Path(__file__).resolve().parents[1]
_K_HOURS_DIR = _ROOT_DIR / "k_hours_based"
_EVENT_DIR = _ROOT_DIR / "event_based"
_AI_JOBS: dict[str, dict] = {}
_AI_JOBS_LOCK = threading.Lock()
_MAX_LOG_LINES = 2000
_RUN_METADATA_DIR = _ROOT_DIR / "sim_dashboard" / "run_metadata"
_LAST_RUNTIME_SUMMARY_PATH = _RUN_METADATA_DIR / "last_run_time_summary.json"
_PC_SPEC_PATH = _RUN_METADATA_DIR / "pc_specification.json"
_RUNTIME_HISTORY_LIMIT = 50
_DEFAULT_BUILD_WORKERS = default_build_workers()
_DEFAULT_SPLIT_WORKERS = default_build_workers()
_TASK_DIRS = {
    "k_hours": _K_HOURS_DIR,
    "event": _EVENT_DIR,
}
_BUILD_OUTPUT_DIRS = {
    "k_hours": _K_HOURS_DIR / "outputs_next_hour_dashboard",
    "event": _EVENT_DIR / "rows_dashboard",
}
_SPLIT_OUTPUT_DIRS = {
    "k_hours": _K_HOURS_DIR / "splits_next_hour_dashboard",
    "event": _EVENT_DIR / "splits_dashboard",
}
_MODEL_OUTPUT_DIRS = {
    "k_hours": {
        "mlp": _K_HOURS_DIR / "fl_weights_dashboard_mlp",
        "lstm": _K_HOURS_DIR / "fl_weights_dashboard_lstm",
    },
    "event": {
        "mlp": _EVENT_DIR / "fl_weights_dashboard_mlp",
    },
}
_ADMISSIONS_CSV_PATH = _ROOT_DIR / "filestorage" / "admissions.csv"


def _simulation_period_from_admissions() -> str:
    if not _ADMISSIONS_CSV_PATH.exists():
        return f"{START_DATE} to {(START_DATE + timedelta(days=DAYS)).strftime('%Y-%m-%d')}"

    min_admitted: datetime | None = None
    max_discharged: datetime | None = None
    with _ADMISSIONS_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            admitted_raw = str(row.get("admitted_at") or "").strip()
            discharged_raw = str(row.get("discharged_at") or "").strip()
            try:
                admitted_at = datetime.fromisoformat(admitted_raw)
                discharged_at = datetime.fromisoformat(discharged_raw)
            except ValueError:
                continue
            if min_admitted is None or admitted_at < min_admitted:
                min_admitted = admitted_at
            if max_discharged is None or discharged_at > max_discharged:
                max_discharged = discharged_at

    if min_admitted is None or max_discharged is None:
        return f"{START_DATE} to {(START_DATE + timedelta(days=DAYS)).strftime('%Y-%m-%d')}"
    return f"{min_admitted.strftime('%Y-%m-%d')} to {max_discharged.strftime('%Y-%m-%d')}"


def _default_federated_client_cpu() -> float:
    return default_federated_client_cpu()


def _default_federated_workers() -> int:
    return default_federated_workers()

try:
    if str(_K_HOURS_DIR) not in sys.path:
        sys.path.insert(0, str(_K_HOURS_DIR))

    from k_hours_based.fl_mlp_client import get_input_dim as _mlp_get_input_dim
    from k_hours_based.fl_mlp_client import make_model as _mlp_make_model
    from k_hours_based.fl_mlp_client import set_params as _mlp_set_params
    from k_hours_based.next_hour_schema import (
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
            "metadata": {},
            "peak_memory_bytes": 0,
        }
    return job_id


def _format_memory_bytes(num_bytes: int | float | None) -> str:
    if not num_bytes:
        return "Pending"
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.2f} {units[unit_index]}"


def _process_tree_memory_bytes(root_pid: int) -> int:
    if root_pid <= 0 or platform.system().lower() != "windows":
        return 0
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,WorkingSetSize | ConvertTo-Json -Compress"
    )
    raw = _run_powershell_query(script)
    if not raw:
        return 0
    try:
        rows = json.loads(raw)
    except Exception:
        return 0
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return 0

    by_parent: dict[int, list[int]] = {}
    working_set: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            pid = int(row.get("ProcessId"))
            parent_pid = int(row.get("ParentProcessId"))
            rss = int(row.get("WorkingSetSize") or 0)
        except Exception:
            continue
        by_parent.setdefault(parent_pid, []).append(pid)
        working_set[pid] = rss

    if root_pid not in working_set:
        return 0

    total = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += working_set.get(pid, 0)
        stack.extend(by_parent.get(pid, []))
    return total


def _start_memory_monitor(job_id: str, root_pid: int) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _monitor() -> None:
        peak = 0
        while not stop_event.wait(1.0):
            current = _process_tree_memory_bytes(root_pid)
            if current <= 0:
                continue
            peak = max(peak, current)
            with _AI_JOBS_LOCK:
                job = _AI_JOBS.get(job_id)
                if job is None:
                    return
                job["peak_memory_bytes"] = max(int(job.get("peak_memory_bytes") or 0), peak)

        current = _process_tree_memory_bytes(root_pid)
        if current > 0:
            peak = max(peak, current)
        if peak > 0:
            with _AI_JOBS_LOCK:
                job = _AI_JOBS.get(job_id)
                if job is not None:
                    job["peak_memory_bytes"] = max(int(job.get("peak_memory_bytes") or 0), peak)

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return stop_event, thread


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

        if job["kind"] == "build":
            if "chunk" in clean.lower():
                job["progress"] = max(job["progress"], 35)
            if "complete" in clean.lower():
                job["progress"] = max(job["progress"], 95)
        elif job["kind"] == "split":
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
        elif job["kind"] == "simulation":
            if "PRE-GENERATION START" in clean:
                job["progress"] = max(job["progress"], 10)
            if "PRE-GENERATION COMPLETE" in clean:
                job["progress"] = max(job["progress"], 35)
            if "START SIM LOOP" in clean:
                job["progress"] = max(job["progress"], 45)
            if "SIMULATION COMPLETE" in clean or "SIMULATION finished" in clean:
                job["progress"] = max(job["progress"], 95)


def _reset_job_peak_memory(job_id: str) -> None:
    with _AI_JOBS_LOCK:
        job = _AI_JOBS.get(job_id)
        if job is not None:
            job["peak_memory_bytes"] = 0


def _log_peak_memory(job_id: str, prefix: str) -> None:
    with _AI_JOBS_LOCK:
        job = _AI_JOBS.get(job_id)
        peak_memory_bytes = int(job.get("peak_memory_bytes") or 0) if job else 0
    _append_log(job_id, f"{prefix} peak_memory={_format_memory_bytes(peak_memory_bytes)}")


def _finish_job(job_id: str, ok: bool, return_code: int | None) -> None:
    snapshot = None
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
        snapshot = dict(job)
    if ok and snapshot is not None:
        _persist_job_runtime(snapshot)


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

        _reset_job_peak_memory(job_id)
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

        monitor_stop, monitor_thread = _start_memory_monitor(job_id, process.pid)

        if process.stdout is not None:
            for line in process.stdout:
                _append_log(job_id, line)

        return_code = process.wait()
        monitor_stop.set()
        monitor_thread.join(timeout=2.0)
        with _AI_JOBS_LOCK:
            job = _AI_JOBS.get(job_id)
            stop_requested = bool(job and job.get("stop_requested"))
            if job:
                job["pid"] = None
                job["current_process"] = None

        if stop_requested:
            if return_code == 0:
                _log_peak_memory(job_id, "[dashboard] job stopped")
            _append_log(job_id, "[dashboard] job stopped by user")
            _mark_stopped(job_id, return_code=return_code)
            return

        if return_code != 0:
            _log_peak_memory(job_id, f"[dashboard] job failed with exit code {return_code}")
            _append_log(job_id, f"[dashboard] job failed with exit code {return_code}")
            _finish_job(job_id, ok=False, return_code=return_code)
            return

    _log_peak_memory(job_id, "[dashboard] job completed successfully")
    _append_log(job_id, "[dashboard] job completed successfully")
    _finish_job(job_id, ok=True, return_code=0)


def _start_ai_job(kind: str, commands: list[list[str]], *, metadata: dict[str, object] | None = None) -> str:
    job_id = _new_job(kind, commands[0] if commands else [])
    if metadata:
        with _AI_JOBS_LOCK:
            job = _AI_JOBS.get(job_id)
            if job is not None:
                job["metadata"] = metadata
    thread = threading.Thread(target=_run_ai_commands, args=(job_id, commands), daemon=True)
    thread.start()
    return job_id


def _active_job_id(kind: str) -> str | None:
    with _AI_JOBS_LOCK:
        for job_id, job in _AI_JOBS.items():
            if job.get("kind") != kind:
                continue
            if job.get("status") == "running":
                return job_id
    return None


def _update_job_metadata(job_id: str, **values: object) -> None:
    with _AI_JOBS_LOCK:
        job = _AI_JOBS.get(job_id)
        if not job:
            return
        metadata = job.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            job["metadata"] = metadata
        metadata.update(values)


def _simulation_command_and_config(payload: dict[str, object]) -> tuple[list[str], dict[str, object]]:
    months_raw = payload.get("months")
    months = None
    if months_raw not in (None, ""):
        try:
            months = max(1, int(months_raw))  # type: ignore[arg-type]
        except Exception:
            months = None
    default_days = int(getattr(sim_config, "DAYS", 1))
    resolved_days = max(1, round(months * 365 / 12)) if months is not None else int(payload.get("days", default_days))
    config_payload = {
        "start_date": str(payload.get("start_date") or _simulation_config_defaults().get("start_date")),
        "days": resolved_days,
        "months": months if months is not None else max(1, round(default_days * 12 / 365)),
        "patient_count": int(payload.get("patient_count", getattr(sim_config, "PATIENT_COUNT", 1))),
        "random_seed": int(payload.get("random_seed", getattr(sim_config, "RANDOM_SEED", 42))),
        "change_room_prob": float(payload.get("change_room_prob", getattr(sim_config, "CHANGE_ROOM_PROB", 0.3))),
        "min_days_before_transfer": int(payload.get("min_days_before_transfer", getattr(sim_config, "MIN_DAYS_BEFORE_TRANSFER", 1))),
        "min_days_after_transfer": int(payload.get("min_days_after_transfer", getattr(sim_config, "MIN_DAYS_AFTER_TRANSFER", 1))),
        "comfort_max_changes_per_day": int(payload.get("comfort_max_changes_per_day", getattr(sim_config, "COMFORT_MAX_CHANGES_PER_DAY", 6))),
        "sim_step_s": int(payload.get("sim_step_s", getattr(sim_config, "SIM_STEP_S", 60))),
        "sensor_sample_every_s": int(payload.get("sensor_sample_every_s", getattr(sim_config, "SENSOR_SAMPLE_EVERY_S", 300))),
        "wall_sleep_s": float(payload.get("wall_sleep_s", getattr(sim_config, "WALL_SLEEP_S", 0.0))),
        "enable_comfort": bool(payload.get("enable_comfort", getattr(sim_config, "ENABLE_COMFORT", True))),
        "enable_medication": bool(payload.get("enable_medication", getattr(sim_config, "ENABLE_MEDICATION", True))),
        "enable_visits": bool(payload.get("enable_visits", getattr(sim_config, "ENABLE_VISITS", True))),
        "enable_toilet_usage": bool(payload.get("enable_toilet_usage", getattr(sim_config, "ENABLE_TOILET_USAGE", False))),
        "enable_sensor_emit": bool(payload.get("enable_sensor_emit", getattr(sim_config, "ENABLE_SENSOR_EMIT", False))),
        "enable_utility_usage": bool(payload.get("enable_utility_usage", getattr(sim_config, "ENABLE_UTILITY_USAGE", False))),
    }
    command = [
        sys.executable,
        "-u",
        str((_ROOT_DIR / "main.py").resolve()),
        "--mode",
        "simulation",
        "--start-date",
        config_payload["start_date"],
        "--days",
        str(config_payload["days"]),
        "--patient-count",
        str(config_payload["patient_count"]),
        "--random-seed",
        str(config_payload["random_seed"]),
        "--change-room-prob",
        str(config_payload["change_room_prob"]),
        "--min-days-before-transfer",
        str(config_payload["min_days_before_transfer"]),
        "--min-days-after-transfer",
        str(config_payload["min_days_after_transfer"]),
        "--comfort-max-changes-per-day",
        str(config_payload["comfort_max_changes_per_day"]),
        "--sim-step-s",
        str(config_payload["sim_step_s"]),
        "--sensor-sample-every-s",
        str(config_payload["sensor_sample_every_s"]),
        "--wall-sleep-s",
        str(config_payload["wall_sleep_s"]),
    ]
    for flag_key, flag_name in (
        ("enable_comfort", "comfort"),
        ("enable_medication", "medication"),
        ("enable_visits", "visits"),
        ("enable_toilet_usage", "toilet-usage"),
        ("enable_sensor_emit", "sensor-emit"),
        ("enable_utility_usage", "utility-usage"),
    ):
        command.append(f"--{'enable' if config_payload[flag_key] else 'no-enable'}-{flag_name}")
    return command, config_payload


def _build_command_and_config(payload: dict[str, object], task_type: str) -> tuple[list[str], dict[str, object]]:
    if task_type == "event":
        medication_effect_minutes = int(payload.get("medication_effect_minutes", 5))
        medication_active_hours = int(payload.get("medication_active_hours", 24))
        command = [
            sys.executable, "-u", _ai_script(task_type, "build_event_rows.py"),
            "--data-dir", "filestorage",
            "--out-dir", str(_BUILD_OUTPUT_DIRS[task_type]),
            "--medication-effect-minutes", str(medication_effect_minutes),
            "--medication-active-hours", str(medication_active_hours),
        ]
        config = {
            "mode": task_type,
            "data_dir": "filestorage",
            "out_dir": str(_BUILD_OUTPUT_DIRS[task_type]),
            "medication_effect_minutes": medication_effect_minutes,
            "medication_active_hours": medication_active_hours,
        }
        return command, config
    step_minutes = int(payload.get("step_minutes", 30))
    horizon_minutes = int(payload.get("horizon_minutes", 60))
    workers = int(payload.get("workers", _DEFAULT_BUILD_WORKERS))
    chunk_size = int(payload.get("chunk_size", _DEFAULT_BUILD_CHUNK_SIZE))
    csv_write_batch_size = int(payload.get("csv_write_batch_size", _DEFAULT_BUILD_CSV_WRITE_BATCH_SIZE))
    command = [
        sys.executable, "-u", _ai_script(task_type, "build_next_hour_rows.py"),
        "--data-dir", "filestorage",
        "--out-dir", str(_BUILD_OUTPUT_DIRS[task_type]),
        "--step-minutes", str(step_minutes),
        "--horizon-minutes", str(horizon_minutes),
        "--workers", str(workers),
        "--chunk-size", str(chunk_size),
        "--csv-write-batch-size", str(csv_write_batch_size),
    ]
    config = {
        "mode": task_type,
        "data_dir": "filestorage",
        "out_dir": str(_BUILD_OUTPUT_DIRS[task_type]),
        "step_minutes": step_minutes,
        "horizon_minutes": horizon_minutes,
        "workers": workers,
        "chunk_size": chunk_size,
        "csv_write_batch_size": csv_write_batch_size,
    }
    return command, config


def _split_command_and_config(payload: dict[str, object], task_type: str) -> tuple[list[str], dict[str, object]]:
    chunk_size = int(payload.get("chunk_size", _DEFAULT_SPLIT_CHUNK_SIZE))
    workers = int(payload.get("workers", _DEFAULT_SPLIT_WORKERS))
    csv_write_batch_size = int(payload.get("csv_write_batch_size", _DEFAULT_SPLIT_CSV_WRITE_BATCH_SIZE))
    train_ratio = float(payload.get("train_ratio", 0.8))
    min_train_rows = int(payload.get("min_train_rows", 1))
    command = [
        sys.executable, "-u", _ai_script(task_type, "split_by_patient_stay.py" if task_type == "event" else "split_next_hour_by_room.py"),
        "--input-dir", str(_BUILD_OUTPUT_DIRS[task_type]),
        "--output-dir", str(_SPLIT_OUTPUT_DIRS[task_type]),
        "--train-ratio", str(train_ratio),
        "--min-train-rows", str(min_train_rows),
        "--chunk-size", str(chunk_size),
    ]
    if task_type != "event":
        command.extend([
            "--workers", str(workers),
            "--csv-write-batch-size", str(csv_write_batch_size),
        ])
    config = {
        "mode": task_type,
        "input_dir": str(_BUILD_OUTPUT_DIRS[task_type]),
        "output_dir": str(_SPLIT_OUTPUT_DIRS[task_type]),
        "chunk_size": chunk_size,
        "workers": workers,
        "csv_write_batch_size": csv_write_batch_size,
        "train_ratio": train_ratio,
        "min_train_rows": min_train_rows,
    }
    return command, config


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


def _default_runtime_summary() -> dict[str, object]:
    return {
        "total_runtime": "Pending",
        "build_runtime": "Pending",
        "split_runtime": "Pending",
        "training_runtime": "Pending",
        "simulation_total_runtime": "Pending",
        "simulation_runtime": "Pending",
        "simulation_seed_runtime": "Pending",
        "simulation_write_runtime": "Pending",
        "stage_seconds": {
            "build": None,
            "split": None,
            "training": None,
            "simulation_total": None,
            "simulation": None,
            "simulation_seed": None,
            "simulation_write": None,
        },
        "last_configs": {
            "build": {},
            "split": {},
            "training": {},
            "simulation": {},
        },
        "history": [],
    }


def _default_pc_spec() -> dict[str, object]:
    return {
        "cpu": "Pending",
        "ram": "Pending",
        "gpu": "Pending",
        "os": "Pending",
    }


def _safe_iso_to_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "Pending"
    total_seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _detect_total_ram_gb() -> str:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return f"{stat.ullTotalPhys / (1024 ** 3):.1f} GB"
    except Exception:
        pass
    return "Pending"


def _run_powershell_query(script: str) -> str | None:
    if platform.system().lower() != "windows":
        return None
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=True,
            timeout=8,
        )
        output = (completed.stdout or "").strip()
        return output or None
    except Exception:
        return None


def _detect_cpu_label() -> str:
    base = platform.processor() or "Unknown"
    ps_value = _run_powershell_query("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
    name = ps_value or base
    return f"{name} ({os.cpu_count() or 0} logical cores)"


def _detect_gpu_label() -> str:
    ps_value = _run_powershell_query(
        "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -and $_.Name -notmatch 'Basic Display|Remote Display|Hyper-V' } | "
        "Select-Object -ExpandProperty Name)"
    )
    if not ps_value:
        return "Pending"
    parts = [line.strip() for line in ps_value.splitlines() if line.strip()]
    unique_parts: list[str] = []
    for part in parts:
        if part not in unique_parts:
            unique_parts.append(part)
    return ", ".join(unique_parts) if unique_parts else "Pending"


def _detect_os_label() -> str:
    system = platform.system()
    if system.lower() != "windows":
        return f"{system} {platform.release()}"
    version = platform.version()
    try:
        build = int(version.split(".")[-1])
    except Exception:
        build = 0
    label = "Windows 11" if build >= 22000 else "Windows 10"
    return label


def _refresh_pc_spec() -> None:
    _RUN_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cpu": _detect_cpu_label(),
        "ram": _detect_total_ram_gb(),
        "gpu": _detect_gpu_label(),
        "os": _detect_os_label(),
    }
    _PC_SPEC_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_runtime_summary_file() -> dict[str, object]:
    _ensure_run_metadata_files()
    with open(_LAST_RUNTIME_SUMMARY_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    merged = _default_runtime_summary()
    if isinstance(payload, dict):
        merged.update({k: v for k, v in payload.items() if k in merged and k not in {"stage_seconds", "last_configs", "history"}})
        if isinstance(payload.get("stage_seconds"), dict):
            merged["stage_seconds"].update(payload["stage_seconds"])
        if isinstance(payload.get("last_configs"), dict):
            for key, value in payload["last_configs"].items():
                if key in merged["last_configs"] and isinstance(value, dict):
                    merged["last_configs"][key] = value
        if isinstance(payload.get("history"), list):
            merged["history"] = payload["history"]
    return merged


def _save_runtime_summary_file(payload: dict[str, object]) -> None:
    _RUN_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    _LAST_RUNTIME_SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _persist_job_runtime(job: dict[str, object]) -> None:
    started_at = _safe_iso_to_datetime(job.get("started_at"))
    ended_at = _safe_iso_to_datetime(job.get("ended_at"))
    if started_at is None or ended_at is None:
        return
    duration_seconds = max(0.0, (ended_at - started_at).total_seconds())
    kind = str(job.get("kind", ""))
    if kind not in {"build", "split", "federated", "simulation"}:
        return

    runtime_summary = _load_runtime_summary_file()
    stage_map = {
        "build": ("build_runtime", "build"),
        "split": ("split_runtime", "split"),
        "federated": ("training_runtime", "training"),
        "simulation": ("simulation_total_runtime", "simulation_total"),
    }
    display_key, seconds_key = stage_map[kind]
    runtime_summary[display_key] = _format_duration(duration_seconds)
    stage_seconds = runtime_summary["stage_seconds"]
    stage_seconds[seconds_key] = duration_seconds
    if kind == "simulation":
        runtime_summary["simulation_runtime"] = _format_duration(duration_seconds)
    else:
        total_seconds = sum(float(stage_seconds[key]) for key in ("build", "split", "training") if stage_seconds.get(key) is not None)
        runtime_summary["total_runtime"] = _format_duration(total_seconds if total_seconds > 0 else None)

    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    peak_memory_bytes = int(job.get("peak_memory_bytes") or 0)
    last_configs = runtime_summary["last_configs"]
    config_payload = {
        "task_type": metadata.get("task_type"),
        "command": job.get("command"),
        "status": job.get("status"),
        "return_code": job.get("return_code"),
        "duration_seconds": round(duration_seconds, 3),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory": _format_memory_bytes(peak_memory_bytes),
        "started_at": job.get("started_at"),
        "ended_at": job.get("ended_at"),
        **(metadata.get("config") if isinstance(metadata.get("config"), dict) else {}),
    }
    config_bucket = "simulation" if kind == "simulation" else seconds_key
    last_configs[config_bucket] = config_payload

    history = runtime_summary["history"]
    history.insert(
        0,
        {
            "kind": kind,
            "task_type": metadata.get("task_type"),
            "status": job.get("status"),
            "duration": _format_duration(duration_seconds),
            "duration_seconds": round(duration_seconds, 3),
            "started_at": job.get("started_at"),
            "ended_at": job.get("ended_at"),
            "return_code": job.get("return_code"),
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory": _format_memory_bytes(peak_memory_bytes),
            "config": metadata.get("config", {}),
            "command": job.get("command"),
        },
    )
    runtime_summary["history"] = history[:_RUNTIME_HISTORY_LIMIT]
    _save_runtime_summary_file(runtime_summary)


def _normalize_task_type(value: object) -> str:
    task_type = str(value or "k_hours").strip().lower()
    return task_type if task_type in _TASK_DIRS else "k_hours"


def _ensure_run_metadata_files() -> None:
    _RUN_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _LAST_RUNTIME_SUMMARY_PATH.exists():
        _save_runtime_summary_file(_default_runtime_summary())
    if not _PC_SPEC_PATH.exists():
        _PC_SPEC_PATH.write_text(json.dumps(_default_pc_spec(), indent=2), encoding="utf-8")
    try:
        with open(_LAST_RUNTIME_SUMMARY_PATH, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        _save_runtime_summary_file(_default_runtime_summary())
    try:
        with open(_PC_SPEC_PATH, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        _PC_SPEC_PATH.write_text(json.dumps(_default_pc_spec(), indent=2), encoding="utf-8")
    _refresh_pc_spec()


def _load_run_metadata() -> tuple[dict[str, object], dict[str, object]]:
    _ensure_run_metadata_files()
    runtime_summary = _load_runtime_summary_file()
    with open(_PC_SPEC_PATH, "r", encoding="utf-8") as f:
        pc_spec = json.load(f)
    return runtime_summary, pc_spec


def _ai_script(task_type: str, name: str) -> str:
    task_dir = _TASK_DIRS[_normalize_task_type(task_type)]
    return str((task_dir / name).resolve())


def _weights_dir_for_model(task_type: str, model_type: str) -> Path:
    task_type = _normalize_task_type(task_type)
    model_dirs = _MODEL_OUTPUT_DIRS[task_type]
    return model_dirs.get(model_type, model_dirs["mlp"])


def _load_metrics_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _to_native(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    return _to_native(value)


def _series_values(frame: pd.DataFrame, column: str) -> list[float | None]:
    values: list[float | None] = []
    for raw in frame[column].tolist():
        if pd.isna(raw):
            values.append(None)
            continue
        numeric = float(raw)
        values.append(numeric if math.isfinite(numeric) else None)
    return values


def _error_analysis_line_charts(frame: pd.DataFrame) -> list[dict[str, object]]:
    round_values = [int(x) for x in frame.get("round", pd.Series(dtype=int)).tolist()]
    if not round_values:
        return []

    chart_specs = [
        ("global_loss", "Global Loss", "#475569"),
        ("local_loss", "Local Loss", "#475569"),
        ("mae_y_temp_main", "MAE Temp Main", "#1d4ed8"),
        ("mse_y_temp_main", "MSE Temp Main", "#3b82f6"),
        ("rmse_y_temp_main", "RMSE Temp Main", "#2563eb"),
        ("threshold_accuracy_y_temp_main", "Threshold Accuracy Temp Main", "#60a5fa"),
        ("mae_y_temp_toilet", "MAE Temp Toilet", "#0f766e"),
        ("mse_y_temp_toilet", "MSE Temp Toilet", "#14b8a6"),
        ("rmse_y_temp_toilet", "RMSE Temp Toilet", "#0d9488"),
        ("threshold_accuracy_y_temp_toilet", "Threshold Accuracy Temp Toilet", "#5eead4"),
        ("mae_y_light", "MAE Light", "#f59e0b"),
        ("mse_y_light", "MSE Light", "#fbbf24"),
        ("rmse_y_light", "RMSE Light", "#d97706"),
        ("threshold_accuracy_y_light", "Threshold Accuracy Light", "#fcd34d"),
        ("mae_y_sound", "MAE Sound", "#dc2626"),
        ("mse_y_sound", "MSE Sound", "#ef4444"),
        ("rmse_y_sound", "RMSE Sound", "#b91c1c"),
        ("threshold_accuracy_y_sound", "Threshold Accuracy Sound", "#fca5a5"),
        ("regression_correct_rate", "Regression Correct Rate", "#7c3aed"),
        ("temperature_correct_rate", "Temperature Correct Rate", "#8b5cf6"),
        ("airflow_accuracy", "Airflow Accuracy", "#0891b2"),
        ("airflow_precision", "Airflow Precision", "#0ea5e9"),
        ("airflow_recall", "Airflow Recall", "#38bdf8"),
        ("airflow_f1", "Airflow F1", "#0284c7"),
        ("change_accuracy", "Change Accuracy", "#16a34a"),
        ("change_precision", "Change Precision", "#22c55e"),
        ("change_recall", "Change Recall", "#4ade80"),
        ("change_f1", "Change F1", "#15803d"),
    ]

    charts: list[dict[str, object]] = []
    for column, label, color in chart_specs:
        if column not in frame.columns:
            continue
        charts.append(
            {
                "metric": column,
                "label": label,
                "rounds": round_values,
                "series": [
                    {
                        "name": label,
                        "color": color,
                        "values": _series_values(frame, column),
                    }
                ],
            }
        )
    return charts


def _summary_line_charts(frame: pd.DataFrame) -> list[dict[str, object]]:
    round_values = [int(x) for x in frame.get("round", pd.Series(dtype=int)).tolist()]
    if not round_values:
        return []

    charts: list[dict[str, object]] = []
    for column, label, color in (("global_loss", "Loss", "#475569"), ("local_loss", "Loss", "#475569")):
        if column in frame.columns:
            charts.append(
                {
                    "metric": column,
                    "label": label,
                    "rounds": round_values,
                    "series": [
                        {
                            "name": label,
                            "color": color,
                            "values": _series_values(frame, column),
                        }
                    ],
                }
            )
            break

    labels = {
        "temp_main": "Temp Main",
        "temp_toilet": "Temp Toilet",
        "light": "Light",
        "sound": "Sound",
    }
    colors = {
        "temp_main": "#1d4ed8",
        "temp_toilet": "#0f766e",
        "light": "#f59e0b",
        "sound": "#b91c1c",
    }
    for metric_key, metric_label in (("mae", "MAE"), ("mse", "MSE"), ("rmse", "RMSE")):
        chart_series = []
        for suffix, target_label in labels.items():
            column = f"{metric_key}_y_{suffix}"
            if column not in frame.columns:
                continue
            chart_series.append(
                {
                    "name": f"{target_label} {metric_label}",
                    "color": colors[suffix],
                    "values": _series_values(frame, column),
                }
            )
        if chart_series:
            charts.append(
                {
                    "metric": metric_key,
                    "label": metric_label,
                    "rounds": round_values,
                    "series": chart_series,
                }
            )
    return charts


def _make_pie(title: str, slices: list[dict[str, object]]) -> dict[str, object]:
    clean_slices = []
    total = 0.0
    for item in slices:
        value = float(item.get("value", 0.0) or 0.0)
        if not math.isfinite(value):
            value = 0.0
        total += value
        clean_slices.append(
            {
                "label": str(item.get("label", "")),
                "value": value,
                "color": str(item.get("color", "#94a3b8")),
            }
        )
    return {"title": title, "total": total, "slices": clean_slices}


def _global_pies(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    latest = frame.sort_values("round").iloc[-1]
    count = max(int(latest.get("evaluated_examples", 0) or 0), 1)
    pies = []
    for suffix, label, color in (
        ("temp_main", "Temp Main Threshold", "#2563eb"),
        ("temp_toilet", "Temp Toilet Threshold", "#0891b2"),
        ("light", "Light Threshold", "#f59e0b"),
        ("sound", "Sound Threshold", "#ef4444"),
    ):
        correct_key = f"threshold_correct_y_{suffix}"
        wrong_key = f"threshold_wrong_y_{suffix}"
        if correct_key in latest.index or wrong_key in latest.index:
            correct = float(latest.get(correct_key, 0.0) or 0.0)
            wrong = float(latest.get(wrong_key, 0.0) or 0.0)
        else:
            accuracy = float(latest.get(f"threshold_accuracy_y_{suffix}", 0.0) or 0.0)
            correct = max(0, min(count, int(round(accuracy * count))))
            wrong = max(0, count - correct)
        pies.append(
            _make_pie(
                label,
                [
                    {"label": "Correct", "value": correct, "color": color},
                    {"label": "Wrong", "value": wrong, "color": "#e5e7eb"},
                ],
            )
        )
    pies.append(
        _make_pie(
            "Airflow Confusion",
            [
                {"label": "TP", "value": latest.get("airflow_tp", 0), "color": "#16a34a"},
                {"label": "TN", "value": latest.get("airflow_tn", 0), "color": "#2563eb"},
                {"label": "FP", "value": latest.get("airflow_fp", 0), "color": "#f59e0b"},
                {"label": "FN", "value": latest.get("airflow_fn", 0), "color": "#dc2626"},
            ],
        )
    )
    pies.append(
        _make_pie(
            "Change Confusion",
            [
                {"label": "TP", "value": latest.get("change_tp", 0), "color": "#16a34a"},
                {"label": "TN", "value": latest.get("change_tn", 0), "color": "#2563eb"},
                {"label": "FP", "value": latest.get("change_fp", 0), "color": "#f59e0b"},
                {"label": "FN", "value": latest.get("change_fn", 0), "color": "#dc2626"},
            ],
        )
    )
    return pies


def _event_summary_line_charts(frame: pd.DataFrame) -> list[dict[str, object]]:
    round_values = [int(x) for x in frame.get("round", pd.Series(dtype=int)).tolist()]
    if not round_values:
        return []

    charts: list[dict[str, object]] = []
    labels = {
        "target_temp_main": "Temp Main",
        "target_temp_toilet": "Temp Toilet",
        "target_light": "Light",
        "target_sound": "Sound",
    }
    colors = {
        "target_temp_main": "#1d4ed8",
        "target_temp_toilet": "#0f766e",
        "target_light": "#f59e0b",
        "target_sound": "#b91c1c",
    }
    for metric_key, metric_label in (("mae", "MAE"), ("rmse", "RMSE")):
        chart_series = []
        for suffix, target_label in labels.items():
            column = f"{metric_key}_y_{suffix}"
            if column not in frame.columns:
                continue
            chart_series.append(
                {
                    "name": f"{target_label} {metric_label}",
                    "color": colors[suffix],
                    "values": _series_values(frame, column),
                }
            )
        if chart_series:
            charts.append(
                {
                    "metric": metric_key,
                    "label": metric_label,
                    "rounds": round_values,
                    "series": chart_series,
                }
            )

    for column, label, color in (
        ("airflow_accuracy", "Airflow Accuracy", "#2563eb"),
        ("airflow_f1", "Airflow F1", "#16a34a"),
    ):
        if column in frame.columns:
            charts.append(
                {
                    "metric": column,
                    "label": label,
                    "rounds": round_values,
                    "series": [
                        {
                            "name": label,
                            "color": color,
                            "values": _series_values(frame, column),
                        }
                    ],
                }
            )
    return charts


def _event_error_analysis_line_charts(frame: pd.DataFrame) -> list[dict[str, object]]:
    round_values = [int(x) for x in frame.get("round", pd.Series(dtype=int)).tolist()]
    if not round_values:
        return []

    chart_specs = [
        ("global_loss", "Global Loss", "#475569"),
        ("mae_y_target_temp_main", "MAE Temp Main", "#1d4ed8"),
        ("rmse_y_target_temp_main", "RMSE Temp Main", "#2563eb"),
        ("threshold_accuracy_y_temp_main", "Threshold Accuracy Temp Main", "#60a5fa"),
        ("mae_y_target_temp_toilet", "MAE Temp Toilet", "#0f766e"),
        ("rmse_y_target_temp_toilet", "RMSE Temp Toilet", "#0d9488"),
        ("threshold_accuracy_y_temp_toilet", "Threshold Accuracy Temp Toilet", "#5eead4"),
        ("mae_y_target_light", "MAE Light", "#f59e0b"),
        ("rmse_y_target_light", "RMSE Light", "#d97706"),
        ("threshold_accuracy_y_light", "Threshold Accuracy Light", "#fcd34d"),
        ("mae_y_target_sound", "MAE Sound", "#dc2626"),
        ("rmse_y_target_sound", "RMSE Sound", "#b91c1c"),
        ("threshold_accuracy_y_sound", "Threshold Accuracy Sound", "#fca5a5"),
        ("airflow_accuracy", "Airflow Accuracy", "#2563eb"),
        ("airflow_precision", "Airflow Precision", "#0ea5e9"),
        ("airflow_recall", "Airflow Recall", "#38bdf8"),
        ("airflow_f1", "Airflow F1", "#16a34a"),
    ]

    charts: list[dict[str, object]] = []
    for column, label, color in chart_specs:
        if column not in frame.columns:
            continue
        charts.append(
            {
                "metric": column,
                "label": label,
                "rounds": round_values,
                "series": [
                    {
                        "name": label,
                        "color": color,
                        "values": _series_values(frame, column),
                    }
                ],
            }
        )
    return charts


def _event_global_pies(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    latest = frame.sort_values("round").iloc[-1]
    pies = []
    for suffix, label, color in (
        ("temp_main", "Temp Main Threshold", "#2563eb"),
        ("temp_toilet", "Temp Toilet Threshold", "#0891b2"),
        ("light", "Light Threshold", "#f59e0b"),
        ("sound", "Sound Threshold", "#ef4444"),
    ):
        pies.append(
            _make_pie(
                label,
                [
                    {"label": "Correct", "value": latest.get(f"threshold_correct_y_{suffix}", 0), "color": color},
                    {"label": "Wrong", "value": latest.get(f"threshold_wrong_y_{suffix}", 0), "color": "#e5e7eb"},
                ],
            )
        )
    pies.append(
        _make_pie(
            "Airflow Confusion",
            [
                {"label": "TP", "value": latest.get("airflow_tp", 0), "color": "#16a34a"},
                {"label": "TN", "value": latest.get("airflow_tn", 0), "color": "#2563eb"},
                {"label": "FP", "value": latest.get("airflow_fp", 0), "color": "#f59e0b"},
                {"label": "FN", "value": latest.get("airflow_fn", 0), "color": "#dc2626"},
            ],
        )
    )
    return pies


def _event_local_pies(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    latest = frame.sort_values("round").iloc[-1]
    count = max(int(latest.get("num_examples", 0) or 0), 1)
    pies = []
    for suffix, label, color in (
        ("temp_main", "Temp Main Threshold", "#2563eb"),
        ("temp_toilet", "Temp Toilet Threshold", "#0891b2"),
        ("light", "Light Threshold", "#f59e0b"),
        ("sound", "Sound Threshold", "#ef4444"),
    ):
        accuracy = float(latest.get(f"threshold_accuracy_y_{suffix}", 0.0) or 0.0)
        correct = max(0, min(count, int(round(accuracy * count))))
        wrong = max(0, count - correct)
        pies.append(
            _make_pie(
                label,
                [
                    {"label": "Correct", "value": correct, "color": color},
                    {"label": "Wrong", "value": wrong, "color": "#e5e7eb"},
                ],
            )
        )
    pies.append(
        _make_pie(
            "Airflow Confusion",
            [
                {"label": "TP", "value": latest.get("airflow_tp", 0), "color": "#16a34a"},
                {"label": "TN", "value": latest.get("airflow_tn", 0), "color": "#2563eb"},
                {"label": "FP", "value": latest.get("airflow_fp", 0), "color": "#f59e0b"},
                {"label": "FN", "value": latest.get("airflow_fn", 0), "color": "#dc2626"},
            ],
        )
    )
    return pies


def _local_pies(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    latest = frame.sort_values("round").iloc[-1]
    count = max(int(latest.get("num_examples", 0) or 0), 1)
    pies = []
    for suffix, label, color in (
        ("temp_main", "Temp Main Threshold", "#2563eb"),
        ("temp_toilet", "Temp Toilet Threshold", "#0891b2"),
        ("light", "Light Threshold", "#f59e0b"),
        ("sound", "Sound Threshold", "#ef4444"),
    ):
        accuracy = float(latest.get(f"threshold_accuracy_y_{suffix}", 0.0) or 0.0)
        correct = max(0, min(count, int(round(accuracy * count))))
        wrong = max(0, count - correct)
        pies.append(
            _make_pie(
                label,
                [
                    {"label": "Correct", "value": correct, "color": color},
                    {"label": "Wrong", "value": wrong, "color": "#e5e7eb"},
                ],
            )
        )
    pies.append(
        _make_pie(
            "Airflow Confusion",
            [
                {"label": "TP", "value": latest.get("airflow_tp", 0), "color": "#16a34a"},
                {"label": "TN", "value": latest.get("airflow_tn", 0), "color": "#2563eb"},
                {"label": "FP", "value": latest.get("airflow_fp", 0), "color": "#f59e0b"},
                {"label": "FN", "value": latest.get("airflow_fn", 0), "color": "#dc2626"},
            ],
        )
    )
    pies.append(
        _make_pie(
            "Change Confusion",
            [
                {"label": "TP", "value": latest.get("change_tp", 0), "color": "#16a34a"},
                {"label": "TN", "value": latest.get("change_tn", 0), "color": "#2563eb"},
                {"label": "FP", "value": latest.get("change_fp", 0), "color": "#f59e0b"},
                {"label": "FN", "value": latest.get("change_fn", 0), "color": "#dc2626"},
            ],
        )
    )
    return pies


def _extract_job_metrics(job: dict) -> dict:
    raw = _parse_kv_lines(job.get("logs", []))
    peak_memory = _format_memory_bytes(int(job.get("peak_memory_bytes") or 0))
    if job.get("kind") == "build":
        items = [
            {
                "key": "assignments_processed",
                "label": "Assignments Processed",
                "value": raw.get("assignments_processed"),
                "explain": "Number of patient-room assignments turned into training rows.",
            },
            {
                "key": "next_hour_rows",
                "label": "Rows Built",
                "value": raw.get("next_hour_rows"),
                "explain": "Total rows created for the k-hours based dataset.",
            },
            {
                "key": "workers_used",
                "label": "Workers Used",
                "value": raw.get("workers_used"),
                "explain": "Parallel worker processes used during row generation.",
            },
            {
                "key": "peak_memory",
                "label": "Peak Memory",
                "value": peak_memory,
                "explain": "Peak RAM observed for the build process tree.",
            },
        ]
        return {"raw": raw, "items": items}

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
            {
                "key": "peak_memory",
                "label": "Peak Memory",
                "value": peak_memory,
                "explain": "Peak RAM observed for the training process tree.",
            },
        ]
        return {"raw": raw, "items": items}

    if job.get("kind") == "split":
        items = [
            {
                "key": "next_hour_total",
                "label": "Total Rows",
                "value": raw.get("next_hour_total"),
                "explain": "Total rows available before train/test split.",
            },
            {
                "key": "train",
                "label": "Train Rows",
                "value": raw.get("train"),
                "explain": "Rows assigned to the training split.",
            },
            {
                "key": "test",
                "label": "Test Rows",
                "value": raw.get("test"),
                "explain": "Rows assigned to the evaluation split.",
            },
            {
                "key": "peak_memory",
                "label": "Peak Memory",
                "value": peak_memory,
                "explain": "Peak RAM observed for the split process tree.",
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


def _sensor_rows_in_window(
    room_sensor_rows: list[dict],
    *,
    start: datetime | None,
    end: datetime | None,
) -> list[dict]:
    rows: list[dict] = []
    for row in room_sensor_rows:
        ts = row.get("timestamp")
        if ts is None:
            continue
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        rows.append(row)
    return rows


def _sensor_label(sensor_type: str | None, location: str | None) -> str:
    sensor_name = str(sensor_type or "sensor").strip() or "sensor"
    sensor_location = str(location or "").strip()
    if sensor_location:
        return f"{sensor_name} ({sensor_location})"
    return sensor_name


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

    simulation_period = _simulation_period_from_admissions()

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

    print(simulation_info)

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
    room_sensor_rows = data["data_by_room"].get(room_id, [])
    room_sensors = data["sensors_by_room"].get(room_id, [])

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
        pref_ts = pref["timestamp"]
        sensor_windows = []
        for sensor in room_sensors:
            before_rows = _sensor_rows_in_window(
                room_sensor_rows,
                start=pref_ts - timedelta(hours=1) if pref_ts else None,
                end=pref_ts,
            )
            before_rows = [row for row in before_rows if row["sensor_id"] == sensor["sensor_id"]][-10:]
            after_rows = _sensor_rows_in_window(
                room_sensor_rows,
                start=pref_ts,
                end=pref_ts + timedelta(hours=1) if pref_ts else None,
            )
            after_rows = [row for row in after_rows if row["sensor_id"] == sensor["sensor_id"]][:10]
            sensor_windows.append(
                {
                    "sensor_id": sensor["sensor_id"],
                    "device_id": sensor["device_id"],
                    "sensor_type": _sensor_label(sensor["sensor_type"], sensor.get("location")),
                    "unit": sensor["unit"],
                    "before_rows": before_rows,
                    "after_rows": after_rows,
                }
            )
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
                "sensor_windows": sensor_windows,
            }
        )

    utility_usages = data["utility_by_room"].get(room_id, [])[:10]
    ventilation_data = data["ventilations_by_room"].get(room_id, [])
    toilet_heater_data = data["toilet_heaters_by_room"].get(room_id, [])
    toilet_light_data = data["toilet_lights_by_room"].get(room_id, [])
    room_data = {"room_id": room["room_id"], "room_number": room["room_number"], "devices": device_data}

    return render_template(
        "room_detail.html",
        room=room_data,
        assignments=assignments,
        comfort_preferences=comfort_preferences,
        utility_usages=utility_usages,
        ventilation_data=ventilation_data,
        toilet_heater_data=toilet_heater_data,
        toilet_light_data=toilet_light_data,
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


@sim_bp.route("/ai-training")
def ai_training():
    runtime_summary, pc_spec = _load_run_metadata()
    diagnosis_names = _DIAGNOSIS_NAMES if _PREDICT_IMPORT_ERROR is None else []
    symptom_names = _SYMPTOM_NAMES if _PREDICT_IMPORT_ERROR is None else []
    medication_names = _MEDICATION_NAMES if _PREDICT_IMPORT_ERROR is None else []
    return render_template(
        "ai_suggestion.html",
        diagnosis_names=diagnosis_names,
        symptom_names=symptom_names,
        medication_names=medication_names,
        predict_import_error=_PREDICT_IMPORT_ERROR,
        runtime_summary=runtime_summary,
        pc_spec=pc_spec,
        runtime_summary_path=str(_LAST_RUNTIME_SUMMARY_PATH),
        pc_spec_path=str(_PC_SPEC_PATH),
    )


@sim_bp.route("/ai-suggestion")
def ai_suggestion_redirect():
    return redirect("/ai-training", code=301)


@sim_bp.route("/runtime-stats")
def runtime_stats():
    runtime_summary, pc_spec = _load_run_metadata()
    return render_template(
        "runtime_stats.html",
        runtime_summary=runtime_summary,
        pc_spec=pc_spec,
        runtime_summary_path=str(_LAST_RUNTIME_SUMMARY_PATH),
        pc_spec_path=str(_PC_SPEC_PATH),
    )


@sim_bp.route("/api/ai/build/start", methods=["POST"])
def api_ai_build_start():
    payload = request.get_json(silent=True) or {}
    task_type = _normalize_task_type(payload.get("task_type", "k_hours"))
    build_command, build_config = _build_command_and_config(payload, task_type)
    job_id = _start_ai_job("build", [build_command], metadata={"task_type": task_type, "config": build_config})
    return jsonify({"job_id": job_id, "status": "running"})


@sim_bp.route("/api/ai/split/start", methods=["POST"])
def api_ai_split_start():
    payload = request.get_json(silent=True) or {}
    task_type = _normalize_task_type(payload.get("task_type", "k_hours"))
    build_command, build_config = _build_command_and_config(payload, task_type)
    split_command, split_config = _split_command_and_config(payload, task_type)
    combined_config = {
        "mode": task_type,
        "build": build_config,
        "split": split_config,
    }
    job_id = _start_ai_job("split", [build_command, split_command], metadata={"task_type": task_type, "config": combined_config})
    return jsonify({"job_id": job_id, "status": "running"})


@sim_bp.route("/api/ai/federated/start", methods=["POST"])
def api_ai_federated_start():
    payload = request.get_json(silent=True) or {}
    task_type = _normalize_task_type(payload.get("task_type", "k_hours"))
    model_type = str(payload.get("model_type", "mlp"))
    aggregation_method = str(payload.get("aggregation_method", "fedavg"))
    proximal_mu = float(payload.get("proximal_mu", 0.0))
    rounds = int(payload.get("rounds", 5))
    local_epochs = int(payload.get("local_epochs", 1))
    lstm_sequence_length = int(payload.get("lstm_sequence_length", 4))
    lstm_batch_size = int(payload.get("lstm_batch_size", 32))
    mlp_hidden_layers = str(payload.get("mlp_hidden_layers", "128,64,32"))
    mlp_batch_size = int(payload.get("mlp_batch_size", 1 if task_type == "event" else 32))
    mlp_learning_rate = float(payload.get("mlp_learning_rate", 1e-3))
    mlp_optimizer = str(payload.get("mlp_optimizer", "adam"))
    mlp_activation = str(payload.get("mlp_activation", "relu"))
    lstm_hidden_dim = int(payload.get("lstm_hidden_dim", 64))
    lstm_num_layers = int(payload.get("lstm_num_layers", 1))
    lstm_head_hidden_dim = int(payload.get("lstm_head_hidden_dim", 64))
    lstm_learning_rate = float(payload.get("lstm_learning_rate", 1e-3))
    lstm_optimizer = str(payload.get("lstm_optimizer", "adam"))
    lstm_activation = str(payload.get("lstm_activation", "relu"))
    max_rooms_raw = payload.get("max_rooms")
    max_rooms = int(max_rooms_raw) if max_rooms_raw not in (None, "") else None
    fraction_fit = float(payload.get("fraction_fit", 1.0))
    fraction_evaluate = float(payload.get("fraction_evaluate", 1.0))
    min_fit_clients = int(payload.get("min_fit_clients", 2))
    min_evaluate_clients = int(payload.get("min_evaluate_clients", 2))
    min_available_clients = int(payload.get("min_available_clients", 2))
    workers_raw = payload.get("workers")
    if workers_raw not in (None, ""):
        workers = max(1, int(workers_raw))
        total_cpus = max(1, int(os.cpu_count() or 1))
        client_cpu = total_cpus / float(workers)
    else:
        workers = _default_federated_workers()
        client_cpu = float(payload.get("client_cpu", _default_federated_client_cpu()))
    chunksize = int(payload.get("chunksize", _DEFAULT_FEDERATED_CHUNKSIZE))

    sim_map = {
        "k_hours": {
            "mlp": "fl_mlp_simulation.py",
            "lstm": "fl_lstm_simulation.py",
        },
        "event": {
            "mlp": "fl_simulation.py",
        },
    }
    sim_script = sim_map.get(task_type, {}).get(model_type)
    if not sim_script:
        return jsonify({"error": f"Unsupported model type: {model_type}"}), 400

    weights_out_dir = _weights_dir_for_model(task_type, model_type)
    summary_dir = weights_out_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_name = (
        {
            "mlp": "dashboard_summary.json",
            "lstm": "dashboard_lstm_summary.json",
        }.get(model_type, "dashboard_summary.json")
        if task_type == "k_hours"
        else "dashboard_event_summary.json"
    )

    command = [
        sys.executable,
        "-u",
        _ai_script(task_type, sim_script),
        "--split-dir",
        str(_SPLIT_OUTPUT_DIRS[task_type]),
        "--rounds",
        str(rounds),
    ]
    if task_type == "k_hours":
        command.extend([
            "--aggregation-method",
            aggregation_method,
            "--local-epochs",
            str(local_epochs),
            "--fraction-fit",
            str(fraction_fit),
            "--fraction-evaluate",
            str(fraction_evaluate),
            "--min-fit-clients",
            str(min_fit_clients),
            "--min-evaluate-clients",
            str(min_evaluate_clients),
            "--min-available-clients",
            str(min_available_clients),
        ])
    else:
        command.extend([
            "--aggregation-method",
            aggregation_method,
            "--local-epochs",
            str(local_epochs),
        ])
    command.extend([
        "--weights-out-dir",
        str(weights_out_dir),
        "--summary-out",
        str(summary_dir / summary_name),
        "--client-cpu",
        str(client_cpu),
        "--chunksize",
        str(chunksize),
    ])
    if max_rooms is not None:
        command.extend(["--max-rooms", str(max_rooms)])
    if aggregation_method == "fedprox":
        command.extend(["--proximal-mu", str(proximal_mu)])
    if task_type == "k_hours" and model_type != "mlp":
        command.extend(
            [
                "--sequence-length",
                str(lstm_sequence_length),
                "--batch-size",
                str(lstm_batch_size),
            ]
        )
    if task_type == "event":
        command.extend(
            [
                "--hidden-layers",
                mlp_hidden_layers,
                "--batch-size",
                str(mlp_batch_size),
                "--learning-rate",
                str(mlp_learning_rate),
                "--optimizer",
                mlp_optimizer,
                "--activation",
                mlp_activation,
            ]
        )
    elif model_type == "mlp":
        command.extend(
            [
                "--hidden-layers",
                mlp_hidden_layers,
                "--batch-size",
                str(mlp_batch_size),
                "--learning-rate",
                str(mlp_learning_rate),
                "--optimizer",
                mlp_optimizer,
                "--activation",
                mlp_activation,
            ]
        )
    elif model_type == "lstm":
        command.extend(
            [
                "--hidden-dim",
                str(lstm_hidden_dim),
                "--num-layers",
                str(lstm_num_layers),
                "--head-hidden-dim",
                str(lstm_head_hidden_dim),
                "--learning-rate",
                str(lstm_learning_rate),
                "--optimizer",
                lstm_optimizer,
                "--activation",
                lstm_activation,
            ]
        )
    training_config = {
        "mode": task_type,
        "model_type": model_type,
        "aggregation_method": aggregation_method,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "split_dir": str(_SPLIT_OUTPUT_DIRS[task_type]),
        "weights_out_dir": str(weights_out_dir),
        "summary_out": str(summary_dir / summary_name),
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "workers": workers,
        "client_cpu": client_cpu,
        "chunksize": chunksize,
    }
    if max_rooms is not None:
        training_config["max_rooms"] = max_rooms
    if aggregation_method == "fedprox":
        training_config["proximal_mu"] = proximal_mu
    if task_type == "event":
        training_config.update(
            {
                "hidden_layers": mlp_hidden_layers,
                "batch_size": mlp_batch_size,
                "learning_rate": mlp_learning_rate,
                "optimizer": mlp_optimizer,
                "activation": mlp_activation,
            }
        )
    elif model_type == "mlp":
        training_config.update(
            {
                "hidden_layers": mlp_hidden_layers,
                "batch_size": mlp_batch_size,
                "learning_rate": mlp_learning_rate,
                "optimizer": mlp_optimizer,
                "activation": mlp_activation,
            }
        )
    elif model_type == "lstm":
        training_config.update(
            {
                "sequence_length": lstm_sequence_length,
                "batch_size": lstm_batch_size,
                "hidden_dim": lstm_hidden_dim,
                "num_layers": lstm_num_layers,
                "head_hidden_dim": lstm_head_hidden_dim,
                "learning_rate": lstm_learning_rate,
                "optimizer": lstm_optimizer,
                "activation": lstm_activation,
            }
        )
    job_id = _start_ai_job("federated", [command], metadata={"task_type": task_type, "config": training_config})
    return jsonify({"job_id": job_id, "status": "running"})


def _simulation_config_defaults() -> dict[str, object]:
    return {
        "start_date": getattr(sim_config, "START_DATE", None),
        "days": getattr(sim_config, "DAYS", None),
        "patient_count": getattr(sim_config, "PATIENT_COUNT", None),
        "change_room_prob": getattr(sim_config, "CHANGE_ROOM_PROB", None),
        "min_days_before_transfer": getattr(sim_config, "MIN_DAYS_BEFORE_TRANSFER", None),
        "min_days_after_transfer": getattr(sim_config, "MIN_DAYS_AFTER_TRANSFER", None),
        "comfort_max_changes_per_day": getattr(sim_config, "COMFORT_MAX_CHANGES_PER_DAY", None),
        "sim_step_s": getattr(sim_config, "SIM_STEP_S", None),
        "sensor_sample_every_s": getattr(sim_config, "SENSOR_SAMPLE_EVERY_S", None),
        "wall_sleep_s": getattr(sim_config, "WALL_SLEEP_S", None),
        "enable_comfort": getattr(sim_config, "ENABLE_COMFORT", None),
        "enable_medication": getattr(sim_config, "ENABLE_MEDICATION", None),
        "enable_visits": getattr(sim_config, "ENABLE_VISITS", None),
        "enable_toilet_usage": getattr(sim_config, "ENABLE_TOILET_USAGE", None),
        "enable_sensor_emit": getattr(sim_config, "ENABLE_SENSOR_EMIT", None),
        "enable_utility_usage": getattr(sim_config, "ENABLE_UTILITY_USAGE", None),
        "random_seed": getattr(sim_config, "RANDOM_SEED", None),
    }


@sim_bp.route("/new-simulation")
def new_simulation():
    runtime_summary, pc_spec = _load_run_metadata()
    config_defaults = _simulation_config_defaults()
    return render_template(
        "new_simulation.html",
        config_defaults=config_defaults,
        simulation_command="python main.py --mode simulation",
        runtime_summary=runtime_summary,
        pc_spec=pc_spec,
        runtime_summary_path=str(_LAST_RUNTIME_SUMMARY_PATH),
        pc_spec_path=str(_PC_SPEC_PATH),
    )


@sim_bp.route("/api/simulation/start", methods=["POST"])
def api_simulation_start():
    active_job_id = _active_job_id("simulation")
    if active_job_id:
        return jsonify({"error": "A simulation is already running.", "job_id": active_job_id}), 409

    payload = request.get_json(silent=True) or {}
    command, config_payload = _simulation_command_and_config(payload)
    job_id = _start_ai_job("simulation", [command], metadata={"task_type": "simulation", "config": config_payload})
    return jsonify({"job_id": job_id, "status": "running"})


@sim_bp.route("/api/ai/results", methods=["GET"])
def api_ai_results():
    task_type = _normalize_task_type(request.args.get("task_type", "k_hours"))
    model_type = str(request.args.get("model_type", "mlp") or "mlp")
    scope = str(request.args.get("scope", "global") or "global")
    dataset = str(request.args.get("dataset", "test") or "test")
    figure_detail = str(request.args.get("figure_detail", "summary") or "summary")
    requested_room_id = str(request.args.get("room_id", "") or "").strip()

    weights_dir = _weights_dir_for_model(task_type, model_type)
    if task_type == "event":
        metric_rows: list[dict[str, object]] = []
        for path in sorted(weights_dir.glob("round_*_total_metrics.csv"), key=lambda item: int(item.stem.split("_")[1])):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if frame.empty:
                continue
            row = frame.iloc[-1].to_dict()
            row["round"] = int(path.stem.split("_")[1])
            metric_rows.append(row)

        total_frame = pd.DataFrame(metric_rows)
        room_metric_rows: list[dict[str, object]] = []
        for path in sorted(weights_dir.glob("round_*_room_metrics.csv"), key=lambda item: int(item.stem.split("_")[1])):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if frame.empty:
                continue
            if "round" not in frame.columns:
                frame["round"] = int(path.stem.split("_")[1])
            room_metric_rows.extend(frame.to_dict(orient="records"))

        room_frame = pd.DataFrame(room_metric_rows)
        if total_frame.empty and room_frame.empty:
            return jsonify({"error": f"No metrics found for model '{model_type}' in {weights_dir}"}), 404

        if not total_frame.empty and "round" in total_frame.columns:
            total_frame = total_frame.sort_values("round").reset_index(drop=True)
        if not room_frame.empty:
            room_frame["room_id"] = room_frame["room_id"].astype(str)
            if "round" in room_frame.columns:
                room_frame = room_frame.sort_values(["room_id", "round"]).reset_index(drop=True)

        room_ids = sorted(room_frame["room_id"].dropna().astype(str).unique().tolist()) if not room_frame.empty else []
        active_room_id = requested_room_id
        if scope == "local":
            if not active_room_id and room_ids:
                active_room_id = room_ids[0]
            if active_room_id:
                room_frame = room_frame[room_frame["room_id"] == active_room_id].copy()

        source_frame = room_frame if scope == "local" else total_frame
        if source_frame.empty:
            return jsonify({"error": f"No metrics available for scope '{scope}'"}), 404

        latest_row = source_frame.sort_values("round").iloc[-1].to_dict()
        cards = [
            {"label": "Round", "value": _to_native(latest_row.get("round")), "explain": "Latest completed federated round."},
            {"label": "MAE Temp Main", "value": _to_native(latest_row.get("mae_y_target_temp_main")), "explain": "Average absolute error for main temperature."},
            {"label": "MAE Temp Toilet", "value": _to_native(latest_row.get("mae_y_target_temp_toilet")), "explain": "Average absolute error for toilet temperature."},
            {"label": "MAE Light", "value": _to_native(latest_row.get("mae_y_target_light")), "explain": "Average absolute error for light intensity."},
            {"label": "MAE Sound", "value": _to_native(latest_row.get("mae_y_target_sound")), "explain": "Average absolute error for sound level."},
            {"label": "RMSE Temp Main", "value": _to_native(latest_row.get("rmse_y_target_temp_main")), "explain": "Root mean squared error for main temperature."},
            {"label": "Airflow Accuracy", "value": _to_native(latest_row.get("airflow_accuracy")), "explain": "Overall correctness of airflow classification."},
            {"label": "Airflow F1", "value": _to_native(latest_row.get("airflow_f1")), "explain": "Balance between airflow precision and recall."},
            {"label": "Evaluated Samples", "value": _to_native(latest_row.get("evaluated_examples") or latest_row.get("num_examples")), "explain": "Number of test examples included in evaluation."},
        ]
        return jsonify(
            _sanitize_for_json(
                {
                    "task_type": task_type,
                    "model_type": model_type,
                    "scope": scope,
                    "dataset": dataset,
                    "figure_detail": figure_detail,
                    "room_id": active_room_id or None,
                    "room_ids": room_ids,
                    "weights_dir": str(weights_dir),
                    "latest_cards": cards,
                    "line_charts": _event_error_analysis_line_charts(source_frame) if figure_detail == "detailed" else _event_summary_line_charts(source_frame),
                    "pie_charts": _event_local_pies(source_frame) if scope == "local" else _event_global_pies(source_frame),
                }
            )
        )

    if scope == "global":
        total_frame = _load_metrics_csv(weights_dir / "total_metrics.csv")
        room_frame = _load_metrics_csv(weights_dir / "room_metrics.csv")
    elif dataset == "training":
        total_frame = _load_metrics_csv(weights_dir / "train_metrics.csv")
        room_frame = _load_metrics_csv(weights_dir / "train_room_metrics.csv")
    else:
        total_frame = _load_metrics_csv(weights_dir / "local_test_metrics.csv")
        room_frame = _load_metrics_csv(weights_dir / "local_test_room_metrics.csv")

    if total_frame.empty and room_frame.empty:
        return jsonify({"error": f"No metrics found for model '{model_type}' in {weights_dir}"}), 404

    if not total_frame.empty and "round" in total_frame.columns:
        total_frame = total_frame.sort_values("round").reset_index(drop=True)
    if not room_frame.empty:
        room_frame["room_id"] = room_frame["room_id"].astype(str)
        if "round" in room_frame.columns:
            room_frame = room_frame.sort_values(["room_id", "round"]).reset_index(drop=True)

    room_ids = sorted(room_frame["room_id"].dropna().astype(str).unique().tolist()) if not room_frame.empty else []
    active_room_id = requested_room_id
    if scope == "local":
        if not active_room_id and room_ids:
            active_room_id = room_ids[0]
        if active_room_id:
            room_frame = room_frame[room_frame["room_id"] == active_room_id].copy()

    source_frame = room_frame if scope == "local" else total_frame
    if source_frame.empty:
        return jsonify({"error": f"No metrics available for scope '{scope}'"}), 404

    latest_row = source_frame.sort_values("round").iloc[-1].to_dict()
    cards = [
        {"label": "Round", "value": _to_native(latest_row.get("round")), "explain": "Latest completed federated round."},
        {"label": "Train Loss", "value": _to_native(latest_row.get("train_loss")), "explain": "Average optimization loss during local client training."},
        {"label": "MAE Temp Main", "value": _to_native(latest_row.get("mae_y_temp_main")), "explain": "Average absolute error for main temperature."},
        {"label": "RMSE Temp Main", "value": _to_native(latest_row.get("rmse_y_temp_main")), "explain": "Root mean squared error for main temperature."},
        {"label": "Regression Correct Rate", "value": _to_native(latest_row.get("regression_correct_rate")), "explain": "Share of samples meeting the regression correctness criterion."},
        {"label": "Airflow F1", "value": _to_native(latest_row.get("airflow_f1")), "explain": "Balance between airflow precision and recall."},
        {"label": "Change Precision", "value": _to_native(latest_row.get("change_precision")), "explain": "Of predicted changes, how many were true changes."},
        {"label": "Change Recall", "value": _to_native(latest_row.get("change_recall")), "explain": "Of real changes, how many the model detected."},
        {"label": "Change F1", "value": _to_native(latest_row.get("change_f1")), "explain": "Balance between change precision and recall."},
        {"label": "Samples", "value": _to_native(latest_row.get("evaluated_examples") or latest_row.get("num_examples") or latest_row.get("trained_examples")), "explain": "Number of rows included in this result set."},
    ]

    return jsonify(
        _sanitize_for_json(
            {
                "model_type": model_type,
                "scope": scope,
                "dataset": dataset,
                "figure_detail": figure_detail,
                "room_id": active_room_id or None,
                "room_ids": room_ids,
                "weights_dir": str(weights_dir),
                "latest_cards": cards,
                "line_charts": _error_analysis_line_charts(source_frame) if figure_detail == "detailed" else _summary_line_charts(source_frame),
                "pie_charts": _local_pies(source_frame) if scope == "local" else _global_pies(source_frame),
            }
        )
    )


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
        runtime_summary, pc_spec = _load_run_metadata()
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
                "peak_memory_bytes": int(job.get("peak_memory_bytes") or 0),
                "peak_memory": _format_memory_bytes(int(job.get("peak_memory_bytes") or 0)),
                "runtime_summary": runtime_summary,
                "pc_spec": pc_spec,
                "metadata": job.get("metadata", {}),
            }
        )


@sim_bp.route("/api/ai/jobs/active", methods=["GET"])
def api_ai_active_jobs():
    with _AI_JOBS_LOCK:
        jobs = [
            {
                "id": job.get("id"),
                "kind": job.get("kind"),
                "status": job.get("status"),
                "progress": job.get("progress"),
                "started_at": job.get("started_at"),
                "metadata": job.get("metadata", {}),
            }
            for job in _AI_JOBS.values()
            if job.get("status") == "running"
        ]
    jobs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    return jsonify({"jobs": jobs})


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
        pid = int(job.get("pid") or 0)

    if pid > 0 and platform.system().lower() == "windows":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:  # noqa: BLE001
            pass
    elif process is not None:
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
    room_sensor_rows_map = data["data_by_room"]

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
            power_text = "--" if usage["power_consumption"] is None else f"{float(usage['power_consumption']):.3f}"
            if usage["category"] == "hvac":
                short_start = _fmt_time(usage["start_time"])
                short_end = _fmt_time(usage["end_time"])
                env_text = (
                    f"{room_number} | {short_start}-{short_end} | Power {power_text} kWh | "
                    f"Water {usage['water_consumption'] if usage['water_consumption'] is not None else '--'} L"
                )
            else:
                env_text = (
                    f"{room_number} | Power {power_text} kWh | "
                    f"Water {usage['water_consumption'] if usage['water_consumption'] is not None else '--'} L"
                )
            event_ts = usage["end_time"] if usage["category"] == "water" and usage["end_time"] is not None else usage["start_time"]
            add_event(
                event_ts,
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
            day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
            day_end = day_start + timedelta(days=1)
            sensor_rows: list[dict[str, object]] = []
            for assignment in admission_assignments:
                room_id = assignment["room_id"]
                assign_start = assignment.get("start_time")
                assign_end = assignment.get("end_time")
                window_start = max(
                    [value for value in [day_start, admission_start, assign_start] if value is not None],
                    default=day_start,
                )
                end_candidates = [value for value in [day_end, admission_end, assign_end] if value is not None]
                window_end = min(end_candidates) if end_candidates else day_end
                if window_start >= window_end:
                    continue
                for row in _sensor_rows_in_window(
                    room_sensor_rows_map.get(room_id, []),
                    start=window_start,
                    end=window_end,
                ):
                    sensor_rows.append(
                        {
                            "timestamp": _fmt_dt(row["timestamp"]),
                            "room_id": room_id,
                            "sensor_type": _sensor_label(row["sensor_type"], row.get("location")),
                            "value": row["value"],
                            "unit": row["unit"],
                        }
                    )
            sensor_rows.sort(key=lambda row: str(row["timestamp"]))
            current_day["sensor_rows"] = sensor_rows
            current_day["sensor_count"] = len(sensor_rows)
            current_day["sensor_types"] = sorted({str(row["sensor_type"]) for row in sensor_rows})
            current_day["sensor_rooms"] = sorted({int(row["room_id"]) for row in sensor_rows})
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
