import asyncio
import os
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from persistence import init_db
from persistence.seed_devices import seed_devices_and_sensors
# imort db_sink and datacolletor 
from data_collection.data_collector import DataCollector
from data_collection.db_sink import db_sink

# ---- SIMULATION ----
from simulation_batch.orchestrator import SimulationOrchestrator, OrchestratorConfig
from simulation_batch.seed_world import seed_simulated_world
from simulation_batch.config import START_DATE, DAYS, PATIENT_COUNT

# ---- BLE ----
from ble import BLEManager, Device, Sensor
from ble.characteristics import (
    TEMP_CHAR_UUID,
    HUMIDITY_CHAR_UUID,
    AIR_QUALITY_CHAR_UUID,
    SOUUND_CHAR_UUID,
    LIGHT_CHAR_UUID,
)
from ble.sensor import (
    parse_temp_thingy,
    parse_humidity_thingy,
    parse_air_quality_thingy,
    parse_sound_thingy,
    parse_light_thingy,
)

# ---- HYBRID ----
from hybrid.event_adapter import handle_ble_event
from hybrid.comfort_service import run_cli
from hybrid.hybrid_context import ensure_room_and_patient
from hybrid.event_adapter import handle_ble_event, event_worker


# ---------------- TIME ----------------
def _to_utc_dt(d) -> datetime:
    return datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)


def _run_python_step(step_name: str, script_path: str, args: list[str]) -> None:
    command = [sys.executable, script_path, *args]
    print(f"\n[AI] {step_name}", flush=True)
    print(f"[AI] command={' '.join(command)}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    print(f"[AI] {step_name} finished in {elapsed:.1f}s (exit_code={completed.returncode})", flush=True)


def _run_federated_targets(args, split_dir: str, fl_weights_out_dir: str) -> dict:
    summary_dir = os.path.join(fl_weights_out_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "next_hour_summary.json")
    sim_args = [
        "--split-dir",
        split_dir,
        "--rounds",
        str(args.ai_rounds),
        "--n-features",
        str(args.ai_n_features),
        "--local-epochs",
        str(args.ai_local_epochs),
        "--fraction-fit",
        str(args.ai_fraction_fit),
        "--fraction-evaluate",
        str(args.ai_fraction_evaluate),
        "--min-fit-clients",
        str(args.ai_min_fit_clients),
        "--min-evaluate-clients",
        str(args.ai_min_evaluate_clients),
        "--min-available-clients",
        str(args.ai_min_available_clients),
        "--weights-out-dir",
        fl_weights_out_dir,
        "--summary-out",
        summary_path,
        "--client-cpu",
        str(args.ai_client_cpu),
        "--chunksize",
        str(args.ai_chunksize),
    ]
    if args.ai_max_rooms is not None:
        sim_args.extend(["--max-rooms", str(args.ai_max_rooms)])
    _run_python_step("FL: Next-Hour Model", os.path.join("ai", "fl_simulation.py"), sim_args)
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_federated_targets_lstm(args, split_dir: str, fl_weights_out_dir: str) -> dict:
    summary_dir = os.path.join(fl_weights_out_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "next_hour_lstm_summary.json")
    sim_args = [
        "--split-dir",
        split_dir,
        "--rounds",
        str(args.ai_rounds),
        "--local-epochs",
        str(args.ai_local_epochs),
        "--sequence-length",
        str(args.ai_lstm_sequence_length),
        "--batch-size",
        str(args.ai_lstm_batch_size),
        "--fraction-fit",
        str(args.ai_fraction_fit),
        "--fraction-evaluate",
        str(args.ai_fraction_evaluate),
        "--min-fit-clients",
        str(args.ai_min_fit_clients),
        "--min-evaluate-clients",
        str(args.ai_min_evaluate_clients),
        "--min-available-clients",
        str(args.ai_min_available_clients),
        "--weights-out-dir",
        fl_weights_out_dir,
        "--summary-out",
        summary_path,
        "--client-cpu",
        str(args.ai_client_cpu),
        "--chunksize",
        str(args.ai_chunksize),
    ]
    if args.ai_max_rooms is not None:
        sim_args.extend(["--max-rooms", str(args.ai_max_rooms)])
    if args.ai_lstm_room_id is not None:
        sim_args.extend(["--room-id", str(args.ai_lstm_room_id)])
    _run_python_step("FL: Next-Hour LSTM Model", os.path.join("ai", "fl_simulation_lstm.py"), sim_args)
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_federated_targets_lstm_mlp(args, split_dir: str, fl_weights_out_dir: str) -> dict:
    summary_dir = os.path.join(fl_weights_out_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "next_hour_lstm_mlp_summary.json")
    sim_args = [
        "--split-dir",
        split_dir,
        "--rounds",
        str(args.ai_rounds),
        "--local-epochs",
        str(args.ai_local_epochs),
        "--sequence-length",
        str(args.ai_lstm_sequence_length),
        "--batch-size",
        str(args.ai_lstm_batch_size),
        "--fraction-fit",
        str(args.ai_fraction_fit),
        "--fraction-evaluate",
        str(args.ai_fraction_evaluate),
        "--min-fit-clients",
        str(args.ai_min_fit_clients),
        "--min-evaluate-clients",
        str(args.ai_min_evaluate_clients),
        "--min-available-clients",
        str(args.ai_min_available_clients),
        "--weights-out-dir",
        fl_weights_out_dir,
        "--summary-out",
        summary_path,
        "--client-cpu",
        str(args.ai_client_cpu),
        "--chunksize",
        str(args.ai_chunksize),
    ]
    if args.ai_max_rooms is not None:
        sim_args.extend(["--max-rooms", str(args.ai_max_rooms)])
    if args.ai_lstm_room_id is not None:
        sim_args.extend(["--room-id", str(args.ai_lstm_room_id)])
    _run_python_step("FL: Next-Hour LSTM+MLP Model", os.path.join("ai", "fl_simulation_lstm_mlp.py"), sim_args)
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)[0]


def _run_federated_targets_2(args, split_dir: str, fl_weights_out_dir: str) -> dict:
    summary_dir = os.path.join(fl_weights_out_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "next_hour_summary_2.json")
    sim_args = [
        "--split-dir",
        split_dir,
        "--rounds",
        str(args.ai_rounds),
        "--local-epochs",
        str(args.ai_local_epochs),
        "--batch-size",
        str(args.ai_2_batch_size),
        "--fraction-fit",
        str(args.ai_fraction_fit),
        "--fraction-evaluate",
        str(args.ai_fraction_evaluate),
        "--min-fit-clients",
        str(args.ai_min_fit_clients),
        "--min-evaluate-clients",
        str(args.ai_min_evaluate_clients),
        "--min-available-clients",
        str(args.ai_min_available_clients),
        "--weights-out-dir",
        fl_weights_out_dir,
        "--summary-out",
        summary_path,
        "--client-cpu",
        str(args.ai_client_cpu),
        "--chunksize",
        str(args.ai_chunksize),
    ]
    if args.ai_max_rooms is not None:
        sim_args.extend(["--max-rooms", str(args.ai_max_rooms)])
    _run_python_step("FL: Next-Hour Model 2", os.path.join("ai", "fl_simulation_2.py"), sim_args)
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_final_summary(summary: dict) -> None:
    print("\n[AI] Step 3/3: give summary", flush=True)
    print("[AI] Next-hour FL summary", flush=True)
    print(
        f"[AI] regression_correct={summary['regression_correct']} "
        f"regression_wrong={summary['regression_wrong']} "
        f"regression_correct_rate={summary['regression_correct_rate']:.4f} "
        f"evaluated_examples={summary['evaluated_examples']}",
        flush=True,
    )
    print(
        f"[AI] mae_temp_main={summary['mae_y_temp_main']:.4f} "
        f"mae_temp_toilet={summary['mae_y_temp_toilet']:.4f} "
        f"mae_light={summary['mae_y_light']:.4f} "
        f"mae_sound={summary['mae_y_sound']:.4f}",
        flush=True,
    )
    print(
        f"[AI] airflow_accuracy={summary['airflow_accuracy']:.4f} "
        f"airflow_precision={summary['airflow_precision']:.4f} "
        f"airflow_recall={summary['airflow_recall']:.4f} "
        f"airflow_f1={summary['airflow_f1']:.4f}",
        flush=True,
    )
    print(
        f"[AI] airflow_tp={summary['airflow_tp']} tn={summary['airflow_tn']} "
        f"fp={summary['airflow_fp']} fn={summary['airflow_fn']}",
        flush=True,
    )
    print(
        f"[AI] change_accuracy={summary['change_accuracy']:.4f} "
        f"change_precision={summary['change_precision']:.4f} "
        f"change_recall={summary['change_recall']:.4f} "
        f"change_f1={summary['change_f1']:.4f}",
        flush=True,
    )
    print(
        f"[AI] change_tp={summary['change_tp']} tn={summary['change_tn']} "
        f"fp={summary['change_fp']} fn={summary['change_fn']}",
        flush=True,
    )


def run_ai_build_split(args) -> None:
    rows_dir = os.path.abspath(args.ai_row_out_dir)
    split_dir = os.path.abspath(args.ai_split_dir)

    print("\n[AI] starting build/split pipeline", flush=True)
    build_args = [
        "--data-dir",
        args.ai_data_dir,
        "--out-dir",
        rows_dir,
        "--step-minutes",
        str(args.ai_step_minutes),
        "--horizon-minutes",
        str(args.ai_horizon_minutes),
    ]
    if args.ai_max_assignments is not None:
        build_args.extend(["--max-assignments", str(args.ai_max_assignments)])
    _run_python_step("Step 1/2: build rows", os.path.join("ai", "build_row.py"), build_args)

    split_args = [
        "--input-dir",
        rows_dir,
        "--output-dir",
        split_dir,
        "--train-ratio",
        str(args.ai_train_ratio),
        "--min-train-rows",
        str(args.ai_min_train_rows),
    ]
    _run_python_step("Step 2/2: split by y", os.path.join("ai", "split_by_y.py"), split_args)

    print("\n[AI] build/split pipeline complete", flush=True)
    print(f"[AI] rows_dir={rows_dir}", flush=True)
    print(f"[AI] split_dir={split_dir}", flush=True)


def run_ai_fl(args) -> None:
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)
    fl_weights_out_dir = os.path.abspath(args.ai_fl_weights_out_dir)

    print("\n[AI] starting FL pipeline from existing split data", flush=True)
    print("\n[AI] Step 1/2: run FL simulations", flush=True)
    summary = _run_federated_targets(args, split_dir, fl_weights_out_dir)
    print("\n[AI] Step 2/2: give summary", flush=True)
    _print_final_summary(summary)


def run_ai_fl_lstm(args) -> None:
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)
    fl_weights_out_dir = os.path.abspath(args.ai_fl_lstm_weights_out_dir)

    print("\n[AI] starting LSTM FL pipeline from existing split data", flush=True)
    print("\n[AI] Step 1/2: run FL LSTM simulations", flush=True)
    summary = _run_federated_targets_lstm(args, split_dir, fl_weights_out_dir)
    print("\n[AI] Step 2/2: give summary", flush=True)
    _print_final_summary(summary)


def run_ai_fl_lstm_mlp(args) -> None:
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)
    fl_weights_out_dir = os.path.abspath(args.ai_fl_lstm_mlp_weights_out_dir)

    print("\n[AI] starting LSTM+MLP FL pipeline from existing split data", flush=True)
    print("\n[AI] Step 1/2: run FL LSTM+MLP simulations", flush=True)
    summary = _run_federated_targets_lstm_mlp(args, split_dir, fl_weights_out_dir)
    print("\n[AI] Step 2/2: give summary", flush=True)
    _print_final_summary(summary)


def run_ai_fl_lstm_mlp_2(args) -> None:
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)
    fl_weights_out_dir = os.path.abspath(args.ai_fl_lstm_mlp_2_weights_out_dir)
    summary_dir = os.path.join(fl_weights_out_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "next_hour_lstm_mlp_2_summary.json")
    sim_args = [
        "--split-dir",
        split_dir,
        "--rounds",
        str(args.ai_rounds),
        "--local-epochs",
        str(args.ai_local_epochs),
        "--sequence-length",
        str(args.ai_lstm_sequence_length),
        "--batch-size",
        str(args.ai_lstm_batch_size),
        "--fraction-fit",
        str(args.ai_fraction_fit),
        "--fraction-evaluate",
        str(args.ai_fraction_evaluate),
        "--min-fit-clients",
        str(args.ai_min_fit_clients),
        "--min-evaluate-clients",
        str(args.ai_min_evaluate_clients),
        "--min-available-clients",
        str(args.ai_min_available_clients),
        "--weights-out-dir",
        fl_weights_out_dir,
        "--summary-out",
        summary_path,
        "--client-cpu",
        str(args.ai_client_cpu),
        "--chunksize",
        str(args.ai_chunksize),
    ]
    if args.ai_max_rooms is not None:
        sim_args.extend(["--max-rooms", str(args.ai_max_rooms)])
    if args.ai_lstm_room_id is not None:
        sim_args.extend(["--room-id", str(args.ai_lstm_room_id)])
    print("\n[AI] starting LSTM+MLP FL pipeline v2 from existing split data", flush=True)
    print("\n[AI] Step 1/2: run FL LSTM+MLP v2 simulations", flush=True)
    _run_python_step("FL: Next-Hour LSTM+MLP Model 2", os.path.join("ai", "fl_simulation_lstm_mlp_2.py"), sim_args)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)[0]
    print("\n[AI] Step 2/2: give summary", flush=True)
    _print_final_summary(summary)


def run_ai_fl_2(args) -> None:
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)
    fl_weights_out_dir = os.path.abspath(args.ai_fl_2_weights_out_dir)

    print("\n[AI] starting FL pipeline 2 from existing split data", flush=True)
    print("\n[AI] Step 1/2: run FL simulations", flush=True)
    summary = _run_federated_targets_2(args, split_dir, fl_weights_out_dir)
    print("\n[AI] Step 2/2: give summary", flush=True)
    _print_final_summary(summary)


def run_ai_pipeline(args) -> None:
    rows_dir = os.path.abspath(args.ai_next_hour_row_out_dir)
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)
    baseline_out_dir = os.path.abspath(args.ai_next_hour_out_dir)
    fl_weights_out_dir = os.path.abspath(args.ai_fl_weights_out_dir)

    print("\n[AI] starting training pipeline", flush=True)
    build_args = [
        "--data-dir",
        args.ai_data_dir,
        "--out-dir",
        rows_dir,
        "--step-minutes",
        str(args.ai_step_minutes),
        "--horizon-minutes",
        str(args.ai_next_hour_horizon_minutes),
    ]
    if args.ai_max_assignments is not None:
        build_args.extend(["--max-assignments", str(args.ai_max_assignments)])
    _run_python_step("Step 1/4: build next-hour rows", os.path.join("ai", "build_next_hour_rows.py"), build_args)

    split_args = [
        "--input-dir",
        rows_dir,
        "--output-dir",
        split_dir,
        "--train-ratio",
        str(args.ai_train_ratio),
        "--min-train-rows",
        str(args.ai_min_train_rows),
    ]
    _run_python_step("Step 2/4: split next-hour rows", os.path.join("ai", "split_next_hour_by_room.py"), split_args)

    baseline_args = [
        "--split-dir",
        split_dir,
        "--out-dir",
        baseline_out_dir,
        "--max-train",
        str(args.ai_max_train_b),
        "--max-test",
        str(args.ai_max_test_b),
    ]
    try:
        _run_python_step("Step 3/5: train next-hour baseline", os.path.join("ai", "train_baseline_next_hour.py"), baseline_args)
    except subprocess.CalledProcessError as exc:
        print(f"[AI] baseline step failed and will be skipped (exit_code={exc.returncode})", flush=True)
    print("\n[AI] Step 4/5: run FL simulations", flush=True)
    summary = _run_federated_targets(args, split_dir, fl_weights_out_dir)
    _print_final_summary(summary)


def run_ai_next_hour_build_split(args) -> None:
    rows_dir = os.path.abspath(args.ai_next_hour_row_out_dir)
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)

    print("\n[AI] starting next-hour build/split pipeline", flush=True)
    build_args = [
        "--data-dir",
        args.ai_data_dir,
        "--out-dir",
        rows_dir,
        "--step-minutes",
        str(args.ai_step_minutes),
        "--horizon-minutes",
        str(args.ai_next_hour_horizon_minutes),
    ]
    if args.ai_max_assignments is not None:
        build_args.extend(["--max-assignments", str(args.ai_max_assignments)])
    _run_python_step("Step 1/2: build next-hour rows", os.path.join("ai", "build_next_hour_rows.py"), build_args)

    split_args = [
        "--input-dir",
        rows_dir,
        "--output-dir",
        split_dir,
        "--train-ratio",
        str(args.ai_train_ratio),
        "--min-train-rows",
        str(args.ai_min_train_rows),
    ]
    _run_python_step("Step 2/2: split next-hour rows", os.path.join("ai", "split_next_hour_by_room.py"), split_args)


def run_ai_next_hour_build_split_2(args) -> None:
    rows_dir = os.path.abspath(args.ai_next_hour_row_out_dir)
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)

    print("\n[AI] starting next-hour build/split pipeline 2", flush=True)
    build_args = [
        "--data-dir",
        args.ai_data_dir,
        "--out-dir",
        rows_dir,
        "--horizon-minutes",
        str(args.ai_next_hour_horizon_minutes),
        "--before-minutes",
        str(args.ai_next_hour_change_before_minutes),
        "--after-minutes",
        str(args.ai_next_hour_change_after_minutes),
        "--sample-minutes",
        str(args.ai_next_hour_event_sample_minutes),
    ]
    if args.ai_max_assignments is not None:
        build_args.extend(["--max-assignments", str(args.ai_max_assignments)])
    _run_python_step("Step 1/2: build next-hour rows 2", os.path.join("ai", "build_next_hour_rows_2.py"), build_args)

    split_args = [
        "--input-dir",
        rows_dir,
        "--output-dir",
        split_dir,
        "--train-ratio",
        str(args.ai_train_ratio),
        "--min-train-rows",
        str(args.ai_min_train_rows),
    ]
    _run_python_step("Step 2/2: split next-hour rows", os.path.join("ai", "split_next_hour_by_room.py"), split_args)


def run_ai_next_hour_build_split_3(args) -> None:
    rows_dir = os.path.abspath(args.ai_next_hour_row_out_dir)
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)

    print("\n[AI] starting next-hour build/split pipeline 3", flush=True)
    build_args = [
        "--data-dir",
        args.ai_data_dir,
        "--out-dir",
        rows_dir,
        "--horizon-minutes",
        str(args.ai_next_hour_horizon_minutes),
        "--before-minutes",
        str(args.ai_next_hour_change_before_minutes),
        "--after-minutes",
        str(args.ai_next_hour_change_after_minutes),
        "--sample-minutes",
        str(args.ai_next_hour_change_sample_minutes),
    ]
    if args.ai_max_assignments is not None:
        build_args.extend(["--max-assignments", str(args.ai_max_assignments)])
    _run_python_step("Step 1/2: build next-hour rows 3", os.path.join("ai", "build_next_hour_rows_3.py"), build_args)

    split_args = [
        "--input-dir",
        rows_dir,
        "--output-dir",
        split_dir,
        "--train-ratio",
        str(args.ai_train_ratio),
        "--min-train-rows",
        str(args.ai_min_train_rows),
    ]
    _run_python_step("Step 2/2: split next-hour rows", os.path.join("ai", "split_next_hour_by_room.py"), split_args)


def run_ai_next_hour_local(args) -> None:
    split_dir = os.path.abspath(args.ai_next_hour_split_dir)
    out_dir = os.path.abspath(args.ai_next_hour_out_dir)
    print("\n[AI] starting next-hour local baseline", flush=True)
    baseline_args = [
        "--split-dir",
        split_dir,
        "--out-dir",
        out_dir,
        "--max-train",
        str(args.ai_max_train_b),
        "--max-test",
        str(args.ai_max_test_b),
    ]
    _run_python_step("Step 1/1: train next-hour baseline", os.path.join("ai", "train_baseline_next_hour.py"), baseline_args)


# ---------------- BLE DEVICE SETUP ----------------
def build_real_devices() -> list[Device]:
    devices = [
        # Device(mac_address="F8:CA:DA:A2:B6:AE", label="Room Sensor B"),
        # Device(mac_address="F8:CA:DA:A2:B6:AE", label="Room Sensor B"),
        # Device(mac_address="F8:CA:DA:A2:B6:AE", label="Room Sensor B"),
        Device(mac_address="FE:14:B2:D8:FD:AB", label="Room Sensor A"),
        # Device(mac_address="F8:CA:DA:A2:B6:AE", label="Room Sensor B"),
           Device(mac_address="D8:48:7F:68:79:D0", label="Room Sensor C"),
    ]

    for d in devices:
        d.add_sensor(Sensor(uuid=TEMP_CHAR_UUID, sensor_type="temperature", unit="°C", parser=parse_temp_thingy))
        # d.add_sensor(Sensor(uuid=HUMIDITY_CHAR_UUID, sensor_type="humidity", unit="%", parser=parse_humidity_thingy))
        # d.add_sensor(Sensor(uuid=AIR_QUALITY_CHAR_UUID, sensor_type="co2", unit="ppm", parser=parse_air_quality_thingy))
        d.add_sensor(Sensor(uuid=LIGHT_CHAR_UUID, sensor_type="light", unit="lux", parser=parse_light_thingy))
        # d.add_sensor(Sensor(uuid=SOUUND_CHAR_UUID, sensor_type="sound", unit="dB", parser=parse_sound_thingy))
        d.room_id = 1

    return devices


# ---------- modes ----------
async def run_sim(db_url: str, echo: bool, reset_db: bool):
    # Simulation mode: everything simulated
    db_path = os.getenv("FL4HOSPITAL_DB_PATH", "fl4hospital.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    init_db(db_url, echo=echo)

    collector = DataCollector(sinks=[db_sink])
    await collector.start()

    devices = seed_simulated_world(
        patient_count=PATIENT_COUNT,
        days=DAYS,
        start_date=START_DATE,
        seed=42,
    )
    seed_devices_and_sensors(devices)

    start = _to_utc_dt(START_DATE)
    end = start + timedelta(days=DAYS)

    sim = SimulationOrchestrator(
        start_time=start,
        end_time=end,
        on_event=collector.ingest,
        config=OrchestratorConfig(
            step_s=60,
            sample_every_s=300,   # simulated sensor sampling
            wall_sleep_s=0.0,
        ),
        seed=42,
    )

    await sim.start()
    await collector.stop()
    print("SIMULATION finished")


# ---------------- HYBRID MODE ----------------
async def run_real_hybrid(db_url: str, echo: bool, reset_db: bool):

    if reset_db and os.path.exists("fl4hospital.db"):
        os.remove("fl4hospital.db")

    init_db(db_url, echo=echo)

    # build BLE devices
    real_devices = build_real_devices()
    seed_devices_and_sensors(real_devices, room_id_default=1)

    # create room + patient + assignment
    ensure_room_and_patient(real_devices)

    # start BLE manager with controller adapter
    mgr = BLEManager(devices=real_devices, on_event=handle_ble_event)

    print("\nHYBRID MODE RUNNING")
    print("Press 'c' then ENTER to input comfort preference\n")

    try:
        await asyncio.gather(
            mgr.start(),
            event_worker(),
            run_cli(room_id=1, patient_id=1),
        )

    finally:
        await mgr.stop()

    print("HYBRID finished")


# ---------------- REAL ONLY MODE ----------------
async def run_real_real(db_url: str, echo: bool, reset_db: bool):

    if reset_db and os.path.exists("fl4hospital_real.db"):
        os.remove("fl4hospital_real.db")

    init_db(db_url, echo=echo)

    devices = build_real_devices()
    seed_devices_and_sensors(devices, room_id_default=1)

    mgr = BLEManager(devices=devices, on_event=lambda e: None)

    print("REAL SENSOR STREAM (no control)")
    await mgr.start()


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["simulation", "hybrid", "real_real", "ai", "ai_build_split", "ai_fl", "ai_fl_2", "ai_fl_lstm", "ai_fl_lstm_MLP", "ai_fl_lstm_MLP_2", "ai_next_hour_build_split", "ai_next_hour_build_split_2", "ai_next_hour_build_split_3", "ai_next_hour_local"],
        default="hybrid",
    )
    p.add_argument("--db", default="sqlite:///fl4hospital.db")
    p.add_argument("--echo", action="store_true")
    p.add_argument("--reset-db", action="store_true")
    p.add_argument("--ai-data-dir", default="filestorage")
    p.add_argument("--ai-row-out-dir", default="ai/outputs")
    p.add_argument("--ai-split-dir", default="ai/splits")
    p.add_argument("--ai-baseline-out-dir", default="ai/baseline_results")
    p.add_argument("--ai-fl-weights-out-dir", default="ai/fl_weights_sim")
    p.add_argument("--ai-fl-2-weights-out-dir", default="ai/fl_weights_sim_2")
    p.add_argument("--ai-fl-lstm-weights-out-dir", default="ai/fl_weights_sim_lstm")
    p.add_argument("--ai-fl-lstm-mlp-weights-out-dir", default="ai/fl_weights_sim_lstm_mlp")
    p.add_argument("--ai-fl-lstm-mlp-2-weights-out-dir", default="ai/fl_weights_sim_lstm_mlp_2")
    p.add_argument("--ai-next-hour-row-out-dir", default="ai/outputs_next_hour")
    p.add_argument("--ai-next-hour-split-dir", default="ai/splits_next_hour")
    p.add_argument("--ai-next-hour-row-out-dir-2", default="ai/outputs_next_hour_2")
    p.add_argument("--ai-next-hour-split-dir-2", default="ai/splits_next_hour_2")
    p.add_argument("--ai-next-hour-row-out-dir-3", default="ai/outputs_next_hour_3")
    p.add_argument("--ai-next-hour-split-dir-3", default="ai/splits_next_hour_3")
    p.add_argument("--ai-next-hour-out-dir", default="ai/baseline_results_next_hour")
    p.add_argument("--ai-step-minutes", type=int, default=30)
    p.add_argument("--ai-horizon-minutes", type=int, default=30)
    p.add_argument("--ai-next-hour-horizon-minutes", type=int, default=60)
    p.add_argument("--ai-next-hour-event-window-hours", type=int, default=1)
    p.add_argument("--ai-next-hour-event-sample-minutes", type=int, default=30)
    p.add_argument("--ai-next-hour-change-before-minutes", type=int, default=60)
    p.add_argument("--ai-next-hour-change-after-minutes", type=int, default=60)
    p.add_argument("--ai-next-hour-change-sample-minutes", type=int, default=30)
    p.add_argument("--ai-max-assignments", type=int, default=None)
    p.add_argument("--ai-train-ratio", type=float, default=0.8)
    p.add_argument("--ai-min-train-rows", type=int, default=1)
    p.add_argument("--ai-max-train-a", type=int, default=250000)
    p.add_argument("--ai-max-test-a", type=int, default=150000)
    p.add_argument("--ai-max-train-b", type=int, default=200000)
    p.add_argument("--ai-max-test-b", type=int, default=120000)
    p.add_argument("--ai-rounds", type=int, default=5)
    p.add_argument("--ai-n-features", type=int, default=256)
    p.add_argument("--ai-local-epochs", type=int, default=1)
    p.add_argument("--ai-fraction-fit", type=float, default=1.0)
    p.add_argument("--ai-fraction-evaluate", type=float, default=1.0)
    p.add_argument("--ai-min-fit-clients", type=int, default=2)
    p.add_argument("--ai-min-evaluate-clients", type=int, default=2)
    p.add_argument("--ai-min-available-clients", type=int, default=2)
    p.add_argument("--ai-max-rooms", type=int, default=None)
    p.add_argument("--ai-client-cpu", type=float, default=1.0)
    p.add_argument("--ai-chunksize", type=int, default=200000)
    p.add_argument("--ai-2-batch-size", type=int, default=32)
    p.add_argument("--ai-lstm-sequence-length", type=int, default=4)
    p.add_argument("--ai-lstm-batch-size", type=int, default=32)
    p.add_argument("--ai-lstm-room-id", default=None)
    return p.parse_args()


async def main():
    args = parse_args()

    if args.mode == "simulation":
        # reset _counters.json to 0
        with open("filestorage/_counters.json", "w") as f:
            counters = {
                "patients": 0,
                "rooms": 0,
                "admissions": 0,
                "room_assignments": 0,
                "medications": 0,
                "visits": 0,
                "comfort_preferences": 0,
                "utility_usages": 0,
                "toilet_lights": 0,
                "toilet_heaters": 0,
                "data": 0,
                "ventilations": 0
            }
            json.dump(counters, f)

        # delete all existing csv files in filestorage
        for filename in os.listdir("filestorage"):
            if filename.endswith(".csv"):
                os.remove(os.path.join("filestorage", filename))

        await run_sim(args.db, args.echo, args.reset_db)
    elif args.mode == "hybrid":
        await run_real_hybrid(args.db, args.echo, args.reset_db)
    elif args.mode == "real_real":
        await run_real_real(args.db, args.echo, args.reset_db)
    elif args.mode == "ai":
        await asyncio.to_thread(run_ai_pipeline, args)
    elif args.mode == "ai_build_split":
        await asyncio.to_thread(run_ai_build_split, args)
    elif args.mode == "ai_fl":
        await asyncio.to_thread(run_ai_fl, args)
    elif args.mode == "ai_fl_2":
        await asyncio.to_thread(run_ai_fl_2, args)
    elif args.mode == "ai_fl_lstm":
        await asyncio.to_thread(run_ai_fl_lstm, args)
    elif args.mode == "ai_fl_lstm_MLP":
        await asyncio.to_thread(run_ai_fl_lstm_mlp, args)
    elif args.mode == "ai_fl_lstm_MLP_2":
        await asyncio.to_thread(run_ai_fl_lstm_mlp_2, args)
    elif args.mode == "ai_next_hour_build_split":
        await asyncio.to_thread(run_ai_next_hour_build_split, args)
    elif args.mode == "ai_next_hour_build_split_2":
        await asyncio.to_thread(run_ai_next_hour_build_split_2, args)
    elif args.mode == "ai_next_hour_build_split_3":
        await asyncio.to_thread(run_ai_next_hour_build_split_3, args)
    elif args.mode == "ai_next_hour_local":
        await asyncio.to_thread(run_ai_next_hour_local, args)


if __name__ == "__main__":
    asyncio.run(main())
