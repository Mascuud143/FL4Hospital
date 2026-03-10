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


def run_ai_pipeline(args) -> None:
    rows_dir = os.path.abspath(args.ai_row_out_dir)
    split_dir = os.path.abspath(args.ai_split_dir)
    baseline_out_dir = os.path.abspath(args.ai_baseline_out_dir)
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
        str(args.ai_horizon_minutes),
    ]
    if args.ai_max_assignments is not None:
        build_args.extend(["--max-assignments", str(args.ai_max_assignments)])
    _run_python_step("Step 1/4: build rows", os.path.join("ai", "build_row.py"), build_args)

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
    _run_python_step("Step 2/4: split by room", os.path.join("ai", "split_by_room.py"), split_args)

    baseline_args = [
        "--split-dir",
        split_dir,
        "--out-dir",
        baseline_out_dir,
        "--max-train-a",
        str(args.ai_max_train_a),
        "--max-test-a",
        str(args.ai_max_test_a),
        "--max-train-b",
        str(args.ai_max_train_b),
        "--max-test-b",
        str(args.ai_max_test_b),
    ]
    _run_python_step("Step 3/4: train baseline", os.path.join("ai", "train_baseline.py"), baseline_args)

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
        "--client-cpu",
        str(args.ai_client_cpu),
        "--chunksize",
        str(args.ai_chunksize),
    ]
    if args.ai_max_rooms is not None:
        sim_args.extend(["--max-rooms", str(args.ai_max_rooms)])
    _run_python_step("Step 4/4: run FL simulation", os.path.join("ai", "fl_simulation.py"), sim_args)

    print("\n[AI] pipeline complete", flush=True)
    print(f"[AI] rows_dir={rows_dir}", flush=True)
    print(f"[AI] split_dir={split_dir}", flush=True)
    print(f"[AI] baseline_out_dir={baseline_out_dir}", flush=True)
    print(f"[AI] fl_weights_out_dir={fl_weights_out_dir}", flush=True)


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
    p.add_argument("--mode", choices=["simulation", "hybrid", "real_real", "ai"], default="hybrid")
    p.add_argument("--db", default="sqlite:///fl4hospital.db")
    p.add_argument("--echo", action="store_true")
    p.add_argument("--reset-db", action="store_true")
    p.add_argument("--ai-data-dir", default="filestorage")
    p.add_argument("--ai-row-out-dir", default="ai/outputs")
    p.add_argument("--ai-split-dir", default="ai/splits")
    p.add_argument("--ai-baseline-out-dir", default="ai/baseline_results")
    p.add_argument("--ai-fl-weights-out-dir", default="ai/fl_weights_sim")
    p.add_argument("--ai-step-minutes", type=int, default=30)
    p.add_argument("--ai-horizon-minutes", type=int, default=30)
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


if __name__ == "__main__":
    asyncio.run(main())
