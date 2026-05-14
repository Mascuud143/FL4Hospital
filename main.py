import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ble import BLEManager, Device, Sensor
from ble.characteristics import LIGHT_CHAR_UUID, TEMP_CHAR_UUID
from ble.sensor import parse_light_thingy, parse_temp_thingy
from hybrid_prototype.comfort_service import run_cli
from hybrid_prototype.event_adapter import event_worker, handle_ble_event
from hybrid_prototype.hybrid_context import ensure_room_and_patient
from persistence import init_db
from persistence.seed_devices import seed_devices_and_sensors
from simulation_batch.config import DAYS, PATIENT_COUNT, START_DATE
from simulation_batch.orchestrator import OrchestratorConfig, SimulationOrchestrator
from simulation_batch.setup_hospital import seed_simulated_world

DEFAULT_DB_URL = "sqlite:///fl4hospital.db"
DEFAULT_DB_PATH = Path("fl4hospital.db")
DEFAULT_REAL_DB_PATH = Path("fl4hospital_real.db")
FILESTORAGE_DIR = Path("filestorage")
REAL_ROOM_ID = 1
REAL_PATIENT_ID = 1

FILESTORAGE_COUNTERS = {
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
    "ventilations": 0,
}


def _to_utc_dt(value) -> datetime:
    return datetime.combine(value, datetime.min.time()).replace(tzinfo=timezone.utc)


def _env_path(name: str, fallback: Path) -> Path:
    return Path(os.getenv(name, str(fallback)))


def _remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _reset_database_file(path: Path, enabled: bool = True) -> None:
    if enabled:
        _remove_file_if_exists(path)


def _reset_filestorage() -> None:
    FILESTORAGE_DIR.mkdir(exist_ok=True)
    counters_path = FILESTORAGE_DIR / "_counters.json"
    with counters_path.open("w", encoding="utf-8") as handle:
        json.dump(FILESTORAGE_COUNTERS, handle)

    for csv_path in FILESTORAGE_DIR.glob("*.csv"):
        csv_path.unlink()


def build_real_devices() -> list[Device]:
    devices = [
        Device(mac_address="FE:14:B2:D8:FD:AB", label="Room Sensor A"),
        Device(mac_address="D8:48:7F:68:79:D0", label="Room Sensor C"),
    ]
    for device in devices:
        device.add_sensor(Sensor(uuid=TEMP_CHAR_UUID, sensor_type="temperature", unit="C", parser=parse_temp_thingy))
        device.add_sensor(Sensor(uuid=LIGHT_CHAR_UUID, sensor_type="light", unit="lux", parser=parse_light_thingy))
        device.room_id = REAL_ROOM_ID
    return devices


def _simulation_config(args: argparse.Namespace) -> OrchestratorConfig:
    return OrchestratorConfig(
        step_s=args.sim_step_s,
        sample_every_s=args.sensor_sample_every_s,
        wall_sleep_s=args.wall_sleep_s,
        comfort_max_changes_per_day=args.comfort_max_changes_per_day,
        enable_comfort=args.enable_comfort,
        enable_medication=args.enable_medication,
        enable_visits=args.enable_visits,
        enable_toilet_usage=args.enable_toilet_usage,
        enable_sensor_emit=args.enable_sensor_emit,
        enable_utility_usage=args.enable_utility_usage,
    )


def _build_real_ble_manager(devices: list[Device], on_event) -> BLEManager:
    return BLEManager(devices=devices, on_event=on_event)


def run_sim(args: argparse.Namespace) -> None:
    _reset_database_file(_env_path("FL4HOSPITAL_DB_PATH", DEFAULT_DB_PATH))
    init_db(args.db, echo=args.echo)

    devices = seed_simulated_world(
        patient_count=args.patient_count,
        days=args.days,
        start_date=args.start_date,
        change_room_prob=args.change_room_prob,
        min_days_before_transfer=args.min_days_before_transfer,
        min_days_after_transfer=args.min_days_after_transfer,
        seed=args.random_seed,
        create_devices=(
            args.enable_toilet_usage
            or args.enable_sensor_emit
            or args.enable_utility_usage
        ),
    )
    seed_devices_and_sensors(devices)

    start = _to_utc_dt(args.start_date)
    end = start + timedelta(days=args.days)
    sim = SimulationOrchestrator(
        start_time=start,
        end_time=end,
        on_event=lambda event: None,
        config=_simulation_config(args),
        seed=args.random_seed,
    )

    sim.start()
    print("SIMULATION finished")


async def run_real_hybrid(db_url: str, echo: bool, reset_db: bool) -> None:
    _reset_database_file(DEFAULT_DB_PATH, enabled=reset_db)
    init_db(db_url, echo=echo)
    devices = build_real_devices()
    seed_devices_and_sensors(devices, room_id_default=REAL_ROOM_ID)
    ensure_room_and_patient(devices)

    manager = _build_real_ble_manager(devices, handle_ble_event)
    print("\nHYBRID MODE RUNNING")
    print("Press 'c' then ENTER to input comfort preference\n")
    try:
        await asyncio.gather(
            manager.start(),
            event_worker(),
            run_cli(room_id=REAL_ROOM_ID, patient_id=REAL_PATIENT_ID),
        )
    finally:
        await manager.stop()
    print("HYBRID finished")


async def run_real_real(db_url: str, echo: bool, reset_db: bool) -> None:
    _reset_database_file(DEFAULT_REAL_DB_PATH, enabled=reset_db)
    init_db(db_url, echo=echo)
    devices = build_real_devices()
    seed_devices_and_sensors(devices, room_id_default=REAL_ROOM_ID)
    manager = _build_real_ble_manager(devices, lambda event: None)

    print("REAL SENSOR STREAM (no control)")
    await manager.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FL4Hospital runtime entry point.")
    parser.add_argument("--mode", choices=["simulation", "hybrid", "real_real"], default="simulation")
    parser.add_argument("--db", default=DEFAULT_DB_URL)
    parser.add_argument("--echo", action="store_true")
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--start-date", default=START_DATE.isoformat(), help="Simulation start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--patient-count", type=int, default=PATIENT_COUNT)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--change-room-prob", type=float, default=0.3)
    parser.add_argument("--min-days-before-transfer", type=int, default=1)
    parser.add_argument("--min-days-after-transfer", type=int, default=1)
    parser.add_argument("--comfort-max-changes-per-day", type=int, default=6)
    parser.add_argument("--sim-step-s", type=int, default=60)
    parser.add_argument("--sensor-sample-every-s", type=int, default=300)
    parser.add_argument("--wall-sleep-s", type=float, default=0.0)
    parser.add_argument("--enable-comfort", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-medication", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-visits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-toilet-usage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-sensor-emit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-utility-usage", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    args.start_date = datetime.fromisoformat(str(args.start_date)).date()
    return args


async def main() -> None:
    args = parse_args()

    if args.mode == "simulation":
        _reset_filestorage()
        run_sim(args)
    elif args.mode == "hybrid":
        await run_real_hybrid(args.db, args.echo, args.reset_db)
    elif args.mode == "real_real":
        await run_real_real(args.db, args.echo, args.reset_db)


if __name__ == "__main__":
    asyncio.run(main())
