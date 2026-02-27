import asyncio
import os
import argparse
import json
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
    p.add_argument("--mode", choices=["simulation", "hybrid", "real_real"], default="hybrid")
    p.add_argument("--db", default="sqlite:///fl4hospital.db")
    p.add_argument("--echo", action="store_true")
    p.add_argument("--reset-db", action="store_true")
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


if __name__ == "__main__":
    asyncio.run(main())
