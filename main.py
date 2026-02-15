# main.py
import asyncio
import os
import argparse
from datetime import datetime, timedelta, timezone

from data_collection import DataCollector
from data_collection.db_sink import db_sink

from persistence import init_db
from persistence.seed_devices import seed_devices_and_sensors

# ---- SIM imports ----
from simulation_batch.orchestrator import SimulationOrchestrator, OrchestratorConfig
from simulation_batch.seed_world import seed_simulated_world
from simulation_batch.config import START_DATE, DAYS, PATIENT_COUNT

# ---- REAL sensor imports (BLE) ----
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


# ---------- time helpers ----------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_dt(d) -> datetime:
    # Combine date -> midnight, UTC-aware
    return datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)


# ---------- REAL device definitions (shared by REAL + REAL_REAL) ----------
def build_real_devices() -> list[Device]:
    devices = [
        Device(mac_address="FE:14:B2:D8:FD:AB", label="Device 1"),
        Device(mac_address="F4:D4:A3:BE:01:9F", label="Device 2"),
        # Add more devices here...
    ]

    for d in devices:
        d.add_sensor(Sensor(uuid=TEMP_CHAR_UUID, sensor_type="temperature", unit="°C", parser=parse_temp_thingy))
        d.add_sensor(Sensor(uuid=HUMIDITY_CHAR_UUID, sensor_type="humidity", unit="%", parser=parse_humidity_thingy))

        d.add_sensor(Sensor(uuid=AIR_QUALITY_CHAR_UUID, sensor_type="co2", unit="ppm", parser=parse_air_quality_thingy))
        d.add_sensor(Sensor(uuid=LIGHT_CHAR_UUID, sensor_type="light", unit="lux", parser=parse_light_thingy))
        d.add_sensor(Sensor(uuid=SOUUND_CHAR_UUID, sensor_type="sound", unit="dB", parser=parse_sound_thingy))

        # Attach room mapping for seeding / downstream joins
        d.room_id = 1

    return devices


# ---------- modes ----------
async def run_sim(db_url: str, echo: bool, reset_db: bool):
    # Simulation mode: everything simulated
    if reset_db and os.path.exists("fl4hospital_sim.db"):
        os.remove("fl4hospital_sim.db")

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


async def run_real_hybrid(db_url: str, echo: bool, reset_db: bool, runtime_s: int | None):
    """
    REAL (hybrid) mode:
      - Sensors are REAL (BLE)
      - Everything else (comfort / room dynamics / utility sessions) is SIMULATED

    Practical wiring:
      - Start BLEManager and ingest BLE events into the collector
      - Run SimulationOrchestrator for the rest of the sim

    IMPORTANT:
      - You usually want to DISABLE simulated sensor sampling so you don't double-write sensors.
        Here we do that by setting sample_every_s=0.
      - If your orchestrator *requires* sensor sampling for some internal logic,
        you'll need a small orchestrator change to "sample from BLE" instead of "simulate".
    """
    if reset_db and os.path.exists("fl4hospital_testing.db"):
        os.remove("fl4hospital_testing.db")

    init_db(db_url, echo=echo)

    # real sensor devices (BLE)
    real_devices = build_real_devices()
    seed_devices_and_sensors(real_devices, room_id_default=1)

    collector = DataCollector(sinks=[db_sink])
    await collector.start()

    # Start BLE (real sensors)
    mgr = BLEManager(devices=real_devices, on_event=collector.ingest)

    # Start simulation world (patients/rooms/etc.)
    sim_devices = seed_simulated_world(
        patient_count=PATIENT_COUNT,
        days=DAYS,
        start_date=START_DATE,
        seed=42,
    )
    seed_devices_and_sensors(sim_devices)

    start = _to_utc_dt(START_DATE)
    end = start + timedelta(days=DAYS)

    sim = SimulationOrchestrator(
        start_time=start,
        end_time=end,
        on_event=collector.ingest,
        config=OrchestratorConfig(
            step_s=60,
            sample_every_s=0,     # disable simulated sensors to avoid duplicates
            wall_sleep_s=0.0,
        ),
        seed=42,
    )

    try:
        await mgr.start()

        if runtime_s is None:
            # Run the full sim (start->end)
            await sim.start()
        else:
            # Run sim + BLE together for a limited wall-clock duration
            sim_task = asyncio.create_task(sim.start())
            await asyncio.sleep(runtime_s)
            # If your sim supports a graceful stop, call it here.
            # Otherwise we just cancel the task:
            sim_task.cancel()
            try:
                await sim_task
            except asyncio.CancelledError:
                pass

    finally:
        await mgr.stop()
        await collector.stop()

    print("REAL (hybrid) finished")


async def run_real_real(db_url: str, echo: bool, reset_db: bool, runtime_s: int | None):
    """
    REAL REAL mode:
      - Everything is done by real devices
      - Here we only show real SENSOR ingestion via BLE.
      - If you also have real actuator states (HVAC/airflow/light/heater/water),
        you'll plug those sources in here too (and build sessions from real transitions).
    """
    if reset_db and os.path.exists("fl4hospital_real.db"):
        os.remove("fl4hospital_real.db")

    init_db(db_url, echo=echo)

    devices = build_real_devices()
    seed_devices_and_sensors(devices, room_id_default=1)

    collector = DataCollector(sinks=[db_sink])
    await collector.start()

    mgr = BLEManager(devices=devices, on_event=collector.ingest)

    try:
        await mgr.start()

        if runtime_s is None:
            while True:
                await asyncio.sleep(1)
        else:
            await asyncio.sleep(runtime_s)

    finally:
        await mgr.stop()
        await collector.stop()

    print("REAL REAL finished")


# ---------- cli ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["simulation", "real", "real_real"],
        default=os.getenv("MODE", "simulation").lower(),
    )
    p.add_argument("--db", default=os.getenv("DB_URL", "sqlite:///fl4hospital.db"))
    p.add_argument("--echo", action="store_true")
    p.add_argument("--reset-db", action="store_true")
    p.add_argument(
        "--runtime-s",
        type=int,
        default=None,
        help="Wall-clock runtime limit for real/real_real (and optional for real hybrid).",
    )
    return p.parse_args()


async def main():
    if os.path.exists("fl4hospital.db"): os.remove("fl4hospital.db")
    args = parse_args()

    # All timestamps should be treated as UTC across the system. :contentReference[oaicite:0]{index=0}

    if args.mode == "simulation":
        await run_sim(db_url=args.db, echo=args.echo, reset_db=args.reset_db)
    elif args.mode == "real":
        await run_real_hybrid(db_url=args.db, echo=args.echo, reset_db=args.reset_db, runtime_s=args.runtime_s)
    else:
        await run_real_real(db_url=args.db, echo=args.echo, reset_db=args.reset_db, runtime_s=args.runtime_s)


if __name__ == "__main__":
    asyncio.run(main())
