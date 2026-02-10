import asyncio
from datetime import datetime, timedelta, timezone

from data_collection import DataCollector
from data_collection.db_sink import db_sink

from persistence import init_db
from persistence.seed_devices import seed_devices_and_sensors

from ble.characteristics import (
    TEMP_CHAR_UUID,
    HUMIDITY_CHAR_UUID,
    AIR_QUALITY_CHAR_UUID,
    LIGHT_CHAR_UUID,
    SOUUND_CHAR_UUID,
)
from ble.sensor import (
    parse_sound_thingy,
    parse_light_thingy,
    parse_air_quality_thingy,
    parse_humidity_thingy,
    parse_temp_thingy,
)

from ble import BLEManager, Device, Sensor

# ✅ New simulation runner (matches your diagram)
from simulation_batch.orchestrator import SimulationOrchestrator, OrchestratorConfig
from simulation_batch.seed_world import seed_simulated_world
from simulation_batch.config import START_DATE, DAYS, PATIENT_COUNT  # room count removed


MODE = "SIMULATED"  # "REAL" or "SIMULATED"


def build_real_devices():
    devices = [
        Device(mac_address="FE:14:B2:D8:FD:AB", label="Device 1"),
        Device(mac_address="F4:D4:A3:BE:01:9F", label="Device 2"),
        Device(mac_address="F8:CA:DA:A2:B6:AE", label="Device 3"),
        Device(mac_address="D8:48:7F:68:79:D0", label="Device 4"),
    ]
    for d in devices:
        d.add_sensor(Sensor(uuid=TEMP_CHAR_UUID, sensor_type="temperature", unit="°C", parser=parse_temp_thingy))
        d.add_sensor(Sensor(uuid=HUMIDITY_CHAR_UUID, sensor_type="humidity", unit="%", parser=parse_humidity_thingy))
        d.add_sensor(Sensor(uuid=AIR_QUALITY_CHAR_UUID, sensor_type="co2", unit="ppm", parser=parse_air_quality_thingy))
        d.add_sensor(Sensor(uuid=LIGHT_CHAR_UUID, sensor_type="light", unit="lux", parser=parse_light_thingy))
        d.add_sensor(Sensor(uuid=SOUUND_CHAR_UUID, sensor_type="sound", unit="dB", parser=parse_sound_thingy))
        d.room_id = 1
        d.device_type = "sensor"
        d.location = "main"
    return devices


async def print_sink(event: dict):
    print("EVENT:", event)


def _to_utc_dt(d) -> datetime:
    """Convert date/datetime from config into timezone-aware UTC datetime at midnight."""
    if isinstance(d, datetime):
        dt = d
    else:
        dt = datetime.combine(d, datetime.min.time())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def main():
    init_db(db_url="sqlite:///fl4hospital.db", echo=False)

    collector = DataCollector(sinks=[print_sink, db_sink])
    await collector.start()

    if MODE == "REAL":
        devices = build_real_devices()
        seed_devices_and_sensors(devices, room_id_default=1)

        mgr = BLEManager(devices, on_event=collector.ingest)
        await mgr.start()

        try:
            while True:
                await asyncio.sleep(5)
                print("Collector Stats:", collector.get_stats())
        finally:
            await mgr.stop()
            await collector.stop()

    else:
        # -------------------------
        # 1) Seed rooms/patients/assignments + runtime devices
        # -------------------------
        devices = seed_simulated_world(
            patient_count=PATIENT_COUNT,
            days=DAYS,
            start_date=START_DATE,
            seed=42,
        )

        # -------------------------
        # 2) Seed devices/sensors into DB so db_sink resolves them
        # -------------------------
        seed_devices_and_sensors(devices, room_id_default=1)

        # -------------------------
        # 3) Run orchestrated simulation (fast, bounded)
        # -------------------------
        start_time = _to_utc_dt(START_DATE)
        end_time = start_time + timedelta(days=int(DAYS))

        sim = SimulationOrchestrator(
            start_time=start_time,
            end_time=end_time,
            on_event=collector.ingest,
            config=OrchestratorConfig(
                step_s=60,                   # 1 simulated minute per tick
                sample_every_s=300,          # sample sensors every 5 simulated minutes
                wall_sleep_s=0.0,            # run as fast as possible
                comfort_max_changes_per_day=3,
            ),
            seed=42,
        )

        await sim.start()

        try:
            # Wait until simulation naturally ends
            # (Orchestrator stops when it reaches end_time)
            while True:
                await asyncio.sleep(1)
                # Optional periodic stats while running
                # print("Collector Stats:", collector.get_stats())

                # if the background task has finished, break
                task = getattr(sim, "_task", None)
                if task is not None and task.done():
                    break
        finally:
            await sim.stop()
            await collector.stop()

        print("✅ Simulation finished.")
        print("Collector Stats:", collector.get_stats())


if __name__ == "__main__":
    asyncio.run(main())
