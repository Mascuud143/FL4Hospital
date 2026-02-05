import asyncio

from ble import BLEManager, Device, Sensor
from ble.characteristics import TEMP_CHAR_UUID, HUMIDITY_CHAR_UUID
from ble.sensor import parse_temp_thingy, parse_humidity_thingy

from data_collection import DataCollector
from data_collection.db_sink import db_sink

from persistence import init_db
from simulation.simulation_runner import run_simulation


# =============================
# CONFIG
# =============================
MODE = "SIMULATION"   # "REAL" or "SIMULATION"


# ---- Debug sink ----
async def print_sink(event: dict):
    print("CLEAN EVENT:", event)


# ---- BLE Devices (REAL MODE) ----
devices = [
    Device(mac_address="FE:14:B2:D8:FD:AB", label="Device 2"),
]

for d in devices:
    d.add_sensor(Sensor(
        uuid=TEMP_CHAR_UUID,
        sensor_type="temperature",
        unit="°C",
        parser=parse_temp_thingy
    ))
    d.add_sensor(Sensor(
        uuid=HUMIDITY_CHAR_UUID,
        sensor_type="humidity",
        unit="%",
        parser=parse_humidity_thingy
    ))


async def main():
    # 0) Init DB
    init_db(db_url="sqlite:///fl4hospital.db", echo=False)

    # 1) Start DataCollector
    collector = DataCollector(
        sinks=[
            print_sink,  # optional
            db_sink      # REQUIRED for saving
        ]
    )
    await collector.start()

    # 2) Choose data source
    if MODE == "REAL":
        print("🔵 Running in REAL (BLE) mode")
        mgr = BLEManager(devices, on_event=collector.ingest)
        await mgr.start()

        try:
            while True:
                await asyncio.sleep(5)
                print("Collector Stats:", collector.get_stats())
        finally:
            await mgr.stop()

    else:
        print("🟣 Running in SIMULATION mode")
        await run_simulation(collector)


if __name__ == "__main__":
    asyncio.run(main())
