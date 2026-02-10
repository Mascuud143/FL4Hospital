import asyncio

from data_collection import DataCollector
from data_collection.db_sink import db_sink

from persistence import init_db
from persistence.seed_devices import seed_devices_and_sensors

from ble.characteristics import TEMP_CHAR_UUID, HUMIDITY_CHAR_UUID, AIR_QUALITY_CHAR_UUID, LIGHT_CHAR_UUID, SOUUND_CHAR_UUID
from ble.sensor import parse_sound_thingy, parse_light_thingy, parse_air_quality_thingy, parse_humidity_thingy, parse_temp_thingy

from ble import BLEManager, Device, Sensor
from ble.simulated_manager import SimulatedBLEManager   # ✅ FIX HERE
from simulation_batch.seed_world import seed_simulated_world

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
    return devices


async def print_sink(event: dict):
    print("EVENT:", event)


async def main():
    init_db(db_url="sqlite:///fl4hospital.db", echo=False)

    collector = DataCollector(sinks=[print_sink, db_sink])
    await collector.start()

    if MODE == "REAL":
        devices = build_real_devices()
        seed_devices_and_sensors(devices, room_id_default=1)
        mgr = BLEManager(devices, on_event=collector.ingest)

    else:
        devices = seed_simulated_world()
        seed_devices_and_sensors(devices, room_id_default=1)
        mgr = SimulatedBLEManager(devices, on_event=collector.ingest)

    await mgr.start()
    try:
        while True:
            await asyncio.sleep(5)
            print("Collector Stats:", collector.get_stats())
    finally:
        await mgr.stop()
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())
