import asyncio

from ble import BLEManager, Device, Sensor
from ble.characteristics import TEMP_CHAR_UUID, HUMIDITY_CHAR_UUID
from ble.sensor import parse_temp_thingy, parse_humidity_thingy

from data_collection import DataCollector
from data_collection.db_sink import db_sink

from persistence import init_db
from persistence.seed_devices import seed_devices_and_sensors  


MODE = "REAL"


async def print_sink(event: dict):
    print("CLEAN EVENT:", event)


devices = [
    Device(mac_address="FE:14:B2:D8:FD:AB", label="Device 1"),
    Device(mac_address="F4:D4:A3:BE:01:9F", label="Device 2"),
    Device(mac_address="F8:CA:DA:A2:B6:AE", label="Device 3"),
    Device(mac_address="D8:48:7F:68:79:D0", label="Device 4"),
]

for d in devices:
    d.add_sensor(Sensor(uuid=TEMP_CHAR_UUID, sensor_type="temperature", unit="°C", parser=parse_temp_thingy))
    d.add_sensor(Sensor(uuid=HUMIDITY_CHAR_UUID, sensor_type="humidity", unit="%", parser=parse_humidity_thingy))


async def main():
    init_db(db_url="sqlite:///fl4hospital.db", echo=False)

    #Seed the known devices + their sensors right away
    seed_devices_and_sensors(devices)

    collector = DataCollector(sinks=[print_sink, db_sink])
    await collector.start()

    if MODE == "REAL":
        mgr = BLEManager(devices, on_event=collector.ingest)
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
