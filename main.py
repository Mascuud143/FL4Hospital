import asyncio
from ble import BLEManager, Device, Sensor
from ble.characteristics import TEMP_CHAR_UUID, HUMIDITY_CHAR_UUID
from ble.sensor import parse_temp_thingy, parse_humidity_thingy


async def on_event(event: dict):
    # Later: forward to DataCollector / DB / CSV
    print(event)


devices = [
    Device(mac_address="F8:CA:DA:A2:B6:AE", label="Device 1"),
    Device(mac_address="FE:14:B2:D8:FD:AB", label="Device 2"),
    Device(mac_address="D8:48:7F:68:79:D0", label="Device 3"),
]

for d in devices:
    d.add_sensor(Sensor(uuid=TEMP_CHAR_UUID, sensor_type="temperature", unit="°C", parser=parse_temp_thingy))
    d.add_sensor(Sensor(uuid=HUMIDITY_CHAR_UUID, sensor_type="humidity", unit="%", parser=parse_humidity_thingy))


async def main():
    mgr = BLEManager(devices, on_event=on_event)
    await mgr.start()

    # Run forever (or until Ctrl+C)
    try:
        while True:
            await asyncio.sleep(2)
    finally:
        await mgr.stop()


asyncio.run(main())
