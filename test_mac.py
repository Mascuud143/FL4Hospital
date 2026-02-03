import asyncio
from bleak import BleakScanner, BleakClient


TARGET_NAME = "Thingy"  


TEMP_CHAR_UUID = "ef680201-9b35-4933-9b10-52ffa9740042"
PRESSURE_CHAR_UUID = "ef680202-9b35-4933-9b10-52ffa9740042"
LIGHT_CHAR_UUID = "ef680205-9b35-4933-9b10-52ffa9740042"


def handle_temp(_, data):
    # data[0] = integer part (signed), data[1] = decimal part (1/100)
    temp = data[0] + data[1] / 100
    print(f"Temperature: {temp:.2f} °C")

def handle_pressure(_, data):
    # data[0-3] = pressure in Pa (little-endian, unsigned)
    pressure = int.from_bytes(data[0:4], "little", signed=False)
    print(f"Pressure: {pressure} Pa")

def handle_light(_, data):
    # data = light intensity in lx
    light = int.from_bytes(data, "little", signed=False)
    print(f"Light Intensity: {light} lx")


async def find_device_by_name(target_name):
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)

    for device in devices:
        print(f"Found: {device.name} ({device.address})")
        if device.name == target_name:
            print(f"Target device found: {device.name} ({device.address})")
            return device

    print(f"ERROR: Could not find device named {target_name}")
    return None

async def main():
    device = await find_device_by_name(TARGET_NAME)
    if device is None:
        return

    # Connect to the device
    async with BleakClient(device.address) as client:
        print(f"Connected: {client.is_connected}")

        # Start notifications
        await client.start_notify(TEMP_CHAR_UUID, handle_temp)
        await client.start_notify(PRESSURE_CHAR_UUID, handle_pressure)
        await client.start_notify(LIGHT_CHAR_UUID, handle_light)

        print("Listening for sensor updates for 20 seconds...")
        await asyncio.sleep(20)  # Keep the connection alive

        # Stop notifications
        await client.stop_notify(TEMP_CHAR_UUID)
        await client.stop_notify(PRESSURE_CHAR_UUID)
        await client.stop_notify(LIGHT_CHAR_UUID)

        print("Disconnected")


asyncio.run(main())
