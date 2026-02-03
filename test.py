import asyncio
from bleak import BleakClient

# ADDRESS = "F8:CA:DA:A2:B6:AE"
ADDRESS = "FE:14:B2:D8:FD:AB"



TEMP_CHAR_UUID = "ef680201-9b35-4933-9b10-52ffa9740042"
PRESSURE_CHAR_UUID = "ef680202-9b35-4933-9b10-52ffa9740042"
LIGHT_CHAR_UUID = "ef680205-9b35-4933-9b10-52ffa9740042"

def handle_temp(_, data):
    # data[0] = integer part (signed)
    # data[1] = decimal part (1/100)
    temp = data[0] + data[1] / 100
    print(f"Temperature: {temp:.2f} °C")

def handle_pressure(_, data):
    # data[0-3] = pressure in Pa (unsigned, little-endian)
    pressure = int.from_bytes(data[0:4], byteorder='little', signed=False)
    print(f"Pressure: {pressure} Pa")

async def main(address):
    async with BleakClient(address) as client:

        await client.start_notify(TEMP_CHAR_UUID, handle_temp)

        # subscribe to pressure notifications 
        await client.start_notify(PRESSURE_CHAR_UUID, handle_pressure)

        # light intensity notifications 
        await client.start_notify(LIGHT_CHAR_UUID, lambda _, data: print(f"Light Intensity: {int.from_bytes(data, byteorder='little')} lx"))

        print("Listening for temperature updates...")
        await asyncio.sleep(20)

        await client.stop_notify(TEMP_CHAR_UUID)

asyncio.run(main(ADDRESS))