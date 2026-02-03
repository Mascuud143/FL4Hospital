# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:38:39 2026

@author: tawit
"""

import asyncio
import csv
from datetime import datetime

import nest_asyncio
from bleak import BleakClient

nest_asyncio.apply()

# ---------------------------
# CONFIG
# ---------------------------
ADDRESS = "F8:CA:DA:A2:B6:AE"     # <-- your Thingy address
CSV_FILE = "thingy_log.csv"

# Thingy base UUID: EF68xxxx-9B35-4933-9B10-52FFA9740042  :contentReference[oaicite:2]{index=2}

# Weather (Environment) service (0x0200) :contentReference[oaicite:3]{index=3}
TEMP_CHAR_UUID = "ef680201-9b35-4933-9b10-52ffa9740042"  # Notify, 2 bytes: int8 + uint8 :contentReference[oaicite:4]{index=4}
HUM_CHAR_UUID  = "ef680203-9b35-4933-9b10-52ffa9740042"  # Notify, 1 byte RH% :contentReference[oaicite:5]{index=5}
GAS_CHAR_UUID  = "ef680204-9b35-4933-9b10-52ffa9740042"  # Notify, 4 bytes: eCO2 + TVOC :contentReference[oaicite:6]{index=6}

# Sound service (0x0500) :contentReference[oaicite:7]{index=7}
SOUND_CFG_UUID = "ef680501-9b35-4933-9b10-52ffa9740042"  # Write/Read 2 bytes config :contentReference[oaicite:8]{index=8}
MIC_CHAR_UUID  = "ef680504-9b35-4933-9b10-52ffa9740042"  # Notify microphone frames :contentReference[oaicite:9]{index=9}

# ---------------------------
# Parsers
# ---------------------------
def parse_temp_c(payload: bytes) -> float:
    # 2 bytes: int8 integer, uint8 decimal (hundredths) :contentReference[oaicite:10]{index=10}
    if len(payload) < 2:
        return float("nan")
    integer = int.from_bytes(payload[0:1], "little", signed=True)
    decimal = payload[1]
    return integer + decimal / 100.0

def parse_humidity(payload: bytes) -> float:
    # 1 byte: RH% :contentReference[oaicite:11]{index=11}
    if len(payload) < 1:
        return float("nan")
    return float(payload[0])

def parse_gas(payload: bytes):
    # 4 bytes: uint16 eCO2 ppm, uint16 TVOC ppb :contentReference[oaicite:12]{index=12}
    if len(payload) < 4:
        return (float("nan"), float("nan"))
    eco2 = int.from_bytes(payload[0:2], "little", signed=False)
    tvoc = int.from_bytes(payload[2:4], "little", signed=False)
    return (eco2, tvoc)

def sound_proxy_from_adpcm(payload: bytes) -> float:
    """
    Thingy mic notifies ADPCM frames by default. SPL mode isn't implemented. :contentReference[oaicite:13]{index=13}
    This returns a simple 'activity/level' proxy from the raw frame bytes.
    Not calibrated, but useful to see 'louder vs quieter' changes.
    """
    if not payload:
        return 0.0
    # center around 128 and take mean absolute deviation
    return sum(abs(b - 128) for b in payload) / len(payload)

# ---------------------------
async def main():
    latest = {
        "temp_c": None,
        "hum_pct": None,
        "eco2_ppm": None,
        "sound_level": None,
    }

    def maybe_write_row(writer, file_handle):
        # Write a row whenever we have at least one set (you can tighten this if you want)
        ts = datetime.now().isoformat(timespec="seconds")
        writer.writerow([
            ts,
            latest["temp_c"],
            latest["hum_pct"],
            latest["eco2_ppm"],
            latest["sound_level"],
        ])
        file_handle.flush()
        print(ts, "T=", latest["temp_c"], "RH=", latest["hum_pct"], "eCO2=", latest["eco2_ppm"], "snd=", latest["sound_level"])

    async with BleakClient(ADDRESS) as client:
        print("Connected:", client.is_connected)

        # Configure sound: mic mode = ADPCM (0x01). Speaker mode can be 0x01 (doesn't matter if you don't use speaker).
        # Config format: [speaker_mode, mic_mode] :contentReference[oaicite:14]{index=14}
        await client.write_gatt_char(SOUND_CFG_UUID, bytes([0x01, 0x01]), response=True)

        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "temp_C", "humidity_pct", "eco2_ppm", "sound_proxy"])

            # Notification callbacks
            def on_temp(_, data: bytearray):
                latest["temp_c"] = round(parse_temp_c(data), 2)
                maybe_write_row(writer, f)

            def on_hum(_, data: bytearray):
                latest["hum_pct"] = round(parse_humidity(data), 2)
                maybe_write_row(writer, f)

            def on_gas(_, data: bytearray):
                eco2, tvoc = parse_gas(data)
                latest["eco2_ppm"] = int(eco2) if eco2 == eco2 else None
                # If you also want TVOC, add a column and store it here.
                maybe_write_row(writer, f)

            def on_mic(_, data: bytearray):
                latest["sound_level"] = round(sound_proxy_from_adpcm(data), 2)
                maybe_write_row(writer, f)

            # Subscribe (these are Notify characteristics) :contentReference[oaicite:15]{index=15}
            await client.start_notify(TEMP_CHAR_UUID, on_temp)
            await client.start_notify(HUM_CHAR_UUID, on_hum)
            await client.start_notify(GAS_CHAR_UUID, on_gas)
            await client.start_notify(MIC_CHAR_UUID, on_mic)

            print("Subscribed. Logging to:", CSV_FILE)
            print("Stop: in Spyder press the red Stop button.")

            while True:
                await asyncio.sleep(1)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
