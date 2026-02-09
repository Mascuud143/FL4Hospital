from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional
import struct


@dataclass(frozen=True)
class Sensor:
    """
    Represents one sensor/characteristic on a BLE device.
    """
    uuid: str
    sensor_type: str      # e.g. "temperature"
    unit: str             # e.g. "°C"
    parser: Callable[[bytes], Any]

    def parse(self, raw: bytes) -> Any:
        return self.parser(raw)


# -------------------------
# Parsers (examples)
# -------------------------

def parse_temp_thingy(raw: bytes) -> float:
    """
    Thingy:52 temperature: 2 bytes:
    - byte0: int8 (signed) whole degrees
    - byte1: uint8 fractional (0..99) -> /100
    """
    if len(raw) < 2:
        raise ValueError(f"Temperature raw too short: {len(raw)}")
    whole = struct.unpack("<b", raw[0:1])[0]
    frac = raw[1] / 100.0
    return float(whole + frac)


def parse_humidity_thingy(raw: bytes) -> float:
    """
    Thingy:52 humidity: 1 byte (0..100)
    """
    if len(raw) < 1:
        raise ValueError(f"Humidity raw too short: {len(raw)}")
    return float(raw[0])



def parse_air_quality_thingy(raw: bytes) -> int:
    """
    Thingy:52 Air Quality characteristic

    Format (little-endian):
      Bytes 0–1: eCO2 (uint16) in ppm
      Bytes 2–3: TVOC (uint16) in ppb (ignored)
    """
    import struct

    if len(raw) < 2:
        raise ValueError(f"Air quality raw too short: {len(raw)} bytes")

    (eco2,) = struct.unpack("<H", raw[:2])
    return int(eco2)




import math

DB_OFFSET = 75.0  # empirical calibration

def parse_sound_thingy(raw: bytes) -> float:
    """
    Convert Thingy:52 microphone audio frame to approximate dB (SPL-like).
    """

    if not raw:
        return 0.0

    # Center unsigned 8-bit samples
    samples = [(b - 128) for b in raw]

    # RMS amplitude
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))

    if rms <= 0:
        return 0.0

    # Digital full-scale reference
    A_ref = 128.0

    # Convert to dBFS
    dbfs = 20.0 * math.log10(rms / A_ref)

    # Convert to dB (approx SPL)
    db = dbfs + DB_OFFSET

    return float(db)




def parse_light_thingy(raw: bytes) -> float:
    import struct
    if len(raw) < 2:
        raise ValueError(f"Light raw too short: {len(raw)}")
    lux_raw = struct.unpack("<H", raw[:2])[0]
    return float(lux_raw)  # change to lux_raw / 100.0 if needed
