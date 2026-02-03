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


def parse_pressure_thingy(raw: bytes) -> float:
    """
    Thingy:52 pressure: 4 bytes little-endian:
    - uint32 Pascal * 10? (varies by firmware)
    Many implementations interpret it as:
    pressure_hpa = (uint32 / 10.0)
    """
    if len(raw) < 4:
        raise ValueError(f"Pressure raw too short: {len(raw)}")
    v = struct.unpack("<I", raw[0:4])[0]
    return float(v) / 10.0


def parse_air_quality_thingy(raw: bytes) -> dict:
    """
    Thingy:52 air quality often has 2 bytes:
    - eCO2 (ppm) uint16
    - TVOC (ppb) uint16  (sometimes separate)
    Here we assume 4 bytes total.
    """
    if len(raw) < 4:
        raise ValueError(f"Air quality raw too short: {len(raw)}")
    eco2, tvoc = struct.unpack("<HH", raw[0:4])
    return {"eco2_ppm": int(eco2), "tvoc_ppb": int(tvoc)}
