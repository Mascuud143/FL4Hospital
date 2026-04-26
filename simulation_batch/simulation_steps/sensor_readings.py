from __future__ import annotations

# Build sensor readings.
# - Splits sensor work into chunks - chunk_sensor_specs()
# - Builds one sensor value - sample_value()
# - Builds sensor output rows - collect_rows_chunk()
# - Picks sensor units - sensor_unit()

import random
from datetime import datetime


def chunk_sensor_specs(items: list[tuple[int, int, str]], size: int) -> list[list[tuple[int, int, str]]]:
    return [items[idx:idx + size] for idx in range(0, len(items), size)]


def sample_value(rng: random.Random, sensor_type: str, room_state: tuple[float, float, float, float, float]) -> float:
    temperature, humidity, co2, light, sound = room_state
    if sensor_type == "temperature":
        return temperature + rng.gauss(0, 0.05)
    if sensor_type == "humidity":
        return humidity + rng.gauss(0, 0.2)
    if sensor_type == "co2":
        return max(400.0, co2 + rng.gauss(0, 20))
    if sensor_type == "light":
        return max(0.0, light + rng.gauss(0, 2))
    if sensor_type == "sound":
        return max(0.0, sound + rng.gauss(0, 1))
    return 0.0


def collect_rows_chunk(
    chunk: list[tuple[int, int, str]],
    room_snapshot: dict[int, tuple[float, float, float, float, float]],
    now: datetime,
    seed: int,
) -> list[tuple[int, float, datetime]]:
    rng = random.Random(seed)
    rows: list[tuple[int, float, datetime]] = []
    for sensor_id, room_id, sensor_type in chunk:
        room_state = room_snapshot.get(room_id)
        if room_state is None:
            continue
        rows.append((sensor_id, sample_value(rng, sensor_type, room_state), now))
    return rows


def sensor_unit(sensor_type: str) -> str:
    return {
        "temperature": "degC",
        "humidity": "%",
        "co2": "ppm",
        "light": "lux",
        "sound": "dB",
    }.get(sensor_type, "")
