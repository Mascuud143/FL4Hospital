from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List


def _chunked(items: list[tuple[int, int, str]], size: int) -> list[list[tuple[int, int, str]]]:
    return [items[idx:idx + size] for idx in range(0, len(items), size)]


def _sample_value(rng: random.Random, sensor_type: str, room_state: tuple[float, float, float, float, float]) -> float:
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


def _collect_rows_chunk(
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
        rows.append((sensor_id, _sample_value(rng, sensor_type, room_state), now))
    return rows


class SensorRuntime:
    """
    Plain runtime sensor object (NO ORM).
    """

    def __init__(
        self,
        *,
        sensor_id: int,
        device_id: int,
        room_id: int,
        mac: str,
        location: str,
        sensor_type: str,
        uuid: str,
    ):
        self.sensor_id = sensor_id
        self.device_id = device_id
        self.room_id = room_id
        self.mac = mac
        self.location = location
        self.sensor_type = sensor_type
        self.uuid = uuid


class SensorSampler:
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.sensors: List[SensorRuntime] = []
        self._sensor_specs: list[tuple[int, int, str]] = []
        self.room_ids: list[int] = []
        self._emit_counter = 0
        self.workers = max(1, min((os.cpu_count() or 2) - 1, 8))
        self.parallel_threshold = 5000
        self._executor: ProcessPoolExecutor | None = None

        # Load everything once, flattening relationships.
        from persistence.database import session_scope
        from persistence.models.device import Device as DeviceModel
        from persistence.models.sensor import Sensor as SensorModel

        with session_scope() as session:
            rows = (
                session.query(
                    SensorModel.sensor_id,
                    SensorModel.sensor_type,
                    SensorModel.uuid,
                    DeviceModel.device_id,
                    DeviceModel.room_id,
                    DeviceModel.mac_address,
                    DeviceModel.location,
                )
                .join(DeviceModel, SensorModel.device_id == DeviceModel.device_id)
                .all()
            )

            for row in rows:
                runtime_sensor = SensorRuntime(
                    sensor_id=row.sensor_id,
                    device_id=row.device_id,
                    room_id=row.room_id,
                    mac=row.mac_address,
                    location=row.location,
                    sensor_type=row.sensor_type,
                    uuid=row.uuid,
                )
                self.sensors.append(runtime_sensor)
                self._sensor_specs.append((runtime_sensor.sensor_id, runtime_sensor.room_id, runtime_sensor.sensor_type))

        self.room_ids = sorted({sensor.room_id for sensor in self.sensors if sensor.room_id is not None})

        print(f"[SensorSampler] Loaded {len(self.sensors)} sensors")

    def collect_data_rows(
        self,
        now: datetime,
        *,
        room_engine,
    ) -> list[tuple[int, float, datetime]]:
        room_snapshot = {
            room_id: (
                float(room.temperature),
                float(room.humidity),
                float(room.co2),
                float(room.light),
                float(room.sound),
            )
            for room_id, room in room_engine.rooms.items()
        }
        self._emit_counter += 1
        if self.workers <= 1 or len(self._sensor_specs) < self.parallel_threshold:
            return _collect_rows_chunk(self._sensor_specs, room_snapshot, now, self.seed + self._emit_counter)

        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers)

        worker_chunk_size = max(1, (len(self._sensor_specs) + self.workers - 1) // self.workers)
        jobs = _chunked(self._sensor_specs, worker_chunk_size)
        rows: list[tuple[int, float, datetime]] = []
        for chunk_rows in self._executor.map(
            _collect_rows_chunk,
            jobs,
            [room_snapshot] * len(jobs),
            [now] * len(jobs),
            [self.seed + self._emit_counter + idx for idx in range(len(jobs))],
        ):
            rows.extend(chunk_rows)
        return rows

    async def emit(
        self,
        now: datetime,
        *,
        room_engine,
        on_event,
    ) -> None:
        for sensor in self.sensors:
            room = room_engine.rooms.get(sensor.room_id)
            if not room:
                continue

            value = _sample_value(
                random.Random(self.seed + self._emit_counter),
                sensor.sensor_type,
                (
                    float(room.temperature),
                    float(room.humidity),
                    float(room.co2),
                    float(room.light),
                    float(room.sound),
                ),
            )

            event = {
                "timestamp": now,
                "sensor_id": sensor.sensor_id,
                "room_id": sensor.room_id,
                "device_id": sensor.device_id,
                "mac": sensor.mac,
                "device_type": "sensor",
                "location": sensor.location,
                "sensor_type": sensor.sensor_type,
                "uuid": sensor.uuid,
                "unit": self._unit(sensor.sensor_type),
                "value": value,
            }

            await on_event(event)

    def _unit(self, sensor_type: str) -> str:
        return {
            "temperature": "degC",
            "humidity": "%",
            "co2": "ppm",
            "light": "lux",
            "sound": "dB",
        }.get(sensor_type, "")

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
