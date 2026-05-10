from __future__ import annotations

# Run sensor sampling.
# - Loads the sensor list - SensorSampler
# - Chooses serial or parallel work
# - Collects sensor rows
# - Closes worker processes

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from simulation_batch.simulation_steps.sensor_readings import chunk_sensor_specs, collect_rows_chunk
from simulation_batch.simulation_steps.sensor_catalog import SensorRuntime, load_sensor_registry


class SensorSampler:
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.sensors: list[SensorRuntime] = load_sensor_registry()
        self._sensor_specs = [(sensor.sensor_id, sensor.room_id, sensor.sensor_type) for sensor in self.sensors]
        self.room_ids = sorted({sensor.room_id for sensor in self.sensors if sensor.room_id is not None})
        self._emit_counter = 0
        self.workers = max(1, min((os.cpu_count() or 2) - 1, 14))
        self.parallel_threshold = 5000
        self._executor: ProcessPoolExecutor | None = None
        print(f"[SensorSampler] Loaded {len(self.sensors)} sensors")

    def collect_data_rows(self, now: datetime, *, room_engine) -> list[tuple[int, float, datetime]]:
        room_snapshot = self._build_room_snapshot(room_engine)
        self._emit_counter += 1
        emit_seed = self.seed + self._emit_counter
        if self._should_collect_serially():
            return self._collect_rows_serial(now=now, room_snapshot=room_snapshot, emit_seed=emit_seed)
        return self._collect_rows_parallel(now=now, room_snapshot=room_snapshot, emit_seed=emit_seed)

    def _build_room_snapshot(self, room_engine) -> dict[int, tuple[float, float, float, float, float]]:
        return {
            room_id: (
                float(room.temperature),
                float(room.humidity),
                float(room.co2),
                float(room.light),
                float(room.sound),
            )
            for room_id, room in room_engine.rooms.items()
        }

    def _should_collect_serially(self) -> bool:
        return self.workers <= 1 or len(self._sensor_specs) < self.parallel_threshold

    def _collect_rows_serial(
        self,
        *,
        now: datetime,
        room_snapshot: dict[int, tuple[float, float, float, float, float]],
        emit_seed: int,
    ) -> list[tuple[int, float, datetime]]:
        return collect_rows_chunk(self._sensor_specs, room_snapshot, now, emit_seed)

    def _collect_rows_parallel(
        self,
        *,
        now: datetime,
        room_snapshot: dict[int, tuple[float, float, float, float, float]],
        emit_seed: int,
    ) -> list[tuple[int, float, datetime]]:
        executor = self._ensure_executor()
        jobs = chunk_sensor_specs(self._sensor_specs, self._worker_chunk_size())
        rows: list[tuple[int, float, datetime]] = []
        for chunk_rows in executor.map(
            collect_rows_chunk,
            jobs,
            [room_snapshot] * len(jobs),
            [now] * len(jobs),
            [emit_seed + idx for idx in range(len(jobs))],
        ):
            rows.extend(chunk_rows)
        return rows

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers)
        return self._executor

    def _worker_chunk_size(self) -> int:
        return max(1, (len(self._sensor_specs) + self.workers - 1) // self.workers)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
