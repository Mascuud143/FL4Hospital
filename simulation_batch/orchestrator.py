from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

from .room_engine import RoomEngine, RoomState
from .sensor_sampler import SensorSampler


@dataclass
class OrchestratorConfig:
    step_s: int = 60
    sample_every_s: int = 300
    wall_sleep_s: float = 0.0
    comfort_max_changes_per_day: int = 3


class SimulationOrchestrator:
    def __init__(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        on_event,
        config: OrchestratorConfig,
        seed: int = 42,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.on_event = on_event
        self.config = config

        self.rooms = {i: RoomState(i) for i in range(1, 101)}
        self.engine = RoomEngine(self.rooms)
        self.sampler = SensorSampler(seed)

        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        now = self.start_time
        last_sample = now

        print("START SIM LOOP", now, self.end_time)

        while now < self.end_time:
            self.engine.apply_targets_from_db(now)
            self.engine.step()

            if (now - last_sample).total_seconds() >= self.config.sample_every_s:
                print(f"Sim Time: {now} - Emitting sensor events...")
                await self.sampler.emit(
                    now,
                    room_engine=self.engine,
                    on_event=self.on_event,
                )
                last_sample = now

            now += timedelta(seconds=self.config.step_s)

            if self.config.wall_sleep_s > 0:
                await asyncio.sleep(self.config.wall_sleep_s)