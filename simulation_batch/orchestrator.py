from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .clock import SimClock
from .comfort_generator import ComfortGenerator, ComfortPolicy
from .room_engine import RoomEngine
from .sensor_sampler import SensorSampler


@dataclass
class OrchestratorConfig:
    step_s: int = 60                 # sim tick
    sample_every_s: int = 300        # sensor sampling cadence
    wall_sleep_s: float = 0.0        # 0 = fast
    comfort_max_changes_per_day: int = 3


class SimulationOrchestrator:
    """
    Runs:
      - (optional) comfort pre-generation for horizon
      - loop over simulated time:
          * apply targets from DB
          * step physics
          * emit sensor readings every sample cadence
      - stops at end time
    """

    def __init__(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        on_event,
        config: Optional[OrchestratorConfig] = None,
        seed: int = 42,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.on_event = on_event
        self.config = config or OrchestratorConfig()

        self.comfort = ComfortGenerator(
            seed=seed,
            policy=ComfortPolicy(max_changes_per_day=self.config.comfort_max_changes_per_day),
        )
        self.engine = RoomEngine(step_s=self.config.step_s, seed=seed + 100)
        self.sampler = SensorSampler(seed=seed + 200)

        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._stop.clear()

        # Pre-generate random comfort changes for the whole horizon (bounded per day).
        # This ensures preference timestamps are coherent with the sim clock.
        self.comfort.generate_for_horizon(self.start_time, self.end_time)

        # Build DB sensor emit map once (devices/sensors are seeded before orchestrator starts)
        self.sampler.build_emit_map_from_db()

        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        clock = SimClock(self.start_time, self.end_time, step_s=self.config.step_s)

        accum = 0
        for now in clock:
            if self._stop.is_set():
                break

            # ensure room states exist for active rooms
            self.engine.load_active_rooms(now)

            # apply targets from latest comfort prefs at this time + get occupancy
            occupancy = self.engine.apply_targets_from_db(now)

            # step physics
            self.engine.step(occupancy=occupancy)

            # sample sensors periodically
            accum += self.config.step_s
            if accum >= self.config.sample_every_s:
                accum = 0
                await self.sampler.emit(now, room_engine=self.engine, on_event=self.on_event)

            if self.config.wall_sleep_s > 0:
                await asyncio.sleep(self.config.wall_sleep_s)
