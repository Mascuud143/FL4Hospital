from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, Optional

from .room_engine import RoomEngine, RoomState, _as_utc
from .sensor_sampler import SensorSampler
from .comfort_generator import ComfortGenerator, ComfortPolicy
from .toilet_usage_generator import ToiletUsageGenerator

OnEvent = Callable[[Dict[str, Any]], Awaitable[None]]


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
        on_event: OnEvent,
        config: Optional[OrchestratorConfig] = None,
        seed: int = 42,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.on_event = on_event
        self.config = config or OrchestratorConfig()
        self.seed = seed

        self.comfort = ComfortGenerator(
            seed=seed,
            policy=ComfortPolicy(max_changes_per_day=self.config.comfort_max_changes_per_day),
        )

        self.rooms = {i: RoomState(i) for i in range(1, 101)}
        self.engine = RoomEngine(self.rooms)
        self.sampler = SensorSampler(seed)

    async def start(self):
        inserted = self.comfort.generate_for_horizon(self.start_time, self.end_time)
        print("[orchestrator] comfort rows inserted:", inserted)

        toilet_gen = ToiletUsageGenerator(seed=self.seed)
        inserted = toilet_gen.generate_for_horizon(self.start_time, self.end_time)
        print("[orchestrator] toilet utility rows inserted:", inserted)

        await self._run()

        # ✅ IMPORTANT: close any still-open utility sessions (especially airflow)
        # so they appear in UtilityUsage even if they never turned off before sim end.
        self.engine.close_all_sessions(self.end_time)

    async def _run(self):
        now = self.start_time
        last_sample = now

        while now < self.end_time:
            self.engine.apply_targets_from_db(now)
            self.engine.step(now, step_s=self.config.step_s)

            if (now - last_sample).total_seconds() >= self.config.sample_every_s:
                await self.sampler.emit(
                    now,
                    room_engine=self.engine,
                    on_event=self.on_event,
                )
                last_sample = now

            now += timedelta(seconds=self.config.step_s)

            if self.config.wall_sleep_s > 0:
                await asyncio.sleep(self.config.wall_sleep_s)


