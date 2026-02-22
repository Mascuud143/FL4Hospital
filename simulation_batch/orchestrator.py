from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, Optional

from .room_engine import RoomEngine, RoomState, _as_utc
from .sensor_sampler import SensorSampler
from .comfort_generator import ComfortGenerator, ComfortPolicy
from .toilet_usage_generator import ToiletUsageGenerator

# Callback type used when emitting sensor events
OnEvent = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class OrchestratorConfig:
    step_s: int = 60
    sample_every_s: int = 300
    wall_sleep_s: float = 0.0
    comfort_max_changes_per_day: int = 3


class SimulationOrchestrator:
    """
    Runs:
      - comfort pre-generation for horizon (writes ComfortPreference rows)
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
        on_event: OnEvent,
        config: Optional[OrchestratorConfig] = None,
        seed: int = 42,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.on_event = on_event
        self.config = config or OrchestratorConfig()
        self.seed = seed

        # Comfort preference generator (writes to DB)
        self.comfort = ComfortGenerator(
            seed=seed,
            policy=ComfortPolicy(
                max_changes_per_day=self.config.comfort_max_changes_per_day
            ),
        )

        # Create room states (1–100)
        self.rooms = {i: RoomState(i) for i in range(1, 101)}

        # Simulation engine
        self.engine = RoomEngine(self.rooms)

        # Sensor sampler (loads all sensors once at init)
        self.sampler = SensorSampler(seed)

        self._stop = asyncio.Event()

    async def start(self) -> None:
        """
        Start simulation:
        1. Generate comfort preferences for the entire horizon
        2. Run simulation loop
        """
        self._stop.clear()

        # Pre-generate comfort settings into DB
        inserted = self.comfort.generate_for_horizon(
            self.start_time,
            self.end_time,
        )
        print(f"[orchestrator] comfort rows inserted: {inserted}")

        toilet_gen = ToiletUsageGenerator(seed=self.seed)
        inserted = toilet_gen.generate_for_horizon(self.start_time, self.end_time)
        print("[orchestrator] toilet utility rows inserted:", inserted)

        await self._run()

        # IMPORTANT: close any still-open utility sessions (especially airflow)
        # so they appear in UtilityUsage even if they never turned off before sim end.
        self.engine.close_all_sessions(self.end_time)

    async def _run(self):
        now = self.start_time
        last_sample = now

        print(
            "[orchestrator] START SIM LOOP",
            self.start_time,
            "->",
            self.end_time,
        )

        while now < self.end_time:
            if self._stop.is_set():
                break

            # Apply latest comfort targets from DB
            self.engine.apply_targets_from_db(now)
            self.engine.step(now, step_s=self.config.step_s)

            # Emit sensors periodically
            if (now - last_sample).total_seconds() >= self.config.sample_every_s:
                await self.sampler.emit(
                    now,
                    room_engine=self.engine,
                    on_event=self.on_event,
                )
                last_sample = now

            # Advance simulation time
            now += timedelta(seconds=self.config.step_s)

            # Optional wall-clock delay
            if self.config.wall_sleep_s > 0:
                await asyncio.sleep(self.config.wall_sleep_s)


