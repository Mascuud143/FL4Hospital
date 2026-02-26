from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, Optional

from .room_engine import RoomEngine, RoomState, _as_utc
from .sensor_sampler import SensorSampler
from .comfort_generator import ComfortGenerator, ComfortPolicy
from .toilet_usage_generator import ToiletUsageGenerator

# NEW: clinical generators
from simulation_batch.generators.medication_generator import MedicationGenerator
from simulation_batch.generators.visit_generator import VisitGenerator
from simulation_batch.generators.patients import DIAGNOSES


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
      - comfort pre-generation
      - medication generation
      - visit generation
      - toilet usage generation
      - simulation loop (physics + sensors)
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

        # Comfort generator
        self.comfort = ComfortGenerator(
            seed=seed,
            policy=ComfortPolicy(
                max_changes_per_day=self.config.comfort_max_changes_per_day
            ),
        )

        # Room states (1–100)
        self.rooms = {i: RoomState(i) for i in range(1, 101)}

        # Simulation engine
        self.engine = RoomEngine(self.rooms)

        # Sensor sampler
        self.sampler = SensorSampler(seed)

        self._stop = asyncio.Event()

    # -------------------------------------------------------
    # START
    # -------------------------------------------------------

    async def start(self) -> None:
        """
        Start simulation:
          1. Generate all time-based events
          2. Run simulation loop
        """
        self._stop.clear()

        print("[orchestrator] PRE-GENERATION START")



        # ---------------------------------------------------
        # 2️⃣ Medication events
        # ---------------------------------------------------
        med_gen = MedicationGenerator(seed=self.seed, diagnoses=DIAGNOSES)
        inserted = med_gen.generate_for_horizon(
            self.start_time,
            self.end_time,
        )
        print(f"[orchestrator] medication rows inserted: {inserted}")

        # ---------------------------------------------------
        # 3️⃣ Visit events
        # ---------------------------------------------------
        visit_gen = VisitGenerator(seed=self.seed)
        inserted = visit_gen.generate_for_horizon(
            self.start_time,
            self.end_time,
        )
        print(f"[orchestrator] visit rows inserted: {inserted}")


        # ---------------------------------------------------
        # 1️⃣ Comfort preferences
        # ---------------------------------------------------
        inserted = self.comfort.generate_for_horizon(
            self.start_time,
            self.end_time,
        )
        print(f"[orchestrator] comfort rows inserted: {inserted}")

        # ---------------------------------------------------
        # 4️⃣ Toilet usage
        # ---------------------------------------------------
        toilet_gen = ToiletUsageGenerator(seed=self.seed)
        inserted = toilet_gen.generate_for_horizon(
            self.start_time,
            self.end_time,
        )
        print(f"[orchestrator] toilet utility rows inserted: {inserted}")

        print("[orchestrator] PRE-GENERATION COMPLETE\n")

        # ---------------------------------------------------
        # 5️⃣ Run simulation loop
        # ---------------------------------------------------
        await self._run()

        # Close open utility sessions (important)
        self.engine.close_all_sessions(self.end_time)

        print("[orchestrator] SIMULATION COMPLETE")

    # -------------------------------------------------------
    # SIM LOOP
    # -------------------------------------------------------

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

            # Apply comfort targets
            self.engine.apply_targets_from_db(now)

            # Step room physics
            self.engine.step(now, step_s=self.config.step_s)

            # Emit sensor readings
            if (now - last_sample).total_seconds() >= self.config.sample_every_s:
                await self.sampler.emit(
                    now,
                    room_engine=self.engine,
                    on_event=self.on_event,
                )
                last_sample = now

            # Advance simulated time
            now += timedelta(seconds=self.config.step_s)

            # Optional wall delay
            if self.config.wall_sleep_s > 0:
                await asyncio.sleep(self.config.wall_sleep_s)

    # -------------------------------------------------------
    # STOP
    # -------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()