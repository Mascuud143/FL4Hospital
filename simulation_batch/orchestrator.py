from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, Optional

from .room_engine import EngineConfig, RoomEngine, RoomState, _as_utc
from .sensor_sampler import SensorSampler
from .comfort_generator import ComfortGenerator, ComfortPolicy
from .toilet_usage_generator import ToiletUsageGenerator
from persistence.models.data import Data
from simulation_batch.csv_filestorage import write_model_row
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
    enable_comfort: bool = True
    enable_medication: bool = True
    enable_visits: bool = True
    enable_toilet_usage: bool = False
    enable_sensor_emit: bool = False
    enable_utility_usage: bool = False


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
        self._on_event_user = on_event
        self.on_event = self._wrap_on_event(on_event)
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
        self.engine = RoomEngine(
            self.rooms,
            config=EngineConfig(enable_utility_usage=self.config.enable_utility_usage),
        )

        # Sensor sampler
        self.sampler = SensorSampler(seed)

        self._stop = asyncio.Event()

    def _wrap_on_event(self, handler: OnEvent) -> OnEvent:
        async def _wrapped(event: Dict[str, Any]) -> None:
            # Save to data table CSV before any downstream sink
            write_model_row(
                Data(
                    sensor_id=event["sensor_id"],
                    value=event["value"],
                    timestamp=event["timestamp"],
                )
            )
            await handler(event)

        return _wrapped

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
        if self.config.enable_medication:
            med_gen = MedicationGenerator(seed=self.seed, diagnoses=DIAGNOSES)
            inserted = med_gen.generate_for_horizon(
                self.start_time,
                self.end_time,
            )
            print(f"[orchestrator] medication rows inserted: {inserted}")
        else:
            print("[orchestrator] medication generation disabled")

        # ---------------------------------------------------
        # 3️⃣ Visit events
        # ---------------------------------------------------
        if self.config.enable_visits:
            visit_gen = VisitGenerator(seed=self.seed)
            inserted = visit_gen.generate_for_horizon(
                self.start_time,
                self.end_time,
            )
            print(f"[orchestrator] visit rows inserted: {inserted}")
        else:
            print("[orchestrator] visit generation disabled")


        # ---------------------------------------------------
        # 1️⃣ Comfort preferences
        # ---------------------------------------------------
        if self.config.enable_comfort:
            inserted = self.comfort.generate_for_horizon(
                self.start_time,
                self.end_time,
            )
            print(f"[orchestrator] comfort rows inserted: {inserted}")
        else:
            print("[orchestrator] comfort generation disabled")

        # ---------------------------------------------------
        # 4️⃣ Toilet usage
        # ---------------------------------------------------
        if self.config.enable_toilet_usage:
            toilet_gen = ToiletUsageGenerator(seed=self.seed)
            inserted = toilet_gen.generate_for_horizon(
                self.start_time,
                self.end_time,
            )
            print(f"[orchestrator] toilet utility rows inserted: {inserted}")
        else:
            print("[orchestrator] toilet usage generation disabled")

        print("[orchestrator] PRE-GENERATION COMPLETE\n")

        # ---------------------------------------------------
        # 5️⃣ Run simulation loop
        # ---------------------------------------------------
        if not self.config.enable_sensor_emit and not self.config.enable_utility_usage:
            print("[orchestrator] skipping simulation loop (no sensor emit and no utility usage)")
            print("[orchestrator] SIMULATION COMPLETE")
            return

        await self._run()

        if self.config.enable_utility_usage:
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
            if self.config.enable_sensor_emit and (now - last_sample).total_seconds() >= self.config.sample_every_s:
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
    
