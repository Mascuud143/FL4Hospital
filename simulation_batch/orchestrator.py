from __future__ import annotations

# Run the full batch flow.
# - Builds run settings with OrchestratorConfig
# - Runs the batch flow with SimulationOrchestrator
# - Starts data generation
# - Starts room simulation and sensor sampling
# - Saves sensor rows and closes utility sessions

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from persistence.database import session_scope
from persistence.models.data import Data
from simulation_batch.clock import run_simulation_loop
from simulation_batch.csv_storage import write_model_rows
from simulation_batch.generation_pipeline import GenerationPipelineConfig, run_generation_pipeline
from simulation_batch.generators.comfort_generation import ComfortGenerator, ComfortPolicy
from simulation_batch.simulation_steps.room_simulation import EngineConfig, RoomEngine
from simulation_batch.simulation_steps.room_dynamics import RoomState
from simulation_batch.simulation_steps.sensor_sampling import SensorSampler


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
    def __init__(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        on_event: Callable | None = None,
        config: Optional[OrchestratorConfig] = None,
        seed: int = 42,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self._on_event_user = on_event
        self.config = config or OrchestratorConfig()
        self.seed = seed
        self.comfort = ComfortGenerator(seed=seed, policy=ComfortPolicy(max_changes_per_day=self.config.comfort_max_changes_per_day))
        self.sampler = SensorSampler(seed)
        self.rooms = {room_id: RoomState(room_id) for room_id in self.sampler.room_ids}
        self.engine = RoomEngine(self.rooms, config=EngineConfig(enable_utility_usage=self.config.enable_utility_usage))
        self._write_batch_size = 500000
        self._data_rows: list[tuple[int, float, datetime]] = []
        self._stop_requested = False

    def _flush_data_rows(self) -> None:
        if not self._data_rows:
            return
        batch = self._data_rows
        self._data_rows = []
        serialized_batch = [{"sensor_id": sensor_id, "value": value, "timestamp": timestamp} for sensor_id, value, timestamp in batch]
        write_model_rows(Data, serialized_batch)
        with session_scope() as session:
            session.bulk_insert_mappings(Data, serialized_batch)

    def _append_sensor_rows(self, rows: list[tuple[int, float, datetime]]) -> None:
        self._data_rows.extend(rows)
        if len(self._data_rows) >= self._write_batch_size:
            self._flush_data_rows()

    def _run_pre_generation(self) -> None:
        run_generation_pipeline(
            start_time=self.start_time,
            end_time=self.end_time,
            config=GenerationPipelineConfig(
                enable_comfort=self.config.enable_comfort,
                enable_medication=self.config.enable_medication,
                enable_visits=self.config.enable_visits,
                enable_toilet_usage=self.config.enable_toilet_usage,
                seed=self.seed,
            ),
            comfort_generator=self.comfort,
        )

    def _prepare_simulation(self) -> None:
        self.engine.preload_simulation_window(self.start_time, self.end_time)

    def _should_skip_simulation_loop(self) -> bool:
        return not self.config.enable_sensor_emit and not self.config.enable_utility_usage

    def _run_simulation(self) -> None:
        run_simulation_loop(
            start_time=self.start_time,
            end_time=self.end_time,
            step_s=self.config.step_s,
            sample_every_s=self.config.sample_every_s,
            wall_sleep_s=self.config.wall_sleep_s,
            stop_requested=lambda: self._stop_requested,
            enable_sensor_emit=self.config.enable_sensor_emit,
            engine=self.engine,
            sampler=self.sampler,
            on_sensor_rows=self._append_sensor_rows,
        )

    def _finalize_simulation(self) -> None:
        if self.config.enable_utility_usage:
            self.engine.close_all_sessions(self.end_time)
        self._flush_data_rows()
        self.sampler.close()
        print("[orchestrator] SIMULATION COMPLETE")

    def start(self) -> None:
        self._stop_requested = False
        self._run_pre_generation()
        self._prepare_simulation()
        if self._should_skip_simulation_loop():
            print("[orchestrator] skipping simulation loop (no sensor emit and no utility usage)")
            print("[orchestrator] SIMULATION COMPLETE")
            return
        self._run_simulation()
        self._finalize_simulation()

    def stop(self) -> None:
        self._stop_requested = True
