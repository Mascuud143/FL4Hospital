from __future__ import annotations

# Run the data generation step.
# - Runs medication generation with _run_medication_generation()
# - Runs visit generation with _run_visit_generation()
# - Runs comfort generation with _run_comfort_generation()
# - Runs water usage generation with _run_water_usage_generation()
# - Runs enabled steps with run_generation_pipeline()

from dataclasses import dataclass
from datetime import datetime

from simulation_batch.generators.comfort_generation import ComfortGenerator
from simulation_batch.generators.diagnosis_profiles import DIAGNOSES
from simulation_batch.generators.medication_generation import MedicationGenerator
from simulation_batch.generators.visit_generation import VisitGenerator
from simulation_batch.generators.water_usage_generation import ToiletUsageGenerator


@dataclass(frozen=True)
class GenerationPipelineConfig:
    enable_comfort: bool
    enable_medication: bool
    enable_visits: bool
    enable_toilet_usage: bool
    seed: int


def _run_medication_generation(*, start_time: datetime, end_time: datetime, seed: int) -> None:
    inserted = MedicationGenerator(seed=seed, diagnoses=DIAGNOSES).generate_for_horizon(start_time, end_time)
    print(f"[orchestrator] medication rows inserted: {inserted}")


def _run_visit_generation(*, start_time: datetime, end_time: datetime, seed: int) -> None:
    inserted = VisitGenerator(seed=seed).generate_for_horizon(start_time, end_time)
    print(f"[orchestrator] visit rows inserted: {inserted}")


def _run_comfort_generation(*, start_time: datetime, end_time: datetime, comfort_generator: ComfortGenerator) -> None:
    inserted = comfort_generator.generate_for_horizon(start_time, end_time)
    print(f"[orchestrator] comfort rows inserted: {inserted}")


def _run_water_usage_generation(*, start_time: datetime, end_time: datetime, seed: int) -> None:
    inserted = ToiletUsageGenerator(seed=seed).generate_for_horizon(start_time, end_time)
    print(f"[orchestrator] toilet utility rows inserted: {inserted}")


def run_generation_pipeline(
    *,
    start_time: datetime,
    end_time: datetime,
    config: GenerationPipelineConfig,
    comfort_generator: ComfortGenerator,
) -> None:
    print("[orchestrator] PRE-GENERATION START")

    if config.enable_medication:
        _run_medication_generation(start_time=start_time, end_time=end_time, seed=config.seed)
    else:
        print("[orchestrator] medication generation disabled")

    if config.enable_visits:
        _run_visit_generation(start_time=start_time, end_time=end_time, seed=config.seed)
    else:
        print("[orchestrator] visit generation disabled")

    if config.enable_comfort:
        _run_comfort_generation(start_time=start_time, end_time=end_time, comfort_generator=comfort_generator)
    else:
        print("[orchestrator] comfort generation disabled")

    if config.enable_toilet_usage:
        _run_water_usage_generation(start_time=start_time, end_time=end_time, seed=config.seed)
    else:
        print("[orchestrator] toilet usage generation disabled")

    print("[orchestrator] PRE-GENERATION COMPLETE\n")
