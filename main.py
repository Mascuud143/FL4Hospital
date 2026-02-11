import asyncio
from datetime import datetime, timedelta, timezone
import os

from data_collection import DataCollector
from data_collection.db_sink import db_sink

from persistence import init_db
from persistence.seed_devices import seed_devices_and_sensors

from simulation_batch.orchestrator import SimulationOrchestrator, OrchestratorConfig
from simulation_batch.seed_world import seed_simulated_world
from simulation_batch.config import START_DATE, DAYS, PATIENT_COUNT




def _to_utc_dt(d) -> datetime:
    return datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)


async def main():
    #delete existing db and re-init 
    if os.path.exists("fl4hospital.db"):
        os.remove("fl4hospital.db")

    init_db("sqlite:///fl4hospital.db", echo=False)

    collector = DataCollector(sinks=[db_sink])
    await collector.start()

    devices = seed_simulated_world(
        patient_count=PATIENT_COUNT,
        days=DAYS,
        start_date=START_DATE,
        seed=42,
    )
    seed_devices_and_sensors(devices)

    start = _to_utc_dt(START_DATE)
    end = start + timedelta(days=DAYS)

    sim = SimulationOrchestrator(
        start_time=start,
        end_time=end,
        on_event=collector.ingest,
        config=OrchestratorConfig(
            step_s=60,
            sample_every_s=300,
            wall_sleep_s=0.0,
        ),
        seed=42,
    )

    await sim.start()
    await sim._task
    await collector.stop()

    print("Simulation finished")


if __name__ == "__main__":
    asyncio.run(main())
