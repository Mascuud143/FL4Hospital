from __future__ import annotations

# Run the simulation loop.
# - Runs one loop with run_simulation_loop()
# - Updates room values with _run_simulation_step()
# - Samples sensors with _collect_sensor_rows_if_due()
# - Moves time forward with _advance_simulation_time()
# - Sleeps between steps with _sleep_if_requested()

from datetime import datetime, timedelta
from typing import Callable


def run_simulation_loop(
    *,
    start_time: datetime,
    end_time: datetime,
    step_s: int,
    sample_every_s: int,
    wall_sleep_s: float,
    stop_requested: Callable[[], bool],
    enable_sensor_emit: bool,
    engine,
    sampler,
    on_sensor_rows: Callable[[list[tuple[int, float, datetime]]], None],
) -> None:
    now = start_time
    last_sample = now
    print("[orchestrator] START SIM LOOP", start_time, "->", end_time)
    while now < end_time:
        if stop_requested():
            break
        _run_simulation_step(now=now, step_s=step_s, engine=engine)
        last_sample = _collect_sensor_rows_if_due(
            now=now,
            last_sample=last_sample,
            sample_every_s=sample_every_s,
            enable_sensor_emit=enable_sensor_emit,
            sampler=sampler,
            engine=engine,
            on_sensor_rows=on_sensor_rows,
        )
        now = _advance_simulation_time(now=now, step_s=step_s)
        _sleep_if_requested(wall_sleep_s)


def _run_simulation_step(*, now: datetime, step_s: int, engine) -> None:
    engine.apply_targets_from_db(now)
    engine.step(now, step_s=step_s)


def _collect_sensor_rows_if_due(
    *,
    now: datetime,
    last_sample: datetime,
    sample_every_s: int,
    enable_sensor_emit: bool,
    sampler,
    engine,
    on_sensor_rows: Callable[[list[tuple[int, float, datetime]]], None],
) -> datetime:
    if not enable_sensor_emit:
        return last_sample
    if (now - last_sample).total_seconds() < sample_every_s:
        return last_sample
    on_sensor_rows(sampler.collect_data_rows(now, room_engine=engine))
    return now


def _advance_simulation_time(*, now: datetime, step_s: int) -> datetime:
    return now + timedelta(seconds=step_s)


def _sleep_if_requested(wall_sleep_s: float) -> None:
    if wall_sleep_s > 0:
        import time

        time.sleep(wall_sleep_s)
