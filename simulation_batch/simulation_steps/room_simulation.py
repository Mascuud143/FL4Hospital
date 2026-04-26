from __future__ import annotations

# Run room simulation.
# - Converts time to UTC - as_utc()
# - Stores room engine settings - EngineConfig
# - Runs room simulation - RoomEngine
# - Loads room data and applies room targets
# - Checks utility sessions

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from simulation_batch.simulation_steps.room_preload import RoomPreloader
from simulation_batch.simulation_steps.room_dynamics import RoomState
from simulation_batch.simulation_steps.utility_sessions import UtilitySessionManager


def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class EngineConfig:
    hvac_power_w: float = 1500.0
    ventilation_power_w: float = 80.0
    toilet_heater_power_w: float = 600.0
    toilet_light_power_w: float = 10.0
    toilet_heater_run_s: int = 15 * 60
    enable_utility_usage: bool = False
    temp_tolerance_c: float = 0.4
    stable_ticks_required: int = 3
    min_hvac_run_s: int = 5 * 60
    merge_target_delta_c: float = 0.30
    preload_batch_size: int = 50000


class RoomEngine:
    def __init__(self, rooms: Dict[int, RoomState], *, config: Optional[EngineConfig] = None):
        self.rooms = rooms
        self.cfg = config or EngineConfig()
        self._preloader = RoomPreloader(preload_batch_size=self.cfg.preload_batch_size, as_utc=as_utc)
        self._cache = None
        self._utility_sessions = UtilitySessionManager(config=self.cfg, as_utc=as_utc)

    def preload_simulation_window(self, start_time, end_time) -> None:
        start_time = as_utc(start_time)
        end_time = as_utc(end_time)
        room_ids = tuple(sorted(self.rooms))
        self._cache = self._preloader.preload(room_ids=room_ids, start_time=start_time, end_time=end_time)
        self._utility_sessions.bind_cache(self._cache)

    def apply_targets_from_db(self, now) -> None:
        now = as_utc(now)
        if self._cache is None:
            raise RuntimeError("RoomEngine.preload_simulation_window() must be called before apply_targets_from_db().")
        for room in self.rooms.values():
            room._occupied_this_tick = False

        for room_id, room in self.rooms.items():
            assignment = self._cache.get_active_assignment(room_id, now)
            if assignment is None:
                continue
            room._occupied_this_tick = True
            room._occupied_until = assignment.end_time
            pref = self._cache.get_latest_preference(patient_id=assignment.patient_id, room_id=room_id, now=now)
            if pref is None:
                room.airflow_requested = False
                room.toilet_heater_requested = False
                room.toilet_light_requested = False
                self._utility_sessions.update_onoff_utilities(room, when=now, airflow=False, toilet_heater=False, toilet_light=False)
                self._utility_sessions.log_toilet_heater_if_changed(room, when=now, state=False)
                self._utility_sessions.log_toilet_light_if_changed(room, when=now, state=False)
                continue

            pref_ts = pref.timestamp
            is_new_pref = room._last_pref_ts is None or pref_ts > room._last_pref_ts
            if pref.temperature_main is not None:
                room.target_temperature = pref.temperature_main
            if pref.light_intensity is not None:
                room.target_light = pref.light_intensity
            if pref.sound_level is not None:
                room.target_sound = pref.sound_level

            airflow_req = pref.airflow
            toilet_heater_req = pref.toilet_heater_requested
            toilet_light_req = toilet_heater_req
            room.airflow_requested = airflow_req
            room.toilet_heater_requested = toilet_heater_req
            room.toilet_light_requested = toilet_light_req
            transition_time = pref_ts if is_new_pref else now

            if is_new_pref:
                room._last_pref_ts = pref_ts
                if pref.temperature_main is not None:
                    new_target = pref.temperature_main
                    if room._hvac_session_target is not None and abs(new_target - room._hvac_session_target) < self.cfg.merge_target_delta_c:
                        room._hvac_session_target = new_target
                    else:
                        if room._hvac_session_start is not None and room._hvac_session_target is not None:
                            self._utility_sessions.close_hvac_session(room, end_time=pref_ts)
                        room._hvac_session_start = pref_ts
                        room._hvac_session_target = new_target
                        room._hvac_stable_ticks = 0
                if toilet_heater_req and room._last_toilet_heater_pref_ts != pref_ts:
                    room._last_toilet_heater_pref_ts = pref_ts
                    room._toilet_heater_active_until = pref_ts + timedelta(seconds=self.cfg.toilet_heater_run_s)
                    if room._toilet_heater_session_start is None:
                        room._toilet_heater_session_start = pref_ts
                if toilet_light_req and room._last_toilet_light_pref_ts != pref_ts:
                    room._last_toilet_light_pref_ts = pref_ts
                    room._toilet_light_active_until = pref_ts + timedelta(seconds=self.cfg.toilet_heater_run_s)
                    if room._toilet_light_session_start is None:
                        room._toilet_light_session_start = pref_ts

            self._utility_sessions.update_onoff_utilities(
                room,
                when=transition_time,
                airflow=airflow_req,
                toilet_heater=toilet_heater_req,
                toilet_light=toilet_light_req,
            )
            self._utility_sessions.log_ventilation_if_changed(room, when=transition_time)
            self._utility_sessions.log_toilet_heater_if_changed(room, when=transition_time, state=toilet_heater_req)
            self._utility_sessions.log_toilet_light_if_changed(room, when=transition_time, state=toilet_light_req)

        for room in self.rooms.values():
            if not room._occupied_this_tick and room._occupied_until is not None:
                departure = room._occupied_until
                self._utility_sessions.close_all_open_sessions(room, end_time=departure)
                room.airflow_requested = False
                room.toilet_heater_requested = False
                room.toilet_light_requested = False
                self._utility_sessions.log_toilet_heater_if_changed(room, when=departure, state=False)
                self._utility_sessions.log_toilet_light_if_changed(room, when=departure, state=False)
                room._occupied_until = None

    def step(self, now, *, step_s: int) -> None:
        now = as_utc(now)
        for room in self.rooms.values():
            room.step_dynamics()
            self._utility_sessions.check_hvac_stabilization(room, now=now)
            if room._toilet_heater_session_start is not None and room._toilet_heater_active_until is not None and now >= room._toilet_heater_active_until:
                self._utility_sessions.close_toilet_heater_session(room, end_time=room._toilet_heater_active_until)
            if room._toilet_light_session_start is not None and room._toilet_light_active_until is not None and now >= room._toilet_light_active_until:
                self._utility_sessions.close_toilet_light_session(room, end_time=room._toilet_light_active_until)

    def close_all_sessions(self, end_time) -> None:
        self._utility_sessions.close_all_sessions(self.rooms, end_time=end_time)
