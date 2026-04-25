from __future__ import annotations

# Track utility sessions.
# - Computes session energy - calculate_energy_usage_kwh()
# - Handles HVAC sessions
# - Handles airflow sessions
# - Handles heater and light sessions
# - Writes utility rows - UtilitySessionManager

from datetime import datetime
from typing import Optional, Tuple

from persistence.database import session_scope
from persistence.models.toilet_heater import ToiletHeater
from persistence.models.toilet_light import ToiletLight
from persistence.models.ventilation import Ventilation
from simulation_batch.csv_storage import flush_utility_usage_writes, insert_utility_usage, write_model_rows


def calculate_energy_usage_kwh(power_watts: float, duration_hours: float) -> float:
    return (power_watts * duration_hours) / 1000


class UtilitySessionManager:
    def __init__(self, *, config, as_utc) -> None:
        self.cfg = config
        self._as_utc = as_utc
        self._write_batch_size = 50000
        self._ventilation_rows: list[tuple[int, str, float, datetime]] = []
        self._toilet_heater_rows: list[tuple[int, bool, datetime]] = []
        self._toilet_light_rows: list[tuple[int, bool, datetime]] = []
        self._cache = None

    def bind_cache(self, cache) -> None:
        self._cache = cache

    def _flush_model_buffer(self, model, rows: list[tuple]) -> None:
        if not rows:
            return
        batch = list(rows)
        rows.clear()
        if model is Ventilation:
            serialized_rows = [{"device_id": device_id, "mode": mode, "level": level, "timestamp": timestamp} for device_id, mode, level, timestamp in batch]
        else:
            serialized_rows = [{"device_id": device_id, "state": state, "timestamp": timestamp} for device_id, state, timestamp in batch]
        write_model_rows(model, serialized_rows)
        with session_scope() as session:
            session.bulk_insert_mappings(model, serialized_rows)

    def _compute_vent_mode(self, room) -> Tuple[str, float]:
        hvac_active = room._hvac_session_start is not None and room._hvac_session_target is not None
        if hvac_active:
            target = float(room._hvac_session_target)
            current = float(room.temperature)
            if current < target - self.cfg.temp_tolerance_c:
                return "heat", 1.0
            if current > target + self.cfg.temp_tolerance_c:
                return "cool", 1.0
            if room.airflow_requested:
                return "airflow", 1.0
            return "airflow", 0.0
        return ("airflow", 1.0) if room.airflow_requested else ("airflow", 0.0)

    def _device_id(self, *, room_id: int, device_type: str) -> Optional[int]:
        if self._cache is None:
            return None
        return self._cache.get_device_id(room_id=room_id, device_type=device_type)

    def log_ventilation_if_changed(self, room, *, when: datetime) -> None:
        if not self.cfg.enable_utility_usage:
            return
        when = self._as_utc(when)
        mode, level = self._compute_vent_mode(room)
        if mode == "airflow" and float(level) == 0.0:
            room._last_vent_mode = mode
            room._last_vent_level = level
            return
        if room._last_vent_mode == mode and room._last_vent_level == level:
            return
        room._last_vent_mode = mode
        room._last_vent_level = level
        dev_id = self._device_id(room_id=room.room_id, device_type="ventilation")
        if dev_id is None:
            return
        self._ventilation_rows.append((int(dev_id), str(mode), float(level), when))
        if len(self._ventilation_rows) >= self._write_batch_size:
            self._flush_model_buffer(Ventilation, self._ventilation_rows)

    def log_toilet_heater_if_changed(self, room, *, when: datetime, state: bool) -> None:
        if not self.cfg.enable_utility_usage:
            return
        when = self._as_utc(when)
        if room._last_toilet_heater_state == state:
            return
        room._last_toilet_heater_state = state
        dev_id = self._device_id(room_id=room.room_id, device_type="toilet_heater")
        if dev_id is None:
            return
        self._toilet_heater_rows.append((int(dev_id), bool(state), when))
        if len(self._toilet_heater_rows) >= self._write_batch_size:
            self._flush_model_buffer(ToiletHeater, self._toilet_heater_rows)

    def log_toilet_light_if_changed(self, room, *, when: datetime, state: bool) -> None:
        if not self.cfg.enable_utility_usage:
            return
        when = self._as_utc(when)
        if room._last_toilet_light_state == state:
            return
        room._last_toilet_light_state = state
        dev_id = self._device_id(room_id=room.room_id, device_type="toilet_light")
        if dev_id is None:
            return
        self._toilet_light_rows.append((int(dev_id), bool(state), when))
        if len(self._toilet_light_rows) >= self._write_batch_size:
            self._flush_model_buffer(ToiletLight, self._toilet_light_rows)

    def update_onoff_utilities(self, room, *, when: datetime, airflow: bool, toilet_heater: bool, toilet_light: bool) -> None:
        when = self._as_utc(when)
        hvac_active = room._hvac_session_start is not None and room._hvac_session_target is not None
        if airflow and not hvac_active:
            if room._airflow_session_start is None:
                room._airflow_session_start = when
        elif room._airflow_session_start is not None:
            self.close_airflow_session(room, end_time=when)
        if not toilet_heater:
            room._toilet_heater_active_until = None
            if room._toilet_heater_session_start is not None:
                self.close_toilet_heater_session(room, end_time=when)
        if not toilet_light:
            room._toilet_light_active_until = None
            if room._toilet_light_session_start is not None:
                self.close_toilet_light_session(room, end_time=when)

    def check_hvac_stabilization(self, room, *, now: datetime) -> None:
        if room._hvac_session_start is None or room._hvac_session_target is None:
            return
        if (now - room._hvac_session_start).total_seconds() < self.cfg.min_hvac_run_s:
            return
        err = abs(room.temperature - room._hvac_session_target)
        room._hvac_stable_ticks = room._hvac_stable_ticks + 1 if err <= self.cfg.temp_tolerance_c else 0
        if room._hvac_stable_ticks >= self.cfg.stable_ticks_required:
            self.close_hvac_session(room, end_time=now)

    def close_hvac_session(self, room, *, end_time: datetime) -> None:
        start = room._hvac_session_start
        if start is None:
            return
        start = self._as_utc(start)
        end_time = self._as_utc(end_time)
        kwh = calculate_energy_usage_kwh(self.cfg.hvac_power_w, max(0.0, (end_time - start).total_seconds() / 3600.0))
        if self.cfg.enable_utility_usage:
            insert_utility_usage(room_id=room.room_id, category="hvac", start_time=start, end_time=end_time, power_kwh=kwh, water_liters=None)
        room._hvac_session_start = None
        room._hvac_session_target = None
        room._hvac_stable_ticks = 0

    def close_airflow_session(self, room, *, end_time: datetime) -> None:
        start = room._airflow_session_start
        if start is None:
            return
        start = self._as_utc(start)
        end_time = self._as_utc(end_time)
        kwh = calculate_energy_usage_kwh(self.cfg.ventilation_power_w, max(0.0, (end_time - start).total_seconds() / 3600.0))
        if self.cfg.enable_utility_usage:
            insert_utility_usage(room_id=room.room_id, category="airflow", start_time=start, end_time=end_time, power_kwh=kwh, water_liters=None)
        room._airflow_session_start = None

    def close_toilet_heater_session(self, room, *, end_time: datetime) -> None:
        start = room._toilet_heater_session_start
        if start is None:
            return
        start = self._as_utc(start)
        end_time = self._as_utc(end_time)
        kwh = calculate_energy_usage_kwh(self.cfg.toilet_heater_power_w, max(0.0, (end_time - start).total_seconds() / 3600.0))
        if self.cfg.enable_utility_usage:
            insert_utility_usage(room_id=room.room_id, category="toilet_heater", start_time=start, end_time=end_time, power_kwh=kwh, water_liters=None)
        room._toilet_heater_session_start = None
        room._toilet_heater_active_until = None

    def close_toilet_light_session(self, room, *, end_time: datetime) -> None:
        start = room._toilet_light_session_start
        if start is None:
            return
        start = self._as_utc(start)
        end_time = self._as_utc(end_time)
        kwh = calculate_energy_usage_kwh(self.cfg.toilet_light_power_w, max(0.0, (end_time - start).total_seconds() / 3600.0))
        if self.cfg.enable_utility_usage:
            insert_utility_usage(room_id=room.room_id, category="toilet_light", start_time=start, end_time=end_time, power_kwh=kwh, water_liters=None)
        room._toilet_light_session_start = None
        room._toilet_light_active_until = None

    def close_all_open_sessions(self, room, *, end_time: datetime) -> None:
        end_time = self._as_utc(end_time)
        if room._hvac_session_start is not None:
            self.close_hvac_session(room, end_time=end_time)
        if room._airflow_session_start is not None:
            self.close_airflow_session(room, end_time=end_time)
        if room._toilet_heater_session_start is not None:
            self.close_toilet_heater_session(room, end_time=end_time)
        if room._toilet_light_session_start is not None:
            self.close_toilet_light_session(room, end_time=end_time)

    def close_all_sessions(self, rooms: dict[int, object], *, end_time: datetime) -> None:
        end_time = self._as_utc(end_time)
        for room in rooms.values():
            self.close_all_open_sessions(room, end_time=end_time)
        self._flush_model_buffer(Ventilation, self._ventilation_rows)
        self._flush_model_buffer(ToiletHeater, self._toilet_heater_rows)
        self._flush_model_buffer(ToiletLight, self._toilet_light_rows)
        flush_utility_usage_writes()
