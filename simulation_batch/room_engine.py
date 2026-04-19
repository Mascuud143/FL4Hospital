from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from persistence.database import session_scope
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.room_assignment import RoomAssignment

from persistence.models.device import Device as DeviceModel
from persistence.models.ventilation import Ventilation
from persistence.models.toilet_heater import ToiletHeater
from persistence.models.toilet_light import ToiletLight

from simulation_batch.generators.utility_calculator import calculate_energy_usage_kwh
from simulation_batch.utility_usage_writer import insert_utility_usage, flush_utility_usage_writes
from simulation_batch.csv_filestorage import write_model_rows


def _as_utc(dt: datetime) -> datetime:
    """
    SQLite commonly returns naive datetimes.
    Treat naive DB timestamps as UTC (simulation time is UTC).
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class EngineConfig:
    # Power assumptions
    hvac_power_w: float = 1500.0
    ventilation_power_w: float = 80.0
    toilet_heater_power_w: float = 600.0
    toilet_light_power_w: float = 10.0
    toilet_heater_run_s: int = 15 * 60
    enable_utility_usage: bool = False

    # Stabilization logic
    temp_tolerance_c: float = 0.4          # wider tolerance prevents "runs forever"
    stable_ticks_required: int = 3         # fewer ticks to declare stable
    min_hvac_run_s: int = 5 * 60           # don't close HVAC before 5 minutes

    # Merge tiny target changes into existing session
    merge_target_delta_c: float = 0.30     # if new target within this delta, don't restart session
    preload_batch_size: int = 50000


@dataclass(frozen=True)
class AssignmentWindow:
    patient_id: int
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class PreferenceSnapshot:
    timestamp: datetime
    temperature_main: Optional[float]
    light_intensity: Optional[float]
    sound_level: Optional[float]
    airflow: bool
    toilet_heater_requested: bool


class RoomState:
    def __init__(self, room_id: int):
        self.room_id = room_id

        # Current state
        self.temperature = 21.0
        self.humidity = 45.0
        self.co2 = 600.0
        self.light = 100.0
        self.sound = 30.0

        # Targets
        self.target_temperature = 21.0
        self.target_light = 100.0
        self.target_sound = 30.0

        # Requests derived from preferences
        self.airflow_requested: bool = False
        self.toilet_heater_requested: bool = False
        self.toilet_light_requested: bool = False

        # Occupancy tracking
        self._occupied_this_tick: bool = False
        self._occupied_until: Optional[datetime] = None

        # Track last preference timestamp used (detect changes)
        self._last_pref_ts: Optional[datetime] = None

        # HVAC temperature session tracking (heat/cool until stabilized)
        self._hvac_session_start: Optional[datetime] = None
        self._hvac_session_target: Optional[float] = None
        self._hvac_stable_ticks: int = 0

        # Airflow session tracking (fan-only, ON/OFF)
        self._airflow_session_start: Optional[datetime] = None

        # Toilet heater energy session tracking
        self._toilet_heater_session_start: Optional[datetime] = None
        self._toilet_heater_active_until: Optional[datetime] = None
        self._last_toilet_heater_pref_ts: Optional[datetime] = None
        self._toilet_light_session_start: Optional[datetime] = None
        self._toilet_light_active_until: Optional[datetime] = None
        self._last_toilet_light_pref_ts: Optional[datetime] = None

        # Avoid duplicate actuator history inserts
        self._last_vent_mode: Optional[str] = None
        self._last_vent_level: Optional[float] = None
        self._last_toilet_heater_state: Optional[bool] = None
        self._last_toilet_light_state: Optional[bool] = None

    def step_dynamics(self) -> None:
        # Simple convergence dynamics
        self.temperature += (self.target_temperature - self.temperature) * 0.05
        self.light += (self.target_light - self.light) * 0.1
        self.sound += (self.target_sound - self.sound) * 0.1

        # Drift / background dynamics
        self.co2 += 5.0
        self.humidity += (45.0 - self.humidity) * 0.01


class RoomEngine:
    def __init__(self, rooms: Dict[int, RoomState], *, config: Optional[EngineConfig] = None):
        self.rooms = rooms
        self.cfg = config or EngineConfig()
        self._write_batch_size = 50000
        self._ventilation_rows: list[tuple[int, str, float, datetime]] = []
        self._toilet_heater_rows: list[tuple[int, bool, datetime]] = []
        self._toilet_light_rows: list[tuple[int, bool, datetime]] = []
        self._device_ids: dict[tuple[int, str], int] = {}
        self._assignments_by_room: dict[int, list[AssignmentWindow]] = {room_id: [] for room_id in rooms}
        self._assignment_start_times: dict[int, list[datetime]] = {room_id: [] for room_id in rooms}
        self._preferences_by_key: dict[tuple[int, int], list[PreferenceSnapshot]] = {}
        self._preference_times: dict[tuple[int, int], list[datetime]] = {}
        self._preloaded_window: Optional[tuple[datetime, datetime]] = None

    def _flush_model_buffer(self, model, buffer_name: str) -> None:
        rows = getattr(self, buffer_name)
        if not rows:
            return
        setattr(self, buffer_name, [])
        if model is Ventilation:
            serialized_rows = [
                {"device_id": device_id, "mode": mode, "level": level, "timestamp": timestamp}
                for device_id, mode, level, timestamp in rows
            ]
        else:
            serialized_rows = [
                {"device_id": device_id, "state": state, "timestamp": timestamp}
                for device_id, state, timestamp in rows
            ]
        write_model_rows(model, serialized_rows)
        with session_scope() as session:
            session.bulk_insert_mappings(model, serialized_rows)

    # -------------------------
    # DB helpers
    # -------------------------
    def _get_device_id(self, session, *, room_id: int, device_type: str) -> Optional[int]:
        cached = self._device_ids.get((room_id, device_type))
        if cached is not None:
            return cached
        d = (
            session.query(DeviceModel)
            .filter(DeviceModel.room_id == room_id)
            .filter(DeviceModel.device_type == device_type)
            .one_or_none()
        )
        if not d:
            return None
        device_id = int(d.device_id)
        self._device_ids[(room_id, device_type)] = device_id
        return device_id

    def preload_simulation_window(self, start_time: datetime, end_time: datetime) -> None:
        start_time = _as_utc(start_time)
        end_time = _as_utc(end_time)
        room_ids = tuple(sorted(self.rooms))
        self._assignments_by_room = {room_id: [] for room_id in room_ids}
        self._assignment_start_times = {room_id: [] for room_id in room_ids}
        self._preferences_by_key = {}
        self._preference_times = {}
        self._device_ids = {}

        with session_scope() as session:
            device_rows = (
                session.query(DeviceModel.room_id, DeviceModel.device_type, DeviceModel.device_id)
                .filter(DeviceModel.room_id.in_(room_ids))
                .yield_per(self.cfg.preload_batch_size)
            )
            for room_id, device_type, device_id in device_rows:
                if room_id is None or device_type is None or device_id is None:
                    continue
                self._device_ids[(int(room_id), str(device_type))] = int(device_id)

            assignment_rows = (
                session.query(
                    RoomAssignment.room_id,
                    RoomAssignment.patient_id,
                    RoomAssignment.start_time,
                    RoomAssignment.end_time,
                )
                .filter(RoomAssignment.room_id.in_(room_ids))
                .filter(RoomAssignment.start_time < end_time)
                .filter(RoomAssignment.end_time > start_time)
                .order_by(RoomAssignment.room_id, RoomAssignment.start_time)
                .yield_per(self.cfg.preload_batch_size)
            )
            pref_keys: set[tuple[int, int]] = set()
            for room_id, patient_id, assignment_start, assignment_end in assignment_rows:
                if room_id is None or patient_id is None or assignment_start is None or assignment_end is None:
                    continue
                room_key = int(room_id)
                start_utc = _as_utc(assignment_start)
                end_utc = _as_utc(assignment_end)
                window = AssignmentWindow(
                    patient_id=int(patient_id),
                    start_time=start_utc,
                    end_time=end_utc,
                )
                self._assignments_by_room.setdefault(room_key, []).append(window)
                self._assignment_start_times.setdefault(room_key, []).append(start_utc)
                pref_keys.add((int(patient_id), room_key))

            if pref_keys:
                patient_ids = sorted({patient_id for patient_id, _ in pref_keys})
                pref_rows = (
                    session.query(
                        ComfortPreference.patient_id,
                        ComfortPreference.room_id,
                        ComfortPreference.timestamp,
                        ComfortPreference.temperature_main,
                        ComfortPreference.light_intensity,
                        ComfortPreference.sound_level,
                        ComfortPreference.airflow,
                        ComfortPreference.temperature_toilet,
                    )
                    .filter(ComfortPreference.patient_id.in_(patient_ids))
                    .filter(ComfortPreference.room_id.in_(room_ids))
                    .filter(ComfortPreference.timestamp <= end_time)
                    .order_by(
                        ComfortPreference.patient_id,
                        ComfortPreference.room_id,
                        ComfortPreference.timestamp,
                    )
                    .yield_per(self.cfg.preload_batch_size)
                )
                for (
                    patient_id,
                    room_id,
                    pref_ts,
                    temperature_main,
                    light_intensity,
                    sound_level,
                    airflow,
                    temperature_toilet,
                ) in pref_rows:
                    if patient_id is None or room_id is None or pref_ts is None:
                        continue
                    key = (int(patient_id), int(room_id))
                    if key not in pref_keys:
                        continue
                    self._preferences_by_key.setdefault(key, []).append(
                        PreferenceSnapshot(
                            timestamp=_as_utc(pref_ts),
                            temperature_main=float(temperature_main) if temperature_main is not None else None,
                            light_intensity=float(light_intensity) if light_intensity is not None else None,
                            sound_level=float(sound_level) if sound_level is not None else None,
                            airflow=bool(airflow),
                            toilet_heater_requested=temperature_toilet is not None,
                        )
                    )
                    self._preference_times.setdefault(key, []).append(_as_utc(pref_ts))

        self._preloaded_window = (start_time, end_time)

    def _get_active_assignment(self, room_id: int, now: datetime) -> Optional[AssignmentWindow]:
        starts = self._assignment_start_times.get(room_id)
        if not starts:
            return None
        idx = bisect_right(starts, now) - 1
        if idx < 0:
            return None
        assignment = self._assignments_by_room[room_id][idx]
        if assignment.start_time <= now < assignment.end_time:
            return assignment
        return None

    def _get_latest_preference(
        self,
        *,
        patient_id: int,
        room_id: int,
        now: datetime,
    ) -> Optional[PreferenceSnapshot]:
        prefs = self._preferences_by_key.get((patient_id, room_id))
        if not prefs:
            return None
        pref_times = self._preference_times[(patient_id, room_id)]
        idx = bisect_right(pref_times, now) - 1
        if idx < 0:
            return None
        return prefs[idx]

    # -------------------------
    # Ventilation mode logic
    # -------------------------
    def _compute_vent_mode(self, room: RoomState) -> Tuple[str, float]:
        """
        mode: "heat" | "cool" | "airflow"
        level: 0.0(off) .. 1.0(on)

        IMPORTANT: We will not log OFF states to DB to avoid extra rows.
        """
        hvac_active = room._hvac_session_start is not None and room._hvac_session_target is not None

        if hvac_active:
            tgt = float(room._hvac_session_target)
            cur = float(room.temperature)

            if cur < tgt - self.cfg.temp_tolerance_c:
                return "heat", 1.0
            if cur > tgt + self.cfg.temp_tolerance_c:
                return "cool", 1.0

            # In tolerance -> no active heat/cool
            # If airflow requested, report airflow on, else off.
            if room.airflow_requested:
                return "airflow", 1.0
            return "airflow", 0.0

        # No heat/cool; airflow can still be on
        if room.airflow_requested:
            return "airflow", 1.0

        return "airflow", 0.0

    def _log_ventilation_if_changed(self, room: RoomState, *, when: datetime) -> None:
        """
        Only log meaningful transitions:
          - heat/cool
          - airflow ON
        Do NOT log OFF rows (airflow level 0) and do NOT log baseline.
        """
        if not self.cfg.enable_utility_usage:
            return
        when = _as_utc(when)
        mode, level = self._compute_vent_mode(room)

        # Skip OFF rows entirely (prevents baseline + "turning off" spam)
        seeing_off = (mode == "airflow" and float(level) == 0.0)
        if seeing_off:
            room._last_vent_mode = mode
            room._last_vent_level = level
            return

        if room._last_vent_mode == mode and room._last_vent_level == level:
            return

        room._last_vent_mode = mode
        room._last_vent_level = level

        with session_scope() as session:
            dev_id = self._get_device_id(session, room_id=room.room_id, device_type="ventilation")
            if dev_id is None:
                return

        self._ventilation_rows.append((int(dev_id), str(mode), float(level), when))
        if len(self._ventilation_rows) >= self._write_batch_size:
            self._flush_model_buffer(Ventilation, "_ventilation_rows")

    # -------------------------
    # Toilet heater state logging
    # -------------------------
    def _log_toilet_heater_if_changed(self, room: RoomState, *, when: datetime, state: bool) -> None:
        if not self.cfg.enable_utility_usage:
            return
        when = _as_utc(when)

        if room._last_toilet_heater_state == state:
            return
        room._last_toilet_heater_state = state

        with session_scope() as session:
            dev_id = self._get_device_id(session, room_id=room.room_id, device_type="toilet_heater")
            if dev_id is None:
                return

        self._toilet_heater_rows.append((int(dev_id), bool(state), when))
        if len(self._toilet_heater_rows) >= self._write_batch_size:
            self._flush_model_buffer(ToiletHeater, "_toilet_heater_rows")

    def _log_toilet_light_if_changed(self, room: RoomState, *, when: datetime, state: bool) -> None:
        if not self.cfg.enable_utility_usage:
            return
        when = _as_utc(when)

        if room._last_toilet_light_state == state:
            return
        room._last_toilet_light_state = state

        with session_scope() as session:
            dev_id = self._get_device_id(session, room_id=room.room_id, device_type="toilet_light")
            if dev_id is None:
                return

        self._toilet_light_rows.append((int(dev_id), bool(state), when))
        if len(self._toilet_light_rows) >= self._write_batch_size:
            self._flush_model_buffer(ToiletLight, "_toilet_light_rows")

    # -------------------------
    # Apply DB preferences + manage sessions
    # -------------------------
    def apply_targets_from_db(self, now: datetime) -> None:
        now = _as_utc(now)
        if self._preloaded_window is None:
            raise RuntimeError("RoomEngine.preload_simulation_window() must be called before apply_targets_from_db().")

        # reset occupancy flags
        for r in self.rooms.values():
            r._occupied_this_tick = False

        for room_id, room in self.rooms.items():
            assignment = self._get_active_assignment(room_id, now)
            if assignment is None:
                continue

            room._occupied_this_tick = True
            room._occupied_until = assignment.end_time

            pref = self._get_latest_preference(
                patient_id=assignment.patient_id,
                room_id=room_id,
                now=now,
            )

            # If no preference, just keep everything OFF but DO NOT log ventilation baseline.
            if pref is None:
                room.airflow_requested = False
                room.toilet_heater_requested = False
                room.toilet_light_requested = False
                self._update_onoff_utilities(room, when=now, airflow=False, toilet_heater=False, toilet_light=False)
                self._log_toilet_heater_if_changed(room, when=now, state=False)
                self._log_toilet_light_if_changed(room, when=now, state=False)
                continue

            pref_ts = pref.timestamp
            is_new_pref = (room._last_pref_ts is None) or (pref_ts > room._last_pref_ts)

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

                    if (
                        room._hvac_session_target is not None
                        and abs(new_target - room._hvac_session_target) < self.cfg.merge_target_delta_c
                    ):
                        room._hvac_session_target = new_target
                    else:
                        if room._hvac_session_start is not None and room._hvac_session_target is not None:
                            self._close_hvac_session(room, end_time=pref_ts)

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

            self._update_onoff_utilities(
                room,
                when=transition_time,
                airflow=airflow_req,
                toilet_heater=toilet_heater_req,
                toilet_light=toilet_light_req,
            )

            self._log_ventilation_if_changed(room, when=transition_time)
            self._log_toilet_heater_if_changed(room, when=transition_time, state=toilet_heater_req)
            self._log_toilet_light_if_changed(room, when=transition_time, state=toilet_light_req)

        # Close sessions on departure
        for room in self.rooms.values():
            if not room._occupied_this_tick and room._occupied_until is not None:
                dep = room._occupied_until
                self._close_all_open_sessions(room, end_time=dep)

                # On departure, turn off requests and log heater OFF
                room.airflow_requested = False
                room.toilet_heater_requested = False
                room.toilet_light_requested = False
                # ventilation OFF is intentionally NOT logged
                self._log_toilet_heater_if_changed(room, when=dep, state=False)
                self._log_toilet_light_if_changed(room, when=dep, state=False)

                room._occupied_until = None

    # -------------------------
    # Simulation step
    # -------------------------
    def step(self, now: datetime, *, step_s: int) -> None:
        now = _as_utc(now)

        for room in self.rooms.values():
            room.step_dynamics()
            self._check_hvac_stabilization(room, now=now)
            if (
                room._toilet_heater_session_start is not None
                and room._toilet_heater_active_until is not None
                and now >= room._toilet_heater_active_until
            ):
                self._close_toilet_heater_session(room, end_time=room._toilet_heater_active_until)
            if (
                room._toilet_light_session_start is not None
                and room._toilet_light_active_until is not None
                and now >= room._toilet_light_active_until
            ):
                self._close_toilet_light_session(room, end_time=room._toilet_light_active_until)

            # If airflow is ON, we still want ventilation rows only on changes,
            # so no need to log here every tick.

    # -------------------------
    # ON/OFF utility sessions (airflow + toilet heater)
    # -------------------------
    def _update_onoff_utilities(self, room: RoomState, *, when: datetime, airflow: bool, toilet_heater: bool, toilet_light: bool) -> None:
        when = _as_utc(when)
        hvac_active = room._hvac_session_start is not None and room._hvac_session_target is not None

        # Airflow is billed only in fan-only mode, not while HVAC is actively heating/cooling.
        if airflow and not hvac_active:
            if room._airflow_session_start is None:
                room._airflow_session_start = when
        else:
            if room._airflow_session_start is not None:
                self._close_airflow_session(room, end_time=when)

        if not toilet_heater:
            room._toilet_heater_active_until = None
            if room._toilet_heater_session_start is not None:
                self._close_toilet_heater_session(room, end_time=when)
        if not toilet_light:
            room._toilet_light_active_until = None
            if room._toilet_light_session_start is not None:
                self._close_toilet_light_session(room, end_time=when)

    # -------------------------
    # HVAC stabilization
    # -------------------------
    def _check_hvac_stabilization(self, room: RoomState, *, now: datetime) -> None:
        if room._hvac_session_start is None or room._hvac_session_target is None:
            return

        # enforce minimum run time
        if (now - room._hvac_session_start).total_seconds() < self.cfg.min_hvac_run_s:
            return

        err = abs(room.temperature - room._hvac_session_target)
        if err <= self.cfg.temp_tolerance_c:
            room._hvac_stable_ticks += 1
        else:
            room._hvac_stable_ticks = 0

        if room._hvac_stable_ticks >= self.cfg.stable_ticks_required:
            self._close_hvac_session(room, end_time=now)
            # Ventilation OFF is not logged; airflow ON would be logged by transitions.

    # -------------------------
    # Session closers -> UtilityUsage
    # -------------------------
    def _close_hvac_session(self, room: RoomState, *, end_time: datetime) -> None:
        start = room._hvac_session_start
        if start is None:
            return

        start = _as_utc(start)
        end_time = _as_utc(end_time)

        hours = max(0.0, (end_time - start).total_seconds() / 3600.0)
        kwh = calculate_energy_usage_kwh(self.cfg.hvac_power_w, hours)

        if self.cfg.enable_utility_usage:
            insert_utility_usage(
                room_id=room.room_id,
                category="hvac",
                start_time=start,
                end_time=end_time,
                power_kwh=kwh,
                water_liters=None,
            )

        room._hvac_session_start = None
        room._hvac_session_target = None
        room._hvac_stable_ticks = 0

    def _close_airflow_session(self, room: RoomState, *, end_time: datetime) -> None:
        start = room._airflow_session_start
        if start is None:
            return

        start = _as_utc(start)
        end_time = _as_utc(end_time)

        hours = max(0.0, (end_time - start).total_seconds() / 3600.0)
        kwh = calculate_energy_usage_kwh(self.cfg.ventilation_power_w, hours)

        # ✅ Separate category so HVAC rows match temperature sessions
        if self.cfg.enable_utility_usage:
            insert_utility_usage(
                room_id=room.room_id,
                category="airflow",
                start_time=start,
                end_time=end_time,
                power_kwh=kwh,
                water_liters=None,
            )

        room._airflow_session_start = None

    def _close_toilet_heater_session(self, room: RoomState, *, end_time: datetime) -> None:
        start = room._toilet_heater_session_start
        if start is None:
            return

        start = _as_utc(start)
        end_time = _as_utc(end_time)

        hours = max(0.0, (end_time - start).total_seconds() / 3600.0)
        kwh = calculate_energy_usage_kwh(self.cfg.toilet_heater_power_w, hours)

        if self.cfg.enable_utility_usage:
            insert_utility_usage(
                room_id=room.room_id,
                category="toilet_heater",
                start_time=start,
                end_time=end_time,
                power_kwh=kwh,
                water_liters=None,
            )

        room._toilet_heater_session_start = None
        room._toilet_heater_active_until = None

    def _close_toilet_light_session(self, room: RoomState, *, end_time: datetime) -> None:
        start = room._toilet_light_session_start
        if start is None:
            return

        start = _as_utc(start)
        end_time = _as_utc(end_time)

        hours = max(0.0, (end_time - start).total_seconds() / 3600.0)
        kwh = calculate_energy_usage_kwh(self.cfg.toilet_light_power_w, hours)

        if self.cfg.enable_utility_usage:
            insert_utility_usage(
                room_id=room.room_id,
                category="toilet_light",
                start_time=start,
                end_time=end_time,
                power_kwh=kwh,
                water_liters=None,
            )

        room._toilet_light_session_start = None
        room._toilet_light_active_until = None

    def _close_all_open_sessions(self, room: RoomState, *, end_time: datetime) -> None:
        end_time = _as_utc(end_time)

        if room._hvac_session_start is not None:
            self._close_hvac_session(room, end_time=end_time)

        if room._airflow_session_start is not None:
            self._close_airflow_session(room, end_time=end_time)

        if room._toilet_heater_session_start is not None:
            self._close_toilet_heater_session(room, end_time=end_time)
        if room._toilet_light_session_start is not None:
            self._close_toilet_light_session(room, end_time=end_time)

    def close_all_sessions(self, end_time: datetime) -> None:
        """
        Force-close any still-open sessions at the end of the simulation horizon.
        Ensures airflow/heater sessions get written even if they never turned off.
        """
        end_time = _as_utc(end_time)
        for room in self.rooms.values():
            self._close_all_open_sessions(room, end_time=end_time)
        self._flush_model_buffer(Ventilation, "_ventilation_rows")
        self._flush_model_buffer(ToiletHeater, "_toilet_heater_rows")
        self._flush_model_buffer(ToiletLight, "_toilet_light_rows")
        flush_utility_usage_writes()
