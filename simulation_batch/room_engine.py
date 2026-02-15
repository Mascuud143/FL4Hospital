from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from persistence.database import session_scope
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.room_assignment import RoomAssignment

from persistence.models.device import Device as DeviceModel
from persistence.models.ventilation import Ventilation
from persistence.models.toilet_heater import ToiletHeater

from domain.utility_calculator import calculate_energy_usage_kwh
from simulation_batch.utility_usage_writer import insert_utility_usage


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

    # Stabilization logic
    temp_tolerance_c: float = 0.4          # wider tolerance prevents "runs forever"
    stable_ticks_required: int = 3         # fewer ticks to declare stable
    min_hvac_run_s: int = 5 * 60           # don't close HVAC before 5 minutes

    # Merge tiny target changes into existing session
    merge_target_delta_c: float = 0.30     # if new target within this delta, don't restart session


class RoomState:
    def __init__(self, room_id: int):
        self.room_id = room_id

        # Current state
        self.temperature = 0
        self.humidity = 45.0
        self.co2 = 600.0
        self.light = 100.0
        self.sound = 30.0

        # Targets
        self.target_temperature = 0
        self.target_light = 100.0
        self.target_sound = 30.0

        # Requests derived from preferences
        self.airflow_requested: bool = False
        self.toilet_heater_requested: bool = False

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

        # Avoid duplicate actuator history inserts
        self._last_vent_mode: Optional[str] = None
        self._last_vent_level: Optional[float] = None
        self._last_toilet_heater_state: Optional[bool] = None

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

    # -------------------------
    # DB helpers
    # -------------------------
    def _get_device_id(self, session, *, room_id: int, device_type: str) -> Optional[int]:
        d = (
            session.query(DeviceModel)
            .filter(DeviceModel.room_id == room_id)
            .filter(DeviceModel.device_type == device_type)
            .one_or_none()
        )
        return int(d.device_id) if d else None

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
            session.add(Ventilation(device_id=dev_id, mode=mode, level=level, timestamp=when))

    # -------------------------
    # Toilet heater state logging
    # -------------------------
    def _log_toilet_heater_if_changed(self, room: RoomState, *, when: datetime, state: bool) -> None:
        when = _as_utc(when)

        if room._last_toilet_heater_state == state:
            return
        room._last_toilet_heater_state = state

        with session_scope() as session:
            dev_id = self._get_device_id(session, room_id=room.room_id, device_type="toilet_heater")
            if dev_id is None:
                return
            session.add(ToiletHeater(device_id=dev_id, state=state, timestamp=when))

    # -------------------------
    # Apply DB preferences + manage sessions
    # -------------------------
    def apply_targets_from_db(self, now: datetime) -> None:
        now = _as_utc(now)

        # reset occupancy flags
        for r in self.rooms.values():
            r._occupied_this_tick = False

        with session_scope() as session:
            active = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.start_time <= now)
                .filter(RoomAssignment.end_time > now)
                .all()
            )

            for a in active:
                room = self.rooms.get(a.room_id)
                if not room:
                    continue

                room._occupied_this_tick = True
                room._occupied_until = _as_utc(a.end_time)

                # latest preference
                pref = (
                    session.query(ComfortPreference)
                    .filter(ComfortPreference.patient_id == a.patient_id)
                    .filter(ComfortPreference.room_id == a.room_id)
                    .filter(ComfortPreference.timestamp <= now)
                    .order_by(ComfortPreference.timestamp.desc())
                    .first()
                )

                # If no preference, just keep everything OFF but DO NOT log ventilation baseline.
                if not pref:
                    room.airflow_requested = False
                    room.toilet_heater_requested = False
                    self._update_onoff_utilities(room, when=now, airflow=False, toilet_heater=False)
                    # do not log ventilation here
                    self._log_toilet_heater_if_changed(room, when=now, state=False)
                    continue

                pref_ts = _as_utc(pref.timestamp)
                is_new_pref = (room._last_pref_ts is None) or (pref_ts > room._last_pref_ts)

                # Apply continuous targets
                if pref.temperature_main is not None:
                    room.target_temperature = float(pref.temperature_main)
                if pref.light_intensity is not None:
                    room.target_light = float(pref.light_intensity)
                if pref.sound_level is not None:
                    room.target_sound = float(pref.sound_level)

                airflow_req = bool(getattr(pref, "airflow", False))
                toilet_heater_req = pref.temperature_toilet is not None

                room.airflow_requested = airflow_req
                room.toilet_heater_requested = toilet_heater_req

                # If a preference just became active, treat that pref_ts as transition moment
                transition_time = pref_ts if is_new_pref else now

                if is_new_pref:
                    room._last_pref_ts = pref_ts

                    # HVAC session logic only when temperature_main is set
                    if pref.temperature_main is not None:
                        new_target = float(pref.temperature_main)

                        # If a session exists and targets are very close: MERGE (avoid restart)
                        if (
                            room._hvac_session_target is not None
                            and abs(new_target - room._hvac_session_target) < self.cfg.merge_target_delta_c
                        ):
                            room._hvac_session_target = new_target
                            # don't restart; keep current start time
                        else:
                            # If there is an active session and target changes meaningfully -> close early
                            if room._hvac_session_start is not None and room._hvac_session_target is not None:
                                self._close_hvac_session(room, end_time=pref_ts)

                            # Start new session
                            room._hvac_session_start = pref_ts
                            room._hvac_session_target = new_target
                            room._hvac_stable_ticks = 0

                    # If temperature_main is None: Option A (do nothing)

                # ON/OFF energy sessions (airflow + toilet heater)
                self._update_onoff_utilities(
                    room,
                    when=transition_time,
                    airflow=airflow_req,
                    toilet_heater=toilet_heater_req,
                )

                # Log meaningful ventilation transitions ONLY (no baseline/off)
                self._log_ventilation_if_changed(room, when=transition_time)

                # Log toilet heater ON/OFF history
                self._log_toilet_heater_if_changed(room, when=transition_time, state=toilet_heater_req)

        # Close sessions on departure
        for room in self.rooms.values():
            if not room._occupied_this_tick and room._occupied_until is not None:
                dep = room._occupied_until
                self._close_all_open_sessions(room, end_time=dep)

                # On departure, turn off requests and log heater OFF
                room.airflow_requested = False
                room.toilet_heater_requested = False
                # ventilation OFF is intentionally NOT logged
                self._log_toilet_heater_if_changed(room, when=dep, state=False)

                room._occupied_until = None

    # -------------------------
    # Simulation step
    # -------------------------
    def step(self, now: datetime, *, step_s: int) -> None:
        now = _as_utc(now)

        for room in self.rooms.values():
            room.step_dynamics()
            self._check_hvac_stabilization(room, now=now)

            # If airflow is ON, we still want ventilation rows only on changes,
            # so no need to log here every tick.

    # -------------------------
    # ON/OFF utility sessions (airflow + toilet heater)
    # -------------------------
    def _update_onoff_utilities(self, room: RoomState, *, when: datetime, airflow: bool, toilet_heater: bool) -> None:
        when = _as_utc(when)

        # airflow session (fan-only)
        if airflow:
            if room._airflow_session_start is None:
                room._airflow_session_start = when
        else:
            if room._airflow_session_start is not None:
                self._close_airflow_session(room, end_time=when)

        # toilet heater session
        if toilet_heater:
            if room._toilet_heater_session_start is None:
                room._toilet_heater_session_start = when
        else:
            if room._toilet_heater_session_start is not None:
                self._close_toilet_heater_session(room, end_time=when)

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

        insert_utility_usage(
            room_id=room.room_id,
            category="toilet_heater",
            start_time=start,
            end_time=end_time,
            power_kwh=kwh,
            water_liters=None,
        )

        room._toilet_heater_session_start = None

    def _close_all_open_sessions(self, room: RoomState, *, end_time: datetime) -> None:
        end_time = _as_utc(end_time)

        if room._hvac_session_start is not None:
            self._close_hvac_session(room, end_time=end_time)

        if room._airflow_session_start is not None:
            self._close_airflow_session(room, end_time=end_time)

        if room._toilet_heater_session_start is not None:
            self._close_toilet_heater_session(room, end_time=end_time)

    def close_all_sessions(self, end_time: datetime) -> None:
        """
        Force-close any still-open sessions at the end of the simulation horizon.
        Ensures airflow/heater sessions get written even if they never turned off.
        """
        end_time = _as_utc(end_time)
        for room in self.rooms.values():
            self._close_all_open_sessions(room, end_time=end_time)
