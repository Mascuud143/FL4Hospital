from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.device import Device as DeviceModel
from persistence.models.sensor import Sensor as SensorModel


# -------------------------
# Helpers
# -------------------------

def _alpha(dt_s: float, tau_s: float) -> float:
    """First-order lag coefficient."""
    if tau_s <= 0:
        return 1.0
    return 1.0 - math.exp(-dt_s / tau_s)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -------------------------
# Environment state (hidden truth)
# -------------------------

@dataclass
class MainZoneState:
    temperature: float = 21.5
    humidity: float = 45.0
    co2: float = 600.0
    light: float = 120.0
    sound: float = 30.0

    target_temperature: float = 22.0
    target_light: float = 120.0
    target_sound: float = 30.0

    airflow_requested: bool = False  # from ComfortPreference.airflow


@dataclass
class ToiletZoneState:
    temperature: float = 20.0
    target_temperature: float = 21.0


@dataclass
class RoomState:
    main: MainZoneState = field(default_factory=MainZoneState)
    toilet: ToiletZoneState = field(default_factory=ToiletZoneState)


# -------------------------
# Simulator
# -------------------------

class SimulatedBLEManager:
    """
    Fast, bounded simulation that emits sensor events compatible with db_sink.

    Sensor set:
      - Main zone: temperature, humidity, co2, light, sound
      - Toilet zone: temperature only

    Device resolution:
      - Emits device_id (so MAC is not required)
      - Includes uuid + sensor_type for exact sensor matching
    """

    def __init__(
        self,
        *,
        on_event,
        start_time: datetime,
        end_time: datetime,
        step_s: int = 60,            # simulated seconds per tick
        wall_sleep_s: float = 0.0,   # 0 = run as fast as possible

        # --- Dynamics time constants ---
        tau_temp_main_s: float = 20 * 60,
        tau_temp_toilet_s: float = 45 * 60,   # slower toilet heating

        tau_light_s: float = 2 * 60,
        tau_sound_s: float = 60,
        tau_humidity_s: float = 25 * 60,
        tau_co2_s: float = 15 * 60,

        # --- Outside / loss terms ---
        outside_temp_c: float = 15.0,
        loss_main: float = 0.002,    # heat loss strength (main)
        loss_toilet: float = 0.007,  # stronger loss (toilet cold airflow)

        # --- Occupancy / air quality ---
        co2_gen_ppm_per_min: float = 3.0,      # occupancy CO2 source (per min)
        co2_decay_ppm_per_min: float = 6.0,    # baseline decay when unoccupied
        airflow_extra_decay_ppm_per_min: float = 14.0,  # extra decay when airflow requested

        humidity_drift_per_min: float = 0.02,  # tiny drift
        airflow_humidity_pull: float = 0.08,   # airflow nudges humidity toward baseline

        # --- Noise ---
        noise_temp: float = 0.05,
        noise_humidity: float = 0.25,
        noise_co2: float = 8.0,
        noise_light: float = 2.0,
        noise_sound: float = 0.6,
    ):
        self.on_event = on_event

        self.start_time = start_time.astimezone(timezone.utc)
        self.end_time = end_time.astimezone(timezone.utc)
        self.step_s = int(step_s)
        self.wall_sleep_s = float(wall_sleep_s)

        self.tau_temp_main_s = tau_temp_main_s
        self.tau_temp_toilet_s = tau_temp_toilet_s
        self.tau_light_s = tau_light_s
        self.tau_sound_s = tau_sound_s
        self.tau_humidity_s = tau_humidity_s
        self.tau_co2_s = tau_co2_s

        self.outside_temp_c = outside_temp_c
        self.loss_main = loss_main
        self.loss_toilet = loss_toilet

        self.co2_gen_ppm_per_min = co2_gen_ppm_per_min
        self.co2_decay_ppm_per_min = co2_decay_ppm_per_min
        self.airflow_extra_decay_ppm_per_min = airflow_extra_decay_ppm_per_min

        self.humidity_drift_per_min = humidity_drift_per_min
        self.airflow_humidity_pull = airflow_humidity_pull

        self.noise_temp = noise_temp
        self.noise_humidity = noise_humidity
        self.noise_co2 = noise_co2
        self.noise_light = noise_light
        self.noise_sound = noise_sound

        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

        # room_id -> RoomState
        self._rooms: Dict[int, RoomState] = {}

        # Sensor emit map built from DB:
        # each item: {device_id, room_id, location, device_type, sensor_type, uuid, unit}
        self._sensor_emit: List[Dict[str, Any]] = []

    # -------------------------
    # Lifecycle
    # -------------------------

    async def start(self):
        self._stop.clear()
        self._build_emit_map_from_db()
        self._init_room_states()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    # -------------------------
    # DB discovery
    # -------------------------

    def _build_emit_map_from_db(self) -> None:
        """
        Pull sensor devices + sensors from DB so the simulator matches what was seeded.

        Assumptions:
          - Sensor devices have Device.device_type == "sensor"
          - Device.location is "main" or "toilet"
          - Sensor.sensor_type is one of: temperature, humidity, co2, light, sound
        """
        self._sensor_emit.clear()
        self._rooms.clear()

        with session_scope() as session:
            sensor_devices = (
                session.query(DeviceModel)
                .filter(DeviceModel.device_type == "sensor")
                .all()
            )

            for dev in sensor_devices:
                if dev.room_id is None:
                    continue

                self._rooms.setdefault(dev.room_id, RoomState())

                sensors = (
                    session.query(SensorModel)
                    .filter(SensorModel.device_id == dev.device_id)
                    .all()
                )

                for s in sensors:
                    st = getattr(s, "sensor_type", None)
                    if not st:
                        continue

                    loc = (dev.location or "main").lower()
                    st = st.lower()

                    # Enforce your rule set:
                    if loc == "main":
                        if st not in {"temperature", "humidity", "co2", "light", "sound"}:
                            continue
                    elif loc == "toilet":
                        if st != "temperature":
                            continue
                    else:
                        # unknown location, skip
                        continue

                    self._sensor_emit.append(
                        {
                            "device_id": dev.device_id,
                            "room_id": dev.room_id,
                            "location": loc,
                            "device_type": dev.device_type,
                            "sensor_type": st,
                            "uuid": getattr(s, "uuid", None),
                            "unit": getattr(s, "unit", None) or "",
                        }
                    )

    def _init_room_states(self) -> None:
        for room_id, rs in self._rooms.items():
            rs.main.temperature = 21.5 + random.uniform(-0.7, 0.7)
            rs.main.humidity = 45.0 + random.uniform(-5.0, 5.0)
            rs.main.co2 = 600.0 + random.uniform(-80.0, 80.0)
            rs.main.light = 120.0 + random.uniform(-25.0, 25.0)
            rs.main.sound = 30.0 + random.uniform(-5.0, 5.0)

            rs.toilet.temperature = 20.0 + random.uniform(-1.0, 1.0)

            rs.main.target_temperature = 22.0
            rs.toilet.target_temperature = 21.0
            rs.main.target_light = 120.0
            rs.main.target_sound = 30.0
            rs.main.airflow_requested = False

    # -------------------------
    # Main loop (sim clock)
    # -------------------------

    async def _loop(self):
        sim_time = self.start_time

        while not self._stop.is_set() and sim_time <= self.end_time:
            targets = self._get_current_targets(sim_time)
            self._apply_targets(targets)

            self._step_physics()

            await self._emit_events(sim_time)

            sim_time = sim_time + timedelta(seconds=self.step_s)

            if self.wall_sleep_s > 0:
                await asyncio.sleep(self.wall_sleep_s)

    # -------------------------
    # Targets from comfort preferences
    # -------------------------

    def _get_current_targets(self, now: datetime) -> Dict[int, Dict[str, Any]]:
        """
        Returns per-room targets based on active patient assignments and latest comfort preference.
        { room_id: {t_main, t_toilet?, light?, sound?, airflow_bool} }
        """
        out: Dict[int, Dict[str, Any]] = {}

        with session_scope() as session:
            active = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.start_time <= now)
                .filter(RoomAssignment.end_time > now)
                .all()
            )

            for a in active:
                pref = (
                    session.query(ComfortPreference)
                    .filter(ComfortPreference.patient_id == a.patient_id)
                    .filter(ComfortPreference.room_id == a.room_id)
                    .filter(ComfortPreference.timestamp <= now)
                    .order_by(ComfortPreference.timestamp.desc())
                    .first()
                )

                if not pref:
                    continue

                out[a.room_id] = {
                    "t_main": float(pref.temperature_main),
                    "t_toilet": float(pref.temperature_toilet) if pref.temperature_toilet is not None else None,
                    "light": float(pref.light_intensity) if pref.light_intensity is not None else None,
                    "sound": float(pref.sound_level) if pref.sound_level is not None else None,
                    "airflow": bool(getattr(pref, "airflow", False)),
                }

        return out

    def _apply_targets(self, targets: Dict[int, Dict[str, Any]]) -> None:
        for room_id, rs in self._rooms.items():
            t = targets.get(room_id)
            if not t:
                continue

            rs.main.target_temperature = t["t_main"]

            if t["t_toilet"] is not None:
                rs.toilet.target_temperature = t["t_toilet"]

            if t["light"] is not None:
                rs.main.target_light = t["light"]

            if t["sound"] is not None:
                rs.main.target_sound = t["sound"]

            rs.main.airflow_requested = t["airflow"]

    # -------------------------
    # Physics
    # -------------------------

    def _step_physics(self) -> None:
        dt_s = float(self.step_s)
        dt_min = dt_s / 60.0

        aT_main = _alpha(dt_s, self.tau_temp_main_s)
        aT_toilet = _alpha(dt_s, self.tau_temp_toilet_s)

        aL = _alpha(dt_s, self.tau_light_s)
        aS = _alpha(dt_s, self.tau_sound_s)
        aH = _alpha(dt_s, self.tau_humidity_s)
        aC = _alpha(dt_s, self.tau_co2_s)

        for room_id, rs in self._rooms.items():
            # ---------- MAIN temperature ----------
            rs.main.temperature += aT_main * (rs.main.target_temperature - rs.main.temperature)
            rs.main.temperature += -self.loss_main * (rs.main.temperature - self.outside_temp_c)
            rs.main.temperature += random.gauss(0.0, self.noise_temp)

            # ---------- TOILET temperature (slower + more loss) ----------
            rs.toilet.temperature += aT_toilet * (rs.toilet.target_temperature - rs.toilet.temperature)
            rs.toilet.temperature += -self.loss_toilet * (rs.toilet.temperature - self.outside_temp_c)
            rs.toilet.temperature += random.gauss(0.0, self.noise_temp)

            # ---------- MAIN light ----------
            rs.main.light += aL * (rs.main.target_light - rs.main.light)
            rs.main.light += random.gauss(0.0, self.noise_light)

            # ---------- MAIN sound ----------
            rs.main.sound += aS * (rs.main.target_sound - rs.main.sound)
            rs.main.sound += random.gauss(0.0, self.noise_sound)

            # ---------- MAIN humidity ----------
            # drift slowly + airflow pulls it toward baseline (45%)
            baseline_h = 45.0
            humidity_target = baseline_h
            if rs.main.airflow_requested:
                humidity_target = baseline_h  # airflow stabilizes humidity

            rs.main.humidity += aH * (humidity_target - rs.main.humidity)
            rs.main.humidity += (random.gauss(0.0, self.noise_humidity) + self.humidity_drift_per_min * dt_min)

            # ---------- MAIN CO2 ----------
            # Occupancy: assume occupied if assigned (targets exist) -> generate CO2.
            # Airflow: accelerates CO2 decay (fresh air).
            occupied = True  # if the room exists in simulation, assume occupied during assignment-driven targets
            co2 = rs.main.co2

            if occupied:
                co2 += self.co2_gen_ppm_per_min * dt_min
            else:
                co2 -= self.co2_decay_ppm_per_min * dt_min

            if rs.main.airflow_requested:
                co2 -= self.airflow_extra_decay_ppm_per_min * dt_min

            # Smooth with lag and noise
            rs.main.co2 = rs.main.co2 + aC * (co2 - rs.main.co2) + random.gauss(0.0, self.noise_co2)

            # ---------- clamps ----------
            rs.main.temperature = _clamp(rs.main.temperature, 15.0, 30.0)
            rs.toilet.temperature = _clamp(rs.toilet.temperature, 10.0, 30.0)

            rs.main.light = _clamp(rs.main.light, 0.0, 2000.0)
            rs.main.sound = _clamp(rs.main.sound, 0.0, 120.0)

            rs.main.humidity = _clamp(rs.main.humidity, 20.0, 80.0)
            rs.main.co2 = _clamp(rs.main.co2, 350.0, 2500.0)

    # -------------------------
    # Emit events
    # -------------------------

    async def _emit_events(self, now: datetime) -> None:
        ts = now.isoformat()

        for s in self._sensor_emit:
            room_id = s["room_id"]
            if room_id not in self._rooms:
                continue

            rs = self._rooms[room_id]
            loc = s["location"]
            st = s["sensor_type"]

            # compute "true" value from the hidden room state
            if loc == "toilet":
                # toilet: temperature only (enforced in mapping)
                value = round(rs.toilet.temperature + random.gauss(0.0, self.noise_temp), 2)
            else:
                # main room sensors
                if st == "temperature":
                    value = round(rs.main.temperature + random.gauss(0.0, self.noise_temp), 2)
                elif st == "humidity":
                    value = round(rs.main.humidity + random.gauss(0.0, self.noise_humidity), 1)
                elif st == "co2":
                    value = round(rs.main.co2 + random.gauss(0.0, self.noise_co2), 0)
                elif st == "light":
                    value = round(rs.main.light + random.gauss(0.0, self.noise_light), 1)
                elif st == "sound":
                    value = round(rs.main.sound + random.gauss(0.0, self.noise_sound), 1)
                else:
                    continue

            event = {
                "timestamp": ts,
                "device_id": s["device_id"],     # ✅ db_sink can resolve by device_id
                "room_id": room_id,
                "location": loc,
                "device_type": s["device_type"],
                "sensor_type": st,
                "unit": s["unit"],
                "uuid": s["uuid"],
                "value": value,
                "raw_hex": None,
                "error": None,
            }

            await self.on_event(event)
