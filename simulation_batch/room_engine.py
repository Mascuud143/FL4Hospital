from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference


def _alpha(dt_s: float, tau_s: float) -> float:
    if tau_s <= 0:
        return 1.0
    return 1.0 - math.exp(-dt_s / tau_s)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class MainZone:
    temperature: float = 21.5
    humidity: float = 45.0
    co2: float = 600.0
    light: float = 120.0
    sound: float = 30.0

    target_temperature: float = 22.0
    target_light: float = 120.0
    target_sound: float = 30.0

    airflow_requested: bool = False


@dataclass
class ToiletZone:
    temperature: float = 20.0
    target_temperature: float = 21.0


@dataclass
class RoomState:
    main: MainZone = field(default_factory=MainZone)
    toilet: ToiletZone = field(default_factory=ToiletZone)


class RoomEngine:
    """
    Updates hidden room state toward targets derived from latest ComfortPreference.
    """

    def __init__(
        self,
        *,
        step_s: int,
        # time constants
        tau_temp_main_s: float = 20 * 60,
        tau_temp_toilet_s: float = 45 * 60,
        tau_light_s: float = 2 * 60,
        tau_sound_s: float = 60,
        tau_humidity_s: float = 25 * 60,
        tau_co2_s: float = 15 * 60,
        # outside & losses
        outside_temp_c: float = 15.0,
        loss_main: float = 0.002,
        loss_toilet: float = 0.007,
        # co2 & humidity behavior
        co2_gen_ppm_per_min: float = 3.0,
        co2_decay_ppm_per_min: float = 6.0,
        airflow_extra_decay_ppm_per_min: float = 14.0,
        # noise
        noise_temp: float = 0.05,
        noise_humidity: float = 0.25,
        noise_co2: float = 8.0,
        noise_light: float = 2.0,
        noise_sound: float = 0.6,
        seed: int = 123,
    ):
        self.step_s = int(step_s)

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

        self.noise_temp = noise_temp
        self.noise_humidity = noise_humidity
        self.noise_co2 = noise_co2
        self.noise_light = noise_light
        self.noise_sound = noise_sound

        self.rng = random.Random(seed)

        # room_id -> RoomState
        self.rooms: Dict[int, RoomState] = {}

    def ensure_room(self, room_id: int) -> None:
        self.rooms.setdefault(room_id, RoomState())

    def load_active_rooms(self, now: datetime) -> None:
        """Ensure states exist for rooms that are active at 'now'."""
        now = now.astimezone(timezone.utc)
        with session_scope() as session:
            active = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.start_time <= now)
                .filter(RoomAssignment.end_time > now)
                .all()
            )
            for a in active:
                self.ensure_room(a.room_id)

    def apply_targets_from_db(self, now: datetime) -> Dict[int, bool]:
        """
        Pull latest ComfortPreference per active assignment and apply to room states.
        Returns occupancy map {room_id: True/False} based on active assignments.
        """
        now = now.astimezone(timezone.utc)
        occupancy: Dict[int, bool] = {}

        with session_scope() as session:
            active = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.start_time <= now)
                .filter(RoomAssignment.end_time > now)
                .all()
            )

            for a in active:
                occupancy[a.room_id] = True
                self.ensure_room(a.room_id)

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

                rs = self.rooms[a.room_id]
                rs.main.target_temperature = float(pref.temperature_main)
                if pref.temperature_toilet is not None:
                    rs.toilet.target_temperature = float(pref.temperature_toilet)

                if pref.light_intensity is not None:
                    rs.main.target_light = float(pref.light_intensity)
                if pref.sound_level is not None:
                    rs.main.target_sound = float(pref.sound_level)

                rs.main.airflow_requested = bool(getattr(pref, "airflow", False))

        return occupancy

    def step(self, *, occupancy: Dict[int, bool]) -> None:
        dt_s = float(self.step_s)
        dt_min = dt_s / 60.0

        aT_main = _alpha(dt_s, self.tau_temp_main_s)
        aT_toilet = _alpha(dt_s, self.tau_temp_toilet_s)
        aL = _alpha(dt_s, self.tau_light_s)
        aS = _alpha(dt_s, self.tau_sound_s)
        aH = _alpha(dt_s, self.tau_humidity_s)
        aC = _alpha(dt_s, self.tau_co2_s)

        for room_id, rs in self.rooms.items():
            occ = occupancy.get(room_id, False)

            # main temp
            rs.main.temperature += aT_main * (rs.main.target_temperature - rs.main.temperature)
            rs.main.temperature += -self.loss_main * (rs.main.temperature - self.outside_temp_c)
            rs.main.temperature += self.rng.gauss(0.0, self.noise_temp)

            # toilet temp
            rs.toilet.temperature += aT_toilet * (rs.toilet.target_temperature - rs.toilet.temperature)
            rs.toilet.temperature += -self.loss_toilet * (rs.toilet.temperature - self.outside_temp_c)
            rs.toilet.temperature += self.rng.gauss(0.0, self.noise_temp)

            # light & sound
            rs.main.light += aL * (rs.main.target_light - rs.main.light) + self.rng.gauss(0.0, self.noise_light)
            rs.main.sound += aS * (rs.main.target_sound - rs.main.sound) + self.rng.gauss(0.0, self.noise_sound)

            # humidity: airflow stabilizes toward baseline
            baseline_h = 45.0
            h_target = baseline_h
            rs.main.humidity += aH * (h_target - rs.main.humidity) + self.rng.gauss(0.0, self.noise_humidity)

            # CO2: occupancy generates, airflow removes faster
            co2_next = rs.main.co2
            if occ:
                co2_next += 3.0 * dt_min
            else:
                co2_next -= 6.0 * dt_min
            if rs.main.airflow_requested:
                co2_next -= 14.0 * dt_min

            rs.main.co2 = rs.main.co2 + aC * (co2_next - rs.main.co2) + self.rng.gauss(0.0, self.noise_co2)

            # clamps
            rs.main.temperature = _clamp(rs.main.temperature, 15.0, 30.0)
            rs.toilet.temperature = _clamp(rs.toilet.temperature, 10.0, 30.0)
            rs.main.humidity = _clamp(rs.main.humidity, 20.0, 80.0)
            rs.main.co2 = _clamp(rs.main.co2, 350.0, 2500.0)
            rs.main.light = _clamp(rs.main.light, 0.0, 2000.0)
            rs.main.sound = _clamp(rs.main.sound, 0.0, 120.0)
