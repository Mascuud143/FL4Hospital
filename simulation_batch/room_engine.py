from __future__ import annotations

from datetime import datetime, timezone 
from typing import Dict

from persistence.database import session_scope
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.room_assignment import RoomAssignment


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

    def step(self):
        self.temperature += (self.target_temperature - self.temperature) * 0.05
        self.light += (self.target_light - self.light) * 0.1
        self.sound += (self.target_sound - self.sound) * 0.1

        self.co2 += 5.0
        self.humidity += (45.0 - self.humidity) * 0.01


class RoomEngine:
    def __init__(self, rooms: Dict[int, RoomState]):
        self.rooms = rooms

    def apply_targets_from_db(self, now: datetime):
        now = now.astimezone(timezone.utc)

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

                if pref.temperature_main is not None:
                    room.target_temperature = float(pref.temperature_main)
                if pref.light_intensity is not None:
                    room.target_light = float(pref.light_intensity)
                if pref.sound_level is not None:
                    room.target_sound = float(pref.sound_level)

                room.airflow_requested = bool(getattr(pref, "airflow", False))


    def step(self):
        for room in self.rooms.values():
            room.step()