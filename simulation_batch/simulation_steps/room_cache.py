from __future__ import annotations

# Store cached room data.
# - Stores one room stay window - AssignmentWindow
# - Stores one room preference row - PreferenceSnapshot
# - Stores cached room data - RoomPreloadCache
# - Keeps room lookups fast

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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


class RoomPreloadCache:
    def __init__(self) -> None:
        self.device_ids: dict[tuple[int, str], int] = {}
        self.assignments_by_room: dict[int, list[AssignmentWindow]] = {}
        self.assignment_start_times: dict[int, list[datetime]] = {}
        self.preferences_by_key: dict[tuple[int, int], list[PreferenceSnapshot]] = {}
        self.preference_times: dict[tuple[int, int], list[datetime]] = {}

    def get_active_assignment(self, room_id: int, now: datetime) -> Optional[AssignmentWindow]:
        starts = self.assignment_start_times.get(room_id)
        if not starts:
            return None
        idx = bisect_right(starts, now) - 1
        if idx < 0:
            return None
        assignment = self.assignments_by_room[room_id][idx]
        if assignment.start_time <= now < assignment.end_time:
            return assignment
        return None

    def get_latest_preference(self, *, patient_id: int, room_id: int, now: datetime) -> Optional[PreferenceSnapshot]:
        prefs = self.preferences_by_key.get((patient_id, room_id))
        if not prefs:
            return None
        pref_times = self.preference_times[(patient_id, room_id)]
        idx = bisect_right(pref_times, now) - 1
        if idx < 0:
            return None
        return prefs[idx]

    def get_device_id(self, *, room_id: int, device_type: str) -> Optional[int]:
        return self.device_ids.get((room_id, device_type))
