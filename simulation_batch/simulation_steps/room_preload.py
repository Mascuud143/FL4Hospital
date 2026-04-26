from __future__ import annotations

# Load room data before simulation.
# - Reads device ids
# - Reads room assignments
# - Reads comfort preferences
# - Builds room preload data - RoomPreloader

from persistence.database import session_scope
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.device import Device as DeviceModel
from persistence.models.room_assignment import RoomAssignment
from simulation_batch.simulation_steps.room_cache import AssignmentWindow, PreferenceSnapshot, RoomPreloadCache


class RoomPreloader:
    def __init__(self, *, preload_batch_size: int, as_utc) -> None:
        self.preload_batch_size = preload_batch_size
        self._as_utc = as_utc

    def preload(self, *, room_ids: tuple[int, ...], start_time, end_time) -> RoomPreloadCache:
        cache = RoomPreloadCache()
        cache.assignments_by_room = {room_id: [] for room_id in room_ids}
        cache.assignment_start_times = {room_id: [] for room_id in room_ids}
        with session_scope() as session:
            device_rows = (
                session.query(DeviceModel.room_id, DeviceModel.device_type, DeviceModel.device_id)
                .filter(DeviceModel.room_id.in_(room_ids))
                .yield_per(self.preload_batch_size)
            )
            for room_id, device_type, device_id in device_rows:
                if room_id is None or device_type is None or device_id is None:
                    continue
                cache.device_ids[(int(room_id), str(device_type))] = int(device_id)

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
                .yield_per(self.preload_batch_size)
            )
            pref_keys: set[tuple[int, int]] = set()
            for room_id, patient_id, assignment_start, assignment_end in assignment_rows:
                if room_id is None or patient_id is None or assignment_start is None or assignment_end is None:
                    continue
                room_key = int(room_id)
                start_utc = self._as_utc(assignment_start)
                end_utc = self._as_utc(assignment_end)
                cache.assignments_by_room.setdefault(room_key, []).append(
                    AssignmentWindow(patient_id=int(patient_id), start_time=start_utc, end_time=end_utc)
                )
                cache.assignment_start_times.setdefault(room_key, []).append(start_utc)
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
                    .yield_per(self.preload_batch_size)
                )
                for patient_id, room_id, pref_ts, temperature_main, light_intensity, sound_level, airflow, temperature_toilet in pref_rows:
                    if patient_id is None or room_id is None or pref_ts is None:
                        continue
                    key = (int(patient_id), int(room_id))
                    if key not in pref_keys:
                        continue
                    cache.preferences_by_key.setdefault(key, []).append(
                        PreferenceSnapshot(
                            timestamp=self._as_utc(pref_ts),
                            temperature_main=float(temperature_main) if temperature_main is not None else None,
                            light_intensity=float(light_intensity) if light_intensity is not None else None,
                            sound_level=float(sound_level) if sound_level is not None else None,
                            airflow=bool(airflow),
                            toilet_heater_requested=temperature_toilet is not None,
                        )
                    )
                    cache.preference_times.setdefault(key, []).append(self._as_utc(pref_ts))
        return cache
