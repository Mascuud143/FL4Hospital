from __future__ import annotations

# Store room state.
# - Stores initial room temperature
# - Stores initial room humidity
# - Stores initial room light and sound
# - Stores one room state - RoomState
# - Updates room values each step


class RoomState:
    def __init__(self, room_id: int):
        self.room_id = room_id
        self.temperature = 21.0
        self.humidity = 45.0
        self.co2 = 600.0
        self.light = 100.0
        self.sound = 30.0
        self.target_temperature = 21.0
        self.target_light = 100.0
        self.target_sound = 30.0
        self.airflow_requested = False
        self.toilet_heater_requested = False
        self.toilet_light_requested = False
        self._occupied_this_tick = False
        self._occupied_until = None
        self._last_pref_ts = None
        self._hvac_session_start = None
        self._hvac_session_target = None
        self._hvac_stable_ticks = 0
        self._airflow_session_start = None
        self._toilet_heater_session_start = None
        self._toilet_heater_active_until = None
        self._last_toilet_heater_pref_ts = None
        self._toilet_light_session_start = None
        self._toilet_light_active_until = None
        self._last_toilet_light_pref_ts = None
        self._last_vent_mode = None
        self._last_vent_level = None
        self._last_toilet_heater_state = None
        self._last_toilet_light_state = None

    def step_dynamics(self) -> None:
        self.temperature += (self.target_temperature - self.temperature) * 0.05
        self.light += (self.target_light - self.light) * 0.1
        self.sound += (self.target_sound - self.sound) * 0.1
        self.co2 += 5.0
        self.humidity += (45.0 - self.humidity) * 0.01
