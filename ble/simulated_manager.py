import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference


# -------------------------
# Helpers
# -------------------------

def _alpha(dt_s: float, tau_s: float) -> float:
    """First-order lag coefficient."""
    if tau_s <= 0:
        return 1.0
    return 1.0 - math.exp(-dt_s / tau_s)


# -------------------------
# Environment state
# -------------------------

@dataclass
class RoomEnvState:
    # current measured values
    temperature: float = 22.0
    humidity: float = 45.0
    co2: float = 600.0
    light: float = 100.0
    sound: float = 30.0

    # targets
    target_temperature: float = 22.0
    target_light: float = 100.0
    target_sound: float = 30.0


# -------------------------
# Simulated BLE Manager
# -------------------------

class SimulatedBLEManager:
    """
    Simulates BLE notifications.
    Sensor values drift slowly toward comfort targets.
    Emits events compatible with db_sink.
    """

    def __init__(
        self,
        devices,
        on_event,
        *,
        tick_s: float = 1.0,
        emit_every_s: float = 5.0,

        # time constants
        tau_temp_s: float = 15 * 60,
        tau_humidity_s: float = 20 * 60,
        tau_co2_s: float = 10 * 60,
        tau_light_s: float = 2 * 60,
        tau_sound_s: float = 60,

        # noise
        noise_temp: float = 0.03,
        noise_humidity: float = 0.2,
        noise_co2: float = 5.0,
        noise_light: float = 1.5,
        noise_sound: float = 0.4,
    ):
        self.devices = devices
        self.on_event = on_event

        self.tick_s = tick_s
        self.emit_every_s = emit_every_s

        self.tau_temp_s = tau_temp_s
        self.tau_humidity_s = tau_humidity_s
        self.tau_co2_s = tau_co2_s
        self.tau_light_s = tau_light_s
        self.tau_sound_s = tau_sound_s

        self.noise_temp = noise_temp
        self.noise_humidity = noise_humidity
        self.noise_co2 = noise_co2
        self.noise_light = noise_light
        self.noise_sound = noise_sound

        self._stop = asyncio.Event()
        self._task = None

        # room_id -> RoomEnvState
        self._rooms: dict[int, RoomEnvState] = {}

        self._emit_accum = 0.0

    # -------------------------
    # Lifecycle
    # -------------------------

    async def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    # -------------------------
    # Main loop
    # -------------------------

    async def _loop(self):
        # initialize room states
        for d in self.devices:
            if getattr(d, "room_id", None) is not None:
                self._rooms.setdefault(d.room_id, RoomEnvState())

        while not self._stop.is_set():
            now = datetime.now(timezone.utc)

            room_targets = self._get_current_targets(now)
            self._step_rooms(room_targets)

            self._emit_accum += self.tick_s
            if self._emit_accum >= self.emit_every_s:
                self._emit_accum = 0.0
                await self._emit_events(now)

            await asyncio.sleep(self.tick_s)

    # -------------------------
    # Comfort targets
    # -------------------------

    def _get_current_targets(self, now: datetime) -> dict[int, dict]:
        targets = {}

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

                if pref:
                    targets[a.room_id] = {
                        "temperature": float(pref.temperature),
                        "light": float(pref.light_intensity),
                        "sound": float(pref.sound_level),
                    }

        return targets

    # -------------------------
    # Physics
    # -------------------------

    def _step_rooms(self, room_targets: dict[int, dict]):
        aT = _alpha(self.tick_s, self.tau_temp_s)
        aH = _alpha(self.tick_s, self.tau_humidity_s)
        aC = _alpha(self.tick_s, self.tau_co2_s)
        aL = _alpha(self.tick_s, self.tau_light_s)
        aS = _alpha(self.tick_s, self.tau_sound_s)

        for room_id, state in self._rooms.items():
            if room_id in room_targets:
                t = room_targets[room_id]
                state.target_temperature = t["temperature"]
                state.target_light = t["light"]
                state.target_sound = t["sound"]

            # temperature
            state.temperature += aT * (state.target_temperature - state.temperature) + random.gauss(0, self.noise_temp)

            # humidity (slow ambient drift)
            state.humidity += random.gauss(0, self.noise_humidity)

            # CO2 (people present → slowly increases, otherwise decays)
            occupancy_source = 5.0 if room_id in room_targets else -8.0
            state.co2 += aC * occupancy_source + random.gauss(0, self.noise_co2)

            # light
            state.light += aL * (state.target_light - state.light) + random.gauss(0, self.noise_light)

            # sound
            state.sound += aS * (state.target_sound - state.sound) + random.gauss(0, self.noise_sound)

            # clamps
            state.temperature = max(15.0, min(30.0, state.temperature))
            state.humidity = max(20.0, min(80.0, state.humidity))
            state.co2 = max(350.0, min(2000.0, state.co2))
            state.light = max(0.0, min(2000.0, state.light))
            state.sound = max(0.0, min(120.0, state.sound))

    # -------------------------
    # Emit BLE-style events
    # -------------------------

    async def _emit_events(self, now: datetime):
        ts = now.isoformat()

        for device in self.devices:
            room_id = getattr(device, "room_id", None)
            if room_id not in self._rooms:
                continue

            env = self._rooms[room_id]
            mac = device.mac_address.upper()

            for sensor in device.sensors:
                st = sensor.sensor_type

                if st == "temperature":
                    value = round(env.temperature, 2)
                elif st == "humidity":
                    value = round(env.humidity, 1)
                elif st == "co2":
                    value = round(env.co2, 0)
                elif st == "light":
                    value = round(env.light, 1)
                elif st == "sound":
                    value = round(env.sound, 1)
                else:
                    print(f"[SIM] No generator for sensor type: {st}")
                    continue

                event = {
                    "timestamp": ts,
                    "mac": mac,
                    "device_label": device.label or device.name or mac,
                    "sensor_type": st,
                    "unit": sensor.unit,
                    "uuid": str(sensor.uuid),
                    "value": value,
                    "raw_hex": None,
                    "error": None,
                }

                await self.on_event(event)
