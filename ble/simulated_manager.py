import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from persistence.database import session_scope
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference


def _alpha(dt_s: float, tau_s: float) -> float:
    # stable for dt << tau and dt >> tau
    if tau_s <= 0:
        return 1.0
    return 1.0 - math.exp(-dt_s / tau_s)


@dataclass
class RoomEnvState:
    # current measured values (what sensors will report)
    temperature: float = 22.0
    light: float = 100.0
    sound: float = 30.0

    # last targets (for debugging / optional logging)
    target_temperature: float = 22.0
    target_light: float = 100.0
    target_sound: float = 30.0


class SimulatedBLEManager:
    """
    Simulates BLE notifications. Sensor values drift toward comfort targets over time.
    Emits events shaped exactly like BLEManager.
    """

    def __init__(
        self,
        devices,
        on_event,
        *,
        tick_s: float = 1.0,              # simulation update tick
        emit_every_s: float = 5.0,        # how often to emit sensor readings
        tau_temp_s: float = 15 * 60,      # 15 minutes time constant
        tau_light_s: float = 2 * 60,      # 2 minutes
        tau_sound_s: float = 60,          # 1 minute
        noise_temp: float = 0.03,         # measurement/process noise
        noise_light: float = 1.5,
        noise_sound: float = 0.4,
    ):
        self.devices = devices
        self.on_event = on_event

        self.tick_s = tick_s
        self.emit_every_s = emit_every_s

        self.tau_temp_s = tau_temp_s
        self.tau_light_s = tau_light_s
        self.tau_sound_s = tau_sound_s

        self.noise_temp = noise_temp
        self.noise_light = noise_light
        self.noise_sound = noise_sound

        self._stop = asyncio.Event()
        self._task = None

        # room_id -> RoomEnvState
        self._rooms: dict[int, RoomEnvState] = {}

        # last time we emitted per device (or globally)
        self._emit_accum = 0.0

    async def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self):
        # initialize room states from something plausible
        for d in self.devices:
            if getattr(d, "room_id", None) is None:
                continue
            self._rooms.setdefault(d.room_id, RoomEnvState())

        while not self._stop.is_set():
            now = datetime.now(timezone.utc)

            # 1) update targets for each room from current patient comfort
            room_targets = self._get_current_targets(now)

            # 2) advance room physics toward targets
            self._step_rooms(room_targets)

            # 3) emit readings occasionally
            self._emit_accum += self.tick_s
            if self._emit_accum >= self.emit_every_s:
                self._emit_accum = 0.0
                await self._emit_events(now)

            await asyncio.sleep(self.tick_s)

    def _get_current_targets(self, now: datetime) -> dict[int, dict]:
        """
        Returns: { room_id: {"temperature": x, "light": y, "sound": z} }
        If no patient/comfort is found, keep previous targets.
        """
        targets = {}

        with session_scope() as session:
            # find active assignments at "now"
            # NOTE: adjust field names if yours differ
            active = (
                session.query(RoomAssignment)
                .filter(RoomAssignment.start_time <= now)
                .filter(RoomAssignment.end_time > now)
                .all()
            )

            for a in active:
                room_id = a.room_id
                patient_id = a.patient_id

                # get most recent comfort pref for this patient/room at or before now
                pref = (
                    session.query(ComfortPreference)
                    .filter(ComfortPreference.patient_id == patient_id)
                    .filter(ComfortPreference.room_id == room_id)
                    .filter(ComfortPreference.timestamp <= now)
                    .order_by(ComfortPreference.timestamp.desc())
                    .first()
                )

                if pref:
                    targets[room_id] = {
                        "temperature": float(pref.temperature),
                        "light": float(pref.light_intensity),
                        "sound": float(pref.sound_level),
                    }

        return targets

    def _step_rooms(self, room_targets: dict[int, dict]):
        aT = _alpha(self.tick_s, self.tau_temp_s)
        aL = _alpha(self.tick_s, self.tau_light_s)
        aS = _alpha(self.tick_s, self.tau_sound_s)

        for room_id, state in self._rooms.items():
            # choose targets: new if available else keep last target
            if room_id in room_targets:
                t = room_targets[room_id]
                state.target_temperature = t["temperature"]
                state.target_light = t["light"]
                state.target_sound = t["sound"]

            # move toward target (slowly), add noise
            state.temperature += aT * (state.target_temperature - state.temperature) + random.gauss(0, self.noise_temp)
            state.light += aL * (state.target_light - state.light) + random.gauss(0, self.noise_light)
            state.sound += aS * (state.target_sound - state.sound) + random.gauss(0, self.noise_sound)

            # clamp to sensible ranges (optional but helps)
            state.temperature = max(15.0, min(30.0, state.temperature))
            state.light = max(0.0, min(2000.0, state.light))
            state.sound = max(0.0, min(120.0, state.sound))

    async def _emit_events(self, now: datetime):
        ts = now.isoformat()

        for device in self.devices:
            mac = device.mac_address.upper()
            room_id = getattr(device, "room_id", None)
            if room_id is None or room_id not in self._rooms:
                continue

            env = self._rooms[room_id]

            for sensor in device.sensors:
                st = sensor.sensor_type

                if st == "temperature":
                    value = round(env.temperature, 2)
                elif st == "light":
                    value = round(env.light, 1)
                elif st == "sound":
                    value = round(env.sound, 1)
                else:
                    # if you simulate more later (humidity/co2), add them here
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

