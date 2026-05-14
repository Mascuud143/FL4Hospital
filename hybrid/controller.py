from datetime import datetime

from .physics_model import step_temperature
from .session_tracker import start_hvac, stop_hvac
from .state_store import get_zone_state
from .ventilation_logic import DEADBAND, HYSTERESIS, decide_mode
from persistence.database import session_scope
from persistence.models import ComfortPreference, Data, Device, HybridZoneState, Sensor, Ventilation


def find_sensor(session, mac: str, uuid: str):
    return (
        session.query(Sensor)
        .join(Sensor.device)
        .filter(Sensor.uuid == uuid)
        .filter(Sensor.device.has(mac_address=mac.upper()))
        .first()
    )


def get_latest_preference(session, room_id: int):
    return (
        session.query(ComfortPreference)
        .filter(ComfortPreference.room_id == room_id)
        .order_by(ComfortPreference.timestamp.desc())
        .first()
    )


def get_vent_device_id(session, room_id: int) -> int | None:
    device = (
        session.query(Device)
        .filter(Device.room_id == room_id, Device.device_type == "ventilation")
        .order_by(Device.device_id.asc())
        .first()
    )
    return device.device_id if device else None


def _zone_location(sensor: Sensor) -> str:
    return sensor.device.location if sensor.device and sensor.device.location else "main"


def _get_or_create_zone_record(session, room_id: int, location: str) -> HybridZoneState:
    record = (
        session.query(HybridZoneState)
        .filter(HybridZoneState.room_id == room_id, HybridZoneState.location == location)
        .first()
    )
    if record is None:
        record = HybridZoneState(room_id=room_id, location=location, hvac_mode="off")
        session.add(record)
        session.flush()
    return record


def _resolve_target_temp(sensor: Sensor, pref: ComfortPreference) -> float:
    if sensor.device and sensor.device.location == "toilet":
        if pref.temperature_toilet is not None:
            return pref.temperature_toilet
        zone_state = get_zone_state(sensor.device.room_id, sensor.device.location)
        if zone_state.virtual_temp is not None:
            return zone_state.virtual_temp
        if zone_state.last_ble_temp is not None:
            return zone_state.last_ble_temp
        return 0.0
    return pref.temperature_main


def _decide_toilet_mode(current_temp: float, target: float, current_mode: str) -> str:
    error = current_temp - target
    stop_threshold = DEADBAND
    start_threshold = DEADBAND + HYSTERESIS

    if current_mode == "heat" and abs(error) <= start_threshold:
        return "off"
    if abs(error) <= stop_threshold:
        return "off"
    if error < -start_threshold:
        return "heat"
    return "off"


def _apply_temperature_control(
    session,
    sensor: Sensor,
    ble_temp: float,
    timestamp: datetime,
    pref: ComfortPreference,
) -> str:
    room_id = sensor.device.room_id if sensor.device else None
    if room_id is None:
        raise ValueError("Sensor has no room assignment.")

    location = _zone_location(sensor)
    zone_state = get_zone_state(room_id, location)
    zone_record = _get_or_create_zone_record(session, room_id, location)
    if zone_state.virtual_temp is None and zone_record.virtual_temp is not None:
        zone_state.virtual_temp = zone_record.virtual_temp
    if zone_state.last_timestamp is None and zone_record.last_timestamp is not None:
        zone_state.last_timestamp = zone_record.last_timestamp.replace(tzinfo=None) if getattr(zone_record.last_timestamp, "tzinfo", None) else zone_record.last_timestamp
    if zone_state.last_ble_temp is None and zone_record.last_ble_temp is not None:
        zone_state.last_ble_temp = zone_record.last_ble_temp
    if zone_record.hvac_mode:
        zone_state.hvac_mode = zone_record.hvac_mode
    target_temp = _resolve_target_temp(sensor, pref)

    if zone_state.virtual_temp is None:
        zone_state.virtual_temp = ble_temp
        zone_state.last_timestamp = timestamp
        zone_state.last_ble_temp = ble_temp

    if zone_state.last_timestamp is None:
        new_temp = ble_temp
    else:
        active_mode = zone_state.hvac_mode
        if location == "toilet" and active_mode not in {"heat", "off"}:
            active_mode = "off"
        new_temp = step_temperature(
            now=timestamp,
            last_time=zone_state.last_timestamp,
            current_temp=zone_state.virtual_temp if zone_state.virtual_temp is not None else ble_temp,
            ble_temp=ble_temp,
            target_temp=target_temp,
            mode=active_mode,
        )

    if location == "toilet":
        new_mode = _decide_toilet_mode(new_temp, target_temp, zone_state.hvac_mode)
    else:
        new_mode = decide_mode(new_temp, target_temp, pref.airflow, current_mode=zone_state.hvac_mode)
        vent_device_id = get_vent_device_id(session, room_id)

        if new_mode != zone_state.hvac_mode:
            if zone_state.hvac_mode in ("heat", "cool"):
                stop_hvac(session, room_id)

            if vent_device_id is not None:
                session.add(Ventilation(device_id=vent_device_id, mode=new_mode, timestamp=timestamp))

            if new_mode in ("heat", "cool") and vent_device_id is not None:
                start_hvac(session, room_id, vent_device_id)

    zone_state.hvac_mode = new_mode
    zone_state.virtual_temp = new_temp
    zone_state.last_timestamp = timestamp
    zone_state.last_ble_temp = ble_temp
    zone_record.hvac_mode = new_mode
    zone_record.virtual_temp = new_temp
    zone_record.last_timestamp = timestamp
    zone_record.last_ble_temp = ble_temp

    return new_mode


def evaluate_room_control(session, room_id: int, pref: ComfortPreference, location: str | None = None) -> None:
    sensors = (
        session.query(Sensor)
        .join(Sensor.device)
        .filter(
            Device.room_id == room_id,
            Sensor.sensor_type == "temperature",
            *( [Device.location == location] if location in {"main", "toilet"} else [] ),
        )
        .all()
    )
    for sensor in sensors:
        latest_data = (
            session.query(Data)
            .filter(Data.sensor_id == sensor.sensor_id)
            .order_by(Data.timestamp.desc(), Data.data_id.desc())
            .first()
        )
        if latest_data is None or latest_data.value is None:
            continue
        timestamp = latest_data.timestamp
        if timestamp is None:
            timestamp = datetime.utcnow()
        elif getattr(timestamp, "tzinfo", None) is not None:
            timestamp = timestamp.astimezone().replace(tzinfo=None)
        _apply_temperature_control(
            session=session,
            sensor=sensor,
            ble_temp=float(latest_data.value),
            timestamp=timestamp,
            pref=pref,
        )


async def process_ble_event(event: dict):
    if event.get("sensor_type") != "temperature":
        return

    ble_temp = float(event["value"])
    timestamp = datetime.fromisoformat(event["timestamp"]).astimezone().replace(tzinfo=None)

    with session_scope() as session:
        sensor = find_sensor(session, event["mac"], event["uuid"])
        if sensor is None:
            print("Sensor not found in DB:", event["mac"], event["uuid"])
            return

        room_id = sensor.device.room_id if sensor.device else None
        if room_id is None:
            print("Sensor has no room assignment:", event["mac"], event["uuid"])
            return

        pref = get_latest_preference(session, room_id)

        if pref is None:
            session.add(Data(sensor_id=sensor.sensor_id, value=ble_temp, timestamp=timestamp))
            return
        _apply_temperature_control(
            session=session,
            sensor=sensor,
            ble_temp=ble_temp,
            timestamp=timestamp,
            pref=pref,
        )
