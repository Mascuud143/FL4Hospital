from datetime import datetime
from persistence.database import session_scope
from persistence.models import Data, Sensor, ComfortPreference, Ventilation

from hybrid.state_store import ROOM_STATE
from hybrid.physics_model import step_temperature
from hybrid.ventilation_logic import decide_mode
from hybrid.session_tracker import start_hvac, stop_hvac


ROOM_ID = 1
VENT_DEVICE_ID = 1


def find_sensor(session, mac: str, uuid: str):
    return (
        session.query(Sensor)
        .join(Sensor.device)
        .filter(Sensor.uuid == uuid)
        .filter(Sensor.device.has(mac_address=mac.upper()))
        .first()
    )


def get_latest_preference(session):
    return (
        session.query(ComfortPreference)
        .filter(ComfortPreference.room_id == ROOM_ID)
        .order_by(ComfortPreference.timestamp.desc())
        .first()
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

        pref = get_latest_preference(session)

        # ------------------------------------------------
        # 1) NO COMFORT → SAVE RAW BLE
        # ------------------------------------------------
        if pref is None:
            session.add(
                Data(sensor_id=sensor.sensor_id, value=ble_temp, timestamp=timestamp)
            )
            return

        # ------------------------------------------------
        # 2) FIRST COMFORT EVENT INITIALIZES DIGITAL TWIN
        # ------------------------------------------------
        if ROOM_STATE.virtual_temp is None:
            ROOM_STATE.virtual_temp = ble_temp
            ROOM_STATE.last_timestamp = timestamp
            ROOM_STATE.last_ble_temp = ble_temp

            session.add(
                Data(sensor_id=sensor.sensor_id, value=ble_temp, timestamp=timestamp)
            )
            return

        # ------------------------------------------------
        # 3) PHYSICAL RESPONSE MODEL
        # ------------------------------------------------
        new_temp = step_temperature(
            now=timestamp,
            last_time=ROOM_STATE.last_timestamp,
            current_temp=ROOM_STATE.virtual_temp,
            ble_temp=ble_temp,
            target_temp=pref.temperature_main,
            mode=ROOM_STATE.hvac_mode,
        )

        new_mode = decide_mode(new_temp, pref.temperature_main, pref.airflow)

        if new_mode != ROOM_STATE.hvac_mode:

        # always close previous running session first
            if ROOM_STATE.hvac_mode in ("heat", "cool"):
                stop_hvac(session)

            session.add(
                Ventilation(device_id=VENT_DEVICE_ID, mode=new_mode, timestamp=timestamp)
            )

            # open new session only for active HVAC
            if new_mode in ("heat", "cool"):
                start_hvac(session, ROOM_ID, VENT_DEVICE_ID)


        ROOM_STATE.hvac_mode = new_mode
        ROOM_STATE.virtual_temp = new_temp
        ROOM_STATE.last_timestamp = timestamp
        ROOM_STATE.last_ble_temp = ble_temp

        # ------------------------------------------------
        # 4) SAVE ADJUSTED ROOM TEMPERATURE (NOT RAW)
        # ------------------------------------------------
        session.add(
            Data(sensor_id=sensor.sensor_id, value=new_temp, timestamp=timestamp)
        )
