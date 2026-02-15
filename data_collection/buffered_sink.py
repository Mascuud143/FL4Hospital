import asyncio
from persistence.database import session_scope
from persistence.models.data import Data
from persistence.models.sensor import Sensor as SensorModel
from persistence.models.device import Device as DeviceModel

BUFFER = []
FLUSH_INTERVAL = 2.0  # seconds


async def buffered_db_sink(event: dict):
    BUFFER.append(event)


async def flush_loop():
    while True:
        await asyncio.sleep(FLUSH_INTERVAL)

        if not BUFFER:
            continue

        batch = BUFFER.copy()
        BUFFER.clear()

        with session_scope() as session:
            for event in batch:
                mac = event.get("mac")
                uuid = str(event.get("uuid"))
                value = event.get("value")

                device = session.query(DeviceModel)\
                    .filter(DeviceModel.mac_address == mac)\
                    .one_or_none()

                if not device:
                    continue

                sensor = session.query(SensorModel)\
                    .filter(SensorModel.device_id == device.device_id)\
                    .filter(SensorModel.uuid == uuid)\
                    .one_or_none()

                if not sensor:
                    continue

                session.add(Data(sensor_id=sensor.sensor_id, value=value))
