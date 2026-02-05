from persistence.database import session_scope
from persistence.models.data import Data


async def db_sink(event: dict):
    """
    Persist clean sensor events to the database.
    """

    sensor_id = event.get("sensor_id")
    if sensor_id is None:
        raise ValueError("Event missing sensor_id")

    with session_scope() as session:
        data_row = Data(
            sensor_id=sensor_id,
            value=event["value"],
        )
        session.add(data_row)
