from datetime import datetime, timezone
from sqlalchemy.orm import Session

from persistence.models import UtilityUsage
from hybrid.state_store import ROOM_STATE

from persistence.models.utility_usage import UtilityUsage
from persistence.models.ventilation import Ventilation


def start_hvac(session: Session, room_id: int, device_id: int):
    usage = UtilityUsage(
        category="hvac",
        room_id=room_id,
        device_id=device_id,
        start_time=datetime.now(timezone.utc),
    )
    session.add(usage)
    session.flush()

    ROOM_STATE.active_hvac_usage_id = usage.usage_id

# kW consumption
HVAC_POWER = {
    "heat": 1.2,
    "cool": 1.4,
}

def stop_hvac(session):

    active = (
        session.query(UtilityUsage)
        .filter(
            UtilityUsage.category == "hvac",
            UtilityUsage.end_time.is_(None),
        )
        .first()
    )

    if not active:
        return

    now = datetime.now(timezone.utc)

    # duration in hours
    start = active.start_time
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    duration_h = (now - start).total_seconds() / 3600.0

    # find last hvac mode
    last_mode = (
        session.query(Ventilation.mode)
        .order_by(Ventilation.timestamp.desc())
        .first()
    )

    mode = last_mode[0] if last_mode else None
    power_kw = HVAC_POWER.get(mode, 0.0)

    # ENERGY = POWER × TIME
    active.power_consumption = power_kw * duration_h
    active.end_time = now