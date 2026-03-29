import random
from datetime import datetime, timedelta


def generate_environment_for_day(
    room_id,
    date,
    readings_per_day,
    temp_range,
    humidity_range
):
    readings = []
    step = 24 // readings_per_day

    for i in range(readings_per_day):
        ts = datetime.combine(date, datetime.min.time()) + timedelta(hours=i * step)

        readings.append({
            "room_id": room_id,
            "timestamp": ts,
            "temperature": round(random.uniform(*temp_range), 1),
            "humidity": round(random.uniform(*humidity_range), 2),
        })

    return readings
