import asyncio
from datetime import datetime, timezone
from persistence.database import session_scope
from persistence.models import ComfortPreference


async def run_cli(room_id: int, patient_id: int):

    loop = asyncio.get_event_loop()

    while True:
        key = await loop.run_in_executor(None, input, "")

        if key.lower() != "c":
            continue

        print("\n--- Comfort Input ---")
        temp = await loop.run_in_executor(None, input, "Target temp: ")
        airflow = await loop.run_in_executor(None, input, "Airflow (on/off): ")

        try:
            with session_scope() as session:
                session.add(
                    ComfortPreference(
                        temperature_main=float(temp),
                        airflow=airflow.lower().startswith("on"),
                        patient_id=patient_id,
                        room_id=room_id,
                        source="manual",
                        timestamp=datetime.now(timezone.utc),
                    )
                )

            print("Saved preference\n")

        except Exception as e:
            print("Invalid input:", e)
