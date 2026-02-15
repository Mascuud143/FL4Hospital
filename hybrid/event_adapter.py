import asyncio
from hybrid.controller import process_ble_event

queue = asyncio.Queue(maxsize=1000)


def _normalize_uuid(u):
    if u is None:
        return None
    if hasattr(u, "uuid"):
        try:
            return str(u.uuid)
        except Exception:
            pass
    return str(u)


async def handle_ble_event(event: dict):
    """
    Fast non-blocking BLE callback
    Also normalizes Bleak objects → plain dict
    """

    # ---- normalize UUID (CRITICAL FIX) ----
    event["uuid"] = _normalize_uuid(event.get("uuid"))

    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        print("Hybrid queue full — dropping event")


async def event_worker():
    """
    Slow processing worker
    """
    while True:
        event = await queue.get()
        try:
            await process_ble_event(event)
        except Exception as e:
            print("Controller error:", e)
        finally:
            queue.task_done()
