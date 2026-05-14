import asyncio

from .controller import process_ble_event

queue = asyncio.Queue(maxsize=1000)


def _normalize_uuid(value):
    if value is None:
        return None
    if hasattr(value, "uuid"):
        try:
            return str(value.uuid)
        except Exception:
            pass
    return str(value)


async def handle_ble_event(event: dict):
    """Queue BLE events quickly and normalize UUID payloads."""
    event["uuid"] = _normalize_uuid(event.get("uuid"))

    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        print("Hybrid queue full - dropping event")


async def event_worker():
    """Process queued BLE events in the background."""
    while True:
        event = await queue.get()
        try:
            await process_ble_event(event)
        except Exception as exc:
            print("Controller error:", exc)
        finally:
            queue.task_done()
