from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

from .device import Device


EventCallback = Callable[[dict], Awaitable[None]] 


async def _probe_device_connection(mac_address: str, timeout: float) -> bool:
    async with BleakClient(mac_address, timeout=timeout) as client:
        return bool(client.is_connected)


def probe_device_connection(mac_address: str, timeout: float = 8.0) -> bool:
    normalized_mac = mac_address.strip().upper()
    try:
        return asyncio.run(_probe_device_connection(normalized_mac, timeout))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_probe_device_connection(normalized_mac, timeout))
        finally:
            loop.close()


@dataclass
class ConnectionState:
    client: Optional[BleakClient] = None
    connected: bool = False
    last_error: Optional[str] = None
    reconnect_attempts: int = 0
    last_seen: Optional[datetime] = None


class BLEManager:
    """
    Manages BLE scanning, connections, notifications, and reconnection.

    Emits events in the form:
    {
      "timestamp": "...",
      "mac": "...",
      "device_label": "...",
      "sensor_type": "...",
      "unit": "...",
      "uuid": "...",
      "value": ...,
      "raw_hex": "...",
    }
    """

    def __init__(
        self,
        devices: List[Device],
        on_event: Optional[EventCallback] = None,
        *,
        connect_timeout: float = 12.0,
        max_reconnect_backoff_s: float = 30.0,
        initial_backoff_s: float = 1.0,
    ) -> None:
        self.devices = {d.mac_address.upper(): d for d in devices}
        self.on_event = on_event

        self.connect_timeout = connect_timeout
        self.max_reconnect_backoff_s = max_reconnect_backoff_s
        self.initial_backoff_s = initial_backoff_s

        self._states: Dict[str, ConnectionState] = {
            mac: ConnectionState() for mac in self.devices.keys()
        }

        self._tasks: Dict[str, asyncio.Task] = {}
        self._stop_event = asyncio.Event()

    # -------------------------
    # Public API
    # -------------------------

    async def scan(self, timeout: float = 5.0) -> List[dict]:
        """
        Scan for BLE devices and return a list of dict results.
        """
        found = await BleakScanner.discover(timeout=timeout)
        results = []
        for dev in found:
            results.append(
                {
                    "name": dev.name,
                    "address": dev.address,
                    "rssi": getattr(dev, "rssi", None),
                }
            )
        return results

    async def start(self) -> None:
        """
        Start connection loops for all configured devices.
        """
        self._stop_event.clear()
        for mac in self.devices.keys():
            if mac not in self._tasks or self._tasks[mac].done():
                print(f"Starting BLEManager loop for device {mac}")
                self._tasks[mac] = asyncio.create_task(self._device_loop(mac))
                print(f"Started BLEManager loop for device {mac}")  

    async def stop(self) -> None:
        """
        Stop all tasks and disconnect cleanly.
        """
        self._stop_event.set()

        # Cancel loops
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        # Disconnect clients
        for mac, state in self._states.items():
            await self._safe_disconnect(mac, state)

    def get_status(self) -> dict:
        """
        Lightweight status for dashboard/healthcheck.
        """
        out = {}
        for mac, state in self._states.items():
            dev = self.devices[mac]
            out[mac] = {
                "label": dev.label or dev.name or mac,
                "connected": state.connected,
                "reconnect_attempts": state.reconnect_attempts,
                "last_error": state.last_error,
                "last_seen": state.last_seen.isoformat() if state.last_seen else None,
            }
        return out

    # -------------------------
    # Internal loops
    # -------------------------

    async def _device_loop(self, mac: str) -> None:
        """
        Keeps a single device connected + subscribed.
        Reconnects on disconnect with exponential backoff.
        """
        state = self._states[mac]
        device = self.devices[mac]
        backoff = self.initial_backoff_s

        while not self._stop_event.is_set():
            try:
                await self._connect_and_subscribe(mac, device, state)
                # If connected/subscribed, just wait until we are stopped.
                # Disconnect callback will flip state and the loop will reconnect.
                while not self._stop_event.is_set() and state.connected:
                    await asyncio.sleep(0.5)

                # If we got here, disconnected
                await self._safe_disconnect(mac, state)

            except asyncio.CancelledError:
                break
            except Exception as e:
                state.last_error = f"{type(e).__name__}: {e}"
                state.connected = False
                await self._safe_disconnect(mac, state)

            # Backoff before reconnect
            if self._stop_event.is_set():
                break

            state.reconnect_attempts += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, self.max_reconnect_backoff_s)

        # Final cleanup
        await self._safe_disconnect(mac, state)



    async def _device_loop(self, mac: str) -> None:
        state = self._states[mac]
        device = self.devices[mac]
        backoff = self.initial_backoff_s

        while not self._stop_event.is_set():
            try:
                state.connecting = True
                print(f"[{mac}] connecting...")

                async with BleakClient(mac, timeout=self.connect_timeout) as client:
                    state.client = client
                    state.connected = True
                    state.connecting = False
                    state.reconnect_attempts = 0
                    state.last_error = None
                    state.last_seen = datetime.now(timezone.utc)
                    print(f"[{mac}] connected={client.is_connected}")

                    # --- device setup writes (like SOUND_CFG_UUID) ---
                    for (uuid, payload, response) in getattr(device, "setup_writes", []):
                        await client.write_gatt_char(uuid, payload, response=response)

                    # --- subscribe ---
                    for sensor in device.sensors:
                        await client.start_notify(
                            sensor.uuid,
                            lambda _uuid, data, s=sensor: asyncio.create_task(
                                self._handle_notification(device, s, _uuid, data)
                            ),
                        )

                    print(f"[{mac}] subscribed")

                    # keep alive until stop
                    while not self._stop_event.is_set():
                        await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                state.connected = False
                state.connecting = False
                state.last_error = f"{type(e).__name__}: {e}"
                print(f"[{mac}] disconnected/error: {state.last_error} -> retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self.max_reconnect_backoff_s)

        state.connected = False
        state.connecting = False



    async def _safe_disconnect(self, mac: str, state: ConnectionState) -> None:
        client = state.client
        if client is None:
            return
        try:
            if client.is_connected:
                await client.disconnect()
        except Exception:
            pass
        finally:
            state.connected = False
            state.client = None

    # -------------------------
    # Notification handler
    # -------------------------

    async def _handle_notification(
        self,
        device: Device,
        sensor,
        uuid: str,
        data: bytes,
    ) -> None:
        """
        Parse the notification bytes -> event dict and forward to callback.
        """
        mac = device.mac_address.upper()
        self._states[mac].last_seen = datetime.now(timezone.utc)

        try:
            value = sensor.parse(data)
            error = None
        except Exception as e:
            value = None
            error = f"{type(e).__name__}: {e}"

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mac": mac,
            "device_label": device.label or device.name or mac,
            "sensor_type": sensor.sensor_type,
            "unit": sensor.unit,
            "uuid": str(getattr(uuid, "uuid", uuid)),
            "value": value,
            "raw_hex": data.hex(),
            "error": error,
        }

        if self.on_event is not None:
            await self.on_event(event)
