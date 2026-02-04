from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .data_processor import DataProcessor


AsyncSink = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class CollectorConfig:
    """
    max_incoming_queue:
      Queue between BLE callbacks and processing.
      Keep this reasonably large to avoid drops during bursts.

    drop_on_full:
      If True: drops incoming events when queue is full (keeps system responsive).
      If False: backpressure (BLE callback await may slow things down).
    """
    max_incoming_queue: int = 5000
    drop_on_full: bool = True


class DataCollector:
    """
    Buffer-free DataCollector:
    BLEManager -> DataCollector.ingest(event) -> DataProcessor -> sinks

    Uses an internal queue to keep BLE callback fast and non-blocking.
    """

    def __init__(
        self,
        *,
        processor: Optional[DataProcessor] = None,
        sinks: Optional[List[AsyncSink]] = None,
        config: Optional[CollectorConfig] = None,
    ) -> None:
        self.config = config or CollectorConfig()
        self.processor = processor or DataProcessor()
        self.sinks: List[AsyncSink] = sinks or []

        self._incoming: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_incoming_queue)
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

        # Stats
        self.received = 0
        self.processed_ok = 0
        self.dropped = 0
        self.sink_errors = 0

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def ingest(self, event: Dict[str, Any]) -> None:
        """
        Async entry point intended to be passed to BLEManager(on_event=...).
        """
        self.received += 1

        if self.config.drop_on_full:
            try:
                self._incoming.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped += 1
        else:
            await self._incoming.put(event)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = await self._incoming.get()

                result = self.processor.process(event)
                if not result.ok or result.event is None:
                    continue

                self.processed_ok += 1
                cleaned = result.event

                # Dispatch to sinks
                for sink in self.sinks:
                    try:
                        await sink(cleaned)
                    except Exception:
                        self.sink_errors += 1

            except asyncio.CancelledError:
                break
            except Exception:
                # protect the pipeline
                continue

    def get_stats(self) -> Dict[str, Any]:
        return {
            "received": self.received,
            "processed_ok": self.processed_ok,
            "dropped": self.dropped,
            "sink_errors": self.sink_errors,
            "incoming_queue_size": self._incoming.qsize(),
        }
