from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .data_processor import DataProcessor

AsyncSink = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class CollectorConfig:
    #max_incoming_queue: int = 10000
    drop_on_full: bool = True


class DataCollector:
    def __init__(
        self,
        *,
        processor: Optional[DataProcessor] = None,
        sinks: Optional[List[AsyncSink]] = None,
        config: Optional[CollectorConfig] = None,
    ):
        self.config = config or CollectorConfig()
        self.processor = processor or DataProcessor()
        self.sinks = sinks or []

        self._incoming = asyncio.Queue()
        self._task: asyncio.Task | None = None

        self.received = 0
        self.processed_ok = 0
        self.dropped = 0
        self.sink_errors = 0

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def ingest(self, event: Dict[str, Any]):
        self.received += 1
        try:
            self._incoming.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1

    async def _run(self):
        while True:
            event = await self._incoming.get()
            try:
                result = self.processor.process(event)
                if result.ok and result.event:
                    self.processed_ok += 1
                    for sink in self.sinks:
                        await sink(result.event)
            except Exception:
                self.sink_errors += 1
            finally:
                self._incoming.task_done()

    async def stop(self):
        await self._incoming.join()
        if self._task:
            self._task.cancel()
