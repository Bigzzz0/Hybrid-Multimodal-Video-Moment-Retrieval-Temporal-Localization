from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(order=True)
class _Job:
    priority: int
    sequence: int
    operation: Callable[[], Any] = field(compare=False)
    future: asyncio.Future = field(compare=False)


class InferenceScheduler:
    """Single GPU queue with interactive work ahead of ingestion work."""

    PRIORITIES = {
        "interactive_search": 0,
        "vqa": 1,
        "ingestion": 2,
    }

    def __init__(self) -> None:
        self.queue: asyncio.PriorityQueue[_Job] = asyncio.PriorityQueue()
        self.worker_task: asyncio.Task | None = None
        self.sequence = 0

    async def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._serve())

    async def stop(self) -> None:
        if self.worker_task is None:
            return
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass
        self.worker_task = None

    async def submit(self, priority: str, operation: Callable[[], Any]) -> Any:
        await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.sequence += 1
        await self.queue.put(_Job(self.PRIORITIES.get(priority, 2), self.sequence, operation, future))
        return await future

    async def _serve(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                result = await asyncio.to_thread(job.operation)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self.queue.task_done()


inference_scheduler = InferenceScheduler()
