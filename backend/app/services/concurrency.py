from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, TypeVar

from fastapi import HTTPException
from opentelemetry.trace import Status, StatusCode

from ..telemetry import active_analyses, queue_wait, tracer

T = TypeVar("T")

logger = logging.getLogger(__name__)

# One heavy analysis at a time: the deployment target has a single vCPU and
# 512 MB of RAM, so parallel analyses would only slow each other down or OOM.
_slot = asyncio.Semaphore(1)
QUEUE_TIMEOUT_S = 60.0
RUN_TIMEOUT_S = 120.0


async def run_exclusive(func: Callable[..., T], *args, **kwargs) -> T:
    active_analyses.add(1)
    try:
        with tracer.start_as_current_span("queue.wait") as span:
            waited_from = time.perf_counter()
            try:
                await asyncio.wait_for(_slot.acquire(), timeout=QUEUE_TIMEOUT_S)
            except asyncio.TimeoutError:
                span.set_status(Status(StatusCode.ERROR, "queue timeout"))
                logger.warning(
                    "Rejected analysis: slot busy for over %.0f s (HTTP 503)",
                    QUEUE_TIMEOUT_S,
                )
                raise HTTPException(
                    status_code=503,
                    detail="Server is busy with other analyses. Please retry shortly.",
                )
            waited = time.perf_counter() - waited_from
            span.set_attribute("pond.queue.wait_s", round(waited, 3))
            queue_wait.record(waited)
            if waited > 1.0:
                logger.info("Analysis waited %.1f s for the processing slot", waited)

        try:
            # to_thread copies the current context, so spans opened inside
            # func become children of the request's span.
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=RUN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Analysis exceeded the %.0f s run limit (HTTP 504)", RUN_TIMEOUT_S
            )
            raise HTTPException(
                status_code=504,
                detail="Analysis took too long. Try a smaller area.",
            )
        finally:
            _slot.release()
    finally:
        active_analyses.add(-1)
