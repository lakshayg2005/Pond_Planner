from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# The browser sends its spans here; they are passed on unchanged to the
# collector, which is only reachable on the internal network.
router = APIRouter(tags=["Telemetry"], include_in_schema=False)

MAX_BODY_BYTES = 1024 * 1024

_client = httpx.AsyncClient(timeout=5.0)
# Forwarding telemetry must not itself produce spans.
HTTPXClientInstrumentor.uninstrument_client(_client)


@router.post("/otel/v1/traces")
async def forward_browser_traces(request: Request) -> Response:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return Response(status_code=204)  # telemetry disabled: drop quietly

    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Telemetry batch too large")

    try:
        upstream = await _client.post(
            endpoint.rstrip("/") + "/v1/traces",
            content=body,
            headers={
                "Content-Type": request.headers.get(
                    "content-type", "application/json"
                )
            },
        )
    except httpx.HTTPError:
        return Response(status_code=503)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )
