"""OpenTelemetry set-up: traces, metrics and logs, exported over OTLP/HTTP.

Configured with the standard OpenTelemetry environment variables, e.g.

    OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:5080/api/default
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<base64 user:password>
    OTEL_SERVICE_NAME=pond-planner

With no endpoint configured the SDK is still installed (so trace ids appear
in the console logs) but nothing is exported. POND_OTEL_CONSOLE=1 also prints
every span to stdout, which is useful when no backend is running.
"""

from __future__ import annotations

import logging
import os
import sys

from opentelemetry import context as otel_context
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "pond-planner")

tracer = trace.get_tracer("pond.planner")
meter = metrics.get_meter("pond.planner")

_configured = False


def _exporting() -> bool:
    return bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


class _TraceIdFormatter(logging.Formatter):
    """Console format with the current trace/span id.

    The ids are added to the formatted text only, not to the record, since
    the OTLP log handler would export extra record fields as attributes.
    """

    def format(self, record: logging.LogRecord) -> str:
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            ids = (
                f"trace_id={context.trace_id:032x} "
                f"span_id={context.span_id:016x}"
            )
        else:
            ids = "trace_id=- span_id=-"
        return super().format(record).replace("{ids}", ids, 1)


class _DropTelemetryForwarding(logging.Filter):
    """Hides httpx's log line for each browser span batch forwarded."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/v1/traces" not in record.getMessage()


def setup_telemetry() -> None:
    """Install the tracer, meter and logger providers (once per process)."""
    global _configured
    if _configured or os.getenv("OTEL_SDK_DISABLED", "").lower() == "true":
        return
    _configured = True

    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("POND_ENVIRONMENT", "dev"),
        }
    )

    # --- Traces -----------------------------------------------------------
    tracer_provider = TracerProvider(resource=resource)
    if _exporting():
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if os.getenv("POND_OTEL_CONSOLE") == "1":
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
    trace.set_tracer_provider(tracer_provider)

    # --- Metrics ----------------------------------------------------------
    readers = []
    if _exporting():
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )

        interval_ms = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "15000"))
        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(), export_interval_millis=interval_ms
            )
        )
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=readers)
    )

    # --- Logs -------------------------------------------------------------
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        _TraceIdFormatter("%(asctime)s %(levelname)s [%(name)s] [{ids}] %(message)s")
    )
    root.addHandler(console)
    logging.getLogger("httpx").addFilter(_DropTelemetryForwarding())

    if _exporting():
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.instrumentation.logging.handler import LoggingHandler

        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter())
        )
        set_logger_provider(logger_provider)
        otel_handler = LoggingHandler(
            level=logging.INFO, logger_provider=logger_provider
        )
        root.addHandler(otel_handler)
        # Uvicorn's own loggers do not propagate to the root logger.
        logging.getLogger("uvicorn").addHandler(otel_handler)

    # --- Automatic instrumentation ---------------------------------------
    # Every outgoing httpx call (elevation tiles, rainfall APIs) becomes a
    # client span. Must run before the services create their httpx clients.
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()

    # Process memory / CPU: the deployment host has only 512 MB and 1 vCPU.
    from opentelemetry.instrumentation.system_metrics import (
        SystemMetricsInstrumentor,
    )

    SystemMetricsInstrumentor(
        config={
            "process.memory.usage": None,
            "process.memory.virtual": None,
            "process.cpu.time": ["user", "system"],
            "process.cpu.utilization": None,
            "process.thread.count": None,
            "process.open_file_descriptor.count": None,
            "process.runtime.gc_count": None,
            "system.memory.usage": ["used", "free", "cached"],
            "system.memory.utilization": ["used", "free", "cached"],
            "system.cpu.utilization": ["idle", "user", "system", "iowait"],
        }
    ).instrument()

    logging.getLogger(__name__).info(
        "OpenTelemetry configured: service=%s exporting=%s endpoint=%s",
        SERVICE_NAME,
        _exporting(),
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "-"),
    )


def in_current_context(func):
    """Wrap func so it runs in the caller's trace context on another thread.

    ThreadPoolExecutor does not carry context over, so spans opened by pooled
    work (tile loads, rainfall providers) would otherwise start new traces.
    """
    parent = otel_context.get_current()

    def run(*args, **kwargs):
        token = otel_context.attach(parent)
        try:
            return func(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return run


def instrument_app(app) -> None:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    # Health checks and static frontend files would only add noise, as would
    # the per-chunk ASGI send/receive spans (a 6 MB upload makes ~500).
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls=r"api/health,assets/.*,favicon,otel/v1/traces",
        exclude_spans=["receive", "send"],
    )


# ---------------------------------------------------------------------------
# Application metrics
# ---------------------------------------------------------------------------

# The SDK's default histogram buckets suit milliseconds; these durations are
# in seconds, so without explicit buckets nearly everything lands in "0-5".
SECONDS_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 60, 120]

analysis_counter = meter.create_counter(
    "pond.analysis.count",
    unit="{analysis}",
    description="Pond analyses run, by mode and outcome",
)
analysis_duration = meter.create_histogram(
    "pond.analysis.duration",
    unit="s",
    description="End-to-end time of one pond analysis",
    explicit_bucket_boundaries_advisory=SECONDS_BUCKETS,
)
stage_duration = meter.create_histogram(
    "pond.analysis.stage.duration",
    unit="s",
    description="Time spent in each stage of the analysis pipeline",
    explicit_bucket_boundaries_advisory=SECONDS_BUCKETS,
)
candidates_found = meter.create_histogram(
    "pond.candidates.found",
    unit="{pond}",
    description="Pond sites returned per analysis",
    explicit_bucket_boundaries_advisory=[0, 1, 2, 3, 5, 10, 20, 50],
)
grid_cells = meter.create_histogram(
    "pond.grid.cells",
    unit="{cell}",
    description="Size of the elevation grid processed per analysis",
    explicit_bucket_boundaries_advisory=[1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
)
area_km2 = meter.create_histogram(
    "pond.area.size",
    unit="km2",
    description="Area drawn by the user",
    explicit_bucket_boundaries_advisory=[0.5, 1, 2, 5, 10, 15, 20, 25],
)
annual_runoff = meter.create_histogram(
    "pond.best_site.annual_runoff",
    unit="m3",
    description="Yearly water reaching the best pond site",
    explicit_bucket_boundaries_advisory=[1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
)
tile_requests = meter.create_counter(
    "pond.tiles.requests",
    unit="{tile}",
    description="Elevation tiles needed, by result (cache_hit, downloaded, failed)",
)
rainfall_lookups = meter.create_counter(
    "pond.rainfall.lookups",
    unit="{lookup}",
    description="Rainfall lookups, by the source that answered",
)
queue_wait = meter.create_histogram(
    "pond.queue.wait",
    unit="s",
    description="Time an analysis waited for the single processing slot",
    explicit_bucket_boundaries_advisory=SECONDS_BUCKETS,
)
active_analyses = meter.create_up_down_counter(
    "pond.analyses.active",
    unit="{analysis}",
    description="Analyses currently running or waiting for the slot",
)
