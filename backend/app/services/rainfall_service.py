from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Callable
from datetime import date
from pathlib import Path

import httpx
from opentelemetry import trace

from ..telemetry import in_current_context, rainfall_lookups, tracer
from .dem_service import CACHE_DIR

logger = logging.getLogger(__name__)

NASA_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Kept small: Open-Meteo's free quota is per IP and charged by data volume.
OPEN_METEO_YEARS = 3

DAYS_IN_MONTH = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_KEYS = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]

# Used only when every rainfall source fails; flagged in the response.
FALLBACK_ANNUAL_MM = 1000.0

_CACHE_FILE = Path(CACHE_DIR) / "rainfall_cache.json"
_cache_lock = threading.Lock()


class RainfallError(RuntimeError):
    """Rainfall could not be retrieved from any source."""


@dataclass(slots=True)
class RainfallResult:
    annual_mm: float
    monthly_mm: list[float]  # 12 values, January..December
    source: str
    measured: bool
    years: str


def manual_rainfall(annual_mm: float) -> RainfallResult:
    return RainfallResult(
        annual_mm=float(annual_mm),
        monthly_mm=[float(annual_mm) / 12.0] * 12,
        source="User supplied",
        measured=False,
        years="n/a",
    )


def fallback_rainfall() -> RainfallResult:
    rainfall_lookups.add(1, {"source": "fallback_default"})
    result = manual_rainfall(FALLBACK_ANNUAL_MM)
    result.source = (
        "Rainfall could not be looked up automatically; a default of "
        f"{FALLBACK_ANNUAL_MM:g} mm/year was used. Enter your local "
        "rainfall in Settings for accurate results."
    )
    return result


# ---------------------------------------------------------------------------
# Persistent cache (rainfall at a location does not change between requests)
# ---------------------------------------------------------------------------


def _cache_key(lat: float, lon: float) -> str:
    return f"{lat:.2f},{lon:.2f}"


def _cache_read(key: str) -> RainfallResult | None:
    try:
        with _cache_lock:
            data = json.loads(_CACHE_FILE.read_text())
        return RainfallResult(**data[key])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _cache_write(key: str, result: RainfallResult) -> None:
    try:
        with _cache_lock:
            try:
                data = json.loads(_CACHE_FILE.read_text())
            except (OSError, ValueError):
                data = {}
            data[key] = asdict(result)
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps(data))
    except OSError:
        pass  # cache is best-effort


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _from_nasa_power(lat: float, lon: float) -> RainfallResult:
    """NASA POWER 20-year rainfall climatology (mm/day per month)."""
    response = httpx.get(
        NASA_URL,
        params={
            "parameters": "PRECTOTCORR",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "format": "JSON",
        },
        timeout=httpx.Timeout(20.0, connect=4.0),
    )
    response.raise_for_status()
    per_day = response.json()["properties"]["parameter"]["PRECTOTCORR"]

    monthly = [
        float(per_day[key]) * days
        for key, days in zip(MONTH_KEYS, DAYS_IN_MONTH)
    ]
    if any(value < 0 for value in monthly):
        raise ValueError("NASA POWER returned no-data values")

    return RainfallResult(
        annual_mm=float(sum(monthly)),
        monthly_mm=monthly,
        source="NASA POWER climatology",
        measured=True,
        years="2001-2020",
    )


def _from_open_meteo(lat: float, lon: float) -> RainfallResult:
    """Average of the last few complete years of ERA5 reanalysis."""
    year1 = date.today().year - 1
    year0 = year1 - OPEN_METEO_YEARS + 1

    response = httpx.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year0}-01-01",
            "end_date": f"{year1}-12-31",
            "daily": "precipitation_sum",
            "timezone": "UTC",
        },
        timeout=httpx.Timeout(10.0, connect=4.0),
    )
    response.raise_for_status()
    daily = response.json()["daily"]

    month_totals = [0.0] * 12
    valid_days = 0
    for day, value in zip(daily["time"], daily["precipitation_sum"]):
        if value is None:
            continue
        month_totals[int(day[5:7]) - 1] += float(value)
        valid_days += 1

    if valid_days < 365:
        raise ValueError("too little data returned")

    years = year1 - year0 + 1
    monthly = [total / years for total in month_totals]

    return RainfallResult(
        annual_mm=float(sum(monthly)),
        monthly_mm=monthly,
        source="Open-Meteo ERA5 reanalysis",
        measured=True,
        years=f"{year0}-{year1}",
    )


PROVIDERS = (_from_nasa_power, _from_open_meteo)


def get_annual_rainfall(latitude: float, longitude: float) -> RainfallResult:
    """Yearly rainfall (mm) for a location, trying each free source in turn.

    Both sources have a native resolution of roughly 0.25-0.5 degrees, so the
    query point is rounded to 0.25 degrees; this also lets nearby areas share
    a cached answer.
    """
    lat = round(latitude * 4) / 4
    lon = round(longitude * 4) / 4
    key = _cache_key(lat, lon)
    span = trace.get_current_span()
    span.set_attribute("pond.rainfall.cell", key)

    cached = _cache_read(key)
    if cached is not None:
        span.set_attribute("pond.rainfall.cache_hit", True)
        span.set_attribute("pond.rainfall.source", cached.source)
        rainfall_lookups.add(1, {"source": "cache"})
        return cached
    span.set_attribute("pond.rainfall.cache_hit", False)

    # Ask every source at once and take the first that answers, so a slow or
    # blocked source cannot hold the request up.
    futures = {
        _pool.submit(in_current_context(_traced_provider), provider, lat, lon): provider
        for provider in PROVIDERS
    }
    pending = set(futures)
    failures = []

    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            try:
                result = future.result()
            except Exception as exc:  # network, HTTP status, bad payload
                failures.append(f"{futures[future].__name__}: {exc}")
                logger.warning(
                    "Rainfall source %s failed for %s: %s",
                    futures[future].__name__, key, exc,
                )
                continue
            _cache_write(key, result)
            span.set_attribute("pond.rainfall.source", result.source)
            rainfall_lookups.add(1, {"source": result.source})
            logger.info(
                "Rainfall for %s: %.0f mm/year from %s",
                key, result.annual_mm, result.source,
            )
            return result

    logger.error("Every rainfall source failed for %s: %s", key, failures)
    raise RainfallError("; ".join(failures))


def _traced_provider(provider, lat: float, lon: float) -> RainfallResult:
    with tracer.start_as_current_span(f"rainfall.{provider.__name__.removeprefix('_from_')}"):
        return provider(lat, lon)


_pool = ThreadPoolExecutor(max_workers=6)


def lookup_async(latitude: float, longitude: float) -> Callable[[], RainfallResult]:
    """Start a rainfall lookup in the background.

    Returns a function that waits for (and returns) the result, falling back
    to the flagged default if every source fails. Lets terrain processing run
    while the network request is in flight.
    """

    def task() -> RainfallResult:
        with tracer.start_as_current_span("rainfall.lookup"):
            try:
                return get_annual_rainfall(latitude, longitude)
            except RainfallError:
                return fallback_rainfall()

    future = _pool.submit(in_current_context(task))

    def result() -> RainfallResult:
        with tracer.start_as_current_span("rainfall.wait"):
            return future.result()

    return result
