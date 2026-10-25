from __future__ import annotations

import asyncio
import io
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
import shapely
from opentelemetry import trace
from PIL import Image
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates
from shapely.geometry import Polygon, shape
from shapely.ops import transform as shapely_transform

from ..telemetry import in_current_context, tile_requests, tracer
from .terrain_service import TerrainResult, _calculate_slope_percent, _utm_crs

logger = logging.getLogger(__name__)

# AWS Open Data "Terrain Tiles": global, free, no API key. Terrarium encoding.
TILE_URL = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)
TILE_SIZE = 256

# The underlying data is ~30 m (SRTM/Copernicus); zooming past 14 adds no detail.
MAX_ZOOM = 14
MIN_ZOOM = 6

MAX_AREA_KM2 = float(os.getenv("POND_MAX_AREA_KM2", "25"))
MAX_TILES = 36
MAX_GRID_CELLS = 120_000
MIN_RESOLUTION_M = 20.0
BUFFER_M = 300.0

CACHE_DIR = Path(
    os.getenv(
        "POND_TILE_CACHE",
        Path(__file__).resolve().parents[2] / ".tile_cache",
    )
)
CACHE_LIMIT_BYTES = int(
    float(os.getenv("POND_TILE_CACHE_MB", "300")) * 1024 * 1024
)


# One shared connection pool: avoids a fresh TLS handshake for every tile.
#
# Some of S3's IP addresses can be unreachable from restricted networks, and a
# connection attempt to one of them hangs until the OS gives up (~2 minutes).
# A short connect timeout plus retries abandons a dead address quickly and
# tries again, instead of waiting.
_client = httpx.Client(
    timeout=httpx.Timeout(10.0, connect=3.0),
    transport=httpx.HTTPTransport(
        retries=2,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
    ),
)


class AreaError(ValueError):
    """The requested area is invalid or outside the supported limits."""


class TileFetchError(RuntimeError):
    """Elevation tiles could not be downloaded."""


@dataclass(slots=True)
class DemInfo:
    source: str
    zoom: int
    tile_count: int
    resolution_m: float
    area_km2: float
    grid_shape: tuple[int, int]


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------


def parse_polygon(geojson: dict | list) -> Polygon:
    """Accept a GeoJSON Polygon/Feature or a bare list of [lon, lat] points."""
    try:
        if isinstance(geojson, list):
            polygon = Polygon(geojson)
        else:
            geometry = geojson.get("geometry", geojson)
            polygon = shape(geometry)
    except Exception as exc:
        raise AreaError(f"Invalid polygon: {exc}") from exc

    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)

    if polygon.geom_type != "Polygon" or polygon.is_empty:
        raise AreaError("Area must be a polygon")

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if polygon.is_empty or polygon.area == 0:
        raise AreaError("Polygon has no area")

    minx, miny, maxx, maxy = polygon.bounds
    if not (-180 <= minx <= maxx <= 180 and -85 <= miny <= maxy <= 85):
        raise AreaError("Polygon coordinates must be [longitude, latitude]")

    return polygon


def polygon_area_km2(polygon: Polygon) -> float:
    centroid = polygon.centroid
    crs = _utm_crs(np.array([centroid.x]), np.array([centroid.y]))
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return shapely_transform(to_utm.transform, polygon).area / 1e6


def polygon_mask(
    terrain: TerrainResult,
    polygon: Polygon,
) -> np.ndarray:
    """Boolean grid: True where a cell centre lies inside the polygon."""
    to_utm = Transformer.from_crs(
        "EPSG:4326", terrain.crs, always_xy=True
    )
    projected = shapely_transform(to_utm.transform, polygon)

    half = terrain.grid_resolution_m / 2.0
    grid_x, grid_y = np.meshgrid(terrain.x_m + half, terrain.y_m + half)
    return shapely.contains_xy(projected, grid_x, grid_y)


# ---------------------------------------------------------------------------
# Web-mercator tile maths
# ---------------------------------------------------------------------------


def _lonlat_to_pixel(
    lon: np.ndarray, lat: np.ndarray, zoom: int
) -> tuple[np.ndarray, np.ndarray]:
    n = TILE_SIZE * (2**zoom)
    lat_rad = np.radians(np.clip(lat, -85.05112878, 85.05112878))
    px = (lon + 180.0) / 360.0 * n
    py = (
        (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return px, py


def _tile_metres_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2**zoom)


def _choose_zoom(lat: float, resolution_m: float) -> int:
    """Coarsest zoom whose pixels are at least as fine as the grid."""
    for zoom in range(MIN_ZOOM, MAX_ZOOM + 1):
        if _tile_metres_per_pixel(lat, zoom) <= resolution_m:
            return zoom
    return MAX_ZOOM


# ---------------------------------------------------------------------------
# Tile download + cache
# ---------------------------------------------------------------------------


def _tile_path(zoom: int, x: int, y: int) -> Path:
    return CACHE_DIR / str(zoom) / str(x) / f"{y}.png"


def _decode_terrarium(png_bytes: bytes) -> np.ndarray:
    rgb = np.asarray(
        Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.float32
    )
    return (
        rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
    )


@lru_cache(maxsize=96)
def _load_tile(zoom: int, x: int, y: int) -> np.ndarray:
    path = _tile_path(zoom, x, y)

    if path.exists():
        return _decode_terrarium(path.read_bytes())

    url = TILE_URL.format(z=zoom, x=x, y=y)
    last_error: Exception | None = None

    for _ in range(5):
        try:
            response = _client.get(url)
            response.raise_for_status()
            break
        except Exception as exc:  # network / HTTP error
            last_error = exc
    else:
        tile_requests.add(1, {"result": "failed"})
        logger.error(
            "Elevation tile %s/%s/%s could not be downloaded: %s",
            zoom, x, y, last_error,
        )
        raise TileFetchError(
            f"Could not download elevation tile {zoom}/{x}/{y}: {last_error}"
        )

    tile_requests.add(1, {"result": "downloaded"})

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    except OSError:
        pass  # cache is best-effort

    return _decode_terrarium(response.content)


async def _download_tiles(coords: list[tuple[int, int, int]]) -> None:
    """Fetch tiles concurrently, writing them to the disk cache.

    The async client races connections to every address the host resolves to
    (RFC 8305 "happy eyeballs"), so an unreachable address costs a fraction
    of a second instead of a full connect timeout. Failures are left for the
    synchronous fallback in _load_tile to retry and report.
    """
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8)
    gate = asyncio.Semaphore(8)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0),
        transport=httpx.AsyncHTTPTransport(retries=2, limits=limits),
    ) as client:

        async def fetch(zoom: int, x: int, y: int) -> None:
            url = TILE_URL.format(z=zoom, x=x, y=y)
            async with gate:
                for attempt in range(4):
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                    except Exception as exc:
                        logger.warning(
                            "Tile %s/%s/%s attempt %d failed: %r",
                            zoom, x, y, attempt + 1, exc,
                        )
                        continue
                    tile_requests.add(1, {"result": "downloaded"})
                    try:
                        path = _tile_path(zoom, x, y)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(response.content)
                    except OSError:
                        pass  # cache is best-effort
                    return

        await asyncio.gather(*(fetch(*c) for c in coords))


def _download_missing(coords: list[tuple[int, int, int]]) -> None:
    missing = [c for c in coords if not _tile_path(*c).exists()]
    cached = len(coords) - len(missing)
    if cached:
        tile_requests.add(cached, {"result": "cache_hit"})

    with tracer.start_as_current_span("tiles.download") as span:
        span.set_attribute("pond.tiles.needed", len(coords))
        span.set_attribute("pond.tiles.cached", cached)
        span.set_attribute("pond.tiles.missing", len(missing))
        if not missing:
            return
        logger.info(
            "Downloading %d of %d elevation tiles (%d cached)",
            len(missing), len(coords), cached,
        )
        try:
            asyncio.run(_download_tiles(missing))
        except RuntimeError:
            pass  # already inside an event loop; the sync fallback handles it


def _prune_cache() -> None:
    try:
        files = [p for p in CACHE_DIR.rglob("*.png")]
        total = sum(p.stat().st_size for p in files)
        if total <= CACHE_LIMIT_BYTES:
            return
        for path in sorted(files, key=lambda p: p.stat().st_mtime):
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
            if total <= CACHE_LIMIT_BYTES * 0.8:
                break
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_terrain_from_polygon(
    polygon: Polygon,
    *,
    buffer_m: float = BUFFER_M,
    min_resolution_m: float = MIN_RESOLUTION_M,
    max_area_km2: float = MAX_AREA_KM2,
) -> tuple[TerrainResult, DemInfo]:
    """Download elevation tiles covering the polygon and build a metric DEM.

    The returned TerrainResult has the same layout as the one built from
    contour lines, so the hydrology pipeline can consume either.
    """
    area_km2 = polygon_area_km2(polygon)
    if area_km2 > max_area_km2:
        raise AreaError(
            f"Selected area is {area_km2:.1f} km², above the "
            f"{max_area_km2:g} km² limit. Please draw a smaller area."
        )

    centroid = polygon.centroid
    crs: CRS = _utm_crs(np.array([centroid.x]), np.array([centroid.y]))
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Buffered bounding box in UTM metres.
    projected = shapely_transform(to_utm.transform, polygon)
    minx, miny, maxx, maxy = projected.bounds
    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m

    # Grid resolution: as fine as allowed, coarser only to respect cell budget.
    width_m = maxx - minx
    height_m = maxy - miny
    resolution = max(
        min_resolution_m,
        math.sqrt(width_m * height_m / MAX_GRID_CELLS),
    )

    cols = max(int(math.ceil(width_m / resolution)), 3)
    rows = max(int(math.ceil(height_m / resolution)), 3)

    x_m = minx + np.arange(cols) * resolution  # cell lower-left corners
    y_m = miny + np.arange(rows) * resolution
    half = resolution / 2.0

    # Sample points = cell centres, converted back to lon/lat.
    grid_x, grid_y = np.meshgrid(x_m + half, y_m + half)
    lon, lat = to_lonlat.transform(grid_x, grid_y)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Zoom / tile selection, reducing zoom if too many tiles are needed.
    zoom = _choose_zoom(float(centroid.y), resolution)
    while True:
        px, py = _lonlat_to_pixel(lon, lat, zoom)
        tx0 = int(px.min() // TILE_SIZE)
        tx1 = int(px.max() // TILE_SIZE)
        ty0 = int(py.min() // TILE_SIZE)
        ty1 = int(py.max() // TILE_SIZE)
        tile_count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        if tile_count <= MAX_TILES or zoom <= MIN_ZOOM:
            break
        zoom -= 1

    coords = [
        (zoom, tx, ty)
        for ty in range(ty0, ty1 + 1)
        for tx in range(tx0, tx1 + 1)
    ]

    span = trace.get_current_span()
    span.set_attribute("pond.area_km2", round(area_km2, 3))
    span.set_attribute("pond.grid.resolution_m", round(resolution, 1))
    span.set_attribute("pond.grid.cells", rows * cols)
    span.set_attribute("pond.tiles.zoom", zoom)
    span.set_attribute("pond.tiles.count", tile_count)

    _download_missing(coords)

    with tracer.start_as_current_span("tiles.decode_and_stitch"):
        load = in_current_context(lambda c: _load_tile(*c))
        with ThreadPoolExecutor(max_workers=8) as pool:
            tiles = list(pool.map(load, coords))

    mosaic = np.zeros(
        ((ty1 - ty0 + 1) * TILE_SIZE, (tx1 - tx0 + 1) * TILE_SIZE),
        dtype=np.float32,
    )
    for (_, tx, ty), tile in zip(coords, tiles):
        r0 = (ty - ty0) * TILE_SIZE
        c0 = (tx - tx0) * TILE_SIZE
        mosaic[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile

    _prune_cache()

    # Bilinear sample the mosaic at each grid cell centre.
    # Pixel centres sit at +0.5, so shift when indexing the array.
    elevation = map_coordinates(
        mosaic,
        [py - ty0 * TILE_SIZE - 0.5, px - tx0 * TILE_SIZE - 0.5],
        order=1,
        mode="nearest",
    ).astype(np.float32)

    if not np.isfinite(elevation).all():
        raise TileFetchError("Elevation data contained invalid values")

    slope = _calculate_slope_percent(elevation, resolution)

    min_lon, min_lat = to_lonlat.transform(minx, miny)
    max_lon, max_lat = to_lonlat.transform(maxx, maxy)

    terrain = TerrainResult(
        elevation_grid_m=elevation,
        x_m=x_m,
        y_m=y_m,
        crs=crs.to_string(),
        bounds_lonlat=(
            float(min_lon),
            float(min_lat),
            float(max_lon),
            float(max_lat),
        ),
        min_elevation_m=float(elevation.min()),
        max_elevation_m=float(elevation.max()),
        grid_resolution_m=float(resolution),
        sample_count=0,
        valid_cell_count=int(elevation.size),
        slope_grid_percent=slope,
        slope_min_percent=float(slope.min()),
        slope_max_percent=float(slope.max()),
        slope_mean_percent=float(slope.mean()),
        contour_rmse_m=0.0,
        contour_max_abs_error_m=0.0,
        contour_p95_abs_error_m=0.0,
    )

    info = DemInfo(
        source="AWS Terrain Tiles (Terrarium, ~30 m SRTM/Copernicus-derived)",
        zoom=zoom,
        tile_count=tile_count,
        resolution_m=float(resolution),
        area_km2=float(area_km2),
        grid_shape=(rows, cols),
    )

    return terrain, info
