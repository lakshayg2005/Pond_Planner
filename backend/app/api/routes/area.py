from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from opentelemetry import trace

from ...schemas.area import AreaAnalyzeRequest, AreaAnalyzeResponse
from ...services.analysis_pipeline import run_hydrology, stage
from ...services.concurrency import run_exclusive
from ...services.contour_service import dem_contours
from ...services.dem_service import (
    AreaError,
    TileFetchError,
    build_terrain_from_polygon,
    parse_polygon,
    polygon_mask,
)
from ...services.rainfall_service import lookup_async, manual_rainfall
from ...telemetry import (
    analysis_counter,
    analysis_duration,
    annual_runoff,
    area_km2,
    candidates_found,
    grid_cells,
    rainfall_lookups,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Area Analysis"],
)


def _analyse(request: AreaAnalyzeRequest, polygon, rainfall_source):
    started = time.perf_counter()
    timings: dict[str, float] = {}

    with stage("terrain.build"):
        terrain, dem_info = build_terrain_from_polygon(polygon)
    with stage("terrain.polygon_mask"):
        mask = polygon_mask(terrain, polygon)
    timings["terrain_s"] = time.perf_counter() - started

    if not mask.any():
        raise AreaError("Selected area is too small to analyse")

    rainfall_wait = [0.0]

    def resolve_rainfall():
        waited_from = time.perf_counter()
        result = rainfall_source()
        rainfall_wait[0] = time.perf_counter() - waited_from
        return result

    hydrology_started = time.perf_counter()
    output = run_hydrology(
        terrain,
        max_slope_percent=request.max_slope_percent,
        minimum_accumulation=request.minimum_accumulation,
        max_candidates=request.max_candidates,
        minimum_distance_cells=request.minimum_distance_cells,
        analysis_mask=mask,
        rainfall=resolve_rainfall,
        runoff_coefficient=request.runoff_coefficient,
    )
    timings["rainfall_wait_s"] = rainfall_wait[0]
    timings["hydrology_s"] = (
        time.perf_counter() - hydrology_started - rainfall_wait[0]
    )

    contours_started = time.perf_counter()
    with stage("contours.generate") as span:
        contours, contour_interval = dem_contours(terrain)
        span.set_attribute("pond.contours.interval_m", float(contour_interval))
    output.map_data["contours"] = contours
    timings["contours_s"] = time.perf_counter() - contours_started

    height, width = terrain.elevation_grid_m.shape
    flow = output.flow

    return (
        terrain, dem_info, output, flow, (height, width),
        contour_interval, timings,
    )


@router.post(
    "/analyze-area",
    response_model=AreaAnalyzeResponse,
)
async def analyze_area(request: AreaAnalyzeRequest) -> AreaAnalyzeResponse:
    span = trace.get_current_span()
    request_started = time.perf_counter()

    def failed(status: int, outcome: str, exc: Exception) -> HTTPException:
        analysis_counter.add(1, {"mode": "area", "outcome": outcome})
        span.set_attribute("pond.outcome", outcome)
        span.record_exception(exc)
        logger.warning("Area analysis rejected (%d %s): %s", status, outcome, exc)
        return HTTPException(status_code=status, detail=str(exc))

    try:
        polygon = parse_polygon(request.polygon)
    except AreaError as exc:
        raise failed(422, "invalid_area", exc) from exc

    centroid = polygon.centroid
    span.set_attribute("pond.centroid.lat", round(centroid.y, 5))
    span.set_attribute("pond.centroid.lon", round(centroid.x, 5))
    logger.info(
        "Area analysis started at (%.4f, %.4f), max_candidates=%d",
        centroid.y, centroid.x, request.max_candidates,
    )

    # The lookup starts now and runs while the terrain is being processed.
    if request.rainfall_mm is not None:
        rainfall_lookups.add(1, {"source": "user_supplied"})
        rainfall_source = lambda: manual_rainfall(request.rainfall_mm)  # noqa: E731
    else:
        rainfall_source = lookup_async(centroid.y, centroid.x)

    try:
        (
            terrain, dem_info, output, flow, (height, width),
            contour_interval, timings,
        ) = await run_exclusive(_analyse, request, polygon, rainfall_source)
    except AreaError as exc:
        raise failed(422, "invalid_area", exc) from exc
    except TileFetchError as exc:
        raise failed(502, "tile_download_failed", exc) from exc
    except HTTPException as exc:
        analysis_counter.add(1, {"mode": "area", "outcome": f"http_{exc.status_code}"})
        raise

    rainfall = output.rainfall
    timings["total_s"] = time.perf_counter() - request_started
    timings = {name: round(value, 2) for name, value in timings.items()}

    candidates = output.candidate_responses
    best = candidates[0] if candidates else None

    analysis_counter.add(1, {"mode": "area", "outcome": "success"})
    analysis_duration.record(timings["total_s"], {"mode": "area"})
    candidates_found.record(len(candidates), {"mode": "area"})
    grid_cells.record(height * width, {"mode": "area"})
    area_km2.record(dem_info.area_km2)
    span.set_attribute("pond.outcome", "success")
    span.set_attribute("pond.area_km2", round(dem_info.area_km2, 3))
    span.set_attribute("pond.candidates.found", len(candidates))
    span.set_attribute("pond.rainfall.annual_mm", round(rainfall.annual_mm, 1))
    if best:
        annual_runoff.record(best["volume"]["annual_runoff_m3"], {"mode": "area"})
        span.set_attribute(
            "pond.best_site.annual_runoff_m3", best["volume"]["annual_runoff_m3"]
        )
    logger.info(
        "Area analysis done: %.2f km2, %d cells, %d ponds, rainfall %.0f mm (%s), "
        "timings %s",
        dem_info.area_km2, height * width, len(candidates),
        rainfall.annual_mm, rainfall.source, timings,
    )

    summary = {
        "pond_count": len(candidates),
        "best_pond": (
            {
                "rank": 1,
                "latitude": best["latitude"],
                "longitude": best["longitude"],
                "catchment_area_m2": best["catchment_area_m2"],
                "catchment_area_hectares": best["catchment_area_hectares"],
                "annual_runoff_m3": best["volume"]["annual_runoff_m3"],
                "storage_capacity_m3": best["volume"]["recommended_pond"][
                    "storage_capacity_m3"
                ],
            }
            if best
            else None
        ),
    }

    message = (
        f"Found {len(candidates)} suitable pond location(s) in the selected area."
        if candidates
        else "No suitable pond location found. Try a larger area or a "
        "higher maximum slope."
    )

    return AreaAnalyzeResponse(
        status="success",
        message=message,
        area={
            "area_km2": dem_info.area_km2,
            "centroid": {"latitude": centroid.y, "longitude": centroid.x},
        },
        terrain={
            "source": dem_info.source,
            "zoom": dem_info.zoom,
            "tile_count": dem_info.tile_count,
            "grid_resolution_m": dem_info.resolution_m,
            "contour_interval_m": contour_interval,
            "width": width,
            "height": height,
            "cell_count": width * height,
            "crs": terrain.crs,
            "min_elevation_m": terrain.min_elevation_m,
            "max_elevation_m": terrain.max_elevation_m,
            "slope_mean_percent": terrain.slope_mean_percent,
            "slope_max_percent": terrain.slope_max_percent,
            "bounds": {
                "min_lon": terrain.bounds_lonlat[0],
                "min_lat": terrain.bounds_lonlat[1],
                "max_lon": terrain.bounds_lonlat[2],
                "max_lat": terrain.bounds_lonlat[3],
            },
        },
        rainfall={
            "annual_mm": rainfall.annual_mm,
            "monthly_mm": rainfall.monthly_mm,
            "source": rainfall.source,
            "measured": rainfall.measured,
            "years": rainfall.years,
        },
        summary=summary,
        timings=timings,
        hydrology={
            "dem_filled": flow.filled_cell_count > 0,
            "filled_cell_count": flow.filled_cell_count,
            "max_fill_depth_m": flow.max_fill_depth_m,
            "valid_cell_count": flow.valid_cell_count,
            "flowing_cell_count": flow.flowing_cell_count,
            "no_flow_cell_count": flow.no_flow_cell_count,
            "d8_algorithm": "D8 steepest-descent",
            "flow_accumulation": output.accumulation_stats,
            "pond_candidates": candidates,
        },
        accumulation=output.accumulation_stats,
        suitability={
            "max_slope_percent": request.max_slope_percent,
            "minimum_accumulation": request.minimum_accumulation,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        analysis={
            "suitability": {
                "max_slope_percent": request.max_slope_percent,
                "minimum_accumulation": request.minimum_accumulation,
                "candidate_count": len(candidates),
            },
            "candidates": candidates,
            "catchments": output.catchment_responses,
        },
        map_data=output.map_data,
    )
