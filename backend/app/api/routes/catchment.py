from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from opentelemetry import trace

from ...schemas.catchment import CatchmentAnalyzeResponse
from ...services.analysis_pipeline import run_hydrology, stage
from ...services.concurrency import run_exclusive
from ...services.kml_parser import parse_kml_bytes
from ...services.rainfall_service import lookup_async, manual_rainfall
from ...services.terrain_service import reconstruct_dem
from ...telemetry import (
    analysis_counter,
    analysis_duration,
    annual_runoff,
    candidates_found,
    grid_cells,
    rainfall_lookups,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/catchment",
    tags=["Catchment Analysis"],
)

_ALLOWED_EXTENSIONS = {".kml", ".kmz"}

# Protects the small (512 MB) deployment host from oversized uploads.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


@router.post(
    "/analyze",
    response_model=CatchmentAnalyzeResponse,
)
async def analyze_catchment(
    contour_map: UploadFile = File(
        ...,
        description="Contour map in KML or KMZ format",
        alias="contour_map",
    ),
    grid_resolution_m: float = Query(
        10.0,
        gt=0,
        le=100,
        description="DEM cell size in metres",
    ),
    sample_spacing_m: float | None = Query(
        None,
        gt=0,
        le=100,
    ),
    interpolation_method: str = Query(
        "linear",
        pattern="^(linear|nearest)$",
    ),
    max_slope_percent: float = Query(
        8.0,
        gt=0,
        le=100,
    ),
    minimum_accumulation: int = Query(
        10,
        ge=1,
    ),
    max_candidates: int = Query(
        10,
        ge=1,
        le=100,
    ),
    minimum_distance_cells: int = Query(
        10,
        ge=0,
    ),
    rainfall_mm: float | None = Query(
        None,
        gt=0,
        le=12000,
        description="Annual rainfall override; looked up automatically if omitted",
    ),
    runoff_coefficient: float | None = Query(
        None,
        gt=0,
        le=1,
        description="Runoff coefficient; estimated from slope if omitted",
    ),
) -> CatchmentAnalyzeResponse:

    # =========================================================
    # Validate file extension
    # =========================================================

    span = trace.get_current_span()
    request_started = time.perf_counter()

    filename = contour_map.filename or ""
    span.set_attribute("pond.upload.filename", filename)

    extension = (
        "." + filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if extension not in _ALLOWED_EXTENSIONS:
        analysis_counter.add(1, {"mode": "contour", "outcome": "bad_file_type"})
        logger.warning("Rejected upload %r: not a .kml/.kmz file", filename)
        raise HTTPException(
            status_code=400,
            detail="Only .kml and .kmz files are supported",
        )

    try:
        # Read one byte past the limit so oversized files are rejected
        # without buffering them entirely.
        data = await contour_map.read(MAX_UPLOAD_BYTES + 1)

        span.set_attribute("pond.upload.bytes", len(data))
        if len(data) > MAX_UPLOAD_BYTES:
            analysis_counter.add(1, {"mode": "contour", "outcome": "too_large"})
            logger.warning("Rejected upload %r: larger than the limit", filename)
            raise HTTPException(
                status_code=413,
                detail=(
                    "File is too large. The maximum upload size is "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                ),
            )

        if not data:
            raise ValueError("Uploaded file is empty")

        with stage("kml.parse") as parse_span:
            contours = parse_kml_bytes(data, filename)
            parse_span.set_attribute("pond.contours.count", len(contours))
        logger.info(
            "Contour analysis started: %r, %d bytes, %d contour lines",
            filename, len(data), len(contours),
        )

        if not contours:
            raise ValueError(
                "No valid LineString contours with numeric "
                "elevations were found"
            )

        # Start the rainfall lookup now so it runs while the (slow) terrain
        # reconstruction is in progress.
        lons = [lon for c in contours for lon, _ in c.coordinates]
        lats = [lat for c in contours for _, lat in c.coordinates]
        centre_lon = (min(lons) + max(lons)) / 2.0
        centre_lat = (min(lats) + max(lats)) / 2.0

        if rainfall_mm is not None:
            rainfall_lookups.add(1, {"source": "user_supplied"})
            rainfall_source = manual_rainfall(rainfall_mm)
        else:
            rainfall_source = lookup_async(centre_lat, centre_lon)

        def reconstruct(*args, **kwargs):
            with stage("terrain.reconstruct_from_contours"):
                return reconstruct_dem(*args, **kwargs)

        terrain = await run_exclusive(
            reconstruct,
            contours,
            grid_resolution_m=grid_resolution_m,
            sample_spacing_m=sample_spacing_m,
            method=interpolation_method,
        )

        output = await run_exclusive(
            run_hydrology,
            terrain,
            max_slope_percent=max_slope_percent,
            minimum_accumulation=minimum_accumulation,
            max_candidates=max_candidates,
            minimum_distance_cells=minimum_distance_cells,
            rainfall=rainfall_source,
            runoff_coefficient=runoff_coefficient,
        )
        rainfall = output.rainfall

        flow = output.flow
        candidate_responses = output.candidate_responses
        accumulation_stats = output.accumulation_stats
        candidates = output.candidates

        # =====================================================
        # 11. Terrain statistics
        # =====================================================

        elevations = {
            round(c.elevation_m, 6)
            for c in contours
        }

        cell_count = int(
            terrain.elevation_grid_m.size
        )

        nan_percentage = (
            100.0
            * (
                cell_count
                - terrain.valid_cell_count
            )
            / cell_count
            if cell_count > 0
            else 0.0
        )

        height, width = (
            terrain.elevation_grid_m.shape
        )

        elapsed = time.perf_counter() - request_started
        analysis_counter.add(1, {"mode": "contour", "outcome": "success"})
        analysis_duration.record(elapsed, {"mode": "contour"})
        candidates_found.record(len(candidates), {"mode": "contour"})
        grid_cells.record(cell_count, {"mode": "contour"})
        span.set_attribute("pond.outcome", "success")
        span.set_attribute("pond.candidates.found", len(candidates))
        span.set_attribute("pond.contour_rmse_m", float(terrain.contour_rmse_m))
        if candidate_responses and candidate_responses[0]["volume"]:
            annual_runoff.record(
                candidate_responses[0]["volume"]["annual_runoff_m3"],
                {"mode": "contour"},
            )
        logger.info(
            "Contour analysis done in %.1f s: %d cells at %.0f m, %d ponds, "
            "contour RMSE %.2f m, rainfall %.0f mm (%s)",
            elapsed, cell_count, terrain.grid_resolution_m, len(candidates),
            terrain.contour_rmse_m, rainfall.annual_mm, rainfall.source,
        )

        # =====================================================
        # 13. Final API response
        # =====================================================

        return CatchmentAnalyzeResponse(
            status="success",

            message=(
                "Contour terrain parsed, reconstructed, "
                "validated, slope calculated, sink-filled, "
                "processed with D8 flow direction, flow "
                "accumulation calculated, pond candidates "
                "ranked, and catchments delineated"
            ),

            terrain={
                "contour_count": len(contours),

                "elevation_levels": len(
                    elevations
                ),

                "min_elevation_m": (
                    terrain.min_elevation_m
                ),

                "max_elevation_m": (
                    terrain.max_elevation_m
                ),

                "grid_resolution_m": (
                    terrain.grid_resolution_m
                ),

                "contour_sample_count": (
                    terrain.sample_count
                ),

                "valid_dem_cells": (
                    terrain.valid_cell_count
                ),

                "crs": terrain.crs,

                "bounds": {
                    "min_lon": terrain.bounds_lonlat[0],
                    "min_lat": terrain.bounds_lonlat[1],
                    "max_lon": terrain.bounds_lonlat[2],
                    "max_lat": terrain.bounds_lonlat[3],
                },

                "dem": {
                    "width": width,
                    "height": height,
                    "cell_count": cell_count,
                    "valid_cell_count": (
                        terrain.valid_cell_count
                    ),
                    "nan_percentage": (
                        nan_percentage
                    ),
                    "resolution_m": (
                        terrain.grid_resolution_m
                    ),
                    "crs": terrain.crs,
                },

                "slope": {
                    "min_percent": (
                        terrain.slope_min_percent
                    ),
                    "max_percent": (
                        terrain.slope_max_percent
                    ),
                    "mean_percent": (
                        terrain.slope_mean_percent
                    ),
                },

                "contour_validation": {
                    "rmse_m": (
                        terrain.contour_rmse_m
                    ),
                    "max_abs_error_m": (
                        terrain.contour_max_abs_error_m
                    ),
                    "p95_abs_error_m": (
                        terrain.contour_p95_abs_error_m
                    ),
                },
            },

            hydrology={
                "dem_filled": (
                    flow.filled_cell_count > 0
                ),

                "filled_cell_count": (
                    flow.filled_cell_count
                ),

                "max_fill_depth_m": (
                    flow.max_fill_depth_m
                ),

                "valid_cell_count": (
                    flow.valid_cell_count
                ),

                "flowing_cell_count": (
                    flow.flowing_cell_count
                ),

                "no_flow_cell_count": (
                    flow.no_flow_cell_count
                ),

                "d8_algorithm": (
                    "D8 steepest-descent"
                ),

                "flow_accumulation": (
                    accumulation_stats
                ),

                "pond_candidates": (
                    candidate_responses
                ),
            },

            accumulation=accumulation_stats,

            suitability={
                "max_slope_percent": (
                    max_slope_percent
                ),

                "minimum_accumulation": (
                    minimum_accumulation
                ),

                "candidate_count": len(
                    candidates
                ),

                "candidates": (
                    candidate_responses
                ),
            },

            analysis={
                "suitability": {
                    "max_slope_percent": (
                        max_slope_percent
                    ),

                    "minimum_accumulation": (
                        minimum_accumulation
                    ),

                    "candidate_count": len(
                        candidates
                    ),
                },

                "candidates": (
                    candidate_responses
                ),

                "catchments": (
                    output.catchment_responses
                ),
            },

            map_data=output.map_data,

            rainfall={
                "annual_mm": rainfall.annual_mm,
                "monthly_mm": rainfall.monthly_mm,
                "source": rainfall.source,
                "measured": rainfall.measured,
                "years": rainfall.years,
            },
        )

    except HTTPException as exc:
        if exc.status_code >= 500:
            analysis_counter.add(
                1, {"mode": "contour", "outcome": f"http_{exc.status_code}"}
            )
        raise

    except ValueError as exc:
        analysis_counter.add(1, {"mode": "contour", "outcome": "invalid_file"})
        span.record_exception(exc)
        logger.warning("Contour file %r rejected: %s", filename, exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        analysis_counter.add(1, {"mode": "contour", "outcome": "error"})
        logger.exception("Contour analysis of %r failed", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Terrain analysis failed: {exc}",
        ) from exc

