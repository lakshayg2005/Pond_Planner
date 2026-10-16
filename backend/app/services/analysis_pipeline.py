from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
from pyproj import Transformer

from ..telemetry import stage_duration, tracer
from .accumulation_service import (
    accumulation_statistics,
    calculate_flow_accumulation,
)
from .catchment_service import catchment_to_geojson, delineate_catchment
from .flow_service import FlowResult, analyze_flow
from .geojson_service import (
    make_feature,
    make_feature_collection,
    transform_geojson_to_wgs84,
)
from .rainfall_service import RainfallResult
from .suitability_service import PondCandidate, detect_pond_candidates
from .terrain_service import TerrainResult
from .volume_service import estimate_water_volume, suggest_runoff_coefficient


@contextmanager
def stage(name: str):
    """A child span for one pipeline stage, also recorded as a metric."""
    started = time.perf_counter()
    with tracer.start_as_current_span(name) as span:
        yield span
    stage_duration.record(time.perf_counter() - started, {"stage": name})


@dataclass(slots=True)
class HydrologyOutput:
    flow: FlowResult
    accumulation_stats: dict
    candidates: list[PondCandidate]
    candidate_responses: list[dict]
    catchment_responses: list[dict]
    map_data: dict
    rainfall: RainfallResult | None = None


def run_hydrology(
    terrain: TerrainResult,
    *,
    max_slope_percent: float,
    minimum_accumulation: int,
    max_candidates: int,
    minimum_distance_cells: int,
    analysis_mask: np.ndarray | None = None,
    rainfall: RainfallResult | Callable[[], RainfallResult] | None = None,
    runoff_coefficient: float | None = None,
) -> HydrologyOutput:
    """Sink filling, D8 flow, accumulation, pond ranking and catchments.

    analysis_mask: optional boolean grid; pond candidates are only chosen
    where it is True, but flow is computed over the whole terrain so that
    catchments can extend beyond the mask.
    rainfall: when given, each pond also gets a water-volume estimate. May be
    a function returning the rainfall, evaluated only once the terrain work is
    done, so a background lookup can overlap with it.
    """
    with stage("hydrology.fill_sinks_and_d8") as span:
        flow = analyze_flow(
            terrain.elevation_grid_m,
            cell_size_x_m=terrain.grid_resolution_m,
            cell_size_y_m=terrain.grid_resolution_m,
        )
        span.set_attribute("pond.grid.cells", int(terrain.elevation_grid_m.size))
        span.set_attribute("pond.flow.filled_cells", int(flow.filled_cell_count))
        span.set_attribute("pond.flow.max_fill_depth_m", float(flow.max_fill_depth_m))

    with stage("hydrology.flow_accumulation"):
        accumulation = calculate_flow_accumulation(flow.flow_direction)
        accumulation_stats = accumulation_statistics(accumulation)

    slope = np.asarray(terrain.slope_grid_percent, dtype=np.float64)
    candidate_slope = slope
    if analysis_mask is not None:
        # Non-finite slope makes the suitability step skip those cells.
        candidate_slope = np.where(analysis_mask, slope, np.nan)

    with stage("hydrology.rank_pond_sites") as span:
        candidates = detect_pond_candidates(
            elevation_m=flow.filled_dem_m,
            slope_percent=candidate_slope,
            flow_accumulation=accumulation,
            flow_direction=flow.flow_direction,
            max_slope_percent=max_slope_percent,
            minimum_accumulation=minimum_accumulation,
            max_candidates=max_candidates,
            minimum_distance_cells=minimum_distance_cells,
        )
        span.set_attribute("pond.candidates.found", len(candidates))
        span.set_attribute("pond.max_slope_percent", float(max_slope_percent))

    if callable(rainfall):
        rainfall = rainfall()

    to_wgs84 = Transformer.from_crs(
        terrain.crs, "EPSG:4326", always_xy=True
    )
    half = terrain.grid_resolution_m / 2.0

    candidate_responses: list[dict] = []
    catchment_responses: list[dict] = []
    candidate_features: list[dict] = []
    catchment_features: list[dict] = []

    with stage("hydrology.catchments_and_volume") as catchment_span:
        for rank, candidate in enumerate(candidates, start=1):
            longitude, latitude = to_wgs84.transform(
                terrain.x_m[candidate.col] + half,
                terrain.y_m[candidate.row] + half,
            )

            catchment = delineate_catchment(
                flow_direction=flow.flow_direction,
                outlet_row=int(candidate.row),
                outlet_col=int(candidate.col),
                cell_size_x_m=terrain.grid_resolution_m,
                cell_size_y_m=terrain.grid_resolution_m,
                x_m=terrain.x_m,
                y_m=terrain.y_m,
            )

            geometry = transform_geojson_to_wgs84(
                catchment_to_geojson(catchment),
                terrain.crs,
            )

            volume = None
            if rainfall is not None:
                coefficient = runoff_coefficient
                if coefficient is None:
                    mean_slope = (
                        float(np.mean(slope[catchment.mask]))
                        if np.any(catchment.mask)
                        else float(terrain.slope_mean_percent)
                    )
                    coefficient = suggest_runoff_coefficient(mean_slope)
                volume = estimate_water_volume(
                    catchment.area_m2, rainfall, coefficient
                )

            candidate_response = {
                "rank": rank,
                "row": int(candidate.row),
                "col": int(candidate.col),
                "longitude": float(longitude),
                "latitude": float(latitude),
                "elevation_m": float(candidate.elevation_m),
                "slope_percent": float(candidate.slope_percent),
                "flow_accumulation": int(candidate.flow_accumulation),
                "score": float(candidate.score),
                "basin_score": float(candidate.basin_score),
                "storage_score": float(candidate.storage_score),
                "channel_penalty": float(candidate.channel_penalty),
                "non_channel_score": float(candidate.non_channel_score),
                "local_relief_score": float(candidate.local_relief_score),
                "reason": candidate.reason,
                "catchment_area_m2": float(catchment.area_m2),
                "catchment_area_hectares": float(catchment.area_hectares),
                "volume": volume,
            }
            candidate_responses.append(candidate_response)

            catchment_responses.append(
                {
                    "rank": rank,
                    "area_m2": float(catchment.area_m2),
                    "area_hectares": float(catchment.area_hectares),
                    "cell_count": int(catchment.cell_count),
                    "geometry": geometry,
                    "volume": volume,
                }
            )

            properties = {
                "rank": rank,
                "elevation_m": candidate_response["elevation_m"],
                "slope_percent": candidate_response["slope_percent"],
                "flow_accumulation": candidate_response["flow_accumulation"],
                "score": candidate_response["score"],
                "reason": candidate.reason,
                "catchment_area_m2": candidate_response["catchment_area_m2"],
                "catchment_area_hectares": candidate_response[
                    "catchment_area_hectares"
                ],
            }
            if volume is not None:
                properties["annual_runoff_m3"] = volume["annual_runoff_m3"]
                properties["storage_capacity_m3"] = volume["recommended_pond"][
                    "storage_capacity_m3"
                ]

            candidate_features.append(
                make_feature(
                    {
                        "type": "Point",
                        "coordinates": [float(longitude), float(latitude)],
                    },
                    properties,
                )
            )
            catchment_features.append(make_feature(geometry, properties))

        catchment_span.set_attribute("pond.catchments", len(catchment_responses))

    return HydrologyOutput(
        flow=flow,
        accumulation_stats=accumulation_stats,
        candidates=candidates,
        candidate_responses=candidate_responses,
        catchment_responses=catchment_responses,
        map_data={
            "candidates": make_feature_collection(candidate_features),
            "catchments": make_feature_collection(catchment_features),
        },
        rainfall=rainfall,
    )
