"""Terrain features for the learned pond-site model.

Every feature describes the land *around* a cell at a fixed distance in
metres, so the same model works on 20 m and 30 m grids. Features that look
only at the cell itself (its own slope, its own fill depth) are avoided on
purpose: an existing pond flattens the elevation model under its water
surface, and a model trained on mapped ponds would otherwise learn to
recognise "flat water" instead of "good place for a pond".
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

FEATURE_NAMES = (
    "log_upstream_area",        # log of contributing area at the cell (m2)
    "log_upstream_area_100m",   # largest contributing area within 100 m
    "slope_ring_60_200m",       # mean slope (%) 60-200 m away
    "slope_ring_200_500m",      # mean slope (%) 200-500 m away
    "tpi_150m",                 # elevation minus local mean (m), 150 m
    "tpi_500m",                 # same at 500 m: valley floor vs ridge
    "relief_250m",              # highest point within 250 m, above the cell (m)
    "relief_500m",
    "ruggedness_500m",          # max - min elevation within 500 m (m)
    "wetness_index",            # ln(upstream area / tan(surrounding slope))
)


def rank_within_area(features: np.ndarray, area_mask: np.ndarray) -> np.ndarray:
    """Replace each feature by its percentile (0-1) among the area's cells.

    Absolute values do not transfer between landscapes: 3 m of relief is a
    hill on the Chhattisgarh plains and nothing on the Deccan plateau. A
    percentile ("lower than 80% of this area") means the same everywhere.
    """
    values = features[area_mask]
    ranked = np.empty_like(features, dtype=np.float32)
    for k in range(features.shape[-1]):
        reference = np.sort(values[:, k])
        positions = np.searchsorted(reference, features[..., k], side="right")
        ranked[..., k] = positions / max(len(reference), 1)
    return ranked


def _cells(metres: float, resolution_m: float) -> int:
    return max(1, int(round(metres / resolution_m)))


def _window_mean(values: np.ndarray, radius: int) -> np.ndarray:
    return uniform_filter(values, size=2 * radius + 1, mode="nearest")


def _ring_mean(values: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Mean over the square ring between two radii (cells)."""
    big = 2 * outer + 1
    small = 2 * inner + 1
    total = _window_mean(values, outer) * big * big
    core = _window_mean(values, inner) * small * small
    return (total - core) / (big * big - small * small)


def compute_feature_grid(
    elevation_m: np.ndarray,
    slope_percent: np.ndarray,
    flow_accumulation: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """Feature stack of shape (rows, cols, len(FEATURE_NAMES)), float32.

    elevation_m should be the original (not sink-filled) elevation grid.
    """
    dem = np.asarray(elevation_m, dtype=np.float64)
    slope = np.nan_to_num(np.asarray(slope_percent, dtype=np.float64), nan=0.0)
    accumulation = np.maximum(np.asarray(flow_accumulation, dtype=np.float64), 1.0)
    dem = np.where(np.isfinite(dem), dem, np.nanmean(dem))

    cell_area = resolution_m * resolution_m
    log_area = np.log(accumulation * cell_area)
    log_area_100 = maximum_filter(
        log_area, size=2 * _cells(100, resolution_m) + 1, mode="nearest"
    )

    r60, r200, r500 = (_cells(m, resolution_m) for m in (60, 200, 500))
    slope_near = _ring_mean(slope, r60, max(r200, r60 + 1))
    slope_far = _ring_mean(slope, r200, max(r500, r200 + 1))

    tpi_150 = dem - _window_mean(dem, _cells(150, resolution_m))
    tpi_500 = dem - _window_mean(dem, r500)

    size_250 = 2 * _cells(250, resolution_m) + 1
    size_500 = 2 * r500 + 1
    max_250 = maximum_filter(dem, size=size_250, mode="nearest")
    max_500 = maximum_filter(dem, size=size_500, mode="nearest")
    min_500 = minimum_filter(dem, size=size_500, mode="nearest")

    wetness = log_area_100 - np.log(np.maximum(slope_near / 100.0, 0.001))

    features = np.stack(
        [
            log_area,
            log_area_100,
            slope_near,
            slope_far,
            tpi_150,
            tpi_500,
            max_250 - dem,
            max_500 - dem,
            max_500 - min_500,
            wetness,
        ],
        axis=-1,
    )
    return features.astype(np.float32)
