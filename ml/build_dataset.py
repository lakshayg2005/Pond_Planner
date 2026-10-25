"""Turn OpenStreetMap ponds + elevation tiles into a training table.

    python ml/build_dataset.py chhattisgarh   # or: india

For each region, tiles of about 4.4 x 4.4 km that contain mapped ponds are
run through the same terrain pipeline the app uses (elevation download, sink
filling, D8, flow accumulation). Then:

  * positives: the cell at each mapped pond (OSM water polygon, 0.1-20 ha)
  * negatives: random cells at least 250 m from any mapped water body

Writes ml/data/samples_<set>.csv (one row per sample: features, label, and the
hand-weighted formula's score for comparison) and ml/data/tiles_<set>/*.npz (full
grids, used to measure top-10 ranking quality per tile).
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box, shape
from shapely.ops import transform

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
os.environ.setdefault("POND_TILE_CACHE", str(DATA / "elevation_cache"))
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
sys.path.insert(0, str(HERE.parent / "backend"))

from app.services.accumulation_service import calculate_flow_accumulation  # noqa: E402
from app.services.dem_service import build_terrain_from_polygon, polygon_mask  # noqa: E402
from app.services.flow_service import analyze_flow  # noqa: E402
from app.services.site_features import FEATURE_NAMES, compute_feature_grid  # noqa: E402
from app.services.suitability_service import heuristic_scores  # noqa: E402

from regions import CHHATTISGARH_TEST, CHHATTISGARH_TRAIN, REGIONS  # noqa: E402

# Region sets: name -> (regions, tiles per region). Outputs are kept apart:
# data/samples_<set>.csv and data/tiles_<set>/.
REGION_SETS = {
    "india": (REGIONS, 8),
    "chhattisgarh": ({**CHHATTISGARH_TRAIN, **CHHATTISGARH_TEST}, 15),
}

TILE_DEG = 0.04
MAX_TILES_PER_REGION = 8
TILES_DIR = DATA / "tiles_india"
MIN_PONDS_PER_TILE = 3
NEGATIVES_PER_POSITIVE = 3
NEGATIVE_MIN_DISTANCE_M = 250.0
SEED = 42


def _load_water(region: str):
    data = json.loads((DATA / "osm" / f"{region}.geojson").read_text())
    polygons = [shape(f["geometry"]) for f in data["features"]]
    is_pond = [f["properties"]["is_pond"] for f in data["features"]]
    return polygons, is_pond


def _choose_tiles(bbox, ponds, rng):
    west, south, east, north = bbox
    tiles = []
    lon = west
    while lon + TILE_DEG <= east + 1e-9:
        lat = south
        while lat + TILE_DEG <= north + 1e-9:
            tile = box(lon, lat, lon + TILE_DEG, lat + TILE_DEG)
            inside = sum(tile.contains(p.representative_point()) for p in ponds)
            if inside >= MIN_PONDS_PER_TILE:
                tiles.append(tile)
            lat += TILE_DEG
        lon += TILE_DEG
    rng.shuffle(tiles)
    return tiles[:MAX_TILES_PER_REGION]


def _rasterise(polygons, terrain, to_utm):
    half = terrain.grid_resolution_m / 2.0
    gx, gy = np.meshgrid(terrain.x_m + half, terrain.y_m + half)
    mask = np.zeros(gx.shape, dtype=bool)
    for polygon in polygons:
        projected = transform(to_utm.transform, polygon)
        mask |= shapely.contains_xy(projected, gx, gy)
        # Small ponds may contain no cell centre: mark the nearest cell.
        point = projected.representative_point()
        col = int((point.x - terrain.x_m[0]) // terrain.grid_resolution_m)
        row = int((point.y - terrain.y_m[0]) // terrain.grid_resolution_m)
        if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
            mask[row, col] = True
    return mask


def process_tile(region, index, tile, water, is_pond, rng, writer):
    terrain, _ = build_terrain_from_polygon(tile)
    res = terrain.grid_resolution_m
    to_utm = Transformer.from_crs("EPSG:4326", terrain.crs, always_xy=True)

    flow = analyze_flow(terrain.elevation_grid_m, cell_size_x_m=res, cell_size_y_m=res)
    accumulation = calculate_flow_accumulation(flow.flow_direction)
    grids = heuristic_scores(
        flow.filled_dem_m,
        terrain.slope_grid_percent,
        accumulation,
        flow_direction=flow.flow_direction,
    )
    features = compute_feature_grid(
        terrain.elevation_grid_m, terrain.slope_grid_percent, accumulation, res
    )

    nearby = [p for p in water if p.intersects(tile.buffer(0.01))]
    nearby_ponds = [
        p for p, pond in zip(water, is_pond) if pond and p.intersects(tile.buffer(0.01))
    ]
    water_mask = _rasterise(nearby, terrain, to_utm)
    pond_mask = _rasterise(nearby_ponds, terrain, to_utm)
    inside = polygon_mask(terrain, tile)
    water_distance = distance_transform_edt(~water_mask) * res
    pond_distance = distance_transform_edt(~pond_mask) * res

    tile_id = f"{region}_{index}"
    rows = []
    for polygon in nearby_ponds:
        point = transform(to_utm.transform, polygon.representative_point())
        col = int((point.x - terrain.x_m[0]) // res)
        row = int((point.y - terrain.y_m[0]) // res)
        if 0 <= row < inside.shape[0] and 0 <= col < inside.shape[1] and inside[row, col]:
            rows.append((row, col, 1))

    candidates = np.argwhere(inside & (water_distance >= NEGATIVE_MIN_DISTANCE_M))
    count = min(len(candidates), NEGATIVES_PER_POSITIVE * len(rows))
    for row, col in candidates[rng.sample(range(len(candidates)), count)]:
        rows.append((int(row), int(col), 0))

    for row, col, label in rows:
        writer.writerow(
            [region, tile_id, row, col, label, round(float(grids.score[row, col]), 5)]
            + [round(float(v), 4) for v in features[row, col]]
        )

    np.savez_compressed(
        TILES_DIR / f"{tile_id}.npz",
        features=features,
        heuristic=grids.score.astype(np.float32),
        valid=grids.valid,
        inside=inside,
        pond_distance=pond_distance.astype(np.float32),
        resolution=res,
    )
    positives = sum(r[2] for r in rows)
    return positives, len(rows) - positives


def main(region_set: str):
    global MAX_TILES_PER_REGION, TILES_DIR
    regions, MAX_TILES_PER_REGION = REGION_SETS[region_set]
    TILES_DIR = DATA / f"tiles_{region_set}"
    rng = random.Random(SEED)
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA / f"samples_{region_set}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["region", "tile", "row", "col", "label", "heuristic_score", *FEATURE_NAMES]
        )
        for region, bbox in regions.items():
            water, is_pond = _load_water(region)
            ponds = [p for p, pond in zip(water, is_pond) if pond]
            tiles = _choose_tiles(bbox, ponds, rng)
            for index, tile in enumerate(tiles):
                started = time.perf_counter()
                try:
                    pos, neg = process_tile(region, index, tile, water, is_pond, rng, writer)
                except Exception as exc:  # one bad tile should not stop the run
                    print(f"{region}_{index}: skipped ({exc})", flush=True)
                    continue
                print(
                    f"{region}_{index}: {pos} ponds, {neg} negatives "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "india")
