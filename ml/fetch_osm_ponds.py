"""Download mapped water bodies from OpenStreetMap for the training regions.

    python ml/fetch_osm_ponds.py

Writes ml/data/osm/<region>.geojson with every water polygon in the region,
each tagged with whether it counts as a pond (a positive training example).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import transform

from regions import CHHATTISGARH_TEST, CHHATTISGARH_TRAIN, REGIONS

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {
    "User-Agent": "pond-planner-training/1.0 "
    "(+https://github.com/lakshayg2005/AI-based-Village-Pond-Planning-System)",
    "Accept": "application/json",
}
OUT_DIR = Path(__file__).parent / "data" / "osm"

# Village ponds and tanks: big enough to show on a 20-30 m grid, small enough
# to be a pond rather than a lake or a dam reservoir.
MIN_POND_M2 = 1_000
MAX_POND_M2 = 200_000
NOT_PONDS = {"river", "canal", "stream", "oxbow", "wastewater", "stream_pool"}


def _query(bbox: tuple[float, float, float, float]) -> dict:
    west, south, east, north = bbox
    box = f"{south},{west},{north},{east}"
    query = f"""
    [out:json][timeout:180];
    (
      way["natural"="water"]({box});
      way["landuse"="reservoir"]({box});
      way["landuse"="basin"]({box});
    );
    out geom;
    """
    for attempt in range(4):
        try:
            response = httpx.post(
                OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=240
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            wait = 20 * (attempt + 1)
            print(f"  overpass error ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Overpass API unavailable")


def _area_m2(polygon: Polygon) -> float:
    lon, lat = polygon.centroid.x, polygon.centroid.y
    zone = int((lon + 180) // 6) + 1
    to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}", always_xy=True
    )
    return transform(to_utm.transform, polygon).area


def fetch_region(name: str, bbox: tuple[float, float, float, float]) -> None:
    out = OUT_DIR / f"{name}.geojson"
    if out.exists():
        print(f"{name}: cached")
        return

    data = _query(bbox)
    features = []
    for element in data.get("elements", []):
        geometry = element.get("geometry") or []
        if len(geometry) < 4:
            continue
        polygon = Polygon([(p["lon"], p["lat"]) for p in geometry])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        tags = element.get("tags", {})
        area = _area_m2(polygon)
        is_pond = (
            tags.get("water") not in NOT_PONDS
            and tags.get("waterway") is None
            and MIN_POND_M2 <= area <= MAX_POND_M2
        )
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "osm_id": element["id"],
                    "water": tags.get("water", ""),
                    "name": tags.get("name", ""),
                    "area_m2": round(area),
                    "is_pond": is_pond,
                },
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    ponds = sum(f["properties"]["is_pond"] for f in features)
    print(f"{name}: {len(features)} water polygons, {ponds} ponds")
    time.sleep(5)  # be polite to the public Overpass server


if __name__ == "__main__":
    every_region = {**REGIONS, **CHHATTISGARH_TRAIN, **CHHATTISGARH_TEST}
    for region_name, region_bbox in every_region.items():
        fetch_region(region_name, region_bbox)
