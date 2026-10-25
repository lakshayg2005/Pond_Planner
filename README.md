# Pond Planner: pond siting and water-harvest estimation

Given a piece of land, this system suggests where a farm pond should go, outlines
the land that drains into it (the **catchment**), and estimates how much water it
can collect in a year.

In Chhattisgarh, candidate sites are ranked by a **machine-learning model trained on
1,063 real village ponds mapped in OpenStreetMap**. On districts it never saw during
training it scores **ROC-AUC 0.70, against 0.51 for the hand-tuned formula it replaced**
(details: [AI: learned site suitability](#ai-learned-site-suitability)).

Two ways to use it:

| Mode | Input | Terrain comes from |
|---|---|---|
| **Draw an area** | A polygon drawn on the map (anywhere in the world) | Downloaded automatically (free elevation tiles) |
| **Upload KML/KMZ** | A contour-map file | Rebuilt from the contour lines in the file |

Live demo: `http://10.1.75.51:3305/`  (API docs: `/docs`)

---

## How it works

```
polygon or KML/KMZ
      |
      v
 elevation grid (DEM)          AWS Terrain Tiles  or  interpolated contour lines
      |
      v
 fill sinks -> D8 flow direction -> flow accumulation
      |
      v
 rank pond sites: ML model in Chhattisgarh, hand-weighted formula elsewhere
 (both keep the hard rules: slope limit, minimum runoff, not a drainage channel)
      |
      v
 catchment of each site = all cells whose flow ends at the site
      |
      v
 water per year = runoff coefficient x annual rainfall x catchment area
```

* **Elevation (draw mode):** AWS Terrain Tiles (Terrarium PNG), a free, keyless, global
  dataset of about 30 m resolution. Tiles are stitched, converted to a local UTM metre
  grid (20 m cells) and cached on disk.
* **Contour mode:** contour lines are sampled and interpolated into the same kind of grid.
* **Hydrology:** depressions are filled, each cell drains to its steepest downhill
  neighbour (D8), and flow is accumulated to find where water gathers.
* **Site ranking:** cells that pass the hard rules (slope limit, minimum runoff, not a
  strong drainage channel) are ranked by the learned model inside Chhattisgarh, and by a
  hand-weighted formula (runoff supply, low slope, basin shape, relief) elsewhere.
  Candidates are spaced apart.
* **Rainfall:** looked up automatically (NASA POWER 20-year climatology, or Open-Meteo
  ERA5 as a fallback, cached per 0.25 degree cell). It can be overridden.
* **Contours (draw mode):** contour lines shown on the map are generated from the same
  grid, at an automatically chosen interval.

### Assumptions (also shown in the app)

| Assumption | Why |
|---|---|
| Runoff coefficient 0.25 (flat) to 0.55 (steep), estimated from catchment slope | No soil or land-cover data available; can be overridden |
| Pond surface = 3% of its catchment | Common farm-pond rule of thumb; a pond larger than its catchment could fill would sit empty |
| Pond depth 3 m | Holds water through the dry season, limits evaporation, diggable with ordinary machinery |
| Pond volume = half of surface x depth | Pond sides slope inwards, so it holds about half of a straight-sided box |
| Rainfall is a long-term average | A single year can differ a lot |
| Terrain is about 30 m resolution | Bunds, roads, drains and existing ponds are not visible; check sites on the ground |
| "Water per year" is total runoff reaching the site | Evaporation and seepage are not subtracted; the pond can refill several times a year |

---

## AI: learned site suitability

The original ranking was a hand-weighted formula. Scored against where villages
actually built ponds, it turned out to be **no better than chance** (ROC-AUC 0.50). It
was replaced, for Chhattisgarh, by a gradient-boosted tree model that learned from real
ponds.

| | |
|---|---|
| **Labels** | 1,063 village ponds and tanks (0.1-20 ha) from OpenStreetMap in 9 districts; 3,189 random points at least 250 m from any mapped water |
| **Features** | 10 terrain descriptors computed by the app's own pipeline: topographic position at 150/500 m, slope in rings 60-200 m and 200-500 m around the point, relief, ruggedness, upstream area, wetness index. They describe the *surroundings* on purpose, so the model cannot cheat by recognising the flat surface of an existing pond |
| **Model** | scikit-learn `HistGradientBoostingClassifier` (200 trees, depth 4), exported to a 227 KB JSON file and evaluated with plain NumPy on the server, so the 512 MB host never loads scikit-learn. A test checks that the export reproduces scikit-learn's predictions exactly |
| **Evaluation** | Trained on 6 districts. The 3 test districts (Mahasamund, Janjgir-Champa, Rajnandgaon) were fixed before any results were seen, share no land with the training districts, and were scored once |

Results on the held-out districts (399 ponds, 1,596 samples):

| Metric | Learned model | Hand-tuned formula | Random |
|---|---|---|---|
| ROC-AUC | **0.697** | 0.505 | 0.50 |
| Precision-recall AUC | **0.477** | 0.260 | 0.25 |
| Top-10 sites within 100 m of a real pond | **9.3%** | 6.4% | 4.5% |

Leave-one-district-out cross-validation on the training districts gave ROC-AUC 0.702,
consistent with the held-out score. The model beats the formula in every test district
(0.65 / 0.72 / 0.70 vs 0.51 / 0.50 / 0.52). Scoring adds no measurable time to an
analysis (2.34 s vs 2.38 s on a 36,600-cell area).

**Scope.** The model is used only inside Chhattisgarh; elsewhere the app falls back to
the formula. An all-India version trained on 8 regions did *not* generalise (ROC-AUC
0.52): ponds in the Kolar tank cascades sit in valleys, while those on the Chhattisgarh
plains sit on gentle slopes, and in Bankura they are dug next to villages regardless of
terrain. `POND_RANKING=heuristic` forces the formula everywhere.

Method, scripts and full results: [ml/README.md](ml/README.md). Map data
(c) OpenStreetMap contributors, ODbL.

## API

Interactive documentation is served at **`/docs`** (Swagger UI).

### `POST /api/analyze-area`: analyse a drawn area

JSON body:

| Field | Type | Default | Notes |
|---|---|---|---|
| `polygon` | GeoJSON Polygon/Feature, or list of `[lon, lat]` | required | At most 25 km2 |
| `rainfall_mm` | number | auto | Annual rainfall override |
| `runoff_coefficient` | number 0-1 | auto | Estimated from slope if omitted |
| `max_slope_percent` | number | 8 | Steepest land allowed at a pond site (8 = rises 8 m per 100 m) |
| `minimum_accumulation` | integer | 10 | Minimum upstream cells draining to a site |
| `max_candidates` | integer 1-50 | 10 | Number of pond sites returned |
| `minimum_distance_cells` | integer | 10 | Minimum spacing between sites |

```bash
curl -X POST http://10.1.75.51:3305/api/analyze-area \
  -H "Content-Type: application/json" \
  -d '{"polygon":[[81.281,21.259],[81.291,21.259],[81.291,21.268],[81.281,21.268],[81.281,21.259]]}'
```

### `POST /api/catchment/analyze`: analyse an uploaded contour map

`multipart/form-data`, file field **`contour_map`** (`.kml` or `.kmz`, at most 15 MB).
Optional query parameters: `grid_resolution_m` (default 10), `sample_spacing_m`,
`interpolation_method` (`linear` or `nearest`), `max_slope_percent`,
`minimum_accumulation`, `max_candidates`, `minimum_distance_cells`, `rainfall_mm`,
`runoff_coefficient`. Very large maps are processed on a coarser grid automatically.

```bash
curl -X POST -F contour_map=@contours_1m.kml http://10.1.75.51:3305/api/catchment/analyze
```

### `GET /api/health`

Returns `{"status": "ok", "max_area_km2": 25.0}`.

### Response (both analysis routes)

| Field | Contents |
|---|---|
| `status`, `message` | `"success"` and a readable summary |
| `terrain` | Grid size and resolution, elevation range, CRS, bounds; contour interval (draw mode) or contour validation error (upload mode) |
| `rainfall` | `annual_mm`, `monthly_mm` (12 values), `source`, `measured` (false means a default or manual value), `years` |
| `summary` | (draw mode) `pond_count` and the `best_pond` |
| `timings` | (draw mode) seconds spent in terrain, rainfall wait, hydrology and contours |
| `analysis.candidates[]` | Ranked pond sites (below) |
| `analysis.catchments[]` | `rank`, `area_m2`, `geometry` (GeoJSON), `volume` |
| `map_data` | GeoJSON FeatureCollections: `candidates`, `catchments` (and `contours` in draw mode) |

Each pond in `analysis.candidates[]` (illustrative values):

```json
{
  "rank": 1,
  "latitude": 20.5263, "longitude": 81.7823,
  "elevation_m": 425.5, "slope_percent": 5.7, "flow_accumulation": 84,
  "score": 0.71,
  "catchment_area_m2": 84400,
  "volume": {
    "annual_runoff_m3": 64490,
    "annual_runoff_litres": 64490000,
    "runoff_coefficient": 0.45,
    "annual_rainfall_mm": 1389,
    "monthly_runoff_m3": { "Jan": 0, "Feb": 0, "...": 0, "Dec": 0 },
    "recommended_pond": { "surface_area_m2": 2532, "depth_m": 3.0, "storage_capacity_m3": 3798 },
    "times_filled_per_year": 17.0
  }
}
```

(`catchment_area_hectares` is also present for compatibility with the Phase 2 output.)

### Errors

| Status | Meaning |
|---|---|
| 400 | File is not `.kml`/`.kmz` |
| 413 | Upload larger than 15 MB |
| 422 | Invalid polygon, area above the limit, area too small, or unreadable/empty contour file |
| 502 | Elevation tiles could not be downloaded |
| 503 | Server busy: other analyses are queued (retry shortly) |
| 504 | Analysis exceeded the time limit; try a smaller area |

---

## Limits and behaviour under load

The service is designed to run inside one small host (512 MB RAM, 1 vCPU).

| Safeguard | Value |
|---|---|
| Maximum drawn area | 25 km2 (`POND_MAX_AREA_KM2`) |
| Maximum upload | 15 MB |
| Grid cell budget | 120,000 (draw mode), 250,000 (upload mode); larger inputs are coarsened |
| Concurrency | One analysis at a time; others queue (60 s wait limit, 120 s run limit) |
| Caching | Elevation tiles on disk (`POND_TILE_CACHE`, capped by `POND_TILE_CACHE_MB`, default 300), rainfall per 0.25 degree cell |
| Network resilience | Rainfall sources are queried in parallel; tiles are downloaded concurrently with fast retries |

Measured on the deployment container (1 vCPU, 512 MB):

| Request | Time |
|---|---|
| Draw area, 1 km2, cold (first time) | about 9 s |
| Draw area, 24 km2, cold | about 10 s |
| Draw area, 1 km2, warm (cached) | 0.5 s |
| Draw area, 24 km2, warm | 4.5 s |
| 4 simultaneous 24 km2 requests | all succeed, queued: 4.5, 8.6, 11.8, 16.1 s |
| Upload of a 6.4 MB contour file | about 15-27 s |
| Peak memory (server process) | 175 MB (draw mode), 286 MB (after the contour upload) |

---

## Running locally

Requirements: Python 3.11+, Node 20+.

```bash
# backend (from the project root)
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                       # http://localhost:5173, talks to port 8000
```

Tests:

```bash
pip install pytest
pytest
```

Configuration (environment variables): `POND_MAX_AREA_KM2`, `POND_TILE_CACHE`,
`POND_TILE_CACHE_MB`. In development the frontend reads `VITE_API_URL` from
`frontend/.env.development`; in a production build it calls the same origin.

## Deployment (single host)

The backend also serves the built frontend, so one process is enough.

```bash
cd frontend && npm run build                       # produces frontend/dist
# copy backend/ and frontend/dist to the host, then on the host:
python3 -m venv venv && venv/bin/pip install --no-cache-dir -r backend/requirements.txt
nohup venv/bin/uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 3000 --workers 1 > server.log 2>&1 &
```

In the course deployment the container's port 3000 is exposed as 3305.

## Observability (OpenTelemetry)

The backend and the browser emit traces, metrics and logs with OpenTelemetry.

```
browser (OTel Web SDK) --traceparent--> FastAPI app (OTel SDK) --OTLP/HTTP--> Collector --> OpenObserve (UI)
```

* Backend set-up: `backend/app/telemetry.py` (FastAPI, httpx, logging and process
  metrics instrumentation, one span per pipeline stage, `pond.*` metrics).
* Browser: `frontend/src/telemetry.js`; spans are posted to `/otel/v1/traces` and
  forwarded to the collector by the backend.
* Enabled by the standard variables `OTEL_SERVICE_NAME` and
  `OTEL_EXPORTER_OTLP_ENDPOINT`; without an endpoint nothing is exported, but
  console logs still carry `trace_id`. `POND_OTEL_CONSOLE=1` prints spans to stdout.
* Collector config and start scripts: `observability/`. In the course deployment the
  collector and OpenObserve run on a second container; the UI is at
  `http://10.1.75.51:3306`.

## Project structure

```
backend/app/
  main.py                    app, CORS, gzip, health, static frontend
  api/routes/area.py         POST /api/analyze-area
  api/routes/catchment.py    POST /api/catchment/analyze
  schemas/                   response models
  services/
    dem_service.py           tile download, stitching, UTM grid, polygon mask
    terrain_service.py       contour lines -> grid
    kml_parser.py            KML/KMZ parsing
    flow_service.py          sink filling, D8 flow direction
    accumulation_service.py  flow accumulation
    suitability_service.py   pond site ranking
    site_features.py         terrain features for the learned model
    site_model.py            learned site model (NumPy evaluation of exported trees)
    catchment_service.py     catchment delineation
    analysis_pipeline.py     shared hydrology pipeline for both routes
    rainfall_service.py      rainfall lookup, caching, fallbacks
    volume_service.py        runoff volume and pond sizing
    contour_service.py       contour lines from the grid (display)
    concurrency.py           one-at-a-time execution with timeouts
frontend/src/
  App.jsx                    state and layout
  components/                MapView, Sidebar, ResultsPanel, PlaceSearch
  services/api.js            API client
backend/app/models/          site_model.json (trained model)
ml/                          training data, scripts and results for the site model
tests/                       pytest suite
```

`backend/app/api/routes/catchment_v2.py` is an experimental alternative kept for
reference; it is not registered in the application.
