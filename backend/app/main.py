from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from .telemetry import instrument_app, setup_telemetry

# Before the routes are imported, so the services' httpx clients are traced.
setup_telemetry()

from .api.routes.catchment import router as catchment_router  # noqa: E402
from .api.routes.area import router as area_router  # noqa: E402
from .api.routes.otel_proxy import router as otel_router  # noqa: E402
from .services.dem_service import MAX_AREA_KM2  # noqa: E402


app = FastAPI(
    title="The Pond Project API",
    description="Contour-based catchment and pond suitability analysis API",
    version="1.0.0",
)

instrument_app(app)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catchment_router)
app.include_router(area_router)
app.include_router(otel_router)


@app.get("/api/health", tags=["Health"])
def health() -> dict:
    return {"status": "ok", "max_area_km2": MAX_AREA_KM2}


# Serve the built frontend (frontend/dist) from the same origin when present,
# so a single container hosts both the UI and the API.
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
