import sys

# Ensure UTF-8 output on Windows consoles to prevent charmap UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.database import init_and_seed_db, get_model_pipeline, get_digital_twin, SessionLocal
from app.models_db import StationDB
from app.routes import trains, stations, network, predictions, whatif, alerts, analytics, model_intel, data_health, demo

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Initializing RAILCAST AI Engine...")
    init_and_seed_db()
    print("[INFO] RAILCAST AI Engine initialized and ready.")
    yield
    print("[INFO] Shutting down RAILCAST AI Engine...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=f"{settings.TAGLINE} — An explainable spatio-temporal intelligence platform for dynamic railway ETA forecasting.",
    lifespan=lifespan
)

# CORS Configuration — allow the local Vite dev server origin
CORS_ORIGINS = [
    "http://localhost:5173",    # Vite default dev port
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers under API Prefix
app.include_router(trains.router, prefix=settings.API_PREFIX)
app.include_router(stations.router, prefix=settings.API_PREFIX)
app.include_router(network.router, prefix=settings.API_PREFIX)
app.include_router(predictions.router, prefix=settings.API_PREFIX)
app.include_router(whatif.router, prefix=settings.API_PREFIX)
app.include_router(alerts.router, prefix=settings.API_PREFIX)
app.include_router(analytics.router, prefix=settings.API_PREFIX)
app.include_router(model_intel.router, prefix=settings.API_PREFIX)
app.include_router(data_health.router, prefix=settings.API_PREFIX)
app.include_router(demo.router, prefix=settings.API_PREFIX)

@app.get("/")
def root():
    return {
        "product": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "tagline": settings.TAGLINE,
        "docs_url": "/docs",
        "health_check": f"{settings.API_PREFIX}/health"
    }

@app.get(f"{settings.API_PREFIX}/health")
def health_check():
    db = SessionLocal()
    try:
        stn_count = db.query(StationDB).count()
        db_status = "CONNECTED" if stn_count > 0 else "EMPTY"
    except Exception as e:
        db_status = f"ERROR: {e}"
    finally:
        db.close()

    model = get_model_pipeline()
    twin = get_digital_twin()

    return {
        "status": "ONLINE",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_version": settings.MODEL_VERSION,
        "database": db_status,
        "model_loaded": bool(model is not None and getattr(model, "is_trained", False)),
        "digital_twin_active": bool(twin is not None)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
