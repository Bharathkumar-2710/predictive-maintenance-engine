"""
FastAPI application with static files and modern dashboard UI.
"""

import sys
import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from datetime import datetime

from src.api.routes import router

app = FastAPI(
    title="Industrial IoT Predictive Maintenance System",
    description="High-performance end-to-end monitoring system for predicting mechanical failures",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount static files
static_dir = os.path.join(PROJECT_ROOT, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the dashboard UI."""
    html_path = os.path.join(PROJECT_ROOT, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api")
def api_info():
    """API entry point."""
    return {
        "name": "Industrial IoT Predictive Maintenance System",
        "role": "Lead Python Developer & Data Analyst",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)