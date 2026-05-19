"""FastAPI entry. Run:

    cd limit_up_project
    set PYTHONPATH=%CD%
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import jobs as jobs_mod
from api.routers import bundles, data, optimize, train_bt

app = FastAPI(
    title="RiseQuant Pattern-Cluster API",
    description="Backend for the Vue frontend (data fetch/clean, train, backtest, optimize).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(train_bt.router, prefix="/api", tags=["train+backtest"])
app.include_router(bundles.router, prefix="/api/bundles", tags=["bundles"])
app.include_router(optimize.router, prefix="/api/optimize", tags=["optimize"])


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/jobs")
def all_jobs(kind: str | None = None, limit: int = 50):
    return jobs_mod.list_jobs(kind=kind, limit=limit)


@app.get("/api/jobs/{job_id}")
def one_job(job_id: str):
    j = jobs_mod.get_job(job_id)
    if j is None:
        from fastapi import HTTPException
        raise HTTPException(404, "job not found")
    return j
