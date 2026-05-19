"""Train + backtest router."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from api import jobs as jobs_mod
from api import runner
from api.schemas import BacktestRequest, TrainRequest

router = APIRouter()


@router.post("/train")
def submit_train(req: TrainRequest, background: BackgroundTasks):
    job = jobs_mod.create_job("train", req.model_dump())
    background.add_task(
        jobs_mod.run_job, job["id"], runner.run_train_via_script,
        req=req.model_dump(),
    )
    return job


@router.post("/backtest")
def submit_backtest(req: BacktestRequest, background: BackgroundTasks):
    job = jobs_mod.create_job("backtest", req.model_dump())
    background.add_task(
        jobs_mod.run_job, job["id"], runner.run_backtest_via_script,
        req=req.model_dump(),
    )
    return job
