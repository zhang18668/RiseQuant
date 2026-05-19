"""Data router: fetch / clean / dryrun jobs."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from api import jobs as jobs_mod
from api import runner
from api.schemas import CleanRequest, DryRunRequest, FetchRequest

router = APIRouter()


@router.post("/fetch")
def submit_fetch(req: FetchRequest, background: BackgroundTasks):
    job = jobs_mod.create_job("fetch", req.model_dump())
    background.add_task(
        jobs_mod.run_job, job["id"], runner.run_fetch,
        source=req.source, tdx_path=req.tdx_path or r"C:\new_tdx\vipdoc",
        start=req.start, end=req.end,
        main_board_only=req.main_board_only, code_list=req.code_list,
    )
    return job


@router.post("/clean")
def submit_clean(req: CleanRequest, background: BackgroundTasks):
    parent = jobs_mod.get_job(req.job_id)
    if parent is None or not parent.get("result"):
        return {"error": "parent fetch job not found or not finished"}
    cache_path = parent["result"]["cache_path"]
    job = jobs_mod.create_job("clean", req.model_dump())
    background.add_task(
        jobs_mod.run_job, job["id"], runner.run_clean,
        cache_path=cache_path,
        exclude_st=req.exclude_st,
        max_abs_change_pct=req.max_abs_change_pct,
    )
    return job


@router.post("/dryrun")
def submit_dryrun(req: DryRunRequest, background: BackgroundTasks):
    if req.fetch_job_id:
        parent = jobs_mod.get_job(req.fetch_job_id)
        start = parent["params"]["start"]
        end = parent["params"]["end"]
        tdx = parent["params"]["tdx_path"] or r"C:\new_tdx\vipdoc"
    else:
        start, end = req.start, req.end
        tdx = req.tdx_path or r"C:\new_tdx\vipdoc"
    job = jobs_mod.create_job("dryrun", req.model_dump())
    background.add_task(
        jobs_mod.run_job, job["id"], runner.run_dryrun,
        start=start, end=end, tdx_path=tdx,
        golden_lo=req.golden_lo, golden_hi=req.golden_hi,
        target_lo=req.target_lo, target_hi=req.target_hi,
        horizon=req.horizon, max_gap=req.max_gap,
        min_events=req.min_events,
    )
    return job
