"""Optimize router: grid/random search + sensitivity analysis.

Both endpoints run multiple backtests, each through wash_pattern_backtest.py.
Heavy: each trial is one full backtest. Use --n-trials to cap.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks

from api import jobs as jobs_mod
from api import runner
from api.schemas import OptimizeSearchRequest, SensitivityRequest

router = APIRouter()


def _run_one_backtest(base_req: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    req = dict(base_req)
    req.update(overrides)
    res = runner.run_backtest_via_script(req)
    metrics = res.get("metrics") or {}
    return {
        "params": overrides,
        "backtest_dir": res.get("backtest_dir"),
        "returncode": res.get("returncode"),
        "metrics_summary": {
            "total_return": metrics.get("total_return"),
            "annual_return": metrics.get("annual_return"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "win_rate": metrics.get("win_rate"),
            "num_trades": metrics.get("num_trades"),
        },
    }


def _run_search_job(req: Dict[str, Any]) -> Dict[str, Any]:
    grid = req["grid"]
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if req.get("n_trials") and len(combos) > req["n_trials"]:
        random.seed(42)
        combos = random.sample(combos, req["n_trials"])
    results: List[Dict[str, Any]] = []
    base_req = {k: v for k, v in req.items() if k not in ("grid", "n_trials", "objective")}
    for combo in combos:
        overrides = {keys[i]: combo[i] for i in range(len(keys))}
        results.append(_run_one_backtest(base_req, overrides))
    # rank by objective
    obj = req.get("objective", "sharpe_ratio")
    ranked = sorted(
        [r for r in results if r["metrics_summary"].get(obj) is not None],
        key=lambda r: r["metrics_summary"][obj], reverse=True,
    )
    best = ranked[0] if ranked else None
    return {
        "n_trials": len(results),
        "objective": obj,
        "best": best,
        "all_results": results,
    }


def _run_sensitivity_job(req: Dict[str, Any]) -> Dict[str, Any]:
    param = req["param"]
    values = req["values"]
    base = dict(req.get("base_params") or {})
    base_req = {k: v for k, v in req.items() if k not in ("param", "values", "base_params")}
    base_req.update(base)
    results: List[Dict[str, Any]] = []
    for v in values:
        overrides = {param: v}
        r = _run_one_backtest(base_req, overrides)
        r["value"] = v
        results.append(r)
    return {
        "param": param,
        "n_points": len(values),
        "results": results,
    }


@router.post("/search")
def submit_search(req: OptimizeSearchRequest, background: BackgroundTasks):
    job = jobs_mod.create_job("optimize_search", req.model_dump())
    background.add_task(
        jobs_mod.run_job, job["id"], _run_search_job,
        req=req.model_dump(),
    )
    return job


@router.post("/sensitivity")
def submit_sensitivity(req: SensitivityRequest, background: BackgroundTasks):
    job = jobs_mod.create_job("optimize_sensitivity", req.model_dump())
    background.add_task(
        jobs_mod.run_job, job["id"], _run_sensitivity_job,
        req=req.model_dump(),
    )
    return job
