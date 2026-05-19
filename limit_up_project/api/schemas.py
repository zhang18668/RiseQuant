"""Pydantic schemas for request bodies."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FetchRequest(BaseModel):
    source: str = Field("tdx", description="tdx or akshare")
    tdx_path: Optional[str] = Field(None, description="vipdoc path for TDX")
    start: str
    end: str
    code_list: Optional[List[str]] = None
    main_board_only: bool = True


class CleanRequest(BaseModel):
    job_id: str = Field(..., description="parent fetch job id whose result will be cleaned")
    exclude_st: bool = True
    max_abs_change_pct: float = 30.0


class DryRunRequest(BaseModel):
    fetch_job_id: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    tdx_path: Optional[str] = None
    golden_lo: List[float] = [0.10, 0.15, 0.20]
    golden_hi: List[float] = [0.40, 0.35, 0.30]
    target_lo: float = 0.15
    target_hi: float = 0.35
    horizon: int = 22
    max_gap: int = 30
    min_events: int = 300


class TrainRequest(BaseModel):
    start: str
    end: str
    train_end: str
    valid_end: str
    test_end: str
    tdx_path: Optional[str] = None
    golden_lo: float = 0.15
    golden_hi: float = 0.35
    golden_horizon: int = 22
    max_gap: int = 30
    positive_window: int = 5
    k_range: List[int] = [3, 8]
    k_fixed: Optional[int] = None
    bundle_type: str = "both"
    min_train_samples_per_cluster: int = 80
    min_auc_test: float = 0.55
    n_estimators: int = 200
    out: str = "./models/pattern_cluster"
    save_training_data: bool = False


class BacktestRequest(BaseModel):
    bundle_dir: str
    bundle_type: str = "per_cluster"
    bt_start: str
    bt_end: str
    tdx_path: Optional[str] = None
    initial_cash: float = 10_000_000
    topk: int = 5
    sell_n: int = 8
    slippage: float = 0.003
    min_score: float = 0.6
    min_score_per_cluster: Optional[Dict[int, float]] = None
    cooldown: int = 5
    clusters: Optional[List[int]] = None


class OptimizeSearchRequest(BaseModel):
    bundle_dir: str
    bundle_type: str = "per_cluster"
    bt_start: str
    bt_end: str
    tdx_path: Optional[str] = None
    grid: Dict[str, List]   # e.g. {"min_score": [0.5,0.6,0.7], "topk": [3,5,8]}
    n_trials: Optional[int] = None    # if set, use random sample instead of full grid
    objective: str = "sharpe_ratio"    # sharpe_ratio / annual_return / win_rate


class SensitivityRequest(BaseModel):
    bundle_dir: str
    bundle_type: str = "per_cluster"
    bt_start: str
    bt_end: str
    tdx_path: Optional[str] = None
    param: str            # e.g. "min_score"
    values: List[float]
    base_params: Dict[str, float] = {}


class JobView(BaseModel):
    id: str
    kind: str
    state: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    params: Dict
    result: Optional[Dict] = None
    error: Optional[Dict] = None
    logs: List[str] = []
