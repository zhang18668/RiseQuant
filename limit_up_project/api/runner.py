"""Adapters: turn long-running pipelines into pure functions that return dict.

Each runner returns a dict suitable for JSON serialization (no DataFrames).
Used by API endpoints to wrap our scripts as jobs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "models" / "_jobs"


def _tail_text(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _subprocess_log_path(kind: str) -> Path:
    env_path = os.environ.get("RISEQUANT_JOB_LOG_PATH")
    if env_path:
        return Path(env_path)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return JOBS_DIR / f"{kind}_{ts}.log"


def _run_subprocess(cmd: List[str], env: Dict[str, str], kind: str) -> Dict[str, Any]:
    """Run a subprocess and stream its full output into the current job log."""
    log_path = _subprocess_log_path(kind)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"[{datetime.now().isoformat(timespec='seconds')}] subprocess start\n")
        log.write("cwd: " + str(ROOT) + "\n")
        log.write("cmd: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] subprocess exit {proc.returncode}\n")
    return {
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "output_tail": _tail_text(log_path),
    }


# ============================================================
def _load_daily(tdx_path: str, start: str, end: str,
                main_board_only: bool = True,
                code_list: Optional[List[str]] = None) -> "pd.DataFrame":
    from src.data_fetch import TDXFetcher
    f = TDXFetcher(tdx_path)
    if code_list:
        codes = code_list
    else:
        sh = f.load_stock_list("sh")["code"].tolist()
        sz = f.load_stock_list("sz")["code"].tolist()
        codes = sh + sz
        if main_board_only:
            codes = [c for c in codes if f.is_main_board(c)]
    df = f.load_batch(codes, start_date=start, end_date=end)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
def run_fetch(source: str, tdx_path: str, start: str, end: str,
              main_board_only: bool = True,
              code_list: Optional[List[str]] = None) -> Dict[str, Any]:
    daily = _load_daily(tdx_path or r"C:\new_tdx\vipdoc", start, end,
                        main_board_only, code_list)
    # cache to parquet for downstream
    cache_dir = ROOT / "models" / "_cache_fetch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"fetch_{start}_{end}.parquet"
    try:
        daily.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.with_suffix(".csv")
        daily.to_csv(out_path, index=False)
    return {
        "n_rows": int(len(daily)),
        "n_codes": int(daily["code"].nunique()),
        "date_min": str(daily["date"].min().date()),
        "date_max": str(daily["date"].max().date()),
        "cache_path": str(out_path),
    }


def run_clean(cache_path: str, exclude_st: bool = True,
              max_abs_change_pct: float = 30.0) -> Dict[str, Any]:
    from src.data_clean import DataCleaner, DataValidatorPro
    p = Path(cache_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, parse_dates=["date"])
    cleaner = DataCleaner(exclude_st=exclude_st,
                          max_abs_change_pct=max_abs_change_pct)
    cleaned, rep = cleaner.clean(df)
    val = DataValidatorPro().validate(cleaned)
    out_path = p.with_name(p.stem + "_cleaned.parquet")
    try:
        cleaned.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.with_suffix(".csv")
        cleaned.to_csv(out_path, index=False)
    return {
        "clean_report": rep.to_dict(),
        "validation": val,
        "cache_path": str(out_path),
    }


def run_dryrun(start: str, end: str, tdx_path: str,
                golden_lo: List[float], golden_hi: List[float],
                target_lo: float, target_hi: float,
                horizon: int, max_gap: int, min_events: int) -> Dict[str, Any]:
    from src.event.data_dryrun import DataDryRun
    daily = _load_daily(tdx_path or r"C:\new_tdx\vipdoc", start, end)
    dry = DataDryRun(
        lo_hi_bins=list(zip(golden_lo, golden_hi)),
        target_bin=(target_lo, target_hi),
        horizon=horizon,
        min_events_required=min_events,
        detector_max_gap=max_gap,
    )
    rep = dry.summarize(daily)
    return rep.to_dict()


# ============================================================
def run_train_via_script(req: Dict[str, Any]) -> Dict[str, Any]:
    """Call wash_pattern_train.py as subprocess to avoid heavy in-process imports."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "wash_pattern_train.py"),
        "--start", req["start"], "--end", req["end"],
        "--train-end", req["train_end"],
        "--valid-end", req["valid_end"],
        "--test-end", req["test_end"],
        "--golden-lo", str(req.get("golden_lo", 0.15)),
        "--golden-hi", str(req.get("golden_hi", 0.35)),
        "--golden-horizon", str(req.get("golden_horizon", 22)),
        "--max-gap", str(req.get("max_gap", 30)),
        "--positive-window", str(req.get("positive_window", 5)),
        "--bundle-type", req.get("bundle_type", "both"),
        "--n-estimators", str(req.get("n_estimators", 200)),
        "--min-train-samples-per-cluster", str(req.get("min_train_samples_per_cluster", 80)),
        "--min-auc-test", str(req.get("min_auc_test", 0.55)),
        "--out", req.get("out", "./models/pattern_cluster"),
    ]
    if req.get("tdx_path"):
        cmd += ["--tdx-path", req["tdx_path"]]
    if req.get("k_fixed"):
        cmd += ["--k-fixed", str(req["k_fixed"])]
    if req.get("k_range"):
        cmd += ["--k-range", str(req["k_range"][0]), str(req["k_range"][1])]
    if req.get("save_training_data"):
        cmd += ["--save-training-data"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    logger.info("running: " + " ".join(cmd))
    proc_info = _run_subprocess(cmd, env, kind="train")
    # discover newest run dir in --out
    out_root = Path(req.get("out", "./models/pattern_cluster"))
    bundle_dir = None
    if out_root.exists():
        latest = out_root / "latest"
        if latest.exists():
            bundle_dir = str(latest.resolve())
        else:
            subs = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith(("run_", "window_"))],
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if subs:
                bundle_dir = str(subs[0].resolve())
    return {
        "returncode": proc_info["returncode"],
        "log_path": proc_info["log_path"],
        "output_tail": proc_info["output_tail"],
        "bundle_dir": bundle_dir,
    }


def run_backtest_via_script(req: Dict[str, Any]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "wash_pattern_backtest.py"),
        "--bundle", req["bundle_dir"],
        "--bundle-type", req.get("bundle_type", "per_cluster"),
        "--bt-start", req["bt_start"], "--bt-end", req["bt_end"],
        "--topk", str(req.get("topk", 5)),
        "--sell-n", str(req.get("sell_n", 8)),
        "--slippage", str(req.get("slippage", 0.003)),
        "--min-score", str(req.get("min_score", 0.6)),
        "--cooldown", str(req.get("cooldown", 5)),
        "--initial-cash", str(req.get("initial_cash", 10_000_000)),
    ]
    if req.get("tdx_path"):
        cmd += ["--tdx-path", req["tdx_path"]]
    if req.get("clusters"):
        cmd += ["--clusters", ",".join(str(c) for c in req["clusters"])]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    logger.info("running: " + " ".join(cmd))
    proc_info = _run_subprocess(cmd, env, kind="backtest")
    # find latest backtest run
    backtests_root = ROOT / "backtests"
    bt_dir = None
    if backtests_root.exists():
        runs = sorted(backtests_root.rglob("run_*"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            bt_dir = str(runs[0].resolve())
    metrics = None
    if bt_dir:
        mp = Path(bt_dir) / "backtest_metrics.json"
        if mp.exists():
            try:
                with open(mp, encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                metrics = None
    return {
        "returncode": proc_info["returncode"],
        "log_path": proc_info["log_path"],
        "output_tail": proc_info["output_tail"],
        "backtest_dir": bt_dir,
        "metrics": metrics,
    }
