"""Bundle browser: list bundles, fetch visualization assets, list backtests."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

router = APIRouter()

ROOT = Path(__file__).resolve().parent.parent.parent


def _list_bundle_dirs(base: Path) -> List[Path]:
    out: List[Path] = []
    if not base.exists():
        return out
    for sub in base.iterdir():
        if not sub.is_dir() or sub.name == "latest":
            continue
        if (sub / "shared" / "shared_meta.json").exists():
            out.append(sub)
        else:
            # walk one more level for walk-forward layouts
            for s2 in sub.iterdir() if sub.is_dir() else []:
                if (s2 / "shared" / "shared_meta.json").exists():
                    out.append(s2)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


@router.get("")
def list_bundles(base: str = "./models/pattern_cluster"):
    base_p = (ROOT / base).resolve() if not Path(base).is_absolute() else Path(base)
    out: List[Dict[str, Any]] = []
    for bd in _list_bundle_dirs(base_p):
        with open(bd / "shared" / "shared_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        ts_path = bd / "training_summary.json"
        summary = None
        if ts_path.exists():
            with open(ts_path, encoding="utf-8") as f:
                summary = json.load(f)
        out.append({
            "bundle_dir": str(bd),
            "name": bd.name,
            "meta": meta,
            "training_summary": summary,
            "has_per_cluster": (bd / "bundle_per_cluster").exists(),
            "has_single_with_cf": (bd / "bundle_single_with_cf").exists(),
            "modified_at": bd.stat().st_mtime,
        })
    return out


@router.get("/detail")
def bundle_detail(bundle_dir: str):
    bd = Path(bundle_dir)
    if not bd.exists():
        raise HTTPException(404, f"bundle not found: {bundle_dir}")
    out: Dict[str, Any] = {"bundle_dir": str(bd)}
    sm = bd / "shared" / "shared_meta.json"
    if sm.exists():
        with open(sm, encoding="utf-8") as f:
            out["shared_meta"] = json.load(f)
    ts = bd / "training_summary.json"
    if ts.exists():
        with open(ts, encoding="utf-8") as f:
            out["training_summary"] = json.load(f)
    # cluster_stats
    cs = bd / "shared" / "cluster_stats.csv"
    if cs.exists():
        import pandas as pd
        out["cluster_stats"] = pd.read_csv(cs).to_dict(orient="records")
    # per-cluster
    pcd = bd / "bundle_per_cluster"
    if pcd.exists():
        per: Dict[str, Any] = {}
        usability = {}
        uf = pcd / "usability_flags.json"
        if uf.exists():
            with open(uf, encoding="utf-8") as f:
                usability = json.load(f)
        for sub in (pcd / "models").iterdir() if (pcd / "models").exists() else []:
            if not sub.is_dir():
                continue
            cid = sub.name.split("_")[1]
            entry: Dict[str, Any] = {"cluster_id": int(cid),
                                     "usable": bool(usability.get(cid, True))}
            tm = sub / "train_metrics.json"
            if tm.exists():
                with open(tm, encoding="utf-8") as f:
                    entry["metrics"] = json.load(f)
            fi = sub / "feature_importance.csv"
            if fi.exists():
                import pandas as pd
                entry["feature_importance"] = pd.read_csv(fi).head(20).to_dict(orient="records")
            per[cid] = entry
        out["per_cluster"] = per
    return out


@router.get("/visualization")
def bundle_visualization(bundle_dir: str, name: str):
    """Return a PNG as base64 string."""
    bd = Path(bundle_dir) / "visualization" / name
    if not bd.exists():
        raise HTTPException(404, f"viz not found: {name}")
    data = bd.read_bytes()
    return {"name": name, "size": len(data),
            "base64": base64.b64encode(data).decode("ascii")}


@router.get("/visualization-list")
def bundle_visualization_list(bundle_dir: str):
    vd = Path(bundle_dir) / "visualization"
    if not vd.exists():
        return {"files": []}
    return {"files": sorted([p.name for p in vd.iterdir() if p.is_file()])}


@router.get("/backtests")
def list_backtests(base: str = "./backtests"):
    base_p = (ROOT / base).resolve() if not Path(base).is_absolute() else Path(base)
    out: List[Dict[str, Any]] = []
    if base_p.exists():
        for run in sorted(base_p.rglob("run_*"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
            if not run.is_dir():
                continue
            entry: Dict[str, Any] = {"run_dir": str(run), "name": run.name,
                                      "modified_at": run.stat().st_mtime}
            mp = run / "backtest_metrics.json"
            if mp.exists():
                with open(mp, encoding="utf-8") as f:
                    entry["metrics"] = json.load(f)
            cp = run / "run_config.json"
            if cp.exists():
                with open(cp, encoding="utf-8") as f:
                    entry["config"] = json.load(f)
            out.append(entry)
    return out


@router.get("/backtest-detail")
def backtest_detail(run_dir: str):
    p = Path(run_dir)
    if not p.exists():
        raise HTTPException(404, "run not found")
    out: Dict[str, Any] = {"run_dir": str(p)}
    for fname in ["backtest_metrics.json", "summary.json", "run_config.json"]:
        f = p / fname
        if f.exists():
            with open(f, encoding="utf-8") as g:
                out[fname.replace(".json", "")] = json.load(g)
    import pandas as pd
    for fname in ["equity_curve.csv", "trades.csv", "signals.csv"]:
        f = p / fname
        if f.exists():
            df = pd.read_csv(f)
            out[fname.replace(".csv", "")] = df.head(1000).to_dict(orient="records")
    return out
