"""Lightweight async job queue using FastAPI BackgroundTasks + JSON status files.

Each job:
  - id: uuid string
  - state: pending / running / done / failed
  - kind: train / backtest / dryrun / optimize / fetch / clean
  - params: input dict
  - result: output dict (when done)
  - logs: list of log lines (tail kept in memory + flushed to file)
  - created_at / started_at / finished_at
  - error: error message + traceback (if failed)
"""
from __future__ import annotations

import io
import json
import logging
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


JOBS_DIR = Path(__file__).resolve().parent.parent / "models" / "_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat()


def _save(job: Dict[str, Any]) -> None:
    fp = JOBS_DIR / f"{job['id']}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2, default=str)


def list_jobs(kind: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """List jobs (memory + on-disk)."""
    out: List[Dict[str, Any]] = []
    seen = set()
    with _LOCK:
        for j in _REGISTRY.values():
            if kind is None or j.get("kind") == kind:
                out.append(j)
                seen.add(j["id"])
    if JOBS_DIR.exists():
        for fp in sorted(JOBS_DIR.glob("*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True):
            jid = fp.stem
            if jid in seen:
                continue
            try:
                with open(fp, encoding="utf-8") as f:
                    j = json.load(f)
            except Exception:
                continue
            if kind is None or j.get("kind") == kind:
                out.append(j)
    out.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return out[:limit]


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        if job_id in _REGISTRY:
            return _REGISTRY[job_id]
    fp = JOBS_DIR / f"{job_id}.json"
    if fp.exists():
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    return None


def create_job(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid,
        "kind": kind,
        "state": "pending",
        "params": dict(params),
        "result": None,
        "logs": [],
        "error": None,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
    }
    with _LOCK:
        _REGISTRY[jid] = job
    _save(job)
    return job


def run_job(job_id: str, func: Callable[..., Any], **kwargs) -> None:
    """Run `func(**kwargs)` in current thread, capturing logs + result + error.

    Designed to be invoked through BackgroundTasks or a worker thread.
    """
    with _LOCK:
        job = _REGISTRY.get(job_id)
    if job is None:
        job = get_job(job_id) or {}
        with _LOCK:
            _REGISTRY[job_id] = job

    job["state"] = "running"
    job["started_at"] = _now()
    _save(job)

    # Capture logs to in-memory stream and tail into job["logs"]
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        result = func(**kwargs)
        job["result"] = result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result)
        job["state"] = "done"
    except Exception as e:
        job["error"] = {"type": type(e).__name__, "msg": str(e),
                        "traceback": traceback.format_exc()}
        job["state"] = "failed"
    finally:
        root_logger.removeHandler(handler)
        log_buf.seek(0)
        text = log_buf.getvalue()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        job["logs"] = lines[-500:]   # keep last 500
        job["finished_at"] = _now()
        _save(job)
