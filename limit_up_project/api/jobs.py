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
import os
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


JOBS_DIR = Path(__file__).resolve().parent.parent / "models" / "_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
LOG_TAIL_LINES = 500

_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat()


def _save(job: Dict[str, Any]) -> None:
    fp = JOBS_DIR / f"{job['id']}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2, default=str)


def _job_log_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.log"


def _tail_file(path: str | Path, limit: int = LOG_TAIL_LINES) -> List[str]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\r\n") for line in f if line.strip()]
    except OSError:
        return []
    return lines[-limit:]


def _hydrate_logs(job: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh log tail from the durable job log file before returning a job."""
    log_path = job.get("log_path")
    if log_path:
        tail = _tail_file(log_path)
        if tail:
            job = dict(job)
            job["logs"] = tail
    return job


def list_jobs(kind: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """List jobs (memory + on-disk)."""
    out: List[Dict[str, Any]] = []
    seen = set()
    with _LOCK:
        for j in _REGISTRY.values():
            if kind is None or j.get("kind") == kind:
                out.append(_hydrate_logs(j))
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
                out.append(_hydrate_logs(j))
    out.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return out[:limit]


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        if job_id in _REGISTRY:
            return _hydrate_logs(_REGISTRY[job_id])
    fp = JOBS_DIR / f"{job_id}.json"
    if fp.exists():
        with open(fp, encoding="utf-8") as f:
            return _hydrate_logs(json.load(f))
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
        "log_path": str(_job_log_path(jid)),
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
    job["log_path"] = job.get("log_path") or str(_job_log_path(job_id))
    _save(job)

    # Capture Python logging both in memory and in a durable per-job log file.
    log_buf = io.StringIO()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    file_handler = logging.FileHandler(job["log_path"], mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    old_level = root_logger.level
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    root_logger.addHandler(file_handler)

    old_env = {
        "RISEQUANT_JOB_ID": os.environ.get("RISEQUANT_JOB_ID"),
        "RISEQUANT_JOB_LOG_PATH": os.environ.get("RISEQUANT_JOB_LOG_PATH"),
    }
    os.environ["RISEQUANT_JOB_ID"] = job_id
    os.environ["RISEQUANT_JOB_LOG_PATH"] = job["log_path"]

    try:
        logging.getLogger(__name__).info("job %s started", job_id)
        result = func(**kwargs)
        job["result"] = result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result)
        job["state"] = "done"
    except Exception as e:
        job["error"] = {"type": type(e).__name__, "msg": str(e),
                        "traceback": traceback.format_exc()}
        job["state"] = "failed"
    finally:
        root_logger.removeHandler(handler)
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(old_level)
        file_handler.close()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        log_buf.seek(0)
        text = log_buf.getvalue()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        file_lines = _tail_file(job["log_path"])
        job["logs"] = file_lines or lines[-LOG_TAIL_LINES:]
        job["finished_at"] = _now()
        _save(job)
