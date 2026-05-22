from __future__ import annotations

import logging
from pathlib import Path

from api import jobs


def test_run_job_writes_durable_log_and_hydrates_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    tmp_path.mkdir(exist_ok=True)

    job = jobs.create_job("unit", {"x": 1})

    def work():
        logging.getLogger("unit-test").info("hello from job")
        return {"ok": True}

    jobs.run_job(job["id"], work)

    loaded = jobs.get_job(job["id"])
    assert loaded["state"] == "done"
    assert loaded["result"] == {"ok": True}
    assert Path(loaded["log_path"]).exists()
    assert any("hello from job" in line for line in loaded["logs"])


def test_get_job_reads_new_lines_from_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    job = jobs.create_job("unit", {})
    Path(job["log_path"]).write_text("line one\nline two\n", encoding="utf-8")

    loaded = jobs.get_job(job["id"])

    assert loaded["logs"][-2:] == ["line one", "line two"]
