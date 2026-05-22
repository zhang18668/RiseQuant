from __future__ import annotations

import os
import sys

from api import runner


def test_run_subprocess_streams_output_to_job_log(tmp_path, monkeypatch):
    log_path = tmp_path / "job.log"
    monkeypatch.setenv("RISEQUANT_JOB_LOG_PATH", str(log_path))
    env = os.environ.copy()
    cmd = [
        sys.executable,
        "-c",
        "import sys; print('hello stdout'); print('hello stderr', file=sys.stderr)",
    ]

    result = runner._run_subprocess(cmd, env, kind="unit")

    text = log_path.read_text(encoding="utf-8")
    assert result["returncode"] == 0
    assert result["log_path"] == str(log_path)
    assert "hello stdout" in text
    assert "hello stderr" in text
    assert "subprocess exit 0" in text
