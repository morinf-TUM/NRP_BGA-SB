"""Drive a single NRPCoreSim trial: materialise config + params, set the env
contract, run the binary from the repo root, and parse the JSON-lines log."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_trial(config: dict, params: dict, run_dir: Path) -> list[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    params_path = run_dir / "params.json"
    log_path = run_dir / "trace.jsonl"
    config_path.write_text(json.dumps(config, indent=2))
    params_path.write_text(json.dumps(params))
    if log_path.exists():
        log_path.unlink()

    env = dict(os.environ,
               NRP_BGA_TRIAL_PARAMS=str(params_path),
               NRP_BGA_LOG=str(log_path))
    # -d REPO is REQUIRED: NRPCoreSim cwd's to the experiment dir, so the
    # repo-root-relative PythonFileName/FileName paths only resolve with an
    # explicit experiment root (verified Phase 0).
    # -c accepts an absolute path: is_regular_file() resolves it cwd-independently,
    # so run_dir can live outside the repo (e.g. tmp_path in tests).
    proc = subprocess.run(
        ["NRPCoreSim", "-c", str(config_path), "-d", str(REPO)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"NRPCoreSim failed (rc={proc.returncode}):\n{proc.stderr}")
    if not log_path.exists():
        return []
    return [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
