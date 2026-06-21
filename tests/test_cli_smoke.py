"""CLI smoke test: verify generate_visuals.py --dry-run exits cleanly."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def test_cli_dry_run_exits_zero():
    """--all --dry-run: all modules import correctly and no I/O is performed."""
    env = {**os.environ, "PYTHONPATH": str(_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, str(_ROOT / "experiments" / "generate_visuals.py"),
         "--all", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=env,
    )
    assert result.returncode == 0, f"CLI exited {result.returncode}:\n{result.stderr}"
    assert "Done." in result.stdout
    assert "DRY RUN" in result.stdout
