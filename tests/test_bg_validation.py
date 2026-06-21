"""Dry-run structural test for bg_validation experiment (Phase 12, Task 12.1)."""
from __future__ import annotations

import importlib.util


def test_bg_validation_module_is_importable() -> None:
    spec = importlib.util.find_spec("nrp_bga_sb.bg_model")
    assert spec is not None


def test_run_bg_validation_returns_three_conditions() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    from bg_validation import run_bg_validation  # type: ignore[import]

    results = run_bg_validation()
    assert len(results) == 3
    labels = [r["conflict_level"] for r in results]
    assert labels == ["low", "medium", "high"]


def test_bg_validation_low_conflict_selects_correctly() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    from bg_validation import run_bg_validation  # type: ignore[import]

    results = run_bg_validation()
    low = next(r for r in results if r["conflict_level"] == "low")
    assert low["selection_accuracy"] == 1.0
    assert low["mean_selection_latency_ms"] is not None
    assert low["mean_selection_latency_ms"] < 20.0  # 13.0 ms expected


def test_bg_validation_high_conflict_suppresses() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    from bg_validation import run_bg_validation  # type: ignore[import]

    results = run_bg_validation()
    high = next(r for r in results if r["conflict_level"] == "high")
    assert high["n_selections"] == 0
