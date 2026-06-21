from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Results live at project_root/results/; this file is at project_root/src/visuals/
_RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _load(filename: str) -> Any:
    path = _RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def load_frequency_sweep() -> list[dict]:
    return _load("frequency_sweep_results.json")


def load_perturbation_gonogo() -> list[dict]:
    return _load("perturbation_sweep_gonogo.json")


def load_perturbation_stopsignal() -> list[dict]:
    return _load("perturbation_sweep_stopsignal.json")


def load_cerebellum_results() -> list[dict]:
    return _load("cerebellum_results.json")


def load_bg_validation() -> list[dict]:
    return _load("bg_validation.json")


def load_opensim_gonogo() -> list[dict]:
    return _load("opensim_gonogo_sweep.json")
