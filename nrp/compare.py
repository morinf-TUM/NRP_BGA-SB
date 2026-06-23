"""Offline comparison of pure-Python prototype vs nrp-core go/no-go results.

Pure functions only — no NRPCoreSim, no module-level I/O. Loaders normalize the
two sides onto a common representation (per-knob and per-frequency
go_success_rate keyed by float Hz), so the prototype's `miss_rate` and the nrp
binding's `go_success_rate` become directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

# --- Knob vocabulary ---

# Canonical nrp-side knob labels, in report order.
KNOBS: tuple[str, ...] = ("sampling", "integration", "emission", "commitment")

# Prototype FrequencyConfig field names -> canonical nrp labels.
KNOB_NAME_MAP: dict[str, str] = {
    "input_sampling_hz": "sampling",
    "integration_step_hz": "integration",
    "output_emission_hz": "emission",
    "commitment_update_hz": "commitment",
}


# --- Loaders (normalize both sides to go_success_rate keyed by float Hz) ---


def load_prototype_ablation(path: str | Path) -> dict[str, dict[float, float]]:
    """Prototype ablation: list of {knob_name, freq_hz, miss_rate}. Converts
    miss_rate -> go_success_rate and maps knob names. The 'all'/baseline record
    is not a per-knob sweep series and is excluded."""
    records = json.loads(Path(path).read_text())
    out: dict[str, dict[float, float]] = {knob: {} for knob in KNOBS}
    for r in records:
        name = r["knob_name"]
        if name == "all":
            continue
        knob = KNOB_NAME_MAP[name]
        out[knob][float(r["freq_hz"])] = 1.0 - float(r["miss_rate"])
    return out


def load_nrp_ablation(path: str | Path) -> dict[str, dict[float, float]]:
    """nrp ablation: {knob: {str_hz: go_success_rate}}. Coerce Hz keys to float."""
    raw = json.loads(Path(path).read_text())
    return {
        knob: {float(hz): float(rate) for hz, rate in series.items()}
        for knob, series in raw.items()
    }


def load_prototype_gonogo_sweep(path: str | Path) -> dict[float, float]:
    """Prototype frequency sweep: filter paradigm == 'go_nogo', average
    go_success_rate per frequency over conflict levels and seeds."""
    records = json.loads(Path(path).read_text())
    by_freq: dict[float, list[float]] = {}
    for r in records:
        if r.get("paradigm") != "go_nogo":
            continue
        by_freq.setdefault(float(r["frequency_hz"]), []).append(float(r["go_success_rate"]))
    return {hz: sum(vals) / len(vals) for hz, vals in by_freq.items()}


def load_nrp_gonogo_sweep(path: str | Path) -> dict[float, float]:
    """nrp frequency sweep: {str_hz: go_success_rate}. Coerce Hz keys to float."""
    raw = json.loads(Path(path).read_text())
    return {float(hz): float(rate) for hz, rate in raw.items()}
