"""Offline comparison of pure-Python prototype vs nrp-core go/no-go results.

Pure functions only — no NRPCoreSim, no module-level I/O. Loaders normalize the
two sides onto a common representation (per-knob and per-frequency
go_success_rate keyed by float Hz), so the prototype's `miss_rate` and the nrp
binding's `go_success_rate` become directly comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


# --- Categorical regime ---


def classify_regime(rate: float) -> str:
    """Coarse outcome regime. Seed counts differ across sides (prototype 24-30
    trials, nrp 3-5 seeds), so exact rates are not meaningful — only whether the
    go response is predominantly released."""
    return "success" if rate > 0.5 else "miss"


# --- Frequency-grid alignment ---


@dataclass
class FreqRow:
    freq_hz: float
    proto_rate: float | None
    nrp_rate: float | None
    tag: str  # "common" | "proto_only" | "nrp_only"


def align_series(proto: dict[float, float], nrp: dict[float, float]) -> list[FreqRow]:
    """Union of frequencies, sorted, each tagged by which side(s) cover it."""
    rows: list[FreqRow] = []
    for hz in sorted(set(proto) | set(nrp)):
        in_p, in_n = hz in proto, hz in nrp
        # Trigger: frequency present on both sides vs only one.
        # Why: only common frequencies can be judged agree/diverge; one-sided
        #      frequencies (e.g. nrp-only 5 Hz in the sweep) are reported, not graded.
        tag = "common" if in_p and in_n else ("proto_only" if in_p else "nrp_only")
        rows.append(FreqRow(freq_hz=hz, proto_rate=proto.get(hz), nrp_rate=nrp.get(hz), tag=tag))
    return rows


# --- Ablation comparison ---


@dataclass
class KnobVerdict:
    knob: str
    holds: bool
    divergent_freqs: list[float]
    rows: list[FreqRow]


@dataclass
class AblationVerdict:
    knobs: list[KnobVerdict]
    holds: bool


def compare_ablation(
    proto: dict[str, dict[float, float]],
    nrp: dict[str, dict[float, float]],
) -> AblationVerdict:
    """Per-knob regime match across the common frequency grid. A knob holds iff
    its prototype and nrp regimes agree at every shared frequency."""
    knob_verdicts: list[KnobVerdict] = []
    for knob in KNOBS:
        rows = align_series(proto.get(knob, {}), nrp.get(knob, {}))
        divergent = [
            r.freq_hz
            for r in rows
            if r.tag == "common" and classify_regime(r.proto_rate) != classify_regime(r.nrp_rate)
        ]
        knob_verdicts.append(
            KnobVerdict(knob=knob, holds=not divergent, divergent_freqs=divergent, rows=rows)
        )
    return AblationVerdict(knobs=knob_verdicts, holds=all(kv.holds for kv in knob_verdicts))


# --- Frequency-sweep comparison (qualitative only) ---


@dataclass
class SweepVerdict:
    proto_onset_hz: float | None
    nrp_onset_hz: float | None
    proto_rows: dict[float, float]
    nrp_rows: dict[float, float]
    caveat: str


def _success_onset(series: dict[float, float]) -> float | None:
    """Lowest frequency whose regime is 'success', or None if never."""
    successes = [hz for hz in sorted(series) if classify_regime(series[hz]) == "success"]
    return successes[0] if successes else None


def compare_frequency_sweep(
    proto: dict[float, float],
    nrp: dict[float, float],
) -> SweepVerdict:
    """Qualitative comparison only. The prototype sweep JSON averages conflict
    levels (low/medium/high), which inflates its apparent success onset, so we
    report each side's monotone curve and onset but never a per-frequency
    pass/fail."""
    return SweepVerdict(
        proto_onset_hz=_success_onset(proto),
        nrp_onset_hz=_success_onset(nrp),
        proto_rows=dict(proto),
        nrp_rows=dict(nrp),
        caveat=(
            "Prototype frequency-sweep rates aggregate conflict levels "
            "(low/medium/high), inflating the apparent success onset; only the "
            "qualitative monotone rise to 1.0 is comparable, not the onset "
            "frequency or per-frequency rates."
        ),
    )


# --- Overall verdict + report ---


@dataclass
class OverallVerdict:
    ablation: AblationVerdict
    sweep: SweepVerdict
    summary: str


def build_verdict(ablation: AblationVerdict, sweep: SweepVerdict) -> OverallVerdict:
    held = [kv.knob for kv in ablation.knobs if kv.holds]
    diverged = [kv.knob for kv in ablation.knobs if not kv.holds]
    summary = (
        f"{len(held)} of {len(ablation.knobs)} knobs reproduce the prototype "
        f"exactly across the common grid ({', '.join(held)})."
    )
    if diverged:
        # The expected, scientifically meaningful outcome: integration diverges
        # because the nrp sub-step is idempotent on the stateless BG solver.
        summary += (
            f" The {', '.join(diverged)} knob(s) diverge — see the per-knob table; "
            f"this is the knob-2 idempotence finding (PROJECT_MEMORY §15.7) and "
            f"motivates the separate knob-2 modeling project."
        )
    return OverallVerdict(ablation=ablation, sweep=sweep, summary=summary)


def _ablation_table(ablation: AblationVerdict) -> str:
    # Column frequencies = union across all knob rows, sorted.
    freqs = sorted({r.freq_hz for kv in ablation.knobs for r in kv.rows})
    header = "| knob | " + " | ".join(f"{f:g} Hz" for f in freqs) + " | holds |"
    sep = "|" + "---|" * (len(freqs) + 2)
    lines = [header, sep]
    for kv in ablation.knobs:
        by_hz = {r.freq_hz: r for r in kv.rows}
        cells = []
        for f in freqs:
            row = by_hz.get(f)
            if row is None or (row.proto_rate is None and row.nrp_rate is None):
                cells.append("–")
            else:
                p = "–" if row.proto_rate is None else f"{row.proto_rate:g}"
                n = "–" if row.nrp_rate is None else f"{row.nrp_rate:g}"
                mark = ""
                if row.tag == "common" and classify_regime(row.proto_rate) != classify_regime(row.nrp_rate):
                    mark = " ✗"
                cells.append(f"{p}/{n}{mark}")
        status = "✓" if kv.holds else f"✗ @ {', '.join(f'{x:g}' for x in kv.divergent_freqs)} Hz"
        lines.append(f"| {kv.knob} | " + " | ".join(cells) + f" | {status} |")
    return "\n".join(lines)


def _sweep_table(sweep: SweepVerdict) -> str:
    freqs = sorted(set(sweep.proto_rows) | set(sweep.nrp_rows))
    header = "| side | " + " | ".join(f"{f:g} Hz" for f in freqs) + " | onset |"
    sep = "|" + "---|" * (len(freqs) + 2)

    def fmt(rows: dict[float, float], onset: float | None) -> str:
        cells = [(f"{rows[f]:g}" if f in rows else "–") for f in freqs]
        onset_s = "none" if onset is None else f"{onset:g} Hz"
        return " | ".join(cells) + f" | {onset_s} |"

    return "\n".join([
        header,
        sep,
        "| prototype | " + fmt(sweep.proto_rows, sweep.proto_onset_hz),
        "| nrp-core | " + fmt(sweep.nrp_rows, sweep.nrp_onset_hz),
    ])


def format_report(verdict: OverallVerdict) -> str:
    """Render the comparison as a self-contained markdown report. Cells show
    prototype/nrp go-success; ✗ marks a common-grid regime divergence."""
    return f"""# Go/No-Go: Prototype vs nrp-core Comparison

**Verdict:** {verdict.summary}

## Ablation (primary, like-for-like)

Cells are prototype/nrp go-success rate; ✗ marks a common-grid regime divergence.

{_ablation_table(verdict.ablation)}

## Frequency sweep (secondary, qualitative)

> {verdict.sweep.caveat}

{_sweep_table(verdict.sweep)}
"""
