"""Phase 11 cerebellar correction experiment (M9 acceptance evidence).

Runs the visuomotor-rotation go/no-go pipeline at five BG frequencies with the
cerebellum off vs on, on the SAME BG decisions. Demonstrates:
  (a) accuracy improves under perturbation (within-trial + across-trial), and
  (b) the BG-frequency onset signature is unchanged by the cerebellum.
"""
from __future__ import annotations

import json

from nrp_bga_sb.cerebellum_sweep import (
    FREQUENCIES_HZ,
    CerebellumSweepResult,
    run_cerebellum_condition,
)

_DEFAULT_SEEDS = [11, 22, 33, 44, 55]


def run_sweep(
    n_trials: int = 30,
    seeds: list[int] | None = None,
    perturbation_deg: float = 30.0,
) -> list[CerebellumSweepResult]:
    """Run every frequency x {cerebellum off, on} x seed condition."""
    seeds = seeds if seeds is not None else _DEFAULT_SEEDS
    results: list[CerebellumSweepResult] = []
    for freq in FREQUENCIES_HZ:
        for enabled in (False, True):
            for seed in seeds:
                results.append(
                    run_cerebellum_condition(
                        freq,
                        n_trials=n_trials,
                        seed=seed,
                        perturbation_deg=perturbation_deg,
                        cerebellum_enabled=enabled,
                    )
                )
    return results


def save_results(results: list[CerebellumSweepResult], path: str) -> None:
    """Write results as a JSON array of model dumps."""
    payload = [r.model_dump() for r in results]
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def format_report(results: list[CerebellumSweepResult]) -> str:
    """Per-frequency on-vs-off comparison table (averaged over seeds)."""
    lines = [
        "Phase 11 — Cerebellar correction (M9)",
        "freq(Hz) | onset off | onset on | dev off | dev on | ang.err on | theta_hat on",
        "-" * 82,
    ]
    for freq in FREQUENCIES_HZ:
        on = [r for r in results if r.frequency_hz == freq and r.cerebellum_enabled]
        off = [r for r in results if r.frequency_hz == freq and not r.cerebellum_enabled]

        def avg(rs: list[CerebellumSweepResult], attr: str) -> float:
            return sum(getattr(r, attr) for r in rs) / len(rs) if rs else 0.0

        lines.append(
            f"{freq:>7.1f} | "
            f"{avg(off, 'movement_onset_rate'):>9.3f} | "
            f"{avg(on, 'movement_onset_rate'):>8.3f} | "
            f"{avg(off, 'mean_endpoint_deviation'):>13.4f} | "
            f"{avg(on, 'mean_endpoint_deviation'):>12.4f} | "
            f"{avg(on, 'mean_angular_error_rad'):>10.4f} | "
            f"{avg(on, 'final_theta_hat'):>12.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    results = run_sweep()
    save_results(results, "results/cerebellum_results.json")
    print(format_report(results))


if __name__ == "__main__":
    main()
