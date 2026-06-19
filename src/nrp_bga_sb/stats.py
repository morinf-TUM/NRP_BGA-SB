"""Statistical reporting for Phase 5 frequency-sweep results (Task 5.2).

Provides bootstrap confidence intervals (percentile method, pure numpy),
frequency-response curve aggregation, a log-frequency OLS slope estimator
(GLM proxy for error probabilities), and a reproducibility checker.
"""

from __future__ import annotations

import numpy as np

from nrp_bga_sb.sweep import SweepConditionResult

# --- Bootstrap CI ---


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`.

    Raises ValueError if values is empty. Uses percentile method (pure numpy).
    rng_seed guarantees deterministic results across calls.
    """
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    # Vectorised bootstrap: shape (n_bootstrap, n_samples) → mean per row
    resampled = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(resampled, 100.0 * alpha / 2.0))
    hi = float(np.percentile(resampled, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


# --- Frequency-curve aggregation ---


def aggregate_by_frequency(
    results: list[SweepConditionResult],
    metric: str,
    paradigm: str | None = None,
    conflict_level: str | None = None,
) -> dict[float, dict]:
    """Group results by frequency_hz and compute mean ± 95% CI for one metric.

    Returns dict mapping frequency_hz → {"mean", "ci_lo", "ci_hi", "n"}.
    Frequencies where all values are None are omitted.
    """
    filtered = results
    if paradigm is not None:
        filtered = [r for r in filtered if r.paradigm == paradigm]
    if conflict_level is not None:
        filtered = [r for r in filtered if r.conflict_level == conflict_level]

    freq_groups: dict[float, list[float]] = {}
    for r in filtered:
        val = getattr(r, metric, None)
        if val is not None:
            freq_groups.setdefault(r.frequency_hz, []).append(float(val))

    curves: dict[float, dict] = {}
    for freq in sorted(freq_groups):
        vals = freq_groups[freq]
        mean = sum(vals) / len(vals)
        lo, hi = bootstrap_ci(vals)
        curves[freq] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}
    return curves


# --- Log-frequency OLS slope ---


def fit_frequency_slope(curves: dict[float, dict]) -> float:
    """OLS slope of metric ~ log(frequency_hz).

    GLM proxy: positive slope → metric rises with frequency; negative → falls.
    Returns 0.0 if fewer than 2 frequency points.
    """
    if len(curves) < 2:
        return 0.0
    freqs_sorted = sorted(curves.keys())
    x = np.log(np.array(freqs_sorted, dtype=float))
    y = np.array([curves[f]["mean"] for f in freqs_sorted], dtype=float)
    # Centre to avoid numerical issues; slope = Cov(x,y) / Var(x)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.dot(x_c, x_c))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_c, y_c) / denom)


# --- Reproducibility check ---


def reproducibility_check(
    results_a: list[SweepConditionResult],
    results_b: list[SweepConditionResult],
    tolerance: float = 1e-9,
) -> bool:
    """Verify two sweep result sets are element-wise identical within tolerance.

    Matches by (frequency_hz, conflict_level, paradigm, seed) key. Returns True
    only if all matched pairs agree on the five key rate metrics.
    """
    def _key(r: SweepConditionResult) -> tuple:
        return (r.frequency_hz, r.conflict_level, r.paradigm, r.seed)

    map_a = {_key(r): r for r in results_a}
    map_b = {_key(r): r for r in results_b}

    if set(map_a.keys()) != set(map_b.keys()):
        return False

    checked = ["miss_rate", "go_success_rate", "wrong_target_rate", "timeout_rate", "false_alarm_rate"]  # noqa: E501
    for k in map_a:
        ra, rb = map_a[k], map_b[k]
        for m in checked:
            va = getattr(ra, m, None)
            vb = getattr(rb, m, None)
            if va is None and vb is None:
                continue
            if va is None or vb is None:
                return False
            if abs(va - vb) > tolerance:
                return False
    return True


# --- Text report ---


def format_sweep_report(
    results: list[SweepConditionResult],
    frequencies: list[float],
    conflict_levels: list[str],
) -> str:
    """Format a human-readable Phase 5 frequency-sweep summary report.

    Includes frequency-response curves with 95% CIs and log-frequency slope
    for each (paradigm, conflict_level) combination.
    """
    lines = ["Phase 5 Frequency-Sweep Report", "=" * 60, ""]

    paradigm_metrics = {
        "go_nogo": ("miss_rate", "Miss rate (go trials)"),
        "two_choice": ("timeout_rate", "Timeout rate (no selection)"),
    }

    for paradigm, (metric_key, metric_label) in paradigm_metrics.items():
        lines.append(f"Paradigm: {paradigm}  |  Primary metric: {metric_label}")
        lines.append("-" * 60)
        header = (
            f"  {'Conflict':<10} {'Freq (Hz)':<12} {'Mean':>8} {'CI lo':>8} {'CI hi':>8} {'N':>5}"
        )
        lines.append(header)

        for conflict in conflict_levels:
            curves = aggregate_by_frequency(
                results, metric_key, paradigm=paradigm, conflict_level=conflict
            )
            if not curves:
                continue
            slope = fit_frequency_slope(curves)
            for freq in sorted(frequencies):
                if freq not in curves:
                    continue
                d = curves[freq]
                lines.append(
                    f"  {conflict:<10} {freq:<12.0f} "
                    f"{d['mean']:>8.3f} {d['ci_lo']:>8.3f} {d['ci_hi']:>8.3f} {d['n']:>5}"
                )
            lines.append(f"  {conflict:<10} {'slope(log-f)':>12} {slope:>+8.4f}")
            lines.append("")

        # Secondary: wrong_target_rate for two_choice
        if paradigm == "two_choice":
            lines.append("  Secondary metric: Wrong-target rate")
            for conflict in conflict_levels:
                curves = aggregate_by_frequency(
                    results, "wrong_target_rate", paradigm=paradigm, conflict_level=conflict
                )
                if not curves:
                    continue
                slope = fit_frequency_slope(curves)
                for freq in sorted(frequencies):
                    if freq not in curves:
                        continue
                    d = curves[freq]
                    lines.append(
                        f"  {conflict:<10} {freq:<12.0f} "
                        f"{d['mean']:>8.3f} {d['ci_lo']:>8.3f} {d['ci_hi']:>8.3f} {d['n']:>5}"
                    )
                lines.append(f"  {conflict:<10} {'slope(log-f)':>12} {slope:>+8.4f}")
                lines.append("")

        lines.append("")

    return "\n".join(lines)
