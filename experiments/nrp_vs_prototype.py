"""Offline go/no-go comparison: pure-Python prototype vs nrp-core binding.

Loads committed result snapshots from both sides (no NRPCoreSim needed), builds
the verdict, writes the markdown report, and prints a one-line summary.

Regenerate the nrp snapshots (live, env-gated, slow) before running this if the
binding changed:
    source $HOME/.local/nrp/bin/.nrp_env
    python experiments/nrp_gonogo_sweep.py   # -> nrp/results/gonogo_sweep.json
    python experiments/nrp_ablation.py       # -> nrp/results/ablation.json
"""

from __future__ import annotations

from pathlib import Path

from nrp.compare import (
    build_verdict,
    compare_ablation,
    compare_frequency_sweep,
    format_report,
    load_nrp_ablation,
    load_nrp_gonogo_sweep,
    load_prototype_ablation,
    load_prototype_gonogo_sweep,
)

REPO = Path(__file__).resolve().parents[1]
PROTO = REPO / "deprecated_toy_prototype_results"
RESULTS = REPO / "nrp" / "results"
REPORT = REPO / "docs" / "nrp_vs_prototype_comparison.md"


def main() -> None:
    ablation = compare_ablation(
        load_prototype_ablation(PROTO / "ablation_frequency_v2.json"),
        load_nrp_ablation(RESULTS / "ablation.json"),
    )
    sweep = compare_frequency_sweep(
        load_prototype_gonogo_sweep(PROTO / "frequency_sweep_results.json"),
        load_nrp_gonogo_sweep(RESULTS / "gonogo_sweep.json"),
    )
    verdict = build_verdict(ablation, sweep)
    REPORT.write_text(format_report(verdict))
    print(verdict.summary)
    print(f"saved -> {REPORT}")


if __name__ == "__main__":
    main()
