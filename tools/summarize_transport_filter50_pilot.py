#!/usr/bin/env python3
"""Summarize paired rollout outcomes for the Transport 50% filtering pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("baseline", "cupid50", "quality50", "random50")
CANDIDATES = ("cupid50", "quality50")


def exact_sign_pvalue(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, value) for value in range(0, min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def paired_stats(
    candidate: pd.Series,
    reference: pd.Series,
    rng: np.random.Generator,
) -> dict:
    difference = candidate.astype(int).to_numpy() - reference.astype(int).to_numpy()
    wins = int(np.sum(difference == 1))
    losses = int(np.sum(difference == -1))
    ties = int(np.sum(difference == 0))
    draws = rng.integers(0, len(difference), size=(10000, len(difference)))
    bootstrap_means = difference[draws].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "paired_bootstrap_95_interval": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "exact_mcnemar_two_sided_p": exact_sign_pvalue(wins, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    outcomes = {}
    summaries = {}
    for arm in ARMS:
        arm_root = args.manifest_root / arm / "rollout100"
        manifest_path = arm_root / "episode_manifest.csv"
        summary_path = arm_root / "rollout_summary.json"
        if not manifest_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing audited output for {arm}")
        frame = pd.read_csv(manifest_path)
        if len(frame) != 100 or frame["seed"].nunique() != 100:
            raise ValueError(f"invalid rollout manifest for {arm}")
        outcomes[arm] = frame.set_index("seed")["success"].astype(bool).sort_index()
        summaries[arm] = json.loads(summary_path.read_text(encoding="utf-8"))

    reference_index = outcomes["baseline"].index
    if any(not series.index.equals(reference_index) for series in outcomes.values()):
        raise ValueError("arms were not evaluated on identical seed sets")

    rng = np.random.default_rng(20260725)
    success_rates = {
        arm: float(series.mean())
        for arm, series in outcomes.items()
    }
    comparisons = {}
    decisions = {}
    for candidate in CANDIDATES:
        versus_baseline = paired_stats(outcomes[candidate], outcomes["baseline"], rng)
        versus_random = paired_stats(outcomes[candidate], outcomes["random50"], rng)
        comparisons[f"{candidate}_vs_baseline"] = versus_baseline
        comparisons[f"{candidate}_vs_random50"] = versus_random
        qualifies = (
            versus_baseline["mean_difference"] >= 0.05
            and versus_random["mean_difference"] >= 0.03
            and versus_baseline["wins"] > versus_baseline["losses"]
            and versus_random["wins"] > versus_random["losses"]
        )
        decisions[candidate] = {
            "advance": qualifies,
            "requirements": {
                "delta_vs_baseline_at_least": 0.05,
                "delta_vs_random50_at_least": 0.03,
                "paired_wins_exceed_losses": True,
            },
        }

    advancing = [arm for arm in CANDIDATES if decisions[arm]["advance"]]
    decision = (
        "ADVANCE_TO_MULTI_SEED_CONFIRMATION"
        if advancing
        else "STOP_PILOT_NO_PRACTICAL_GAIN"
    )
    result = {
        "experiment_id": "transport_filter50_pilot_20260725",
        "evaluation_seed_start": int(reference_index.min()),
        "evaluation_seed_end": int(reference_index.max()),
        "evaluation_count": len(reference_index),
        "success_rates": success_rates,
        "paired_comparisons": comparisons,
        "candidate_decisions": decisions,
        "advancing_candidates": advancing,
        "decision": decision,
        "interpretation": (
            "Pilot advancement is a practical-effect gate, not strong statistical "
            "confirmation. Multi-training-seed confirmation remains required."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )

    lines = [
        "# CUPID Transport-MH 50% filtering pilot result",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        "- Verification Status: ANALYZED",
        "- Version Label: transport_filter50_pilot_v1",
        "",
        "## Success rates",
        "",
        "| Arm | Success rate | Successes / 100 |",
        "|---|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | `{success_rates[arm]:.4f}` | "
            f"{int(round(success_rates[arm] * 100))} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Comparison | Difference | Win/Tie/Loss | Paired bootstrap 95% interval | Exact McNemar p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in comparisons.items():
        interval = item["paired_bootstrap_95_interval"]
        lines.append(
            f"| {name} | `{item['mean_difference']:+.4f}` | "
            f"{item['wins']}/{item['ties']}/{item['losses']} | "
            f"`[{interval[0]:+.4f}, {interval[1]:+.4f}]` | "
            f"`{item['exact_mcnemar_two_sided_p']:.6g}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{decision}`",
            "",
            f"Advancing candidates: {', '.join(advancing) if advancing else 'none'}.",
            "",
            "This 100-rollout pilot uses a pre-frozen practical-effect gate. "
            "It does not replace multi-training-seed confirmation.",
            "",
        ]
    )
    (args.output_dir / "pilot_report.md").write_text(
        "\n".join(lines),
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(decision)


if __name__ == "__main__":
    main()
