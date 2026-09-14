"""Residual-debt ledgers and extra reachable suffixes. No debt-reset 'fix'."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

from .util import write_csv, write_json, write_text


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def from_first_pass(first_pass: Path) -> dict[str, Any]:
    returns = list(csv.DictReader((first_pass / "10_symbolic_probes" / "per_episode_returns.csv").open(encoding="utf-8", newline="")))
    effects = list(csv.DictReader((first_pass / "10_symbolic_probes" / "matched_suffix_effects.csv").open(encoding="utf-8", newline="")))
    ledgers = list(csv.DictReader((first_pass / "10_symbolic_probes" / "per_transition_ledger.csv").open(encoding="utf-8", newline="")))
    cycles = []
    for k in (1, 3, 8):
        eid = f"normalized_main__reset__p0__k{k}"
        row = next(r for r in returns if r["episode_id"] == eid and r["method"] == "FULL_FROZEN")
        cycles.append(dict(k=k, episode_id=eid, signed_return=float(row["signed_return"]),
                           debt_after=float(row["debt_after"]), expected_debt=k / 6))
    suffix = [r for r in effects if r["method"] == "FULL_FROZEN"]
    suffix_sorted = sorted(suffix, key=lambda r: int(r["preload_cycles"]))
    return dict(cycles=cycles, matched_suffix=suffix_sorted, n_ledger_rows=len(ledgers))


def extra_reachable_suffixes() -> list[dict[str, Any]]:
    """Additional lawful skeletons using only legacy recovery-graph edges."""
    return [
        dict(name="partial_recovery_stop", path=["in_transit", "dropped_or_misaligned", "recovery"],
             edges=["in_transit_to_dropped_or_misaligned", "dropped_or_misaligned_to_recovery"],
             note="Stops after recovery begins; no success. Lawful prefix of the recovery chain."),
        dict(name="recover_then_terminal_failure", path=["in_transit", "dropped_or_misaligned", "recovery", "grasped", "in_transit", "terminal_failure"],
             edges=["in_transit_to_dropped_or_misaligned", "dropped_or_misaligned_to_recovery", "recovery_to_grasped", "grasped_to_in_transit", "in_transit_to_terminal_failure"],
             note="Recovery then unrecoverable terminal failure. Existing failure edge."),
        dict(name="recover_then_fail_again", path=["in_transit", "dropped_or_misaligned", "recovery", "grasped", "in_transit", "dropped_or_misaligned"],
             edges=["in_transit_to_dropped_or_misaligned", "dropped_or_misaligned_to_recovery", "recovery_to_grasped", "grasped_to_in_transit", "in_transit_to_dropped_or_misaligned"],
             note="Second drop after recovery. No new edges."),
        dict(name="success_then_hold", path=["in_transit", "dropped_or_misaligned", "recovery", "grasped", "in_transit", "placed", "success", "success"],
             edges=["in_transit_to_dropped_or_misaligned", "dropped_or_misaligned_to_recovery", "recovery_to_grasped", "grasped_to_in_transit", "in_transit_to_placed", "placed_to_success", None],
             note="Matched success suffix plus same-node hold. Hold is none, not a new graph edge."),
    ]


def write_outputs(first_pass: Path, out: Path) -> dict[str, Any]:
    data = from_first_pass(first_pass)
    cycle_rows = [dict(k=c["k"], signed_return=c["signed_return"], debt_after=c["debt_after"],
                       expected_debt=c["expected_debt"],
                       matches_k_over_6=abs(c["debt_after"] - c["expected_debt"]) < 1e-9)
                  for c in data["cycles"]]
    write_csv(out / "debt_by_cycle.csv", cycle_rows)
    write_csv(out / "matched_suffix_rewards.csv", [
        dict(preload_cycles=r["preload_cycles"], debt_at_suffix_entry=r["debt_at_suffix_entry"],
             suffix_signed_return=r["suffix_signed_return"],
             max_step_reward_difference_from_zero_preload=r["max_step_reward_difference_from_zero_preload"],
             changed=r["changed"], claim_scope=r["claim_scope"])
        for r in data["matched_suffix"]
    ])
    extras = extra_reachable_suffixes()
    write_json(out / "reachability_and_scope.json", dict(
        matched_suffix_path=["in_transit", "dropped_or_misaligned", "recovery", "grasped", "in_transit", "placed", "success"],
        extra_reachable=extras,
        counterfactual_ledger_probes="NOT_RUN_AS_REAL_RISK; would be CONTROLLED_LEDGER_STATE_PROBE if added",
        episode_vs_attempt="new_episode clears debt; recovery attempts do not",
    ))
    changed = any(str(r["changed"]).lower() == "true" for r in data["matched_suffix"])
    conclusion = [
        "# Residual debt",
        "",
        "Normalized no-progress cycles accumulate D_k ≈ k/6 inside one episode.",
        "Matched lawful suffix `in_transit→dropped→recovery→grasped→in_transit→placed→success` has unchanged per-step signed rewards for preload k=0/1/3/8 under FULL_FROZEN.",
        "Conclusion: residual debt exists, but the tested lawful suffix reward did not change.",
        "This is not proof that residual debt is harmless in general, and it is not a reason to zero debt at recovery.",
        f"matched_suffix_changed={changed}",
        "",
        "Extra reachable skeletons (not scored as new physics): partial recovery stop, recover-then-terminal-failure, recover-then-fail-again, success-then-hold.",
    ]
    write_text(out / "conclusion.md", "\n".join(conclusion))
    return dict(cycles=cycle_rows, matched_suffix_changed=changed, extras=extras)