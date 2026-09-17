"""P2C-RM command line."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import METHOD_V2, SMOKE_METHODS
from .io_utils import dump_replace, load, sha256_file
from .reward_audit import (
    V1_METHOD,
    audit_export_mechanism,
    audit_export_signal,
    combine_signal,
    confirmation_signal_passed,
    contract_from_mdp,
    development_signal_passed,
    load_json,
    mechanism_passed,
    oracle_alignment,
    planner_import_audit,
    score_one_step,
)
from .statistics import mean

DEFAULTS = {
    "wt": "/home/__compress_data/xushijie/graph_pathgraph_p2c_rm_v1_worktree",
    "data": "/home/__compress_data/xushijie/graph_pathgraph_p2c_rm_data/run_v1",
    "p2cq_data": "/home/__compress_data/xushijie/graph_pathgraph_p2c_q_data/run_v1",
    "p2cq_pkg": "/home/__compress_data/xushijie/PathGraph_P2C_Q_Agent_Package_V1.0",
    "pkg": "/home/__compress_data/xushijie/PathGraph_P2C_RM_RewardV2_MaskablePPO_Agent_Package_V1.0/PathGraph_P2C_RM_RewardV2_MaskablePPO_Agent_Package_V1.0",
    "art": "artifacts/pathgraph_sarm/upgrade_v2/p2c_rm_reward_v2_maskable_v1",
}


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _add_p2cq(pkg):
    if pkg and pkg not in sys.path:
        sys.path.insert(0, pkg)


def cmd_import_audit(args):
    wt = Path(args.repo)
    path = wt / "experiments/pathgraph_p2c_rm_v1/p2crm/remaining_work_model.py"
    rec = planner_import_audit(path)
    dump_replace(Path(args.out), rec)
    print(json.dumps(rec, indent=2))
    if not rec["passed"]:
        raise SystemExit(1)


def cmd_audit_reward_development(args):
    data = Path(args.out)
    data.mkdir(parents=True, exist_ok=True)
    p2cq = Path(args.p2cq_data)
    exports = [p2cq / "development" / "export", p2cq / "confirmation" / "export"]
    mechs = []
    sigs = []
    for exp in exports:
        print("MECH", exp, flush=True)
        mech = audit_export_mechanism(exp)
        mechs.append(mech)
        dump_replace(data / (exp.parent.name + "_mechanism.json"), mech)
        print("SIGNAL", exp, flush=True)
        sig, rows = audit_export_signal(exp, args.p2cq_pkg)
        sigs.append((sig, rows))
        dump_replace(data / (exp.parent.name + "_signal.json"), sig)
    mech = {
        "states_checked": sum(m["states_checked"] for m in mechs),
        "cost_parity_mismatches": sum(m["cost_parity_mismatches"] for m in mechs),
        "bellman_mismatches": sum(m["bellman_mismatches"] for m in mechs),
        "unreachable_mismatches": sum(m["unreachable_mismatches"] for m in mechs),
        "search_truncations": sum(m["search_truncations"] for m in mechs),
        "cache_equivalence_mismatches": sum(m["cache_equivalence_mismatches"] for m in mechs),
        "stage_handoff_regressions": sum(m["stage_handoff_regressions"] for m in mechs),
        "pbrs_max_abs_error": max(m["pbrs_max_abs_error"] for m in mechs),
        "mdps": sum(m["mdps"] for m in mechs),
    }
    sig = combine_signal(sigs)
    dump_replace(data / "mechanism_combined.json", mech)
    dump_replace(data / "signal_combined.json", sig)
    c = sig["critical"]
    audit = {
        "schema": "P2C_RM_REWARD_V2_AUDIT_V1",
        "phase": "development",
        "candidate_commit": None,
        "states_checked": mech["states_checked"],
        "cost_parity_mismatches": mech["cost_parity_mismatches"],
        "bellman_mismatches": mech["bellman_mismatches"],
        "unreachable_mismatches": mech["unreachable_mismatches"],
        "search_truncations": mech["search_truncations"],
        "pbrs_max_abs_error": mech["pbrs_max_abs_error"],
        "cache_equivalence_mismatches": mech["cache_equivalence_mismatches"],
        "stage_handoff_regressions": mech["stage_handoff_regressions"],
        "v1_critical_conservative_hit": c["v1_hit"],
        "v2_critical_conservative_hit": c["v2_hit"],
        "v1_mean_regret": c["v1_regret"],
        "v2_mean_regret": c["v2_regret"],
        "per_motif": {
            m: {"v1_hit": rec["v1_hit"], "v2_hit": rec["v2_hit"], "v1_regret": rec["v1_regret"], "v2_regret": rec["v2_regret"], "n": rec["n"]}
            for m, rec in sig["per_motif"].items()
        },
        "passed": mechanism_passed(mech) and development_signal_passed(sig),
        "mechanism_passed": mechanism_passed(mech),
        "signal_passed": development_signal_passed(sig),
    }
    dump_replace(data / "reward_v2_audit.json", audit)
    print(json.dumps({k: audit[k] for k in audit if k != "per_motif"}, indent=2, default=str))
    return audit


def _table_mdp(p2cq_pkg, data):
    _add_p2cq(p2cq_pkg)
    from p2cq.mdp import TableMDP
    return TableMDP(data)


def _pair_informative(l, r, limits):
    from p2cq.metrics import optimal_set
    lq, rq = l["q"], r["q"]
    lopt, ropt = optimal_set(lq), optimal_set(rq)
    same = l["nongraph"] == r["nongraph"]
    public_distinct = l["obs"] != r["obs"]
    valid_same = l["valid"] == r["valid"]
    nontrivial = sum(l["valid"]) >= 2 and sum(r["valid"]) >= 2
    solvable = min(max(lq), max(rq)) > 0
    gaps = [l["gap"], r["gap"]]
    sep = not bool(lopt & ropt)
    margin = all(g is not None and g >= limits["min_oracle_gap"] for g in gaps)
    informative = same and public_distinct and solvable and sep and margin and nontrivial
    return informative, valid_same


def cmd_generate_confirmation(args):
    from .export_v2 import collision_scan, export_split
    from .generator_v2 import family_id, iter_split
    from p2cq_research.generator import MOTIFS, ROOTS_PER_MOTIF

    out = Path(args.out)
    new_ids = [family_id("confirmation", m, r) for m in MOTIFS for r in range(ROOTS_PER_MOTIF)]
    scan = collision_scan([
        str(Path(args.p2cq_data) / "development" / "export"),
        str(Path(args.p2cq_data) / "confirmation" / "export"),
    ], new_ids)
    dump_replace(out / "collision_scan.json", scan)
    if not scan["passed"]:
        raise SystemExit("family id collision")
    rec = export_split("confirmation", out, write_mdp=True)
    print(json.dumps({k: rec[k] for k in rec if k not in ("mdps", "registry")}, indent=2))
    return rec


def cmd_audit_reward_confirmation(args):
    data = Path(args.out)
    data.mkdir(parents=True, exist_ok=True)
    exp = Path(args.data)
    mech = audit_export_mechanism(exp)
    sig, rows = audit_export_signal(exp, args.p2cq_pkg)
    c = sig["critical"]
    audit = {
        "schema": "P2C_RM_REWARD_V2_AUDIT_V1",
        "phase": "confirmation",
        "candidate_commit": args.candidate_commit,
        "states_checked": mech["states_checked"],
        "cost_parity_mismatches": mech["cost_parity_mismatches"],
        "bellman_mismatches": mech["bellman_mismatches"],
        "unreachable_mismatches": mech["unreachable_mismatches"],
        "search_truncations": mech["search_truncations"],
        "pbrs_max_abs_error": mech["pbrs_max_abs_error"],
        "cache_equivalence_mismatches": mech["cache_equivalence_mismatches"],
        "stage_handoff_regressions": mech["stage_handoff_regressions"],
        "v1_critical_conservative_hit": c["v1_hit"],
        "v2_critical_conservative_hit": c["v2_hit"],
        "v1_mean_regret": c["v1_regret"],
        "v2_mean_regret": c["v2_regret"],
        "per_motif": {
            m: {"v1_hit": rec["v1_hit"], "v2_hit": rec["v2_hit"], "n": rec["n"]}
            for m, rec in sig["per_motif"].items()
        },
        "passed": mechanism_passed(mech) and confirmation_signal_passed(sig),
        "mechanism_passed": mechanism_passed(mech),
        "signal_passed": confirmation_signal_passed(sig),
        "environment_summary": sig.get("environment_summary"),
    }
    dump_replace(data / "reward_v2_audit.json", audit)
    dump_replace(data / "mechanism.json", mech)
    dump_replace(data / "signal.json", sig)
    print(json.dumps({k: audit[k] for k in audit if k not in ("per_motif", "environment_summary")}, indent=2, default=str))
    return audit


def cmd_audit_masks(args):
    from .mask_audit import audit_export_masks
    tot = audit_export_masks(args.data, args.p2cq_pkg, args.out)
    print(json.dumps({k: tot[k] for k in tot if k not in ("dummy",)}, indent=2, default=str))
    if not tot["passed"]:
        raise SystemExit(1)
    return tot


def cmd_smoke_all(args):
    from .masked_smoke import run_all
    rec = run_all(args.out)
    print(json.dumps({"status": rec["status"], "parity": rec["initial_policy_parity"], "n": len(rec["jobs"])}, indent=2))
    if rec["status"] != "SMOKE_PASS_NOT_RESEARCH_RESULT":
        raise SystemExit(1)
    return rec


def cmd_freeze_candidate(args):
    rec = {
        "schema": "P2C_RM_CANDIDATE_FREEZE_V1",
        "candidate_commit": args.candidate_commit,
        "frozen_at_utc": _utc(),
        "note": "No further candidate or threshold edits after confirmation starts.",
    }
    dump_replace(Path(args.out), rec)
    print(json.dumps(rec, indent=2))


def build():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--repo", default=DEFAULTS["wt"])
        sp.add_argument("--p2cq-data", default=DEFAULTS["p2cq_data"])
        sp.add_argument("--p2cq-pkg", default=DEFAULTS["p2cq_pkg"])
        return sp

    s = add_common(sub.add_parser("import-audit"))
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_import_audit)

    s = add_common(sub.add_parser("audit-reward-development"))
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_audit_reward_development)

    s = add_common(sub.add_parser("generate-confirmation"))
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_generate_confirmation)

    s = add_common(sub.add_parser("audit-reward-confirmation"))
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--candidate-commit", default=None)
    s.set_defaults(func=cmd_audit_reward_confirmation)

    s = add_common(sub.add_parser("audit-masks"))
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_audit_masks)

    s = add_common(sub.add_parser("smoke-all"))
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_smoke_all)

    s = add_common(sub.add_parser("freeze-candidate"))
    s.add_argument("--out", required=True)
    s.add_argument("--candidate-commit", required=True)
    s.set_defaults(func=cmd_freeze_candidate)
    return p


def main(argv=None):
    args = build().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()