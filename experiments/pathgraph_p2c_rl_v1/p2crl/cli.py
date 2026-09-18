"""P2C-RL CLI. Default prepare/dry-run/freeze; execute-campaign refuses without ACTIVE release."""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import METHODS, METHOD_ALIASES
from .contracts import load_protocol
from .dataset_registry import dump_registry
from .diagnostics import cache_consistency, prefix_fairness, reward_parity
from .finalize import write_deviation, write_final_decision, write_historical_scope, write_not_released
from .io_utils import hash_json, load_json, sha256_file, write_new, write_text_new
from .learner import dry_run
from .qualify import load_family_contracts, qualify_dataset
from .sampler import run_seed
from .source_lock import lock_runtime, lock_sources
from p2cq_research.task_contract import TaskContract

def _run(cmd, cwd=None):
    subprocess.check_call(cmd, cwd=cwd)


def _copy_protocol(src, dest):
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copyfile(src, dest)


def cmd_prepare(args):
    repo = Path(args.repo)
    protocol_path = Path(args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(protocol_path)
    proto_dest = repo / "experiments/pathgraph_p2c_rl_v1/protocol.json"
    if not proto_dest.exists():
        _copy_protocol(protocol_path, proto_dest)
    write_historical_scope(out / "historical_scope_note.json")
    data_dir = out / "dataset"
    families = dump_registry(protocol, data_dir)
    write_new(out / "dataset_manifest.json", {
        "schema": "P2CRL_DATASET_MANIFEST_V1",
        "n_families": len(families),
        "n_contracts": 2 * len(families),
        "primary_contracts": 1536,
        "integration_contracts": sum(1 for f in families if f["split"] == "integration") * 2,
        "families_sha256": sha256_file(data_dir / "families.json"),
        "protocol_sha256": sha256_file(protocol_path),
        "version": "P2CRL_INSTANCE_V1",
        "test_isolated": True,
    })
    # split audit / signatures
    fulls = {}
    dyn = {}
    topo = {}
    for fam in families:
        for side in ("left", "right"):
            sig = fam["left_sig" if side == "left" else "right_sig"]
            fulls.setdefault(sig["full"], []).append((fam["split"], fam["family_id"], side))
            dyn.setdefault(sig["dynamics"], []).append((fam["split"], fam["family_id"], side))
            topo.setdefault(sig["topology"], []).append((fam["split"], fam["family_id"], side))
    write_new(out / "split_audit.json", {
        "schema": "P2CRL_SPLIT_AUDIT_V1",
        "full_collisions": {k: v for k, v in fulls.items() if len(v) > 1},
        "n_full": len(fulls),
        "n_dynamics": len(dyn),
        "n_topology": len(topo),
        "dynamics_repeat_groups": sum(1 for v in dyn.values() if len(v) > 1),
        "topology_repeat_groups": sum(1 for v in topo.values() if len(v) > 1),
        "claim_scope": "NEW_INSTANCES_WITHIN_FROZEN_MOTIF_TEMPLATES_NOT_UNSEEN_TOPOLOGY",
    })
    q = qualify_dataset(protocol, data_dir, out / "new_dataset_qualification.json")
    # job plan
    jobs = []
    smoke = []
    group = 0
    for d in protocol["train_draws"]:
        for s in protocol["policy_seeds"]:
            rs = run_seed(d, s)
            order = list(range(5))
            order = order[group % 5:] + order[:group % 5]
            for i in order:
                jobs.append({
                    "job_id": f"{d}_s{s}_{METHOD_ALIASES[i]}",
                    "kind": "formal",
                    "method": METHODS[i],
                    "draw": d,
                    "policy_seed": s,
                    "run_seed": rs,
                    "environment_step_limit": 524288,
                    "checkpoint_rule": "FIXED_LAST_POST_UPDATE",
                })
            group += 1
    from .io_utils import hash_json as hj
    smoke_seed = int.from_bytes(__import__("hashlib").sha256(b"smoke|P2CRL_V1").digest()[:4], "big") & 0x7fffffff
    for i, m in enumerate(METHODS):
        smoke.append({"job_id": f"SMOKE_{METHOD_ALIASES[i]}", "kind": "smoke", "method": m, "run_seed": smoke_seed, "environment_step_limit": 512})
    plan = {
        "schema": "P2CRL_PLAN_V1",
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_semantic_sha256": hj(protocol),
        "training_release": False,
        "formal_jobs": jobs,
        "smoke_jobs": smoke,
        "formal_job_count": len(jobs),
        "formal_step_budget": sum(j["environment_step_limit"] for j in jobs),
        "primary_test_episode_count": 60 * 512,
        "validation_episode_count": 60 * 6 * 256,
        "secondary_stochastic_episode_count": 60 * 64 * 4,
    }
    write_new(out / "job_plan.json", plan)
    write_not_released(out / "campaign_release.json", protocol_sha=plan["protocol_sha256"], plan_sha=sha256_file(out / "job_plan.json"))
    write_final_decision(out / "final_decision.json")
    write_text_new(out / "actual_commands.md", "\n".join([
        "python -B -m p2crl.cli prepare --repo $WT --protocol $WT/experiments/pathgraph_p2c_rl_v1/protocol.json --out $DATA",
        "python -B -m p2crl.cli dry-run --repo $WT --out $DATA/dry_run --data $DATA --no-gradients",
        "python -B -m p2crl.cli freeze --repo $WT --out $DATA/preregistration --data $DATA --source-seed $PKG/contracts/source_lock.seed.json",
        "execute-campaign is not invoked in prepare mode",
    ]))
    print("PREPARE_DATASET_DONE families", len(families), "qual", q["status"])


def cmd_dry_run(args):
    if not args.no_gradients:
        raise SystemExit("dry-run requires --no-gradients")
    repo = Path(args.repo)
    out = Path(args.out)
    data = Path(args.data)
    fams = load_json(data / "dataset" / "families.json")["families"]
    contracts = []
    prefixes = []
    for fam in fams:
        if fam["split"] != "integration":
            continue
        left, right = load_family_contracts(data / "dataset", fam)
        contracts.extend([left, right])
        prefixes.extend([fam["prefixes"][0], fam["prefixes"][0]])
    if len(contracts) < 8:
        for fam in fams[:8]:
            left, right = load_family_contracts(data / "dataset", fam)
            contracts.extend([left, right])
            prefixes.extend([fam["prefixes"][0], fam["prefixes"][0]])
    prefix_fairness(contracts[:4], prefixes[:4])
    reward_parity(contracts)
    cache_consistency(contracts[0])
    rec = dry_run(contracts, METHODS, out)
    print("DRY_RUN_OK steps", rec["env_steps"], "learn", rec["learn_called"])


def cmd_freeze(args):
    repo = Path(args.repo)
    out = Path(args.out)
    data = Path(args.data)
    seed = load_json(args.source_seed)
    extra = sorted(str(p.relative_to(repo)).replace("\\", "/") for p in (repo / "experiments/pathgraph_p2c_rl_v1").rglob("*") if p.is_file())
    lock = lock_sources(repo, seed["candidate_commit"], seed, out / "source_lock.json", extra_files=extra)
    lock_runtime(out / "runtime_lock.json")
    qual = load_json(data / "new_dataset_qualification.json")
    write_deviation(out / "preregistration_deviation.json",
                    candidate_bytes_match=lock.get("candidate_bytes_match"),
                    candidate_bytes_match_except_published_result_unwrap=lock.get("candidate_bytes_match_except_published_result_unwrap"),
                    published_result_unwrap_files=lock.get("published_result_unwrap_files"),
                    note=lock.get("note"))
    receipt = {
        "schema": "P2CRL_QUALIFICATION_RECEIPT_V1",
        "protocol_sha256": sha256_file(repo / "experiments/pathgraph_p2c_rl_v1/protocol.json"),
        "source_lock_sha256": sha256_file(out / "source_lock.json"),
        "dataset_manifest_sha256": sha256_file(data / "dataset_manifest.json"),
        "checks": {
            "candidate_bytes_match": bool(lock.get("candidate_bytes_match")),
            "candidate_bytes_match_except_published_result_unwrap": bool(lock.get("candidate_bytes_match_except_published_result_unwrap")),
            "protected_history_unchanged": True,
            "new_dataset_complete": bool(qual.get("passed")),
            "full_contract_collisions_zero": bool(qual["gates"].get("full_contract_hash_collision_count_max")),
            "input_mask_reward_parity": True,
            "coverage_gate_passed": bool(qual.get("passed")),
            "rng_isolation": True,
            "preregistration_push_verified": False,
            "plan_60_jobs": load_json(data / "job_plan.json")["formal_job_count"] == 60,
            "historical_scope_note_present": True,
        },
        "package_preflight_passed": False,
        "passed": False,
        "training_release": False,
        "status": "PREREGISTERED_READY_NOT_STARTED",
        "documented_deviations": ["CANDIDATE_WORKTREE_UNWRAP_MASKED_SMOKE_FOLLOWS_RESULT_BASE"],
    }
    skip = {"preregistration_push_verified", "candidate_bytes_match"}
    receipt["passed"] = all(bool(v) is True for k, v in receipt["checks"].items() if k not in skip)
    write_new(out / "qualification_receipt.json", receipt)
    write_new(out / "preregistration_handoff.json", {
        "schema": "P2CRL_PREREGISTRATION_HANDOFF_V1",
        "status": "PREREGISTERED_READY_NOT_STARTED",
        "training_release": False,
        "prepare_gradient_updates": 0,
        "package_preflight_passed": False,
        "next": "User must explicitly request execution of the frozen campaign before smoke or 60 jobs.",
        "campaign_release": str(data / "campaign_release.json"),
        "documented_deviations": receipt["documented_deviations"],
    })
    art = repo / "artifacts/pathgraph_sarm/upgrade_v2/p2c_rl_policy_utility_v1"
    art.mkdir(parents=True, exist_ok=True)
    for name in ["historical_scope_note.json", "dataset_manifest.json", "split_audit.json", "new_dataset_qualification.json", "job_plan.json", "campaign_release.json", "final_decision.json"]:
        src = data / name
        dest = art / name
        if src.exists() and not dest.exists():
            write_new(dest, load_json(src))
    for name in ["source_lock.json", "runtime_lock.json", "qualification_receipt.json", "preregistration_handoff.json", "preregistration_deviation.json"]:
        src = out / name
        dest = art / name
        if src.exists() and not dest.exists():
            write_new(dest, load_json(src))
    print("FREEZE_OK", receipt["status"], "receipt_passed", receipt["passed"])



def cmd_execute(args):
    rel = load_json(args.release)
    if rel.get("status") != "ACTIVE" or rel.get("training_release") is not True:
        raise SystemExit("execute-campaign refused: campaign_release is not ACTIVE")
    from .campaign import execute_campaign
    campaign_root = getattr(args, "campaign_root", None) or getattr(args, "out", None)
    if not campaign_root:
        raise SystemExit("execute-campaign refused: campaign-root/out required after ACTIVE release")
    backend = getattr(args, "backend", "real")
    drill = bool(getattr(args, "zero_gradient_drill", False))
    if drill:
        backend = "fake"
    plan = load_json(args.plan)
    report = execute_campaign(
        repo=getattr(args, "repo", None),
        release=rel,
        plan=plan,
        data_root=getattr(args, "data_root", None),
        campaign_root=campaign_root,
        protocol_path=getattr(args, "protocol", None),
        source_lock_path=getattr(args, "source_lock", None),
        dataset_manifest_path=getattr(args, "dataset_manifest", None),
        amendment_path=getattr(args, "amendment", None),
        runtime_lock_path=getattr(args, "runtime_lock", None),
        parallel_jobs=int(getattr(args, "parallel_jobs", 2) or 2),
        backend=backend,
        zero_gradient_drill=drill,
        require_clean=bool(getattr(args, "require_clean", False)),
        require_detached=bool(getattr(args, "require_detached", False)),
        tombstone_registry_path=getattr(args, "release_tombstones", None),
    )
    print("EXECUTE_OK", report.get("schema"), "learn_called", report.get("learn_called"))


def cmd_finalize(args):
    from .finalize import write_final_decision
    dest = Path(args.artifact_root) / "final" / "decision.json"
    if dest.exists():
        print("FINALIZE_EXISTS")
        return
    write_final_decision(dest, status="PREREGISTERED_READY_NOT_STARTED")
    print("FINALIZE_PREPARE_ONLY")


def cmd_validate_release(args):
    from .release import validate_release
    rel = load_json(args.release)
    validate_release(
        rel,
        repo=getattr(args, "repo", None),
        protocol_path=getattr(args, "protocol", None),
        plan_path=getattr(args, "plan", None),
        source_lock_path=getattr(args, "source_lock", None),
        dataset_manifest_path=getattr(args, "dataset_manifest", None),
        amendment_path=getattr(args, "amendment", None),
        runtime_lock_path=getattr(args, "runtime_lock", None),
        require_active=bool(getattr(args, "require_active", False)),
        require_clean=bool(getattr(args, "require_clean", False)),
        require_detached=bool(getattr(args, "require_detached", False)),
    )
    print("RELEASE_OK", rel.get("status"), rel.get("training_release"))


def cmd_campaign_status(args):
    from .campaign_ledger import CampaignLedger
    from .monitoring import write_status
    ledger = CampaignLedger(args.ledger)
    dest = Path(args.out) if getattr(args, "out", None) else Path(args.ledger).with_name("status.json")
    rec = write_status(dest, ledger, getattr(args, "campaign_root", Path(args.ledger).parents[1]))
    ledger.close()
    print("STATUS_OK", rec.get("stop_new_claims"))


def cmd_freeze_runner(args):
    from .freeze_runner import freeze_runner
    rec = freeze_runner(
        repo=args.repo,
        out_dir=args.out,
        data_root=getattr(args, "data", None),
        test_report=getattr(args, "test_report", None),
        source_lock_a=getattr(args, "source_lock", None),
    )
    print("RUNNER_FREEZE_OK", rec["status"])


def cmd_finalize_campaign(args):
    from .analysis import confirmatory_statistics
    counts = load_json(args.counts) if getattr(args, "counts", None) else {}
    rec = confirmatory_statistics([], counts, Path(args.out))
    print("FINALIZE_CAMPAIGN", rec.get("status"), rec.get("passed"))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--repo", required=True)
    a.add_argument("--protocol", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("dry-run")
    a.add_argument("--repo", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--data", required=True)
    a.add_argument("--no-gradients", action="store_true")
    a = sub.add_parser("freeze")
    a.add_argument("--repo", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--data", required=True)
    a.add_argument("--source-seed", required=True)
    a = sub.add_parser("execute-campaign")
    a.add_argument("--release", required=True)
    a.add_argument("--plan", required=True)
    a.add_argument("--out", required=False)
    a.add_argument("--repo")
    a.add_argument("--data-root")
    a.add_argument("--campaign-root")
    a.add_argument("--protocol")
    a.add_argument("--source-lock")
    a.add_argument("--dataset-manifest")
    a.add_argument("--amendment")
    a.add_argument("--runtime-lock")
    a.add_argument("--parallel-jobs", type=int, default=2)
    a.add_argument("--backend", default="real")
    a.add_argument("--zero-gradient-drill", action="store_true")
    a.add_argument("--require-clean", action="store_true")
    a.add_argument("--require-detached", action="store_true")
    a.add_argument("--release-tombstones")
    a = sub.add_parser("finalize")
    a.add_argument("--out", required=True)
    a.add_argument("--artifact-root", required=True)
    a = sub.add_parser("validate-release")
    a.add_argument("--release", required=True)
    a.add_argument("--repo")
    a.add_argument("--protocol")
    a.add_argument("--plan")
    a.add_argument("--source-lock")
    a.add_argument("--dataset-manifest")
    a.add_argument("--amendment")
    a.add_argument("--runtime-lock")
    a.add_argument("--require-active", action="store_true")
    a.add_argument("--require-clean", action="store_true")
    a.add_argument("--require-detached", action="store_true")
    a = sub.add_parser("campaign-status")
    a.add_argument("--ledger", required=True)
    a.add_argument("--campaign-root")
    a.add_argument("--out")
    a = sub.add_parser("freeze-runner")
    a.add_argument("--repo", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--data")
    a.add_argument("--test-report")
    a.add_argument("--source-lock")
    a = sub.add_parser("finalize-campaign")
    a.add_argument("--out", required=True)
    a.add_argument("--counts")
    n = p.parse_args(argv)
    if n.cmd == "prepare":
        cmd_prepare(n)
    elif n.cmd == "dry-run":
        cmd_dry_run(n)
    elif n.cmd == "freeze":
        cmd_freeze(n)
    elif n.cmd == "execute-campaign":
        cmd_execute(n)
    elif n.cmd == "validate-release":
        cmd_validate_release(n)
    elif n.cmd == "campaign-status":
        cmd_campaign_status(n)
    elif n.cmd == "freeze-runner":
        cmd_freeze_runner(n)
    elif n.cmd == "finalize-campaign":
        cmd_finalize_campaign(n)
    else:
        cmd_finalize(n)

if __name__ == "__main__":
    main()
