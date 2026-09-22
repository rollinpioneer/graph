#!/usr/bin/env python3
"""Write Plan v1.1 migration deliverables. Does not start training."""
from __future__ import annotations
import csv, json, hashlib, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
MIG = ROOT / "migration"
REP = ROOT / "reports"
ST = ROOT / "status"
PKG = ROOT / "packages" / "cp_disr_final_v31"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def loadj(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()

def git_branch():
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT), text=True).strip()

def main():
    MIG.mkdir(parents=True, exist_ok=True)
    s0a = loadj(ST / "stage_0a.json", {})
    s0b = loadj(ST / "stage_0b.json", {})
    s0c = loadj(ST / "stage_0c.json", {})
    s0d = loadj(ST / "stage_0d.json", {})
    inv = (MIG / "legacy_run_inventory.csv").read_text(encoding="utf-8") if (MIG / "legacy_run_inventory.csv").exists() else ""
    receipt = {
        "execution_reason": "SUPERSEDED_BY_PLAN_V1_1",
        "verified_at": utc(),
        "reported_clues": {
            "repository": str(ROOT),
            "reported_branch": "codex/cp-disr-v2.1-stage-2a-exploration",
            "reported_commit": "6141e47",
            "reported_orchestrator_pid": 2275072,
            "reported_gpus": [2, 1],
        },
        "actual_head_at_stop": "6141e47212671526de1e6bb86699f20484ecfb24",
        "new_branch": git_branch(),
        "current_head": git_hash(),
        "processes": {
            "orchestrator_2275072": "not_running",
            "bash_2275069": "not_running",
            "trainer_2688488": "not_running",
            "trainer_2726993": "not_running",
            "watch_2387622": "not_running",
        },
        "stop_method": "Verified PIDs dead; wrote STOP_SUPERSEDED_BY_PLAN_V1_1.json; run_matrix.py exits if that file exists. No blind kill of other users. No resume-to-64. No seed1/2 dispatch. Did not start queued T_C Full to complete 8/8 first updates.",
        "guard_file": "experiments/part_2_exploration/stage_2a/orchestrator/STOP_SUPERSEDED_BY_PLAN_V1_1.json",
        "old_stage_2a_status": "RUNNING with execution_reason=SUPERSEDED_BY_PLAN_V1_1 (not PASS)",
        "interrupted_jobs": [
            {"run_id": "stage2a_T_C_B2_seed0", "last_safe_point": "760 valid transitions, 0 PPO updates, eval update_0.pt only, KeyboardInterrupt inside MuJoCo step"},
            {"run_id": "stage2a_T_C_Full_seed0", "last_safe_point": "609 valid transitions, 0 PPO updates, eval update_0.pt only, KeyboardInterrupt inside MuJoCo step"},
        ],
        "preserved": ["transitions jsonl", "checkpoints present at interrupt", "optimizer/RNG where saved", "logs", "actual consumption"],
        "new_profile_rl_runs_this_round": 0,
    }
    write_json(MIG / "legacy_stop_receipt.json", receipt)

    # planned runs mapping
    src_csv = PKG / "experiments" / "planned_runs.csv"
    if not src_csv.exists():
        src_csv = ROOT / "packages/cp_disr_final_v31/experiments/planned_runs.csv"
    rows = list(csv.DictReader(src_csv.open(encoding="utf-8-sig")))
    first = [r for r in rows if r.get("phase") == "first_pause"]
    mapped = []
    for r in first:
        mapped.append({
            "planned_id": r["planned_id"],
            "stage": r["stage"],
            "task": r["task"],
            "method": r["method"],
            "seed": r["seed"],
            "N_cap": r["N_cap"],
            "updates_cap": r["updates_cap"],
            "status": "NOT_STARTED",
            "actual_run_id": "",
            "control_reference": r.get("control_reference", ""),
        })
    out_csv = MIG / "planned_runs_v11_mapping.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mapped[0].keys()))
        w.writeheader()
        w.writerows(mapped)
        for r in rows:
            if r.get("phase") != "first_pause":
                w.writerow({
                    "planned_id": r["planned_id"],
                    "stage": r["stage"],
                    "task": r["task"],
                    "method": r["method"],
                    "seed": r["seed"],
                    "N_cap": r["N_cap"],
                    "updates_cap": r["updates_cap"],
                    "status": "NOT_STARTED",
                    "actual_run_id": "",
                    "control_reference": r.get("control_reference", ""),
                })

    infra = {
        "reused": [
            "LIBERO_CP_DISR_CLEAN@8f1084e3132a39270c3a13ebe37270a43ece2a01",
            "robosuite 1.4.0 / mujoco 3.6.0 via lerobotpi0 site-packages",
            ".venv-stage0a python for unit tests",
            "DashScope cn-beijing https://dashscope.aliyuncs.com/api/v1 qwen3.8-max-0902",
            "frozen prompt/schema/3 few-shots",
            "D0 8 robosuite Stage-1A VLM caches",
            "T_C 8 Stage-2A VLM caches",
            "skill timeouts OPEN16/PICK9/PLACE8/PLACE_BUFFER8",
        ],
        "not_reused_as_new_results": [
            "old B0 RGCN / old B1-H checkpoints",
            "old actor_episode_discount_weight=true trajectories",
            "old Tcap=2*Ncap*d_ref",
            "T_A family and T_A caches as T_B",
            "old weighted checkpoint selection",
        ],
        "new": [
            "T_B split configs/splits/T_B_stage_0a.json",
            "T_B resolved task configs/tasks/resolved/T_B.yaml",
            "T_B 8 VLM caches under experiments/vlm_cache/plan_v11",
            "suite H from new 0A reference",
            "branch codex/cp-disr-method-2.1.1-plan-1.1-migration",
        ],
    }
    (MIG / "infrastructure_reuse_manifest.yaml").write_text(
        "reused:\n" + "".join("  - %s\n" % x for x in infra["reused"]) +
        "not_reused_as_new_results:\n" + "".join("  - %s\n" % x for x in infra["not_reused_as_new_results"]) +
        "new:\n" + "".join("  - %s\n" % x for x in infra["new"]),
        encoding="utf-8",
    )

    log = [
        "# Manifest resolution log (Plan v1.1 / Method 2.1.1 / Document 3.1)",
        "",
        "Standalone Experimental Plan SHA256 (zip copy) = `96588729F3B5649F73B6BFE4BB39C4FF1E540467FC45E7E87A1F3F48AA372EA5`.",
        "",
        "Identity: document_version=3.1, method_version=2.1.1, training_profile_revision=2.1.1, plan_version=1.1.",
        "",
        "## Template leftovers (source package not silently edited)",
        "",
        "- `interfaces/runtime_manifest.template.yaml` still labels method_version=2.1 and document_version=3.0. Derived runtime uses 2.1.1 / 3.1.",
        "- `paper_method_manifest.yaml` generic evaluation seed lists do not replace Stage-specific seeds. First pause uses seed0 only.",
        "- Package status templates remain NOT_STARTED. They were not copied over old `experiments/stage_status/*.json` actual ledgers.",
        "",
        "## Runtime derivation",
        "",
        "- New runtime: `experiments/manifests/runtime_manifest_v211.yaml`.",
        "- Old D0 `experiments/manifests/runtime_manifest.yaml` left in place.",
        "- T_A exits the default matrix. New 2A = T_B/T_C x B0/B1-K/B2/Full x seed0.",
        "- fewshot_manifest bound to existing frozen 3 examples; model access verified only if Stage 0C issued a live T_B request.",
        "",
    ]
    (MIG / "manifest_resolution_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")

    first_ids = [r["planned_id"] for r in first]
    readiness = {
        "plan_version": "1.1",
        "method_version": "2.1.1",
        "document_version": "3.1",
        "new_profile_rl_runs_this_round": 0,
        "first_pause_planned_ids": first_ids,
        "first_pause_status": {pid: "NOT_STARTED" for pid in first_ids},
        "stage_0a": s0a.get("status"),
        "stage_0b": s0b.get("status"),
        "stage_0c": s0c.get("status"),
        "stage_0d": s0d.get("status"),
        "next_authorizable": None,
        "blocked": [],
    }
    chain = [s0a.get("status"), s0b.get("status"), s0c.get("status"), s0d.get("status")]
    if "BLOCKED" in chain:
        readiness["next_authorizable"] = "repair blocked Part 0 stage; do not authorize 1A"
        readiness["blocked"] = [k for k, v in [("0A", s0a), ("0B", s0b), ("0C", s0c), ("0D", s0d)] if v.get("status") == "BLOCKED"]
    elif s0d.get("status") in (None, "NOT_STARTED"):
        readiness["next_authorizable"] = "Stage 0D still required before any 1A authorization"
    elif s0d.get("status") == "NEEDS_RERUN":
        readiness["next_authorizable"] = "limited task/interface review; not Stage 1A multi-seed"
    elif s0d.get("status") == "PASS_WITH_NOTES":
        readiness["next_authorizable"] = "Stage 1A single-seed smoke only, if explicitly authorized"
    elif s0d.get("status") == "PASS":
        readiness["next_authorizable"] = "Stage 1A (D0 x B2/Full x seed0) only after explicit authorization"
    write_json(MIG / "new_profile_readiness.json", readiness)

    # cost ledger
    n_ref = 0
    ap = ROOT / "runs/stage_0a/reference_attempts.jsonl"
    if ap.exists():
        n_ref = sum(1 for line in ap.read_text(encoding="utf-8").splitlines() if line.strip())
    n_vlm = int(s0c.get("materialized_tb") or 0)
    n_0d = 0
    if (ROOT / "runs/stage_0d/random_episode_log.jsonl").exists():
        n_0d = sum(1 for line in (ROOT / "runs/stage_0d/random_episode_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    cost_rows = [
        {"bucket": "legacy_stage2a_old_profile", "unit": "valid_transitions", "quantity": 1024*6 + 760 + 609, "note": "old 8 jobs; not new-profile"},
        {"bucket": "legacy_stage2a_old_profile", "unit": "ppo_updates", "quantity": 1*6, "note": "T_A four + T_C B0/B1; T_C B2/Full 0 updates"},
        {"bucket": "part0_0a_reference", "unit": "scripted_episodes", "quantity": n_ref, "note": "not RL"},
        {"bucket": "part0_0b_unittests", "unit": "pytest_sessions", "quantity": 1, "note": "not RL"},
        {"bucket": "part0_0c_vlm", "unit": "new_api_requests", "quantity": n_vlm, "note": "T_B only; D0/T_C reused"},
        {"bucket": "part0_0d_random", "unit": "diagnostic_episodes", "quantity": n_0d, "note": "not PPO/BC"},
        {"bucket": "new_profile_rl", "unit": "training_runs", "quantity": 0, "note": "authorization excluded 1A/2A"},
    ]
    with (MIG / "cumulative_cost_ledger.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "unit", "quantity", "note"])
        w.writeheader()
        w.writerows(cost_rows)

    md = [
        "# Plan v1.1 migration summary",
        "",
        "Identity: Research Specification v3.1 / Method 2.1.1 / Experimental Plan v1.1.",
        "",
        "## 1. Old orchestrator",
        "",
        "Stopped. PIDs 2275072/2275069/2688488/2726993/2387622 are not running. Guard file forbids dispatch/resume-to-64/seed1-2. Old Stage 2A remains RUNNING+SUPERSEDED, not PASS.",
        "",
        "## 2. Old 8 first-update jobs (actual, not the stale 5/8 snapshot)",
        "",
        "See `migration/legacy_run_inventory.csv`. T_A B0/B1/B2/Full seed0: 1 PPO update, 1024 transitions, latest.pt. T_C B0: 1 update/1024/latest.pt. T_C B1: 1 update recorded, optimizer_steps=60, latest.pt. T_C B2: 760 transitions, 0 updates, SIGINT. T_C Full: 609 transitions, 0 updates, SIGINT.",
        "",
        "## 3. Old checkpoints not reusable as v2.1.1 results",
        "",
        "Old B0/B1 definitions, actor prefix weights, Tcap=2*Ncap*d_ref, T_A in default matrix, weighted checkpoint rule. Historical only.",
        "",
        "## 4. T_B binding / T_A exit",
        "",
        "T_B goals are Inside(target,container) AND AtBuffer(second_object,buffer). T_A is out of the default matrix. T_A caches were not reused as T_B.",
        "",
        "## 5. Production code",
        "",
        "B0 two-layer Set Transformer; B1=B1-K phiK(c,0); A_CAT 4-view concat + contract-repeat anchor; actor_episode_discount_weight=false; GRU prefix recompute kept; Gamma=2^(-d/H); Tcap=Ncap*d_ref.",
        "",
        "## 6. H and d_ref",
        "",
        json.dumps(s0a.get("families") or s0a.get("H_seconds"), ensure_ascii=False),
        "",
        "H=%s seconds from 5/5 legal scripted successes per family." % s0a.get("H_seconds"),
        "",
        "## 7. New 0A-0D",
        "",
        "- 0A: %s" % s0a.get("status"),
        "- 0B: %s" % s0b.get("status"),
        "- 0C: %s" % s0c.get("status"),
        "- 0D: %s" % s0d.get("status"),
        "",
        "## 8. Cache reuse",
        "",
        "See `migration/cache_reuse_manifest.csv`. D0/T_C exact reuse; T_B newly captured+requested; T_A not reused.",
        "",
        "## 9. First-pause 11 planned runs",
        "",
        "All remain NOT_STARTED. No new-profile RL this round (count=0).",
        "",
        "## 10. Next authorizable stage",
        "",
        str(readiness.get("next_authorizable")),
        "",
        "Blocked: %s" % readiness.get("blocked"),
        "",
        "This round stops here. Do not start Stage 1A/1B/2A/2B/3A without a new explicit authorization.",
        "",
    ]
    (MIG / "migration_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"readiness": readiness, "receipt": str(MIG / "legacy_stop_receipt.json")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
