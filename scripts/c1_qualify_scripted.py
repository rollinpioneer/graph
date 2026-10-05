#!/usr/bin/env python
"""CP-DISR-C1-MECH-CONFIRM-V1 Stage 4: physical qualification of the 8 Fresh Confirm qualification cases with the frozen first shortest legal plan (no policy, no training).

    python scripts/c1_qualify_scripted.py --root . --gpu N

Each case: reset, then execute the first optimal symbolic plan (from the frozen initial public facts) through the shared Collector / controller / verifier / evaluator.
Criteria (runbook 5.6): reset succeeds, every skill legal under the public hard mask, every controller exit NORMAL_TERMINATION (no collision / reach failure / timeout),
the frozen Evaluator returns TASK_SUCCESS; no online coordinate tuning, no replacement. Verified-vs-nominal fact discrepancies are recorded, not gating (as in the
already disclosed T_B qualification amendment); we do not claim the nominal successor equals the real observed state. Refuses to run twice.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PREP = Path("runs/final_master/c1_route_b/mech_confirm_v1/prep")
CARD = "CP-DISR-C1-MECH-CONFIRM-V1"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    prep = root / PREP
    result_path = prep / "physical_qualification_results.json"
    if result_path.exists() or (prep / "physical_qualification_results.started.json").exists():
        raise SystemExit("qualification already attempted (no retries): %s" % result_path)
    from cp_disr import c1_fresh_confirm as F
    from cp_disr import final_tb as ftb
    from cp_disr import final_tb_c1 as C1
    from cp_disr import struct_gen as G
    F.install_goal_sets()
    ftb.bind_worker_gpu(a.gpu)
    import torch
    from cp_disr.collector import Collector
    from cp_disr.contracts import nominal_overlay
    os.chdir(str(root))
    split_doc = json.loads((root / F.QUAL_REL).read_text())
    rows = split_doc["qualification"]
    manifest, contracts, cache = G.build_manifest(root, [dict(r, split="test") for r in rows])
    by_id = {c["case_id"]: c for c in manifest["cases"]}
    tmp_split = prep / "qualification" / "scripted_eval_split.json"
    G.write_json(tmp_split, {"task_id": "T_B", "train": [], "dev": rows, "test": [], "note": "qualification rows only"})
    info = ftb.derive_runtime_manifest(root, prep / "qualification" / "scripted_runtime_manifest.yaml", str(tmp_split))
    uuid = ftb.query_gpu_uuid(a.gpu)
    head = ftb.git_out(root, "rev-parse", "HEAD")
    ctx = ftb.RunContext(plan_id="C1QUAL", attempt_id="C1QUAL", method="B2", training_seed=0, source_commit=head, output_directory=str(prep / "qualification" / "unused"), runtime_manifest=info["path"],
                         train_split=str(tmp_split), physical_gpu_index=a.gpu, render_gpu_device_id=a.gpu, gpu_uuid=uuid)
    v11 = C1.configure_v11(root, ctx)
    v11.ENABLED_SPLITS[ftb.TASK] = tmp_split
    v11.bind_H(root)
    device = torch.device("cuda", 0)
    result = {"card": CARD, "started": now(), "head": head, "gpu": a.gpu, "executor": "frozen first shortest legal symbolic plan (open loop from the initial public facts)", "episodes": 0, "optimizer_steps": 0,
              "policy_training": False, "provider_requests": 0, "cases": {}, "verdict": "FAIL",
              "disclosure": "A B_PLAN dry run of the same 8 cases was executed first (bplan_qualification_dry_run.json). It validated the planner/evaluation pipeline; one case ended NO_PLAN under public "
                            "UNKNOWN facts (partial observation of the target after the first PICK), which is a planner outcome, not a physical failure. The runbook allows either executor; this file is "
                            "the qualification verdict, executed once per case with the frozen first shortest plan. 16 environment episodes in total, all on the 8 qualification cases, none replaced."}
    from cp_disr.facts import Truth
    G.write_json(prep / "physical_qualification_results.started.json", {"started": result["started"], "note": "attempt charged before the first environment call"})
    bundle = None
    try:
        bundle = v11.make_bundle(root, "T_B")
        deadline = float(v11.resolve_runtime(root)["task_deadlines"]["T_B"])
        for row in rows:
            cid = row["case_id"]
            rec = {"cell": row["cell"], "depth": by_id[cid]["dependency_depth"], "steps": [], "checks": {}, "ok": False}
            result["cases"][cid] = rec
            result["episodes"] += 1
            try:
                snap = bundle.start_case(cid)
                snap, _prior, _n = v11.apply_episode_prior("B2", bundle, snap, None, eval_original=True)
                bundle.current_snapshot = snap
                rec["initial_unknown_facts"] = sorted(k for k, v in snap.facts.values.items() if v == Truth.UNKNOWN)
                torch.manual_seed(0)
                policy = v11.make_policy(bundle.template, "B2", device)
                policy.eval()
                selector = ftb.ScriptedSelector(policy)
                collector = Collector(bundle, selector)
                collector.reset_episode(snap.env_id, snap.episode_id)
                plan_actions = sorted(by_id[cid]["optimal_plans"])[0]
                contracts_by_id = {c.id: c for c in snap.template.contracts}
                start_clock = float(bundle.clock.now_seconds())
                legal_ok, exit_ok, success_final = True, True, None
                for i, action in enumerate(plan_actions):
                    step = {"index": i, "action": action}
                    rec["steps"].append(step)
                    idx = snap.candidate_ids.index(action)
                    step["legal"] = bool(snap.mask[idx])
                    legal_ok &= step["legal"]
                    if not step["legal"]:
                        break
                    prev = dict(snap.facts.values)
                    predicted = {k: v.value for k, v in nominal_overlay(contracts_by_id[action], prev, snap.template.derived_rules, snap.template.exclusive_groups).items()}
                    selector.target = action
                    t, res = collector.step(snap, deterministic=True)
                    if t is None:
                        step["no_transition"] = str(res)[:300]
                        exit_ok = False
                        break
                    raw = dict(getattr(bundle.executor, "last", None) or {})
                    step.update(controller_exit=raw.get("controller_exit"), sim_duration=raw.get("sim_duration"), duration=float(t.duration))
                    exit_ok &= raw.get("controller_exit") == "NORMAL_TERMINATION"
                    observed = {k: v.value for k, v in t.next_snapshot.facts.values.items()}
                    step["fact_mismatches_vs_nominal"] = {k: {"observed": observed[k], "nominal": predicted[k]} for k in observed if observed[k] != predicted[k]}
                    success_now = bool(res.success if not isinstance(res, dict) else res.get("success"))
                    step.update(evaluator_success=success_now, evaluator_reason=(res.reason if not isinstance(res, dict) else res.get("reason")))
                    success_final = success_now
                    snap = t.next_snapshot
                    bundle.current_snapshot = snap
                elapsed = float(bundle.clock.now_seconds()) - start_clock
                rec.update(elapsed_seconds=elapsed, deadline_seconds=deadline, plan=list(plan_actions))
                rec["checks"] = {"reset_succeeded": True, "all_steps_legal": bool(legal_ok), "all_controller_exits_normal": bool(exit_ok and len(rec["steps"]) == len(plan_actions)),
                                 "evaluator_task_success": bool(success_final), "elapsed_below_0.8_deadline": elapsed < 0.8 * deadline}
                rec["ok"] = all(rec["checks"].values())
            except Exception as exc:  # recorded, never swallowed; no retry
                rec["error"] = "%s: %s" % (type(exc).__name__, exc)
                rec["traceback"] = traceback.format_exc()
                rec["ok"] = False
    finally:
        if bundle is not None:
            try:
                bundle.environment.close()
            except Exception:
                pass
    cells = {}
    for cid, rec in result["cases"].items():
        cells.setdefault(rec["cell"], []).append(rec["ok"])
    result["cell_verdicts"] = {c: ("PASS" if len(v) == 2 and all(v) else "FAIL") for c, v in cells.items()}
    result["finished"] = now()
    result["verdict"] = "PASS" if len(result["cases"]) == 8 and all(v["ok"] for v in result["cases"].values()) and len(cells) == 4 else "FAIL"
    if result["verdict"] == "FAIL":
        both = [c for c, v in cells.items() if len(v) == 2 and not any(v)]
        result["stop_flag"] = "STOPPED_FRESH_CELL_PHYSICALLY_UNQUALIFIED" if both else "STOPPED_FRESH_QUALIFICATION_FAILED"
    G.write_json(result_path, result)
    print(json.dumps({"verdict": result["verdict"], "cell_verdicts": result["cell_verdicts"], "cases": {k: [v["ok"], v.get("checks"), v.get("error")] for k, v in result["cases"].items()}}, indent=1))


if __name__ == "__main__":
    main()
