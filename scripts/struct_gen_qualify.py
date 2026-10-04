#!/usr/bin/env python
"""CP-DISR-TB-STRUCT-GEN-V1 minimal physical qualification (<= 6 scripted episodes, no policy training, no optimizer).

    python scripts/struct_gen_qualify.py --root . --gpu N

Runs the first optimal symbolic plan of each pre-registered qualification case through the frozen controller / verifier / evaluator
and checks the pre-registered criteria. No coordinate is changed, no case is re-sampled, no retry is made.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_REL = Path("runs/final_master/2.1.1/structgen/prep")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = root / OUT_REL
    result_path = out / "physical_qualification_results.json"
    if result_path.exists():
        raise SystemExit("qualification already attempted (no retries): %s" % result_path)
    from cp_disr import final_tb as ftb
    from cp_disr import struct_gen as G
    ftb.bind_worker_gpu(args.gpu)
    import torch
    from cp_disr import struct_gen_runtime as SR
    from cp_disr.collector import Collector
    from cp_disr.contracts import nominal_overlay
    os.chdir(str(root))
    plan = json.loads((out / "physical_qualification_plan.json").read_text())
    manifest = json.loads((out / "representation_challenge_manifest.json").read_text())
    by_id = {c["case_id"]: c for c in manifest["cases"]}
    rows_all = {r["case_id"]: r for r in SR.read_rows(root / "configs/splits/struct_gen_v1_train_dev.json") + SR.read_rows(root / "configs/splits/struct_gen_v1_test.json")}
    case_ids = [cid for level in ("DEPTH-1", "DEPTH-2", "DEPTH-3") for cid in plan["cases"][level]]
    assert len(case_ids) == len(set(case_ids)) <= 6
    qual_split = {"task_id": "T_B", "train": [rows_all[c] for c in case_ids], "dev": [], "test": [], "note": "qualification only"}
    split_path = out / "physical_qualification_split.json"
    G.write_json(split_path, qual_split)
    manifest_info = ftb.derive_runtime_manifest(root, out / "physical_qualification_runtime_manifest.yaml", str(split_path))
    uuid = ftb.query_gpu_uuid(args.gpu)
    head = ftb.git_out(root, "rev-parse", "HEAD")
    ctx = ftb.RunContext(plan_id="QUAL", attempt_id="QUAL", method="B2", training_seed=0, source_commit=head, output_directory=str(out / "unused_run_dir"),
                         runtime_manifest=manifest_info["path"], train_split=str(split_path), physical_gpu_index=args.gpu, render_gpu_device_id=args.gpu, gpu_uuid=uuid)
    v11 = ftb.configure_v11(root, ctx)
    SR.install(v11)
    v11.bind_H(root)
    device = torch.device("cuda", 0)
    result = {"card": G.CARD, "started": now(), "head": head, "gpu": args.gpu, "gpu_uuid": uuid, "criteria": plan["criteria"], "episodes": 0, "optimizer_steps": 0,
              "policy_training": False, "provider_requests": 0, "cases": {}, "verdict": "FAIL"}
    ftb.write_json_atomic(out / "physical_qualification_results.started.json", {"started": result["started"], "note": "attempt charged before the first environment call"})
    bundle = None
    try:
        bundle = v11.make_bundle(root, "T_B")
        deadline = float(v11.resolve_runtime(root)["task_deadlines"]["T_B"])
        for level, ids in plan["cases"].items():
            for cid in ids:
                rec = {"level": level, "cell": by_id[cid]["cell"], "depth": by_id[cid]["dependency_depth"], "steps": [], "checks": {}, "ok": False}
                result["cases"][cid] = rec
                result["episodes"] += 1
                try:
                    snap = bundle.start_case(cid)
                    snap, _prior, _n = v11.apply_episode_prior("B2", bundle, snap, None, eval_original=True)
                    bundle.current_snapshot = snap
                    expect = {k: v for k, v in by_id[cid]["initial_facts"].items()}
                    got = {k: v.value for k, v in snap.facts.values.items()}
                    rec["initial_facts_equal_symbolic"] = got == expect
                    rec["initial_unknown_facts"] = sorted(k for k, v in got.items() if v == "UNKNOWN")
                    torch.manual_seed(0)
                    policy = v11.make_policy(bundle.template, "B2", device)
                    policy.eval()
                    selector = ftb.ScriptedSelector(policy)
                    collector = Collector(bundle, selector)
                    collector.reset_episode(snap.env_id, snap.episode_id)
                    plan_actions = sorted(by_id[cid]["optimal_plans"])[0]
                    contracts = {c.id: c for c in snap.template.contracts}
                    start_clock = float(bundle.clock.now_seconds())
                    all_exit_ok, all_facts_ok, legal_ok, continue_ok, success_final = True, True, True, True, None
                    for i, action in enumerate(plan_actions):
                        step = {"index": i, "action": action}
                        rec["steps"].append(step)
                        idx = snap.candidate_ids.index(action)
                        step["legal"] = bool(snap.mask[idx])
                        legal_ok &= step["legal"]
                        if not step["legal"]:
                            break
                        prev_values = dict(snap.facts.values)
                        predicted = {k: v.value for k, v in nominal_overlay(contracts[action], prev_values, snap.template.derived_rules, snap.template.exclusive_groups).items()}
                        selector.target = action
                        t, res = collector.step(snap, deterministic=True)
                        if t is None:
                            step["no_transition"] = str(res)[:300]
                            all_exit_ok = False
                            break
                        raw = dict(getattr(bundle.executor, "last", None) or {})
                        step.update(controller_exit=raw.get("controller_exit"), sim_duration=raw.get("sim_duration"), duration=float(t.duration))
                        all_exit_ok &= raw.get("controller_exit") == "NORMAL_TERMINATION"
                        observed = {k: v.value for k, v in t.next_snapshot.facts.values.items()}
                        diffs = {k: {"observed": observed[k], "nominal": predicted[k]} for k in observed if observed[k] != predicted[k]}
                        step["fact_mismatches_vs_nominal"] = diffs
                        all_facts_ok &= not diffs
                        success_now = bool(res.success if not isinstance(res, dict) else res.get("success"))
                        reason = getattr(res, "reason", None) if not isinstance(res, dict) else res.get("reason")
                        step.update(evaluator_success=success_now, evaluator_reason=reason)
                        last = i == len(plan_actions) - 1
                        if not last:
                            continue_ok &= (not success_now) and reason == "CONTINUE"
                        success_final = success_now
                        snap = t.next_snapshot
                        bundle.current_snapshot = snap
                    elapsed = float(bundle.clock.now_seconds()) - start_clock
                    rec.update(elapsed_seconds=elapsed, deadline_seconds=deadline)
                    rec["checks"] = {"initial_facts_equal_symbolic_and_known": bool(rec["initial_facts_equal_symbolic"] and not rec["initial_unknown_facts"]), "all_steps_legal": bool(legal_ok),
                                     "all_controller_exits_normal": bool(all_exit_ok and len(rec["steps"]) == len(plan_actions)), "verified_facts_equal_nominal_every_step": bool(all_facts_ok),
                                     "evaluator_continue_until_last_then_success": bool(continue_ok and success_final), "elapsed_below_0.8_deadline": elapsed < 0.8 * deadline}
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
    levels = {}
    for cid, rec in result["cases"].items():
        levels.setdefault(rec["level"], []).append(rec["ok"])
    result["level_verdicts"] = {lvl: ("PASS" if oks and all(oks) else "FAIL") for lvl, oks in levels.items()}
    result["finished"] = now()
    result["verdict"] = "PASS" if result["episodes"] <= 6 and len(result["cases"]) == 6 and all(v == "PASS" for v in result["level_verdicts"].values()) else "FAIL"
    G.write_json(result_path, result)
    print(json.dumps({"verdict": result["verdict"], "levels": result["level_verdicts"], "episodes": result["episodes"],
                      "cases": {k: {"ok": v["ok"], "checks": v.get("checks"), "elapsed": v.get("elapsed_seconds"), "error": v.get("error")} for k, v in result["cases"].items()}}, indent=1))


if __name__ == "__main__":
    main()
