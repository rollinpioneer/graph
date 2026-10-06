#!/usr/bin/env python
"""CP-DISR-C1-BW-IMITATION-A02: imitation ID gate on dev36 (each final imitation checkpoint once), then - only if all three pass - the one-shot A0, A1, A2, B evaluation and the exact planner baseline.

    python scripts/c1_bw_il_eval.py --root . --run-root <imitation_a02 run root> --gpu N

Never called by any training path. dev36 is opened here, after all three trainings are COMPLETE. Each (model, case) is evaluated once: an existing output file refuses the run. Deterministic argmax, the
FINAL imitation checkpoint only (no mid checkpoint, no PPO checkpoint), no optimizer. A1 / A2 / B are opened only after the A0 gate passed. Any ID-gate failure stops immediately:
IMITATION_ID_GATE_FAIL, limitation_label = LIMIT_TRAINING, no REP label, NEXT_ACTION = WAIT_FOR_USER_MAINLINE_DECISION.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--planner-only", action="store_true", help="only write the exact-planner baseline file (dry-run use only)")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch
    from cp_disr.blocksworld import classification as CL
    from cp_disr.blocksworld import imitation_launch as IL
    from cp_disr.blocksworld import launch as L
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld import train as T
    from cp_disr.blocksworld.imitation import make_imitation_policy, sha256_file
    from cp_disr.rl import set_suite_half_life
    from cp_disr.torch_rl import load_checkpoint
    root, run_root = Path(a.root).resolve(), Path(a.run_root).resolve()
    plan = json.loads((run_root / "registration" / "launch_plan.json").read_text())
    IL.check_source_identity_il(root, plan["prep_commit"], exact=False)
    state = json.loads((run_root / "registration" / "launch_state.json").read_text())
    runs = dict(state["runs"])
    receipts = run_root / "receipts"
    receipts.mkdir(exist_ok=True)
    for rid, r in runs.items():
        if r["status"] != "COMPLETE":
            raise SystemExit("%s is %s: the ID gate needs all three imitation runs COMPLETE" % (rid, r["status"]))
        if sha256_file(r["accounting"]["final_checkpoint"]) != r["accounting"]["final_checkpoint_sha256"]:
            raise SystemExit("final checkpoint of %s changed" % rid)
    cfg = IL.frozen_config(root)
    set_suite_half_life(float(cfg["half_life"]))
    device = torch.device("cuda", 0)
    models = {}
    for rid, r in runs.items():
        m = make_imitation_policy(r["method"], device, 0)
        load_checkpoint(r["accounting"]["final_checkpoint"], m)
        m.eval()
        models[rid] = m
    hops = T.Hops()
    # ---- ID gate: dev36, each final imitation checkpoint once
    dev_cases, _doc = T.load_cases(root / L.TRAIN_SPLIT_REL, "dev")
    gate = {}
    for rid, m in models.items():
        acct = runs[rid]["accounting"]
        out_path = Path(acct["final_checkpoint"]).parent.parent / "eval_id_final.json"
        if out_path.exists():
            raise SystemExit("already evaluated (each final checkpoint once): %s" % out_path)
        eps = T.evaluate(m, dev_cases, P.Solver(), hops, record_decisions=True)
        s = T.summarize(eps)
        out_path.write_text(json.dumps({"run_id": rid, "method": runs[rid]["method"], "split": "dev", "summary": s, "episodes": eps, "checkpoint": acct["final_checkpoint"],
                                        "checkpoint_sha256": acct["final_checkpoint_sha256"]}, indent=1, default=str), encoding="utf-8")
        gate[rid] = {"id_dev_success_n": s["success_n"], "id_dev_decision_perfect_n": s["decision_perfect_n"], "n": s["n"], "nan": acct["nan_events"], "hard_fail": acct["hard_fail"],
                     "pass": CL.id_gate(s["success_n"], s["decision_perfect_n"], acct["nan_events"], acct["hard_fail"])}
        print(json.dumps({"id_gate": rid, **gate[rid]}), flush=True)
    all_pass = all(g["pass"] for g in gate.values())
    (receipts / "id_gate.json").write_text(json.dumps({"gate": gate, "all_pass": all_pass, "rule": CL.RULES["id_gate"]}, indent=1) + "\n")
    if not all_pass:
        (receipts / "STATE.txt").write_text("IMITATION_ID_GATE_FAIL\nlimitation_label = LIMIT_TRAINING\nrepresentation_label = NOT_ISSUED\nNEXT_ACTION = WAIT_FOR_USER_MAINLINE_DECISION\n")
        print(json.dumps({"state": "IMITATION_ID_GATE_FAIL", "gate": gate}, indent=1))
        return
    (receipts / "STATE.txt").write_text("IMITATION_ID_GATE_PASS\n")
    files = {"a0": "configs/splits/c1_bw_a0_iso_v1.json", "a1": "configs/splits/c1_bw_a1_color_reverse_v1.json", "a2": "configs/splits/c1_bw_a2_noniso_v1.json", "b": "configs/splits/c1_bw_b_scale_v1.json"}
    results = {}
    for split in ("a0", "a1", "a2", "b"):
        if a.planner_only:
            continue
        cases, _doc = T.load_cases(root / files[split], "cases")
        for rid, m in models.items():
            out_path = Path(runs[rid]["accounting"]["final_checkpoint"]).parent.parent / ("eval_%s_final.json" % split)
            if out_path.exists():
                raise SystemExit("already evaluated (each model x case once): %s" % out_path)
            t0 = time.time()
            eps = T.evaluate(m, cases, P.Solver(), hops, record_decisions=True)
            s = T.summarize(eps)
            out_path.write_text(json.dumps({"run_id": rid, "method": runs[rid]["method"], "split": split, "summary": s, "episodes": eps, "wall_seconds": time.time() - t0,
                                            "checkpoint": runs[rid]["accounting"]["final_checkpoint"], "checkpoint_sha256": runs[rid]["accounting"]["final_checkpoint_sha256"]}, indent=1, default=str),
                                encoding="utf-8")
            with (out_path.parent / "decision_log.jsonl").open("a", encoding="utf-8") as f:
                for e in eps:
                    for d in e["decisions"]:
                        f.write(json.dumps({"run_id": rid, "split": split, "case_id": e["case_id"], **d}, default=str) + "\n")
            results[(rid, split)] = s
            print(json.dumps({"run": rid, "split": split, **s}), flush=True)
        if split == "a0":
            ok = {rid: CL.a0_gate(results[(rid, "a0")]["decision_perfect_rate"]) for rid in models}
            (receipts / "a0_gate.json").write_text(json.dumps({"a0_decision_perfect": {rid: results[(rid, "a0")]["decision_perfect_rate"] for rid in models}, "pass": ok}, indent=1) + "\n")
            if not all(ok.values()):
                (receipts / "STATE.txt").write_text("EQUIVARIANCE_OR_BINDING_GATE_FAIL\nNEXT_ACTION = STOP_MAIN_RESULT_INTERPRETATION\n")
                print(json.dumps({"state": "EQUIVARIANCE_OR_BINDING_GATE_FAIL", "a0": ok}))
                return
    pl_path = run_root / "eval" / "planner_baseline.json"
    if not pl_path.exists():
        out = {"splits": {}}
        for split in ("a0", "a1", "a2", "b"):
            cases, _ = T.load_cases(root / files[split], "cases")
            rows = []
            for c in cases:
                s = P.Solver()
                t0 = time.perf_counter()
                plan_ = s.one_optimal_plan(c.problem.init, c.problem.goal)
                cpu = time.perf_counter() - t0
                state_ = c.problem.init
                for act in plan_:
                    state_ = S.apply(state_, act)
                rows.append({"case_id": c.case_id, "n_blocks": c.problem.n, "optimal_length": c.optimal_length, "plan_length": len(plan_), "success": S.goal_satisfied(c.problem, state_),
                             "expanded_nodes": s.stats.expanded, "generated_nodes": s.stats.generated, "wall_seconds": cpu})
            out["splits"][split] = rows
        pl_path.parent.mkdir(exist_ok=True)
        pl_path.write_text(json.dumps(out, indent=1) + "\n")
    (receipts / "STATE.txt").write_text("FORMAL_EVALUATION_DONE\n")
    print(json.dumps({"state": "FORMAL_EVALUATION_DONE"}))


if __name__ == "__main__":
    main()
