#!/usr/bin/env python
"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1 E2 / E3: ID gate, then the one-shot formal evaluation (A0, A1, A2, B, planner) of the three final checkpoints.

    python scripts/c1_bw_eval.py --root . --run-root <registered run root> --gpu N

Never called by any training path. Each (model, case) is evaluated once: an existing output file refuses the run. Deterministic argmax, final checkpoint only, no checkpoint
selection, no optimizer. The test split files are opened here only after every run passed the ID gate; A1 / A2 / B are opened only after the A0 gate passed.
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
    ap.add_argument("--only", help="comma separated split names (a0,a1,a2,b); default = all in order with the A0 gate")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch
    from cp_disr.blocksworld import classification as CL
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld import train as T
    from cp_disr.blocksworld import launch as L
    from cp_disr.c1_blocksworld_policies import make_policy
    from cp_disr.rl import set_suite_half_life
    from cp_disr.torch_rl import load_checkpoint
    root, run_root = Path(a.root).resolve(), Path(a.run_root).resolve()
    plan = json.loads((run_root / "registration" / "launch_plan.json").read_text())
    L.check_source_identity(root, plan["prep_commit"])
    state = json.loads((run_root / "registration" / "launch_state.json").read_text())
    runs = {rid: r for rid, r in state["runs"].items()}
    receipts = run_root / "receipts"
    receipts.mkdir(exist_ok=True)
    # ---- E2: ID gate (all three must have finished)
    gate = {}
    for rid, r in runs.items():
        if r["status"] != "COMPLETE":
            raise SystemExit("%s is %s: formal evaluation needs all three runs COMPLETE" % (rid, r["status"]))
        acct = r["accounting"]
        gate[rid] = {"id_dev_success_n": acct["final_id_dev_success_n"], "id_dev_decision_perfect_n": acct["final_id_dev_decision_perfect_n"], "nan": acct["nan_events"], "hard_fail": acct["hard_fail"],
                     "stop_reason": acct["stop_reason"], "N": acct["N"], "pass": CL.id_gate(acct["final_id_dev_success_n"], acct["final_id_dev_decision_perfect_n"], acct["nan_events"], acct["hard_fail"])}
    (receipts / "id_gate.json").write_text(json.dumps({"gate": gate, "all_pass": all(g["pass"] for g in gate.values())}, indent=1) + "\n")
    if not all(g["pass"] for g in gate.values()):
        (receipts / "STATE.txt").write_text("PPO_ID_GATE_FAIL\nNEXT_ACTION = WAIT_FOR_USER_DECISION_ON_PLANNER_IMITATION\n")
        print(json.dumps({"state": "PPO_ID_GATE_FAIL", "gate": gate}, indent=1))
        return
    spec = json.loads((root / L.PREP_REL / "task_spec.json").read_text())
    set_suite_half_life(float(spec["half_life_H"]))
    device = torch.device("cuda", 0)
    only = a.only.split(",") if a.only else None
    files = {"a0": "configs/splits/c1_bw_a0_iso_v1.json", "a1": "configs/splits/c1_bw_a1_color_reverse_v1.json", "a2": "configs/splits/c1_bw_a2_noniso_v1.json", "b": "configs/splits/c1_bw_b_scale_v1.json"}
    models = {}
    for rid, r in runs.items():
        m = make_policy(r["method"], device, 0)
        load_checkpoint(r["accounting"]["final_checkpoint"], m)
        m.eval()
        models[rid] = m
    hops = T.Hops()
    results = {}
    for split in ("a0", "a1", "a2", "b"):
        if only and split not in only:
            continue
        cases, _doc = T.load_cases(root / files[split], "cases")
        for rid, m in models.items():
            out_path = Path(runs[rid]["accounting"]["final_checkpoint"]).parent.parent / ("eval_%s_final.json" % split)
            if out_path.exists():
                raise SystemExit("already evaluated (each model x case once): %s" % out_path)
            solver = P.Solver()
            t0 = time.time()
            eps = T.evaluate(m, cases, solver, hops, record_decisions=True)
            s = T.summarize(eps)
            out_path.write_text(json.dumps({"run_id": rid, "method": runs[rid]["method"], "split": split, "summary": s, "episodes": eps, "wall_seconds": time.time() - t0,
                                            "checkpoint": runs[rid]["accounting"]["final_checkpoint"]}, indent=1, default=str), encoding="utf-8")
            with (out_path.parent / "decision_log.jsonl").open("a", encoding="utf-8") as f:
                for e in eps:
                    for d in e["decisions"]:
                        f.write(json.dumps({"run_id": rid, "split": split, "case_id": e["case_id"], **d}, default=str) + "\n")
            results[(rid, split)] = s
            print(json.dumps({"run": rid, "split": split, **s}))
        if split == "a0":
            ok = {rid: CL.a0_gate(results[(rid, "a0")]["decision_perfect_rate"]) for rid in models}
            (receipts / "a0_gate.json").write_text(json.dumps({"a0_decision_perfect": {rid: results[(rid, "a0")]["decision_perfect_rate"] for rid in models}, "pass": ok}, indent=1) + "\n")
            if not all(ok.values()):
                (receipts / "STATE.txt").write_text("EQUIVARIANCE_OR_BINDING_GATE_FAIL\nNEXT_ACTION = STOP_MAIN_RESULT_INTERPRETATION\n")
                print(json.dumps({"state": "EQUIVARIANCE_OR_BINDING_GATE_FAIL", "a0": ok}))
                return
    pl_path = run_root / "eval" / "planner_baseline.json"
    if not pl_path.exists() and not only:
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
