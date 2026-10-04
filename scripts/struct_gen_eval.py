#!/usr/bin/env python
"""CP-DISR-TB-STRUCT-GEN-V1 post-hoc structural-generalization test evaluation (final checkpoint of each finished run only).

    python scripts/struct_gen_eval.py --root . --out <registration dir> --plan R-TB-SG-B2-0 --gpu N

Never called by any training code path. Reads the separate test split file; no optimizer, no provider, deterministic argmax, isolated RNG.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    from cp_disr import final_tb as ftb
    from cp_disr import final_tb_structgen as SG
    from cp_disr import struct_gen as G
    ftb.bind_worker_gpu(a.gpu)
    import torch
    state = json.loads((out / "launch_state.json").read_text())
    plan = state["plans"][a.plan]
    if plan["status"] != "COMPLETE":
        raise SystemExit("%s is %s; the structural test is evaluated on finished runs only" % (a.plan, plan["status"]))
    run_dir = Path(plan["run_dir"])
    target = run_dir / "eval_struct_test_final.json"
    if target.exists():
        raise SystemExit("already evaluated: %s" % target)
    ckpts = sorted((run_dir / "checkpoints").glob("final_n_*.pt"))
    if len(ckpts) != 1:
        raise SystemExit("expected one final checkpoint in %s, found %s" % (run_dir, ckpts))
    os.chdir(str(root))
    launch = json.loads((out / "launch_plan.json").read_text())
    base = SG.load_run_config(launch["configs"][a.plan]["path"])
    ctx = SG.dataclasses.replace(base, physical_gpu_index=a.gpu, render_gpu_device_id=a.gpu, gpu_uuid=ftb.query_gpu_uuid(a.gpu))
    v11 = SG.configure_v11(root, ctx)
    test_doc = json.loads((root / SG.TEST_REL).read_text())
    rows = test_doc["test"]
    tmp_split = out / "struct_gen_test_eval_split.json"
    G.write_json(tmp_split, {"task_id": "T_B", "train": [], "dev": rows, "test": [], "note": "post-hoc structural test evaluation file (derived from the frozen test split)"})
    v11.ENABLED_SPLITS[ftb.TASK] = tmp_split
    split_index = {r["case_id"]: r for r in rows}
    device = torch.device("cuda", 0)
    hashes = ftb.make_source_hashes(root, ctx)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ev = v11.eval_episodes(root, ftb.TASK, ctx.method, ckpts[0], [r["case_id"] for r in rows], split_index, device, hashes, run_dir / "eval_struct_test_final.tmp.json",
                           n_episodes=len(rows), label="struct_test")
    by_cell = {}
    for r in ev["rows"]:
        cell = split_index[r["case_id"]]["cell"]
        by_cell.setdefault(cell, []).append(r)
    ev["struct_test"] = {"plan_id": a.plan, "method": ctx.method, "seed": ctx.training_seed, "checkpoint": ckpts[0].name, "started": started,
                         "per_cell_success": {c: sum(1 for r in rs if r.get("success")) for c, rs in by_cell.items()}, "per_cell_n": {c: len(rs) for c, rs in by_cell.items()},
                         "cell_of_case": {r["case_id"]: split_index[r["case_id"]]["cell"] for r in ev["rows"]}}
    G.write_json(target, ev)
    (run_dir / "eval_struct_test_final.tmp.json").unlink()
    print(json.dumps({"plan": a.plan, "success": "%s/%s" % (ev["success_n"], ev["n"]), "per_cell": ev["struct_test"]["per_cell_success"]}))


if __name__ == "__main__":
    main()
