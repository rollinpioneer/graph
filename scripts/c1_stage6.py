#!/usr/bin/env python
"""CP-DISR-C1-MECH-CONFIRM-V1 Stage 6 / 7 launcher: start the formal Fresh Confirm evaluations (one background process per model, one GPU each; no model is evaluated twice).

    python scripts/c1_stage6.py --root . --reg <QMARK registration dir> --prep-commit <sha> --seed 0          # B2, ABS, +E, NC, QMARK, B_PLAN
    python scripts/c1_stage6.py --root . --reg <...> --prep-commit <sha> --seed 1 --methods B2,ABS,QMARK      # conditional matched seed

QMARK is evaluated only if its ID qualification holds (final dev12 = 12/12, 0 NaN, no hard failure). Existing checkpoints are verified against their recorded sha256
before launch. Writes <fresh_confirm>/launch_seed<k>.json. Each process is `scripts/c1_eval.py --formal` (resume-only; a finished case is never re-run).
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PREP = Path("runs/final_master/c1_route_b/mech_confirm_v1/prep")
FRESH = Path("runs/final_master/c1_route_b/mech_confirm_v1/fresh_confirm")
TEST_REL = "configs/splits/c1_fresh_confirm_v1_test.json"
METHOD_NAME = {"B2": "B2", "ABS": "B2-ABS", "E": "B1-K+E", "NC": "B1-K+NC", "QMARK": "B1-K+QMARK", "B_PLAN": "B_PLAN"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gpu_free(exclude=()):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True).splitlines()
    rows = [(int(l.split(",")[1]), int(l.split(",")[0])) for l in out]
    return [g for _m, g in sorted(rows) if g not in exclude and g != 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--reg", required=True)
    ap.add_argument("--prep-commit", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="B2,ABS,E,NC,QMARK,B_PLAN")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    methods = a.methods.split(",")
    inv = json.loads((root / PREP / "existing_evidence_inventory.json").read_text())["runs"]
    state = json.loads((Path(a.reg) / "launch_state.json").read_text())
    qplan = state["plans"]["R-TB-C1-QMARK-%d" % a.seed]
    plan = {}
    problems = []
    for m in methods:
        if m == "B_PLAN":
            if a.seed == 0:
                plan[m] = {"checkpoint": None, "model_seed": None, "out": "B_PLAN.jsonl"}
            continue
        if m == "QMARK":
            if qplan["status"] != "COMPLETE":
                problems.append("QMARK seed %d is %s" % (a.seed, qplan["status"]))
                continue
            run_dir = Path(qplan["run_dir"])
            job = json.loads((run_dir / "job_summary.json").read_text())
            rows = list(csv.DictReader((run_dir / "eval_metrics.csv").open()))
            final = [r for r in rows if "final_" in r["checkpoint"]]
            final = final[-1] if final else rows[-1]
            qualified = int(final["success_n"]) == 12 and job.get("NaN_n") == 0 and not job.get("hard_fail")
            if not qualified:
                problems.append("QMARK seed %d is CONTROL_NOT_ID_QUALIFIED (final dev12 %s/12, NaN %s, hard_fail %s)" % (a.seed, final["success_n"], job.get("NaN_n"), job.get("hard_fail")))
                continue
            man = json.loads((run_dir / "checkpoint_manifest.json").read_text())
            name = [k for k in man["checkpoint_files"] if k.startswith("final_n_") and k.endswith(".pt")][0]
            ck = run_dir / "checkpoints" / name
            if sha(ck) != man["checkpoint_files"][name]:
                problems.append("QMARK checkpoint hash mismatch")
                continue
            plan[m] = {"checkpoint": str(ck), "model_seed": a.seed, "out": "QMARK_s%d.jsonl" % a.seed}
            continue
        rec = inv["%s-%d" % (m, a.seed)]
        ck = root / rec["run_dir"] / "checkpoints" / rec["final_checkpoint"]
        if sha(ck) != rec["sha256_recorded"]:
            problems.append("%s checkpoint hash mismatch" % m)
            continue
        plan[m] = {"checkpoint": str(ck), "model_seed": a.seed, "out": "%s_s%d.jsonl" % (m, a.seed)}
    fresh = root / FRESH
    fresh.mkdir(parents=True, exist_ok=True)
    gpus = gpu_free()
    launches = {}
    for m, spec in plan.items():
        out = fresh / spec["out"]
        if out.exists() and sum(1 for _ in out.open()) >= 32:
            problems.append("%s already evaluated (%s)" % (m, out.name))
            continue
        gpu = gpus[len(launches) % len(gpus)]
        cmd = [sys.executable, str(root / "scripts/c1_eval.py"), "--root", str(root), "--gpu", str(gpu), "--method", METHOD_NAME[m], "--split-file", TEST_REL, "--rows-key", "test", "--out", str(out),
               "--label", "fresh_confirm", "--formal", "--prep-commit", a.prep_commit]
        if spec["checkpoint"]:
            cmd += ["--checkpoint", spec["checkpoint"]]
        if spec["model_seed"] is not None:
            cmd += ["--model-seed", str(spec["model_seed"])]
        launches[m] = {"gpu": gpu, "cmd": cmd, "log": str(fresh / ("eval_%s.log" % out.stem))}
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": a.seed, "launches": launches, "problems": problems, "dry_run": a.dry_run}
    (fresh / ("launch_seed%d.json" % a.seed)).write_text(json.dumps(record, indent=2) + "\n")
    if a.dry_run:
        print(json.dumps(record, indent=1))
        return
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "src")}
    for m, spec in launches.items():
        subprocess.Popen(spec["cmd"], stdout=open(spec["log"], "ab"), stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=str(root))
    print(json.dumps({"launched": {m: s["gpu"] for m, s in launches.items()}, "problems": problems}, indent=1))


if __name__ == "__main__":
    main()
