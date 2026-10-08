#!/usr/bin/env python
"""C1-BW-GOAL-LOOKAHEAD-INTEGRATION-V1 (plan of 2026-10-08, sections 5 and 6).

  zero-training 3x2 completion: GOAL_DENSE / GOAL_REL selected checkpoints of the method-serial v3 card x fixed two-step look-ahead (DENSE_LOOK2, REL_LOOK2) on the Confirm128 problems (now development material),
  plus ONE training of the sparsity-matched random-mask control (GOAL_RAND) with its one-step and look-ahead evaluations.

    init     --gpus 0,1,...  --authorization-text-file F
    run-all  --run-root R            parallel scheduler (one task per GPU, dependency-respecting, resumable)
    fixtures | look_eval --condition X | report            stage handlers (train / devsel of GOAL_RAND are delegated to scripts/c1_method_serial.py)
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CARD = "C1-BW-GOAL-LOOKAHEAD-INTEGRATION-V1"
BASE_COMMIT = "225cbae0c160b9fa4361a60a6b475f586f3f90a4"
BRANCH = "codex/cp-disr-c1-bw-goal-lookahead-integration-v1"
PLAN_REL = "docs/c1_blocksworld/CP_DISR_C1_Five_Methods_Closeout_and_Integration_Plan_20261008.md"
V3_REL = "runs/final_master/c1_route_b/blocksworld_main_v1/method_serial_v3/20261007T165611Z_a994451e"
SUITE_REL = "runs/final_master/c1_route_b/blocksworld_main_v1/goal_lookahead_integration_v1"
LOOK = {"DENSE_LOOK2": ("v3", "GOAL_DENSE", True), "REL_LOOK2": ("v3", "GOAL_REL", True), "RAND_C3": ("new", "GOAL_RAND", False), "RAND_LOOK2": ("new", "GOAL_RAND", True)}
DONE = ("DONE", "REUSED_EXACT", "MATHEMATICALLY_EQUIVALENT_REUSED")
CORE = ("src/cp_disr/blocksworld/method_serial/model.py", "src/cp_disr/blocksworld/method_serial/common.py", "src/cp_disr/blocksworld/method_serial/lookahead.py", "src/cp_disr/blocksworld/method_serial/trainer.py",
        "src/cp_disr/blocksworld/method_serial/evalkit.py", "scripts/c1_method_serial.py", "scripts/c1_goal_lookahead.py")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def wj(p, doc):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def wcsv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        if rows:
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fp_of(task):
    return hashlib.sha256(json.dumps({k: task.get(k) for k in ("id", "argv", "input_hashes")}, sort_keys=True).encode()).hexdigest()


def D():
    from types import SimpleNamespace
    from cp_disr.blocksworld import eval_a03 as E, planner as P, scale_order as SO, scorer_control as SC, state as S, train as T
    from cp_disr.blocksworld.method_serial import common as C, evalkit as EK, lookahead as LA, model as MD
    from cp_disr.rl import set_suite_half_life
    return SimpleNamespace(E=E, P=P, SO=SO, SC=SC, S=S, T=T, C=C, EK=EK, LA=LA, MD=MD, set_suite_half_life=set_suite_half_life)


# ------------------------------------------------------------------------------------------------ init
def cmd_init(gpus, auth):
    d = D()
    if d.E.git(ROOT, "branch", "--show-current") != BRANCH or d.E.git(ROOT, "rev-parse", "HEAD") != BASE_COMMIT:
        raise SystemExit("branch %s at base commit %s required (new files may be untracked)" % (BRANCH, BASE_COMMIT))
    storage = d.C.check_storage("/home/xushijie3")
    repo = d.C.check_storage(ROOT)
    hashes = {s: d.C.sha_file(ROOT / s) for s in CORE}
    code8 = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:8]
    rr = repo / SUITE_REL / ("%s_%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), code8))
    for sub in ("plan", "assets", "prep", "runs", "development", "confirmation", "results", "receipts", "driver_logs"):
        (rr / sub).mkdir(parents=True, exist_ok=True)
    (rr / "receipts" / "case_ledger.jsonl").write_text("")
    py = str(storage / "envs" / "cpdisr" / "bin" / "python")
    cfg = {"storage_root": str(storage), "repo_root": str(repo), "asset_root": str(repo), "run_root": str(rr), "cache_root": str(storage / "cp_disr_cache"), "tmp_root": str(storage / "cp_disr_tmp"), "venv_python": py,
           "device": "cuda:0", "gpus": gpus, "base_commit": BASE_COMMIT, "branch": BRANCH, "head": d.E.git(ROOT, "rev-parse", "HEAD"), "v3_run_root": str(repo / V3_REL), "storage_owner": "xushijie3", "execution_authorized": True}
    wj(rr / "suite_runtime.json", cfg)
    ids = {}
    v3 = repo / V3_REL
    for name, path in (("v3_GOAL_DENSE_selected", v3 / "runs/GOAL_DENSE/checkpoint_selection.json"), ("v3_GOAL_REL_selected", v3 / "runs/GOAL_REL/checkpoint_selection.json"), ("v3_confirm_manifest", v3 / "confirmation/manifest.json"),
                       ("v3_confirm_split", repo / "configs/splits/c1_bw_serial_confirm128_v1.json"), ("M1_GOAL", repo / d.SC.M1GOAL["path"]), ("D_train", repo / d.SC.A.GP_ROOT / "datasets/D_train.json")):
        ids[name] = {"path": str(path.relative_to(repo)), "sha256": sha_file(path)}
    for c in ("GOAL_DENSE", "GOAL_REL"):
        sel = json.loads((v3 / "runs" / c / "checkpoint_selection.json").read_text())
        ck = repo / sel["selected_checkpoint"]["path"]
        assert sha_file(ck) == sel["selected_checkpoint"]["sha256"]
        ids["weights_" + c] = {"epoch": sel["selected_epoch"], **sel["selected_checkpoint"]}
    wj(rr / "assets" / "asset_identity.json", ids)
    (rr / "plan" / "PLAN.md").write_text((ROOT / PLAN_REL).read_text(encoding="utf-8"), encoding="utf-8")
    S_ = str(ROOT / "scripts" / "c1_goal_lookahead.py")
    V3S = str(ROOT / "scripts" / "c1_method_serial.py")

    def task(tid, argv, requires=(), gpu=True, group="eval", training=None):
        t = {"id": tid, "argv": [py] + argv, "receipt": str(rr / "receipts" / ("%s.json" % tid)), "input_hashes": hashes, "requires": list(requires), "gpu": gpu, "group": group}
        if training:
            t["training_run_id"] = training
        return t
    tasks = [task("fixtures", [S_, "fixtures", "--run-root", str(rr)], gpu=True, group="fixtures"),
             task("train_GOAL_RAND", [V3S, "train", "--run-root", str(rr), "--condition", "GOAL_RAND"], ["fixtures"], True, "train", "R-C1-GLI-GOAL-RAND-0"),
             task("devsel_GOAL_RAND", [V3S, "devsel", "--run-root", str(rr), "--condition", "GOAL_RAND"], ["train_GOAL_RAND"], True, "train")]
    for name in ("DENSE_LOOK2", "REL_LOOK2"):
        tasks.append(task("look_" + name, [S_, "look_eval", "--run-root", str(rr), "--condition", name], ["fixtures"]))
    for name in ("RAND_C3", "RAND_LOOK2"):
        tasks.append(task("look_" + name, [S_, "look_eval", "--run-root", str(rr), "--condition", name], ["devsel_GOAL_RAND"]))
    env = {"PYTHONPATH": "src", "TMPDIR": cfg["tmp_root"], "XDG_CACHE_HOME": cfg["cache_root"], "TORCH_HOME": str(Path(cfg["cache_root"]) / "torch"), "PYTHONUNBUFFERED": "1"}
    wj(rr / "manifest.json", {"card": CARD, "execution_authorized": True, "authorization_source": auth, "gpus": gpus, "env": env, "tasks": tasks, "max_new_training_runs": 1})
    wj(rr / "plan" / "registration.json", {"card": CARD, "plan": PLAN_REL, "plan_sha256": sha_file(ROOT / PLAN_REL), "base_commit": BASE_COMMIT, "authorization_text": auth, "authorization_text_sha256": hashlib.sha256(auth.encode()).hexdigest(),
                                            "scope": "section 5 (zero-training DENSE_LOOK2 and REL_LOOK2 on Confirm128, 256 development episodes) and section 6 (at most one training: sparsity-matched random-mask control GOAL_RAND)",
                                            "control_definition": "same module/parameters/pair features/initialisation/budget as GOAL_REL and GOAL_DENSE; per query row keep self + k randomly ranked other goals, k = non-self targets of the REL mask in that row; "
                                                                  "ranking = fixed integer hash of (scored-state proposition truth vector, row, column), identical on every call (cached-by-construction, state-only, CPU = GPU); "
                                                                  "not invariant to object renaming (block names b0.. fixed in all splits); no true relation, label or history enters; same selection rule and 5 checkpoints as the other conditions",
                                            "extra_evaluations_of_the_control": ["GOAL_RAND one-step+C3 on Confirm128", "GOAL_RAND + fixed two-step on Confirm128 (execution-matched control for REL_LOOK2)"],
                                            "reused_rows": ["MG_C3", "LOOK2_MG", "GOAL_DENSE", "GOAL_REL", "B_G1C3 (method-serial v3, same code/problems/weights)"], "core_source_hashes": hashes,
                                            "role_of_confirm128": "development material from this card on; no independent confirmation claim", "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"run_root": str(rr), "tasks": len(tasks)}))


# ------------------------------------------------------------------------------------------------ scheduler (parallel GPUs)
def run_all(rr):
    rr = Path(rr)
    man = json.loads((rr / "manifest.json").read_text())
    tasks = {t["id"]: t for t in man["tasks"]}
    gpus = list(man["gpus"])
    env0 = {**os.environ, **man["env"]}
    state = {}
    attempts = Counter()

    def receipt(tid):
        p = Path(tasks[tid]["receipt"])
        try:
            r = json.loads(p.read_text()) if p.is_file() else None
        except json.JSONDecodeError:
            r = None
        return r if r and r.get("status") in DONE and r.get("input_fingerprint") == fp_of(tasks[tid]) else None
    running, free = {}, list(gpus)
    pending = [t for t in tasks if not receipt(t)]
    failed = set()
    while pending or running:
        for tid, (proc, g) in list(running.items()):
            if proc.poll() is not None:
                free.append(g)
                del running[tid]
                if not receipt(tid):
                    attempts[tid] += 1
                    if attempts[tid] < (3 if tasks[tid].get("training_run_id") else 1):
                        pending.append(tid)
                    else:
                        failed.add(tid)
        for tid in list(pending):
            if tid in failed:
                pending.remove(tid)
                continue
            req = tasks[tid].get("requires", [])
            if any(r in failed for r in req):
                failed.add(tid)
                pending.remove(tid)
                continue
            if free and all(receipt(r) for r in req):
                g = free.pop(0)
                pending.remove(tid)
                log = open(rr / "driver_logs" / ("%s.log" % tid), "ab")
                running[tid] = (subprocess.Popen(tasks[tid]["argv"], cwd=str(ROOT), env={**env0, "CUDA_VISIBLE_DEVICES": str(g), "TASK_FINGERPRINT": fp_of(tasks[tid])}, stdout=log, stderr=subprocess.STDOUT), g)
        time.sleep(10)
    wj(rr / "scheduler_receipt.json", {"done": sorted(t for t in tasks if receipt(t)), "failed": sorted(failed), "attempts": dict(attempts), "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 1 if failed else 0


def finish(rr, tid, status, outputs=(), **extra):
    d = D()
    outs = [{"path": str(Path(o).resolve()), "sha256": sha_file(o)} for o in outputs]
    wj(Path(rr) / "receipts" / ("%s.json" % tid), {"task": tid, "status": status, "input_fingerprint": os.environ.get("TASK_FINGERPRINT"), "outputs": outs, **extra})


# ------------------------------------------------------------------------------------------------ stages
def cmd_fixtures(rr):
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_goal_lookahead.py", "tests/test_method_serial.py", "tests/test_method_serial_math.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=str(ROOT),
                       env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
    log = Path(rr) / "prep" / "fixtures.log"
    log.write_text(r.stdout + r.stderr)
    finish(rr, "fixtures", "DONE" if r.returncode == 0 else "GLOBAL_INTEGRITY_FAILURE", [log], pytest_tail=r.stdout[-300:])
    return r.returncode


def cmd_look_eval(rr, name):
    d = D()
    which, cond, look = LOOK[name]
    td = json.loads((ROOT / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    d.set_suite_half_life(float(td["half_life"]))
    v3rr = ROOT / V3_REL
    c = d.C.Ctx(ROOT, v3rr if which == "v3" else rr)
    model, sel = c.load_selected(cond)
    man = json.loads((v3rr / "confirmation" / "manifest.json").read_text())
    assert d.C.sha_file(ROOT / man["split"]) == man["sha256"], "Confirm128 split changed"
    groups, meta = d.EK.load_cases(ROOT, man["split"])
    gl = [(k, groups[k]) for k in sorted(groups)]
    cnt = d.LA.Counters() if look else None
    chooser = d.LA.make_chooser(model, "MG", cnt) if look else d.EK.c3_chooser()
    path = Path(rr) / "confirmation" / ("by_case_%s.jsonl" % name)
    d.EK.run_set(Path(rr), "%s@confirm" % name, gl, d.MD.SerialPolicy(model), chooser, path)
    extra = {"selected_epoch": sel["selected_epoch"], "checkpoint_sha256": sel["selected_checkpoint"]["sha256"]}
    if cnt is not None:
        extra["counters"] = cnt.as_dict()
    from cp_disr.blocksworld import imitation as I
    vals = [model.mask_density(I.snapshot_at(cs, cs.problem.init, 0)) for g in groups.values() for cs in g]
    extra["mask_density_nonself_initial_states_confirm"] = round(sum(vals) / len(vals), 4)
    finish(rr, "look_" + name, "DONE", [path], **extra)
    return 0


def cmd_report(rr):
    d = D()
    rr = Path(rr)
    v3 = ROOT / V3_REL
    man = json.loads((v3 / "confirmation" / "manifest.json").read_text())
    groups, meta = d.EK.load_cases(ROOT, man["split"])
    names = {"MG_C3": v3, "LOOK2_MG": v3, "GOAL_DENSE": v3, "GOAL_REL": v3, "B_G1C3": v3, "DENSE_LOOK2": rr, "REL_LOOK2": rr, "RAND_C3": rr, "RAND_LOOK2": rr}
    ep = {}
    for n, root in names.items():
        p = root / "confirmation" / ("by_case_%s.jsonl" % n)
        if p.is_file():
            ep[n] = {e["case_id"]: e for e in d.EK.jl(p)}
    layers = {"ALL": lambda m: True, "h2_seen": lambda m: m["max_tower_height"] == 2, "h4_unseen": lambda m: m["max_tower_height"] == 4, "k1": lambda m: m["k_nontrivial_towers"] == 1, "k2": lambda m: m["k_nontrivial_towers"] == 2,
              "n6": lambda m: m["n_blocks"] == 6, "n8": lambda m: m["n_blocks"] == 8, "BREAK": lambda m: m["labels"]["requires_goal_destruction"] is True, "MONO": lambda m: m["labels"]["requires_goal_destruction"] is False}
    for cell in sorted(groups):
        layers["cell=" + cell] = (lambda cell: lambda m: m["cell"] == cell)(cell)
    res = rr / "results"
    rows = []
    for n, eps in ep.items():
        for ln, pr in layers.items():
            es = [e for cid, e in eps.items() if pr(meta[cid])]
            rows.append({"condition": n, "layer": ln, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.EK.summarize(es).items()}})
    wcsv(res / "by_condition_and_layer.csv", rows)
    mat = [{"scorer": s, "one_step_C3": "%d / %d" % ((lambda x: (x["success"], x["perfect"]))(d.EK.summarize(ep[o].values()))) if o in ep else "NA",
            "fixed_two_step": "%d / %d" % ((lambda x: (x["success"], x["perfect"]))(d.EK.summarize(ep[t].values()))) if t in ep else "NA", "one_step_row": o, "two_step_row": t}
           for s, o, t in (("MG", "MG_C3", "LOOK2_MG"), ("GOAL_DENSE", "GOAL_DENSE", "DENSE_LOOK2"), ("GOAL_REL", "GOAL_REL", "REL_LOOK2"), ("GOAL_RAND (sparsity-matched control)", "RAND_C3", "RAND_LOOK2"))]
    wcsv(res / "matrix_3x2.csv", mat)
    pairs = (("REL_LOOK2", "LOOK2_MG"), ("REL_LOOK2", "DENSE_LOOK2"), ("REL_LOOK2", "GOAL_REL"), ("DENSE_LOOK2", "LOOK2_MG"), ("DENSE_LOOK2", "GOAL_DENSE"), ("GOAL_REL", "GOAL_DENSE"), ("GOAL_REL", "MG_C3"), ("REL_LOOK2", "MG_C3"),
             ("RAND_C3", "GOAL_REL"), ("RAND_C3", "GOAL_DENSE"), ("RAND_C3", "MG_C3"), ("RAND_LOOK2", "REL_LOOK2"), ("RAND_LOOK2", "DENSE_LOOK2"), ("RAND_LOOK2", "LOOK2_MG"), ("RAND_LOOK2", "RAND_C3"))
    prs = []
    for a, b in pairs:
        if a in ep and b in ep:
            for ln, pr in layers.items():
                ids = {cid for cid in ep[a] if pr(meta[cid])}
                common = [i for i in ids if ep[a][i]["success"] and ep[b][i]["success"]]
                diff = [ep[a][i]["steps"] - ep[b][i]["steps"] for i in common]
                for field, nm in (("success", "S"), ("decision_perfect", "D")):
                    pc = d.EK.pair_counts({i: ep[a][i] for i in ids}, {i: ep[b][i] for i in ids}, field)
                    prs.append({"first": a, "second": b, "layer": ln, "metric": nm, **pc, "common_success_n": len(common), "common_mean_steps_first_minus_second": round(statistics.mean(diff), 3) if diff else None,
                                "failure_penalised_ratio_first": round(statistics.mean(d.EK.penalty_ratio(ep[a][i]) for i in ids), 4), "failure_penalised_ratio_second": round(statistics.mean(d.EK.penalty_ratio(ep[b][i]) for i in ids), 4)})
    wcsv(res / "paired_comparisons.csv", prs)
    costs = []
    for n in ("DENSE_LOOK2", "REL_LOOK2", "RAND_C3", "RAND_LOOK2"):
        p = rr / "receipts" / ("look_%s.json" % n)
        if p.is_file():
            r = json.loads(p.read_text())
            costs.append({"condition": n, **{k: v for k, v in (r.get("counters") or {}).items()}, "mask_density_confirm_initial_states": r.get("mask_density_nonself_initial_states_confirm"), "selected_epoch": r.get("selected_epoch"),
                          "episode_wall_seconds": round(sum(e["wall_seconds"] for e in ep[n].values()), 2)})
    for n in ("MG_C3", "LOOK2_MG", "GOAL_DENSE", "GOAL_REL"):
        costs.append({"condition": n + " (v3 reuse)", "episode_wall_seconds": round(sum(e["wall_seconds"] for e in ep[n].values()), 2)})
    p = rr / "runs" / "GOAL_RAND" / "training_accounting.json"
    if p.is_file():
        a = json.loads(p.read_text())
        costs.append({"condition": "TRAIN_GOAL_RAND", **{k: a[k] for k in ("optimizer_steps", "decision_samples_shown", "wall_seconds", "parameters_new_module", "new_module_param_change_l2", "nan_events", "attempts")}})
    wcsv(res / "compute_costs.csv", costs)
    lines = ["# C1-BW-GOAL-LOOKAHEAD-INTEGRATION-V1 (auto-generated numbers; Confirm128 is development material here)", "", "success / all-steps-optimal (n)", "", "| condition | ALL | h2 | h4 | must-destroy |", "|---|---|---|---|---|"]
    for n, eps in ep.items():
        cells = []
        for ln in ("ALL", "h2_seen", "h4_unseen", "BREAK"):
            s = d.EK.summarize([e for cid, e in eps.items() if layers[ln](meta[cid])])
            cells.append("%d / %d (%d)" % (s["success"], s["perfect"], s["n"]))
        lines.append("| %s | %s |" % (n, " | ".join(cells)))
    (res / "final_summary_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finish(rr, "report", "DONE", [res / "by_condition_and_layer.csv", res / "matrix_3x2.csv"])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run-root")
    ap.add_argument("--condition")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--authorization-text-file")
    a = ap.parse_args()
    if a.cmd == "init":
        return cmd_init([int(x) for x in a.gpus.split(",")], Path(a.authorization_text_file).read_text(encoding="utf-8"))
    rr = a.run_root
    return {"run-all": lambda: run_all(rr), "fixtures": lambda: cmd_fixtures(rr), "look_eval": lambda: cmd_look_eval(rr, a.condition), "report": lambda: cmd_report(rr)}[a.cmd]() or 0


if __name__ == "__main__":
    sys.exit(main())
