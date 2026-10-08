#!/usr/bin/env python
"""C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3 (plan CP_DISR_C1_Depots_FAST_ALT_Runbook_v3_20261009.md).

    init                     create the run root (authorisation text, plan and config copies)
    prepare                  frozen identities, IPC evidence review, workload counts (no search)
    profile                  bounded input library, split timings, micro-benchmarks, call-cost description (GPU)
    verify-fast              FAST equivalence: tensors, values against the reference's own repeat noise, search prefixes (GPU)
    fixtures                 pytest of the ALT / engine fixtures
    freeze                   decide the FAST status, write the frozen record, registration and DAG manifest
    run-all                  parallel scheduler of the main searches (452 at most)
    search / report / final_receipt   stage handlers
No training, no external fit, no label, no new problem.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CARD = "C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3"
BASE_COMMIT = "57f011656ea2ccb6915869eeead265d6c98514f2"
BRANCH = "codex/cp-disr-c1-depots-fast-alt-v3"
PLAN_REL = "docs/c1_blocksworld/CP_DISR_C1_Depots_FAST_ALT_Runbook_v3_20261009.md"
CONFIG_REL = "configs/c1_depots_fast_alt_v3.yaml"
PKG = "src/cp_disr/pddl/fast_alt/"
CORE = tuple(PKG + n for n in ("__init__.py", "fast_value.py", "alt_engine.py", "evaluators.py", "profiling.py", "verify.py", "runner.py")) + (
    "src/cp_disr/pddl/search_match/engine.py", "src/cp_disr/pddl/search_match/evaluators.py", "src/cp_disr/pddl/search_match/budget.py", "src/cp_disr/pddl/search_match/runner.py",
    "scripts/c1_depots_fast_alt.py", CONFIG_REL, "tests/test_depots_fast_alt.py")


def rj(p):
    return json.loads(Path(p).read_text())


def wj(p, doc):
    from cp_disr.pddl.sched import wj as _wj
    _wj(p, doc)


def sha_file(p):
    from cp_disr.pddl.sched import sha_file as s
    return s(p)


def finish(rr, tid, status, outs=(), **extra):
    from cp_disr.pddl.sched import finish as f
    f(rr, tid, status, outs, **extra)


def load_cfg():
    import yaml
    return yaml.safe_load((ROOT / CONFIG_REL).read_text())


def search_cfg(cfg):
    s = cfg["search"]
    return {"wall_seconds": float(s["wall_seconds"]), "max_expansions": int(s["max_expansions"]), "max_rss_growth_bytes": int(s["max_rss_growth_gib"] * 2 ** 30), "grace_seconds": float(s["grace_seconds"])}


def old_run(cfg):
    from cp_disr.pddl.search_match.runner import OldRun
    return OldRun(ROOT / cfg["old_asset_root"])


def runtime(rr):
    return rj(Path(rr) / "suite_runtime.json")


def wcsv(p, rows):
    from cp_disr.pddl.search_match.analysis import wcsv as w
    return w(p, rows)


# ------------------------------------------------------------------------------------------------ init
def cmd_init(auth):
    cfg = load_cfg()
    git = lambda *a: subprocess.check_output(["git", "-C", str(ROOT)] + list(a), text=True).strip()
    if git("branch", "--show-current") != BRANCH or subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"]).returncode != 0:
        raise SystemExit("branch %s descending from %s required" % (BRANCH, BASE_COMMIT))
    for k in ("old_search_root", "old_asset_root"):
        if not (ROOT / cfg[k]).is_dir():
            raise SystemExit("missing %s" % cfg[k])
    parent = Path(os.environ["FA_RUN_PARENT"]) if os.environ.get("FA_RUN_PARENT") else ROOT / cfg["suite_rel"]
    rr = parent / ("%s_fastalt" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    for sub in ("plan", "assets", "prep", "profile", "checks", "runs", "plans", "results", "receipts", "driver_logs"):
        (rr / sub).mkdir(parents=True, exist_ok=True)
    (rr / "plan" / "PLAN.md").write_text((ROOT / PLAN_REL).read_text(encoding="utf-8"), encoding="utf-8")
    (rr / "plan" / "config.yaml").write_text((ROOT / CONFIG_REL).read_text(encoding="utf-8"), encoding="utf-8")
    storage = Path("/home/xushijie3")
    wj(rr / "suite_runtime.json", {"repo_root": str(ROOT), "run_root": str(rr), "venv_python": str(storage / "envs" / "cpdisr" / "bin" / "python"), "base_commit": BASE_COMMIT, "branch": BRANCH, "head": git("rev-parse", "HEAD"),
                                    "storage_owner": "xushijie3", "old_search_root": cfg["old_search_root"], "old_asset_root": cfg["old_asset_root"], "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    wj(rr / "plan" / "authorisation.json", {"card": CARD, "authorization_text": auth, "authorization_text_sha256": hashlib.sha256(auth.encode()).hexdigest(), "plan_sha256": sha_file(ROOT / PLAN_REL)})
    print(json.dumps({"run_root": str(rr)}))


# ------------------------------------------------------------------------------------------------ prepare
def cmd_prepare(rr):
    import csv
    from collections import defaultdict
    from cp_disr.pddl.search_match import evaluators as EV
    from cp_disr.pddl.search_match.runner import NEURAL, sha_file as sf
    cfg = load_cfg()
    old = old_run(cfg)
    ck = old.checkpoint("dense")
    assert sf(ck["path"]) == ck["sha256"], "DENSE checkpoint changed"
    ident = {"dense": {"update": ck["update"], "checkpoint": str(ck["path"]), "sha256": ck["sha256"], "init_seed": ck["init_seed"]}, "wl": {}, "h_add": {"binary": str(EV.HADD_BIN), "sha256": sf(EV.HADD_BIN)}}
    for s in ("struct", "ipc"):
        w = old.wl_params(s)
        ident["wl"][w["track"]] = {"model_sha256": sf(w["model"]), "params_sha256": sf(w["params"]), "opts_sha256": sf(w["opts"])}
    old_ident = rj(ROOT / cfg["old_search_root"] / "assets" / "evaluator_identity.json")
    assert old_ident["neural"]["V_DENSE"]["sha256_now"] == ck["sha256"]
    for tr in ("typed", "ipc"):
        assert old_ident["wl"][tr]["model_sha256"] == ident["wl"][tr]["model_sha256"]
    assert old_ident["heuristics"]["H_ADD"]["sha256"] == ident["h_add"]["sha256"], "h_add binary changed"
    bad = [c["case_id"] for s in ("struct", "joint", "ipc") for c in old.cases(s) if sf(c["file"]) != c["sha256"]]
    assert not bad, bad[:5]
    ident["problems"] = {s: len(old.cases(s)) for s in ("struct", "joint", "ipc")}
    ident["software"] = old_ident["software"]
    ident["previous_card_receipt_sha256"] = sf(ROOT / cfg["old_search_root"] / "receipts" / "final_receipt.json")
    ident["previous_card_commit"] = BASE_COMMIT
    wj(Path(rr) / "assets" / "frozen_identity.json", ident)
    # ---- IPC evidence review (reread of stored rows; no search)
    rows = list(csv.DictReader(open(ROOT / cfg["old_search_root"] / "results" / "search_by_case.csv", encoding="utf-8")))
    by = defaultdict(dict)
    for r in rows:
        if r["set"] == "ipc":
            by[r["case_id"]][r["scorer"]] = r
    out = []
    for cid in sorted(by):
        d, w, a = by[cid]["V_DENSE"], by[cid]["H_WL"], by[cid]["H_ADD"]
        g = lambda r, k: (float(r[k]) if r.get(k) not in (None, "") else None)
        rec = {"case_id": cid, "n_ground_actions": d["n_ground_actions"], "dense_status": d["status"], "dense_expanded": d["expanded"], "dense_solved_at_expansion": d["solved_at_expansion"], "dense_wall_s": d["wall_total_s"],
               "dense_logical_batches": d["eval_batches"], "dense_outer_physical_calls": d["forward_calls"], "dense_states_scored": d["evaluated_states"],
               "dense_mean_batch": round(g(d, "evaluated_states") / max(1, g(d, "eval_batches")), 2) if d["eval_batches"] else None,
               "dense_physical_calls_per_logical_batch": round(g(d, "forward_calls") / max(1, g(d, "eval_batches")), 3) if d["eval_batches"] else None,
               "dense_ms_per_expansion": round(1000 * g(d, "wall_total_s") / max(1, g(d, "expanded")), 2), "wl_status": w["status"], "wl_expanded": w["expanded"], "wl_solved_at_expansion": w["solved_at_expansion"], "wl_wall_s": w["wall_total_s"],
               "hadd_status": a["status"], "hadd_expanded": a["expanded"], "hadd_solved_at_expansion": a["solved_at_expansion"], "hadd_wall_s": a["wall_total_s"]}
        if d["status"] != "SOLVED":
            rec["dense_stopped_by"] = d["status"]
            wl_need = w["solved_at_expansion"]
            rec["wl_needed_expansions"] = wl_need or None
            rec["dense_to_wl_expansion_ratio"] = round(float(d["expanded"]) / float(wl_need), 2) if wl_need not in (None, "") and w["status"] == "SOLVED" else None
            rec["relative_guidance_evidence"] = ("DENSE expanded more nodes than WL needed to solve" if rec["dense_to_wl_expansion_ratio"] and rec["dense_to_wl_expansion_ratio"] > 1 else
                                                 ("WL unsolved within its node budget" if w["status"] != "SOLVED" else "none"))
            rec["hadd_alternative"] = ("h_add solved at %s expansions" % a["solved_at_expansion"]) if a["status"] == "SOLVED" else "h_add unsolved (%s at %s expansions)" % (a["status"], a["expanded"])
            rec["note"] = "resource stop and relative guidance disadvantage can coexist; 'unsolved' is not an unsolvability proof"
        out.append(rec)
    wcsv(Path(rr) / "prep" / "ipc_evidence_review.csv", out)
    wl = [("REF_DENSE", "ipc", 22), ("FAST_DENSE", "ipc", 22), ("ALT_DENSE_ADD", "struct", 32), ("ALT_DENSE_ADD", "joint", 128), ("ALT_DENSE_ADD", "ipc", 22), ("ALT_WL_ADD", "struct", 32), ("ALT_WL_ADD", "joint", 128),
          ("ALT_WL_ADD", "ipc", 22), ("EAGER_WL", "ipc", 22), ("EAGER_HADD", "ipc", 22)]
    wcsv(Path(rr) / "prep" / "workload_counts.csv", [{"condition": c, "set": s, "planned_runs": n} for c, s, n in wl] + [{"condition": "TOTAL_MAIN_SEARCHES_MAX", "set": "all", "planned_runs": sum(n for _, _, n in wl)}])
    finish(rr, "prepare", "DONE", [Path(rr) / "assets" / "frozen_identity.json", Path(rr) / "prep" / "ipc_evidence_review.csv"], planned=sum(n for _, _, n in wl))
    return 0


# ------------------------------------------------------------------------------------------------ profile / verify
def dense_model(device):
    from cp_disr.pddl.search_match.runner import load_neural
    cfg = load_cfg()
    old = old_run(cfg)
    m, ck = load_neural(old, "V_DENSE", device)
    return old, cfg, m, ck


def cmd_profile(rr):
    import torch
    from cp_disr.pddl.fast_alt import profiling as PF
    from cp_disr.pddl.fast_alt.evaluators import RefNeuralEval
    dev = torch.device("cuda", 0)
    old, cfg, model, ck = dense_model(dev)
    t0 = time.time()
    pick = PF.pick_templates(old)
    ref = RefNeuralEval("REF", model, dev)
    inputs = PF.collect_inputs(old, ref, pick, cfg["profile"]["collect_wall_seconds"], cfg["profile"]["max_expansions"], cfg["profile"]["max_states"])
    PF.save_inputs(inputs, Path(rr) / "profile" / "natural_batches.jsonl")
    lib = PF.load_state_library(Path(rr) / "profile" / "natural_batches.jsonl")
    rows = PF.microbench(old, model, dev, lib)
    PF.write_csv(rows, Path(rr) / "profile" / "timings.csv")
    wj(Path(rr) / "profile" / "call_cost_model.json", {"fit": PF.fit_cost_model(rows), "speedups": PF.speedups(rows), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                                       "templates": {k: {"case_id": v["case_id"], "edges": v["template_edges"], "nodes": v["template_nodes"], "goals": v["goals"], "natural_batches": len(v["batch_sizes"]), "distinct_states": v["n_distinct_states"],
                                                                         "collection_status": v["status"], "collection_wall_s": round(v["wall"], 2)} for k, v in lib.items()}})
    outs = [Path(rr) / "profile" / n for n in ("natural_batches.jsonl", "timings.csv", "call_cost_model.json")]
    finish(rr, "profile", "DONE", outs, seconds=round(time.time() - t0, 1), natural_state_collection_expansions=sum(v["expanded"] for v in inputs.values()), microbenchmark_rows=len(rows))
    return 0


def cmd_verify_fast(rr):
    import torch
    from cp_disr.pddl.fast_alt import profiling as PF, verify as VF
    dev = torch.device("cuda", 0)
    old, cfg, model, ck = dense_model(dev)
    lib = PF.load_state_library(Path(rr) / "profile" / "natural_batches.jsonl")
    tens = VF.check_tensors(old, model, dev, lib)
    vals, ok_vals = VF.check_values(old, model, dev, lib)
    rows, summ = VF.check_prefixes(old, model, dev, cfg["verify"]["prefix_wall_seconds"], cfg["verify"]["prefix_expansions"])
    tens_ok = all(v["codes_identical"] and v["edge_tensors_identical"] and v["static_embedding_identical"] for v in tens.values())
    status = "PASS" if (tens_ok and ok_vals and summ["pass"]) else "NOT_EQUIVALENT_OR_INCOMPLETE"
    wj(Path(rr) / "checks" / "fast_equivalence.json", {"fast_status": status, "tensors": tens, "tensors_all_identical": tens_ok, "values": vals, "values_pass_noise_rule": ok_vals,
                                                       "rule": {"noise": "max|FAST-REF| <= %.1f * max|REF-REF'| + %g per template and group" % (cfg["verify"]["noise_factor"], cfg["verify"]["noise_floor"]),
                                                                "plan_tolerance_reported": "abs <= %g + %g*|ref|" % (cfg["verify"]["plan_atol"], cfg["verify"]["plan_rtol"])}})
    wj(Path(rr) / "checks" / "search_prefixes.json", {"summary": summ, "problems": rows})
    finish(rr, "verify_fast", "DONE", [Path(rr) / "checks" / "fast_equivalence.json", Path(rr) / "checks" / "search_prefixes.json"], fast_status=status, prefix_summary=summ)
    return 0


def cmd_fixtures(rr):
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_depots_fast_alt.py", "tests/test_depots_search_match.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=str(ROOT), env=env, capture_output=True, text=True)
    p = Path(rr) / "checks" / "alt_fixtures.json"
    wj(p, {"returncode": r.returncode, "passed": r.returncode == 0, "tail": r.stdout[-600:]})
    finish(rr, "fixtures", "DONE" if r.returncode == 0 else "GLOBAL_INTEGRITY_FAILURE", [p], tail=r.stdout[-200:])
    return r.returncode


# ------------------------------------------------------------------------------------------------ freeze
def pick_gpus(n=7):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = sorted((int(m), int(u), int(i)) for i, m, u in (l.split(",") for l in out.strip().splitlines()))
    return [i for m, u, i in rows if m < 2000 and u < 30][:n]


def cmd_freeze(rr, gpus):
    cfg = load_cfg()
    rr = Path(rr)
    fe = rj(rr / "checks" / "fast_equivalence.json")
    fx = rj(rr / "checks" / "alt_fixtures.json")
    if not fx["passed"]:
        raise SystemExit("fixtures failed: no freeze")
    fast_status = fe["fast_status"]
    backend = "fast" if fast_status == "PASS" else "reference"
    hashes = {s: sha_file(ROOT / s) for s in CORE if (ROOT / s).is_file()}
    gpus = gpus or pick_gpus()
    storage = Path("/home/xushijie3")
    py = str(storage / "envs" / "cpdisr" / "bin" / "python")
    warm = next(c["file"] for c in old_run(cfg).manifest["train"] if c["case_id"] == "train_n3_000")
    env = {"PYTHONPATH": "src", "TMPDIR": str(storage / "cp_disr_tmp"), "XDG_CACHE_HOME": str(storage / "cp_disr_cache"), "TORCH_HOME": str(storage / "cp_disr_cache" / "torch"), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
           "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "CPDISR_EXT": str(Path.home() / "ext"), "CPDISR_WARMUP_PROBLEM": warm, "LD_LIBRARY_PATH": "/home/xushijie3/.uv_python/cpython-3.10.19-linux-x86_64-gnu/lib"}
    me = str(ROOT / "scripts" / "c1_depots_fast_alt.py")
    tasks = []

    def add(tid, args, resource):
        tasks.append({"id": tid, "argv": [py, me] + args + ["--run-root", str(rr), "--dense-backend", backend, "--task-id", tid], "receipt": str(rr / "receipts" / ("%s.json" % tid)), "input_hashes": hashes, "requires": [], "resource": resource})
    sh = cfg["shards"]
    ipc_neural = "REF_DENSE,FAST_DENSE,ALT_DENSE_ADD" if backend == "fast" else "REF_DENSE,ALT_DENSE_ADD"
    for k in range(sh["ipc_gpu_blocks"]):
        add("ipc_block_%d" % k, ["search", "--conds", ipc_neural, "--set", "ipc", "--shard", str(k), "--nshards", str(sh["ipc_gpu_blocks"]), "--block"], "gpu")
    for s, n in (("joint", sh["joint"]), ("struct", sh["struct"])):
        for k in range(n):
            add("alt_dense_%s_%d" % (s, k), ["search", "--conds", "ALT_DENSE_ADD", "--set", s, "--shard", str(k), "--nshards", str(n)], "gpu")
    for k in range(sh["ipc_cpu"]):
        add("cpu_ipc_%d" % k, ["search", "--conds", "EAGER_WL,EAGER_HADD,ALT_WL_ADD", "--set", "ipc", "--shard", str(k), "--nshards", str(sh["ipc_cpu"])], "cpu")
    for s, n in (("joint", sh["joint"]), ("struct", sh["struct"])):
        for k in range(n):
            add("alt_wl_%s_%d" % (s, k), ["search", "--conds", "ALT_WL_ADD", "--set", s, "--shard", str(k), "--nshards", str(n)], "cpu")
    planned = 22 * (3 if backend == "fast" else 2) + 22 * 3 + 2 * (32 + 128)
    manifest = {"card": CARD, "execution_authorized": True, "gpus": gpus, "env": env, "repo_root": str(ROOT), "tasks": tasks, "max_new_training_runs": 0, "cpu_slots": int(cfg["limits"]["cpu_slots"])}
    wj(rr / "manifest.json", manifest)
    wj(rr / "plan" / "frozen.json", {"fast_status": fast_status, "alt_dense_backend": backend, "gpus": gpus, "planned_main_searches": planned, "alt_rotation": cfg["alt"]["rotation"], "frozen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    auth = rj(rr / "plan" / "authorisation.json")
    reg = {"card": CARD, "plan": PLAN_REL, "plan_sha256": sha_file(ROOT / PLAN_REL), "config_sha256": sha_file(ROOT / CONFIG_REL), "base_commit": BASE_COMMIT, "authorization": auth, "new_training_runs": 0, "optimizer_steps": 0,
           "new_training_labels": 0, "new_problem_generation": 0, "new_external_fits": 0, "scope": "FAST (equivalent speed-up of the frozen DENSE-G value) + ALT (two alternated open lists, main score + h_add) on the existing 182 problems; <= 452 main searches",
           "fast_status": fast_status, "alt_dense_backend": backend, "frozen_search_protocol": cfg["search"], "alt_protocol": cfg["alt"], "verify_rules": cfg["verify"], "core_source_hashes": hashes,
           "preceding_checks": {"fast_equivalence": "checks/fast_equivalence.json", "search_prefixes": "checks/search_prefixes.json", "fixtures": "checks/alt_fixtures.json", "profile": "profile/"},
           "note": "FAST development and its checks used Train96 templates and (as the plan allows) IPC p15 / p21 / p22 and the fixed Struct / Joint prefix problems; no main-run outcome of this card had been seen when the rotation rule and the "
                   "equivalence rule were frozen", "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    wj(rr / "plan" / "registration.json", reg)
    print(json.dumps({"fast_status": fast_status, "backend": backend, "tasks": len(tasks), "planned_main_searches": planned, "gpus": gpus}))


# ------------------------------------------------------------------------------------------------ search / report
def cmd_search(rr, conds, set_name, shard, nshards, block, backend, task_id=None):
    import torch
    from cp_disr.pddl.fast_alt.runner import NEURAL_CONDS, run_shard
    cfg = load_cfg()
    old = old_run(cfg)
    conds = conds.split(",")
    device = torch.device("cuda", 0) if any(c in NEURAL_CONDS for c in conds) else None
    if device is not None and not torch.cuda.is_available():
        raise SystemExit("neural condition without a GPU")
    tid = task_id or "%s_%s_%d" % ("-".join(conds), set_name, shard)
    out = Path(rr) / "runs" / ("%s.jsonl" % tid)
    ident, load_s = run_shard(old, conds, set_name, shard, nshards, search_cfg(cfg), out, Path(rr) / "plans", device, backend, block)
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    from collections import Counter
    finish(rr, tid, "DONE", [out], cases=len(recs), solved=sum(r["solved"] for r in recs), statuses=dict(Counter(r["status"] for r in recs)), identity=ident, model_load_seconds=load_s)
    return 0


def runtime_tasks(rr):
    return rj(Path(rr) / "manifest.json")["tasks"]


def cmd_report(rr):
    from cp_disr.pddl.fast_alt import report as RP
    cfg = load_cfg()
    outs = RP.build(rr, old_run(cfg), cfg, ROOT)
    finish(rr, "report", "DONE", outs)
    return 0


def cmd_final_receipt(rr):
    rr = Path(rr)
    from collections import Counter
    recs = []
    for p in sorted((rr / "runs").glob("*.jsonl")):
        if p.name == "search_results.jsonl":                                    # merged copy written by `report`
            continue
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    st = Counter(r["status"] for r in recs)
    fr = rj(rr / "plan" / "frozen.json")
    prof = rr / "receipts" / "profile.json"
    pre = rj(rr / "checks" / "search_prefixes.json")
    nprob = len(pre["problems"])
    rows = sum(1 for _ in open(rr / "profile" / "timings.csv", encoding="utf-8")) - 1
    coll = sum(1 for _ in open(rr / "profile" / "natural_batches.jsonl", encoding="utf-8"))
    out = {"card": CARD, "base_commit": BASE_COMMIT, "result_commit": "THIS_COMMIT (a commit cannot contain its own hash; see the final report / git log of the branch)", "storage_owner": "xushijie3", "new_training_runs": 0,
           "new_external_fits": 0, "optimizer_steps": 0, "new_training_labels": 0,
           "new_problem_generation": 0, "primary_searches_planned_max": 452, "primary_searches_planned_after_freeze": fr["planned_main_searches"], "primary_searches_completed": len(recs),
           "bounded_probe_searches": {"state_collection_runs": coll, "state_collection_cap": "<= 32 expansions each", "prefix_problems": nprob, "prefix_runs": 3 * nprob, "prefix_runs_breakdown": "REF, REF repeat, FAST per problem (checks/search_prefixes.json)",
                                      "plan_literal_cap": 40, "over_literal_cap_by": 3 * nprob - 40, "note": "plan 7.3 item 6 requires a REF repeat reading; the 20 REF-repeat runs are counted separately and exceed the literal cap of 40 (REF + FAST only) by 20"},
           "microbenchmark_calls": {"timed_rows": rows, "file": "profile/timings.csv"}, "fast_status": fr["fast_status"],
           "alt_dense_backend": fr["alt_dense_backend"].upper(), "solved": sum(r["solved"] for r in recs), "status_counts": dict(st), "resource_limited": sum(st.get(k, 0) for k in ("TIMEOUT", "MEMORY_LIMIT", "NODE_LIMIT")),
           "technical_incomplete": sum(st.get(k, 0) for k in ("ADAPTER_ERROR", "INVALID_PLAN", "MODEL_NONFINITE")), "old_results_modified": False, "weights_modified": False, "no_pt_committed": True,
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "next_action": "WAIT_FOR_METHOD_DECISION"}
    p = rr / "receipts" / "final_receipt.json"
    wj(p, out)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run-root")
    ap.add_argument("--authorization-text-file")
    ap.add_argument("--conds")
    ap.add_argument("--set")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--block", action="store_true")
    ap.add_argument("--dense-backend", default="fast")
    ap.add_argument("--gpus")
    ap.add_argument("--task-id")
    a = ap.parse_args()
    if a.cmd == "init":
        return cmd_init(Path(a.authorization_text_file).read_text(encoding="utf-8"))
    rr = a.run_root
    if a.cmd == "run-all":
        from cp_disr.pddl.sched import run_all
        return run_all(rr, cpu_slots=rj(Path(rr) / "manifest.json").get("cpu_slots", 3))
    fn = {"prepare": lambda: cmd_prepare(rr), "profile": lambda: cmd_profile(rr), "verify-fast": lambda: cmd_verify_fast(rr), "fixtures": lambda: cmd_fixtures(rr),
          "freeze": lambda: cmd_freeze(rr, [int(x) for x in a.gpus.split(",")] if a.gpus else None), "search": lambda: cmd_search(rr, a.conds, a.set, a.shard, a.nshards, a.block, a.dense_backend, a.task_id),
          "report": lambda: cmd_report(rr), "final_receipt": lambda: cmd_final_receipt(rr)}[a.cmd]
    r = fn()
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
