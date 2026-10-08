#!/usr/bin/env python
"""C1-DEPOTS-SEARCH-MATCH-V1 (plan CP_DISR_C1_Depots_Search_Matched_Runbook_v2_20261008.md).

    init   --gpus 0,1,..  --authorization-text-file F     resolve assets, run root, evaluator identities, registration, DAG manifest
    run-all --run-root R                                    dependency-respecting parallel scheduler (resumable by receipts)
    assets | fixtures | topology | ref_sensitivity | search --scorer S --set X --shard K --nshards N | report | final_receipt     stage handlers

Frozen scorers V_DENSE / V_MG / V_REL (previous card, selected updates), H_WL (fitted WL-GOOSE model), H_ADD, H_COUNT are run in ONE shared search engine; no training, no new labels, no new problems.
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
CARD = "C1-DEPOTS-SEARCH-MATCH-V1"
BASE_COMMIT = "e10e80f56236cb7c43cd4113c8bc345d932f629f"
BRANCH = "codex/cp-disr-c1-depots-search-match-v1"
PLAN_REL = "docs/c1_blocksworld/CP_DISR_C1_Depots_Search_Matched_Runbook_v2_20261008.md"
CONFIG_REL = "configs/c1_depots_search_match_v1.yaml"
PKG = "src/cp_disr/pddl/search_match/"
CORE = tuple(PKG + n for n in ("__init__.py", "budget.py", "engine.py", "evaluators.py", "runner.py", "hadd_server.cc")) + ("scripts/c1_depots_search_match.py", CONFIG_REL, "tests/test_depots_search_match.py")


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


def runtime(rr):
    return rj(Path(rr) / "suite_runtime.json")


def old_run(rr):
    from cp_disr.pddl.search_match.runner import OldRun
    return OldRun(runtime(rr)["old_run_root"])


# ------------------------------------------------------------------------------------------------ identities
def evaluator_identity(old):
    from cp_disr.pddl.search_match import evaluators as EV
    from cp_disr.pddl.search_match.runner import NEURAL, sha_file as sf
    ident = {"neural": {}, "wl": {}, "heuristics": {}, "software": {}}
    for sc, mode in NEURAL.items():
        ck = old.checkpoint(mode)
        ident["neural"][sc] = {"mode": mode, "selected_update": ck["update"], "checkpoint": str(ck["path"]), "sha256_recorded": ck["sha256"], "sha256_now": sf(ck["path"]), "init_seed": ck["init_seed"]}
        assert ident["neural"][sc]["sha256_recorded"] == ident["neural"][sc]["sha256_now"], "checkpoint changed: " + sc
    for s in ("struct", "ipc"):
        w = old.wl_params(s)
        ident["wl"][w["track"]] = {"model": str(w["model"]), "model_sha256": sf(w["model"]), "params_sha256": sf(w["params"]), "opts_sha256": sf(w["opts"]), "used_for_sets": ["struct", "joint"] if w["track"] == "typed" else ["ipc"],
                                   "fit_config": "configurations/classic.toml (rank-svm, downward state representation, wl features, 2 iterations, ilg graph)", "rounding": "round(prediction) as in the native heuristic"}
    ident["heuristics"] = {"H_ADD": {"binary": str(EV.HADD_BIN), "sha256": sf(EV.HADD_BIN), "definition": "additive delete relaxation, unit action cost, positive preconditions only; unreachable goal -> +inf (sorted last)"},
                           "H_COUNT": {"definition": "number of unmet final goal atoms"}}
    git = lambda *a, cwd: subprocess.check_output(["git", "-C", str(cwd)] + list(a), text=True).strip()
    ext = Path(os.environ.get("CPDISR_EXT", str(Path.home() / "ext")))
    import torch
    ident["software"] = {"goose": git("rev-parse", "HEAD", cwd=ext / "goose"), "scorpion": git("rev-parse", "HEAD", cwd=ext / "goose" / "ext" / "planners" / "scorpion"), "wlplan": git("rev-parse", "HEAD", cwd=ext / "goose" / "ext" / "wlplan"),
                         "fast_downward": git("rev-parse", "HEAD", cwd=ext / "downward"), "pddl_generators": git("rev-parse", "HEAD", cwd=ext / "pddl-generators"),
                         "downward_benchmarks": git("rev-parse", "HEAD", cwd=ext / "downward-benchmarks"), "torch": torch.__version__, "python": sys.version.split()[0]}
    old_sw = rj(Path(old.root) / "assets" / "public_domain_identity.json")["ext"]
    ident["software"]["recorded_by_previous_card"] = old_sw
    return ident


# ------------------------------------------------------------------------------------------------ init
def cmd_init(gpus, auth):
    cfg = load_cfg()
    git = lambda *a: subprocess.check_output(["git", "-C", str(ROOT)] + list(a), text=True).strip()
    if git("branch", "--show-current") != BRANCH or subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"]).returncode != 0:
        raise SystemExit("branch %s descending from %s required" % (BRANCH, BASE_COMMIT))
    old_root = ROOT / cfg["old_run_root"]
    if not (old_root / "receipts" / "final_receipt.json").is_file():
        raise SystemExit("old result root incomplete: %s" % old_root)
    hashes = {s: sha_file(ROOT / s) for s in CORE if (ROOT / s).is_file()}
    code8 = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:8]
    parent = Path(os.environ["SM_RUN_PARENT"]) if os.environ.get("SM_RUN_PARENT") else ROOT / cfg["suite_rel"]
    rr = parent / ("%s_%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), code8))
    for sub in ("plan", "assets", "search", "plans", "results", "receipts", "driver_logs"):
        (rr / sub).mkdir(parents=True, exist_ok=True)
    storage = Path("/home/xushijie3")
    py = str(storage / "envs" / "cpdisr" / "bin" / "python")
    from cp_disr.pddl.search_match.runner import OldRun
    old = OldRun(old_root)
    ident = evaluator_identity(old)
    wj(rr / "assets" / "evaluator_identity.json", ident)
    old_final = rj(old_root / "receipts" / "final_receipt.json")
    wj(rr / "assets" / "old_run_identity.json", {"old_run_root": str(old_root), "old_final_receipt_sha256": sha_file(old_root / "receipts" / "final_receipt.json"), "old_manifest_sha256": sha_file(old_root / "data" / "manifest.json"),
                                                  "old_task_statuses": len(old_final["task_statuses"]), "base_commit": BASE_COMMIT, "note": "historical results are read only and never rewritten"})
    warm = next(c["file"] for c in old.manifest["train"] if c["case_id"] == "train_n3_000")
    env = {"PYTHONPATH": "src", "TMPDIR": str(storage / "cp_disr_tmp"), "XDG_CACHE_HOME": str(storage / "cp_disr_cache"), "TORCH_HOME": str(storage / "cp_disr_cache" / "torch"), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
           "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "CPDISR_EXT": str(Path.home() / "ext"), "CPDISR_WARMUP_PROBLEM": warm,
           "LD_LIBRARY_PATH": "/home/xushijie3/.uv_python/cpython-3.10.19-linux-x86_64-gnu/lib"}
    me = str(ROOT / "scripts" / "c1_depots_search_match.py")
    tasks = []

    def add(tid, args, requires=(), resource="gpu"):
        tasks.append({"id": tid, "argv": [py, me] + args + ["--run-root", str(rr)], "receipt": str(rr / "receipts" / ("%s.json" % tid)), "input_hashes": hashes, "requires": list(requires), "resource": resource})
    add("assets", ["assets"], [], "cpu")
    add("fixtures", ["fixtures"], ["assets"], "cpu")
    for s in ("ipc", "joint", "struct"):                              # longest tasks first (makespan); the written report is ordered struct -> joint -> ipc
        n = cfg["shards"][s]
        for k in range(n):
            for sc in cfg["scorers"]:
                tid = "search_%s_%s_%d" % (sc, s, k)
                add(tid, ["search", "--scorer", sc, "--set", s, "--shard", str(k), "--nshards", str(n)], ["fixtures"], "gpu" if sc.startswith("V_") else "cpu")
    # topology / reference sensitivity / report are run by the CLI after the searches (they read results only and do not influence them)

    manifest = {"card": CARD, "execution_authorized": True, "authorization_source": auth, "gpus": gpus, "env": env, "repo_root": str(ROOT), "tasks": tasks, "max_new_training_runs": 0, "cpu_slots": int(cfg["limits"]["cpu_slots"])}
    wj(rr / "manifest.json", manifest)
    wj(rr / "suite_runtime.json", {"repo_root": str(ROOT), "run_root": str(rr), "venv_python": py, "gpus": gpus, "base_commit": BASE_COMMIT, "branch": BRANCH, "head": git("rev-parse", "HEAD"), "old_run_root": str(old_root), "storage_owner": "xushijie3"})
    (rr / "plan" / "PLAN.md").write_text((ROOT / PLAN_REL).read_text(encoding="utf-8"), encoding="utf-8")
    (rr / "plan" / "config.yaml").write_text((ROOT / CONFIG_REL).read_text(encoding="utf-8"), encoding="utf-8")
    reg = {"card": CARD, "plan": PLAN_REL, "plan_sha256": sha_file(ROOT / PLAN_REL), "config_sha256": sha_file(ROOT / CONFIG_REL), "base_commit": BASE_COMMIT, "authorization_text": auth,
           "authorization_text_sha256": hashlib.sha256(auth.encode()).hexdigest(), "new_training_runs": 0, "optimizer_steps": 0, "new_gate_labels": 0, "new_problem_generation": 0,
           "scope": "six frozen scorers x 182 problems in ONE shared search engine (1,092 runs), reference-length sensitivity (derived), problem topology table, full report; no training",
           "frozen_search_protocol": cfg["search"], "scorers": cfg["scorers"], "shards": cfg["shards"], "evaluator_identity": "assets/evaluator_identity.json", "core_source_hashes": hashes,
           "fixture_policy": "interface fixtures use Train96 problems (and synthetic graphs) only; no Struct / Joint / IPC search outcome is looked at before the lock; the WL-GOOSE state conversion was additionally compared with the native planner on states of six IPC problem files (heuristic values only, no search, no outcome)",
           "decision_rules": {"primary_comparison": "V_DENSE vs H_WL at the 10,000-expansion milestone (1,000 and 100,000 are the budget curve), per data set; coverage, common-solved plan length, expansions to first solution; H_ADD and H_COUNT are the scale references",
                              "outcomes": "plan section 15 classes A-F decide the written method recommendation; no automatic new training"},
           "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    wj(rr / "plan" / "registration.json", reg)
    print(json.dumps({"run_root": str(rr), "tasks": len(tasks)}))


# ------------------------------------------------------------------------------------------------ stages
def cmd_assets(rr):
    cfg = load_cfg()
    old = old_run(rr)
    ident = evaluator_identity(old)
    reg = rj(Path(rr) / "assets" / "evaluator_identity.json")
    for sc in ident["neural"]:
        assert ident["neural"][sc]["sha256_now"] == reg["neural"][sc]["sha256_now"]
    for tr in ident["wl"]:
        assert ident["wl"][tr]["model_sha256"] == reg["wl"][tr]["model_sha256"]
    bad = []
    for s in cfg["sets"]:
        for c in old.cases(s):
            if sha_file(c["file"]) != c["sha256"]:
                bad.append(c["case_id"])
    if bad:
        raise SystemExit("problem files differ from the manifest: %s" % bad[:5])
    counts = {s: len(old.cases(s)) for s in cfg["sets"]}
    finish(rr, "assets", "DONE", [], problems=counts, total=sum(counts.values()), hadd_binary_sha256=sha_file(__import__("cp_disr.pddl.search_match.evaluators", fromlist=["x"]).HADD_BIN))
    return 0


def cmd_fixtures(rr):
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_depots_search_match.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=str(ROOT), env=env, capture_output=True, text=True)
    log = Path(rr) / "results" / "fixtures.log"
    log.write_text(r.stdout + r.stderr)
    finish(rr, "fixtures", "DONE" if r.returncode == 0 else "GLOBAL_INTEGRITY_FAILURE", [log], pytest_tail=r.stdout[-300:])
    return r.returncode


def cmd_search(rr, scorer, set_name, shard, nshards):
    import torch
    from cp_disr.pddl.search_match.runner import run_shard
    cfg = load_cfg()
    old = old_run(rr)
    device = None
    if scorer.startswith("V_"):
        if not torch.cuda.is_available():
            raise SystemExit("neural scorer without a GPU")
        device = torch.device("cuda", 0)
    out = Path(rr) / "search" / ("%s__%s__%d.jsonl" % (scorer, set_name, shard))
    ident, load_seconds = run_shard(old, scorer, set_name, shard, nshards, search_cfg(cfg), out, Path(rr) / "plans", device)
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    from collections import Counter
    finish(rr, "search_%s_%s_%d" % (scorer, set_name, shard), "DONE", [out], cases=len(recs), solved=sum(r["solved"] for r in recs), statuses=dict(Counter(r["status"] for r in recs)), identity=ident, model_load_seconds=load_seconds)
    return 0


def cmd_topology(rr):
    from cp_disr.pddl.search_match import analysis as AN
    p = AN.topology(old_run(rr), Path(rr) / "results" / "topology.csv")
    finish(rr, "topology", "DONE", [p])
    return 0


def cmd_ref_sensitivity(rr):
    from cp_disr.pddl.search_match import analysis as AN
    p, summary = AN.reference_sensitivity(old_run(rr), Path(rr) / "results" / "reference_length_sensitivity.csv")
    finish(rr, "ref_sensitivity", "DONE", [p], **summary)
    return 0


def cmd_report(rr):
    from cp_disr.pddl.search_match import report as RP
    outs = RP.build(rr, old_run(rr), load_cfg())
    finish(rr, "report", "DONE", outs)
    return 0


def cmd_final_receipt(rr, wl_verified):
    rr = Path(rr)
    cfg = load_cfg()
    recs = []
    for p in sorted((rr / "search").glob("*.jsonl")):
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    from collections import Counter
    st = Counter(r["status"] for r in recs)
    technical = sum(st.get(k, 0) for k in ("ADAPTER_ERROR", "INVALID_PLAN", "MODEL_NONFINITE"))
    resource = sum(st.get(k, 0) for k in ("TIMEOUT", "MEMORY_LIMIT", "NODE_LIMIT"))
    statuses = {r_["task"]: r_["status"] for r_ in (rj(p) for p in sorted((rr / "receipts").glob("*.json")) if p.name != "final_receipt.json")}
    out = {"card": CARD, "base_commit": BASE_COMMIT, "new_training_runs": 0, "optimizer_steps": 0, "new_gate_labels": 0, "new_problem_generation": 0, "search_conditions_planned": len(cfg["scorers"]), "search_cases_planned": 182,
           "search_runs_planned": 1092, "completed_search_runs": len(recs), "solved": sum(r["solved"] for r in recs), "status_counts": dict(st), "technical_incomplete": technical, "resource_limited": resource,
           "common_engine": "src/cp_disr/pddl/search_match/engine.py (eager GBFS, common adaptation protocol)", "wl_identity_verified": bool(wl_verified), "historical_results_modified": False, "path_owner": "xushijie3",
           "task_statuses": statuses, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "next_action": "WAIT_FOR_METHOD_DECISION"}
    p = rr / "receipts" / "final_receipt.json"
    wj(p, out)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run-root")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--authorization-text-file")
    ap.add_argument("--scorer")
    ap.add_argument("--set")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--wl-verified", default="true")
    a = ap.parse_args()
    if a.cmd == "init":
        return cmd_init([int(x) for x in a.gpus.split(",")], Path(a.authorization_text_file).read_text(encoding="utf-8"))
    rr = a.run_root
    if a.cmd == "run-all":
        from cp_disr.pddl.sched import run_all
        return run_all(rr, cpu_slots=rj(Path(rr) / "manifest.json").get("cpu_slots", 4))
    fn = {"assets": lambda: cmd_assets(rr), "fixtures": lambda: cmd_fixtures(rr), "search": lambda: cmd_search(rr, a.scorer, a.set, a.shard, a.nshards), "topology": lambda: cmd_topology(rr),
          "ref_sensitivity": lambda: cmd_ref_sensitivity(rr), "report": lambda: cmd_report(rr), "final_receipt": lambda: cmd_final_receipt(rr, a.wl_verified == "true")}[a.cmd]
    r = fn()
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
