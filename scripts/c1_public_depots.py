#!/usr/bin/env python
"""C1-PUBLIC-DEPOTS-REL-V1 (plan CP_DISR_C1_Next_Direction_v2_Depots_REL_20261008).

    init    --gpus 0,1,..  --authorization-text-file F      resolve paths, run root, identities, frozen design, manifest
    run-all --run-root R                                    parallel scheduler (dependency DAG, one task per free GPU, bounded CPU tasks, resumable)
    <stage> --run-root R [--model M] [--exec E] [--set S] [--condition C] [--track T]

Stages: prep_ext gen_data labels ref goose_fit train devsel lock eval goose_eval bw_tau bw_eval bw_probes gate_decision report
Smoke mode (DEPOTS_SMOKE=1): tiny counts / budgets / time limits for an end-to-end dry run outside the repository tree.
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
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SMOKE = bool(os.environ.get("DEPOTS_SMOKE"))
CARD = "C1-PUBLIC-DEPOTS-REL-V1"
BASE_COMMIT = "e3b1e65b3725174cc6169d7857d0983793037b88"
BRANCH = "codex/cp-disr-c1-public-depots-rel-v1"
PLAN_REL = "docs/c1_blocksworld/CP_DISR_C1_Next_Direction_v2_Depots_REL_20261008.md"
SUITE_REL = "runs/final_master/c1_route_b/public_depots_rel_v1"
V3_REL = "runs/final_master/c1_route_b/blocksworld_main_v1/method_serial_v3/20261007T165611Z_a994451e"
EXT = Path.home() / "ext"
GOOSE_PY = str(Path.home() / "envs" / "goose" / "bin" / "python")
GOOSE_LD = str(Path.home() / ".uv_python" / "cpython-3.10.19-linux-x86_64-gnu" / "lib")
MODES = ("mg", "dense", "rel")
MODEL_NAME = {"mg": "MG-G", "dense": "DENSE-G", "rel": "REL-G"}
EXECS = ("one_step", "two_step", "gated")
SETS = ("struct", "joint", "ipc")
CORE = ("src/cp_disr/pddl/model.py", "src/cp_disr/pddl/task.py", "src/cp_disr/pddl/parse.py", "src/cp_disr/pddl/data.py", "src/cp_disr/pddl/depots.py", "src/cp_disr/pddl/control.py", "src/cp_disr/pddl/train.py",
        "src/cp_disr/pddl/stages.py", "src/cp_disr/pddl/bw_stage.py", "src/cp_disr/pddl/sched.py", "src/cp_disr/pddl/report.py", "src/cp_disr/pddl/exact_dist.cc", "scripts/c1_public_depots.py", "tests/test_pddl_c1.py",
        "src/cp_disr/blocksworld/method_serial/model.py", "src/cp_disr/blocksworld/method_serial/trainer.py", "src/cp_disr/blocksworld/method_serial/lookahead.py")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def wj(p, doc):
    from cp_disr.pddl.sched import wj as _wj
    _wj(p, doc)


def wcsv(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
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


def rj(p):
    return json.loads(Path(p).read_text())


def sha_file(p):
    from cp_disr.pddl.sched import sha_file as s
    return s(p)


def S():
    """Lazy stage-library import with smoke overrides applied."""
    from cp_disr.pddl import stages
    if SMOKE:
        stages.TRAIN_PER_N, stages.DEV_PER_N, stages.STRUCT_PER, stages.JOINT_PER_CELL = 2, 2, 1, 1
        stages.LAMA_SECONDS, stages.LMCUT_SECONDS, stages.DEADLINE_SECONDS = 5, 5, 30
        stages.BUDGET.update({"updates": 40, "checkpoints": [8, 16, 24, 32, 40]})
    return stages


def finish(rr, tid, status, outs=(), **extra):
    from cp_disr.pddl.sched import finish as f
    f(rr, tid, status, outs, **extra)


# ------------------------------------------------------------------------------------------------ init
def cmd_init(gpus, auth):
    import torch
    from cp_disr.pddl import stages
    from cp_disr.pddl.sched import sha_file as shaf
    st = S()
    git = lambda *a, cwd=ROOT: subprocess.check_output(["git", "-C", str(cwd)] + list(a), text=True).strip()
    if git("branch", "--show-current") != BRANCH or git("rev-parse", "HEAD") != BASE_COMMIT:
        raise SystemExit("branch %s at %s required" % (BRANCH, BASE_COMMIT))
    storage = Path("/home/xushijie3")
    hashes = {s: shaf(ROOT / s) for s in CORE if (ROOT / s).is_file()}
    code8 = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:8]
    parent = Path(os.environ["DEPOTS_RUN_PARENT"]) if os.environ.get("DEPOTS_RUN_PARENT") else ROOT / SUITE_REL
    rr = parent / ("%s_%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), code8))
    for sub in ("plan", "assets", "data", "stages", "runs", "eval", "results", "receipts", "driver_logs", "bw"):
        (rr / sub).mkdir(parents=True, exist_ok=True)
    (rr / "receipts" / "case_ledger.jsonl").write_text("")
    py = str(storage / "envs" / "cpdisr" / "bin" / "python")
    ext = {}
    for name, d in (("fast_downward", EXT / "downward"), ("pddl_generators", EXT / "pddl-generators"), ("downward_benchmarks", EXT / "downward-benchmarks"), ("goose", EXT / "goose")):
        ext[name] = git("rev-parse", "HEAD", cwd=d)
    ext["goose_wlplan"] = git("rev-parse", "HEAD", cwd=EXT / "goose" / "ext" / "wlplan")
    ext["goose_scorpion"] = git("rev-parse", "HEAD", cwd=EXT / "goose" / "ext" / "planners" / "scorpion")
    ext["exact_dist_sha256"] = shaf(EXT / "bin" / "exact_dist")
    cfg = {"storage_root": str(storage), "repo_root": str(ROOT), "run_root": str(rr), "venv_python": py, "gpus": gpus, "base_commit": BASE_COMMIT, "branch": BRANCH, "head": git("rev-parse", "HEAD"),
           "v3_run_root": str(ROOT / V3_REL), "ext": ext, "smoke": SMOKE, "storage_owner": "xushijie3", "execution_authorized": True}
    wj(rr / "suite_runtime.json", cfg)
    shutil_copy = lambda a, b: Path(b).write_text(Path(a).read_text(encoding="utf-8"), encoding="utf-8")
    shutil_copy(ROOT / PLAN_REL, rr / "plan" / "PLAN.md")
    ids = {"M1_GOAL_final": {"path": "runs/final_master/c1_route_b/blocksworld_main_v1/gp_minimal_attribution_v1/20261006T143512Z_3cb37301/runs/R-C1-GPA-M1-GOAL-0/checkpoints/final.pt"}}
    sel = rj(Path(cfg["v3_run_root"]) / "runs" / "GOAL_REL" / "checkpoint_selection.json")
    ids["v3_GOAL_REL_selected"] = sel["selected_checkpoint"]
    ids["v3_GOAL_DENSE_selected"] = rj(Path(cfg["v3_run_root"]) / "runs" / "GOAL_DENSE" / "checkpoint_selection.json")["selected_checkpoint"]
    for k, v in ids.items():
        v["sha256_now"] = shaf(ROOT / v["path"])
    wj(rr / "assets" / "asset_identity.json", ids)
    env = {"PYTHONPATH": "src", "TMPDIR": str(storage / "cp_disr_tmp"), "XDG_CACHE_HOME": str(storage / "cp_disr_cache"), "TORCH_HOME": str(storage / "cp_disr_cache" / "torch"), "PYTHONUNBUFFERED": "1",
           "CPDISR_EXT": str(EXT), "CPDISR_EXACT_BIN": str(EXT / "bin" / "exact_dist")}
    if SMOKE:
        env["DEPOTS_SMOKE"] = "1"
    me = str(ROOT / "scripts" / "c1_public_depots.py")
    tasks = []

    def add(tid, args, requires=(), resource="gpu", training=None):
        t = {"id": tid, "argv": [py, me] + args + ["--run-root", str(rr)], "receipt": str(rr / "receipts" / ("%s.json" % tid)), "input_hashes": hashes, "requires": list(requires), "resource": resource}
        if training:
            t["training_run_id"] = training
        tasks.append(t)
    add("prep_ext", ["prep_ext"], resource="cpu")
    add("gen_data", ["gen_data"], ["prep_ext"], "cpu")
    add("labels", ["labels"], ["gen_data"], "cpu")
    for s in ("dev", "struct", "joint", "ipc"):
        add("ref_" + s, ["ref", "--set", s], ["gen_data"], "cpu")
    add("goose_fit_typed", ["goose_fit", "--track", "typed"], ["labels"], "cpu")
    add("goose_fit_ipc", ["goose_fit", "--track", "ipc"], ["labels"], "cpu")
    # Blocksworld zero-training stage (independent of the Depots data)
    add("bw_tau", ["bw_tau"])
    for c in ("GATED_MG", "GATED_REL"):
        add("bw_eval_" + c, ["bw_eval", "--condition", c], ["bw_tau"])
    add("bw_eval_LOOK2_HADD", ["bw_eval", "--condition", "LOOK2_HADD"])
    add("bw_probes", ["bw_probes"])
    add("gate_decision", ["gate_decision"], ["bw_eval_GATED_MG", "bw_eval_GATED_REL", "bw_eval_LOOK2_HADD"], "cpu")
    for m in MODES:
        add("train_" + m, ["train", "--model", m], ["labels"], "gpu", "R-C1-PD-%s-0" % MODEL_NAME[m])
        add("devsel_" + m, ["devsel", "--model", m], ["train_" + m, "labels"])
    add("lock", ["lock"], ["devsel_" + m for m in MODES] + ["goose_fit_typed", "goose_fit_ipc"], "gpu")
    for m in MODES:
        for e in EXECS:
            for s in SETS:
                req = ["lock", "ref_" + s] + (["gate_decision"] if e == "gated" else [])
                if s in ("struct",):
                    req.append("labels")
                add("eval_%s_%s_%s" % (m, e, s), ["eval", "--model", m, "--exec", e, "--set", s], req)
    for s in SETS:
        add("goose_eval_" + s, ["goose_eval", "--set", s], ["lock", "ref_" + s], "cpu")
    add("report", ["report"], [], "cpu")
    tasks[-1]["after"] = [t["id"] for t in tasks[:-1]]                                          # runs last, also when some task failed (reports what exists)
    manifest = {"card": CARD, "execution_authorized": True, "authorization_source": auth, "gpus": gpus, "env": env, "repo_root": str(ROOT), "tasks": tasks, "max_new_training_runs": 3}
    wj(rr / "manifest.json", manifest)
    reg = {"card": CARD, "plan": PLAN_REL, "plan_sha256": shaf(ROOT / PLAN_REL), "base_commit": BASE_COMMIT, "authorization_text": auth, "authorization_text_sha256": hashlib.sha256(auth.encode()).hexdigest(),
           "scope": "plan sections 3-12: Blocksworld zero-training (gated look-ahead, h_add leaf, two probes) + generic relation interface + public Depots (3 internal trainings MG-G/DENSE-G/REL-G) + external LAMA and WL-GOOSE; "
                    "optional AIW-AD / KR 2023 not run unless stated in the final receipt",
           "frozen_design": {"seeds": {"train": st.SEEDS["train"], "dev": st.SEEDS["dev"], "struct": st.SEEDS["struct"], "joint0": st.JOINT_SEED0}, "per": {"train_per_n": st.TRAIN_PER_N, "dev_per_n": st.DEV_PER_N, "struct": st.STRUCT_PER, "joint_per_cell": st.JOINT_PER_CELL},
                             "budget": st.BUDGET, "fd": {"lama_seconds": st.LAMA_SECONDS, "lama_first_seconds": st.LAMA_SECONDS, "lmcut_seconds": st.LMCUT_SECONDS, "memory_mb": st.FD_MEMORY_MB},
                             "time_budget": {"per_problem_wall_seconds_all_time_limited_methods": st.DEADLINE_SECONDS, "neural_controllers": "checked between decisions (a decision is never interrupted)", "wl_goose": "whole planner run incl. preprocessing, process group killed at the limit",
                                             "lama": "anytime configuration with the same wall limit; first-plan time and length recorded separately"},
                             "places": 2, "trucks": 1, "hoists": 2, "pallets": "= crates", "train_goal_shape": "exactly one stack of height 2 or 3, all other crates alone on a pallet; complete layout (every crate has a goal)",
                             "struct": "4 crates 2+2 (16) and 5 crates 3+2 (16)", "joint_cells": {"n6k1h2": [2, 1, 1, 1, 1], "n6k1h4": [4, 1, 1], "n6k2h2": [2, 2, 1, 1], "n6k2h4": [4, 2], "n8k1h2": [2, 1, 1, 1, 1, 1, 1], "n8k1h4": [4, 1, 1, 1, 1],
                                                                                                                                                             "n8k2h2": [2, 2, 1, 1, 1, 1], "n8k2h4": [4, 2, 1, 1]},
                             "selection": "dev (24 problems): most successes, then smaller failure-penalised cost ratio, then earlier checkpoint", "gate_tau": "10th percentile of top-2 logit margin over unique training decisions with >=2 legal actions",
                             "step_cap": "2*L_ref+4 (L_ref: exact optimum, else A*+LM-cut optimum, else best known LAMA plan)", "external": {"lama": "seq-sat-lama-2011 (anytime) + lama-first + seq-opt-lmcut", "wl_goose": "configurations/classic.toml, trained on the same 96 training problems with optimal plans"},
                             "exec": list(EXECS),
                             "gated_depots_runs_only_if": "pooled over the MG and REL scorers on the Confirm128 problems the selective look-ahead (a) evaluates fewer leaf states than the fixed two-step tree, (b) keeps >= 90% of its completions and (c) makes no more avoidable destructions of satisfied goals (gate_decision stage); the thresholds tau are the 10th percentile of the top-2 margin over TRAINING decisions of each scorer, frozen before any evaluation episode",
                             "primary_comparisons": "REL-G vs DENSE-G at one_step and at two_step on struct and on joint: completion counts (only-first / only-second), failure-penalised cost ratio and common-success cost difference with a two-sided sign test; every other comparison is exploratory and uncorrected",
                             "blocksworld_stage": {"scorers": {"MG": "M1-GOAL final checkpoint (zero-initialised SerialModel wrapper)", "REL": "GOAL_REL checkpoint selected by the method-serial v3 card"}, "problems": "Confirm128 (development material), 128 episodes per condition",
                                                   "conditions": ["GATED_MG", "GATED_REL", "LOOK2_HADD (MG roots/C3, same two-step tree, leaf = h_add of the grounded contracts)"], "reused_rows": ["MG_C3", "LOOK2_MG", "GOAL_REL", "GOAL_DENSE", "REL_LOOK2", "DENSE_LOOK2"],
                                                   "probes": "module-level isolation and end-to-end augmentation on the 240-state bank (DENSE/REL/MG)"},
                             "labels": "exact BFS distances (full reachable state space) for train/dev/struct and 6-crate joint problems (optimum + goal-destruction label); 8-crate joint and IPC problems: A*+LM-cut within the budget, else best known LAMA plan; no label enters a forward pass",
                             "external_optional": "AIW-AD / KR 2023: not part of the formal run; status is stated in the final receipt"},
           "core_source_hashes": hashes, "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    wj(rr / "plan" / "registration.json", reg)
    print(json.dumps({"run_root": str(rr), "tasks": len(tasks), "trainings": 3}))


# ------------------------------------------------------------------------------------------------ helpers shared by stages
def runtime(rr):
    return rj(Path(rr) / "suite_runtime.json")


def load_man(rr):
    return rj(Path(rr) / "data" / "manifest.json")


def load_exact(rr):
    return rj(Path(rr) / "data" / "exact.json")


def load_refs(rr, s):
    p = Path(rr) / "data" / ("refs_%s.json" % s)
    return rj(p) if p.is_file() else {}


def domain_for(s):
    from cp_disr.pddl import depots as DP
    return DP.DOMAIN_IPC if s == "ipc" else DP.DOMAIN_TYPED


def build_cases(rr, fam, with_refs=True):
    st = S()
    man, exact = load_man(rr), load_exact(rr)
    refs = load_refs(rr, fam) if fam in ("dev", "struct", "joint", "ipc") else {}
    cases = []
    for c in man[fam]:
        cid = c["case_id"]
        ex = exact.get(cid)
        if ex and ex.get("status") == "OK":
            L, kind = ex["optimal_length"], "exact"
        elif cid in refs:
            L, kind = st.reference_length(refs[cid])
        else:
            L, kind = None, "missing"
        if L is None:
            L, kind = 150, "no_reference_cap150"           # IPC instance without any reference plan: fixed cap, no cost ratio
        meta = {"exact": bool(ex and ex.get("status") == "OK"), "length_kind": kind}
        if "cell" in c:
            meta["cell"] = c["cell"]
        if "analysis" in c:
            a = c["analysis"]
            meta.update({"n_crates": a["n_crates"], "k_towers": a["k_goal_towers"], "max_goal_height": a["max_goal_height"], "needs_transport": a["needs_transport"], "max_init_height": a["max_init_height"]})
        if ex and ex.get("status") == "OK":
            meta["requires_goal_destruction"] = ex["requires_goal_destruction"]
        cases.append(st.make_case(c, fam, domain_for(fam), L, kind, **meta))
    return cases


def make_device():
    import torch
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


# ------------------------------------------------------------------------------------------------ stages: data and external identity
def cmd_prep_ext(rr):
    cfg = runtime(rr)
    out = Path(rr) / "assets" / "public_domain_identity.json"
    from cp_disr.pddl import depots as DP
    doc = {"ext": cfg["ext"], "typed_domain_sha256": hashlib.sha256(DP.DOMAIN_TYPED.read_bytes()).hexdigest(), "ipc_domain_sha256": hashlib.sha256(DP.DOMAIN_IPC.read_bytes()).hexdigest(),
           "generator_binary_sha256": sha_file(DP.GEN), "fd_version": subprocess.run(["python3", str(EXT / "downward" / "fast-downward.py"), "--version"], capture_output=True, text=True).stdout.strip()}
    wj(out, doc)
    finish(rr, "prep_ext", "DONE", [out])
    return 0


def cmd_gen_data(rr):
    st = S()
    man = st.gen_data(rr)
    p = Path(rr) / "data" / "manifest.json"
    wj(p, man)
    short = {k: len(v) for k, v in man.items() if isinstance(v, list)}
    bad = {k: v for k, v in man["generation_report"].items() if v["built"] != v["requested"]}
    finish(rr, "gen_data", "DONE", [p], counts=short, shortfalls=bad)
    return 0


def cmd_labels(rr):
    st = S()
    man = load_man(rr)
    ex = st.exact_labels(man)
    p = Path(rr) / "data" / "exact.json"
    wj(p, ex)
    ok = sum(1 for v in ex.values() if v["status"] == "OK")
    finish(rr, "labels", "DONE", [p], cases=len(ex), ok=ok, too_big=len(ex) - ok)
    return 0


def cmd_ref(rr, s):
    st = S()
    man = load_man(rr)
    cases = man[s]
    if SMOKE and s == "ipc":
        cases = cases[:3]
    refs = st.fd_references(cases, domain_for(s), Path(rr) / "data" / "fd" / s, workers=24 if SMOKE else 30, lama_s=st.LAMA_SECONDS, lmcut_s=st.LMCUT_SECONDS)
    for cid, r in refs.items():
        r["L_ref"], r["L_kind"] = st.reference_length(r)
    p = Path(rr) / "data" / ("refs_%s.json" % s)
    wj(p, refs)
    finish(rr, "ref_" + s, "DONE", [p], solved=sum(1 for r in refs.values() if r["L_ref"] is not None), n=len(refs))
    return 0


# ------------------------------------------------------------------------------------------------ stages: GOOSE (external learned heuristic)
def plan_file_text(ids):
    return "\n".join("(" + " ".join(a.split(":")[1:-1]) + ")" for a in ids) + "\n"


def to_ipc_encoding(text):
    """Typed generator problem -> the untyped IPC-2002 encoding (type facts, no declared types, goal unchanged)."""
    from cp_disr.pddl.parse import parse_problem
    p = parse_problem(text)
    objs = sorted(p.objects)
    facts = []
    for o in objs:
        t = p.objects[o]
        if t in ("depot", "distributor", "place"):
            facts.append("(place %s)" % o)
        elif t in ("pallet", "crate"):
            facts += ["(%s %s)" % (t, o), "(surface %s)" % o]
        else:
            facts.append("(%s %s)" % (t, o))
    init = ["(%s %s)" % (a[0], " ".join(a[1])) for a in sorted(p.init)]
    goal = ["(%s %s)" % (a[0], " ".join(a[1])) for a in p.goal]
    return "(define (problem %s) (:domain depot)\n(:objects %s)\n(:init\n%s\n)\n(:goal (and\n%s\n)))\n" % (p.name, " ".join(objs), "\n".join(facts + init), "\n".join(goal))


def cmd_goose_fit(rr, track):
    man, exact = load_man(rr), load_exact(rr)
    d = Path(rr) / "goose" / ("train_" + track)
    (d / "training").mkdir(parents=True, exist_ok=True)
    (d / "training_plans").mkdir(parents=True, exist_ok=True)
    dom = domain_for("ipc" if track == "ipc" else "struct")
    (d / "domain.pddl").write_text(dom.read_text())
    n = 0
    for c in man["train"]:
        ex = exact[c["case_id"]]
        if ex["status"] != "OK":
            continue
        text = Path(c["file"]).read_text()
        (d / "training" / ("%s.pddl" % c["case_id"])).write_text(to_ipc_encoding(text) if track == "ipc" else text)
        (d / "training_plans" / ("%s.plan" % c["case_id"])).write_text(plan_file_text(ex["plan"]))
        n += 1
    model = d / "wl_goose.model"
    env = {**os.environ, "LD_LIBRARY_PATH": GOOSE_LD, "PATH": str(Path.home() / "envs" / "goose" / "bin") + ":" + str(EXT / "cmake" / "bin") + ":" + os.environ["PATH"]}
    t0 = time.time()
    r = subprocess.run([GOOSE_PY, str(EXT / "goose" / "train.py"), str(d), str(EXT / "goose" / "configurations" / "classic.toml"), "-s", str(model)], capture_output=True, text=True, env=env, cwd=str(d))
    (d / "train.log").write_text(r.stdout + r.stderr)
    ok = r.returncode == 0 and model.is_file()
    finish(rr, "goose_fit_" + track, "DONE" if ok else "TECHNICAL_INCOMPLETE", [model, d / "train.log"] if ok else [d / "train.log"], training_problems=n, wall_seconds=round(time.time() - t0, 1), returncode=r.returncode,
           config="classic.toml (recommended WL-GOOSE configuration; no tuning)")
    return 0 if ok else 1


def _goose_env():
    return {**os.environ, "LD_LIBRARY_PATH": GOOSE_LD, "PATH": str(Path.home() / "envs" / "goose" / "bin") + ":" + str(EXT / "cmake" / "bin") + ":" + os.environ["PATH"]}


def _goose_worker(a):
    """One WL-GOOSE planning run in its own working directory (the author script writes sas_plan / intermediate.tmp relative to the cwd). Success = a plan that replays to the goal under OUR semantics
    within ``limit`` wall seconds for the whole run (preprocessing + search)."""
    import signal
    from cp_disr.pddl import task as T
    from cp_disr.pddl.data import plan_ids_from_file
    track, cid, domain, problem, model, limit, workdir = a
    wd = Path(workdir) / cid
    wd.mkdir(parents=True, exist_ok=True)
    plan, inter = wd / "sas_plan", wd / "intermediate.tmp"
    for p in (plan, inter):
        if p.exists():
            p.unlink()
    cmd = [GOOSE_PY, str(EXT / "goose" / "plan.py"), str(domain), str(problem), "-m", str(model), "--timeout", str(limit), "--plan-file", str(plan), "--intermediate-file", str(inter)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_goose_env(), cwd=str(wd), start_new_session=True)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    wall = time.time() - t0
    (wd / "planner.log").write_text(out or "")
    stats = {}
    for line in (out or "").splitlines():
        for key, tag in (("expanded", "Expanded "), ("evaluated", "Evaluated "), ("generated", "Generated ")):
            if tag in line and "state(s)" in line:
                try:
                    stats[key] = int(line.split(tag)[1].split()[0])
                except (IndexError, ValueError):
                    pass
        if "Peak memory:" in line:
            stats["peak_kb"] = int(line.split("Peak memory:")[1].split()[0])
        if "Search time:" in line:
            stats["search_seconds"] = float(line.split("Search time:")[1].split("s")[0])
    valid, length = False, None
    if plan.is_file() and not timed_out:
        ids = plan_ids_from_file(plan)
        task = T.depots_task(domain, problem)
        fin = task.replay(ids)
        valid = fin is not None and task.goal_satisfied(fin)
        length = len(ids)
    return {"case_id": cid, "solved": bool(valid), "plan_length": length if valid else None, "reported_plan_length": length, "plan_valid": valid if plan.is_file() else None, "timed_out": timed_out,
            "wall_seconds": round(wall, 2), **stats}


def cmd_goose_eval(rr, s):
    from multiprocessing import Pool
    man = load_man(rr)
    track = "ipc" if s == "ipc" else "typed"
    model = Path(rr) / "goose" / ("train_" + track) / "wl_goose.model"
    cases = man[s][:3] if (SMOKE and s == "ipc") else man[s]
    limit = 30 if SMOKE else 300
    wd = Path(rr) / "goose" / ("eval_" + s)
    jobs = [(track, c["case_id"], domain_for(s), c["file"], model, limit, wd) for c in cases]
    with Pool(min(16, len(jobs))) as p:
        res = p.map(_goose_worker, jobs, chunksize=1)
    out = Path(rr) / "eval" / ("goose__%s.json" % s)
    wj(out, {"time_limit_seconds": limit, "model_sha256": sha_file(model), "results": res})
    finish(rr, "goose_eval_" + s, "DONE", [out], solved=sum(r["solved"] for r in res), n=len(res))
    return 0

# ------------------------------------------------------------------------------------------------ stages: training / selection / lock / evaluation
def train_inputs(rr):
    st = S()
    man, exact = load_man(rr), load_exact(rr)
    cases, trajs, labels = [], [], {}
    for c in man["train"]:
        ex = exact[c["case_id"]]
        if ex["status"] != "OK":
            continue
        cases.append(st.make_case(c, "train", domain_for("struct"), ex["optimal_length"], "exact"))
        trajs += ex["trajectories"]
        labels.update(ex["rank_labels"])
    return cases, trajs, labels


def cmd_train(rr, mode):
    st = S()
    cases, trajs, labels = train_inputs(rr)
    out = Path(rr) / "runs" / mode
    acct_path = out / "training_accounting.json"
    if acct_path.is_file():
        finish(rr, "train_" + mode, "DONE", [acct_path], note="already complete")
        return 0
    acct, rows = st.train_loop(out, mode, make_device(), cases, trajs, labels)
    wj(acct_path, acct)
    wcsv(out / "updates.csv", [{k: v for k, v in r.items()} for r in rows])
    finish(rr, "train_" + mode, "DONE", [acct_path, out / "updates.csv"], wall=acct["wall_seconds"])
    return 0


def load_model(rr, mode, update):
    from cp_disr.pddl import model as PM
    from cp_disr.torch_rl import load_checkpoint
    acct = rj(Path(rr) / "runs" / mode / "training_accounting.json")
    info = acct["checkpoints"][str(update)]
    assert sha_file(info["path"]) == info["sha256"], "checkpoint hash mismatch"
    dev = make_device()
    m = PM.make_model(mode, dev, seed=acct["budget"]["init_seed"])
    load_checkpoint(info["path"], m)
    m.eval()
    return m


def cmd_devsel(rr, mode):
    st = S()
    acct = rj(Path(rr) / "runs" / mode / "training_accounting.json")
    dev_cases = build_cases(rr, "dev")
    rows, table = {}, []
    for u in acct["checkpoints"]:
        m = load_model(rr, mode, u)
        res = st.run_set(Path(rr) / "runs" / mode / ("dev_u%s.jsonl" % u), dev_cases, m, "one_step")
        rows[u] = st.summarise(res)
        table.append({"model": mode, "update": u, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in rows[u].items()}})
    best = st.select_checkpoint(rows)
    sel = {"model": mode, "selected_update": best, "checkpoint": acct["checkpoints"][best], "dev_by_update": rows,
           "rule": "most dev successes, then smaller failure-penalised cost ratio, then earlier checkpoint", "dev_cases": len(dev_cases)}
    p = Path(rr) / "runs" / mode / "selection.json"
    wj(p, sel)
    wcsv(Path(rr) / "runs" / mode / "dev_by_checkpoint.csv", table)
    finish(rr, "devsel_" + mode, "DONE", [p], selected=best)
    return 0


def cmd_lock(rr):
    st = S()
    cases, trajs, labels = train_inputs(rr)
    lock = {"locked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": {}}
    for mode in MODES:
        sel = rj(Path(rr) / "runs" / mode / "selection.json")
        m = load_model(rr, mode, sel["selected_update"])
        tau, n = st.gate_threshold(m, cases, trajs)
        lock["models"][mode] = {"selected_update": sel["selected_update"], "checkpoint_sha256": sel["checkpoint"]["sha256"], "gate_tau": tau, "gate_tau_states": n}
        del m
    man = load_man(rr)
    lock["test_manifest_sha256"] = {s: hashlib.sha256(json.dumps([c.get("sha256") for c in man[s]]).encode()).hexdigest() for s in SETS}
    p = Path(rr) / "runs" / "lock.json"
    wj(p, lock)
    finish(rr, "lock", "DONE", [p])
    return 0


def cmd_eval(rr, mode, ex, s):
    import torch
    st = S()
    lock = rj(Path(rr) / "runs" / "lock.json")
    if ex == "gated":
        dec = rj(Path(rr) / "bw" / "gate_decision.json")
        if not dec["run_gated_on_depots"]:
            finish(rr, "eval_%s_%s_%s" % (mode, ex, s), "SKIPPED_BY_RULE", [], reason=dec["reason"])
            return 0
    cases = build_cases(rr, s)
    if SMOKE and s == "ipc":
        cases = cases[:3]
    m = load_model(rr, mode, lock["models"][mode]["selected_update"])
    out = Path(rr) / "eval" / ("%s__%s__%s.jsonl" % (mode, ex, s))
    res = st.run_set(out, cases, m, ex, tau=lock["models"][mode]["gate_tau"], use_labels=True)
    summ = st.summarise(res)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    dens = None
    if mode != "mg":
        from cp_disr.pddl import task as T
        dens = [m.mask_density(T.PddlEpisode(c).snapshot()) for c in cases]
        dens = {"mean_nonself_mask_density_initial_states": round(sum(dens) / len(dens), 4), "min": round(min(dens), 4), "max": round(max(dens), 4)}
    finish(rr, "eval_%s_%s_%s" % (mode, ex, s), "DONE", [out], summary=summ, peak_gpu_bytes=peak, deadline_seconds=st.DEADLINE_SECONDS, mask_density=dens)
    return 0


# ------------------------------------------------------------------------------------------------ stages: Blocksworld zero training
GL_REL = "runs/final_master/c1_route_b/blocksworld_main_v1/goal_lookahead_integration_v1/20261008T020121Z_5ccadc39"
SO_REL = "runs/final_master/c1_route_b/blocksworld_main_v1/scale_order_prototype_v1/20261007T083206Z_ea7f69e8"


def bw_ctx(device=None):
    from cp_disr.blocksworld import eval_a03 as E, planner as P, scale_order as SO, scorer_control as SC, state as S, train as T
    from cp_disr.blocksworld.method_serial import common as C, evalkit as EK, lookahead as LA, model as MD
    from cp_disr.rl import set_suite_half_life
    from types import SimpleNamespace
    td = rj(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(float(td["half_life"]))
    dev = device or make_device()
    c = C.Ctx(ROOT, ROOT / V3_REL, dev)
    return SimpleNamespace(E=E, P=P, SO=SO, SC=SC, S=S, T=T, C=C, EK=EK, LA=LA, MD=MD, c=c, device=dev)


def bw_scorers(d, which=("MG", "REL")):
    out = {}
    if "MG" in which:
        mg = d.MD.SerialModel(d.c.mg_model(), "MG")
        mg.eval()
        out["MG"] = mg
    for name, cond in (("REL", "GOAL_REL"), ("DENSE", "GOAL_DENSE")):
        if name in which:
            out[name], _sel = d.c.load_selected(cond)
    return out


def cmd_bw_tau(rr):
    from cp_disr.blocksworld import gp_attribution as A, imitation as I
    from cp_disr.pddl import bw_stage as BWS
    d = bw_ctx()
    models = bw_scorers(d)
    cases, _h = I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    d_train = rj(ROOT / A.GP_ROOT / "datasets" / "D_train.json")
    if SMOKE:
        d_train = d_train[:40]
    doc = {"rule": "10th percentile of the top-2 logit margin over unique training (case,state) decisions with >= 2 legal actions; frozen before any evaluation episode"}
    for name in ("MG", "REL"):
        ms = BWS.training_margins(models[name], cases, d_train)
        doc[name] = {"tau": BWS.percentile(ms, 10), "n_states": len(ms), "p50": BWS.percentile(ms, 50), "p90": BWS.percentile(ms, 90), "min": min(ms)}
    p = Path(rr) / "bw" / "tau.json"
    wj(p, doc)
    finish(rr, "bw_tau", "DONE", [p], tau={k: doc[k]["tau"] for k in ("MG", "REL")})
    return 0


BW_CONDS = {"GATED_MG": ("MG", "gated"), "GATED_REL": ("REL", "gated"), "LOOK2_HADD": ("MG", "hadd")}


def cmd_bw_eval(rr, cond):
    from cp_disr.pddl import bw_stage as BWS
    d = bw_ctx()
    scorer_name, kind = BW_CONDS[cond]
    scorer = bw_scorers(d, (scorer_name,))[scorer_name]
    man = rj(ROOT / V3_REL / "confirmation" / "manifest.json")
    assert d.C.sha_file(ROOT / man["split"]) == man["sha256"], "Confirm128 split changed"
    groups, meta = d.EK.load_cases(ROOT, man["split"])
    gl = [(k, groups[k]) for k in sorted(groups)]
    if SMOKE:
        gl = [(k, v[:1]) for k, v in gl[:3]]
    cnt = BWS.GateCounters()
    if kind == "gated":
        tau = rj(Path(rr) / "bw" / "tau.json")[scorer_name]["tau"]
        chooser = BWS.BwChooser(scorer, "gated", cnt, tau)
    else:
        chooser = BWS.BwChooser(BWS.HAddModel(), "fixed", cnt)
    out = Path(rr) / "bw" / ("by_case_%s.jsonl" % cond)
    BWS.run_condition(rr, cond, scorer, chooser, gl, out)
    eps = d.EK.jl(out)
    finish(rr, "bw_eval_" + cond, "DONE", [out], counters=cnt.as_dict(), scorer=scorer_name, kind=kind, summary=d.EK.summarize(eps, meta))
    return 0


def cmd_bw_probes(rr):
    from cp_disr.pddl import bw_stage as BWS
    d = bw_ctx()
    models = bw_scorers(d, ("MG", "DENSE", "REL"))
    bank = d.EK.bank_records(ROOT / SO_REL)
    cases, _bm = d.EK.bank_cases(ROOT)
    if SMOKE:
        bank = bank[:12]
    rows = []
    for name in ("DENSE", "REL"):
        rows.append({"probe": "module_isolation", "model": name, **BWS.probe_module_isolation(models[name], bank, cases)})
    for name in ("MG", "DENSE", "REL"):
        for variant, r in BWS.probe_augmentation(models[name], bank, cases).items():
            rows.append({"probe": "augmentation_" + variant, "model": name, **r})
    p = Path(rr) / "results" / "probe_results.csv"
    wcsv(p, rows)
    finish(rr, "bw_probes", "DONE", [p], states=len(bank))
    return 0


def _fixed_reference(which):
    v3 = ROOT / V3_REL
    if which == "MG":
        return jl(v3 / "confirmation" / "by_case_LOOK2_MG.jsonl"), rj(v3 / "receipts" / "confirm_LOOK2_MG.json")["counters"]
    gl = ROOT / GL_REL
    return jl(gl / "confirmation" / "by_case_REL_LOOK2.jsonl"), rj(gl / "receipts" / "look_REL_LOOK2.json")["counters"]


def cmd_gate_decision(rr):
    """Frozen rule (registration): the selective look-ahead is carried to Depots iff, pooled over the MG and REL scorers on the same Confirm128 problems, it (a) evaluates fewer leaf states than the fixed
    two-step tree, (b) keeps >= 90% of the fixed two-step completions and (c) makes no more avoidable destructions of satisfied goals (the placement-harm proxy of the logs) than the fixed two-step tree."""
    ev, tot = {}, {"g_succ": 0, "f_succ": 0, "g_leaves": 0, "f_leaves": 0, "g_harm": 0, "f_harm": 0}
    for scorer, cond in (("MG", "GATED_MG"), ("REL", "GATED_REL")):
        g = jl(Path(rr) / "bw" / ("by_case_%s.jsonl" % cond))
        fixed, fcnt = _fixed_reference(scorer)
        ids = {e["case_id"] for e in g}
        f = [e for e in fixed if e["case_id"] in ids]
        gcnt = rj(Path(rr) / "receipts" / ("bw_eval_%s.json" % cond))["counters"]
        full = len(ids) == len(fixed)
        scale = 1.0 if full else len(f) / len(fixed)
        row = {"scorer": scorer, "episodes": len(g), "gated_success": sum(e["success"] for e in g), "fixed_success": sum(e["success"] for e in f),
               "gated_leaves": gcnt["leaves"], "fixed_leaves": fcnt["leaves"] if full else None, "gated_unique_leaves": gcnt["unique_leaves"], "fixed_unique_leaves": fcnt["unique_leaves"] if full else None,
               "gated_expansions": gcnt["gate_expansions"], "gate_checks": gcnt["gate_checks"], "trap_unfolds": gcnt["trap_unfolds"], "gated_model_calls": gcnt["model_calls"],
               "fixed_model_calls": fcnt["model_calls"] if full else None,
               "gated_avoidable_destruction": sum(e["avoidable_destruction_steps"] for e in g), "fixed_avoidable_destruction": sum(e["avoidable_destruction_steps"] for e in f),
               "gated_interventions": sum(e["interventions"] for e in g), "fixed_interventions": sum(e["interventions"] for e in f), "full_set": full}
        ev[scorer] = row
        tot["g_succ"] += row["gated_success"]
        tot["f_succ"] += row["fixed_success"]
        tot["g_leaves"] += row["gated_leaves"]
        tot["f_leaves"] += (row["fixed_leaves"] if full else row["gated_leaves"] + 1)
        tot["g_harm"] += row["gated_avoidable_destruction"]
        tot["f_harm"] += row["fixed_avoidable_destruction"]
    a, b, c = tot["g_leaves"] < tot["f_leaves"], tot["g_succ"] >= 0.9 * tot["f_succ"], tot["g_harm"] <= tot["f_harm"]
    ok = bool(a and b and c) or SMOKE
    reason = ("(a) fewer leaf states %s (%d vs %d); (b) completions >= 90%% of fixed two-step %s (%d vs %d); (c) avoidable destruction <= fixed %s (%d vs %d)"
              % (a, tot["g_leaves"], tot["f_leaves"], b, tot["g_succ"], tot["f_succ"], c, tot["g_harm"], tot["f_harm"]))
    dec = {"run_gated_on_depots": ok, "reason": reason, "pooled": tot, "per_scorer": ev, "rule": cmd_gate_decision.__doc__.strip(), "smoke": SMOKE}
    p = Path(rr) / "bw" / "gate_decision.json"
    wj(p, dec)
    finish(rr, "gate_decision", "DONE", [p], run_gated=ok, reason=reason)
    return 0


# ------------------------------------------------------------------------------------------------ report
BW_REUSED = {"MG_C3": ("v3", "MG_C3"), "LOOK2_MG": ("v3", "LOOK2_MG"), "GOAL_REL": ("v3", "GOAL_REL"), "GOAL_DENSE": ("v3", "GOAL_DENSE"), "REL_LOOK2": ("gl", "REL_LOOK2"), "DENSE_LOOK2": ("gl", "DENSE_LOOK2")}
BW_NEW = ("GATED_MG", "GATED_REL", "LOOK2_HADD")


def cmd_bw_tables(rr):
    """Blocksworld tables of the zero-training stage: gate / h_add rows next to the REUSED rows of the earlier cards (same code, same Confirm128 problems, same weights), layers, paired comparisons, pipeline log."""
    import statistics as stats
    from cp_disr.pddl import report as RP
    d = bw_ctx()
    v3, gl = ROOT / V3_REL, ROOT / GL_REL
    man = rj(v3 / "confirmation" / "manifest.json")
    groups, meta = d.EK.load_cases(ROOT, man["split"])
    eps, counters, ident = {}, {}, {}
    for name, (which, cond) in BW_REUSED.items():
        root = v3 if which == "v3" else gl
        p = root / "confirmation" / ("by_case_%s.jsonl" % cond)
        eps[name] = {e["case_id"]: e for e in jl(p)}
        ident[name] = {"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "status": "REUSED_EXACT (same Confirm128 problems, same code path, same weights)"}
        rp = root / "receipts" / (("confirm_%s.json" if which == "v3" else "look_%s.json") % cond)
        counters[name] = (rj(rp).get("counters") if rp.is_file() else None)
    for name in BW_NEW:
        p = Path(rr) / "bw" / ("by_case_%s.jsonl" % name)
        if p.is_file():
            eps[name] = {e["case_id"]: e for e in jl(p)}
            counters[name] = rj(Path(rr) / "receipts" / ("bw_eval_%s.json" % name))["counters"]
            ident[name] = {"path": str(p.relative_to(Path(rr))), "sha256": sha_file(p), "status": "NEW"}
    layers = {"ALL": lambda m: True, "h2_seen": lambda m: m["max_tower_height"] == 2, "h4_unseen": lambda m: m["max_tower_height"] == 4, "k1": lambda m: m["k_nontrivial_towers"] == 1, "k2": lambda m: m["k_nontrivial_towers"] == 2,
              "n6": lambda m: m["n_blocks"] == 6, "n8": lambda m: m["n_blocks"] == 8, "BREAK": lambda m: m["labels"]["requires_goal_destruction"] is True, "MONO": lambda m: m["labels"]["requires_goal_destruction"] is False}
    res = Path(rr) / "results"
    rows = []
    for n, es in eps.items():
        for ln, pr in layers.items():
            sel = [e for cid, e in es.items() if pr(meta[cid])]
            if sel:
                rows.append({"condition": n, "layer": ln, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.EK.summarize(sel).items()}})
    wcsv(res / "bw_by_condition_and_layer.csv", rows)
    cost = []
    for n, es in eps.items():
        c = counters.get(n) or {}
        cost.append({"condition": n, "status": ident[n]["status"], "episodes": len(es), "wall_seconds_total": round(sum(e["wall_seconds"] for e in es.values()), 2), **{k: c.get(k) for k in
                     ("total_decisions", "decisions", "gate_checks", "gate_expansions", "trap_unfolds", "roots", "leaves", "unique_leaves", "model_calls", "fallback_decisions", "terminal_decisions")}})
    wcsv(res / "bw_gate_compute_table.csv", cost)
    pairs = (("GATED_MG", "LOOK2_MG"), ("GATED_MG", "MG_C3"), ("GATED_REL", "REL_LOOK2"), ("GATED_REL", "GOAL_REL"), ("GATED_REL", "GATED_MG"), ("LOOK2_HADD", "LOOK2_MG"), ("LOOK2_HADD", "MG_C3"), ("REL_LOOK2", "LOOK2_MG"),
             ("GATED_REL", "GOAL_DENSE"), ("GATED_MG", "GATED_REL"))
    prs = []
    for a, b in pairs:
        if a in eps and b in eps:
            for ln, pr in layers.items():
                ids = {cid for cid in eps[a] if pr(meta[cid]) and cid in eps[b]}
                if not ids:
                    continue
                common = [i for i in ids if eps[a][i]["success"] and eps[b][i]["success"]]
                diff = [eps[a][i]["steps"] - eps[b][i]["steps"] for i in common]
                better, worse = sum(x < 0 for x in diff), sum(x > 0 for x in diff)
                for field, nm in (("success", "S"), ("decision_perfect", "D")):
                    pc = d.EK.pair_counts({i: eps[a][i] for i in ids}, {i: eps[b][i] for i in ids}, field)
                    prs.append({"first": a, "second": b, "layer": ln, "metric": nm, **pc, "common_success_n": len(common), "first_cheaper": better, "first_costlier": worse,
                                "common_mean_steps_first_minus_second": round(stats.mean(diff), 3) if diff else None, "sign_test_p": round(RP.sign_test_p(better, worse), 6),
                                "failure_penalised_ratio_first": round(stats.mean(d.EK.penalty_ratio(eps[a][i]) for i in ids), 4), "failure_penalised_ratio_second": round(stats.mean(d.EK.penalty_ratio(eps[b][i]) for i in ids), 4)})
    wcsv(res / "bw_paired_comparisons.csv", prs)
    pl = []
    for n in BW_NEW:
        if n not in eps:
            continue
        decs = [(e, x, p) for e in eps[n].values() for x, p in zip(e["decisions"], e.get("pipeline", []))]
        opt = [(x, p) for _e, x, p in decs]
        row = {"condition": n, "decisions": len(opt), "c3_changed_raw": sum(1 for x, p in opt if p["c3"] and p["c3"] != p["raw"]), "final_differs_from_c3": sum(1 for x, p in opt if p["c3"] and p["final"] != p["c3"]),
               "raw_optimal": sum(x["raw_is_optimal"] for x, p in opt), "c3_optimal": sum(1 for x, p in opt if p["c3"] in x["optimal_actions"]), "final_optimal": sum(x["selected_is_optimal"] for x, p in opt),
               "raw_opt_to_final_not": sum(1 for x, p in opt if x["raw_is_optimal"] and not x["selected_is_optimal"]),
               "raw_opt_to_final_not_raw_survives_c3": sum(1 for x, p in opt if x["raw_is_optimal"] and not x["selected_is_optimal"] and p["c3"] == p["raw"]),
               "raw_not_to_final_opt": sum(1 for x, p in opt if not x["raw_is_optimal"] and x["selected_is_optimal"]), "c3_opt_to_final_not": sum(1 for x, p in opt if p["c3"] in x["optimal_actions"] and not x["selected_is_optimal"]),
               "c3_not_to_final_opt": sum(1 for x, p in opt if p["c3"] and p["c3"] not in x["optimal_actions"] and x["selected_is_optimal"]), "gate_reasons": dict(Counter(p["gate"] for x, p in opt if p.get("gate")))}
        pl.append(row)
    wcsv(res / "bw_raw_c3_final_actions.csv", pl)
    wj(res / "bw_reused_row_identity.json", ident)
    return [res / "bw_by_condition_and_layer.csv", res / "bw_gate_compute_table.csv", res / "bw_paired_comparisons.csv", res / "bw_raw_c3_final_actions.csv"]


def cmd_docs(rr):
    """Plan 11.3 artifacts that are compositions of existing files: source_manifest, data roles, label coverage, training and selection."""
    rr = Path(rr)
    cfg = runtime(rr)
    man, exact = load_man(rr), load_exact(rr)
    ident = rj(rr / "assets" / "public_domain_identity.json")
    import platform
    import torch
    src = {"card": CARD, "plan": PLAN_REL, "plan_sha256": sha_file(ROOT / PLAN_REL), "base_commit": BASE_COMMIT, "branch": BRANCH, "head_at_init": cfg["head"], "public_domain_sources": ident,
           "user_review_materials": "the review report and the two audit scripts named in plan section 1 were read when the plan was written; they are not part of this repository and were not re-executed",
           "software": {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.node()}, "run_root": str(rr)}
    wj(rr / "source_manifest.json", src)
    roles = {"train": "gradients and exact labels only (96 problems, 3-5 crates, one tower of height 2-3)", "dev": "checkpoint selection only (24 problems, same structure)", "struct": "sealed structure test (4 crates 2+2, 5 crates 3+2; 32 problems), opened after lock",
             "joint": "sealed joint extrapolation test (8 cells x 16 problems: 6/8 crates, 1/2 towers, height 2/4), opened after lock", "ipc": "Track A: the 22 public IPC Depots problems, unchanged, opened after lock",
             "generation": {k: v for k, v in man["generation_report"].items()}, "isomorphism_dedup": "problems across ALL custom splits are pairwise non-isomorphic (typed objects, init and goal under renaming)",
             "files": {s: [{"case_id": c["case_id"], "sha256": c["sha256"], "file": os.path.relpath(c["file"], rr)} for c in man[s]] for s in ("train", "dev", "struct", "joint", "ipc")}, "domains": man["domains"]}
    wj(rr / "data_roles_and_split_manifest.json", roles)
    cov = {"cases": len(exact), "exact_ok": sum(1 for v in exact.values() if v["status"] == "OK"), "unknown_too_big": [k for k, v in exact.items() if v["status"] != "OK"], "per_family": {}}
    for fam in ("train", "dev", "struct", "joint"):
        ids = [c["case_id"] for c in man[fam] if c["case_id"] in exact and exact[c["case_id"]]["status"] == "OK"]
        vs = [exact[i] for i in ids]
        if not vs:
            continue
        lab = sum(len(v["rank_labels"]) for v in vs)
        succ = sum(len(d) for v in vs for d in v["rank_labels"].values())
        dead = sum(1 for v in vs for d in v["rank_labels"].values() for x in d.values() if x < 0)
        cov["per_family"][fam] = {"problems": len(ids), "optimal_length_mean": round(statistics.mean(v["optimal_length"] for v in vs), 2), "optimal_length_min": min(v["optimal_length"] for v in vs), "optimal_length_max": max(v["optimal_length"] for v in vs),
                                  "labelled_states": lab, "successor_cost_labels": succ, "dead_end_successor_labels": dead, "trajectories": sum(len(v["trajectories"]) for v in vs),
                                  "requires_goal_destruction": dict(Counter(str(v["requires_goal_destruction"]) for v in vs)), "reachable_states_mean": round(statistics.mean(v["n_states"] for v in vs)),
                                  "mean_optimal_first_actions": round(statistics.mean(v["n_optimal_first_actions"] for v in vs), 2), "label_seconds": round(sum(v["seconds"] for v in vs), 1)}
    cov["note"] = "UNKNOWN is a label state of its own: problems whose complete state space exceeded the budget have no exact label (8-crate joint and IPC problems); they are never treated as dead ends or as destruction-free"
    wj(rr / "training_label_coverage.json", cov)
    ts = {"budget_frozen_in_registration": rj(rr / "plan" / "registration.json")["frozen_design"]["budget"], "models": {}}
    for m in MODES:
        p = rr / "runs" / m / "training_accounting.json"
        s = rr / "runs" / m / "selection.json"
        if p.is_file():
            a = rj(p)
            ts["models"][MODEL_NAME[m]] = {"accounting": {k: a[k] for k in ("optimizer_steps", "decision_samples_shown", "wall_seconds", "attempts", "nan_events", "parameters_trainable", "parameters_new_module", "new_module_param_change_l2", "batches_per_epoch", "trajectories")},
                                           "checkpoints": {u: {"sha256": c["sha256"], "bytes": c["bytes"]} for u, c in a["checkpoints"].items()}, "selection": rj(s) if s.is_file() else None}
    wj(rr / "training_and_selection.json", ts)
    return [rr / n for n in ("source_manifest.json", "data_roles_and_split_manifest.json", "training_label_coverage.json", "training_and_selection.json")]


def cmd_final_receipt(rr, optional_status):
    rr = Path(rr)
    statuses = {}
    for p in sorted((rr / "receipts").glob("*.json")):
        r = rj(p)
        statuses[r.get("task", p.stem)] = r.get("status")
    sch = rj(rr / "scheduler_receipt.json") if (rr / "scheduler_receipt.json").is_file() else None
    out = {"card": CARD, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "task_statuses": statuses, "scheduler": sch, "new_blocksworld_trainings": 0, "new_internal_depots_trainings": 3,
           "external": {"LAMA": "run (anytime and first-plan, uniform wall budget)", "A*+LM-cut": "run (reference lengths and optimality labels, 300 s)", "WL-GOOSE": "run (author code, classic.toml, one fit per domain encoding on the same 96 training problems)",
                        "AIW-AD_or_KR2023": optional_status},
           "no_pt_committed": True, "no_pr_merge_or_force_push": True}
    p = rr / "receipts" / "final_receipt.json"
    wj(p, out)
    return p


def cmd_report(rr):
    from cp_disr.pddl import report as RP
    rows, probs, recs = RP.build(rr)
    outs = [Path(rr) / "results" / n for n in ("by_problem.csv", "by_structure.csv", "paired_costs.csv", "completion_by_time.csv", "compute_costs.csv", "training_and_label_costs.csv", "raw_c3_final_actions.csv")]
    try:
        outs += cmd_bw_tables(rr)
    except FileNotFoundError as e:
        print("[report] Blocksworld tables incomplete:", e)
    p = Path(rr) / "results" / "final_summary_numbers.md"
    p.write_text(RP.numbers_md(rr, rows), encoding="utf-8")
    outs += cmd_docs(rr)
    finish(rr, "report", "DONE", outs + [p])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run-root")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--authorization-text-file")
    ap.add_argument("--model")
    ap.add_argument("--exec")
    ap.add_argument("--set")
    ap.add_argument("--condition")
    ap.add_argument("--track")
    a = ap.parse_args()
    if a.cmd == "init":
        return cmd_init([int(x) for x in a.gpus.split(",")], Path(a.authorization_text_file).read_text(encoding="utf-8"))
    rr = a.run_root
    if a.cmd == "run-all":
        from cp_disr.pddl.sched import run_all
        return run_all(rr, cpu_slots=2)
    fn = {"prep_ext": lambda: cmd_prep_ext(rr), "gen_data": lambda: cmd_gen_data(rr), "labels": lambda: cmd_labels(rr), "ref": lambda: cmd_ref(rr, a.set), "goose_fit": lambda: cmd_goose_fit(rr, a.track),
          "train": lambda: cmd_train(rr, a.model), "devsel": lambda: cmd_devsel(rr, a.model), "lock": lambda: cmd_lock(rr), "eval": lambda: cmd_eval(rr, a.model, a.exec, a.set), "goose_eval": lambda: cmd_goose_eval(rr, a.set),
          "docs": lambda: cmd_docs(rr), "final_receipt": lambda: cmd_final_receipt(rr, a.condition or "not run"), "bw_tau": lambda: cmd_bw_tau(rr), "bw_eval": lambda: cmd_bw_eval(rr, a.condition), "bw_probes": lambda: cmd_bw_probes(rr), "gate_decision": lambda: cmd_gate_decision(rr), "report": lambda: cmd_report(rr)}[a.cmd]
    r = fn()
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
