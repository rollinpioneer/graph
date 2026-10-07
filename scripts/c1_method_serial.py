#!/usr/bin/env python
"""C1-BW-METHOD-SERIAL-SUITE-V3 stage handlers and suite driver.

    init       --gpu N --authorization-text-file F    # resolve xushijie3 paths, run root, identity, manifest (run once by the operator)
    run-all    --run-root R                            # blocking serial driver (one instance only)
    status     --run-root R
    <stage>    --run-root R [--condition C]           # fixtures lookahead_dev event_labels cal_prep train devsel lock_selected confirm_prepare confirm_eval report
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


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


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


def D():
    """Lazy heavy imports (module namespace)."""
    import torch
    from cp_disr.blocksworld import a04p_controls as AC, eval_a03 as E, gp_attribution as A, goal_progress as GP, imitation as I, planner as P, scale_order as SO, scorer_control as SC
    from cp_disr.blocksworld import splits as SP, state as S, train as T
    from cp_disr.blocksworld.method_serial import common as C, evalkit as EK, labels as LB, lookahead as LA, model as MD, trainer as TR
    from cp_disr.rl import set_suite_half_life
    from cp_disr.torch_rl import load_checkpoint, save_checkpoint
    from types import SimpleNamespace
    return SimpleNamespace(**{k: v for k, v in locals().items() if k != "SimpleNamespace"})


# ------------------------------------------------------------------------------------------------ init
CORE_SOURCES = ("src/cp_disr/blocksworld/method_serial/model.py", "src/cp_disr/blocksworld/method_serial/trainer.py", "src/cp_disr/blocksworld/method_serial/labels.py", "src/cp_disr/blocksworld/method_serial/heads_math.py",
                "src/cp_disr/blocksworld/method_serial/lookahead.py", "src/cp_disr/blocksworld/method_serial/evalkit.py", "src/cp_disr/blocksworld/method_serial/common.py", "scripts/c1_method_serial.py")


def tasks_for(rr, py, env, hashes):
    S_ = str(ROOT / "scripts" / "c1_method_serial.py")
    t = []

    def add(tid, stage, requires=(), group=None, cond=None, training=None, fail_scope=None):
        argv = [py, S_, stage, "--run-root", str(rr)] + (["--condition", cond] if cond else [])
        d = {"id": tid, "argv": argv, "receipt": str(rr / "receipts" / ("%s.json" % tid)), "input_hashes": hashes, "requires": list(requires), "group": group or stage}
        if training:
            d["training_run_id"] = training
        if fail_scope:
            d["failure_scope"] = fail_scope
        t.append(d)
    add("fixtures", "fixtures", fail_scope="global")
    add("lookahead_dev", "lookahead_dev", ["fixtures"], "lookahead")
    add("event_labels", "event_labels", ["fixtures"], "event")
    for c in ("SG_BASE", "SG_FACT", "SG_JOINT"):
        add("train_" + c, "train", ["event_labels"], "event", c, "R-C1-SERIAL-%s-0" % c.replace("_", "-"))
        add("devsel_" + c, "devsel", ["train_" + c], "event", c)
    t.append({"id": "order_cal_rec", "kind": "order_cal_rec", "argv": ["true"], "receipt": str(rr / "receipts" / "order_cal_rec.json"), "input_hashes": hashes, "group": "order"})
    add("cal_prep", "cal_prep", ["fixtures"], "calibration")
    for c in ("CAL_ABS", "CAL_REL"):
        add("train_" + c, "train", ["cal_prep"], "calibration", c, "R-C1-SERIAL-%s-0" % c.replace("_", "-"))
        add("devsel_" + c, "devsel", ["train_" + c], "calibration", c)
    for c in ("REC_SELF", "REC_REL"):
        add("train_" + c, "train", ["fixtures"], "recurrent", c, "R-C1-SERIAL-%s-0" % c.replace("_", "-"))
        add("devsel_" + c, "devsel", ["train_" + c], "recurrent", c)
    for c in ("GOAL_DENSE", "GOAL_REL"):
        add("train_" + c, "train", ["fixtures"], "goal_attention", c, "R-C1-SERIAL-%s-0" % c.replace("_", "-"))
        add("devsel_" + c, "devsel", ["train_" + c], "goal_attention", c)
    add("lock_selected", "lock_selected", [], "lock")
    add("confirm_prepare", "confirm_prepare", ["lock_selected"], "confirm")
    return t


def conf_tasks(rr, py, hashes):
    S_ = str(ROOT / "scripts" / "c1_method_serial.py")
    from cp_disr.blocksworld.method_serial.common import CONFIRM
    out = []
    for name, (src, _how) in CONFIRM.items():
        req = ["confirm_prepare"] + (["devsel_" + src] if src else [])
        out.append({"id": "confirm_" + name, "argv": [py, S_, "confirm_eval", "--run-root", str(rr), "--condition", name], "receipt": str(rr / "receipts" / ("confirm_%s.json" % name)), "input_hashes": hashes, "requires": req, "group": "confirm"})
    out.append({"id": "report", "argv": [py, S_, "report", "--run-root", str(rr)], "receipt": str(rr / "receipts" / "report.json"), "input_hashes": {}, "requires": [], "group": "report"})
    return out


def cmd_init(gpu, auth):
    d = D()
    E, C = d.E, d.C
    if E.git(ROOT, "branch", "--show-current") != C.BRANCH:
        raise SystemExit("branch must be %s" % C.BRANCH)
    storage = C.check_storage("/home/xushijie3")
    if not os.access(storage, os.W_OK | os.X_OK):
        raise SystemExit("storage root not writable")
    repo = C.check_storage(ROOT)
    hashes = {s: C.sha_file(ROOT / s) for s in CORE_SOURCES if (ROOT / s).is_file()}
    code8 = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:8]
    parent = Path(os.environ["SERIAL_RUN_PARENT"]) if os.environ.get("SERIAL_RUN_PARENT") else repo / C.SUITE_REL
    rr = C.check_storage(parent) / ("%s_%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), code8))
    for sub in ("plan", "assets", "prep", "stages", "runs", "development", "confirmation", "results", "receipts", "driver_logs"):
        (rr / sub).mkdir(parents=True, exist_ok=True)
    (rr / "receipts" / "case_ledger.jsonl").write_text("")
    cache, tmp = storage / "cp_disr_cache", storage / "cp_disr_tmp"
    cache.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    py = str(storage / "envs" / "cpdisr" / "bin" / "python")
    assert Path(py).is_file()
    cfg = {"storage_root": str(storage), "repo_root": str(repo), "asset_root": str(repo), "run_root": str(rr), "cache_root": str(cache), "tmp_root": str(tmp), "venv_python": py, "device": "cuda:%d" % gpu, "gpu": gpu,
           "base_commit": C.BASE_COMMIT, "branch": C.BRANCH, "head": E.git(ROOT, "rev-parse", "HEAD"), "storage_owner": "xushijie3", "legacy_storage_owner": "xushijie2",
           "legacy_prefix": "/home/xushijie2/graph_cp_disr_c1_bw", "execution_authorized": True}
    C.wj(rr / "suite_runtime.json", cfg)
    # assets
    ids = {}
    for name, rel in (("A02_B2", d.E.MODELS["B2"]["path"]), ("M1_GOAL", d.SC.M1GOAL["path"]), ("D_train", d.A.GP_ROOT + "/datasets/D_train.json"), ("rank_labels", d.A.GP_ROOT + "/datasets/rank_labels.json"),
                      ("board48", d.SO.BOARD_REL), ("fresh112", d.SC.FRESH_REL), ("train_dev_split", "configs/splits/c1_bw_train_dev_v1.json"), ("state_bank", C.SO_RUN_REL + "/prep/state_bank.json")):
        p = (repo / rel).resolve()
        C.check_storage(p)
        ids[name] = {"path": rel, "resolved": str(p), "sha256": C.sha_file(p), "bytes": p.stat().st_size}
    exp = {"A02_B2": d.E.MODELS["B2"]["sha256"], "M1_GOAL": d.SC.M1GOAL["sha256"], "D_train": d.A.D_TRAIN["file_sha256"], "rank_labels": d.A.D_TRAIN["rank_labels_file_sha256"]}
    for k, v in exp.items():
        assert ids[k]["sha256"] == v, "asset identity mismatch: %s" % k
    C.wj(rr / "assets" / "asset_identity.json", ids)
    C.wj(rr / "assets" / "path_map.json", {"legacy_prefix": cfg["legacy_prefix"], "new_prefix": cfg["repo_root"], "note": "historical manifests keep their original absolute paths; relative paths resolve against repo_root; no file rewritten"})
    (rr / "plan" / "RUNBOOK.md").write_text((ROOT / C.RUNBOOK_REL).read_text(encoding="utf-8"), encoding="utf-8")
    env = {"PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": str(gpu), "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(cache), "TORCH_HOME": str(cache / "torch")}
    tasks = tasks_for(rr, py, env, hashes)
    # insert confirmation tasks and report
    tasks += conf_tasks(rr, py, hashes)
    manifest = {"card": C.CARD, "execution_authorized": True, "authorization_source": auth, "storage_root": str(storage), "repo_root": str(repo), "run_root": str(rr), "max_new_training_runs": 9, "priority_receipt": str(rr / "stages" / "lookahead" / "priority.json"),
                "env": env, "tasks": tasks}
    C.wj(rr / "manifest.json", manifest)
    C.wj(rr / "plan" / "registration.json", {"card": C.CARD, "runbook": C.RUNBOOK_REL, "runbook_sha256": C.sha_file(ROOT / C.RUNBOOK_REL), "base_commit": C.BASE_COMMIT, "authorization_text": auth,
                                              "authorization_text_sha256": hashlib.sha256(auth.encode()).hexdigest(), "scope": "FULL_SERIAL_5_FAMILIES_MAX_9_TRAININGS_AND_ONE_COMMON_CONFIRM", "requires_intermediate_confirmation": False,
                                              "core_source_hashes": hashes, "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"run_root": str(rr), "tasks": len(tasks), "trainings": sum(1 for t in tasks if t.get("training_run_id"))}))


# ------------------------------------------------------------------------------------------------ driver
def run_all(rr):
    from cp_disr.blocksworld.method_serial import schedule
    return schedule.safe_cli_run(Path(rr) / "manifest.json")


# ------------------------------------------------------------------------------------------------ shared helpers
def ctx(rr):
    d = D()
    c = d.C.Ctx(ROOT, rr)
    td = json.loads((ROOT / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    d.set_suite_half_life(float(td["half_life"]))
    return d, c


def policy_for(c, d, cond_name, model=None):
    """(policy, chooser, counters or None) of a confirmation / development condition name."""
    src, how = d.C.CONFIRM[cond_name]
    if how == "b2_rule":
        return d.SC.load_scorer(ROOT, "B_G1C3", c.device), d.SC.chooser_for("B_G1C3"), None
    if cond_name == "MG_C3":
        return d.SC.load_scorer(ROOT, "MG_C3", c.device), d.EK.c3_chooser(), None
    cnt = None
    if how in ("look_mg", "look_count", "look_w"):
        cnt = d.LA.Counters()
        if how == "look_count":
            m = d.MD.SerialModel(c.mg_model(), "MG")
        elif how == "look_mg":
            m = d.MD.SerialModel(c.mg_model(), "MG")
        else:
            m, _sel = c.load_selected(src)
        m.eval()
        return d.MD.SerialPolicy(m), d.LA.make_chooser(m, "COUNT" if how == "look_count" else "MG", cnt), cnt
    m, _sel = c.load_selected(src)
    steps = 4 if how == "T4" else 8 if how == "T8" else (4 if CONDS_KIND(d, src).startswith("REC") else None)
    return d.MD.SerialPolicy(m, steps), d.EK.c3_chooser(), None


def CONDS_KIND(d, src):
    return d.C.CONDS[src][0]


def reuse_eps(c, d, name, set_name):
    """Episodes of a frozen reference condition from the earlier cards (same code, same problems, same weights)."""
    so_eval = ROOT / d.C.SO_RUN_REL / "eval"
    sc_eval = ROOT / d.SO.SC_RUN_REL / "eval"
    if set_name == "board48":
        return d.EK.jl(so_eval / name / "board48.jsonl")
    if set_name == "fresh112":
        return d.EK.jl(sc_eval / name / "episodes.jsonl")
    raise KeyError(set_name)


# ------------------------------------------------------------------------------------------------ stage: fixtures
def cmd_fixtures(rr):
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_method_serial.py", "tests/test_method_serial_math.py", "tests/test_method_serial_driver.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=str(ROOT), env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
    log = Path(rr) / "prep" / "fixtures.log"
    log.write_text(r.stdout + r.stderr)
    d, c = ctx(rr)
    # data profile of the training scope (runbook 2.3)
    from cp_disr.blocksworld import goal_probe as GPB
    cases, _h = d.I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    prof = Counter()
    rows = []
    for cs in cases:
        comps = [len(x) for x in GPB.goal_components(cs.problem.goal)]
        nt = [x for x in comps if x >= 2]
        ih = max(len(t) for t in d.S.towers(cs.problem.init))
        rows.append((cs.problem.n, len(nt), max(comps), ih))
        prof[("n", cs.problem.n)] += 1
        prof[("goal_height", max(comps))] += 1
        prof[("nontrivial_goal_towers", len(nt))] += 1
        prof[("initial_height", ih)] += 1
    scope = {"train_cases": len(cases), "counts": {"%s=%s" % k: v for k, v in sorted(prof.items())}, "train_object_count": sorted({r[0] for r in rows}), "train_goal_tower_height": sorted({r[2] for r in rows}),
             "train_nontrivial_goal_towers": sorted({r[1] for r in rows}), "pad_training_used": False, "height4_training_used": False, "height_seen_set": [2, 3], "height_unseen": [4]}
    d.C.wj(Path(rr) / "prep" / "training_scope.json", scope)
    ok = r.returncode == 0 and scope["train_goal_tower_height"] in ([2, 3], [3], [2]) and scope["train_nontrivial_goal_towers"] == [1]
    c.receipt("fixtures", "DONE" if ok else "GLOBAL_INTEGRITY_FAILURE", [log, Path(rr) / "prep" / "training_scope.json"], pytest_returncode=r.returncode, pytest_tail=r.stdout[-400:])
    return 0 if ok else 3


# ------------------------------------------------------------------------------------------------ stage: lookahead (zero training)
def cmd_lookahead(rr):
    d, c = ctx(rr)
    mg = d.MD.SerialModel(c.mg_model(), "MG")
    mg.eval()
    out_dir = Path(rr) / "stages" / "lookahead"
    sets = {"board48": d.EK.load_cases(ROOT, d.SO.BOARD_REL), "fresh112": d.EK.load_cases(ROOT, d.SC.FRESH_REL)}
    summ, outs, costs = {}, [], []
    for sname, (groups, meta) in sets.items():
        for leaf, label in (("MG", "LOOK2_MG"), ("COUNT", "LOOK2_COUNT")):
            cnt = d.LA.Counters()
            ch = d.LA.make_chooser(mg, leaf, cnt)
            path = out_dir / ("%s_%s.jsonl" % (sname, label))
            d.EK.run_set(Path(rr), "%s@%s" % (label, sname), [(k, groups[k]) for k in sorted(groups)], d.MD.SerialPolicy(mg), ch, path)
            outs.append(path)
            eps = d.EK.jl(path)
            summ[(sname, label)] = d.EK.summarize(eps, meta)
            costs.append({"set": sname, "condition": label, **cnt.as_dict()})
        for ref in ("MG_C3", "B_G1C3"):
            summ[(sname, ref)] = d.EK.summarize(reuse_eps(c, d, ref, sname), meta)
    # shared-tree diagnostic on the common state bank (visited = {s}, identical roots for both leaf scorers)
    bank = d.EK.bank_records(ROOT / d.C.SO_RUN_REL)
    cases, bmeta = d.EK.bank_cases(ROOT)
    rows = []
    for r in bank:
        case = cases[r["case_id"]]
        from cp_disr.blocksworld.environment import BwEpisode
        ep = BwEpisode(case)
        ep.state = tuple(r["state"])
        snap = ep.snapshot()
        ids = snap.candidate_ids
        roots = [ids[i] for i, m in enumerate(snap.mask) if m and d.S.apply(ep.state, ep._action_of[ids[i]]) != ep.state]
        row = {"bank_id": r["bank_id"], "cell": bmeta[r["case_id"]]["cell"], "h": bmeta[r["case_id"]]["max_tower_height"], "n_roots": len(roots)}
        for leaf, label in (("MG", "LOOK2_MG"), ("COUNT", "LOOK2_COUNT")):
            aid, info = d.LA.choose_root(mg, leaf, snap.template, case.problem, ep.state, roots, ep._action_of, {ep.state})
            row[label + "_root"] = aid
            row[label + "_optimal_root"] = (aid in r["optimal"]) if aid else None
            row[label + "_local_excess"] = (1 + r["dist"][aid] - r["L"]) if aid else None
            row[label + "_terminal"] = info["terminal"]
            row["n_leaves"] = info["n_leaves"]
        rows.append(row)
    wcsv(Path(rr) / "results" / "lookahead_leaf_diagnostics.csv", rows)
    wcsv(Path(rr) / "results" / "lookahead_costs.csv", costs)
    root_mg = [r["LOOK2_MG_optimal_root"] for r in rows if r["LOOK2_MG_optimal_root"] is not None]
    root_ct = [r["LOOK2_COUNT_optimal_root"] for r in rows if r["LOOK2_COUNT_optimal_root"] is not None]
    rate = lambda xs: (sum(xs) / len(xs)) if xs else None
    ev = {}
    weak = True
    for sname in sets:
        a, b = summ[(sname, "LOOK2_MG")], summ[(sname, "LOOK2_COUNT")]
        ev[sname] = {"MG_h4_success": a["h4_success"], "COUNT_h4_success": b["h4_success"], "MG_penalty": a["mean_penalty_ratio"], "COUNT_penalty": b["mean_penalty_ratio"], "MG_h4_n": a["h4_n"]}
        weak &= a["h4_success"] <= b["h4_success"] and a["mean_penalty_ratio"] >= b["mean_penalty_ratio"]
    ev["cross_root_optimal_rate"] = {"MG": rate(root_mg), "COUNT": rate(root_ct)}
    weak &= rate(root_mg) is not None and rate(root_ct) is not None and rate(root_mg) < rate(root_ct)
    prio = "CALIBRATION_THEN_RECURRENT" if weak else "RECURRENT_THEN_CALIBRATION"
    d.C.wj(out_dir / "priority.json", {"priority": prio, "rule": "plan 4.2: neural leaf not better than COUNT on h4 completions in Board48 and fresh112, penalty cost not lower, and worse cross-root choice on the shared-tree state bank",
                                        "evidence": ev, "note": "ordering aid only; not a root-cause proof or significance test"})
    table = [{"set": s, "condition": k, **{kk: (json.dumps(vv) if isinstance(vv, (dict, list)) else vv) for kk, vv in v.items()}} for (s, k), v in summ.items()]
    wcsv(Path(rr) / "results" / "lookahead_panel_summary.csv", table)
    c.receipt("lookahead_dev", "DONE", outs + [out_dir / "priority.json"], priority=prio)
    return 0


# ------------------------------------------------------------------------------------------------ stages: labels
def _train_data(d):
    G = ROOT / d.A.GP_ROOT / "datasets"
    return json.loads((G / "D_train.json").read_text()), json.loads((G / "rank_labels.json").read_text())


def cmd_event_labels(rr):
    d, c = ctx(rr)
    cases, _h = d.I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    d_train, _rl = _train_data(d)
    labels, wall = d.LB.build_event_labels(cases, d_train, 48)
    info = d.LB.event_information(labels, d_train)
    info["label_wall_seconds"] = wall
    unknown = sum(1 for v in labels.values() if not v["complete"])
    d.C.wj(Path(rr) / "prep" / "event_labels.json", labels)
    d.C.wj(Path(rr) / "prep" / "event_information.json", info)
    f = info["unique_goal_state_pairs"]["fractions_of_complete"]
    uninformative = (f["event_information_Q_ne_U0"] == 0.0 and f["pair_information_Y_ne_QxA"] == 0.0)
    c.receipt("event_labels", "UNINFORMATIVE_LABELS" if uninformative else "DONE", [Path(rr) / "prep" / "event_labels.json", Path(rr) / "prep" / "event_information.json"], unknown_labels=unknown)
    return 0


def cmd_cal_prep(rr):
    d, c = ctx(rr)
    cases, _h = d.I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    d_train, rl = _train_data(d)
    cal = d.LB.build_calibration_targets(cases, d_train, 48)
    scale = d.LB.calibration_scale(cal, rl, d_train)
    mg = d.MD.SerialModel(c.mg_model(), "MG")
    cc = d.TR.calibration_constants(mg, cases, rl)
    cc["scale_c"] = scale
    cc["endpoint_coverage"] = dict(Counter(len(v["endpoints"]) for v in cal.values()))
    d.C.wj(Path(rr) / "prep" / "calibration_targets.json", cal)
    d.C.wj(Path(rr) / "prep" / "calibration_constants.json", cc)
    c.receipt("cal_prep", "DONE", [Path(rr) / "prep" / "calibration_targets.json", Path(rr) / "prep" / "calibration_constants.json"], constants=cc)
    return 0


# ------------------------------------------------------------------------------------------------ stage: train
def _rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _atomic_torch_save(obj, path):
    import torch
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def cmd_train(rr, cond):
    import torch
    d, c = ctx(rr)
    kind, loss, _fam = d.C.CONDS[cond]
    out = Path(rr) / "runs" / cond
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    acct_path = out / "training_accounting.json"
    if acct_path.is_file():
        c.receipt("train_" + cond, "DONE", [acct_path], note="already complete")
        return 0
    cases, _h = d.I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    d_train, rl = _train_data(d)
    if d.C.SMOKE:
        d_train = d_train[:64]
    ev = cal = None
    scale = 1.0
    if loss in ("BASE", "FACT", "JOINT"):
        ev = json.loads((Path(rr) / "prep" / "event_labels.json").read_text())
    model = c.build(kind)
    if loss in ("ABS", "REL"):
        cal = json.loads((Path(rr) / "prep" / "calibration_targets.json").read_text())
        cc = json.loads((Path(rr) / "prep" / "calibration_constants.json").read_text())
        model.set_calibration(cc["a0"], cc["c0"])
        scale = cc["scale_c"]
    trainer = d.TR.SerialTrainer(model, loss, cases, rl, c.device, event_labels=ev, cal_targets=cal, cal_scale=scale)
    init_new = [p.detach().clone().cpu() for p in model.new_params() + model.eta_params()]
    rows, start_epoch, attempts, resumed_steps = [], 0, 1, 0
    resume = out / "resume_last.pt"
    t_prev = 0.0
    if resume.is_file():
        st = torch.load(resume, map_location=c.device, weights_only=False)
        if st["attempts"] >= 3:
            c.receipt("train_" + cond, "TECHNICAL_INCOMPLETE", [], reason="resume attempts exhausted")
            return 1
        model.load_state_dict(st["model"])
        trainer.optimizer.load_state_dict(st["optimizer"])
        rows, start_epoch, attempts = st["rows"], st["epoch"], st["attempts"] + 1
        trainer.steps, trainer.nan_events, trainer.eta_clamped = st["steps"], st["nan"], st["eta_clamped"]
        init_new = st["init_new"]
        resumed_steps = st["steps"]
        t_prev = st["wall"]
    t0 = time.time()
    ckpts = {}
    for name in ("epoch020", "epoch040", "epoch060", "epoch080", "final"):
        p = out / "checkpoints" / ("%s.pt" % name)
        if p.is_file():
            ckpts[name] = {"path": _rel(p), "sha256": d.C.sha_file(p), "bytes": p.stat().st_size}
    for ep in range(start_epoch, d.C.EPOCHS):
        row = trainer.epoch(d_train, 0 * 100003 + ep)
        row.update({"epoch": ep + 1, "optimizer_steps": trainer.steps, "wall_seconds": round(t_prev + time.time() - t0, 1)})
        rows.append(row)
        if ep == 0 or (ep + 1) % 10 == 0:
            print("[serial] %s epoch %d %s" % (cond, ep + 1, {k: round(v, 5) for k, v in row.items() if isinstance(v, float)}), flush=True)
        wcsv(out / "epochs.csv", rows)
        if ep + 1 in d.C.CKPT_EPOCHS:
            name = "final" if ep + 1 == d.C.EPOCHS else "epoch%03d" % (ep + 1)
            p = out / "checkpoints" / ("%s.pt" % name)
            d.save_checkpoint(p, model, trainer.optimizer, {"condition": cond, "epoch": ep + 1, "optimizer_steps": trainer.steps})
            ckpts[name] = {"path": _rel(p), "sha256": d.C.sha_file(p), "bytes": p.stat().st_size}
        _atomic_torch_save({"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(), "rows": rows, "epoch": ep + 1, "attempts": attempts, "steps": trainer.steps, "nan": trainer.nan_events,
                            "eta_clamped": trainer.eta_clamped, "init_new": init_new, "wall": t_prev + time.time() - t0}, resume)
    new_now = [p.detach().cpu() for p in model.new_params() + model.eta_params()]
    delta = float(sum(((a - b) ** 2).sum() for a, b in zip(new_now, init_new)) ** 0.5) if new_now else 0.0
    g = [r["new_module_grad_norm"] for r in rows]
    acct = {"run_id": "R-C1-SERIAL-%s-0" % cond.replace("_", "-"), "condition": cond, "kind": kind, "loss": loss, "epochs": d.C.EPOCHS, "optimizer_steps": trainer.steps, "effective_optimizer_steps": trainer.steps,
            "resumed_from_step": resumed_steps, "attempts": attempts, "decision_samples_shown": sum(r["decisions"] for r in rows), "nan_events": trainer.nan_events, "wall_seconds": round(t_prev + time.time() - t0, 1),
            "mean_epoch_wall_seconds": round((t_prev + time.time() - t0) / d.C.EPOCHS, 2), "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in trainer.params), "parameters_new_module": sum(p.numel() for p in model.new_params()) + sum(p.numel() for p in model.eta_params()),
            "new_module_param_change_l2": delta, "new_module_grad_norm_first_epoch": g[0], "new_module_grad_norm_last_epoch": g[-1], "new_module_grad_norm_max": max(g), "eta_clamped_total": trainer.eta_clamped,
            "checkpoints": ckpts, "final_epoch": rows[-1], "init": "M1-GOAL final (SHA %s)" % d.SC.M1GOAL["sha256"], "new_module_seed": d.MD.NEW_SEED, "same_start_for_all_conditions": True,
            "evaluation_checkpoint": "selected by Board48 key among epoch 20/40/60/80/100 (devsel)"}
    d.C.wj(acct_path, acct)
    c.receipt("train_" + cond, "DONE", [acct_path, out / "checkpoints" / "final.pt"], wall=acct["wall_seconds"])
    return 0


# ------------------------------------------------------------------------------------------------ stage: dev selection and support
def _event_diag(d, c, model, bank, cases, rr):
    """Q top-1 hit, top joint pair in Y, P_Y / P_F / P_A on the common bank states (labels only for scoring)."""
    import torch
    from cp_disr.blocksworld import imitation as I
    solver = d.P.Solver()
    rows = []
    for r in bank:
        case = cases[r["case_id"]]
        lab = d.LB.event_label(case.problem, tuple(r["state"]), solver)
        if not lab["complete"]:
            rows.append({"bank_id": r["bank_id"], "complete": False})
            continue
        snap = I.snapshot_at(case, tuple(r["state"]), 0)
        jo, pending, legal = model.eval_event(snap)
        gids = [g.fact_id for g in snap.template.goals]
        ids = snap.candidate_ids
        Y = {(g, a) for g, a in lab["Y"]}
        Q = {g for g, _ in Y}
        A_ = set(lab["A"])
        lq = jo.log_goal[0]
        lj = jo.log_joint[0]
        top_goal = gids[int(lq.argmax())]
        flat = lj.masked_fill(~legal[0][None, :].expand_as(lj), -torch.inf)
        gi, ai = divmod(int(flat.flatten().argmax()), lj.shape[1])
        py = sum(float(torch.exp(lj[gids.index(g), ids.index(a)])) for g, a in Y)
        pf = sum(float(torch.exp(lj[gids.index(g), ids.index(a)])) for g in Q for a in A_)
        pa = sum(float(torch.exp(jo.log_action[0, ids.index(a)])) for a in A_)
        pair_info = Y != {(g, a) for g in Q for a in A_}
        rows.append({"bank_id": r["bank_id"], "complete": True, "Q_top1_hit": top_goal in Q, "joint_top_pair_in_Y": (gids[gi], ids[ai]) in Y, "P_Y": py, "P_F": pf, "P_A": pa, "pair_informative": pair_info,
                     "singleton_event": len(Q) == 1, "event_kind": "OnTable" if any(":OnTable:" in g for g in Q) and not any(":On:" in g for g in Q) else ("On" if not any(":OnTable:" in g for g in Q) else "mixed")})
    return rows


def cmd_devsel(rr, cond):
    d, c = ctx(rr)
    kind, loss, fam = d.C.CONDS[cond]
    dev = Path(rr) / "development" / cond
    dev.mkdir(parents=True, exist_ok=True)
    groups, meta = d.EK.load_cases(ROOT, d.SO.BOARD_REL)
    gl = [(k, groups[k]) for k in sorted(groups)]
    steps = 4 if kind.startswith("REC") else None
    rows, outs = {}, []
    solver = d.P.Solver()
    for ep in d.C.CKPT_EPOCHS:
        model, info = c.load_epoch(cond, ep)
        pol = d.MD.SerialPolicy(model, steps)
        path = dev / ("board48_e%03d.jsonl" % ep)
        d.EK.run_set(Path(rr), "%s@e%d@board48" % (cond, ep), gl, pol, d.EK.c3_chooser(), path, solver)
        rows[ep] = d.EK.summarize(d.EK.jl(path), meta)
        outs.append(path)
    best = d.EK.choose_checkpoint(rows)
    name = "final" if best == d.C.EPOCHS else "epoch%03d" % best
    acct = d.C.rj(Path(rr) / "runs" / cond / "training_accounting.json")
    sel = {"condition": cond, "selected_epoch": best, "selected_checkpoint": acct["checkpoints"][name], "selection_key": "success, h4 decision-perfect, -penalised length ratio, decision-perfect, earlier epoch",
           "board48_by_epoch": {str(k): v for k, v in rows.items()}, "support_can_reselect": False}
    d.C.wj(Path(rr) / "runs" / cond / "checkpoint_selection.json", sel)
    wcsv(dev / "all_checkpoint_scores.csv", [{"condition": cond, "epoch": k, **{a: (json.dumps(b) if isinstance(b, dict) else b) for a, b in v.items()}, "selected": k == best} for k, v in rows.items()])
    # support evaluations of the selected checkpoint only
    model, _sel = c.load_selected(cond)
    pol = d.MD.SerialPolicy(model, steps)
    g112, m112 = d.EK.load_cases(ROOT, d.SC.FRESH_REL)
    p112 = dev / "fresh112.jsonl"
    d.EK.run_set(Path(rr), "%s@sel@fresh112" % cond, [(k, g112[k]) for k in sorted(g112)], pol, d.EK.c3_chooser(), p112, solver)
    dg, td = d.EK.load_dev36(ROOT)
    pdv = dev / "dev36.jsonl"
    d.EK.run_set(Path(rr), "%s@sel@dev36" % cond, dg, pol, d.EK.raw_chooser(), pdv, solver)
    outs += [p112, pdv]
    bank = d.EK.bank_records(ROOT / d.C.SO_RUN_REL)
    bcases, bmeta = d.EK.bank_cases(ROOT)
    brows = d.EK.bank_eval(pol, bank, bcases)
    wcsv(dev / "state_bank.csv", brows)
    mgrows = d.EK.bank_eval(d.SC.load_scorer(ROOT, "MG_C3", c.device), bank, bcases)
    d.C.wj(dev / "state_bank_summary.json", d.EK.bank_summary(brows, mgrows))
    extra = {}
    if kind.startswith("REC"):
        m8 = d.MD.SerialPolicy(model, 8)
        for sname, g, mt in (("board48", gl, meta), ("fresh112", [(k, g112[k]) for k in sorted(g112)], m112)):
            pth = dev / ("%s_T8.jsonl" % sname)
            d.EK.run_set(Path(rr), "%s@sel@T8@%s" % (cond, sname), g, m8, d.EK.c3_chooser(), pth, solver)
            outs.append(pth)
        wcsv(dev / "state_bank_T8.csv", d.EK.bank_eval(m8, bank, bcases))
    if kind == "CAL":
        for sname, g in (("board48", gl), ("fresh112", [(k, g112[k]) for k in sorted(g112)])):
            cnt = d.LA.Counters()
            pth = dev / ("%s_LOOK2.jsonl" % sname)
            d.EK.run_set(Path(rr), "%s@sel@LOOK2@%s" % (cond, sname), g, pol, d.LA.make_chooser(model, "MG", cnt), pth, solver)
            outs.append(pth)
            extra["look2_counters_" + sname] = cnt.as_dict()
        extra["beta"] = float(model.log_beta.exp())
        extra["a0"], extra["c0"] = float(model.a0), float(model.c0)
    if kind == "EVENT":
        erows = _event_diag(d, c, model, bank, bcases, rr)
        wcsv(dev / "event_bank_diagnostics.csv", erows)
    if kind.startswith("GOAL"):
        dens = {}
        from cp_disr.blocksworld import imitation as I
        for sname, g in (("board48", gl), ("fresh112", [(k, g112[k]) for k in sorted(g112)])):
            vals = []
            for cell, cs in g:
                for cs_ in cs:
                    vals.append(model.mask_density(I.snapshot_at(cs_, cs_.problem.init, 0)))
            dens[sname] = round(sum(vals) / len(vals), 4)
        cases, _h = d.I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
        tv = [model.mask_density(I.snapshot_at(cs_, cs_.problem.init, 0)) for cs_ in cases]
        dens["train_initial_states"] = round(sum(tv) / len(tv), 4)
        extra["mask_density_nonself_initial_states"] = dens
    d.C.wj(dev / "support_extra.json", extra)
    outs += [dev / "all_checkpoint_scores.csv", Path(rr) / "runs" / cond / "checkpoint_selection.json", dev / "state_bank.csv"]
    c.receipt("devsel_" + cond, "DONE", outs, selected_epoch=best)
    return 0


# ------------------------------------------------------------------------------------------------ stage: lock and confirmation
def _lock_candidates(rr):
    """Developing-set information used to choose ONE system candidate (recorded before the confirmation set exists)."""
    rows = []
    for p in sorted((Path(rr) / "runs").glob("*/checkpoint_selection.json")):
        sel = json.loads(p.read_text())
        r = sel["board48_by_epoch"][str(sel["selected_epoch"])]
        rows.append({"condition": sel["condition"], "epoch": sel["selected_epoch"], "board48_success": r["success"], "board48_h4_perfect": r["h4_perfect"], "board48_penalty": r["mean_penalty_ratio"], "board48_perfect": r["perfect"]})
    return rows


def cmd_lock(rr):
    d, c = ctx(rr)
    rows = _lock_candidates(rr)
    best = max(rows, key=lambda r: (r["board48_success"], r["board48_h4_perfect"], -r["board48_penalty"], r["board48_perfect"])) if rows else None
    prio = json.loads((Path(rr) / "stages" / "lookahead" / "priority.json").read_text()) if (Path(rr) / "stages" / "lookahead" / "priority.json").is_file() else {}
    doc = {"preselected_candidate": best["condition"] if best else "MG_C3 (no trained condition available)", "selected_by": "Board48 key only", "candidates": rows,
           "families_and_matched_controls": {"lookahead": ["LOOK2_MG vs LOOK2_COUNT vs MG_C3"], "next_event": ["SG_JOINT vs SG_FACT vs SG_BASE"], "calibration": ["CAL_REL vs CAL_ABS (C3 and LOOK2)"],
                                             "recurrent": ["REC_REL vs REC_SELF (T4 and T8)"], "goal_attention": ["GOAL_REL vs GOAL_DENSE"]},
           "frozen_references": ["MG_C3", "B_G1C3"], "lookahead_priority": prio.get("priority"), "expected_side_effects": "recorded in the final claim boundary",
           "locked_before_confirmation": True, "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    d.C.wj(Path(rr) / "development" / "preconfirmation_selection.json", doc)
    d.C.wj(Path(rr) / "results" / "selection_history.json", {"candidates": rows, "preselected": doc["preselected_candidate"]})
    c.receipt("lock_selected", "DONE", [Path(rr) / "development" / "preconfirmation_selection.json"])
    return 0


def confirm_bank(case, solver, d):
    """<= 6 states per confirmation problem: init, ~1/3, ~2/3, pre-final, up to two fixed non-optimal successors of the middle state (no model score)."""
    base = d.SO.bank_states(case, solver)
    states = [(k, s) for k, s, _i in base if k != "deviation"]
    plan = solver.one_optimal_plan(case.problem.init, case.problem.goal)
    cur = case.problem.init
    path = [cur]
    for a in plan:
        cur = d.S.apply(cur, a)
        path.append(cur)
    mid = path[len(plan) // 2]
    _l, opt = solver.optimal_actions(mid, case.problem.goal)
    optset = {d.S.action_id(case.problem.names, a) for a in opt}
    seen = {s for _k, s in states}
    k = 0
    for a in sorted(d.S.legal_actions(mid), key=lambda x: d.S.action_id(case.problem.names, x)):
        if d.S.action_id(case.problem.names, a) in optset:
            continue
        s2 = d.S.apply(mid, a)
        if s2 in seen or solver.cost_to_go(s2, case.problem.goal) in (None, 0):
            continue
        states.append(("deviation%d" % (k + 1), s2))
        seen.add(s2)
        k += 1
        if k == 2:
            break
    return states


def cmd_confirm_prepare(rr):
    d, c = ctx(rr)
    cases, _h = d.I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    d_train, _rl = _train_data(d)
    forbidden, by_source = d.SO.board_excluded(ROOT, d_train, cases)
    b48 = {x["problem_iso_hash"] for x in json.loads((ROOT / d.SO.BOARD_REL).read_text())["cases"]}
    forbidden |= b48
    by_source["board48"] = len(b48)
    start = set(forbidden)
    b = d.SP.Builder()
    out, report, rejects = [], {}, {}
    quota = 2 if d.C.SMOKE else 16
    for cell, n, k, h, heights in d.SO.BOARD_CELLS:
        rej = Counter()
        made, att = d.SO.gen_board_cell(b, forbidden, "%s:%s" % (d.C.CONFIRM_NAMESPACE + ("-SMOKE" if d.C.SMOKE else ""), cell), n, heights, quota, d.SO.BOARD_MAX_ATTEMPTS, rej)
        for i, (p, q, hh) in enumerate(made):
            out.append(d.SP.case_record("BW_c128_%s_%02d" % (cell, i), "serial_confirm", p, q, d.A._extra(p, q, **{"namespace": d.C.CONFIRM_NAMESPACE, "cell": cell, "n_cell": n, "k_cell": k, "h_cell": h, "goal_heights": sorted(heights, reverse=True)})))
        report[cell] = {"planned": quota, "built": len(made), "attempts": att, "shortfall": quota - len(made)}
        rejects[cell] = dict(rej)
    hashes = [x["problem_iso_hash"] for x in out]
    assert not (set(hashes) & start) and len(set(hashes)) == len(hashes)
    rel = str(Path(rr) / "confirmation" / "smoke_split.json") if d.C.SMOKE else "configs/splits/c1_bw_serial_confirm128_v1.json"
    d.C.wj(ROOT / rel, {"namespace": d.C.CONFIRM_NAMESPACE, "cells": report, "counts": len(out), "cases": out, "generated_after_selection_lock": True, "note": "no model or rule queried; exclusion by colour-preserving problem class"})
    solver = d.P.Solver()
    bank = []
    for x in out:
        case = d.T.case_from_json(x)
        for kind, st in confirm_bank(case, solver, d):
            r = d.SO.state_record(case, st, kind, solver)
            r.update({"bank_id": "%s#%s" % (case.case_id, kind), "cell": x["cell"]})
            bank.append(r)
    d.C.wj(Path(rr) / "confirmation" / "state_bank.json", bank)
    man = {"split": rel, "sha256": d.C.sha_file(ROOT / rel), "cases": len(out), "cells": report, "rejections": rejects, "excluded_by_source": by_source, "excluded_total": len(start), "overlap": 0, "bank_states": len(bank),
           "layers": dict(Counter("%s|%s" % (x["cell"], x["labels"]["requires_goal_destruction"]) for x in out))}
    d.C.wj(Path(rr) / "confirmation" / "manifest.json", man)
    c.receipt("confirm_prepare", "DONE", [Path(rr) / "confirmation" / "manifest.json", Path(rr) / "confirmation" / "state_bank.json", ROOT / rel], cases=len(out))
    return 0


def cmd_confirm_eval(rr, name):
    d, c = ctx(rr)
    man = d.C.rj(Path(rr) / "confirmation" / "manifest.json")
    groups, meta = d.EK.load_cases(ROOT, man["split"])
    gl = [(k, groups[k]) for k in sorted(groups)]
    pol, chooser, cnt = policy_for(c, d, name)
    path = Path(rr) / "confirmation" / ("by_case_%s.jsonl" % name)
    d.EK.run_set(Path(rr), "%s@confirm" % name, gl, pol, chooser, path)
    outs = [path]
    extra = {}
    src, how = d.C.CONFIRM[name]
    if cnt is not None:
        extra["counters"] = cnt.as_dict()
    if src and d.C.CONDS[src][0].startswith("GOAL"):
        from cp_disr.blocksworld import imitation as I
        vals = [pol.model.mask_density(I.snapshot_at(cs_, cs_.problem.init, 0)) for g in groups.values() for cs_ in g]
        extra["mask_density_nonself_initial_states_confirm"] = round(sum(vals) / len(vals), 4)
    if how not in ("look_mg", "look_count", "look_w", "b2_rule") or name == "MG_C3":
        bank = d.C.rj(Path(rr) / "confirmation" / "state_bank.json")
        cases = {cid: d.T.case_from_json(cr) for cid, cr in meta.items()}
        if name == "MG_C3" or how != "b2_rule":
            rows = d.EK.bank_eval(pol, bank, cases)
            bp = Path(rr) / "confirmation" / ("bank_%s.csv" % name)
            wcsv(bp, rows)
            outs.append(bp)
    c.receipt("confirm_" + name, "DONE", outs, **extra)
    return 0


# ------------------------------------------------------------------------------------------------ stage: planner references + report
def cmd_report(rr):
    d, c = ctx(rr)
    res = Path(rr) / "results"
    man = d.C.rj(Path(rr) / "confirmation" / "manifest.json") if (Path(rr) / "confirmation" / "manifest.json").is_file() else None
    statuses = {}
    for p in (Path(rr) / "receipts").glob("*.json"):
        r = json.loads(p.read_text())
        statuses[r.get("task", p.stem)] = r.get("status")
    d.C.wj(res / "technical_gaps.json", {k: v for k, v in sorted(statuses.items()) if v not in ("DONE", "REUSED_EXACT", "MATHEMATICALLY_EQUIVALENT_REUSED")})
    if man is None:
        c.receipt("report", "DONE", [res / "technical_gaps.json"], note="no confirmation set")
        return 0
    groups, meta = d.EK.load_cases(ROOT, man["split"])
    ep_by = {}
    for name in d.C.CONFIRM:
        pth = Path(rr) / "confirmation" / ("by_case_%s.jsonl" % name)
        if pth.is_file():
            ep_by[name] = {e["case_id"]: e for e in d.EK.jl(pth)}
    # planner reference
    solver = d.P.Solver()
    plan_rows = []
    for k, g in groups.items():
        for case in g:
            t0 = time.perf_counter()
            plan = solver.one_optimal_plan(case.problem.init, case.problem.goal)
            plan_rows.append({"case_id": case.case_id, "optimal_length": case.optimal_length, "plan_length": len(plan), "wall": round(time.perf_counter() - t0, 5)})
    wcsv(res / "planner_reference.csv", plan_rows)
    layers = {"ALL": lambda m: True, "h2_seen": lambda m: m["max_tower_height"] == 2, "h4_unseen": lambda m: m["max_tower_height"] == 4, "k1": lambda m: m["k_nontrivial_towers"] == 1, "k2": lambda m: m["k_nontrivial_towers"] == 2,
              "n6": lambda m: m["n_blocks"] == 6, "n8": lambda m: m["n_blocks"] == 8, "BREAK": lambda m: m["labels"]["requires_goal_destruction"] is True, "MONO": lambda m: m["labels"]["requires_goal_destruction"] is False}
    fam, hs, gs = [], [], []
    for name, eps in ep_by.items():
        for ln, pr in layers.items():
            es = [e for cid, e in eps.items() if pr(meta[cid])]
            s = d.EK.summarize(es)
            row = {"condition": name, "layer": ln, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in s.items()}}
            (fam if ln == "ALL" else hs if ln in ("h2_seen", "h4_unseen") else gs).append(row)
        for cell in sorted(groups):
            es = [e for cid, e in eps.items() if meta[cid]["cell"] == cell]
            gs.append({"condition": name, "layer": "cell=" + cell, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.EK.summarize(es).items()}})
    wcsv(res / "by_family.csv", fam)
    wcsv(res / "by_height_seen_unseen.csv", hs)
    wcsv(res / "by_goal_structure.csv", gs)
    pairs = (("LOOK2_MG", "MG_C3"), ("LOOK2_MG", "LOOK2_COUNT"), ("LOOK2_COUNT", "MG_C3"), ("SG_JOINT", "SG_FACT"), ("SG_FACT", "SG_BASE"), ("SG_JOINT", "SG_BASE"), ("SG_BASE", "MG_C3"), ("CAL_REL_C3", "CAL_ABS_C3"), ("CAL_REL_LOOK2", "CAL_ABS_LOOK2"),
             ("CAL_ABS_C3", "MG_C3"), ("CAL_REL_C3", "MG_C3"), ("CAL_ABS_LOOK2", "CAL_ABS_C3"), ("CAL_REL_LOOK2", "CAL_REL_C3"), ("REC_REL_T4", "REC_SELF_T4"), ("REC_REL_T8", "REC_REL_T4"), ("REC_SELF_T8", "REC_SELF_T4"),
             ("REC_REL_T8", "REC_SELF_T8"), ("REC_REL_T4", "MG_C3"), ("GOAL_REL", "GOAL_DENSE"), ("GOAL_DENSE", "MG_C3"), ("GOAL_REL", "MG_C3"), ("B_G1C3", "MG_C3"))
    prs = []
    for a, b in pairs:
        if a in ep_by and b in ep_by:
            for ln in ("ALL", "h2_seen", "h4_unseen", "BREAK", "MONO"):
                ids = {cid for cid in ep_by[a] if layers[ln](meta[cid])}
                for field, nm in (("success", "S"), ("decision_perfect", "D")):
                    pc = d.EK.pair_counts({i: ep_by[a][i] for i in ids}, {i: ep_by[b][i] for i in ids}, field)
                    common = [i for i in ids if ep_by[a][i]["success"] and ep_by[b][i]["success"]]
                    diff = [ep_by[a][i]["steps"] - ep_by[b][i]["steps"] for i in common]
                    prs.append({"first": a, "second": b, "layer": ln, "metric": nm, **pc, "common_success_n": len(common), "common_mean_steps_first_minus_second": round(statistics.mean(diff), 3) if diff else None})
    wcsv(res / "paired_repairs_harms.csv", prs)
    # first-error types via the taxonomy
    fe = []
    for name, eps in ep_by.items():
        agg = Counter()
        for cid, e in eps.items():
            case = d.T.case_from_json(meta[cid])
            st = d.SO.episode_error_stats(case.problem, e, False)
            if st["first_raw_class"]:
                agg[("first_raw", st["first_raw_class"])] += 1
            for cl, n in st["raw_classes"].items():
                agg[("raw_errors", cl)] += n
            agg[("rewrites", "raw_optimal_to_nonoptimal")] += st["rewrite"]
        fe += [{"condition": name, "kind": k[0], "class": k[1], "count": v} for k, v in sorted(agg.items())]
    wcsv(res / "first_error_types.csv", fe)
    # per-condition compute costs (inference) and training costs
    costs = []
    for name in d.C.CONFIRM:
        p = Path(rr) / "receipts" / ("confirm_%s.json" % name)
        if p.is_file():
            r = json.loads(p.read_text())
            costs.append({"condition": name, "status": r.get("status"), **{k: v for k, v in (r.get("counters") or {}).items()}, "episode_wall_seconds": round(sum(e["wall_seconds"] for e in ep_by.get(name, {}).values()), 2)})
    for cond in d.C.CONDS:
        p = Path(rr) / "runs" / cond / "training_accounting.json"
        if p.is_file():
            a = json.loads(p.read_text())
            costs.append({"condition": "TRAIN_" + cond, **{k: a[k] for k in ("optimizer_steps", "decision_samples_shown", "wall_seconds", "mean_epoch_wall_seconds", "parameters_new_module", "new_module_param_change_l2", "new_module_grad_norm_first_epoch", "new_module_grad_norm_last_epoch", "nan_events", "attempts")}})
    wcsv(res / "compute_costs.csv", costs)
    # family-specific tables from the development folders
    nei = []
    for cond in ("SG_BASE", "SG_FACT", "SG_JOINT"):
        p = Path(rr) / "development" / cond / "event_bank_diagnostics.csv"
        if p.is_file():
            rows = [r for r in csv.DictReader(open(p)) if r["complete"] == "True"]
            nei.append({"condition": cond, "states": len(rows), "Q_top1_hit": sum(r["Q_top1_hit"] == "True" for r in rows), "joint_top_pair_in_Y": sum(r["joint_top_pair_in_Y"] == "True" for r in rows),
                        "mean_P_Y": round(statistics.mean(float(r["P_Y"]) for r in rows), 4), "mean_P_F": round(statistics.mean(float(r["P_F"]) for r in rows), 4), "mean_P_A": round(statistics.mean(float(r["P_A"]) for r in rows), 4),
                        "pair_informative_states": sum(r["pair_informative"] == "True" for r in rows)})
    wcsv(res / "next_event_information_and_accuracy.csv", nei)
    cal = []
    for cond in ("CAL_ABS", "CAL_REL"):
        p = Path(rr) / "development" / cond / "support_extra.json"
        if p.is_file():
            e = json.loads(p.read_text())
            cal.append({"condition": cond, "beta": e.get("beta"), "a0": e.get("a0"), "c0": e.get("c0"), **{k: json.dumps(v) for k, v in e.items() if k.startswith("look2")}})
    wcsv(res / "calibration_and_temperature.csv", cal)
    rec = []
    for cond in ("REC_SELF", "REC_REL"):
        for t in (4, 8):
            nm = "%s_T%d" % (cond, t)
            if nm in ep_by:
                rec.append({"condition": nm, **{k: json.dumps(v) if isinstance(v, dict) else v for k, v in d.EK.summarize(ep_by[nm].values()).items()}})
    wcsv(res / "recurrent_T4_T8.csv", rec)
    gm = []
    for cond in ("GOAL_DENSE", "GOAL_REL"):
        p = Path(rr) / "development" / cond / "support_extra.json"
        if p.is_file():
            gm.append({"condition": cond, **{k: json.dumps(v) for k, v in json.loads(p.read_text()).items()}})
    wcsv(res / "goal_mask_density.csv", gm)
    lines = ["# Method serial suite v3 — confirmation numbers (auto-generated)", "", "success / all-steps-optimal (n)", "", "| condition | ALL | h2 (seen) | h4 (unseen) | BREAK |", "|---|---|---|---|---|"]
    for name, eps in ep_by.items():
        cells = []
        for ln in ("ALL", "h2_seen", "h4_unseen", "BREAK"):
            s = d.EK.summarize([e for cid, e in eps.items() if layers[ln](meta[cid])])
            cells.append("%d / %d (%d)" % (s["success"], s["perfect"], s["n"]))
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    (res / "final_summary_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    c.receipt("report", "DONE", [res / "by_family.csv", res / "final_summary_numbers.md"])
    return 0


# ------------------------------------------------------------------------------------------------ status
def cmd_status(rr):
    st = json.loads((Path(rr) / "stage_state.json").read_text()) if (Path(rr) / "stage_state.json").is_file() else {"stages": {}}
    for k, v in st["stages"].items():
        print(k, v.get("status"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--authorization-text-file", default=None)
    a = ap.parse_args()
    if a.cmd == "init":
        cmd_init(a.gpu, Path(a.authorization_text_file).read_text(encoding="utf-8"))
        return 0
    rr = a.run_root
    fn = {"run-all": lambda: run_all(rr), "status": lambda: cmd_status(rr), "fixtures": lambda: cmd_fixtures(rr), "lookahead_dev": lambda: cmd_lookahead(rr), "event_labels": lambda: cmd_event_labels(rr),
          "cal_prep": lambda: cmd_cal_prep(rr), "train": lambda: cmd_train(rr, a.condition), "devsel": lambda: cmd_devsel(rr, a.condition), "lock_selected": lambda: cmd_lock(rr),
          "confirm_prepare": lambda: cmd_confirm_prepare(rr), "confirm_eval": lambda: cmd_confirm_eval(rr, a.condition), "report": lambda: cmd_report(rr)}[a.cmd]
    return fn() or 0


if __name__ == "__main__":
    sys.exit(main())
