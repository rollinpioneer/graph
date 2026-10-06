#!/usr/bin/env python
"""C1-BW-GP-MINIMAL-ATTRIBUTION-V1 entry point.

    prep   --gpu N --authorization-text-file F   # run root, old destruction labels, G1 dev36, equivalence fixtures, frozen confirm set, registration/identity
    train  --run-root R --model B2RANK|M1GOAL --gpu N
    eval   --run-root R --condition C0|M0|M1|B2RANK|M1GOAL|G1|C3|G1C3|M1C3 --gpu N
    planner --run-root R
    report --run-root R
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NEW_SOURCES = ("src/cp_disr/blocksworld/gp_attribution.py", "scripts/c1_bw_gp_attribution.py", "tests/test_c1_bw_gp_attribution.py", "docs/c1_blocksworld/CP_DISR_C1_GP_Minimal_Attribution_Runbook_v2.md",
               "configs/splits/c1_bw_gp_attr_confirm112_v2.json")
LAYERS = (("T-MONO", "MONO"), ("T-BREAK", "BREAK"))


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def wj(p, doc):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def shaobj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ prep
def cmd_prep(root, gpu, auth, resume=None):
    import torch
    from cp_disr.blocksworld import a04p_controls as AC
    from cp_disr.blocksworld import canonical as K
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import goal_progress as GP
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import splits as SP
    from cp_disr.blocksworld import train as T
    from cp_disr.rl import set_suite_half_life
    if E.git(root, "branch", "--show-current") != A.BRANCH or E.git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("clean tracked tree on %s required" % A.BRANCH)
    # assets
    ck = root / E.MODELS["B2"]["path"]
    assert E.sha256_file(ck) == E.MODELS["B2"]["sha256"] and ck.stat().st_size == E.MODELS["B2"]["bytes"]
    for k, v in A.OLD.items():
        assert E.sha256_file(root / v["path"]) == v["sha256"] and (root / v["path"]).stat().st_size == v["bytes"], k
    dtp, rlp = root / A.GP_ROOT / "datasets" / "D_train.json", root / A.GP_ROOT / "datasets" / "rank_labels.json"
    assert E.sha256_file(dtp) == A.D_TRAIN["file_sha256"] and E.sha256_file(rlp) == A.D_TRAIN["rank_labels_file_sha256"]
    d_train = json.loads(dtp.read_text())
    assert I.dataset_identity(d_train)["sha256"] == A.D_TRAIN["semantic_sha256"]
    labels = json.loads(rlp.read_text())
    assert shaobj(labels) == A.D_TRAIN["rank_labels_semantic_sha256"]
    cases, half_life = I.load_train_cases(root / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(half_life)
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    dev_cases = [T.case_from_json(c) for c in td["dev"]]
    if resume:
        run_root = Path(resume).resolve()
    else:
        run_root = root / A.GPA_REL / ("%s_%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), E.sha256_file(root / A.RUNBOOK_REL)[:8]))
    for sub in ("prep", "runs", "eval", "results", "receipts"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    if not (run_root / "receipts" / "case_ledger.jsonl").exists():
        (run_root / "receipts" / "case_ledger.jsonl").write_text("")
    prep = run_root / "prep"
    solver = P.Solver()
    # 4.1 old train/dev destruction labels (UNKNOWN is never coerced to False); a resumed run reuses the files already written (the G1 dev36 evaluation is NOT repeated)
    if (prep / "old_train_dev_destruction_labels.json").exists():
        lab = json.loads((prep / "old_train_dev_destruction_labels.json").read_text())
    else:
        lab = {"train": {}, "dev": {}}
        for name, cs in (("train", cases), ("dev", dev_cases)):
            for c in cs:
                f = A.destruction_flag(c.problem, solver)
                lab[name][c.case_id] = "UNKNOWN" if f is None else bool(f)
        lab["counts"] = {n: {"true": sum(1 for v in d.values() if v is True), "false": sum(1 for v in d.values() if v is False), "unknown": sum(1 for v in d.values() if v == "UNKNOWN"), "n": len(d)} for n, d in (("train", lab["train"]), ("dev", lab["dev"]))}
        lab["note"] = "audit-reported expectation train 19/144, dev 5/36 is not a gate; labels are never used as a training signal or model input"
        wj(prep / "old_train_dev_destruction_labels.json", lab)
    print(json.dumps(lab["counts"]), flush=True)
    # 4.2 G1 on dev36, once
    device = torch.device("cuda", 0)
    if not (prep / "g1_dev36.json").exists():
        b2 = E.load_model(root, "B2", device)
        g1 = []
        for c in dev_cases:
            r = AC.run_episode(b2, c, P.Solver(), "G1", record_decisions=False)
            r["destruction_label"] = lab["dev"][c.case_id]
            g1.append(r)
        wj(prep / "g1_dev36.json", {"summary": {"n": len(g1), "success": sum(r["success"] for r in g1), "decision_perfect": sum(r["decision_perfect"] for r in g1),
                                               "break_label_true_cases": {r["case_id"]: (r["success"], r["decision_perfect"]) for r in g1 if r["destruction_label"] is True}}, "episodes": g1})
    gs = json.loads((prep / "g1_dev36.json").read_text())["summary"]
    print(json.dumps({"g1_dev36": [gs["success"], gs["decision_perfect"]]}), flush=True)
    # fixtures
    eq1 = A.check_rank_zero_equivalence(root, cases, d_train, labels, device)
    eq2 = A.check_goal_switch_equivalence(root, cases, d_train, labels, device)
    wj(prep / "equivalence_checks.json", {"b2_rank_zero": eq1, "m1_goal_switch": eq2})
    print(json.dumps({"eq_rank_zero": eq1["pass"], "eq_goal_switch": eq2["pass"]}), flush=True)
    if not (eq1["pass"] and eq2["pass"]):
        raise SystemExit("pre-training equivalence fixture failed: fix the implementation before registering")
    # confirm set
    forbidden = A.seen_problem_hashes(root, d_train, cases)
    n0 = len(forbidden)
    b = SP.Builder()
    out, report = [], {}
    rejects_all = {}

    def rec(cid, p, q, extra):
        return SP.case_record(cid, "gpa_confirm", p, q, A._extra(p, q, **extra))
    for gname, n, heights in A.STRUCTURE_GROUPS + (("H4", 6, (4, 2)),):
        for stratum in ("MONO", "BREAK"):
            quota = 8
            rej = Counter()
            made, att = A.gen_cell(b, forbidden, "%s:%s:%s" % (A.NAMESPACE, gname, stratum), n, heights, S_RED(), stratum == "BREAK", quota, 0.88, A.CELL_MAX_ATTEMPTS, rej)
            layer = ("T-" + stratum) if gname != "H4" else "H4"
            for i, (p, q, h) in enumerate(made):
                out.append(rec("BW_gpa_%s_%s_%02d" % (gname, stratum, i), p, q, {"cell": "%s|%s" % (gname, stratum), "structure_group": gname, "stratum": stratum, "layer": layer, "goal_heights": sorted(heights, reverse=True)}))
            report["%s|%s" % (gname, stratum)] = {"planned": quota, "built": len(made), "attempts": att, "shortfall": quota - len(made)}
            rejects_all["%s|%s" % (gname, stratum)] = dict(rej)
            print(json.dumps({"%s|%s" % (gname, stratum): report["%s|%s" % (gname, stratum)]}), flush=True)
    for stratum in ("MONO", "BREAK"):
        rej = Counter()
        pairs, att = A.gen_color_pairs(b, forbidden, "%s:COLOR:%s" % (A.NAMESPACE, stratum), stratum == "BREAK", 8, A.COLOR_MAX_ATTEMPTS, rej)
        for i, ((p, q, h), (pb, qb, hb)) in enumerate(pairs):
            skel = shaobj([list(p.names), list(p.init), list(p.goal)])
            for var, (pp, qq, hh) in (("RED", (p, q, h)), ("BLUE", (pb, qb, hb))):
                out.append(rec("BW_gpa_COLOR_%s_%02d_%s" % (stratum, i, var), pp, qq, {"cell": "COLOR|%s" % stratum, "structure_group": "COLOR", "stratum": stratum, "layer": "COLOR", "goal_heights": [3, 1, 1],
                                                                                        "pair_id": "CP_%s_%02d" % (stratum, i), "color_variant": var, "skeleton_hash": skel}))
        report["COLOR|%s" % stratum] = {"planned_pairs": 8, "built_pairs": len(pairs), "attempts": att, "shortfall_pairs": 8 - len(pairs)}
        rejects_all["COLOR|%s" % stratum] = dict(rej)
        print(json.dumps({"COLOR|%s" % stratum: report["COLOR|%s" % stratum]}), flush=True)
    doc = {"namespace": A.NAMESPACE, "excluded_problem_classes_at_start": n0, "cells": report, "counts": len(out), "cases": out, "note": "independent of every earlier set; frozen before any new model is trained or any model sees it"}
    wj(root / A.CONFIRM_REL, doc)
    wj(prep / "generation_shortfalls.json", {"cells": report, "rejection_counts": rejects_all, "frozen_cases": len(out)})
    wj(prep / "confirm_manifest.json", {"sha256": E.sha256_file(root / A.CONFIRM_REL), "counts": len(out), "layers": dict(Counter(c["layer"] + "|" + c["stratum"] for c in out)),
                                        "case_ids": [c["case_id"] for c in out], "color_pairs": sorted({c["pair_id"] for c in out if "pair_id" in c})})
    # registration
    diffs = {"B2RANK_vs_M0": {"scientific": {"rank_weight": {"M0": 0.0, "B2RANK": 1.0}}, "metadata_only": ["run_id", "output path", "source commit"], "identical": ["init A02 B2 final", "optimizer reset Adam lr 1e-4 (all trainable)", "epochs 100", "batch 32",
                                                                                                                                                        "shuffle 0*100003+ep", "grad clip 0.5", "decision chunk 256", "trainable modules", "D_train", "trajectory weights"]},
             "M1GOAL_vs_M1": {"scientific": {"encoder_goal_marks": {"M1": "zero", "M1GOAL": "production"}}, "metadata_only": ["run_id", "output path", "source commit"], "identical": ["init encoder A02 B2 final", "head seed 20261006", "phi/rho heads", "386-d per-goal features",
                                                                                                                                                                   "global aggregation", "no GRU/time/candidate context", "rank_weight 1.0", "encoder lr 1e-4, heads 3e-4", "epochs 100", "batch 32", "decision chunk 192", "D_train"]}}
    wj(prep / "paired_config_diffs.json", diffs)
    wj(prep / "initial_weight_identity.json", {"A02_B2": {k: E.MODELS["B2"][k] for k in ("path", "sha256", "bytes")}, "M0": A.OLD["M0"], "M1": A.OLD["M1"], "m1_goal_head_initial_sha256_expected": A.M1_HEAD_SHA})
    wj(prep / "dataset_identity.json", {**A.D_TRAIN, "d_train_trajectories": 1296, "d_train_decisions": 7446, "path": A.GP_ROOT + "/datasets/D_train.json", "rank_labels_path": A.GP_ROOT + "/datasets/rank_labels.json", "no_new_data_collection": True})
    wj(prep / "source_identity.json", A.source_identity(root, NEW_SOURCES))
    reg = {"card": A.CARD, "runbook": A.RUNBOOK_REL, "runbook_sha256": E.sha256_file(root / A.RUNBOOK_REL), "base_commit": A.BASE_COMMIT, "branch": A.BRANCH, "authorization_text": auth,
           "authorization_text_sha256": hashlib.sha256(auth.encode()).hexdigest(), "training": {"planned": 2, "runs": A.RUN_IDS, "epochs": A.EPOCHS, "batch": A.BATCH, "shuffle_seed": A.SHUFFLE_SEED, "lr_all_B2RANK": A.LR_ALL,
                                                                                                    "lr_encoder_M1GOAL": GP.EXISTING_LR, "lr_heads_M1GOAL": GP.NEW_HEAD_LR, "rank_weight": 1.0, "grad_clip": 0.5,
                                                                                                    "expected_steps": 4100, "expected_samples": 744600},
           "confirm": {"path": A.CONFIRM_REL, "sha256": E.sha256_file(root / A.CONFIRM_REL), "frozen_cases": len(out), "color_pairs": len({c["pair_id"] for c in out if "pair_id" in c})},
           "conditions": list(A.CONDITIONS), "planned_confirm_episodes": 9 * len(out), "planner_references": len(out), "dev36": {"new_models": 72, "g1": "done in prep"}, "thresholds": {"N>=32": 5, "16<=N<32": "ceil(5N/32)", "N<16": "no label", "pairs16": 3},
           "equivalence": {"b2_rank_zero": eq1["pass"], "m1_goal_switch": eq2["pass"]}, "no_score_gate": True, "conditional_third_run": "NOT_AUTHORIZED", "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "run_root_naming": "<UTC>_<runbook sha8> (the registration commit sha is not known before the commit)"}
    wj(prep / "registration.json", reg)
    print(json.dumps({"run_root": str(run_root), "confirm_cases": len(out)}))


def S_RED():
    from cp_disr.blocksworld import state as S
    return S.RED


# ------------------------------------------------------------------------------------------------ train
def cmd_train(root, run_root, model, gpu):
    import torch
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import goal_progress as GP
    from cp_disr.blocksworld import imitation as I
    from cp_disr.rl import set_suite_half_life
    from cp_disr.torch_rl import save_checkpoint
    A.check_identity(root, run_root)
    out = run_root / "runs" / A.RUN_IDS[model]
    try:
        (out / "checkpoints").mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit("duplicate start refused")
    dtp, rlp = root / A.GP_ROOT / "datasets" / "D_train.json", root / A.GP_ROOT / "datasets" / "rank_labels.json"
    assert E.sha256_file(dtp) == A.D_TRAIN["file_sha256"] and E.sha256_file(rlp) == A.D_TRAIN["rank_labels_file_sha256"]
    d_train, labels = json.loads(dtp.read_text()), json.loads(rlp.read_text())
    cases, half_life = I.load_train_cases(root / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(half_life)
    device = torch.device("cuda", 0)
    base = E.load_model(root, "B2", device)
    if model == "B2RANK":
        net = base
        trainer = A.RankTrainer(net, "b2", cases, device, labels, 1.0)
        head_sha = None
    else:
        net = A.GoalProgressModelGoal(base, "global", "production").to(device)
        trainer = GP.GPTrainer(net, cases, labels, device)
        head_sha = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in net.heads.parameters())).hexdigest()
        assert head_sha == A.M1_HEAD_SHA
    t0, rows, ckpts = time.time(), [], {}
    for ep in range(A.EPOCHS):
        row = trainer.epoch(d_train, A.SHUFFLE_SEED * 100003 + ep)
        row.update({"epoch": ep + 1, "optimizer_steps": trainer.steps, "wall_seconds": round(time.time() - t0, 1)})
        rows.append(row)
        if ep == 0 or (ep + 1) % 10 == 0:
            print("[gpa] %s epoch %d %s" % (model, ep + 1, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}), flush=True)
        with open(out / "epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if ep + 1 in (50, A.EPOCHS):
            name = "final" if ep + 1 == A.EPOCHS else "epoch050_trace_only"
            path = out / "checkpoints" / ("%s.pt" % name)
            save_checkpoint(path, net, trainer.optimizer, {"model": model, "epoch": ep + 1, "optimizer_steps": trainer.steps})
            ckpts[name] = {"path": str(path.relative_to(root)), "sha256": E.sha256_file(path), "bytes": path.stat().st_size}
    params = trainer.params
    acct = {"model": model, "run_id": A.RUN_IDS[model], "epochs": A.EPOCHS, "optimizer_steps": trainer.steps, "decision_samples_shown": sum(r["decisions"] for r in rows), "nan_events": trainer.nan_events,
            "wall_seconds": round(time.time() - t0, 1), "parameters_total": sum(p.numel() for p in net.parameters()), "parameters_trainable": sum(p.numel() for p in params), "new_head_initial_sha256": head_sha,
            "checkpoints": ckpts, "final_epoch": rows[-1], "init_checkpoint_sha256": E.MODELS["B2"]["sha256"], "d_train_file_sha256": A.D_TRAIN["file_sha256"], "rank_labels_file_sha256": A.D_TRAIN["rank_labels_file_sha256"],
            "evaluation_checkpoint": "final (epoch 100); epoch050 is trace-only"}
    wj(out / "training_accounting.json", acct)
    print(json.dumps({"done": model, "steps": trainer.steps, "wall": acct["wall_seconds"]}))


# ------------------------------------------------------------------------------------------------ eval / planner
def confirm_cases(root, split_rel):
    from cp_disr.blocksworld import train as T
    doc = json.loads((root / split_rel).read_text())
    by = defaultdict(list)
    meta = {}
    for c in doc["cases"]:
        by[c["cell"]].append(T.case_from_json(c))
        meta[c["case_id"]] = c
    return by, meta


def cmd_eval(root, run_root, cond, gpu):
    import torch
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import train as T
    from cp_disr.rl import set_suite_half_life
    A.check_identity(root, run_root)
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    set_suite_half_life(float(td["half_life"]))
    device = torch.device("cuda", 0)
    policy = A.load_condition(root, cond, device, run_root)
    chooser = A.chooser_for(cond)
    by, _meta = confirm_cases(root, A.CONFIRM_REL)
    cells = sorted(by)
    groups = [(c, by[c]) for c in cells]
    if cond in ("B2RANK", "M1GOAL"):
        groups.append(("dev", [T.case_from_json(c) for c in td["dev"]]))
    solver = P.Solver()
    print(json.dumps(R.run_cases(run_root, cond, groups, lambda c: A.run_episode_with(policy, c, solver, chooser), run_root / "eval" / cond / "episodes.jsonl")))


def cmd_planner(root, run_root):
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import state as S
    A.check_identity(root, run_root)
    by, _ = confirm_cases(root, A.CONFIRM_REL)

    def ref(c):
        s = P.Solver()
        t0 = time.perf_counter()
        plan = s.one_optimal_plan(c.problem.init, c.problem.goal)
        cpu = time.perf_counter() - t0
        st = c.problem.init
        for a in plan:
            st = S.apply(st, a)
        return {"case_id": c.case_id, "n_blocks": c.problem.n, "optimal_length": c.optimal_length, "plan_length": len(plan), "success": S.goal_satisfied(c.problem, st), "expanded_nodes": s.stats.expanded,
                "wall_seconds_cold_cache": cpu}
    print(json.dumps(R.run_cases(run_root, "planner", [(c, by[c]) for c in sorted(by)], ref, run_root / "eval" / "planner" / "episodes.jsonl")))


# ------------------------------------------------------------------------------------------------ report
def cmd_report(root, run_root):
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    res = run_root / "results"
    reg = json.loads((run_root / "prep" / "registration.json").read_text())
    conf = json.loads((root / A.CONFIRM_REL).read_text())
    meta = {c["case_id"]: c for c in conf["cases"]}
    labels_old = json.loads((run_root / "prep" / "old_train_dev_destruction_labels.json").read_text())
    eps = {c: [e for e in jl(run_root / "eval" / c / "episodes.jsonl") if e["group"] != "dev"] for c in A.CONDITIONS}
    for c in A.CONDITIONS:
        ids = [e["case_id"] for e in eps[c]]
        assert len(ids) == len(set(ids)) == len(meta), (c, len(ids))
    ep = {c: {e["case_id"]: e for e in eps[c]} for c in A.CONDITIONS}

    def layer_of(m):
        return {"T-MONO": "T-MONO", "T-BREAK": "T-BREAK"}.get(m["layer"], m["layer"] + "-" + m["stratum"])
    percase = []
    for c in A.CONDITIONS:
        for cid, e in ep[c].items():
            m = meta[cid]
            percase.append({"condition": c, "case_id": cid, "cell": m["cell"], "layer": m["layer"], "stratum": m["stratum"], "n": m["n_blocks"], "k": m["k_nontrivial_towers"], "h": m["max_tower_height"],
                            "L": m["optimal_length"], "pair_id": m.get("pair_id"), "color_variant": m.get("color_variant"), "success": e["success"], "decision_perfect": e["decision_perfect"], "steps": e["steps"],
                            "excess_steps": e["excess_steps"], "cycle": e["cycle"], "first_divergence": e["first_divergence"], "interventions": e["interventions"], "reason": e["reason"],
                            "necessary_destruction_steps": e["necessary_destruction_steps"], "avoidable_destruction_steps": e["avoidable_destruction_steps"]})
    wcsv(res / "by_case.csv", percase)

    def cnt(c, pred):
        es = [ep[c][cid] for cid, m in meta.items() if pred(m)]
        return {"n": len(es), "success": sum(e["success"] for e in es), "perfect": sum(e["decision_perfect"] for e in es), "mean_steps": round(statistics.mean(e["steps"] for e in es), 2) if es else None,
                "cycles": sum(e["cycle"] for e in es), "no_unvisited": sum(1 for e in es if e["reason"] == "NO_UNVISITED_SUCCESSOR")}
    layers = {"T-MONO": lambda m: m["layer"] == "T-MONO", "T-BREAK": lambda m: m["layer"] == "T-BREAK", "H4-MONO": lambda m: m["layer"] == "H4" and m["stratum"] == "MONO",
              "H4-BREAK": lambda m: m["layer"] == "H4" and m["stratum"] == "BREAK", "COLOR-MONO": lambda m: m["layer"] == "COLOR" and m["stratum"] == "MONO",
              "COLOR-BREAK": lambda m: m["layer"] == "COLOR" and m["stratum"] == "BREAK", "COLOR-RED": lambda m: m.get("color_variant") == "RED", "COLOR-BLUE": lambda m: m.get("color_variant") == "BLUE",
              "H4": lambda m: m["layer"] == "H4", "COLOR": lambda m: m["layer"] == "COLOR", "ALL": lambda m: True}
    rows = []
    for c in A.CONDITIONS:
        for ln, pred in layers.items():
            rows.append({"condition": c, "layer": ln, **cnt(c, pred)})
    wcsv(res / "by_destructive_stratum.csv", rows)
    # n/k/h/L strata
    st = []
    for c in A.CONDITIONS:
        g = defaultdict(list)
        for r in percase:
            if r["condition"] == c:
                g["n=%d k=%d h=%d layer=%s" % (r["n"], r["k"], r["h"], r["layer"] + "|" + r["stratum"])].append(r)
                g["L " + ("1-8" if r["L"] <= 8 else "9-16" if r["L"] <= 16 else "17-24" if r["L"] <= 24 else ">24")].append(r)
        for k, rs in sorted(g.items()):
            st.append({"condition": c, "stratum": k, "n": len(rs), "success": sum(r["success"] for r in rs), "decision_perfect": sum(r["decision_perfect"] for r in rs), "cycles": sum(r["cycle"] for r in rs),
                       "mean_L": round(statistics.mean(r["L"] for r in rs), 2)})
    wcsv(res / "by_n_k_h_l.csv", st)

    def pair4(a, b, pred, field):
        ids = [cid for cid, m in meta.items() if pred(m)]
        both = sum(ep[a][i][field] and ep[b][i][field] for i in ids)
        oa = sum(ep[a][i][field] and not ep[b][i][field] for i in ids)
        ob = sum(ep[b][i][field] and not ep[a][i][field] for i in ids)
        return {"n": len(ids), "both": both, "only_first": oa, "only_second": ob, "neither": len(ids) - both - oa - ob, "net": oa - ob}
    matched = []
    for a, b in (("B2RANK", "M0"), ("M1GOAL", "M1")):
        for ln in ("T-MONO", "T-BREAK", "H4", "COLOR", "ALL"):
            for field, nm in (("success", "S"), ("decision_perfect", "D")):
                p = pair4(a, b, layers[ln], field)
                matched.append({"first": a, "second": b, "layer": ln, "metric": nm, **p, "direction": A.direction(p["net"], p["n"])})
    wcsv(res / "matched_training_comparisons.csv", matched)
    # colour twins
    tw = []
    for c in A.CONDITIONS:
        for strat in ("MONO", "BREAK", "ALL"):
            pids = sorted({m["pair_id"] for m in meta.values() if m.get("pair_id") and (strat == "ALL" or m["stratum"] == strat)})
            for field, nm in (("success", "S"), ("decision_perfect", "D")):
                both = ro = bo = nei = 0
                for pid in pids:
                    r = next(i for i, m in meta.items() if m.get("pair_id") == pid and m["color_variant"] == "RED")
                    bl = next(i for i, m in meta.items() if m.get("pair_id") == pid and m["color_variant"] == "BLUE")
                    x, y = ep[c][r][field], ep[c][bl][field]
                    both += x and y
                    ro += x and not y
                    bo += y and not x
                    nei += (not x) and (not y)
                tw.append({"condition": c, "stratum": strat, "metric": nm, "pairs": len(pids), "both": both, "RED_only": ro, "BLUE_only": bo, "neither": nei, "net_RED_minus_BLUE": ro - bo,
                           "direction": A.direction(ro - bo, len(pids)) if len(pids) >= 16 else ("NO_LABEL_P_LT_16")})
    wcsv(res / "colour_twin_pairs.csv", tw)
    # G1 agreement
    ag = []
    for c in A.CONDITIONS:
        if c == "G1":
            continue
        for ln in ("T-MONO", "T-BREAK", "H4", "COLOR", "ALL"):
            for field, nm in (("success", "S"), ("decision_perfect", "D")):
                p = pair4(c, "G1", layers[ln], field)
                n = p["n"]
                ag.append({"condition": c, "layer": ln, "metric": nm, **p, "agreement_rate": round((p["both"] + p["neither"]) / n, 4), "iou_correct": round(p["both"] / max(1, p["both"] + p["only_first"] + p["only_second"]), 4)})
    wcsv(res / "g1_agreement.csv", ag)
    # rule combinations
    rc = []
    for c in ("C0", "G1", "C3", "G1C3", "M1", "M1C3"):
        for ln in ("T-MONO", "T-BREAK", "H4", "COLOR", "ALL"):
            rc.append({"condition": c, "layer": ln, **cnt(c, layers[ln])})
    wcsv(res / "rule_combination_comparisons.csv", rc)
    # dev + confirmation
    dev = {}
    dev_g1 = json.loads((run_root / "prep" / "g1_dev36.json").read_text())["summary"]
    dev["G1"] = (dev_g1["success"], dev_g1["decision_perfect"])
    dev["C0"] = (36, 35)
    gp = root / A.GP_ROOT
    for c, cond_dir in (("M0", "M0"), ("M1", "M1")):
        d = [e for e in jl(gp / "eval" / cond_dir / "episodes.jsonl") if e["group"] == "dev"]
        dev[c] = (sum(e["success"] for e in d), sum(e["decision_perfect"] for e in d))
    for c in ("B2RANK", "M1GOAL"):
        d = [e for e in jl(run_root / "eval" / c / "episodes.jsonl") if e["group"] == "dev"]
        dev[c] = (sum(e["success"] for e in d), sum(e["decision_perfect"] for e in d))
    dc = []
    for c in A.CONDITIONS:
        row = {"condition": c, "dev36_success": dev.get(c, ("NA", "NA"))[0], "dev36_perfect": dev.get(c, ("NA", "NA"))[1]}
        for ln in ("T-MONO", "T-BREAK", "H4-MONO", "H4-BREAK", "COLOR-MONO", "COLOR-BREAK", "ALL"):
            x = cnt(c, layers[ln])
            row[ln + "_S"] = "%d/%d" % (x["success"], x["n"])
            row[ln + "_D"] = "%d/%d" % (x["perfect"], x["n"])
        dc.append(row)
    wcsv(res / "dev_and_confirmation.csv", dc)
    # identities / verify
    accts = {m: json.loads((run_root / "runs" / A.RUN_IDS[m] / "training_accounting.json").read_text()) for m in ("B2RANK", "M1GOAL")}
    wj(res / "checkpoint_identity.json", {m: {"final": a["checkpoints"]["final"], "parameters_trainable": a["parameters_trainable"], "optimizer_steps": a["optimizer_steps"], "decision_samples_shown": a["decision_samples_shown"],
                                            "nan_events": a["nan_events"], "wall_seconds": a["wall_seconds"], "committed_to_git": False} for m, a in accts.items()})
    ledger = jl(run_root / "receipts" / "case_ledger.jsonl")
    started, completed = Counter(), Counter()
    for e in ledger:
        k = (e["actor"], e["split"], e["case_id"])
        if e["event"] == "STARTED":
            started[k] += 1
        if e["event"] == "COMPLETED":
            completed[k] += 1
    tech = len({(e["actor"], e["split"], e["case_id"]) for e in ledger if e["event"] == "TECHNICAL_INCOMPLETE"})
    srcid = json.loads((run_root / "prep" / "source_identity.json").read_text())
    verify = {"confirm_episodes_completed": sum(v for k, v in completed.items() if k[0] in A.CONDITIONS and k[1] != "dev"), "confirm_episodes_planned": reg["planned_confirm_episodes"],
              "dev36_new_models_completed": sum(v for k, v in completed.items() if k[1] == "dev"), "planner_completed": sum(v for k, v in completed.items() if k[0] == "planner"), "technical_incomplete": tech,
              "ledger_once_each": all(v == 1 for v in started.values()) and all(v == 1 for v in completed.values()),
              "old_roots_unchanged": all(E.git(root, "rev-parse", "HEAD:%s" % rel) == srcid["old_root_tree_ids_at_base"][rel] for rel in A.OLD_ROOTS), "head": E.git(root, "rev-parse", "HEAD"),
              "old_checkpoints_unchanged": all(E.sha256_file(root / v["path"]) == v["sha256"] for v in A.OLD.values()) and E.sha256_file(root / E.MODELS["B2"]["path"]) == E.MODELS["B2"]["sha256"],
              "frozen_sources_unchanged": srcid["all_frozen_unchanged"], "unknown_labels_not_coerced": True, "destruction_label_scope": "ALL_OPTIMAL_PLANS"}
    verify["verdict"] = "PASS" if (verify["confirm_episodes_completed"] == verify["confirm_episodes_planned"] and verify["dev36_new_models_completed"] == 72 and tech == 0 and verify["ledger_once_each"]
                                   and verify["old_roots_unchanged"] and verify["old_checkpoints_unchanged"]) else "FAIL"
    wj(res / "verify.json", verify)
    print(json.dumps({"verify": verify["verdict"], "completed": verify["confirm_episodes_completed"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "train", "eval", "planner", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        if name == "prep":
            p.add_argument("--authorization-text-file", required=True)
            p.add_argument("--resume-run-root", default=None)
        else:
            p.add_argument("--run-root", required=True)
        if name == "train":
            p.add_argument("--model", required=True, choices=("B2RANK", "M1GOAL"))
        if name == "eval":
            p.add_argument("--condition", required=True, choices=("C0", "M0", "M1", "B2RANK", "M1GOAL", "G1", "C3", "G1C3", "M1C3"))
        if name in ("prep", "train", "eval"):
            p.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    if hasattr(a, "gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    root = Path(a.root).resolve()
    if a.cmd == "prep":
        cmd_prep(root, a.gpu, Path(a.authorization_text_file).read_text(encoding="utf-8"), a.resume_run_root)
        return
    rr = Path(a.run_root).resolve()
    if a.cmd == "train":
        cmd_train(root, rr, a.model, a.gpu)
    elif a.cmd == "eval":
        cmd_eval(root, rr, a.condition, a.gpu)
    elif a.cmd == "planner":
        cmd_planner(root, rr)
    else:
        cmd_report(root, rr)


if __name__ == "__main__":
    main()
