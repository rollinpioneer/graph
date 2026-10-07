#!/usr/bin/env python
"""C1-BW-SCALE-ORDER-PROTOTYPE-V1 entry point.

    prep     --authorization-text-file F [--procs N]   # fixtures, padded + relabelled training data, 48-problem board, state bank, registration / identity
    train    --run-root R --gpu N                       # R-C1-SO-MG-S8-PAD-0 (the only new training of this card)
    eval     --run-root R --actor MG_C3|B_G1C3|PAD_C3|PAD_RAW --set board48|fresh112|dev36 --gpu N
    planner  --run-root R
    taxonomy --run-root R                               # step 0: error classes and denominators from saved logs
    bank     --run-root R --gpu N                       # common-state ranking of MG-final vs MG-S8-PAD
    report   --run-root R
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NEW_SOURCES = ("src/cp_disr/blocksworld/scale_order.py", "scripts/c1_bw_scale_order.py", "tests/test_c1_bw_scale_order.py", "docs/c1_blocksworld/CP_DISR_C1_Scale_Order_Interaction_Next_Experiment_v1.md", "configs/splits/c1_bw_scale_order_board48_v1.json")


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


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ prep
def cmd_prep(root, auth, procs):
    import torch
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import scale_order as SO
    from cp_disr.blocksworld import scorer_control as SC
    from cp_disr.blocksworld import splits as SP
    from cp_disr.blocksworld import state as S
    from cp_disr.rl import set_suite_half_life
    if E.git(root, "branch", "--show-current") != SO.BRANCH or E.git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("clean tracked tree on %s required" % SO.BRANCH)
    if E.git(root, "rev-parse", "HEAD") != SO.BASE_COMMIT:
        raise SystemExit("HEAD must equal the base commit before registration")
    b2, mg = root / E.MODELS["B2"]["path"], root / SC.M1GOAL["path"]
    assert E.sha256_file(b2) == E.MODELS["B2"]["sha256"] and E.sha256_file(mg) == SC.M1GOAL["sha256"]
    dtp, rlp = root / A.GP_ROOT / "datasets" / "D_train.json", root / A.GP_ROOT / "datasets" / "rank_labels.json"
    assert E.sha256_file(dtp) == A.D_TRAIN["file_sha256"] and E.sha256_file(rlp) == A.D_TRAIN["rank_labels_file_sha256"]
    d_train, labels = json.loads(dtp.read_text()), json.loads(rlp.read_text())
    assert I.dataset_identity(d_train)["sha256"] == A.D_TRAIN["semantic_sha256"] and len(d_train) == 1296
    cases, half_life = I.load_train_cases(root / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(half_life)
    run_root = root / SO.SO_REL / ("%s_%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), E.sha256_file(root / SO.PLAN_REL)[:8]))
    for sub in ("prep", "datasets", "runs", "eval", "results", "receipts"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    (run_root / "receipts" / "case_ledger.jsonl").write_text("")
    prep = run_root / "prep"
    fx = SO.check_fixtures(cases, d_train, torch.device("cpu"))
    wj(prep / "fixtures.json", fx)
    print(json.dumps({"fixtures_pass": fx["pass"]}), flush=True)
    if not fx["pass"]:
        raise SystemExit("fixture failed: %s" % json.dumps(fx, default=str))
    # padded training data
    t0 = time.time()
    pcases, ptrajs, plabels, val = SO.build_padding(cases, d_train, procs)
    print(json.dumps({"padding": {k: v for k, v in val.items() if k != "exception_tids"}}), flush=True)
    wj(run_root / "datasets" / "padded_cases.json", pcases)
    wj(run_root / "datasets" / "padded_trajectories.json", ptrajs)
    wj(run_root / "datasets" / "padded_rank_labels.json", plabels)
    wj(prep / "padding_label_validation.json", val)
    sched = SO.schedule_counts(len(d_train))
    wj(prep / "train_padding_manifest.json", {"parents": len(d_train), "versions": {"0": "original", "1": "pad6", "2": "pad7", "3": "pad8"}, "pad_rule": "new blocks b{n}..b{N-1} on the table, goal OnTable, colours alternate from a hashed start, original names/colours/init/goal untouched, full planner relabelling",
                                              "schedule": sched, "schedule_rule": "epoch even -> original; epoch odd -> pad version 1 + (slot + epoch//2) % 3", "per_epoch_trajectories": len(d_train), "per_epoch_decisions": sum(len(t["actions"]) for t in d_train),
                                              "dataset_files": {n: sha_file(run_root / "datasets" / n) for n in ("padded_cases.json", "padded_trajectories.json", "padded_rank_labels.json")}, "original_D_train_sha256": A.D_TRAIN["file_sha256"],
                                              "exception_trajectories_flagged": val["trajectories_with_exception"], "extra_label_wall_seconds": round(time.time() - t0, 1)})
    # 48-problem factorial board (planner labels only; no model)
    forbidden, by_source = SO.board_excluded(root, d_train, cases)
    excluded_at_start = set(forbidden)
    b = SP.Builder()
    out, report, rejects_all = [], {}, {}
    for cell, n, k, h, heights in SO.BOARD_CELLS:
        from collections import Counter as C_
        rej = C_()
        made, att = SO.gen_board_cell(b, forbidden, "%s:%s" % (SO.NAMESPACE, cell), n, heights, SO.BOARD_PER_CELL, SO.BOARD_MAX_ATTEMPTS, rej)
        for i, (p, q, hh) in enumerate(made):
            out.append(SP.case_record("BW_so_%s_%02d" % (cell, i), "scale_order_board", p, q, A._extra(p, q, **{"namespace": SO.NAMESPACE, "cell": cell, "n_cell": n, "k_cell": k, "h_cell": h, "goal_heights": sorted(heights, reverse=True)})))
        report[cell] = {"planned": SO.BOARD_PER_CELL, "built": len(made), "attempts": att, "shortfall": SO.BOARD_PER_CELL - len(made)}
        rejects_all[cell] = dict(rej)
        print(json.dumps({cell: report[cell]}), flush=True)
    hashes = [c["problem_iso_hash"] for c in out]
    assert not (set(hashes) & excluded_at_start) and len(set(hashes)) == len(hashes)
    wj(root / SO.BOARD_REL, {"namespace": SO.NAMESPACE, "cells": report, "counts": len(out), "cases": out, "note": "development board; generated from structure, legality, planner labels and dedup only"})
    wj(prep / "board_generation.json", {"cells": report, "rejection_counts": rejects_all, "frozen_cases": len(out), "excluded_by_source": by_source, "excluded_total_at_start": len(excluded_at_start), "overlap_with_excluded": 0})
    # common state bank (before any model scores anything)
    from cp_disr.blocksworld import train as T
    solver = P.Solver()
    bank = []
    for c in out:
        case = T.case_from_json(c)
        for kind, st, idx in SO.bank_states(case, solver):
            r = SO.state_record(case, st, kind, solver)
            r.update({"bank_id": "%s#%s" % (case.case_id, kind), "cell": c["cell"], "plan_index": idx})
            bank.append(r)
    wj(prep / "state_bank.json", bank)
    wj(prep / "weight_identity.json", {"A02_B2": {k: E.MODELS["B2"][k] for k in ("run_id", "path", "sha256", "bytes")}, "M1_GOAL_reference": SC.M1GOAL, "m1_head_initial_sha256": A.M1_HEAD_SHA, "training_init": "A02 B2 final + seed-20261006 heads (NOT the M1-GOAL final)", "committed_to_git": False})
    wj(prep / "source_identity.json", SO.source_identity(root, NEW_SOURCES))
    reg = {"card": SO.CARD, "plan": SO.PLAN_REL, "plan_sha256": E.sha256_file(root / SO.PLAN_REL), "base_commit": SO.BASE_COMMIT, "branch": SO.BRANCH, "authorization_text": auth, "authorization_text_sha256": hashlib.sha256(auth.encode()).hexdigest(),
           "scope": "tier 1 only: step-0 error taxonomy, one training R-C1-SO-MG-S8-PAD-0, development evaluation, method-branch recommendation; no second training", "new_training_runs": 1,
           "training": {"epochs": SO.EPOCHS, "steps": 4100, "decision_samples": 744600, "checkpoints_saved": list(SO.CKPT_EPOCHS), "primary": "epoch 100", "no_score_gate": True},
           "board": {"path": SO.BOARD_REL, "sha256": E.sha256_file(root / SO.BOARD_REL), "frozen_cases": len(out), "cells": report}, "state_bank": {"states": len(bank), "sha256": sha_file(prep / "state_bank.json")},
           "planned_new_episodes": {"fresh112_PAD_C3": 112, "board48": 144, "dev36": 36}, "planned_planner_refs": len(out), "fixtures_pass": fx["pass"], "padding_exceptions": val["trajectories_with_exception"],
           "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "evaluation_role": "DEVELOPMENT_NOT_FINAL_CONFIRMATION"}
    wj(prep / "registration.json", reg)
    print(json.dumps({"run_root": str(run_root), "board_cases": len(out), "bank_states": len(bank)}))


# ------------------------------------------------------------------------------------------------ train
def cmd_train(root, run_root, gpu):
    import torch
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import goal_progress as GP
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import scale_order as SO
    from cp_disr.blocksworld.train import case_from_json
    from cp_disr.rl import set_suite_half_life
    from cp_disr.torch_rl import save_checkpoint
    SO.check_identity(root, run_root)
    out = run_root / "runs" / SO.RUN_ID
    try:
        (out / "checkpoints").mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit("duplicate start refused")
    dtp, rlp = root / A.GP_ROOT / "datasets" / "D_train.json", root / A.GP_ROOT / "datasets" / "rank_labels.json"
    assert E.sha256_file(dtp) == A.D_TRAIN["file_sha256"] and E.sha256_file(rlp) == A.D_TRAIN["rank_labels_file_sha256"]
    man = json.loads((run_root / "prep" / "train_padding_manifest.json").read_text())
    for n, h in man["dataset_files"].items():
        assert sha_file(run_root / "datasets" / n) == h, n
    d_train, labels = json.loads(dtp.read_text()), json.loads(rlp.read_text())
    cases, half_life = I.load_train_cases(root / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(half_life)
    pcases = [case_from_json(c) for c in json.loads((run_root / "datasets" / "padded_cases.json").read_text())]
    ptrajs = json.loads((run_root / "datasets" / "padded_trajectories.json").read_text())
    padded = {t["tid"]: t for t in ptrajs}
    labels = {**labels, **json.loads((run_root / "datasets" / "padded_rank_labels.json").read_text())}
    device = torch.device("cuda", 0)
    base = E.load_model(root, "B2", device)
    net = A.GoalProgressModelGoal(base, "global", "production").to(device)
    head_sha = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in net.heads.parameters())).hexdigest()
    assert head_sha == A.M1_HEAD_SHA
    trainer = GP.GPTrainer(net, cases + pcases, labels, device)
    t0, rows, ckpts, vcount = time.time(), [], {}, Counter()
    for ep in range(SO.EPOCHS):
        trajs = SO.epoch_trajectories(d_train, padded, ep)
        vcount.update(t["n_blocks"] for t in trajs)
        row = trainer.epoch(trajs, SO.SHUFFLE_SEED * 100003 + ep)
        row.update({"epoch": ep + 1, "optimizer_steps": trainer.steps, "wall_seconds": round(time.time() - t0, 1), "n6": sum(t["n_blocks"] == 6 for t in trajs), "n7": sum(t["n_blocks"] == 7 for t in trajs), "n8": sum(t["n_blocks"] == 8 for t in trajs)})
        rows.append(row)
        if ep == 0 or (ep + 1) % 10 == 0:
            print("[so] epoch %d %s" % (ep + 1, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}), flush=True)
        with open(out / "epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if ep + 1 in SO.CKPT_EPOCHS:
            name = "final" if ep + 1 == SO.EPOCHS else "epoch%03d" % (ep + 1)
            path = out / "checkpoints" / ("%s.pt" % name)
            save_checkpoint(path, net, trainer.optimizer, {"model": "MG-S8-PAD", "epoch": ep + 1, "optimizer_steps": trainer.steps})
            ckpts[name] = {"path": str(path.relative_to(root)), "sha256": E.sha256_file(path), "bytes": path.stat().st_size}
    acct = {"run_id": SO.RUN_ID, "epochs": SO.EPOCHS, "optimizer_steps": trainer.steps, "decision_samples_shown": sum(r["decisions"] for r in rows), "nan_events": trainer.nan_events, "wall_seconds": round(time.time() - t0, 1),
            "parameters_total": sum(p.numel() for p in net.parameters()), "parameters_trainable": sum(p.numel() for p in trainer.params), "new_head_initial_sha256": head_sha, "checkpoints": ckpts,
            "slot_epochs_by_block_count": dict(vcount), "final_epoch": rows[-1], "init_checkpoint_sha256": E.MODELS["B2"]["sha256"], "d_train_file_sha256": A.D_TRAIN["file_sha256"], "padding_manifest_sha256": sha_file(run_root / "prep" / "train_padding_manifest.json"),
            "mean_epoch_wall_seconds": round((time.time() - t0) / SO.EPOCHS, 2), "evaluation_checkpoint": "final (epoch 100); epoch020/040/060/080 saved, not used for any reported number"}
    wj(out / "training_accounting.json", acct)
    print(json.dumps({"done": SO.RUN_ID, "steps": trainer.steps, "wall": acct["wall_seconds"]}))


# ------------------------------------------------------------------------------------------------ eval / planner
def board_cases(root, rel):
    from cp_disr.blocksworld import train as T
    doc = json.loads((root / rel).read_text())
    by, meta = defaultdict(list), {}
    for c in doc["cases"]:
        by[c["cell"]].append(T.case_from_json(c))
        meta[c["case_id"]] = c
    return by, meta


def cmd_eval(root, run_root, actor, set_name, gpu):
    import torch
    from cp_disr.blocksworld import a04p_controls as AC
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import scale_order as SO
    from cp_disr.blocksworld import scorer_control as SC
    from cp_disr.blocksworld import train as T
    from cp_disr.rl import set_suite_half_life
    assert (actor, set_name) in {("MG_C3", "board48"), ("B_G1C3", "board48"), ("PAD_C3", "board48"), ("PAD_C3", "fresh112"), ("PAD_RAW", "dev36")}, "combination not registered"
    SO.check_identity(root, run_root)
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    set_suite_half_life(float(td["half_life"]))
    device = torch.device("cuda", 0)
    if actor.startswith("PAD"):
        policy = SO.load_pad_scorer(root, run_root, "final", device)
    else:
        policy = SC.load_scorer(root, actor, device)
    d0 = SC.weights_digest(policy)
    chooser = (lambda _c, ep, snap, logits, mem: AC.choose("C0", ep, snap, logits, mem)) if actor == "PAD_RAW" else SC.chooser_for("MG_C3" if actor in ("MG_C3", "PAD_C3") else "B_G1C3")
    if set_name == "board48":
        by, _ = board_cases(root, SO.BOARD_REL)
        groups = [(c, by[c]) for c in sorted(by)]
    elif set_name == "fresh112":
        by, _ = board_cases(root, SC.FRESH_REL)
        groups = [(c, by[c]) for c in sorted(by)]
    else:
        groups = [("dev", [T.case_from_json(c) for c in td["dev"]])]
    solver = P.Solver()
    counts = R.run_cases(run_root, "%s@%s" % (actor, set_name), groups, lambda c: A.run_episode_with(policy, c, solver, chooser), run_root / "eval" / actor / ("%s.jsonl" % set_name))
    d1 = SC.weights_digest(policy)
    wj(run_root / "eval" / actor / ("weight_integrity_%s.json" % set_name), {"digest_before": d0, "digest_after": d1, "unchanged": d0 == d1, "counts": counts})
    print(json.dumps(counts))


def cmd_planner(root, run_root):
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import scale_order as SO
    from cp_disr.blocksworld import state as S
    SO.check_identity(root, run_root)
    by, _ = board_cases(root, SO.BOARD_REL)

    def ref(c):
        s = P.Solver()
        t0 = time.perf_counter()
        plan = s.one_optimal_plan(c.problem.init, c.problem.goal)
        cpu = time.perf_counter() - t0
        st = c.problem.init
        for a in plan:
            st = S.apply(st, a)
        return {"case_id": c.case_id, "n_blocks": c.problem.n, "optimal_length": c.optimal_length, "plan_length": len(plan), "success": S.goal_satisfied(c.problem, st), "expanded_nodes": s.stats.expanded, "wall_seconds_cold_cache": cpu}
    print(json.dumps(R.run_cases(run_root, "planner@board48", [(c, by[c]) for c in sorted(by)], ref, run_root / "eval" / "planner" / "board48.jsonl")))


# ------------------------------------------------------------------------------------------------ step 0: taxonomy
def cmd_taxonomy(root, run_root):
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import scale_order as SO
    from cp_disr.blocksworld import scorer_control as SC
    from cp_disr.blocksworld import train as T
    SO.check_identity(root, run_root)
    sources = (("gpa_v2_confirm112", root / SC.GPA_RUN_REL, A.CONFIRM_REL, ("C0", "M0", "M1", "B2RANK", "M1GOAL", "G1", "C3", "G1C3", "M1C3"), {"C3", "G1C3", "M1C3"}),
               ("scorer_control_fresh112", root / SO.SC_RUN_REL, SC.FRESH_REL, SC.CONDITIONS, set(SC.CONDITIONS)))
    cls_rows, den_rows = [], []
    for src, rr, split_rel, conds, c3 in sources:
        doc = json.loads((root / split_rel).read_text())
        meta = {c["case_id"]: c for c in doc["cases"]}
        probs = {cid: T.case_from_json(c).problem for cid, c in meta.items()}
        tabs = {cid: SO.action_table(p) for cid, p in probs.items()}
        for cond in conds:
            eps = [e for e in jl(rr / "eval" / cond / "episodes.jsonl") if e["group"] != "dev"]
            agg = {"episodes": len(eps), "decisions": 0, "raw_err": 0, "exec_err": 0, "rewrite": 0, "no_opt_available": 0, "ep_first_raw": 0, "ep_first_exec": 0, "uniq": set()}
            rows = Counter()
            for e in eps:
                m = meta[e["case_id"]]
                st = SO.episode_error_stats(probs[e["case_id"]], e, cond in c3, tabs[e["case_id"]])
                for k in ("decisions", "raw_err", "exec_err", "rewrite", "no_opt_available"):
                    agg[k] += st[k]
                agg["ep_first_raw"] += st["first_raw_err"] is not None
                agg["ep_first_exec"] += st["first_exec_err"] is not None
                agg["uniq"] |= {(e["case_id"], s) for s in st["unique_raw_err_states"]}
                for scope, cc in (("raw", st["raw_classes"]), ("executed", st["exec_classes"])):
                    for cl, n in cc.items():
                        for gt, gv in (("ALL", "ALL"), ("n", m["n_blocks"]), ("h", m["max_tower_height"]), ("k", m["k_nontrivial_towers"]), ("cell", m["cell"])):
                            rows[(scope, cl, gt, str(gv))] += n
                if st["first_raw_class"]:
                    rows[("first_raw_error_class", st["first_raw_class"], "ALL", "ALL")] += 1
            for (scope, cl, gt, gv), n in sorted(rows.items()):
                cls_rows.append({"source": src, "condition": cond, "scope": scope, "class": cl, "group_type": gt, "group_value": gv, "count": n})
            den_rows.append({"source": src, "condition": cond, "episodes": agg["episodes"], "decisions": agg["decisions"], "raw_nonoptimal_decisions_with_repeats": agg["raw_err"], "executed_nonoptimal_decisions_with_repeats": agg["exec_err"],
                             "rewrites_raw_optimal_to_nonoptimal": agg["rewrite"], "control_left_no_optimal_action": agg["no_opt_available"] if cond in c3 else "NA", "episodes_with_a_raw_error": agg["ep_first_raw"], "episodes_with_an_executed_error": agg["ep_first_exec"],
                             "unique_problem_state_raw_errors": len(agg["uniq"])})
    res = run_root / "results"
    wcsv(res / "error_taxonomy.csv", cls_rows)
    wcsv(res / "error_denominators.csv", den_rows)
    wj(res / "error_taxonomy_notes.json", {"classification": "plan 3.3 priority order, all/exists over the FULL optimal set; implemented here (audit script not obtained): approximate, not a digit-for-digit reproduction of the audit's 102",
                                           "raw": "saved original prediction", "executed": "action actually executed", "rewrites": "single-step immediate rewrites; not a causal effect on the whole trajectory (C3 changes later visited states)",
                                           "history_models": "B2/M0/B2RANK use saved raw predictions (history/time inputs not re-queried)", "sources_not_pooled": True})
    for r in den_rows:
        print(r["source"], r["condition"], r["raw_nonoptimal_decisions_with_repeats"], r["executed_nonoptimal_decisions_with_repeats"], r["rewrites_raw_optimal_to_nonoptimal"], r["unique_problem_state_raw_errors"])


# ------------------------------------------------------------------------------------------------ common-state bank
def cmd_bank(root, run_root, gpu):
    import torch
    from cp_disr.blocksworld import scale_order as SO
    from cp_disr.blocksworld import scorer_control as SC
    from cp_disr.blocksworld import train as T
    from cp_disr.rl import set_suite_half_life
    SO.check_identity(root, run_root)
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    set_suite_half_life(float(td["half_life"]))
    device = torch.device("cuda", 0)
    bank = json.loads((run_root / "prep" / "state_bank.json").read_text())
    reg = json.loads((run_root / "prep" / "registration.json").read_text())
    assert sha_file(run_root / "prep" / "state_bank.json") == reg["state_bank"]["sha256"]
    _, meta = board_cases(root, SO.BOARD_REL)
    cases = {cid: T.case_from_json(c) for cid, c in meta.items()}
    pols = {"MG": SC.load_scorer(root, "MG_C3", device), "PAD": SO.load_pad_scorer(root, run_root, "final", device)}
    rows = []
    for r in bank:
        c = cases[r["case_id"]]
        m = meta[r["case_id"]]
        sc = {k: SO.score_state(p, c, r) for k, p in pols.items()}
        row = {"bank_id": r["bank_id"], "case_id": r["case_id"], "cell": m["cell"], "n": m["n_blocks"], "k": m["k_nontrivial_towers"], "h": m["max_tower_height"], "kind": r["kind"], "L": r["L"],
               "all_optimal_neutral": r["all_optimal_neutral"], "must_destroy": r["must_destroy"], "direct_progress_available": r["direct_progress_available"]}
        for k, s in sc.items():
            row.update({k + "_top1_optimal": s["top1_optimal"], k + "_optimal_mass": round(s["optimal_mass"], 6), k + "_margin": s["margin"], k + "_local_excess": s["local_excess"]})
        row["repaired"] = (not sc["MG"]["top1_optimal"]) and sc["PAD"]["top1_optimal"]
        row["newly_wrong"] = sc["MG"]["top1_optimal"] and not sc["PAD"]["top1_optimal"]
        rows.append(row)
    wcsv(run_root / "results" / "common_state_ranking.csv", rows)

    def summ(pred, name):
        rs = [r for r in rows if pred(r)]
        if not rs:
            return None
        o = {"stratum": name, "states": len(rs)}
        for k in ("MG", "PAD"):
            o[k + "_top1_optimal"] = sum(r[k + "_top1_optimal"] for r in rs)
            o[k + "_mean_optimal_mass"] = round(statistics.mean(r[k + "_optimal_mass"] for r in rs), 4)
            ms = [r[k + "_margin"] for r in rs if r[k + "_margin"] is not None]
            o[k + "_median_margin"] = round(statistics.median(ms), 3) if ms else None
            o[k + "_mean_local_excess"] = round(statistics.mean(r[k + "_local_excess"] for r in rs), 4)
        o["repaired"] = sum(r["repaired"] for r in rs)
        o["newly_wrong"] = sum(r["newly_wrong"] for r in rs)
        return o
    strata = [("ALL", lambda r: True)] + [("kind=" + k, (lambda k: lambda r: r["kind"] == k)(k)) for k in ("init", "p33", "p66", "pre_final", "deviation")] + \
             [("all_optimal_neutral", lambda r: r["all_optimal_neutral"]), ("must_destroy", lambda r: r["must_destroy"]), ("direct_progress_available", lambda r: r["direct_progress_available"]), ("neutral_and_not_must_destroy", lambda r: r["all_optimal_neutral"] and not r["must_destroy"])] + \
             [("n=%d" % n, (lambda n: lambda r: r["n"] == n)(n)) for n in (6, 8)] + [("k=%d" % k, (lambda k: lambda r: r["k"] == k)(k)) for k in (1, 2)] + [("h=%d" % h, (lambda h: lambda r: r["h"] == h)(h)) for h in (2, 4)] + \
             [("cell=" + c[0], (lambda c: lambda r: r["cell"] == c)(c[0])) for c in SO.BOARD_CELLS]
    wcsv(run_root / "results" / "common_state_ranking_summary.csv", [x for x in (summ(p, n) for n, p in strata) if x])
    print(json.dumps(summ(lambda r: True, "ALL")))


# ------------------------------------------------------------------------------------------------ report
def cmd_report(root, run_root):
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import scale_order as SO
    from cp_disr.blocksworld import scorer_control as SC
    res = run_root / "results"
    reg = json.loads((run_root / "prep" / "registration.json").read_text())
    _, meta48 = board_cases(root, SO.BOARD_REL)
    _, meta112 = board_cases(root, SC.FRESH_REL)
    sc_eval = root / SO.SC_RUN_REL / "eval"
    data = {("board48", c): jl(run_root / "eval" / c / "board48.jsonl") for c in ("MG_C3", "PAD_C3", "B_G1C3")}
    data[("fresh112", "PAD_C3")] = jl(run_root / "eval" / "PAD_C3" / "fresh112.jsonl")
    for c in ("MG_C3", "B_G1C3", "MG_G1C3", "B_C3"):
        data[("fresh112", c)] = jl(sc_eval / c / "episodes.jsonl")            # reused: same code, same problems, same weights
    dev = jl(run_root / "eval" / "PAD_RAW" / "dev36.jsonl")
    planner = {e["case_id"]: e for e in jl(run_root / "eval" / "planner" / "board48.jsonl")}
    metas = {"board48": meta48, "fresh112": meta112}
    ep = {k: {e["case_id"]: e for e in v} for k, v in data.items()}
    for (s, c), d in ep.items():
        assert len(d) == len(metas[s]), (s, c, len(d))
    percase = []
    for (s, c), d in ep.items():
        for cid, e in d.items():
            m = metas[s][cid]
            percase.append({"set": s, "condition": c, "case_id": cid, "cell": m["cell"], "n": m["n_blocks"], "k": m["k_nontrivial_towers"], "h": m["max_tower_height"], "L": m["optimal_length"],
                            "initial_goal_atoms_satisfied": m["initial_goal_atoms_satisfied"], "requires_destruction": m["labels"]["requires_goal_destruction"], "success": e["success"], "decision_perfect": e["decision_perfect"],
                            "steps": e["steps"], "step_cap": e["step_cap"], "excess_steps": e["excess_steps"], "reason": e["reason"], "cycle": e["cycle"], "first_divergence": e["first_divergence"], "interventions": e["interventions"],
                            "avoidable_destruction_steps": e["avoidable_destruction_steps"], "wall_seconds": round(e["wall_seconds"], 4)})
    wcsv(res / "development_episodes.csv", percase)
    wcsv(res / "dev36_episodes.csv", [{"case_id": e["case_id"], "success": e["success"], "decision_perfect": e["decision_perfect"], "steps": e["steps"], "reason": e["reason"]} for e in dev])

    def cnt(s, c, pred):
        es = [d for cid, d in ep[(s, c)].items() if pred(metas[s][cid])]
        succ = [e for e in es if e["success"]]
        fail = [e for e in es if not e["success"]]
        return {"n": len(es), "success": len(succ), "perfect": sum(e["decision_perfect"] for e in es), "total_actions": sum(e["steps"] for e in es), "sum_excess_success": sum(e["excess_steps"] for e in succ),
                "failures": len(fail), "failure_reasons": json.dumps(dict(Counter(e["reason"] for e in fail)), sort_keys=True), "failure_actions_spent": sum(e["steps"] for e in fail), "interventions": sum(e["interventions"] for e in es),
                "avoidable_destruction_steps": sum(e["avoidable_destruction_steps"] for e in es), "wall_seconds": round(sum(e["wall_seconds"] for e in es), 2)}
    conds48 = ("MG_C3", "PAD_C3", "B_G1C3")
    fac, costs = [], []
    preds = {"ALL": lambda m: True}
    for cell, n, k, h, _ in SO.BOARD_CELLS:
        preds[cell] = (lambda cell: lambda m: m["cell"] == cell)(cell)
    for n in (6, 8):
        preds["n=%d" % n] = (lambda n: lambda m: m["n_blocks"] == n)(n)
    for k in (1, 2):
        preds["k=%d" % k] = (lambda k: lambda m: m["k_nontrivial_towers"] == k)(k)
    for h in (2, 4):
        preds["h=%d" % h] = (lambda h: lambda m: m["max_tower_height"] == h)(h)
    preds["BREAK"] = lambda m: m["labels"]["requires_goal_destruction"] is True
    preds["MONO"] = lambda m: m["labels"]["requires_goal_destruction"] is False
    for c in conds48:
        for ln, pr in preds.items():
            fac.append({"condition": c, "layer": ln, **cnt("board48", c, pr)})
    wcsv(res / "development_factorial.csv", fac)
    pair = []

    def pair4(s, a, b, pr, field):
        ids = [cid for cid, m in metas[s].items() if pr(m)]
        both = sum(ep[(s, a)][i][field] and ep[(s, b)][i][field] for i in ids)
        oa = sum(ep[(s, a)][i][field] and not ep[(s, b)][i][field] for i in ids)
        ob = sum(ep[(s, b)][i][field] and not ep[(s, a)][i][field] for i in ids)
        return {"n": len(ids), "both": both, "only_first": oa, "only_second": ob, "neither": len(ids) - both - oa - ob, "net": oa - ob}
    pairs = (("board48", "PAD_C3", "MG_C3"), ("board48", "PAD_C3", "B_G1C3"), ("board48", "MG_C3", "B_G1C3"), ("fresh112", "PAD_C3", "MG_C3"), ("fresh112", "PAD_C3", "B_G1C3"), ("fresh112", "MG_C3", "B_G1C3"))
    lay112 = {"ALL": lambda m: True, "T-MONO": lambda m: m["layer"] == "T-MONO", "T-BREAK": lambda m: m["layer"] == "T-BREAK", "H4": lambda m: m["layer"] == "H4", "COLOR": lambda m: m["layer"] == "COLOR"}
    for s, a, b in pairs:
        for ln, pr in (preds if s == "board48" else lay112).items():
            for field, nm in (("success", "S"), ("decision_perfect", "D")):
                p = pair4(s, a, b, pr, field)
                common = [cid for cid, m in metas[s].items() if pr(m) and ep[(s, a)][cid]["success"] and ep[(s, b)][cid]["success"]]
                diff = [ep[(s, a)][i]["steps"] - ep[(s, b)][i]["steps"] for i in common]
                pair.append({"set": s, "first": a, "second": b, "layer": ln, "metric": nm, **p, "common_success_n": len(common), "common_mean_steps_first_minus_second": round(statistics.mean(diff), 3) if diff else None})
    wcsv(res / "paired_comparisons.csv", pair)
    for (s, c) in ep:
        costs.append({"set": s, "condition": c, **cnt(s, c, lambda m: True)})
    wcsv(res / "costs.csv", costs)
    acct = json.loads((run_root / "runs" / SO.RUN_ID / "training_accounting.json").read_text())
    wj(res / "training_cost_summary.json", {k: acct[k] for k in ("optimizer_steps", "decision_samples_shown", "wall_seconds", "mean_epoch_wall_seconds", "nan_events", "slot_epochs_by_block_count", "parameters_trainable")})
    wcsv(res / "planner_reference.csv", [{"case_id": k, "optimal_length": v["optimal_length"], "plan_length": v["plan_length"], "success": v["success"], "expanded_nodes": v["expanded_nodes"]} for k, v in planner.items()])
    # verify
    ledger = jl(run_root / "receipts" / "case_ledger.jsonl")
    started, completed = Counter(), Counter()
    for e in ledger:
        k = (e["actor"], e["split"], e["case_id"])
        started[k] += e["event"] == "STARTED"
        completed[k] += e["event"] == "COMPLETED"
    tech = len({(e["actor"], e["split"], e["case_id"]) for e in ledger if e["event"] == "TECHNICAL_INCOMPLETE"})
    srcid = json.loads((run_root / "prep" / "source_identity.json").read_text())
    wi = {p.name: json.loads(p.read_text()) for p in (run_root / "eval").glob("*/weight_integrity_*.json")}
    verify = {"new_episodes_completed": sum(v for k, v in completed.items() if not k[0].startswith("planner")), "planned": sum(reg["planned_new_episodes"].values()), "planner_completed": sum(v for k, v in completed.items() if k[0].startswith("planner")),
              "technical_incomplete": tech, "ledger_once_each": all(v == 1 for v in started.values()) and all(v == 1 for v in completed.values()), "weights_unchanged_in_eval": all(v["unchanged"] for v in wi.values()),
              "old_roots_unchanged": all(E.git(root, "rev-parse", "HEAD:%s" % rel) == srcid["old_root_tree_ids_at_base"][rel] for rel in SO.OLD_ROOTS), "frozen_sources_unchanged": srcid["all_frozen_unchanged"],
              "new_training_runs": 1, "padding_exceptions": reg["padding_exceptions"], "head": E.git(root, "rev-parse", "HEAD"), "final_checkpoint_sha256": acct["checkpoints"]["final"]["sha256"],
              "m1goal_reference_unchanged": E.sha256_file(root / SC.M1GOAL["path"]) == SC.M1GOAL["sha256"]}
    verify["verdict"] = "PASS" if (verify["new_episodes_completed"] == verify["planned"] and verify["planner_completed"] == len(meta48) and tech == 0 and verify["ledger_once_each"] and verify["weights_unchanged_in_eval"]
                                   and verify["old_roots_unchanged"] and verify["m1goal_reference_unchanged"]) else "FAIL"
    wj(res / "verify.json", verify)
    # numbers
    lines = ["# C1-BW-SCALE-ORDER-PROTOTYPE-V1 results (auto-generated numbers)", "", "Cells are success / all-steps-optimal (n).", ""]
    names = ["ALL"] + [c[0] for c in SO.BOARD_CELLS]
    lines += ["## Board48 (factorial development board)", "", "| condition | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for c in conds48:
        lines.append("| %s | %s |" % (c, " | ".join("%d / %d (%d)" % (lambda x: (x["success"], x["perfect"], x["n"]))(cnt("board48", c, preds[nm])) for nm in names)))
    ln2 = list(lay112)
    lines += ["", "## Fresh 112 (scorer-control suite, now development material)", "", "| condition | " + " | ".join(ln2) + " |", "|---|" + "---|" * len(ln2)]
    for c in ("B_C3", "B_G1C3", "MG_C3", "MG_G1C3", "PAD_C3"):
        lines.append("| %s | %s |" % (c, " | ".join("%d / %d (%d)" % (lambda x: (x["success"], x["perfect"], x["n"]))(cnt("fresh112", c, lay112[nm])) for nm in ln2)))
    lines += ["", "dev36 (PAD raw argmax, no controller): %d success / %d decision-perfect of %d" % (sum(e["success"] for e in dev), sum(e["decision_perfect"] for e in dev), len(dev)), "",
              "Verify: %s (%d/%d new episodes, %d planner references)" % (verify["verdict"], verify["new_episodes_completed"], verify["planned"], verify["planner_completed"]), ""]
    (res / "final_summary_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verify": verify["verdict"], "completed": verify["new_episodes_completed"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "train", "eval", "planner", "taxonomy", "bank", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        if name == "prep":
            p.add_argument("--authorization-text-file", required=True)
            p.add_argument("--procs", type=int, default=48)
        else:
            p.add_argument("--run-root", required=True)
        if name == "eval":
            p.add_argument("--actor", required=True, choices=("MG_C3", "B_G1C3", "PAD_C3", "PAD_RAW"))
            p.add_argument("--set", required=True, choices=("board48", "fresh112", "dev36"))
        if name in ("train", "eval", "bank"):
            p.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    if hasattr(a, "gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    root = Path(a.root).resolve()
    if a.cmd == "prep":
        cmd_prep(root, Path(a.authorization_text_file).read_text(encoding="utf-8"), a.procs)
        return
    rr = Path(a.run_root).resolve()
    if a.cmd == "train":
        cmd_train(root, rr, a.gpu)
    elif a.cmd == "eval":
        cmd_eval(root, rr, a.actor, a.set, a.gpu)
    elif a.cmd == "planner":
        cmd_planner(root, rr)
    elif a.cmd == "taxonomy":
        cmd_taxonomy(root, rr)
    elif a.cmd == "bank":
        cmd_bank(root, rr, a.gpu)
    else:
        cmd_report(root, rr)


if __name__ == "__main__":
    main()
