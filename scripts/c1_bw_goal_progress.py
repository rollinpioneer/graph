#!/usr/bin/env python
"""CP-DISR C1 goal-progress runbook, phase 2 (C1-BW-GOAL-PROGRESS-V1): M0 / M1 / M2 training and the independent Confirm112 evaluation.

    register --gpu N --authorization-text-file F   # D_train (A02 D2 + C0/C3 train rollouts), rank labels, Confirm112, registration, identity
    train  --run-root R --model M0|M1|M2 --gpu N
    eval   --run-root R --condition M0|M1|M2|C0|C3|G1 --gpu N
    planner --run-root R
    report --run-root R
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CONFIRM_REL = "configs/splits/c1_bw_gp_confirm112_v1.json"
NAMESPACE = "C1-BW-GP-CONFIRM-v1"
# (slice, n, goal heights, top colour: 0 = training pattern RED top, 1 = BLUE top = colour reversal)
CONFIRM_SLICES = (("A1-4", 4, (2, 1, 1), 1), ("A1-5", 5, (3, 1, 1), 1), ("C5-1", 5, (2, 1, 1, 1), 0), ("C5-2", 5, (2, 2, 1), 0), ("C5-23", 5, (3, 2), 0), ("S6-2", 6, (2, 2, 1, 1), 0),
                  ("S7-2", 7, (2, 2, 1, 1, 1), 0), ("S8-2", 8, (2, 2, 1, 1, 1, 1), 0), ("K6-1", 6, (2, 1, 1, 1, 1), 0), ("K6-3", 6, (2, 2, 2), 0), ("H6-13", 6, (3, 1, 1, 1), 0),
                  ("H6-14", 6, (4, 1, 1), 0), ("H6-23", 6, (3, 3), 0), ("H6-24", 6, (4, 2), 0))
PER_SLICE = 8
MAX_CANDIDATES = 10000
NEW_SOURCES = ("src/cp_disr/blocksworld/goal_progress.py", "scripts/c1_bw_goal_progress.py", "tests/test_c1_bw_goal_progress.py", "configs/splits/c1_bw_gp_confirm112_v1.json")
CONDITIONS = ("M0", "M1", "M2", "C0", "C3", "G1")


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


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seen_hashes(root, d_train, cases):
    """Problem classes of every earlier set (train/dev, A0-B, Pilot80) and of every hand-empty (colours, state, goal) problem occurring in D_train."""
    from cp_disr.blocksworld import canonical as K
    from cp_disr.blocksworld import state as S
    forbidden = set()
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    forbidden |= {c["problem_iso_hash"] for c in td["train"] + td["dev"]}
    for f in ("c1_bw_a0_iso_v1", "c1_bw_a1_color_reverse_v1", "c1_bw_a2_noniso_v1", "c1_bw_b_scale_v1", "c1_bw_a04p_pilot80_v1"):
        forbidden |= {c["problem_iso_hash"] for c in json.loads((root / ("configs/splits/%s.json" % f)).read_text())["cases"]}
    by_id = {c.case_id: c for c in cases}
    n0 = len(forbidden)
    for t in d_train:
        p = by_id[t["case_id"]].problem
        for s in t["states"][:-1]:
            if S.held(tuple(s)) == -1 and tuple(s) != tuple(p.goal):
                forbidden.add(K.problem_iso_hash(S.Problem(p.names, p.colors, tuple(s), p.goal)))
    return forbidden, n0


def build_confirm(root, d_train, cases):
    from cp_disr.blocksworld import canonical as K
    from cp_disr.blocksworld import contracts as C
    from cp_disr.blocksworld import generator as Gen
    from cp_disr.blocksworld import goal_probe as GP
    from cp_disr.blocksworld import splits as SP
    from cp_disr.blocksworld import state as S
    b = SP.Builder()
    forbidden, n0 = seen_hashes(root, d_train, cases)
    out, report = [], {}
    for name, n, heights, top in CONFIRM_SLICES:
        made, attempt = 0, 0
        top_color = S.RED if top == 0 else S.BLUE
        while made < PER_SLICE and attempt < MAX_CANDIDATES:
            p = Gen.make_problem("%s:%s" % (NAMESPACE, name), n, made, heights, top_color, attempt, p_stack=0.88 if n >= 6 else 0.55)
            attempt += 1
            if p is None:
                continue
            h = K.problem_iso_hash(p)
            if h in forbidden:
                continue
            q = b.qualify(p, need_destruction=True)
            if q is None:
                continue
            comps = [len(c) for c in GP.goal_components(p.goal)]
            extra = {"slice": name, "goal_heights": sorted(heights, reverse=True), "k_nontrivial_towers": sum(1 for t in comps if t >= 2), "max_tower_height": max(comps), "n_goal_atoms": n,
                     "n_candidates": len(C.template_for(p).contracts), "namespace": NAMESPACE}
            forbidden.add(h)
            out.append(SP.case_record("BW_gp_%s_%03d" % (name.replace("-", "").lower(), made), "gp_confirm", p, q, extra))
            made += 1
        report[name] = {"planned": PER_SLICE, "built": made, "candidates_tried": attempt, "shortfall": PER_SLICE - made}
        print(json.dumps({name: report[name]}), flush=True)
    doc = {"namespace": NAMESPACE, "excluded_problem_classes_at_start": n0, "slices": report, "cases": out, "counts": len(out), "note": "independent of Pilot80; frozen before any new model is trained"}
    wj(root / CONFIRM_REL, doc)
    return doc


def collect_and_register(root, gpu, auth_text):
    import torch
    from cp_disr.blocksworld import a04p_controls as AC
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import goal_progress as GP
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld.environment import BwEpisode
    from cp_disr.rl import set_suite_half_life
    if E.git(root, "branch", "--show-current") != R.BRANCH or E.git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("clean tracked tree on %s required" % R.BRANCH)
    cases, half_life = I.load_train_cases(root / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(half_life)
    a02 = root / E.A02_ROOT / "datasets"
    d0 = json.loads((a02 / "D0.json").read_text())
    r1 = {t: json.loads((a02 / ("rollouts_round1_%s.json" % t)).read_text()) for t in I.TAGS}
    r2 = {t: json.loads((a02 / ("rollouts_round2_%s.json" % t)).read_text()) for t in I.TAGS}
    d2 = I.aggregate(I.aggregate(d0, r1), r2)
    d2_id = I.dataset_identity(d2)
    assert d2_id["sha256"] == (a02 / "D2_sha256.txt").read_text().strip(), "A02 D2 hash mismatch"
    device = torch.device("cuda", 0)
    policy = E.load_model(root, "B2", device)
    solver = P.Solver()
    shared = []
    for ctrl in ("C0", "C3"):
        for c in cases:
            res = AC.run_episode(policy, c, solver, ctrl)
            ep = BwEpisode(c)
            states = [ep.state]
            for d in res["decisions"]:
                ep.step(d["selected"])
                states.append(ep.state)
            shared.append({"tid": "%s_final:%s" % (ctrl, c.case_id), "source": "%s_final" % ctrl, "case_id": c.case_id, "n_blocks": c.problem.n, "step_cap": c.step_cap, "states": [list(s) for s in states],
                           "actions": [d["selected"] for d in res["decisions"]], "astar": [d["optimal_actions"] for d in res["decisions"]], "remaining": [d["optimal_remaining"] for d in res["decisions"]], "success": res["success"]})
    shared = [t for t in shared if t["actions"]]
    d_train = d2 + shared
    I.verify_labels(d_train, cases)
    labels = GP.build_rank_labels(d_train, cases)
    confirm = build_confirm(root, d_train, cases)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = root / R.GP_REL / "train_confirm" / ("%s_%s" % (stamp, E.sha256_file(root / R.RUNBOOK_REL)[:8]))
    for sub in ("datasets", "runs", "eval", "receipts", "results"):
        (run_root / sub).mkdir(parents=True)
    (run_root / "receipts" / "case_ledger.jsonl").write_text("")
    wj(run_root / "datasets" / "D_train.json", d_train)
    wj(run_root / "datasets" / "rank_labels.json", labels)
    ident = I.dataset_identity(d_train)
    train_ident = {"D_train": ident, "A02_D2": d2_id, "shared_rollouts": {"C0_final": sum(1 for t in shared if t["source"] == "C0_final"), "C3_final": sum(1 for t in shared if t["source"] == "C3_final")},
                   "case_scope": "train144 single-tower goals only", "d_train_file_sha256": E.sha256_file(run_root / "datasets" / "D_train.json")}
    wj(run_root / "training_dataset_identity.json", train_ident)
    lab_ident = {"states_labelled": len(labels), "sha256": sha_json(labels), "file_sha256": E.sha256_file(run_root / "datasets" / "rank_labels.json"),
                 "rule": "d_a = exact d*(T(F,a),G) for every legal a; A* = argmin; labels never enter a forward pass"}
    wj(run_root / "candidate_rank_labels_identity.json", lab_ident)
    hyper = {"init_checkpoint": "A02_B2_final", "optimizer_reset": True, "epochs": GP.EPOCHS, "trajectory_batch_size": GP.BATCH, "optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
             "existing_module_lr": GP.EXISTING_LR, "new_head_lr": GP.NEW_HEAD_LR, "grad_clip": GP.GRAD_CLIP, "shuffle_seed": GP.SHUFFLE_SEED, "new_head_init_seed": GP.HEAD_SEED, "rank_weight": GP.RANK_WEIGHT,
             "entropy_bonus": 0, "additional_collection_rounds_after_freeze": 0, "decision_chunk": GP.CHUNK, "x_dim": GP.X_DIM}
    reg = {"card": "C1-BW-GOAL-PROGRESS-V1", "runbook": R.RUNBOOK_REL, "runbook_sha256": E.sha256_file(root / R.RUNBOOK_REL), "base_commit": R.BASE_COMMIT, "branch": R.BRANCH, "phase1_registration_commit": "8c8c32e1c9bc1ebbce596bcd9e3d7b340be2c8eb",
           "phase1_result_commit": "fe611a4a5e62eb8e59a8a22084cb48062062b40e", "phase1_state": "GOAL_PROGRESS_READY", "authorization_text": auth_text, "authorization_text_sha256": hashlib.sha256(auth_text.encode()).hexdigest(),
           "runs": GP.RUN_IDS, "training_run_cap": 3, "hyperparameters": hyper, "training": {"dataset": "D_train.json (A02 D2 + C0/C3 train144 rollouts of the A02 B2, collected once)", "d_train_sha256": ident["sha256"], "rank_labels_sha256": lab_ident["sha256"]},
           "losses": {"M0": "L_IL = -log sum_{a in A*} pi(a|F,G) (original B2 policy, GRU path kept)", "M1_M2": "L_IL over softmax(score) + L_rank (mean pairwise softplus over strictly ordered legal candidates per decision)"},
           "confirm": {"path": CONFIRM_REL, "sha256": E.sha256_file(root / CONFIRM_REL), "slices": confirm["slices"], "counts": confirm["counts"], "core_slices": list(GP.CORE)},
           "label_rules": {"ADVANTAGE_D_core_M2_minus_M1": GP.ADVANTAGE, "CLOSE_D_core": GP.CLOSE, "protected_A1_max_success_deficit_vs_C0": 1, "min_core_slices_M2_ge_M1": 4,
                           "note": "section-13 labels are computed only after all six conditions are complete; 'close' is registered as a D_core gap < 0.10"},
           "conditions": list(CONDITIONS), "confirm_episodes_planned": 6 * confirm["counts"], "planner_references_planned": confirm["counts"], "dev36_models": ["M0", "M1", "M2"],
           "no_score_gate": True, "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    wj(run_root / "registration.json", reg)
    wj(run_root / "source_identity.json", R.source_identity(root, NEW_SOURCES + PHASE1_SOURCES))
    wj(run_root / "checkpoint_identity.json", {"init": R.B2_CKPT["path"], "init_sha256": R.B2_CKPT["sha256"], "committed_to_git": False})
    wj(run_root / "split_manifest.json", {"confirm": {s: [c["case_id"] for c in confirm["cases"] if c["slice"] == s] for s in confirm["slices"]}})
    print(json.dumps({"run_root": str(run_root), "d_train": ident["trajectories"], "decisions": ident["decisions"], "labels": len(labels), "confirm": confirm["counts"]}))


PHASE1_SOURCES = ("src/cp_disr/blocksworld/a04p_controls.py", "src/cp_disr/blocksworld/goal_probe.py", "src/cp_disr/blocksworld/a04p_registry.py", "scripts/c1_bw_a04p.py")


def counts_of(model):
    return sum(p.numel() for p in model.parameters())


def cmd_train(root, run_root, model_id, gpu):
    import torch
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import goal_progress as GP
    from cp_disr.blocksworld import imitation as I
    from cp_disr.torch_rl import save_checkpoint
    R.check_identity(root, run_root)
    reg = json.loads((run_root / "registration.json").read_text())
    out = run_root / "runs" / GP.RUN_IDS[model_id]
    try:
        (out / "checkpoints").mkdir(parents=True, exist_ok=False)                           # a second start of the same run is refused
    except FileExistsError:
        raise SystemExit("duplicate start refused: %s" % out)
    d_train_path = run_root / "datasets" / "D_train.json"
    if E.sha256_file(d_train_path) != json.loads((run_root / "training_dataset_identity.json").read_text())["d_train_file_sha256"]:
        raise SystemExit("D_train changed")
    cases, half_life = I.load_train_cases(root / "configs/splits/c1_bw_train_dev_v1.json")
    from cp_disr.rl import set_suite_half_life
    set_suite_half_life(half_life)
    d_train = json.loads(d_train_path.read_text())
    assert I.dataset_identity(d_train)["sha256"] == reg["training"]["d_train_sha256"]
    device = torch.device("cuda", 0)
    base = E.load_model(root, "B2", device)
    t0 = time.time()
    rows, init_hash = [], None
    if model_id == "M0":
        policy = base
        tr = I.ImitationTrainer(policy, "b2", cases, device)
        tr.optimizer = torch.optim.Adam(tr.params, lr=GP.EXISTING_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)       # optimizer reset, lr 1e-4
        net, epoch_fn = policy, lambda ep: tr.epoch(d_train, GP.SHUFFLE_SEED * 100003 + ep)
        trainer = tr
        trainable = sum(p.numel() for p in tr.params)
    else:
        labels = json.loads((run_root / "datasets" / "rank_labels.json").read_text())
        mode = "global" if model_id == "M1" else "additive"
        net = GP.GoalProgressModel(base, mode).to(device)
        trainer = GP.GPTrainer(net, cases, labels, device)
        epoch_fn = lambda ep: trainer.epoch(d_train, GP.SHUFFLE_SEED * 100003 + ep)
        trainable = sum(p.numel() for p in trainer.params)
    head_init = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in net.heads.parameters())).hexdigest() if model_id != "M0" else None
    ckpts = {}
    for ep in range(GP.EPOCHS):
        row = epoch_fn(ep)
        row.update({"epoch": ep + 1, "optimizer_steps": trainer.steps, "wall_seconds": round(time.time() - t0, 1)})
        rows.append(row)
        if ep == 0 or (ep + 1) % 10 == 0:
            print("[gp] %s epoch %d %s" % (model_id, ep + 1, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}), flush=True)
        with open(out / "epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if ep + 1 in (50, GP.EPOCHS):
            name = "final" if ep + 1 == GP.EPOCHS else "epoch050_trace_only"
            path = out / "checkpoints" / ("%s.pt" % name)
            save_checkpoint(path, net, trainer.optimizer, {"model": model_id, "epoch": ep + 1, "optimizer_steps": trainer.steps})
            ckpts[name] = {"path": str(path.relative_to(root)), "sha256": E.sha256_file(path), "bytes": path.stat().st_size}
    acct = {"model": model_id, "run_id": GP.RUN_IDS[model_id], "epochs": GP.EPOCHS, "optimizer_steps": trainer.steps, "decision_samples_shown": sum(r["decisions"] for r in rows), "nan_events": trainer.nan_events,
            "wall_seconds": round(time.time() - t0, 1), "d_train_sha256": reg["training"]["d_train_sha256"], "rank_labels_sha256": reg["training"]["rank_labels_sha256"], "parameters_total": counts_of(net),
            "parameters_trainable": trainable, "new_head_parameters": (sum(p.numel() for p in net.heads.parameters()) if model_id != "M0" else 0), "new_head_initial_sha256": head_init, "checkpoints": ckpts,
            "final_epoch": rows[-1], "init_checkpoint_sha256": R.B2_CKPT["sha256"], "evaluation_checkpoint": "final (epoch 100); epoch050 is trace-only and never evaluated"}
    wj(out / "training_accounting.json", acct)
    print(json.dumps({"done": model_id, "steps": trainer.steps, "wall": acct["wall_seconds"]}))


def cmd_eval(root, run_root, cond, gpu):
    import torch
    from cp_disr.blocksworld import a04p_controls as AC
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import goal_progress as GP
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import train as T
    from cp_disr.torch_rl import load_checkpoint
    R.check_identity(root, run_root)
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    from cp_disr.rl import set_suite_half_life
    set_suite_half_life(float(td["half_life"]))
    device = torch.device("cuda", 0)
    controller = "C0"
    if cond in ("C0", "C3", "G1"):
        policy, controller = E.load_model(root, "B2", device), cond
    else:
        acct = json.loads((run_root / "runs" / GP.RUN_IDS[cond] / "training_accounting.json").read_text())
        ck = root / acct["checkpoints"]["final"]["path"]
        assert E.sha256_file(ck) == acct["checkpoints"]["final"]["sha256"]
        if cond == "M0":
            policy = I.make_imitation_policy("B2-CACHED", device, 0)
            load_checkpoint(ck, policy)
            policy.eval()
        else:
            model = GP.GoalProgressModel(E.load_model(root, "B2", device), "global" if cond == "M1" else "additive").to(device)
            load_checkpoint(ck, model)
            model.eval()
            policy = GP.ScorePolicy(model)
    conf = json.loads((root / CONFIRM_REL).read_text())
    by = defaultdict(list)
    for c in conf["cases"]:
        by[c["slice"]].append(T.case_from_json(c))
    groups = [(s, by[s]) for s in [x[0] for x in CONFIRM_SLICES] if s in by]
    if cond in ("M0", "M1", "M2"):
        groups.append(("dev", [T.case_from_json(c) for c in td["dev"]]))               # ID dev36: read once per final model, no gate
    solver = P.Solver()
    print(json.dumps(R.run_cases(run_root, cond, groups, lambda c: AC.run_episode(policy, c, solver, controller), run_root / "eval" / cond / "episodes.jsonl")))


def cmd_planner(root, run_root):
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld import train as T
    R.check_identity(root, run_root)
    conf = json.loads((root / CONFIRM_REL).read_text())
    by = defaultdict(list)
    for c in conf["cases"]:
        by[c["slice"]].append(T.case_from_json(c))

    def ref(c):
        s = P.Solver()
        t0 = time.perf_counter()
        plan = s.one_optimal_plan(c.problem.init, c.problem.goal)
        cpu = time.perf_counter() - t0
        st = c.problem.init
        for a in plan:
            st = S.apply(st, a)
        return {"case_id": c.case_id, "n_blocks": c.problem.n, "optimal_length": c.optimal_length, "plan_length": len(plan), "success": S.goal_satisfied(c.problem, st), "expanded_nodes": s.stats.expanded,
                "generated_nodes": s.stats.generated, "wall_seconds_cold_cache": cpu}
    print(json.dumps(R.run_cases(run_root, "planner", [(s, by[s]) for s in [x[0] for x in CONFIRM_SLICES]], ref, run_root / "eval" / "planner" / "episodes.jsonl")))


def cmd_report(root, run_root):
    from cp_disr.blocksworld import a04p_controls as AC
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import goal_progress as GP
    res = run_root / "results"
    reg = json.loads((run_root / "registration.json").read_text())
    conf = json.loads((root / CONFIRM_REL).read_text())
    meta = {c["case_id"]: c for c in conf["cases"]}
    eps = {c: jl(run_root / "eval" / c / "episodes.jsonl") for c in CONDITIONS}
    table, strata_rows, percase = {}, [], []
    order = [s for s, *_ in CONFIRM_SLICES]
    for c in CONDITIONS:
        for s in order + ["dev"]:
            es = [e for e in eps[c] if e["group"] == s]
            if not es:
                continue
            table[(c, s)] = {"n": len(es), "success": sum(e["success"] for e in es), "perfect": sum(e["decision_perfect"] for e in es)}
        for e in eps[c]:
            m = meta.get(e["case_id"], {})
            percase.append({"condition": c, "slice": e["group"], "case_id": e["case_id"], "n_blocks": e["n_blocks"], "k_towers": m.get("k_nontrivial_towers"), "max_height": m.get("max_tower_height"),
                            "optimal_length": e["optimal_length"], "success": e["success"], "decision_perfect": e["decision_perfect"], "steps": e["steps"], "excess_steps": e["excess_steps"], "cycle": e["cycle"],
                            "first_divergence": e["first_divergence"], "interventions": e["interventions"], "reason": e["reason"], "destroyed_satisfied_goal_count": e["destroyed_satisfied_goal_count"]})
    wcsv(res / "per_case.csv", percase)
    main = []
    for c in CONDITIONS:
        row = {"condition": c}
        if (c, "dev") in table:
            row["ID_dev_success"] = "%d/%d" % (table[(c, "dev")]["success"], table[(c, "dev")]["n"])
            row["ID_dev_perfect"] = "%d/%d" % (table[(c, "dev")]["perfect"], table[(c, "dev")]["n"])
        for s in order:
            t = table[(c, s)]
            row[s + "_success"] = "%d/%d" % (t["success"], t["n"])
            row[s + "_perfect"] = "%d/%d" % (t["perfect"], t["n"])
        row["D_core"] = round(GP.core_rates(table, c, "perfect"), 4)
        row["S_core"] = round(GP.core_rates(table, c, "success"), 4)
        main.append(row)
    wcsv(res / "id_and_confirm_table.csv", main)
    for c in CONDITIONS:
        groups = defaultdict(list)
        for r in percase:
            if r["condition"] == c and r["slice"] != "dev":
                groups[("n=%s k=%s h=%s" % (r["n_blocks"], r["k_towers"], r["max_height"]))].append(r)
                lb = "L<=8" if r["optimal_length"] <= 8 else ("L9-12" if r["optimal_length"] <= 12 else ("L13-16" if r["optimal_length"] <= 16 else "L>16"))
                groups[lb].append(r)
        for k, rs in sorted(groups.items()):
            strata_rows.append({"condition": c, "stratum": k, "n": len(rs), "success": sum(r["success"] for r in rs), "decision_perfect": sum(r["decision_perfect"] for r in rs), "cycle_episodes": sum(r["cycle"] for r in rs),
                                "mean_optimal_length": round(statistics.mean(r["optimal_length"] for r in rs), 2)})
    wcsv(res / "n_k_h_l_strata.csv", strata_rows)
    labels = GP.direction_labels(table)
    wj(res / "direction_labels.json", labels)
    # first-error types from each condition's own trajectories
    cases_by_id = {c["case_id"]: c for c in conf["cases"]}
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld import train as T
    fe = []
    for c in CONDITIONS:
        for e in eps[c]:
            if e["group"] == "dev":
                continue
            prob = T.case_from_json(cases_by_id[e["case_id"]]).problem
            f = AC.first_error_fields(e["decisions"], prob)
            fe.append({"condition": c, "slice": e["group"], "case_id": e["case_id"], "success": e["success"], **f})
    wcsv(res / "first_error_by_condition.csv", fe)
    accts = {m: json.loads((run_root / "runs" / GP.RUN_IDS[m] / "training_accounting.json").read_text()) for m in ("M0", "M1", "M2")}
    wj(run_root / "training_accounting.json", accts)
    wj(run_root / "model_parameter_and_input_identity.json", {m: {k: a[k] for k in ("parameters_total", "parameters_trainable", "new_head_parameters", "new_head_initial_sha256", "d_train_sha256", "rank_labels_sha256", "init_checkpoint_sha256")} for m, a in accts.items()})
    ledger = jl(run_root / "receipts" / "case_ledger.jsonl")
    from collections import Counter
    st, cp = Counter(), Counter()
    for e in ledger:
        k = (e["actor"], e["split"], e["case_id"])
        if e["event"] == "STARTED":
            st[k] += 1
        if e["event"] == "COMPLETED":
            cp[k] += 1
    tech = len({(e["actor"], e["split"], e["case_id"]) for e in ledger if e["event"] == "TECHNICAL_INCOMPLETE"})
    conf_done = sum(v for k, v in cp.items() if k[0] in CONDITIONS and k[1] != "dev")
    verify = {"confirm_episodes_completed": conf_done, "confirm_episodes_planned": reg["confirm_episodes_planned"], "dev36_completed": sum(v for k, v in cp.items() if k[1] == "dev"), "technical_incomplete": tech,
              "ledger_once_each": all(v == 1 for v in st.values()) and all(v == 1 for v in cp.values()), "planner_completed": sum(v for k, v in cp.items() if k[0] == "planner"),
              "old_roots_unchanged": all(E.git(root, "rev-parse", "HEAD:%s" % rel) == json.loads((run_root / "source_identity.json").read_text())["old_root_tree_ids_at_base"][rel] for rel in R.OLD_ROOTS),
              "head": E.git(root, "rev-parse", "HEAD"), "b2_checkpoint_unchanged": E.sha256_file(root / R.B2_CKPT["path"]) == R.B2_CKPT["sha256"]}
    verify["verdict"] = "PASS" if conf_done == reg["confirm_episodes_planned"] and verify["dev36_completed"] == 108 and tech == 0 and verify["ledger_once_each"] and verify["old_roots_unchanged"] and verify["b2_checkpoint_unchanged"] else "FAIL"
    wj(res / "verify.json", verify)
    print(json.dumps({"verify": verify["verdict"], "labels": labels["labels"], "D_core": labels["D_core"], "S_core": labels["S_core"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("register", "train", "eval", "planner", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        if name == "register":
            p.add_argument("--authorization-text-file", required=True)
        else:
            p.add_argument("--run-root", required=True)
        if name == "train":
            p.add_argument("--model", required=True, choices=("M0", "M1", "M2"))
        if name == "eval":
            p.add_argument("--condition", required=True, choices=CONDITIONS)
        if name in ("register", "train", "eval"):
            p.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    if hasattr(a, "gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    root = Path(a.root).resolve()
    if a.cmd == "register":
        collect_and_register(root, a.gpu, Path(a.authorization_text_file).read_text(encoding="utf-8"))
    else:
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
