"""Extension commands of the compact-diagnosis driver."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GOOSE_PY = str(Path.home() / "envs" / "goose" / "bin" / "python")
GOOSE_LD = str(Path.home() / ".uv_python" / "cpython-3.10.19-linux-x86_64-gnu" / "lib")


def _cfg():
    import yaml
    return yaml.safe_load((ROOT / "configs" / "c1_compact_diagnosis_v2.yaml").read_text())


def _ledger(rr, stage, **kv):
    p = Path(rr) / "receipts" / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stage=stage, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kv), default=str) + "\n")


def _wj(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def goose_run(args, timeout):
    env = {**os.environ, "LD_LIBRARY_PATH": GOOSE_LD, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    return subprocess.run([GOOSE_PY, str(ROOT / "scripts" / "c1_compact_w1_fit.py")] + args, capture_output=True, text=True, env=env, timeout=timeout)


def w1_fixture(rr):
    """Fixture 5: D0-aligned pair list, weights, no hidden second fit (dry run), typed state mapping == deployed adapter."""
    from cp_disr.pddl.compact import w1 as W1
    from cp_disr.pddl.search_match import evaluators as EV
    from cp_disr.pddl.search_match.budget import Budget
    from cp_disr.pddl import depots as DP, stages as ST
    from cp_disr.pddl.compact import train as CT
    rr = Path(rr)
    cfg = _cfg()
    root = ROOT / cfg["old_run_root"]
    params = root / "goose" / "train_typed" / "wl_goose.model.params"
    d = rr / "checks" / "w1_fixture"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = W1.export(root, params, d / "data.json", max_cases=4)
    r = goose_run([str(d / "data.json"), str(d / "dry.params"), "--report", str(d / "dry_report.json"), "--dry-run"], 600)
    rep = json.loads((d / "dry_report.json").read_text()) if (d / "dry_report.json").is_file() else {}
    data = json.loads((d / "data.json").read_text())
    # deployed adapter scores of the same states
    wl = EV.WlEval(str(d / "dry.params"))
    maxdiff, n = 0.0, 0
    man, exact, _ = CT.load_old(root)
    byid = {c["case_id"]: c for c in man["train"]}
    for p, off in zip(data["problems"], rep.get("offsets", [])):
        c = byid[p["case_id"]]
        task = ST.get_task(DP.DOMAIN_TYPED, c["file"])
        wl.prepare(task, {"domain": str(DP.DOMAIN_TYPED), "problem": c["file"]}, Budget(120, 10 ** 6, 8 * 2 ** 30))
        for k in range(0, len(p["states"]), max(1, len(p["states"]) // 12)):
            s = int(p["state_masks_hex"][k], 16)
            maxdiff = max(maxdiff, abs(wl.raw(s) - rep["scores"][off + k]))
            n += 1
    # pair semantics: recompute strict pair count from the labels
    pairs_from_labels = 0
    for c in [c for c in man["train"]][:4]:
        ex = exact[c["case_id"]]
        seen = set()
        for t in ex["trajectories"]:
            for s in t["states"][:-1]:
                k = ",".join(map(str, s))
                if k in seen:
                    continue
                seen.add(k)
                dist = ex["rank_labels"]["%s|%s" % (c["case_id"], k)]
                pairs_from_labels += sum(1 for a in dist for b in dist if dist[a] < dist[b])
    out = {"export_stats": {k: v for k, v in stats.items()}, "dry_run_returncode": r.returncode, "dry_run_tail": (r.stdout + r.stderr)[-300:], "fits_executed_in_fixture": rep.get("fits"), "features": rep.get("features"),
           "deployed_vs_training_embedding_max_abs_diff": maxdiff, "states_compared": n, "strict_pairs_from_labels": pairs_from_labels, "pairs_raw_exported": stats["pairs_raw"],
           "pairs_equal_up_to_identical_successors": stats["pairs_raw"] <= pairs_from_labels and stats["pairs_raw"] >= 0.95 * pairs_from_labels, "weight_mass": stats["weight_mass"], "track": "typed", "ipc_track": "NOT_COMPARABLE (one fit only)",
           "seconds": round(time.time() - t0, 1)}
    out["ok"] = bool(r.returncode == 0 and out["fits_executed_in_fixture"] == 0 and maxdiff < 1e-4 and n > 20 and out["pairs_equal_up_to_identical_successors"])
    _wj(rr / "checks" / "fixture_w1.json", out)
    _ledger(rr, "fixture_w1", seconds=out["seconds"], passed=out["ok"], fits=0, cpu_core_hours=round(2 * out["seconds"] / 3600, 3))
    print("fixture W1 ok=%s maxdiff=%.2e states=%d pairs=%d/%d" % (out["ok"], maxdiff, n, stats["pairs_raw"], pairs_from_labels))
    return 0 if out["ok"] else 3


def w1_fit(rr):
    """THE single W1 fit."""
    from cp_disr.pddl.compact import w1 as W1
    rr = Path(rr)
    cfg = _cfg()
    root = ROOT / cfg["old_run_root"]
    d = rr / "training" / "W1"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "wl_goose_w1.model.params").is_file():
        print("W1 already fitted")
        return 0
    t0 = time.time()
    stats = W1.export(root, root / "goose" / "train_typed" / "wl_goose.model.params", d / "data.json")
    t1 = time.time()
    r = goose_run([str(d / "data.json"), str(d / "wl_goose_w1.model.params"), "--report", str(d / "fit_report.json")], 3600)
    (d / "fit.log").write_text(r.stdout + r.stderr)
    ok = r.returncode == 0 and (d / "wl_goose_w1.model.params").is_file()
    wall = time.time() - t0
    rep = json.loads((d / "fit_report.json").read_text()) if (d / "fit_report.json").is_file() else {}
    from cp_disr.pddl.compact.train import sha_file
    acct = {"status": "DONE" if ok else "TECHNICAL_INCOMPLETE", "fits": 1, "export_seconds": round(t1 - t0, 1), "fit": rep, "export_stats": stats, "wall_seconds": round(wall, 1), "threads": 2, "wall_cap_seconds": 3600,
            "params_sha256": sha_file(d / "wl_goose_w1.model.params") if ok else None, "opts_copied_from": "goose/train_typed/wl_goose.model.opts (same configuration; no hyper-parameter was chosen or changed)"}
    _wj(d / "fit_accounting.json", acct)
    if ok:
        import shutil
        shutil.copy(root / "goose" / "train_typed" / "wl_goose.model.opts", d / "wl_goose_w1.model.opts")
    _ledger(rr, "w1_fit", seconds=round(wall, 1), cpu_core_hours=round(2 * wall / 3600, 3), fits=1, ok=ok)
    print("W1 fit ok=%s %s" % (ok, json.dumps(rep)))
    return 0 if ok else 4


# ------------------------------------------------------------------------------------------------ shared state library
def make_evaluator(rr, name, device_name="cpu"):
    """Scorer factory used by every stage: frozen old models, selected D0 / C0, W1."""
    import torch
    from cp_disr.pddl.search_match import evaluators as EV
    from cp_disr.pddl.search_match.runner import OldRun, load_neural
    cfg = _cfg()
    old = OldRun(ROOT / cfg["old_run_root"])
    dev = torch.device(device_name)
    if name in ("MG", "DENSE", "REL"):
        m, _ck = load_neural(old, "V_" + name, dev)
        return EV.NeuralEval(name, m, dev)
    if name == "WL":
        return EV.WlEval(old.wl_params("struct")["params"])
    if name in ("D0", "C0", "T1"):
        from cp_disr.pddl.compact import train as CT
        sel = json.loads((Path(rr) / "training" / name / "selection.json").read_text())
        m = CT.load_model_v2({"D0": "dense", "C0": "c0", "T1": "dense"}[name], sel["checkpoint"]["path"], dev)
        return EV.NeuralEval(name, m, dev)
    if name == "W1":
        return EV.WlEval(str(Path(rr) / "training" / "W1" / "wl_goose_w1.model.params"))
    raise ValueError(name)


def fixture_b(rr):
    """Fixture 3 + equivariance construction: label relations (strict / equal / unknown), transformed problems are the same problem (ground sizes, init mapping, exact distance)."""
    from cp_disr.pddl import depots as DP, stages as ST, task as T
    from cp_disr.pddl.compact import diagb as DB, statelib as SL
    from cp_disr.pddl.compact import train as CT
    rr = Path(rr)
    cfg = _cfg()
    t0 = time.time()
    ex = lambda lo, up, src="UNKNOWN": {"source": src, "lower": lo, "upper": up}
    rel = {"strict_x": SL.relation(ex(3, 3, "EXACT"), ex(5, 5, "EXACT")), "strict_y": SL.relation(ex(5, 5, "EXACT"), ex(3, 3, "EXACT")), "equal": SL.relation(ex(4, 4, "EXACT"), ex(4, 4, "EXACT")),
           "upper_vs_exact": SL.relation(ex(None, 3, "UPPER_ONLY"), ex(5, 5, "EXACT")), "overlap_is_unknown": SL.relation(ex(None, 6, "UPPER_ONLY"), ex(5, 5, "EXACT")), "touching_is_unknown": SL.relation(ex(None, 5, "UPPER_ONLY"), ex(5, 5, "EXACT")),
           "unknown_end": SL.relation(ex(None, None), ex(5, 5, "EXACT"))}
    expect = {"strict_x": "X_BETTER", "strict_y": "Y_BETTER", "equal": "EQUAL", "upper_vs_exact": "X_BETTER", "overlap_is_unknown": "UNKNOWN", "touching_is_unknown": "UNKNOWN", "unknown_end": "UNKNOWN"}
    root = ROOT / cfg["old_run_root"]
    man, exact, _ = CT.load_old(root)
    checks = []
    domain_text = DP.DOMAIN_TYPED.read_text()
    for c in sorted(man["struct"], key=lambda c: c["sha256"])[:3] + sorted(man["train"], key=lambda c: c["sha256"])[:2]:
        text = Path(c["file"]).read_text()
        ta = ST.get_task(DP.DOMAIN_TYPED, c["file"])
        for kind, seed in (("rename", 1), ("rename", 2), ("goalperm", 3)):
            ntext, phi = DB.transformed_problem(text, kind, seed)
            tb = T.StripsTask(domain_text, ntext, type_map=T.DEPOTS_TYPE_MAP, type_preds=T.DEPOTS_TYPE_PREDS, type_order=T.DEPOTS_TYPE_ORDER)
            sb = DB.map_state(ta, ta.init_mask, tb, phi)
            oa, ob = T.ExactOracle(ta), T.ExactOracle(tb)
            da, db = oa.query(ta.init_mask)[0], ob.query(tb.init_mask)[0]
            oa.close(), ob.close()
            checks.append({"case": c["case_id"], "kind": kind, "seed": seed, "dyn_atoms_equal": len(ta.dyn_atoms) == len(tb.dyn_atoms), "actions_equal": len(ta.actions) == len(tb.actions), "init_mapped": sb == tb.init_mask,
                           "goal_mapped": DB.map_state(ta, ta.goal_mask, tb, phi) == tb.goal_mask, "exact_init_distance_equal": da == db, "renaming_nontrivial": kind != "rename" or any(k != v for k, v in phi.items())})
    ok_rel = all(rel[k] == expect[k] for k in expect)
    ok_chk = all(all(v for k, v in ch.items() if k not in ("case", "kind", "seed")) for ch in checks)
    out = {"relations": rel, "relations_expected": expect, "relations_ok": ok_rel, "transform_checks": checks, "transform_ok": ok_chk, "ok": ok_rel and ok_chk, "seconds": round(time.time() - t0, 1)}
    _wj(rr / "checks" / "fixture_b.json", out)
    _ledger(rr, "fixture_b", seconds=out["seconds"], passed=out["ok"], exact_oracle_builds=2 * len(checks))
    print("fixture B ok=%s" % out["ok"])
    return 0 if out["ok"] else 3


def lib_build(rr):
    from cp_disr.pddl.compact import diagb as DB
    cfg = _cfg()
    t0 = time.time()
    st = DB.build(rr, ROOT / cfg["old_run_root"], cfg)
    _ledger(rr, "lib_build", seconds=round(time.time() - t0, 1), new_exact_queries=st["new_exact_queries"], cpu_core_hours=round((time.time() - t0) / 3600, 3))
    print(json.dumps(st))


def lib_score(rr, model, device_name):
    from cp_disr.pddl.compact import diagb as DB
    t0 = time.time()
    r = DB.score(rr, model, lambda: make_evaluator(rr, model, device_name))
    _ledger(rr, "lib_score_" + model, seconds=round(time.time() - t0, 1), gpu_seconds=round(time.time() - t0, 1) if device_name != "cpu" else 0, device=device_name)
    print(json.dumps(r))


def lib_equiv(rr, model, device_name):
    from cp_disr.pddl.compact import diagb as DB
    t0 = time.time()
    rows = DB.equivariance(rr, model, lambda: make_evaluator(rr, model, device_name))
    _wj(Path(rr) / "diagnostics" / ("invariance_%s.json" % model), rows)
    _ledger(rr, "lib_equiv_" + model, seconds=round(time.time() - t0, 1), gpu_seconds=round(time.time() - t0, 1) if device_name != "cpu" else 0, device=device_name)
    print("equivariance %s: %d rows, max variant diff %.3g, max repeat diff %.3g" % (model, len(rows), max(r["abs_variant_diff"] for r in rows), max(r["abs_repeat_diff"] for r in rows)))


def variants_build(rr):
    from cp_disr.pddl.compact import variants as VR
    rr = Path(rr)
    cfg = _cfg()
    t0 = time.time()
    rows, solves = VR.make_all(ROOT / cfg["old_run_root"], cfg, rr / "diagnostics" / "variants", rr / "diagnostics" / "variant_reference", 60)
    from collections import Counter
    summ = Counter((r["kind"], bool(r.get("feasible"))) for r in rows)
    _wj(rr / "diagnostics" / "problem_transform_manifest.json", {"rows": rows, "reference_solves": solves, "summary": {"%s|%s" % k: v for k, v in sorted(summ.items())}})
    _ledger(rr, "variants_build", seconds=round(time.time() - t0, 1), reference_solves=solves, cpu_core_hours=round((time.time() - t0) / 3600, 3))
    print(json.dumps({"%s|%s" % k: v for k, v in sorted(summ.items())}), "reference solves", solves)


def fixture_variants(rr):
    """Fixture 6: the transformed problems differ from their base exactly as defined (goal / object / init counts), P and R replay the base optimal plan, W is feasible by a bounded LAMA call."""
    from cp_disr.pddl.parse import parse_problem
    rr = Path(rr)
    rows = json.loads((rr / "diagnostics" / "problem_transform_manifest.json").read_text())["rows"]
    byid = {r["problem_id"]: r for r in rows}
    checks = []
    for r in rows:
        if r["kind"] == "BASE":
            continue
        b = byid[r["base"]]
        pb, pv = parse_problem(Path(b["file"]).read_text()), parse_problem(Path(r["file"]).read_text())
        cnt = lambda p, t: sum(1 for x in p.objects.values() if x == t)
        if r["kind"] == "P":
            ok = len(pv.goal) == len(pb.goal) - 1 and set(pv.goal) < set(pb.goal) and pv.init == pb.init and pv.objects == pb.objects
        elif r["kind"] == "R":
            ok = cnt(pv, "pallet") == cnt(pb, "pallet") + 1 and pv.goal == pb.goal and len(pv.init) == len(pb.init) + 2 and set(pb.init) < set(pv.init)
        else:
            moved = set(r["moved_objects"])
            at_new = {a[0] for pred, a in pv.init if pred == "at" and a[1] == r["new_place"]}
            ok = (cnt(pv, "distributor") == cnt(pb, "distributor") + 1 and cnt(pv, "hoist") == cnt(pb, "hoist") + 1 and cnt(pv, "pallet") == cnt(pb, "pallet") + 1 and pv.goal == pb.goal and moved <= at_new
                  and all(a[1] != r["from_place"] for pred, a in pv.init if pred == "at" and a[0] in moved))
        checks.append({"problem": r["problem_id"], "kind": r["kind"], "structure_ok": bool(ok), "feasible": bool(r["feasible"]), "reference_length": r["reference_length"]})
    out = {"checks": checks, "ok": all(c["structure_ok"] and c["feasible"] for c in checks), "n": len(checks), "reference_solves": json.loads((rr / "diagnostics" / "problem_transform_manifest.json").read_text())["reference_solves"]}
    _wj(rr / "checks" / "fixture_variants.json", out)
    _ledger(rr, "fixture_variants", passed=out["ok"])
    print("fixture variants ok=%s n=%d" % (out["ok"], len(checks)))
    return 0 if out["ok"] else 3


def fixture_s(rr, device_name):
    """Fixture for T1-S: with fraction 0 the S trainer equals D0's trainer (loss, gradient norm); with 0.25 the construction is strict, deterministic, never above D0's pair count and uses only Train96 labels."""
    import torch
    from cp_disr.pddl import train as PT
    from cp_disr.pddl.compact import crossparent as XP, train as CT
    from cp_disr.pddl.compact.models import make_model_v2
    rr = Path(rr)
    cfg = _cfg()
    root = ROOT / cfg["old_run_root"]
    dev = torch.device(device_name)
    t0 = time.time()
    cases, trajs, labels = CT.train_inputs(root)
    plan0, plan = XP.SPlan(cases, labels, 0.0), XP.SPlan(cases, labels, 0.25)
    m1, m2 = make_model_v2("dense", dev), make_model_v2("dense", dev)
    tr1 = PT.PddlTrainer(m1, cases, labels, dev, lr=CT.BUDGET["lr"], chunk=CT.BUDGET["chunk_decisions"])
    tr2 = XP.PddlTrainerS(m2, cases, labels, dev, plan0, lr=CT.BUDGET["lr"], chunk=CT.BUDGET["chunk_decisions"])
    batch = tr1.update_batches(trajs, 0)[0]
    r1, r2 = tr1.step(batch), tr2.step(batch)
    same = {k: (r1[k], r2[k]) for k in ("loss", "nll", "rank", "grad_norm", "decisions")}
    ok0 = all(abs(a - b) <= 1e-4 * max(1.0, abs(a)) for a, b in same.values())
    m3 = make_model_v2("dense", dev)
    tr3 = XP.PddlTrainerS(m3, cases, labels, dev, plan, lr=CT.BUDGET["lr"], chunk=CT.BUDGET["chunk_decisions"])
    r3 = tr3.step(batch)
    byc = {c.case_id: c for c in cases}
    bad_order, bad_own, checked, deterministic = 0, 0, 0, True
    plan_b = XP.SPlan(cases, labels, 0.25)
    for key, info in list(plan.by_key.items())[::7]:
        cid = info["cid"]
        task = byc[cid].task
        dist = labels[key]
        st = key.split("|")[1]
        parent = task.state_from_list([int(x) for x in st.split(",")])
        own = {parent} | {task.apply(parent, task.action_by_id[a]) for a in dist}
        for a, y in info["cross"]:
            if y is None:
                continue
            checked += 1
            bad_order += not (plan.pool[cid][y] > dist[a])
            bad_own += y in own
        deterministic &= plan_b.by_key[key]["cross"] == info["cross"] and plan_b.by_key[key]["keep"] == info["keep"]
    st = plan.summary()
    out = {"fraction0_equals_D0": ok0, "fraction0_values": same, "s_step": {k: r3[k] for k in ("loss", "nll", "rank", "grad_norm", "decisions", "cross_pairs_used")}, "plan_stats": st, "cross_pairs_checked": checked, "cross_not_strictly_farther": bad_order,
           "cross_inside_own_decision": bad_own, "deterministic": bool(deterministic), "pairs_not_above_D0": st["pairs"] - st["masked_slots"] <= st["pairs"], "train_labels_only": all(k.split("|")[0].startswith("train_") for k in labels),
           "fixture_optimizer_steps": 3, "seconds": round(time.time() - t0, 1)}
    out["ok"] = bool(ok0 and bad_order == 0 and bad_own == 0 and deterministic and out["train_labels_only"] and r3["cross_pairs_used"] > 0)
    _wj(rr / "checks" / "fixture_s.json", out)
    _ledger(rr, "fixture_s", seconds=out["seconds"], passed=out["ok"], fixture_optimizer_steps=3, gpu_seconds=out["seconds"])
    print("fixture S ok=%s stats=%s" % (out["ok"], json.dumps(st)))
    return 0 if out["ok"] else 3


def train_t1(rr, device_name):
    import torch
    from cp_disr.pddl.compact import crossparent as XP, train as CT
    rr = Path(rr)
    cfg = _cfg()
    out = rr / "training" / "T1"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "training_accounting.json").is_file():
        print("already complete")
        return 0
    cases, trajs, labels = CT.train_inputs(ROOT / cfg["old_run_root"])
    dev = torch.device(device_name)
    plan = XP.SPlan(cases, labels, 0.25)
    _wj(out / "s_plan_summary.json", plan.summary())
    t0 = time.time()
    factory = lambda m, c, l, d: XP.PddlTrainerS(m, c, l, d, plan, lr=CT.BUDGET["lr"], chunk=CT.BUDGET["chunk_decisions"])
    acct, rows, m = CT.train_loop_v2(out, "dense", dev, cases, trajs, labels, log=print, trainer_factory=factory)
    led = CT.schedule_ledger(trajs)
    acct["schedule_decisions_match_ledger"] = [r["decisions"] for r in rows] == [r["decisions"] for r in led]
    acct["s_plan"] = plan.summary()
    acct["cross_pairs_used_total"] = sum(r.get("cross_pairs_used", 0) for r in rows)
    _wj(out / "training_accounting.json", acct)
    CT.write_csv(out / "updates.csv", rows)
    _ledger(rr, "train_T1", seconds=round(time.time() - t0, 1), accepted_updates=acct["optimizer_steps"], attempts=acct["attempts"], device=device_name)
    return 0


def panel(rr, arms, set_name, shard, nshards, device_name):
    from cp_disr.pddl.compact import panel as PN
    t0 = time.time()
    arms = arms.split(",")
    cache = {}

    def mk(a):
        return make_evaluator(rr, a, device_name)
    PN.run_panel(rr, ROOT / _cfg()["old_run_root"], arms, set_name, shard, nshards, mk, "_".join(arms))
    _ledger(rr, "panel_%s_%s" % ("_".join(arms), set_name), shard=[shard, nshards], seconds=round(time.time() - t0, 1), gpu_device_hours=round((time.time() - t0) / 3600, 3) if device_name != "cpu" else 0, device=device_name)


def report_a(rr):
    from cp_disr.pddl.compact import reports_a as RA
    rr = Path(rr)
    cfg = _cfg()
    t0 = time.time()
    w1 = None
    p = rr / "training" / "W1" / "fit_accounting.json"
    if p.is_file():
        acc = json.loads(p.read_text())
        w1 = {"parents": acc["export_stats"]["parents"], "pairs_raw": acc["export_stats"]["pairs_raw"], "pairs_unique_nonzero": acc["fit"].get("pairs_unique_nonzero"), "features": acc["fit"].get("features"),
              "train_pair_accuracy_weighted": acc["fit"].get("train_pair_accuracy_weighted"), "weights_nonzero": acc["fit"].get("weights_nonzero")}
    res = rr / "results"
    root, sm, fa, sc = (ROOT / cfg[k] for k in ("old_run_root", "search_match_root", "fast_alt_root", "scope_root"))
    RA.wcsv(res / "supervision_contract.csv", RA.supervision_rows(root, w1))
    RA.wcsv(res / "historical_paired.csv", RA.historical_rows(sm, fa))
    RA.wcsv(res / "cost_breakdown.csv", RA.cost_rows(sm, fa))
    RA.wcsv(res / "ipc_failure_map.csv", RA.ipc_map(sm, fa, sc))
    _ledger(rr, "report_a", seconds=round(time.time() - t0, 1))
    print("part A tables written")


def report_b(rr, names):
    from cp_disr.pddl.compact import diagb as DB, reports_a as RA
    rr = Path(rr)
    names = names.split(",")
    rows_dec, rows_pair, decisions = DB.metrics(rr, names)
    pk, man, pairs = DB.load_library(rr)
    summ = DB.summarise(rows_dec, rows_pair, pk, names)
    _wj(rr / "diagnostics" / "b_summary.json", summ)
    RA.wcsv(rr / "diagnostics" / "b_decisions.csv", rows_dec)
    RA.wcsv(rr / "diagnostics" / "b_pair_outcomes.csv", rows_pair)
    # per-problem table of the decidable pair inversion rates
    from collections import defaultdict
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows_pair:
        if r["outcome"] in ("CORRECT", "INVERTED", "TIE"):
            c = per[(r["problem"], r["kind"])][r["model"]]
            c[0] += 1
            c[1] += r["outcome"] == "INVERTED"
    rowsp = []
    for (p, kind), d in sorted(per.items()):
        row = {"problem": p, "family": pk[p]["family"], "kind": kind}
        for n in names:
            c = d.get(n, [0, 0])
            row[n + "_decidable"] = c[0]
            row[n + "_inversion_rate"] = c[1] / c[0] if c[0] else None
        rowsp.append(row)
    RA.wcsv(rr / "diagnostics" / "b_by_problem.csv", rowsp)
    # inversion rate by exact distance gap (same-problem pairs; WL ties reported separately)
    bucket = lambda g: "1" if g == 1 else "2" if g == 2 else "3-4" if g <= 4 else "5+"
    gap = defaultdict(lambda: [0, 0, 0])
    for r in rows_pair:
        if r["outcome"] in ("CORRECT", "INVERTED", "TIE") and r.get("distance_gap"):
            c = gap[(r["kind"], bucket(r["distance_gap"]), r["model"])]
            c[0] += 1
            c[1] += r["outcome"] == "INVERTED"
            c[2] += r["outcome"] == "TIE"
    RA.wcsv(rr / "diagnostics" / "b_by_distance_gap.csv", [{"kind": k, "gap": b, "model": m, "decidable": c[0], "inverted": c[1], "ties": c[2], "inversion_rate": c[1] / c[0]} for (k, b, m), c in sorted(gap.items())])
    # repair / damage between models on identical items
    idx = {}
    for r in rows_pair:
        if r["outcome"] in ("CORRECT", "INVERTED", "TIE"):
            idx.setdefault((r["kind"], r["problem"], r["x"], r["y"]), {})[r["model"]] = r["outcome"]
    rd = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for kind in ("same_parent", "cross_parent"):
                c = {"both_correct": 0, "a_only": 0, "b_only": 0, "neither": 0}
                for (k, p, x, y), d in idx.items():
                    if k != kind or a not in d or b not in d:
                        continue
                    ca, cb = d[a] == "CORRECT", d[b] == "CORRECT"
                    c["both_correct" if ca and cb else "a_only" if ca else "b_only" if cb else "neither"] += 1
                rd.append({"model_a": a, "model_b": b, "kind": kind, **c})
    dd = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = {"both_correct": 0, "a_only": 0, "b_only": 0, "neither": 0}
            for (cid, k), (ca, ra) in decisions[a].items():
                if (cid, k) in decisions[b]:
                    cb = decisions[b][(cid, k)][0]
                    c["both_correct" if ca and cb else "a_only" if ca else "b_only" if cb else "neither"] += 1
            dd.append({"model_a": a, "model_b": b, "kind": "same_parent_decision", **c})
    RA.wcsv(rr / "diagnostics" / "b_repair_damage.csv", rd + dd)
    # equivariance summary
    eq = []
    for n in names:
        p = rr / "diagnostics" / ("invariance_%s.json" % n)
        if p.is_file():
            rows = json.loads(p.read_text())
            import statistics
            eq.append({"model": n, "rows": len(rows), "max_abs_repeat_diff": max(r["abs_repeat_diff"] for r in rows), "max_abs_variant_diff": max(r["abs_variant_diff"] for r in rows), "median_abs_variant_diff": statistics.median(r["abs_variant_diff"] for r in rows),
                       "max_abs_variant_diff_rename": max(r["abs_variant_diff"] for r in rows if r["variant"].startswith("rename")), "max_abs_variant_diff_goalperm": max(r["abs_variant_diff"] for r in rows if r["variant"] == "goalperm")})
    RA.wcsv(rr / "diagnostics" / "invariance_summary.csv", eq)
    print(json.dumps(summ)[:1500])


def trigger_s(rr, names="DENSE,WL"):
    """T1-S trigger (plan 6.1) from the library B of the OLD models only."""
    from cp_disr.pddl.compact import diagb as DB
    rr = Path(rr)
    cfg = _cfg()["t1"]["S"]
    rows_dec, rows_pair, _ = DB.metrics(rr, ["DENSE", "WL"])
    pk, man, pairs = DB.load_library(rr)
    from collections import defaultdict
    sampled = defaultdict(int)
    for r in pairs:
        if r["kind"] == "cross_parent":
            sampled[r["problem"]] += 1
    stat = defaultdict(lambda: {"DENSE": [0, 0], "WL": [0, 0]})
    for r in rows_pair:
        if r["kind"] == "cross_parent" and r["outcome"] in ("CORRECT", "INVERTED", "TIE"):
            c = stat[r["problem"]][r["model"]]
            c[0] += 1
            c[1] += r["outcome"] == "INVERTED"
    out, qual, worse = [], [], []
    for p, d in pk.items():
        if d["family"] == "train":
            continue
        n_dec = stat[p]["DENSE"][0]
        rate = n_dec / sampled[p] if sampled[p] else 0
        ok = n_dec >= cfg["comparable_pairs_per_problem_min"] and rate >= cfg["decidable_rate_min"]
        dr = stat[p]["DENSE"][1] / n_dec if n_dec else None
        wr = stat[p]["WL"][1] / stat[p]["WL"][0] if stat[p]["WL"][0] else None
        w = bool(ok and dr is not None and wr is not None and dr > wr)
        out.append({"problem": p, "family": d["family"], "cross_pairs_sampled": sampled[p], "cross_decidable": n_dec, "decidable_rate": rate, "qualifies": ok, "dense_inversion_rate": dr, "wl_inversion_rate": wr, "dense_worse_than_wl": w})
        qual += [p] if ok else []
        worse += [p] if w else []
    trig = len(qual) >= cfg["problems_min"] and len(worse) >= cfg["dense_worse_than_wl_problems_min"]
    res = {"rule": cfg, "qualifying_non_train_problems": len(qual), "dense_worse_than_wl_among_qualifying": len(worse), "S_triggered_by_B": trig, "cache_available_for_same_problem_cross_pairs": "Train96 exact cache: see second condition", "per_problem": out}
    _wj(rr / "diagnostics" / "t1_trigger_S_from_B.json", res)
    print("S trigger from B: qualifying=%d worse=%d triggered=%s" % (len(qual), len(worse), trig))


def commands(rr, a):
    return {"fixture-s": lambda: fixture_s(rr, a.device), "train-t1": lambda: train_t1(rr, a.device), "fixture-variants": lambda: fixture_variants(rr), "report-a": lambda: report_a(rr), "report-b": lambda: report_b(rr, a.models), "trigger-s": lambda: trigger_s(rr),
            "fixture-w1": lambda: w1_fixture(rr), "w1-fit": lambda: w1_fit(rr), "fixture-b": lambda: fixture_b(rr), "lib-build": lambda: lib_build(rr), "lib-score": lambda: lib_score(rr, a.model, a.device),
            "lib-equiv": lambda: lib_equiv(rr, a.model, a.device), "variants-build": lambda: variants_build(rr), "panel": lambda: panel(rr, a.arms, a.set, a.shard, a.nshards, a.device)}
