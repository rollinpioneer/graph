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
    if name in ("D0", "C0"):
        from cp_disr.pddl.compact import train as CT
        sel = json.loads((Path(rr) / "training" / name / "selection.json").read_text())
        m = CT.load_model_v2({"D0": "dense", "C0": "c0"}[name], sel["checkpoint"]["path"], dev)
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


def commands(rr, a):
    return {"fixture-w1": lambda: w1_fixture(rr), "w1-fit": lambda: w1_fit(rr), "fixture-b": lambda: fixture_b(rr), "lib-build": lambda: lib_build(rr), "lib-score": lambda: lib_score(rr, a.model, a.device),
            "lib-equiv": lambda: lib_equiv(rr, a.model, a.device)}
