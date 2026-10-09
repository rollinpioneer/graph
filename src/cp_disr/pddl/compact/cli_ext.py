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


def commands(rr, a):
    return {"fixture-w1": lambda: w1_fixture(rr), "w1-fit": lambda: w1_fit(rr)}
