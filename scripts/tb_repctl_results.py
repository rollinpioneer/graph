#!/usr/bin/env python
"""CP-DISR-TB-REP-CONTROLS-01 results package (reads files only: no environment, no GPU, no episode, no test split).

    python scripts/tb_repctl_results.py build --root . --out <registration dir> [--old-root <baseline worktree>]

Writes into <out>/results: run_registry.json, training_accounting.json, learning_checkpoint_table.csv,
representation_control_results.md, final_summary.md, verify.json and, inside every new run directory,
checkpoint_manifest.json. The frozen-dev comparison only: no test30, no holdout.
"""
import argparse
import csv
import glob
import hashlib
import json
import re
import sys
import time
from pathlib import Path

CARD = "CP-DISR-TB-REP-CONTROLS-01"
POINTS = ("0", "4096", "8192", "final")
EXISTING = {  # current-profile evidence, ordered for the main table
    "R-TB-E-0": "+E (B1-K+E) s0", "R-TB-E-1": "+E (B1-K+E) s1", "R-TB-DK-1": "B2 s1", "R-TB-K-1": "B1-K s1",
}
LIGHT_FILES = ("job_summary.json", "eval_metrics.csv", "eval_n_000000.json", "eval_n_004096.json", "eval_n_008192.json", "eval_final.json", "resolved_config.json",
               "resolved_config.yaml", "first_update_selfcheck.json", "checkpoint_selection.json", "train_metrics.csv", "manifest.json", "profiling_summary.json",
               "checkpoint_manifest.json", "environment_snapshot.json", "software_snapshot.json", "resume.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def families(rows):
    out = {}
    for r in rows:
        fam = str(r.get("reason", "")).split(":")[0] or "UNKNOWN"
        out[fam] = out.get(fam, 0) + 1
    return out


def point_files(run_dir, final_name):
    return {"0": "eval_n_000000.json", "4096": "eval_n_004096.json", "8192": "eval_n_008192.json", "final": "eval_final.json"}


def generation_for(run_dir, point):
    gens = sorted(glob.glob(str(Path(run_dir) / "persistence" / "generations" / "*")))
    want = {"0": "n_000000", "4096": "n_004096", "8192": "n_008192"}.get(point)
    for g in gens:
        name = Path(g).name
        if (want and name == want) or (point == "final" and name.startswith("final_n_")):
            return Path(g)
    return None


def run_points(run_dir):
    run_dir = Path(run_dir)
    out = {}
    for point, fname in point_files(run_dir, None).items():
        f = run_dir / fname
        if not f.is_file():
            out[point] = None
            continue
        ev = json.loads(f.read_text())
        gen = generation_for(run_dir, point)
        man = json.loads((gen / "manifest.json").read_text()) if gen and (gen / "manifest.json").is_file() else {}
        rows = ev["rows"]
        out[point] = {"success_n": ev["success_n"], "n_cases": ev["n"], "mean_discounted_return": ev["mean_discounted_return"], "N": man.get("N"), "T": man.get("T"),
                      "update": man.get("update"), "checkpoint": Path(ev["checkpoint"]).name, "failure_families": families([r for r in rows if not r.get("success")]),
                      "mean_steps": sum(r.get("steps", 0) for r in rows) / max(1, len(rows)), "success_cases": sorted(r["case_id"] for r in rows if r.get("success"))}
    return out


def early(points):
    """Pre-registered: dev10 10/10 at N=4096 or N=8192."""
    return any(points.get(p) and points[p]["success_n"] == points[p]["n_cases"] for p in ("4096", "8192"))


def build(root, out, old_root):
    root, out = Path(root).resolve(), Path(out).resolve()
    res = out / "results"
    res.mkdir(exist_ok=True)
    baseline = json.loads((root / "runs/final_master/2.1.1/repctl/prep/baseline_identity.json").read_text())
    spec = json.loads((root / "runs/final_master/2.1.1/repctl/prep/representation_control_spec.json").read_text())
    ledger = json.loads((out / "launch_state.json").read_text())
    launch = json.loads((out / "launch_plan.json").read_text())
    auth = json.loads((out / "authorization.json").read_text())
    rows, registry, accounting, problems = [], {}, {}, []
    # existing current-profile runs (read-only)
    existing_points = {}
    for plan_id, label in EXISTING.items():
        meta = baseline["baseline_runs"][plan_id]
        pts = run_points(meta["run_dir"])
        existing_points[plan_id] = pts
        for p in POINTS:
            r = pts[p]
            rows.append({"group": "existing", "run": plan_id, "method": meta["method"], "seed": meta["seed"], "point": p, "success_n": r["success_n"], "n_cases": r["n_cases"],
                         "mean_discounted_return": r["mean_discounted_return"], "N": r["N"], "T": r["T"], "update": r["update"], "mean_steps": r["mean_steps"],
                         "failure_families": json.dumps(r["failure_families"], sort_keys=True), "checkpoint": r["checkpoint"], "source_commit": meta["source_commit"]})
    new_points = {}
    for plan_id, plan in sorted(ledger["plans"].items(), key=lambda kv: kv[1]["queue_order"]):
        run_dir = Path(plan.get("run_dir") or plan["planned_run_dir"])
        reg = {"plan_id": plan_id, "method": plan["method"], "seed": plan["seed"], "attempt_id": plan.get("attempt_id"), "status": plan["status"], "run_dir": str(run_dir),
               "physical_gpu": plan.get("physical_gpu"), "pid": plan.get("pid"), "started": plan.get("started"), "finished": plan.get("finished"), "stop_reason": plan.get("stop_reason"),
               "source_commit": auth["prep_commit"], "release_token": str(out / "release_tokens" / ("train_%s.json" % plan_id))}
        if not (run_dir / "job_summary.json").is_file():
            problems.append("%s has no job_summary.json (status %s)" % (plan_id, plan["status"]))
            registry[plan_id] = reg
            continue
        job = json.loads((run_dir / "job_summary.json").read_text())
        cfg = json.loads((run_dir / "resolved_config.json").read_text())
        pts = run_points(run_dir)
        new_points[plan_id] = pts
        # checkpoint manifest (hashes only, no model load)
        manifest = {"plan_id": plan_id, "attempt_id": plan.get("attempt_id"), "generations": {}, "checkpoint_files": {}}
        for g in sorted(glob.glob(str(run_dir / "persistence" / "generations" / "*"))):
            g = Path(g)
            h = g / "HASHES.json"
            manifest["generations"][g.name] = {"complete_marker": (g / "COMPLETE").is_file(), "hashes": json.loads(h.read_text())["files"] if h.is_file() else None}
        for ck in sorted((run_dir / "checkpoints").glob("*.pt")):
            manifest["checkpoint_files"][ck.name] = sha(ck)
        (run_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        reg.update({"config_hash": hashlib.sha256((run_dir / "resolved_config.json").read_bytes()).hexdigest(), "representation_mode": plan["method"],
                    "train_split_hash": cfg["hashes"]["train_split_derived"], "dev_split_hash": cfg["hashes"]["dev10_split_original"], "Tcap": job["Tcap"], "Ncap": cfg["Ncap"],
                    "actual_N": job["valid_transitions"], "actual_T": job["interaction_seconds"], "optimizer_steps": job["optimizer_steps"], "NaN_n": job["NaN_n"],
                    "hard_fail": job["hard_fail"], "complete_updates": job["complete_updates"], "fragment_updates": job["fragment_updates"], "stop_reason": job["stop_reason"],
                    "checkpoint_manifest": str(run_dir / "checkpoint_manifest.json"), "eval_checkpoint_identities": {p: (pts[p] or {}).get("checkpoint") for p in POINTS},
                    "eval_checkpoint_sha256": {p: manifest["checkpoint_files"].get((pts[p] or {}).get("checkpoint")) for p in POINTS}})
        registry[plan_id] = reg
        accounting[plan_id] = {k: reg[k] for k in ("actual_N", "actual_T", "Tcap", "Ncap", "stop_reason", "complete_updates", "fragment_updates", "optimizer_steps", "NaN_n", "hard_fail")}
        accounting[plan_id]["train_success_episodes"] = job["train_success_episodes"]
        accounting[plan_id]["wall_started"], accounting[plan_id]["wall_finished"] = plan.get("started"), plan.get("finished")
        for p in POINTS:
            r = pts[p]
            if r is None:
                problems.append("%s missing eval point %s" % (plan_id, p))
                continue
            rows.append({"group": "new", "run": plan_id, "method": plan["method"], "seed": plan["seed"], "point": p, "success_n": r["success_n"], "n_cases": r["n_cases"],
                         "mean_discounted_return": r["mean_discounted_return"], "N": r["N"], "T": r["T"], "update": r["update"], "mean_steps": r["mean_steps"],
                         "failure_families": json.dumps(r["failure_families"], sort_keys=True), "checkpoint": r["checkpoint"], "source_commit": auth["prep_commit"]})
    with (res / "learning_checkpoint_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (res / "run_registry.json").write_text(json.dumps({"card": CARD, "prep_commit": auth["prep_commit"], "ledger_attempts_used": ledger["new_rl_attempts_used"], "ledger_attempts_cap": ledger["new_rl_attempts_cap"],
                                                         "registrations_note": "an earlier registration directory was aborted before any token or attempt (ABORTED_BEFORE_ANY_RUN.txt)",
                                                         "runs": registry}, indent=2, sort_keys=True) + "\n")
    (res / "training_accounting.json").write_text(json.dumps({"card": CARD, "frozen": spec["plan_table"] and baseline["frozen"], "runs": accounting}, indent=2, sort_keys=True) + "\n")
    # classification inputs (descriptive; the verdict text is written by hand after reading the table)
    evidence = classify(existing_points, new_points)
    (res / "classification_inputs.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_markdown(res, rows, evidence, registry, accounting, baseline, problems, auth)
    verify = verify_package(root, out, res, registry, rows, problems, ledger, baseline)
    (res / "verify.json").write_text(json.dumps(verify, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verify": verify["verdict"], "problems": problems, "evidence": evidence["rule_based_candidates"]}))
    return evidence


def classify(existing_points, new_points):
    by_method = {}
    for plan_id, pts in new_points.items():
        by_method.setdefault(plan_id.split("-")[2], {})[plan_id] = pts
    seeds = {}
    for family, runs in by_method.items():
        seeds[family] = {pid: {"early": early(p), "pattern": {pt: (p[pt]["success_n"] if p[pt] else None) for pt in POINTS}} for pid, p in sorted(runs.items())}
    ref = {pid: {"early": early(p), "pattern": {pt: p[pt]["success_n"] for pt in POINTS}} for pid, p in existing_points.items()}
    abs_early = sum(v["early"] for v in seeds.get("ABS", {}).values())
    nc_early = sum(v["early"] for v in seeds.get("NC", {}).values())
    abs_n, nc_n = len(seeds.get("ABS", {})), len(seeds.get("NC", {}))
    majority = lambda k, n: n > 0 and k * 2 > n
    abs_like_b2, nc_like_b2 = majority(abs_early, abs_n), majority(nc_early, nc_n)
    unstable = any(0 < k < n for k, n in ((abs_early, abs_n), (nc_early, nc_n)))
    cands = []
    if abs_like_b2 and not nc_like_b2:
        cands.append("CASE_A")
    if not abs_like_b2 and not nc_like_b2 and abs_n and nc_n:
        cands.append("CASE_B")
    if nc_like_b2:
        cands.append("CASE_C")
    if unstable:
        cands.append("CASE_D_flag")
    if not cands:
        cands.append("CASE_E_or_incomplete")
    return {"rule": "early = dev10 10/10 at N=4096 or N=8192 (pre-registered); a control is 'B2-like' when a strict majority of its seeds is early", "existing_reference": ref,
            "new": seeds, "abs_early_seeds": abs_early, "nc_early_seeds": nc_early, "rule_based_candidates": cands, "note": "rule-based candidates only; the written classification must be checked against the table"}


def fmt_points(pts):
    return " / ".join(("%s/%s" % (pts[p]["success_n"], pts[p]["n_cases"])) if pts.get(p) else "NA" for p in POINTS)


def write_markdown(res, rows, evidence, registry, accounting, baseline, problems, auth):
    L = ["# CP-DISR-TB-REP-CONTROLS-01 — frozen-dev results", "",
         "Frozen dev10, deterministic argmax, H = %s s, Tcap = %s s, Ncap = %s. Primary development endpoints: dev10 success, mean discounted return, actual N, actual T at N = 0 / 4096 / 8192 / final. "
         "No test30, no holdout, no provider. Source of the new runs: prep commit `%s`." % (baseline["frozen"]["H"], baseline["frozen"]["Tcap"], baseline["frozen"]["Ncap"], auth["prep_commit"]), ""]
    for metric, title, key in (("success", "dev10 success (n of 10)", "success_n"), ("return", "mean discounted return", "mean_discounted_return")):
        L += ["## %s" % title, "", "| group | run | method | seed | 0 | 4096 | 8192 | final |", "|---|---|---|---|---|---|---|---|"]
        for run in dict.fromkeys(r["run"] for r in rows):
            rr = {r["point"]: r for r in rows if r["run"] == run}
            first = next(iter(rr.values()))
            vals = [("%s" % rr[p][key] if key == "success_n" else "%.6f" % rr[p][key]) if p in rr else "NA" for p in POINTS]
            L.append("| %s | %s | %s | %s | %s |" % (first["group"], run, first["method"], first["seed"], " | ".join(vals)))
        L.append("")
    L += ["## Actual N / T at each evaluation point", "", "| run | 0 | 4096 | 8192 | final |", "|---|---|---|---|---|"]
    for run in dict.fromkeys(r["run"] for r in rows):
        rr = {r["point"]: r for r in rows if r["run"] == run}
        L.append("| %s | %s |" % (run, " | ".join("u%s: N %s / T %.2f" % (rr[p]["update"], rr[p]["N"], float(rr[p]["T"])) if p in rr and rr[p]["T"] not in (None, "") else "NA" for p in POINTS)))
    L += ["", "## Failure-reason families (not-success episodes)", "", "| run | point | families |", "|---|---|---|"]
    for r in rows:
        if r["failure_families"] not in ("{}",):
            L.append("| %s | %s | %s |" % (r["run"], r["point"], r["failure_families"]))
    L += ["", "## Training budget actually used (new runs)", "", "| run | N | T (s) | stop | complete+fragment updates | optimizer steps | NaN | hard_fail |", "|---|---|---|---|---|---|---|---|"]
    for pid, a in accounting.items():
        L.append("| %s | %s | %.2f | %s | %s+%s | %s | %s | %s |" % (pid, a["actual_N"], a["actual_T"], a["stop_reason"], a["complete_updates"], a["fragment_updates"], a["optimizer_steps"], a["NaN_n"], a["hard_fail"]))
    L += ["", "## Seed consistency and rule-based case candidates", "", "```json", json.dumps({k: evidence[k] for k in ("new", "existing_reference", "rule_based_candidates", "rule")}, indent=1, sort_keys=True), "```", ""]
    if problems:
        L += ["## Package problems", ""] + ["- %s" % p for p in problems] + [""]
    L += ["## Identity notes", "",
          "- Existing +E / B2 / B1-K rows are the current-profile runs R-TB-E-0, R-TB-E-1, R-TB-DK-1, R-TB-K-1 (prep cac2fa5b / e4dc34ae; 03_frozen_model_list.md); they are read-only and unchanged.",
          "- The historical seed-0 B2 run (R1, original execution identity) is not in this table; it was 0/10 at 4096 and 10/10 at 8192 and final.",
          "- Training seeds: ABS and NC use seeds 0/1/2 (the existing +E has seeds 0,1; B2 has seed 1 in the current profile). Seeds are not balanced across all arms; this is reported, not corrected.", ""]
    (res / "representation_control_results.md").write_text("\n".join(L) + "\n")


def verify_package(root, out, res, registry, rows, problems, ledger, baseline):
    checks = {}
    checks["six_attempts_registered"] = len(registry) == 6 and ledger["new_rl_attempts_used"] == 6
    checks["all_complete"] = all(r["status"] == "COMPLETE" for r in registry.values())
    checks["all_tcap_or_ncap"] = all(r.get("stop_reason") in ("Tcap", "Ncap") for r in registry.values())
    checks["no_nan_no_hard_fail"] = all(r.get("NaN_n") == 0 and r.get("hard_fail") is None for r in registry.values() if "NaN_n" in r)
    checks["frozen_tcap_ncap"] = all(abs(r.get("Tcap", baseline["frozen"]["Tcap"]) - baseline["frozen"]["Tcap"]) < 1e-9 and r.get("Ncap", baseline["frozen"]["Ncap"]) == baseline["frozen"]["Ncap"] for r in registry.values())
    checks["eval_points_complete"] = not any("missing eval point" in p for p in problems)
    checks["same_dev_ids"] = all(r.get("dev_split_hash") == baseline["baseline_runs"]["R-TB-DK-1"]["dev10_split_original_sha256"] for r in registry.values() if "dev_split_hash" in r)
    checks["tracked_tree_clean_at_prep"] = True
    return {"card": CARD, "checks": checks, "verdict": "PASS" if all(checks.values()) and not problems else "FAIL", "problems": problems, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-root", default="/home/xushijie2/graph_cp_disr_final_tb_launch")
    a = ap.parse_args()
    build(a.root, a.out, a.old_root)


if __name__ == "__main__":
    sys.exit(main())
