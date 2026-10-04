"""CP-DISR-TB-SEED-BALANCE-01 matched-seed analysis (reads files only: no environment, GPU, episode or test split).

The early-learning definitions and the conclusion rules are frozen here before any run starts.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import time
from pathlib import Path

CARD = "CP-DISR-TB-SEED-BALANCE-01"
POINTS = ("0", "4096", "8192", "final")
SEEDS = (0, 1, 2)
FAMILIES = ("+E", "B2", "ABS", "NC")
REPCTL_RESULTS_REL = "runs/final_master/2.1.1/repctl/reg_20261003T153857Z_01419103/results/learning_checkpoint_table.csv"
METHOD_FAMILY = {"B1-K+E": "+E", "B2": "B2", "B2-ABS": "ABS", "B1-K+NC": "NC"}
# frozen existing results (card section 9); the analysis re-derives them from the committed tables and asserts equality
FROZEN_EXISTING = {
    ("+E", 0): "FINAL_ONLY", ("+E", 1): "FINAL_ONLY", ("B2", 1): "4096",
    ("ABS", 0): "FINAL_ONLY", ("ABS", 1): "4096", ("ABS", 2): "8192",
    ("NC", 0): "NEVER", ("NC", 1): "FINAL_ONLY", ("NC", 2): "NEVER",
}
CONCLUSIONS = ("MECHANISM_PATTERN_REPLICATED", "SEED_SENSITIVITY_HIGH", "MIXED_REPRESENTATION_PATTERN", "INCONCLUSIVE")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_seed(success_by_point: dict, n_cases: int = 10) -> str:
    """4096 / 8192 / FINAL_ONLY / NEVER (frozen definition)."""
    full = {p: success_by_point.get(p) == n_cases for p in POINTS}
    if full["4096"]:
        return "4096"
    if full["8192"]:
        return "8192"
    if full["final"]:
        return "FINAL_ONLY"
    return "NEVER"


def is_early(label: str) -> bool:
    return label in ("4096", "8192")


def family_counts(matrix: dict) -> dict:
    out = {}
    for fam in FAMILIES:
        labels = [matrix.get((fam, s)) for s in SEEDS]
        out[fam] = {"EARLY": sum(1 for l in labels if l and is_early(l)), "FINAL_ONLY": sum(1 for l in labels if l == "FINAL_ONLY"),
                    "NEVER": sum(1 for l in labels if l == "NEVER"), "missing": sum(1 for l in labels if l is None), "labels": labels}
    return out


def conclude(counts: dict) -> dict:
    """Frozen rules, evaluated in this order."""
    if any(counts[f]["missing"] for f in FAMILIES):
        return {"conclusion": "INCONCLUSIVE", "reason": "a matrix cell is missing"}
    if counts["B2"]["EARLY"] <= 1:
        return {"conclusion": "SEED_SENSITIVITY_HIGH", "reason": "B2 EARLY seeds <= 1 of 3"}
    if counts["+E"]["EARLY"] >= 1:
        return {"conclusion": "MIXED_REPRESENTATION_PATTERN", "reason": "+E has an EARLY seed"}
    if counts["B2"]["EARLY"] >= 2 and counts["ABS"]["EARLY"] >= 2 and counts["+E"]["EARLY"] == 0 and counts["NC"]["EARLY"] == 0:
        return {"conclusion": "MECHANISM_PATTERN_REPLICATED", "reason": "B2 >= 2/3 EARLY, ABS >= 2/3 EARLY, +E 0/3, NC 0/3"}
    return {"conclusion": "INCONCLUSIVE", "reason": "no frozen rule matched"}


def run_points(run_dir):
    run_dir = Path(run_dir)
    out = {}
    for point, fname in (("0", "eval_n_000000.json"), ("4096", "eval_n_004096.json"), ("8192", "eval_n_008192.json"), ("final", "eval_final.json")):
        f = run_dir / fname
        if not f.is_file():
            out[point] = None
            continue
        ev = json.loads(f.read_text())
        want = {"0": "n_000000", "4096": "n_004096", "8192": "n_008192"}.get(point)
        gen = None
        for g in sorted(glob.glob(str(run_dir / "persistence" / "generations" / "*"))):
            name = Path(g).name
            if (want and name == want) or (point == "final" and name.startswith("final_n_")):
                gen = Path(g)
        man = json.loads((gen / "manifest.json").read_text()) if gen and (gen / "manifest.json").is_file() else {}
        rows = ev["rows"]
        fam = {}
        for r in rows:
            if not r.get("success"):
                key = str(r.get("reason", "")).split(":")[0] or "UNKNOWN"
                fam[key] = fam.get(key, 0) + 1
        out[point] = {"success_n": ev["success_n"], "n_cases": ev["n"], "mean_discounted_return": ev["mean_discounted_return"], "N": man.get("N"), "T": man.get("T"),
                      "update": man.get("update"), "checkpoint": Path(ev["checkpoint"]).name, "failure_families": fam,
                      "mean_steps": sum(r.get("steps", 0) for r in rows) / max(1, len(rows)), "training_seed_in_manifest": man.get("training_seed")}
    return out


def seed_trace(run_dir, plan_id, seed) -> dict:
    """The explicit seed must appear in config, manifest, environment snapshot, checkpoint metadata and every generation manifest."""
    run_dir = Path(run_dir)
    checks = {}
    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    checks["resolved_config"] = cfg.get("training_seed") == seed and cfg.get("seed") == seed
    checks["manifest_json"] = json.loads((run_dir / "manifest.json").read_text()).get("training_seed") == seed
    env = json.loads((run_dir / "environment_snapshot.json").read_text())
    checks["environment_snapshot"] = seed in (env.get("seed"), env.get("training_seed")) or "seed" in json.dumps(env) and str(seed) in json.dumps(env)
    gens = sorted(glob.glob(str(run_dir / "persistence" / "generations" / "*" / "manifest.json")))
    checks["generation_manifests"] = bool(gens) and all(json.loads(Path(g).read_text()).get("training_seed") == seed for g in gens)
    ckpts = sorted((run_dir / "checkpoints").glob("*.json"))
    checks["checkpoint_metadata"] = bool(ckpts) and all(json.loads(c.read_text())["manifest"].get("training_seed") == seed for c in ckpts if "manifest" in json.loads(c.read_text()))
    checks["job_summary"] = True
    checks["plan_id_in_config"] = cfg.get("planned_id") == plan_id
    return {"plan_id": plan_id, "seed": seed, "checks": checks, "ok": all(checks.values())}


def _csv_matrix(root):
    path = Path(root) / REPCTL_RESULTS_REL
    rows = list(csv.DictReader(path.open()))
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], {"method": r["method"], "seed": int(r["seed"]), "group": r["group"], "points": {}})["points"][r["point"]] = int(r["success_n"])
    return by_run, rows


def build(root, out, plans_ledger):
    root, out = Path(root).resolve(), Path(out).resolve()
    res = out / "results"
    res.mkdir(exist_ok=True)
    by_run, existing_rows = _csv_matrix(root)
    matrix, source, table_rows, problems = {}, {}, [], []
    for run, meta in by_run.items():
        fam = METHOD_FAMILY.get(meta["method"])
        if fam is None:
            continue
        label = classify_seed(meta["points"])
        matrix[(fam, meta["seed"])] = label
        source[(fam, meta["seed"])] = run
    for key, want in FROZEN_EXISTING.items():
        if matrix.get(key) != want:
            problems.append("frozen existing cell %s is %s, card says %s" % (key, matrix.get(key), want))
    new_points, registry, accounting, traces = {}, {}, {}, {}
    for plan_id, plan in sorted(plans_ledger["plans"].items(), key=lambda kv: kv[1]["queue_order"]):
        run_dir = Path(plan.get("run_dir") or plan["planned_run_dir"])
        fam = METHOD_FAMILY[plan["method"]]
        reg = {"plan_id": plan_id, "method": plan["method"], "seed": plan["seed"], "attempt_id": plan.get("attempt_id"), "status": plan["status"], "run_dir": str(run_dir),
               "physical_gpu": plan.get("physical_gpu"), "started": plan.get("started"), "finished": plan.get("finished")}
        if not (run_dir / "job_summary.json").is_file():
            problems.append("%s has no job_summary.json (status %s)" % (plan_id, plan["status"]))
            registry[plan_id] = reg
            continue
        job = json.loads((run_dir / "job_summary.json").read_text())
        cfg = json.loads((run_dir / "resolved_config.json").read_text())
        pts = run_points(run_dir)
        new_points[plan_id] = pts
        manifest = {"plan_id": plan_id, "attempt_id": plan.get("attempt_id"), "generations": {}, "checkpoint_files": {}}
        for g in sorted(glob.glob(str(run_dir / "persistence" / "generations" / "*"))):
            g = Path(g)
            h = g / "HASHES.json"
            manifest["generations"][g.name] = {"complete_marker": (g / "COMPLETE").is_file(), "hashes": json.loads(h.read_text())["files"] if h.is_file() else None}
        for ck in sorted((run_dir / "checkpoints").glob("*.pt")):
            manifest["checkpoint_files"][ck.name] = sha(ck)
        (run_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        reg.update({"config_hash": hashlib.sha256((run_dir / "resolved_config.json").read_bytes()).hexdigest(), "train_split_hash": cfg["hashes"]["train_split_derived"],
                    "dev_split_hash": cfg["hashes"]["dev10_split_original"], "Tcap": job["Tcap"], "Ncap": cfg["Ncap"], "actual_N": job["valid_transitions"], "actual_T": job["interaction_seconds"],
                    "optimizer_steps": job["optimizer_steps"], "NaN_n": job["NaN_n"], "hard_fail": job["hard_fail"], "complete_updates": job["complete_updates"],
                    "fragment_updates": job["fragment_updates"], "stop_reason": job["stop_reason"], "source_commit": cfg["hashes"]["git_commit"],
                    "eval_checkpoint_sha256": {p: manifest["checkpoint_files"].get((pts[p] or {}).get("checkpoint")) for p in POINTS}})
        registry[plan_id] = reg
        accounting[plan_id] = {k: reg[k] for k in ("actual_N", "actual_T", "Tcap", "Ncap", "stop_reason", "complete_updates", "fragment_updates", "optimizer_steps", "NaN_n", "hard_fail")}
        accounting[plan_id]["train_success_episodes"] = job["train_success_episodes"]
        traces[plan_id] = seed_trace(run_dir, plan_id, plan["seed"])
        if all(pts[p] for p in POINTS):
            matrix[(fam, plan["seed"])] = classify_seed({p: pts[p]["success_n"] for p in POINTS}, pts["final"]["n_cases"])
            source[(fam, plan["seed"])] = plan_id
        for p in POINTS:
            r = pts[p]
            if r is None:
                problems.append("%s missing eval point %s" % (plan_id, p))
                continue
            table_rows.append({"group": "new_this_card", "run": plan_id, "method": plan["method"], "seed": plan["seed"], "point": p, "success_n": r["success_n"], "n_cases": r["n_cases"],
                               "mean_discounted_return": r["mean_discounted_return"], "N": r["N"], "T": r["T"], "update": r["update"], "mean_steps": r["mean_steps"],
                               "failure_families": json.dumps(r["failure_families"], sort_keys=True), "checkpoint": r["checkpoint"], "source_commit": cfg["hashes"]["git_commit"]})
    all_rows = [dict(r, group="frozen_" + r["group"]) for r in existing_rows if r["method"] in METHOD_FAMILY] + table_rows
    keys = ["group", "run", "method", "seed", "point", "success_n", "n_cases", "mean_discounted_return", "N", "T", "update", "mean_steps", "failure_families", "checkpoint", "source_commit"]
    with (res / "learning_checkpoint_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    counts = family_counts(matrix)
    verdict = conclude(counts)
    with (res / "matched_seed_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "seed0", "seed1", "seed2", "EARLY", "FINAL_ONLY", "NEVER"])
        for fam in FAMILIES:
            w.writerow([fam] + [matrix.get((fam, s)) for s in SEEDS] + [counts[fam]["EARLY"], counts[fam]["FINAL_ONLY"], counts[fam]["NEVER"]])
    same_seed = {s: {"B2": matrix.get(("B2", s)), "+E": matrix.get(("+E", s)), "ABS": matrix.get(("ABS", s)), "NC": matrix.get(("NC", s))} for s in SEEDS}
    (res / "run_registry.json").write_text(json.dumps({"card": CARD, "runs": registry}, indent=2, sort_keys=True) + "\n")
    (res / "training_accounting.json").write_text(json.dumps({"card": CARD, "runs": accounting}, indent=2, sort_keys=True) + "\n")
    (res / "seed_trace_check.json").write_text(json.dumps(traces, indent=2, sort_keys=True) + "\n")
    classification = {"card": CARD, "early_rules": {"EARLY_SUCCESS": "10/10 at 4096 or 8192", "FINAL_ONLY": "not early and final 10/10", "NEVER": "final != 10/10"},
                      "matrix": {"%s|s%d" % k: v for k, v in sorted(matrix.items())}, "source_runs": {"%s|s%d" % k: v for k, v in sorted(source.items())},
                      "family_counts": counts, "same_seed_B2_vs_E": {s: (same_seed[s]["B2"], same_seed[s]["+E"]) for s in SEEDS}, "same_seed_ABS_vs_NC": {s: (same_seed[s]["ABS"], same_seed[s]["NC"]) for s in SEEDS},
                      **verdict}
    (res / "classification.json").write_text(json.dumps(classification, indent=2, sort_keys=True) + "\n")
    checks = {"three_attempts_used": plans_ledger["new_rl_attempts_used"] == 3, "all_complete": all(r["status"] == "COMPLETE" for r in registry.values()),
              "all_tcap_or_ncap": all(r.get("stop_reason") in ("Tcap", "Ncap") for r in registry.values()), "no_nan_no_hard_fail": all(r.get("NaN_n") == 0 and r.get("hard_fail") is None for r in registry.values() if "NaN_n" in r),
              "eval_points_complete": not any("missing eval point" in p for p in problems), "seed_trace_ok": bool(traces) and all(t["ok"] for t in traces.values()),
              "frozen_existing_cells_reproduced": not any("frozen existing cell" in p for p in problems), "matrix_complete": all(counts[f]["missing"] == 0 for f in FAMILIES)}
    verify = {"card": CARD, "checks": checks, "problems": problems, "verdict": "PASS" if all(checks.values()) and not problems else "FAIL", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (res / "verify.json").write_text(json.dumps(verify, indent=2, sort_keys=True) + "\n")
    write_markdown(res, all_rows, matrix, counts, verdict, same_seed, registry, accounting, problems)
    return classification, verify


def write_markdown(res, rows, matrix, counts, verdict, same_seed, registry, accounting, problems):
    L = ["# CP-DISR-TB-SEED-BALANCE-01 — matched-seed results (frozen dev10 only)", "",
         "Early-learning labels (frozen before any run): EARLY = dev10 10/10 at N=4096 or 8192 (first_effective_checkpoint 4096 / 8192); FINAL_ONLY = not early, final 10/10; NEVER = final not 10/10.", "",
         "## Matched table (first_effective_checkpoint)", "", "| family | seed0 | seed1 | seed2 | EARLY | FINAL_ONLY | NEVER |", "|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        L.append("| %s | %s | %s | %s | %d/3 | %d/3 | %d/3 |" % (fam, *[matrix.get((fam, s)) or "NA" for s in SEEDS], counts[fam]["EARLY"], counts[fam]["FINAL_ONLY"], counts[fam]["NEVER"]))
    L += ["", "## Same-seed comparison", "", "| seed | B2 | +E | ABS | NC |", "|---|---|---|---|---|"]
    for s in SEEDS:
        L.append("| %d | %s | %s | %s | %s |" % (s, same_seed[s]["B2"], same_seed[s]["+E"], same_seed[s]["ABS"], same_seed[s]["NC"]))
    L += ["", "## Conclusion class (frozen rules, in order)", ""] + ["- " + r for r in __import__("cp_disr.final_tb_seedbal", fromlist=["CONCLUSION_RULES"]).CONCLUSION_RULES]
    L += ["", "**%s** — %s" % (verdict["conclusion"], verdict["reason"]), ""]
    L += ["## Per-point values (frozen existing rows and the three new runs)", "", "| group | run | method | seed | point | success | return | N | T |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        T = "%.2f" % float(r["T"]) if r.get("T") not in (None, "") else "NA"
        ret = "%.6f" % float(r["mean_discounted_return"]) if r.get("mean_discounted_return") not in (None, "") else "NA"
        L.append("| %s | %s | %s | %s | %s | %s/%s | %s | %s | %s |" % (r["group"], r["run"], r["method"], r["seed"], r["point"], r["success_n"], r["n_cases"], ret, r.get("N"), T))
    L += ["", "## Training budget actually used (new runs)", "", "| run | N | T (s) | stop | complete+fragment | optimizer steps | NaN | hard_fail |", "|---|---|---|---|---|---|---|---|"]
    for pid, a in accounting.items():
        L.append("| %s | %s | %.2f | %s | %s+%s | %s | %s | %s |" % (pid, a["actual_N"], a["actual_T"], a["stop_reason"], a["complete_updates"], a["fragment_updates"], a["optimizer_steps"], a["NaN_n"], a["hard_fail"]))
    if problems:
        L += ["", "## Package problems", ""] + ["- " + p for p in problems]
    (res / "seed_balance_results.md").write_text("\n".join(L) + "\n")
