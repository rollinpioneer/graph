"""R1.4 frozen scene-paired analysis. Reads pulse outputs; no model execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ["R0_original", "RW_remove_q0_width", "RW_local_keep_final_width_only",
        "Rrandom_count_matched", "RW_actual_step_norm_matched",
        "RW_local_actual_step_norm_matched", "Rrandom_actual_step_norm_matched"]
FOLDS = [[2, 31, 68, 76], [9, 33, 64, 82]]
SCENES = sorted(sum(FOLDS, []))
COMBOS = ["G0S0", "G1S0", "G0S1", "G1S1"]
SEED = 20261023
SPEC = {
    "statistical_unit": "heldout scene; average its four independent source pulses first",
    "scene_count": 8, "folds": FOLDS, "steps_reported_separately": [1, 5],
    "bootstrap": {"method": "paired percentile scene bootstrap", "replicates": 10000, "seed": SEED},
    "dependence_limit": "Four heldout scenes within a fold share the same four source pulses. Eight-scene bootstrap is conditional on these sources and not a full independent-source/population interval. Report fold means and crossed source/target resampling as sensitivity, not independent replication.",
    "unknown": "Exclude unknown from numerator and denominator; report and fail completeness when any remain. Never convert unknown to failure.",
    "primary": "128 fixed queries; scene-equal analytic feasibility difference",
    "comparisons": [[ARMS[1], ARMS[0]], [ARMS[4], ARMS[0]],
                    [ARMS[1], ARMS[3]], [ARMS[4], ARMS[6]],
                    [ARMS[2], ARMS[0]], [ARMS[5], ARMS[0]],
                    [ARMS[1], ARMS[2]], [ARMS[4], ARMS[5]],
                    [ARMS[3], ARMS[0]], [ARMS[6], ARMS[0]]],
    "rescue_harm": "Paired candidate IDs vs original checkpoint and same-source/step R0; report both rather than only net change.",
    "pipeline": "G0S0/G1S0/G0S1/G1S1 each use full official collision/NMS/objectTop10/Top50 selection; top1, feasible count and candidate coverage separate.",
    "ranking": "Within-object success/failure pair inversions on the same known fixed-query ID intersection; compare old/new scores at each frozen geometry. Equal weight objects within source/scene, then sources within scene. Diagnostic, not AP.",
    "decision": "No best-step selection. Loss descent, gradient norm or oracle gains alone are not benefit. Require consistent treated-vs-R0 evidence surviving count and actual-step controls; small pulse evidence cannot establish a method or generalization."
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def labels(z, prefix):
    mu = np.asarray(z[prefix + "_mu"], dtype=float)
    collision = np.asarray(z[prefix + "_collision"], dtype=bool)
    empty = np.asarray(z[prefix + "_empty"], dtype=bool)
    reason = np.asarray(z[prefix + "_unknown_reason"]).astype(str)
    known = (np.char.str_len(reason) == 0) & np.isfinite(mu)
    good = known & (mu > 0) & (mu <= .4) & ~collision & ~empty
    return {"known": known, "good": good, "mu": mu, "collision": collision,
            "empty": empty, "object_id": np.asarray(z[prefix + "_object_id"])}


def rate(a, mask=None):
    x = np.asarray(a, dtype=float)
    if mask is not None:
        x = x[mask]
    return float(x.mean()) if len(x) else None


def paired_counts(old, new):
    known = old["known"] & new["known"]
    a, b = old["good"], new["good"]
    return {"n": int(known.sum()), "rescue": int((known & ~a & b).sum()),
            "harm": int((known & a & ~b).sum()), "old_good": int((known & a).sum()),
            "new_good": int((known & b).sum()), "delta": rate(b.astype(float)-a, known)}


def inversion(scores, lab, known):
    vals, pairs = [], 0
    for oid in np.unique(lab["object_id"][known]):
        k = known & (lab["object_id"] == oid)
        pos, neg = scores[k & lab["good"]], scores[k & ~lab["good"]]
        if not len(pos) or not len(neg):
            continue
        d = neg[:, None] - pos[None, :]
        vals.append(float(((d > 0) + .5*(d == 0)).mean()))
        pairs += d.size
    return rate(vals), len(vals), pairs


def scene_summary(rows, field="value"):
    scene_vals = {}
    for s in SCENES:
        v = [r[field] for r in rows if r["target_scene"] == s and r.get(field) is not None]
        if v:
            scene_vals[s] = float(np.mean(v))
    arr = np.asarray(list(scene_vals.values()))
    if not len(arr):
        return {"mean": None, "ci95": [None, None], "scenes": 0, "scene_values": {}}
    rng = np.random.default_rng(SEED)
    boot = arr[rng.integers(0, len(arr), size=(10000, len(arr)))].mean(axis=1)
    fold_means = [rate([scene_vals[s] for s in fold if s in scene_vals]) for fold in FOLDS]
    source_vals = {s: float(np.mean([r[field] for r in rows if r["source_scene"] == s and r.get(field) is not None]))
                   for s in SCENES if any(r["source_scene"] == s and r.get(field) is not None for r in rows)}
    # Source effects are crossed with heldout scenes within each fold. Keep both folds fixed.
    crossed = []
    if len(scene_vals) == 8 and len(rows) == 32:
        grid = {(r["source_scene"], r["target_scene"]): r[field] for r in rows}
        if len(grid) == 32 and all(x is not None for x in grid.values()):
            means = []
            for i in range(2):
                matrix = np.array([[grid[(src, tgt)] for tgt in FOLDS[i]] for src in FOLDS[1-i]])
                src = rng.integers(0, 4, (10000, 4))
                tgt = rng.integers(0, 4, (10000, 4))
                means.append(matrix[src[:, :, None], tgt[:, None, :]].mean(axis=(1, 2)))
            crossed = np.mean(means, axis=0)
    return {"mean": float(arr.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "scenes": len(arr), "source_target_units": len(rows), "scene_values": scene_vals,
            "fold_means": fold_means, "source_values": source_vals,
            "source_equal_mean": rate(list(source_vals.values())),
            "crossed_source_target_ci95": np.quantile(crossed, [.025, .975]).tolist() if len(crossed) else None}


def load_jobs(root):
    jobs = {}
    for path in sorted((root / "evaluation").glob("src_*/*/step*/train_*_0128.npz")):
        source = int(path.parts[-4].split("_")[-1])
        arm = path.parts[-3]
        step = int(path.parts[-2][4:])
        target = int(path.stem.split("_")[1])
        if arm not in ARMS or source not in SCENES or target not in SCENES or step not in (1, 5):
            raise ValueError(f"Unregistered evaluation {path}")
        if any(source in f and target in f for f in FOLDS):
            raise ValueError(f"Non-heldout comparison {path}")
        key = (source, target, step, arm)
        if key in jobs:
            raise ValueError(f"Duplicate {key}")
        z = dict(np.load(path, allow_pickle=False))
        ppath = root / "predictions" / path.relative_to(root / "evaluation")
        p = dict(np.load(ppath, allow_pickle=False))
        b = dict(np.load(root / "baseline" / path.name, allow_pickle=False))
        ids = np.asarray(z["primary_ids"], dtype=int)
        if len(ids) != 128 or len(set(ids.tolist())) != 128:
            raise ValueError(f"Expected 128 unique fixed ids {key}")
        if not np.array_equal(ids, p["mechanism_ids"]) or not np.array_equal(ids, b["mechanism_ids"]):
            raise ValueError(f"Fixed query mismatch {key}")
        jobs[key] = {"z": z, "p": p, "b": b, "path": path,
                     "base": labels(z, "primary_G0"), "new": labels(z, "primary_G1")}
    return jobs


def analyze(root, partial=False, out_root=None):
    jobs = load_jobs(root)
    if not jobs:
        raise SystemExit("No R1.4 evaluations present; analysis not run.")
    expected = 8*7*2*4
    if len(jobs) != expected and not partial:
        raise ValueError(f"Incomplete registered design: {len(jobs)}/{expected} jobs")
    units, recoveries, ranking, diagnostics, comparison_units = [], [], [], [], []
    baseline_seen, total_unknown, eval_calls = {}, 0, 0
    for key, job in jobs.items():
        source, target, step, arm = key
        z, p, b, base, new = (job[x] for x in ["z", "p", "b", "base", "new"])
        meta = {"source_scene": source, "target_scene": target, "step": step, "arm": arm}
        total_unknown += int((~base["known"]).sum() + (~new["known"]).sum())
        signature = tuple(np.asarray(base[k]).tobytes() for k in base)
        if target in baseline_seen and baseline_seen[target] != signature:
            raise ValueError(f"Original checkpoint labels vary for scene {target}")
        baseline_seen[target] = signature
        c = paired_counts(base, new)
        recoveries.append({**meta, "reference": "original_checkpoint", **c})
        units.append({**meta, "scope": "fixed128", "metric": "feasibility", "value": rate(new["good"], new["known"]),
                      "base_value": rate(base["good"], base["known"]), "n": int(new["known"].sum())})
        ref = jobs.get((source, target, step, ARMS[0]))
        if ref:
            if not np.array_equal(z["primary_ids"], ref["z"]["primary_ids"]):
                raise ValueError("R0 pairing failed")
            recoveries.append({**meta, "reference": ARMS[0], **paired_counts(ref["new"], new)})
        for combo in COMBOS:
            lab = labels(z, combo)
            total_unknown += int((~lab["known"]).sum())
            n = len(z[combo + "_final_ids"])
            if n != len(lab["good"]) or n > 50:
                raise ValueError(f"Pipeline length {key} {combo}")
            top1 = bool(lab["good"][0]) if n and lab["known"][0] else (False if not n else None)
            for metric, value in [("top1", top1), ("success_count", int(lab["good"].sum())),
                                  ("any_success", bool(lab["good"].any())),
                                  ("selected_count", n), ("success_fraction", rate(lab["good"], lab["known"]))]:
                units.append({**meta, "scope": combo, "metric": metric,
                              "value": float(value) if value is not None else None, "n": n})
        ids = z["primary_ids"].astype(int)
        common = base["known"] & new["known"]
        if ref:
            common &= ref["new"]["known"]
        for geo, lab in [("G0", base), ("G1", new)]:
            for scorename, score in [("S0", b["pred"][ids, 0]), ("S1", p["pred"][ids, 0])]:
                inv, objects, pairs = inversion(score, lab, common)
                ranking.append({**meta, "geometry": geo, "score": scorename, "inversion_rate": inv,
                                "eligible_objects": objects, "success_failure_pairs": pairs,
                                "known_query_intersection": int(common.sum())})
        oldraw, raw = np.ravel(b["raw_width"]), np.ravel(p["raw_width"])
        diagnostics.append({**meta, "raw_width_abs_delta_mean": float(np.mean(np.abs(raw-oldraw))),
                            "width_upper_saturation_rate": float(np.mean(raw*.12 >= .1)),
                            "width_lower_saturation_rate": float(np.mean(raw <= 0)),
                            "base_width_upper_saturation_rate": float(np.mean(oldraw*.12 >= .1)),
                            "base_width_lower_saturation_rate": float(np.mean(oldraw <= 0)),
                            "decoded_width_abs_delta_m_mean": float(np.mean(np.abs(p["pred"][:, 1]-b["pred"][:, 1]))),
                            "quality_logits_abs_delta_mean": float(np.mean(np.abs(p["quality_logits"]-b["quality_logits"]))),
                            "angle_class_change": float(np.mean(np.argmax(p["angle_logits"], axis=0)!=np.argmax(b["angle_logits"], axis=0))),
                            "depth_class_change": float(np.mean(np.argmax(p["depth_logits"], axis=0)!=np.argmax(b["depth_logits"], axis=0)))})
        receipt = job["path"].with_suffix(".json")
        if receipt.exists():
            eval_calls += int(json.loads(receipt.read_text(encoding="utf-8")).get("calls", 0))
    if total_unknown and not partial:
        raise ValueError(f"Unknown labels remain ({total_unknown} repeated appearances); do not treat as negatives")
    unit_map = {(r["source_scene"], r["target_scene"], r["step"], r["arm"], r["scope"], r["metric"]): r for r in units}
    summaries, contrasts = [], []
    for arm in ARMS:
        for step in (1, 5):
            for scope, metric in sorted(set((r["scope"], r["metric"]) for r in units)):
                selected = [r for r in units if (r["arm"], r["step"], r["scope"], r["metric"]) == (arm, step, scope, metric)]
                summaries.append({"arm": arm, "step": step, "scope": scope, "metric": metric, **scene_summary(selected)})
    for treated, control in SPEC["comparisons"]:
        for step in (1, 5):
            for scope, metric in sorted(set((r["scope"], r["metric"]) for r in units)):
                diffs = []
                for r in units:
                    if (r["arm"], r["step"], r["scope"], r["metric"]) != (treated, step, scope, metric):
                        continue
                    c = unit_map.get((r["source_scene"], r["target_scene"], step, control, scope, metric))
                    if c and r["value"] is not None and c["value"] is not None:
                        row = {"source_scene": r["source_scene"], "target_scene": r["target_scene"], "value": r["value"]-c["value"]}
                        diffs.append(row)
                        comparison_units.append({"treated": treated, "control": control, "step": step, "scope": scope, "metric": metric, **row})
                contrasts.append({"treated": treated, "control": control, "step": step, "scope": scope, "metric": metric, **scene_summary(diffs)})
    # GPU receipts remain verbatim as a compact provenance table; no step treated as replication.
    training_steps, mask_meta, preflights = [], [], []
    for p in sorted((root / "receipts").rglob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("steps"), list):
            for s in d["steps"]:
                training_steps.append({"receipt": str(p.relative_to(root)), "source_scene": d.get("source_scene"),
                                       "arm": d.get("arm"), **{k: v for k, v in s.items() if not isinstance(v, (list, dict))}})
        elif "random_mask" in p.name or "random_mask_rows" in d:
            mask_meta.append({"path": str(p.relative_to(root)), "value": d})
        if "preflight" in p.name:
            preflights.append({"path": str(p.relative_to(root)), "value": d})
    out = (out_root or root) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("frame_metrics", units), ("rescue_harm", recoveries), ("ranking", ranking),
                       ("prediction_drift", diagnostics), ("paired_contrast_units", comparison_units), ("training_steps", training_steps)]:
        write_csv(out / f"{name}.csv", rows)
    write_csv(out / "primary_effects.csv", [{"treated": r["treated"], "control": r["control"], "step": r["step"],
                "effect_pp": 100*r["mean"], "ci95_low_pp": 100*r["ci95"][0], "ci95_high_pp": 100*r["ci95"][1],
                "fold0_pp": 100*r["fold_means"][0], "fold1_pp": 100*r["fold_means"][1], "scenes": r["scenes"],
                "source_target_units": r["source_target_units"]}
                for r in contrasts if r["scope"] == "fixed128" and r["metric"] == "feasibility" and r["mean"] is not None])
    save_json(out / "analysis_spec.json", SPEC)
    preflight_path = root / "evaluation" / "preflight.json"
    preflight_calls = int(json.loads(preflight_path.read_text(encoding="utf-8")).get("calls", 0)) if preflight_path.exists() else 0
    total_calls = eval_calls + preflight_calls
    if total_calls > 131072:
        raise ValueError(f"Analytic evaluation budget exceeded: {total_calls}")
    result = {"status": "partial" if len(jobs) != expected else "complete", "jobs": len(jobs), "expected_jobs": expected,
              "input_root": str(root.resolve()),
              "unknown_appearances": total_unknown, "analytic_job_calls_receipts": eval_calls,
              "analytic_preflight_calls": preflight_calls, "analytic_calls_receipts": total_calls, "spec": SPEC,
              "summaries": summaries, "contrasts": contrasts, "mask_receipts": mask_meta, "preflights": preflights,
              "original_checkpoint_fixed_feasible_unique": sum(int(jobs[next(k for k in jobs if k[1] == t)]["base"]["good"].sum()) for t in baseline_seen),
              "original_checkpoint_fixed_queries_unique": 128*len(baseline_seen),
              "receipt_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "receipts").rglob("*.json"))}}
    save_json(out / "pulse_analysis.json", result)
    print(json.dumps({k: result[k] for k in ["status", "jobs", "expected_jobs", "unknown_appearances", "analytic_calls_receipts"]}))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--write-spec-only", action="store_true")
    ap.add_argument("--out-root", type=Path)
    args = ap.parse_args()
    if args.write_spec_only:
        save_json((args.out_root or args.root) / "analysis" / "analysis_spec.json", SPEC)
    else:
        analyze(args.root, args.allow_partial, args.out_root)
