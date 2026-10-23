"""Frozen-grid R3: local cache analysis only; no models, server, or pose evaluation."""
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "work/graph_archive/eg_module_controlled_diagnosis_20261023"
OLD = Path("C:/Users/jackx/Documents/Codex/2026-10-21/ssh-aita-xushijie-172-19-164/outputs/EG_pose_quality_20261022/data/cache")
OUT = ROOT / "outputs/eg_causal_v2_20261023/r3"
MANIFEST = Path(__file__).with_name("R3_manifest.json")
P = json.loads(MANIFEST.read_text())
OUT.mkdir(parents=True, exist_ok=True)
shutil.copyfile(MANIFEST, OUT / MANIFEST.name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def softmax(v):
    x = np.exp(v.astype(float) - np.max(v))
    return x / x.sum()

def comb(n, k):
    return math.comb(int(n), k) if n >= k else 0

def boot(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"scenes": 0, "mean": None, "ci95": None}
    r = np.random.default_rng(P["bootstrap"]["seed"])
    b = v[r.integers(0, len(v), (P["bootstrap"]["replicates"], len(v)))].mean(axis=1)
    return {"scenes": len(v), "mean": float(v.mean()), "ci95": np.quantile(b, [.025, .975]).tolist(),
            "positive_scenes": int((v > 1e-12).sum()), "negative_scenes": int((v < -1e-12).sum()),
            "zero_scenes": int((abs(v) <= 1e-12).sum())}

def paired(d, a, b):
    vals = d.assign(delta=d[a].astype(float) - d[b].astype(float)).groupby("scene").delta.mean()
    return {**boot(vals), "by_scene": {str(k): float(v) for k, v in vals.items()}}

def distributions(d, cols):
    return {c: {"mean": float(d[c].mean()), "median": float(d[c].median()),
                "q10": float(d[c].quantile(.1)), "q90": float(d[c].quantile(.9))} for c in cols}

g = pd.read_csv(SRC / "local_AD_grid.csv")
assert len(g) == 256 * 48
assert not g.mu.isna().any()
rows = []
sources = {str(SRC / "local_AD_grid.csv"): sha(SRC / "local_AD_grid.csv"), str(MANIFEST): sha(MANIFEST)}
checks = {"cached_pose_exact_frames": 0, "cached_logits_exact_frames": 0, "grid_baseline_exact_queries": 0,
          "grid_top2_exact_queries": 0, "candidate_id_exact_frames": 0, "width_decode_max_error_m": 0.0}
for (scene, frame), gf in g.groupby(["scene", "frame"], sort=True):
    tag = f"train_{scene:04d}_{frame:04d}.npz"
    zpath, opath = SRC / "data/cache" / tag, OLD / tag
    z, old = np.load(zpath), np.load(opath)
    sources[str(zpath)] = sha(zpath)
    sources[str(opath)] = sha(opath)
    assert np.array_equal(z["candidate_id"], old["candidate_id"])
    checks["candidate_id_exact_frames"] += 1
    assert np.array_equal(z["H0_pred"][:, 1:], old["pred"][:, 1:])
    checks["cached_pose_exact_frames"] += 1
    assert np.array_equal(z["H0_angle_logits"], old["angle_logits"])
    assert np.array_equal(z["H0_depth_logits"], old["depth_logits"])
    checks["cached_logits_exact_frames"] += 1
    for cid, q in gf.groupby("candidate_id", sort=False):
        q = q.sort_values(["angle", "depth"])
        assert len(q) == 48 and q[["angle", "depth"]].drop_duplicates().shape[0] == 48
        F = ((q.mu > 0) & (q.mu <= .4) & ~q.collision & ~q["empty"]).to_numpy().reshape(12, 4)
        la, ld = z["H0_angle_logits"][:, cid], z["H0_depth_logits"][:, cid]
        aa = np.argsort(-la[:12], kind="stable")[:2]
        dd = np.argsort(-ld[:4], kind="stable")[:2]
        a1, a2, d1, d2 = int(aa[0]), int(aa[1]), int(dd[0]), int(dd[1])
        assert a1 == old["angle_index"][cid] and d1 == old["depth_index"][cid]
        b = bool(F[a1, d1])
        top2 = bool(F[np.ix_(aa, dd)].any())
        assert b == bool(q.baseline_good.iloc[0])
        assert top2 == bool(q.top2_any.iloc[0])
        assert bool(F.any()) == bool(q.grid_any.iloc[0])
        checks["grid_baseline_exact_queries"] += 1
        checks["grid_top2_exact_queries"] += 1
        pa_full, pd_full = softmax(la), softmax(ld)
        pa, pd_ = softmax(la[:12]), softmax(ld[:4])
        ma = float(pa @ F.max(axis=1))
        md = float(pd_ @ F.max(axis=0))
        joint = float((pa[:, None] * pd_[None, :] * F).sum())
        local = [(a1, d1), ((a1 - 1) % 12, d1), ((a1 + 1) % 12, d1), (a1, d1 + 1 if d1 < 3 else d1 - 1)]
        assert len(set(local)) == 4
        m = int(F.sum())
        f21, f12, f22 = bool(F[a2, d1]), bool(F[a1, d2]), bool(F[a2, d2])
        raw = float(old["raw_width"][cid])
        scaled = raw * 1.2 / 10
        decoded = float(z["H0_pred"][cid, 1])
        derr = abs(decoded - np.clip(scaled, 0, .1))
        assert derr < 2e-8, derr
        checks["width_decode_max_error_m"] = max(checks["width_decode_max_error_m"], derr)
        valid = bool(old["batch_valid_mask"][cid])
        qref = float(old["batch_grasp_score"][cid])
        wref = float(old["batch_grasp_width"][cid])
        row = dict(scene=int(scene), frame=int(frame), candidate_id=int(cid), query_stratum=str(q.query_stratum.iloc[0]),
                   object_id=int(old["matched_object_id"][cid]), a1=a1, a2=a2, d1=d1, d2=d2,
                   baseline_good=b, feasible_cells=m, grid_any=bool(m), grid_none=(m == 0),
                   F_a2_d1=f21, F_a1_d2=f12, F_a2_d2=f22,
                   a1_elsewhere_feasible=bool(F[a1].any()), d1_elsewhere_feasible=bool(F[:, d1].any()),
                   individually_usable_combination_failed=bool(~np.bool_(b) and F[a1].any() and F[:, d1].any()),
                   recovery_pattern=(f"{int(f21)}{int(f12)}{int(f22)}" if not b else "baseline_feasible"),
                   any_single_change_recovery=bool(not b and (f21 or f12)),
                   both_change_only_recovery=bool(not b and f22 and not (f21 or f12)),
                   top2x2_any=top2, local4_any=any(F[a, d] for a, d in local),
                   random4_expectation=1 - comb(48 - m, 4) / comb(48, 4),
                   base_random3_expectation=1.0 if b else 1 - comb(47 - m, 3) / comb(47, 3),
                   angle_feasible_mass=ma, depth_feasible_mass=md, independent_joint_feasible_mass=joint,
                   marginal_joint_gap=min(ma, md) - joint,
                   high_marginals_low_joint=(ma >= .5 and md >= .5 and joint < .25),
                   angle_extra_class_mass=float(pa_full[12]), depth_extra_class_mass=float(pd_full[4]),
                   angle_full_argmax_extra=(int(la.argmax()) == 12), depth_full_argmax_extra=(int(ld.argmax()) == 4),
                   angle_prob_margin=float(pa[a1] - pa[a2]), depth_prob_margin=float(pd_[d1] - pd_[d2]),
                   angle_logit_margin=float(la[a1] - la[a2]), depth_logit_margin=float(ld[d1] - ld[d2]),
                   raw_width=raw, preclip_width_m=scaled, decoded_width_m=decoded,
                   width_upper_saturated=bool(scaled >= .1), width_lower_saturated=bool(scaled <= 0),
                   valid_match=valid, qref=qref, qref_zero_valid=bool(valid and qref == 0),
                   width_ref_m=wref, width_ref_zero_valid=bool(valid and wref == 0),
                   width_ref_over100mm_valid=bool(valid and wref > .1),
                   local4_cells=json.dumps(local), frozen_grid_key=f"train_{scene:04d}_{frame:04d}:{cid}")
        rows.append(row)

d = pd.DataFrame(rows).sort_values(["scene", "frame", "candidate_id"])
assert len(d) == 256 and d.baseline_good.sum() == 119 and (~d.baseline_good).sum() == 137
assert d.top2x2_any.sum() == 178 and d.grid_any.sum() == 192
assert ((~d.baseline_good) & d.top2x2_any).sum() == 59
d.to_csv(OUT / "AD_joint_structure.csv", index=False)

METRICS = ["baseline_good", "top2x2_any", "local4_any", "random4_expectation", "base_random3_expectation", "grid_any"]
SCALARS = ["angle_feasible_mass", "depth_feasible_mass", "independent_joint_feasible_mass", "marginal_joint_gap", "angle_prob_margin", "depth_prob_margin"]
def subset_summary(x):
    fail = x[~x.baseline_good]
    obs = {c: dict(sum=float(x[c].sum()), mean=float(x[c].mean()),
                   recovery_sum=float(x.loc[~x.baseline_good, c].sum()),
                   harm_sum=float((1 - x.loc[x.baseline_good, c].astype(float)).sum())) for c in METRICS}
    return dict(n=len(x), scenes=int(x.scene.nunique()), original_feasible=int(x.baseline_good.sum()),
                original_failed=len(fail), coverage=obs,
                comparisons={f"{a}_minus_{b}": paired(x, a, b) for a, b in [
                    ("top2x2_any", "random4_expectation"), ("top2x2_any", "local4_any"),
                    ("top2x2_any", "base_random3_expectation"), ("top2x2_any", "baseline_good")]},
                failed_comparisons={f"{a}_minus_{b}": paired(fail, a, b) for a, b in [
                    ("top2x2_any", "random4_expectation"), ("top2x2_any", "local4_any"),
                    ("top2x2_any", "base_random3_expectation")]},
                failed_recovery_intersections={str(k): int(v) for k, v in fail.recovery_pattern.value_counts().sort_index().items()},
                failed_counts={c: int(fail[c].sum()) for c in ["F_a2_d1", "F_a1_d2", "F_a2_d2", "any_single_change_recovery", "both_change_only_recovery", "individually_usable_combination_failed", "high_marginals_low_joint"]},
                actual_pose_swaps={c: dict(feasible=int(x[c].sum()), recovery=int((x[c] & ~x.baseline_good).sum()),
                    harm=int((~x[c] & x.baseline_good).sum()), scene_paired=paired(x, c, "baseline_good"))
                    for c in ["F_a2_d1", "F_a1_d2", "F_a2_d2"]},
                distributions=distributions(x, SCALARS), failed_distributions=distributions(fail, SCALARS))

width_assoc = {}
for name, x in [("all", d), ("valid_matches", d[d.valid_match])]:
    group = {}
    for is_none, t in x.groupby("grid_none"):
        group[str(bool(is_none))] = dict(n=len(t), scenes=int(t.scene.nunique()), width_mean_mm=float(t.decoded_width_m.mean() * 1000),
            width_median_mm=float(t.decoded_width_m.median() * 1000), valid=int(t.valid_match.sum()),
            upper_saturated=int(t.width_upper_saturated.sum()), lower_saturated=int(t.width_lower_saturated.sum()),
            qref_zero_valid=int(t.qref_zero_valid.sum()), ref_zero_valid=int(t.width_ref_zero_valid.sum()),
            ref_over100mm_valid=int(t.width_ref_over100mm_valid.sum()))
    comparisons = {}
    for c in ["decoded_width_m", "width_upper_saturated", "qref_zero_valid", "width_ref_zero_valid", "width_ref_over100mm_valid"]:
        t = x.groupby(["scene", "grid_none"])[c].mean().unstack("grid_none")
        both = t.dropna()
        vals = both[True] - both[False]
        comparisons[c] = {**boot(vals), "by_scene": {str(k): float(v) for k, v in vals.items()}}
    width_assoc[name] = {"group_grid_none": group, "scene_paired_none_minus_some": comparisons}

scene = d.groupby("scene").agg(n=("candidate_id", "size"), baseline_good=("baseline_good", "sum"),
    top2x2_any=("top2x2_any", "sum"), local4_any=("local4_any", "sum"), random4_expectation=("random4_expectation", "sum"),
    grid_any=("grid_any", "sum"), individually_usable_combination_failed=("individually_usable_combination_failed", "sum"))
scene.to_csv(OUT / "R3_scene_counts.csv")
patterns = {key: distributions(t, ["angle_prob_margin", "depth_prob_margin", "feasible_cells"]) for key, t in d[~d.baseline_good].groupby("recovery_pattern")}
summary = dict(population=dict(queries=256, frames=16, scenes=8, split="train", original_feasible=119, original_failed=137,
                               raw_top1=16, rank_stratified=112, random=128, grid_unknown=0),
               invariants=checks, all=subset_summary(d), strata={k: subset_summary(t) for k, t in d.groupby("query_stratum")},
               scene_counts=scene.reset_index().to_dict("records"), width_association=width_assoc,
               extra_classes={c: {"mean": float(d[c].mean()), "max": float(d[c].max())} for c in ["angle_extra_class_mass", "depth_extra_class_mass"]},
               extra_argmax={c: int(d[c].sum()) for c in ["angle_full_argmax_extra", "depth_full_argmax_extra"]},
               failed_patterns_margin=patterns, source_sha256=sources,
               permissions={"server_access": False, "parameter_updates": 0, "new_analytic_evaluations": 0, "git": False},
               limitations=["Eight training scenes; query selection mixture is not deployment Top-1 distribution", "Scene bootstrap with only eight units is imprecise",
                   "Set feasibility is oracle coverage at frozen baseline width", "Softmax values are descriptive mass, not calibrated success probability",
                   "Baseline cache uses fixed inference queries with attached labels, not historical training exposure"])
(OUT / "R3_scene_paired_statistics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"all": summary["all"], "width": width_assoc, "extra": summary["extra_classes"], "scene": summary["scene_counts"]}, indent=2))
