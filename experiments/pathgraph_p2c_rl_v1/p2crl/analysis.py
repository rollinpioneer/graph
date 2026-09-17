"""Statistics. Missing panels never become confirmatory PASS."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .errors import IncompletePanel
from .io_utils import write_new

PRIMARY = "PATHGRAPH_REMAINING_WORK_PBRS_V2"
GEOM = "GEOM_COUNT_EVENTS_PBRS_V1"
FLAT = "FLAT_CONTRACT_PBRS_V1"
N_BOOT = 20000
BOOT_SEED = 2026091806
MARGIN = 0.03
CI_LEVEL = 0.975

def panel_is_complete(counts):
    need = {
        "formal_jobs": 60,
        "validation_panels": 360,
        "main_rows": 30720,
        "stochastic_rows": 15360,
    }
    missing = {k: counts.get(k, 0) for k in need if counts.get(k, 0) < need[k]}
    return missing

def _unit_success(records, method):
    """Equal-weight motif macro: sides mean, then root, seed, draw, motif."""
    # records: dicts with draw, policy_seed, motif, family_id, side, method, success
    grouped = defaultdict(list)
    for r in records:
        if r["method"] != method:
            continue
        key = (r["draw"], r["policy_seed"], r["motif"], r["family_id"])
        grouped[key].append(float(r["success"]))
    if not grouped:
        return 0.0
    per_root = {k: float(np.mean(v)) for k, v in grouped.items()}
    per_motif = defaultdict(list)
    for (draw, seed, motif, fid), val in per_root.items():
        per_motif[motif].append(val)
    motif_means = [float(np.mean(v)) for v in per_motif.values()]
    return float(np.mean(motif_means)) if motif_means else 0.0

def bootstrap_paired_delta(records, method_a, method_b, n_boot=N_BOOT, seed=BOOT_SEED):
    empty = {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_boot": int(n_boot), "seed": int(seed), "margin_ok": False, "ci_ok": False}
    if not records:
        return empty
    rng = np.random.RandomState(int(seed))
    draws = sorted({r["draw"] for r in records})
    if not draws:
        return empty
    seeds_of = {d: sorted({r["policy_seed"] for r in records if r["draw"] == d}) for d in draws}
    motifs = sorted({r["motif"] for r in records})
    fams = {m: sorted({r["family_id"] for r in records if r["motif"] == m}) for m in motifs}
    index = defaultdict(list)
    for r in records:
        index[(r["method"], r["draw"], r["policy_seed"], r["motif"], r["family_id"], r.get("side", "left"))].append(float(r["success"]))

    def score(method, draw_list, seed_map, fam_map):
        motif_vals = []
        for m in motifs:
            vals = []
            for d in draw_list:
                for s in seed_map[d]:
                    for f in fam_map[m]:
                        sides = []
                        for side in ("left", "right"):
                            xs = index.get((method, d, s, m, f, side), [])
                            if xs:
                                sides.append(float(np.mean(xs)))
                        if sides:
                            vals.append(float(np.mean(sides)))
            if vals:
                motif_vals.append(float(np.mean(vals)))
        return float(np.mean(motif_vals)) if motif_vals else 0.0

    point = _unit_success(records, method_a) - _unit_success(records, method_b)
    deltas = []
    for _ in range(int(n_boot)):
        draw_list = [draws[i] for i in rng.randint(0, len(draws), size=len(draws))]
        seed_map = {}
        for d in draws:
            src = seeds_of[d]
            seed_map[d] = [src[i] for i in rng.randint(0, len(src), size=len(src))]
        fam_map = {}
        for m in motifs:
            src = fams[m]
            fam_map[m] = [src[i] for i in rng.randint(0, len(src), size=len(src))]
        deltas.append(score(method_a, draw_list, seed_map, fam_map) - score(method_b, draw_list, seed_map, fam_map))
    lo = float(np.quantile(deltas, 1.0 - CI_LEVEL))
    hi = float(np.quantile(deltas, CI_LEVEL))
    return {
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "margin_ok": float(point) >= MARGIN,
        "ci_ok": lo > 0.0,
    }

def confirmatory_statistics(records, counts, out_path=None):
    missing = panel_is_complete(counts)
    if missing:
        rec = {
            "schema": "P2CRL_PRIMARY_COMPARISONS_V1",
            "status": "INCOMPLETE_NO_CONFIRMATORY_PASS",
            "passed": False,
            "missing": missing,
        }
        if out_path is not None:
            write_new(out_path, rec)
        return rec
    g = bootstrap_paired_delta(records, PRIMARY, GEOM)
    f = bootstrap_paired_delta(records, PRIMARY, FLAT)
    passed = bool(g["margin_ok"] and g["ci_ok"] and f["margin_ok"] and f["ci_ok"])
    rec = {
        "schema": "P2CRL_PRIMARY_COMPARISONS_V1",
        "status": "RW_V2_POLICY_UTILITY_SUPPORTED_IN_REGISTERED_ABSTRACT_DOMAIN" if passed else "POLICY_UTILITY_NOT_ESTABLISHED_OR_MIXED",
        "passed": passed,
        "RW_V2_minus_GEOM": g,
        "RW_V2_minus_FLAT": f,
        "global_confirmation_passed": False,
    }
    if out_path is not None:
        write_new(out_path, rec)
    return rec
