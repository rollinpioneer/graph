"""POST_HOC_DIAGNOSTIC (evaluation only): prior-strength sweep on the saved PRIOR_BIAS checkpoints.

Weights are frozen (no backward, no optimizer). Contract logits are the policy's logits with the prior residual
removed (shadow pass); the final logits are
    logits = base_logits + alpha * centered_scaled_prior_score          (the existing PRIOR_BIAS score, in [-1, 1])
alpha in {0, 0.25, 0.5} on the same dev episodes and the same per-(game, repeat, seed) RNG as the pilot evaluations.
A harness check reuses the trained alpha = B*tanh(beta) and must reproduce the pilot's own evaluation exactly.
The best alpha is NOT a method result. Dev only.
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
W = {}
ALPHAS = [0.0, 0.25, 0.5]


def _init(run_dir, data_root, prior_path):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from cp_disr.platforms.alfworld import data
    from cp_disr.platforms.alfworld.prior_provider import PriorCache
    from cp_disr.platforms.alfworld.trainer import make_policy

    W["tables"] = data.Tables(json.load(open(run_dir + "/tables.json")))
    W["prior"] = PriorCache.load(prior_path)
    W["root"] = data_root
    W["policy"] = make_policy("PRIOR_BIAS", "cpu", 0).eval()
    W["ckpt"] = None


def _episode(policy, path, game, repeat, alpha):
    import torch
    from cp_disr.common import seed32
    from cp_disr.platforms.alfworld.adapter import AlfEpisode
    from cp_disr.platforms.alfworld.pddl_contracts import candidate_key
    from cp_disr.platforms.alfworld.snapshot import build_snapshot
    from cp_disr.platforms.alfworld.trainer import ENV_ID, H, full_prior_edges
    from cp_disr.torch_rl import prefix_hidden

    torch.manual_seed(seed32("alfworld_eval", game, repeat, 0))  # identical stream to the pilot evaluations
    ep = AlfEpisode(path, W["tables"])
    prefix, d, checks = [], 0, 0
    try:
        pub = ep.public()
        edges = full_prior_edges(pub, W["prior"], W["tables"])
        snap = build_snapshot(pub, edges, ENV_ID, "eval|%s|%d" % (game, repeat), d)
        while pub.legal and not ep.done:
            with torch.no_grad():
                policy.shadow_zero_prior = True  # contract (base) logits only
                out = policy(snap, prefix_hidden(policy, prefix))
                policy.shadow_zero_prior = False
                scores = torch.as_tensor(policy.prior_bias_scores(snap, snap.mask), dtype=out.logits.dtype)
                logits = (out.logits + alpha * scores).masked_fill(~out.mask, -torch.inf)
                idx = int(torch.distributions.Categorical(logits=logits).sample())
            cid = snap.candidate_ids[idx]
            kind, target = candidate_key(next(c for c in snap.template.contracts if c.id == cid))
            ep.execute(kind, target)
            checks += int(kind == "CHECK")
            prefix.append(snap)
            pub = ep.public()
            d += 1
            snap = build_snapshot(pub, edges, ENV_ID, "eval|%s|%d" % (game, repeat), d)
        return {"game": game, "repeat": repeat, "alpha": alpha, "J": (2.0 ** (-ep.success_step / H) if ep.won else 0.0), "won": bool(ep.won),
                "raw_steps": ep.raw_steps, "n_checks": checks}
    finally:
        ep.close()


def _chunk(args):
    ckpt, alphas, cells = args
    import torch
    from cp_disr.torch_rl import load_checkpoint

    pol = W["policy"]
    if W["ckpt"] != ckpt:
        load_checkpoint(ckpt, pol)  # weights only; the optimizer is not touched
        W["ckpt"] = ckpt
    for p in pol.parameters():
        p.requires_grad_(False)
    if alphas == "trained":
        alphas = [float(pol.B * torch.tanh(pol.prior_beta))]
    return [_episode(pol, os.path.join(W["root"], g), g, r, a) for g, r in cells for a in alphas]


def boot(vals_by_cluster, seed=0, n=10000):
    import numpy as np

    ks = sorted(vals_by_cluster)
    per = [float(np.mean(vals_by_cluster[k])) for k in ks]
    rng = np.random.default_rng(seed)
    means = [float(np.mean([per[i] for i in rng.integers(0, len(per), len(per))])) for _ in range(n)]
    return [float(np.mean(per)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(ks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/alfworld_prior_reliance")
    ap.add_argument("--data-root", default="/home/xushijie2/xsj2_alf/data")
    ap.add_argument("--prior", default="runs/alfworld_prior_reliance/prior_cache.json")
    ap.add_argument("--ckpt-dir", default="/home/xushijie2/xsj2_alf/e2_rescue/PRIOR_BIAS_s0")
    ap.add_argument("--pilot-eval-dir", default="runs/alfworld_prior_reliance/e2_rescue/PRIOR_BIAS_s0")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="runs/alfworld_prior_reliance/e2_rescue/alpha_diagnostic_POST_HOC.json")
    a = ap.parse_args()
    import numpy as np

    pool = get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.run_dir, a.data_root, a.prior))
    splits = json.load(open(a.run_dir + "/splits.json"))
    rows = {r["gamefile"]: r for r in json.load(open(a.run_dir + "/catalog.json"))}
    dev = splits["dev"]
    cells = [(g, r) for r in range(2) for g in dev]
    chunks = lambda: [cells[i::a.workers] for i in range(a.workers)]
    scene = {g: rows[g]["scene"] for g in dev}
    out = {"label": "POST_HOC_DIAGNOSTIC", "note": "evaluation only; frozen weights; best alpha is not a method result; single training seed",
           "alphas": ALPHAS, "checkpoints": {}}
    for tag, mark in (("earliest", 2048), ("latest", 8192)):
        ckpt = os.path.join(a.ckpt_dir, "ckpt_%09d.pt" % mark)
        meta = json.load(open(ckpt[:-3] + ".json"))["manifest"]
        recs = [x for ch in pool.map(_chunk, [(ckpt, ALPHAS, c) for c in chunks()]) for x in ch]
        chk = [x for ch in pool.map(_chunk, [(ckpt, "trained", c) for c in chunks()]) for x in ch]
        res = {"transitions": meta["transitions"], "per_alpha": {}, "paired_vs_alpha0": {}}
        by = {al: {(r["game"], r["repeat"]): r for r in recs if r["alpha"] == al} for al in ALPHAS}
        for al in ALPHAS:
            v = list(by[al].values())
            res["per_alpha"][str(al)] = {"J": float(np.mean([r["J"] for r in v])), "success": float(np.mean([r["won"] for r in v])),
                                         "raw_steps": float(np.mean([r["raw_steps"] for r in v])), "n_checks": float(np.mean([r["n_checks"] for r in v])), "episodes": len(v)}
        for al in ALPHAS[1:]:
            ent = {}
            for key in ("J", "raw_steps", "n_checks"):
                d_game, d_scene = {}, {}
                for (g, r), rec in by[al].items():
                    dv = rec[key] - by[0.0][(g, r)][key]
                    d_game.setdefault(g, []).append(dv)
                    d_scene.setdefault(scene[g], []).append(dv)
                ent[key] = {"mean_diff": float(np.mean([x for v in d_game.values() for x in v])), "game_cluster_ci95": boot(d_game)[1:3] + [len(d_game)],
                            "scene_cluster_ci95": boot(d_scene)[1:3] + [len(d_scene)]}
            res["paired_vs_alpha0"][str(al)] = ent
        pilot = {(r["game"], r["repeat"]): r for r in json.load(open(os.path.join(a.pilot_eval_dir, "eval_%09d_original.json" % meta["transitions"])))}
        same = sum(1 for x in chk if abs(pilot[(x["game"], x["repeat"])]["J"] - x["J"]) < 1e-12 and pilot[(x["game"], x["repeat"])]["raw_steps"] == x["raw_steps"])
        res["harness_check"] = {"alpha_trained": chk[0]["alpha"] if chk else None, "episodes": len(chk), "identical_to_pilot_eval": same}
        out["checkpoints"][tag] = res
    json.dump(out, open(a.out, "w"), indent=1)
    for tag, res in out["checkpoints"].items():
        print("==", tag, "transitions", res["transitions"], "| harness:", res["harness_check"])
        for al, v in res["per_alpha"].items():
            print("  alpha=%-5s J=%.4f steps=%.2f checks=%.2f succ=%.3f" % (al, v["J"], v["raw_steps"], v["n_checks"], v["success"]))
        for al, e in res["paired_vs_alpha0"].items():
            print("  alpha=%s vs 0: dJ=%+.4f game-CI[%+.4f,%+.4f] scene-CI[%+.4f,%+.4f] | dSteps=%+.3f | dChecks=%+.3f" % (
                al, e["J"]["mean_diff"], *e["J"]["game_cluster_ci95"][:2], *e["J"]["scene_cluster_ci95"][:2], e["raw_steps"]["mean_diff"], e["n_checks"]["mean_diff"]))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
