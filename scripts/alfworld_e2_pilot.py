"""E2-rescue pilot run: one method, seed 0, PPO to 2,048 real transitions (validity gate) then on to 8,192.

Learning code is the frozen trainer (run_episode / PPO / PriorSampler untouched). This script only adds:
  * a process pool (spawn, CPU policy copies) for dev evaluation with the paired per-(game, repeat, seed) RNG,
  * dev evaluation at marks 0,1024,...,8192 with Original prior; Absent-prior evaluation at 2048/4096/8192,
  * logging of strata, prior-induced TV, residual spread, PRIOR_BIAS beta, throughput.
Dev only: the test split and official valid_unseen are never read here.
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

W = {}


def _worker_init(run_dir, data_root, prior_path, method):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from cp_disr.platforms.alfworld import data
    from cp_disr.platforms.alfworld.prior_provider import PriorCache
    from cp_disr.platforms.alfworld.trainer import make_policy

    W["tables"] = data.Tables(json.load(open(run_dir + "/tables.json")))
    W["prior"] = PriorCache.load(prior_path)
    W["root"] = data_root
    W["policy"] = make_policy(method, "cpu", 0).eval()
    W["version"] = None


def _eval_chunk(args):
    """Evaluate (game, repeat) cells with a CPU copy of the learner policy; stochastic, per-episode seeded."""
    version, state, cells, seed, prior_mode = args
    import torch
    from cp_disr.common import seed32
    from cp_disr.platforms.alfworld.pilot_eval import eval_episode_record

    pol = W["policy"]
    if W["version"] != version:
        pol.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        W["version"] = version
    out = []
    for g, r in cells:
        torch.manual_seed(seed32("alfworld_eval", g, r, seed))
        out.append(eval_episode_record(pol, os.path.join(W["root"], g), W["tables"], W["prior"], g, r, prior_mode))
    return out


def _quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"mean": sum(xs) / len(xs), "p50": q(0.5), "p90": q(0.9), "max": xs[-1], "n": len(xs)}


def summarise(recs):
    n = len(recs)
    tv = [r["tv_mean"] for r in recs if r["tv_mean"] is not None]
    spread = [s for r in recs for s in r["resid_spread"]]
    absr = [s for r in recs for s in r["resid_abs_mean"]]
    return {"J": sum(r["J"] for r in recs) / n, "success": sum(r["won"] for r in recs) / n, "raw_steps": sum(r["raw_steps"] for r in recs) / n,
            "n_checks": sum(r["n_checks"] for r in recs) / n, "tv_mean": (sum(tv) / len(tv) if tv else None), "episodes": n,
            "residual_candidate_spread": _quantiles(spread), "residual_abs_mean": _quantiles(absr)}


def main():
    import faulthandler
    faulthandler.dump_traceback_later(1800, repeat=True, file=sys.stderr)  # diagnostic only
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["B2", "Full", "PRIOR_BIAS"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default="runs/alfworld_prior_reliance")
    ap.add_argument("--data-root", default="/home/xushijie2/xsj2_alf/data")
    ap.add_argument("--prior", default="runs/alfworld_prior_reliance/prior_cache.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--gate", type=int, default=2048)
    ap.add_argument("--final", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=1024)
    ap.add_argument("--eval-repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    ctx = get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=_worker_init, initargs=(a.run_dir, a.data_root, a.prior, a.method))  # before CUDA init

    import torch

    torch.set_num_threads(a.threads)
    from cp_disr.platforms.alfworld import data
    from cp_disr.platforms.alfworld.prior_provider import PriorCache
    from cp_disr.platforms.alfworld.trainer import GameSchedule, full_prior_edges, make_policy, run_episode
    from cp_disr.prior import PriorSampler
    from cp_disr.rl import Rollout, set_suite_half_life
    from cp_disr.torch_rl import PPO, save_checkpoint

    set_suite_half_life(50.0)
    os.makedirs(a.out, exist_ok=True)
    splits = json.load(open(a.run_dir + "/splits.json"))
    tables = data.Tables(json.load(open(a.run_dir + "/tables.json")))
    prior = PriorCache.load(a.prior)
    dev = splits["dev"]
    policy = make_policy(a.method, a.device, a.seed)
    ppo = PPO(policy, lr=3e-4)
    sampler, sched = PriorSampler(a.seed), GameSchedule(splits["train"], a.seed)
    fn = lambda pub: full_prior_edges(pub, prior, tables)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    evals = open(os.path.join(a.out, "eval_log.jsonl"), "a")
    meta = {"method": a.method, "seed": a.seed, "rollout": a.rollout, "prior_hash": prior.hash(), "prior_model": prior.meta.get("model"), "H": 50,
            "n_dev": len(dev), "eval_repeats": a.eval_repeats, "device": a.device, "workers": a.workers,
            "n_params": sum(p.numel() for p in policy.parameters())}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    version = [0]

    def do_eval(transitions, mode, wall):
        t0 = time.time()
        version[0] += 1
        state = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}  # plain arrays: no torch shared-memory pickling
        cells = [(g, r) for r in range(a.eval_repeats) for g in dev]
        chunks = [cells[i::a.workers] for i in range(a.workers)]
        res = pool.map(_eval_chunk, [(version[0], state, c, a.seed, mode) for c in chunks])
        recs = sorted([x for ch in res for x in ch], key=lambda r: (r["game"], r["repeat"]))
        json.dump(recs, open(os.path.join(a.out, "eval_%09d_%s.json" % (transitions, mode)), "w"))
        s = summarise(recs)
        s.update(transitions=transitions, mode=mode, t_eval=time.time() - t0, wall=wall)
        if a.method == "PRIOR_BIAS":
            s["beta"] = float(policy.prior_beta)
            s["tanh_beta"] = math.tanh(float(policy.prior_beta))
        evals.write(json.dumps(s) + "\n")
        evals.flush()
        return s

    t_start = time.time()
    done_t = n_ep = 0
    t_collect = t_update = t_eval = 0.0
    eval_marks = list(range(0, a.final + 1, a.eval_every))
    absent_marks = {a.gate, a.gate * 2, a.final}
    strata = {"original": [0, 0.0, 0.0], "absent": [0, 0.0, 0.0]}  # n episodes, sum J, sum success
    checks_total = [0, 0]  # CHECK macros, episodes (training)
    gate_checked = False
    saved = set()

    def maybe_eval(force_final=False):
        nonlocal t_eval
        while eval_marks and done_t >= eval_marks[0]:
            mark = eval_marks.pop(0)
            e0 = time.time()
            s = do_eval(done_t, "original", time.time() - t_start)
            if mark in absent_marks or (force_final and mark == a.final):
                do_eval(done_t, "absent", time.time() - t_start)
            t_eval += time.time() - e0
            print(json.dumps({"eval": {k: s[k] for k in ("transitions", "J", "success", "n_checks", "tv_mean")}}), flush=True)

    maybe_eval()  # initialisation (0 transitions)
    while done_t < a.final:
        rollout, ep_stats, c0 = Rollout(), [], time.time()
        while len(rollout.transitions) < a.rollout:
            g = sched.next()
            trs, st = run_episode(policy, os.path.join(a.data_root, g), tables, fn, "train|%d|%s" % (n_ep, g), sampler, "train")
            n_ep += 1
            rollout.transitions.extend(trs)
            ep_stats.append(st)
            sm = strata[st["prior_mode"]]
            sm[0] += 1; sm[1] += st["J"]; sm[2] += float(st["won"])
            checks_total[0] += sum(1 for t in trs if ":CHECK:" in t.selected_candidate_id)
            checks_total[1] += 1
        t_collect += time.time() - c0
        u0 = time.time()
        n_tr = len(rollout.transitions)
        logs = ppo.update(rollout, epochs=4, minibatch=64, sequence_length=16)
        t_update += time.time() - u0
        done_t += n_tr
        loss = sum(l["total"] for l in logs) / len(logs)
        row = {"transitions": done_t, "episodes": n_ep, "train_J": sum(s["J"] for s in ep_stats) / len(ep_stats),
               "train_success": sum(s["won"] for s in ep_stats) / len(ep_stats), "loss": loss, "grad_norm": sum(l["grad_norm"] for l in logs) / len(logs),
               "strata": {k: {"episodes": v[0], "mean_J": (v[1] / v[0] if v[0] else None), "success": (v[2] / v[0] if v[0] else None)} for k, v in strata.items()},
               "train_checks_per_episode": checks_total[0] / max(1, checks_total[1]), "wall": time.time() - t_start,
               "t_collect": t_collect, "t_update": t_update, "t_eval": t_eval, "trans_per_hour_train_only": done_t / max(1e-9, t_collect + t_update) * 3600}
        if a.method == "PRIOR_BIAS":
            row["beta"] = float(policy.prior_beta)
        log.write(json.dumps(row) + "\n")
        log.flush()
        for mark in (a.gate, a.gate * 2, a.final):
            if done_t >= mark and mark not in saved:
                saved.add(mark)
                p = os.path.join(a.out, "ckpt_%09d.pt" % mark)
                save_checkpoint(p, policy, ppo.optimizer, {"method": a.method, "seed": a.seed, "transitions": done_t, "episodes": n_ep, "nominal_mark": mark})
        maybe_eval()
        if not gate_checked and done_t >= a.gate:
            gate_checked = True
            finite = math.isfinite(loss) and all(torch.isfinite(p).all() for p in policy.parameters())
            last = [json.loads(l) for l in open(os.path.join(a.out, "eval_log.jsonl"))][-2:]
            ok = finite and n_ep > 0 and strata["original"][0] > 0 and strata["absent"][0] > 0 and all(e["episodes"] > 0 for e in last)
            json.dump({"gate_transitions": done_t, "technically_valid": bool(ok), "finite": bool(finite), "episodes": n_ep,
                       "strata_episodes": {k: v[0] for k, v in strata.items()}}, open(os.path.join(a.out, "gate_2048.json"), "w"), indent=1)
            if not ok:
                print("TECHNICALLY INVALID at the gate; stopping", flush=True)
                sys.exit(3)
    maybe_eval(force_final=True)
    json.dump({"transitions": done_t, "episodes": n_ep, "wall_s": time.time() - t_start, "t_collect": t_collect, "t_update": t_update, "t_eval": t_eval},
              open(os.path.join(a.out, "done.json"), "w"), indent=1)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
