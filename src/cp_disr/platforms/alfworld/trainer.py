"""On-policy PPO collection, dev evaluation and checkpointing for ALFWorld (reuses rl.py / torch_rl.py).

One macro = one semi-MDP transition; duration = raw ALFWorld commands; reward is the discounted
terminal success (H = 50 raw steps), so the start-discounted return of an episode is exactly
J = 2 ** (-T / 50) for a success at raw step T and 0 otherwise.
"""
import json
import os
import random
import time

import torch

from ...common import seed32
from ...prior import PriorSampler
from ...rl import Rollout, Transition, gamma, interval_reward, set_suite_half_life
from ...torch_rl import PPO, prefix_hidden, save_checkpoint
from .adapter import AlfEpisode
from .pddl_contracts import KINDS, PREDICATE_TYPES, candidate_key, episode_template
from .snapshot import BASE_DIM, CAND_DIM, build_snapshot, prior_edges_for

H = 50.0
ENV_ID = "alfworld"


def make_policy(method, device="cpu", seed=0):
    from ...neural_batched import BatchedPolicy

    torch.manual_seed(seed)
    return BatchedPolicy(set(KINDS), set(PREDICATE_TYPES), {"receptacle"}, BASE_DIM, CAND_DIM, method=method).to(device)


class GameSchedule:
    """Same task order for the same seed across methods (epoch-wise shuffled train games)."""

    def __init__(self, games, seed):
        self.games, self.rng, self.queue, self.n = list(games), random.Random(seed32("alfworld_train_order", 0, 0, seed)), [], 0

    def next(self):
        if not self.queue:
            self.queue = list(self.games)
            self.rng.shuffle(self.queue)
        self.n += 1
        return self.queue.pop()


def full_prior_edges(pub, prior, tables):
    return prior_edges_for(episode_template(pub.feasible, pub.goal_instance), prior.scores(pub, tables))


def run_episode(policy, game_path, tables, prior_edges_fn, episode_id, sampler=None, mode="train", deterministic=False, record=False):
    """Collect one episode. mode='train' uses the 80/20 Original/Absent sampler; 'eval' always Original."""
    ep = AlfEpisode(game_path, tables)
    transitions, prefix, weight, decision = [], [], 1.0, 0
    stats = {"J": 0.0, "won": False, "raw_steps": 0, "decisions": 0, "tv": [], "prior_mode": None}
    try:
        pub = ep.public()
        original = prior_edges_fn(pub)
        if mode == "train":
            pe = sampler.start(ENV_ID, episode_id, original)
            edges, stats["prior_mode"] = pe.edges, pe.audit_mode
        else:
            edges, stats["prior_mode"] = original, "original"
        snap = build_snapshot(pub, edges, ENV_ID, episode_id, decision)
        while pub.legal and not ep.done:
            with torch.no_grad():
                out = policy(snap, prefix_hidden(policy, prefix))
                if record:
                    policy.shadow_zero_prior = True
                    shadow = policy(snap, prefix_hidden(policy, prefix))
                    policy.shadow_zero_prior = False
                    stats["tv"].append(0.5 * float((out.distribution.probs - shadow.distribution.probs).abs().sum()))
            cand, index = out.select(deterministic)
            contract = next(c for c in snap.template.contracts if c.id == cand)
            kind, target = candidate_key(contract)
            res = ep.execute(kind, target)
            duration = res.duration
            events = [(float(duration), 1)] if res.status == "WON" else []
            reward = interval_reward(duration, events, H)
            terminated = ep.done
            reason = "SUCCESS" if ep.won else "DEADLINE" if ep.truncated else "RUNNING"
            pub = ep.public()
            nxt = build_snapshot(pub, edges, ENV_ID, episode_id, decision + 1)
            with torch.no_grad():
                next_v = 0.0 if terminated else float(policy(nxt, out.hidden).value)
            if mode == "train":
                transitions.append(Transition(snap, nxt, cand, float(out.distribution.log_prob(torch.tensor(index, device=out.logits.device))), float(out.value), next_v,
                                              reward, float(duration), weight, terminated, False, reason, tuple(prefix)))
            prefix.append(snap)
            weight *= gamma(duration, H)
            snap, decision = nxt, decision + 1
        stats.update(won=ep.won, raw_steps=ep.raw_steps, decisions=decision, J=(2.0 ** (-ep.success_step / H) if ep.won else 0.0))
    finally:
        ep.close()
    return transitions, stats


def evaluate(policy, games, tables, prior, data_root, repeats=4, seed=0, record=True):
    """Stochastic-policy evaluation on a fixed episode list (same RNG seeds for every method)."""
    policy.eval()
    Js, wins, steps, tvs = [], [], [], []
    fn = lambda pub: full_prior_edges(pub, prior, tables)
    for r in range(repeats):
        torch.manual_seed(seed32("alfworld_eval", 0, r, seed))
        for g in games:
            _, st = run_episode(policy, os.path.join(data_root, g), tables, fn, "eval|%s|%d" % (g, r), mode="eval", record=record)
            Js.append(st["J"]); wins.append(float(st["won"])); steps.append(st["raw_steps"]); tvs += st["tv"]
    policy.train()
    n = len(Js)
    return {"J": sum(Js) / n, "success": sum(wins) / n, "raw_steps": sum(steps) / n, "tv": (sum(tvs) / len(tvs) if tvs else None), "episodes": n}


def train(cfg):
    """cfg: dict(method, seed, games, eval_games, tables, prior, data_root, out_dir, n_total, rollout, eval_every, device, lr, save_at)"""
    set_suite_half_life(H)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    policy = make_policy(cfg["method"], cfg.get("device", "cpu"), cfg["seed"])
    ppo = PPO(policy, lr=cfg.get("lr", 3e-4))
    sampler = PriorSampler(cfg["seed"])
    sched = GameSchedule(cfg["games"], cfg["seed"])
    tables, prior, root = cfg["tables"], cfg["prior"], cfg["data_root"]
    fn = lambda pub: full_prior_edges(pub, prior, tables)
    log = open(os.path.join(cfg["out_dir"], "train_log.jsonl"), "a")
    done_t, n_ep, t0 = 0, 0, time.time()
    save_at = sorted(cfg.get("save_at", []))
    next_eval = cfg["eval_every"]
    t_collect = t_update = 0.0
    while done_t < cfg["n_total"]:
        rollout, ep_stats, c0 = Rollout(), [], time.time()
        while len(rollout.transitions) < cfg["rollout"]:
            g = sched.next()
            trs, st = run_episode(policy, os.path.join(root, g), tables, fn, "train|%d|%s" % (n_ep, g), sampler, "train")
            n_ep += 1
            for t in trs:
                rollout.append(t)
            ep_stats.append(st)
        t_collect += time.time() - c0
        u0 = time.time()
        n_tr = len(rollout.transitions)
        logs = ppo.update(rollout, epochs=cfg.get("epochs", 4), minibatch=cfg.get("minibatch", 64), sequence_length=cfg.get("seq", 16))
        t_update += time.time() - u0
        done_t += n_tr
        row = {"transitions": done_t, "episodes": n_ep, "train_J": sum(s["J"] for s in ep_stats) / len(ep_stats),
               "train_success": sum(s["won"] for s in ep_stats) / len(ep_stats), "loss": sum(l["total"] for l in logs) / len(logs),
               "wall": time.time() - t0, "t_collect": t_collect, "t_update": t_update, "trans_per_hour": done_t / max(1e-9, time.time() - t0) * 3600}
        while save_at and done_t >= save_at[0]:
            mark = save_at.pop(0)
            path = os.path.join(cfg["out_dir"], "ckpt_%09d.pt" % mark)
            if not os.path.exists(path):
                save_checkpoint(path, policy, ppo.optimizer, {"method": cfg["method"], "seed": cfg["seed"], "transitions": done_t, "episodes": n_ep})
            row["saved"] = path
        if done_t >= next_eval or done_t >= cfg["n_total"]:
            e0 = time.time()
            row["dev"] = evaluate(policy, cfg["eval_games"], tables, prior, root, repeats=cfg.get("eval_repeats", 2), seed=cfg["seed"])
            row["t_eval"] = time.time() - e0
            next_eval += cfg["eval_every"]
        log.write(json.dumps(row) + "\n")
        log.flush()
    log.close()
    return policy
