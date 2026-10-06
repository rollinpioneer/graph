"""Training / evaluation loops for the Blocksworld cross-task card (runbook 12, 13). Reuses the production PPO, Transition, Rollout and recurrence code unchanged."""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

from ..rl import Rollout, Transition, gamma, set_suite_half_life
from ..torch_rl import PPO, prefix_hidden, save_checkpoint
from . import metrics as M
from . import state as S
from .environment import BwEpisode, Case, step_cap_for
from .planner import Solver


def case_from_json(d):
    problem = S.Problem(tuple(d["names"]), tuple(d["colors"]), tuple(d["init"]), tuple(d["goal"]))
    return Case(d["case_id"], d["split"], problem, int(d["optimal_length"]), int(d["step_cap"]))


def load_cases(path, key):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [case_from_json(c) for c in doc[key]], doc


def interleave_by_size(cases):
    """Deterministic training order that cycles through the block counts (every window covers every size)."""
    groups = {}
    for c in cases:
        groups.setdefault(c.problem.n, []).append(c)
    order = []
    for i in range(max(len(g) for g in groups.values())):
        for n in sorted(groups):
            if i < len(groups[n]):
                order.append(groups[n][i])
    return order


class Hops:
    """Per-template HopIndex cache for decision logging."""

    def __init__(self):
        self._cache = {}

    def get(self, template):
        hit = self._cache.get(id(template))
        if hit is None or hit[0] is not template:
            hit = (template, M.HopIndex(template))
            self._cache[id(template)] = hit
        return hit[1]


@torch.no_grad()
def evaluate_case(policy, case, solver, hops, record_decisions=True):
    """One deterministic-argmax episode with the exact planner labelling every visited state (runbook 8.3)."""
    ep = BwEpisode(case)
    hopindex = hops.get(ep.template)
    hidden = policy.initial_hidden()
    decisions, trace = [], []
    problem = case.problem
    while not ep.done:
        snap = ep.snapshot()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        out = policy(snap, hidden)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t_fwd
        hidden = out.hidden
        selected_id, idx = out.select(True)
        state = ep.state
        L_before, opt = solver.optimal_actions(state, problem.goal)
        opt_ids = [S.action_id(problem.names, a) for a in opt]
        before = S.goal_atom_truth(problem, state)
        legal_ids = [snap.candidate_ids[i] for i, m in enumerate(snap.mask) if m]
        hops_by_action = hopindex.candidate_goal_hops(problem, state, legal_ids)
        d_hop = M.decision_hop(hops_by_action, opt_ids)
        ep.step(selected_id)
        after = S.goal_atom_truth(problem, ep.state)
        trace.append((state, selected_id))
        probs = out.distribution.probs
        rec = {"decision_index": len(decisions), "state": list(state), "selected": selected_id, "optimal_actions": opt_ids, "selected_is_optimal": selected_id in opt_ids, "optimal_remaining": L_before,
               "selected_excess_cost": None, "decision_hop": None if d_hop == M.INF else d_hop, "hop_stratum": M.stratum(d_hop), "selected_hop": None if hops_by_action[selected_id] == M.INF else hops_by_action[selected_id],
               "satisfied_before": sum(before.values()), "satisfied_after": sum(after.values()), "destroyed_satisfied_goal": M.goal_destroyed(before, after),
               "probs": {i: round(float(probs[snap.candidate_ids.index(i)]), 6) for i in legal_ids}, "forward_seconds": t_fwd, "encoder_graph_encodings": (out.diagnostics or {}).get("encoder_graph_encodings"),
               "n_legal": len(legal_ids)}
        L_after = solver.cost_to_go(ep.state, problem.goal)
        rec["selected_excess_cost"] = (L_after + 1 - L_before) if L_after is not None and L_before is not None else None
        decisions.append(rec)
    flags = [d["selected_is_optimal"] for d in decisions]
    flags_le4 = [d["selected_is_optimal"] for d in decisions if d["decision_hop"] is not None and d["decision_hop"] <= 4]
    flags_gt4 = [d["selected_is_optimal"] for d in decisions if d["decision_hop"] is None or d["decision_hop"] > 4]
    summary = {"case_id": case.case_id, "split": case.split, "n_blocks": problem.n, "success": ep.success, "reason": ep.reason, "steps": ep.step_index, "optimal_length": case.optimal_length,
               "step_cap": case.step_cap, "decision_perfect": all(flags), "decision_perfect_hop_le4": all(flags_le4), "decision_perfect_hop_gt4": all(flags_gt4),
               "n_decisions_hop_le4": len(flags_le4), "n_decisions_hop_gt4": len(flags_gt4), "first_divergence": M.first_divergence(flags),
               "destroyed_satisfied_goal_count": sum(len(d["destroyed_satisfied_goal"]) for d in decisions), "cycle": M.repeated_state_action_cycle(trace),
               "excess_steps": M.excess_steps(ep.success, ep.step_index, case.optimal_length)}
    if record_decisions:
        summary["decisions"] = decisions
    return summary


def evaluate(policy, cases, solver, hops, record_decisions=True):
    policy.eval()
    return [evaluate_case(policy, c, solver, hops, record_decisions) for c in cases]


def summarize(episodes):
    n = len(episodes)
    return {"n": n, "success_n": sum(e["success"] for e in episodes), "decision_perfect_n": sum(e["decision_perfect"] for e in episodes),
            "success_rate": sum(e["success"] for e in episodes) / n if n else None, "decision_perfect_rate": sum(e["decision_perfect"] for e in episodes) / n if n else None}


class BatchedPPO(PPO):
    """The production PPO update (same chunking, minibatches, targets, losses, clipping and optimizer) with the policy outputs of a minibatch computed by one grouped forward.

    Only the kernel grouping differs from ``torch_rl.PPO.update``: the graphs of all transitions of a minibatch that share a template go through the encoder as one batch; the GRU still runs
    sequentially per chunk. Equivalence of the resulting losses / gradients is a pre-training test."""

    def update(self, rollout, epochs=4, minibatch=64, sequence_length=16):
        from ..c1_blocksworld_policies import batched_outputs
        from ..rl import scalar_targets
        from ..torch_rl import ppo_losses, sequence_chunks, global_grad_norm
        from ..common import DataIntegrityError
        ts = tuple(rollout.transitions)
        if not ts:
            raise DataIntegrityError("No real transitions")
        device = next(self.policy.parameters()).device
        a, v, q = scalar_targets(ts)
        targets = [torch.tensor(x, device=device, dtype=torch.float32).detach() for x in (a, v, q)]
        chunks = sequence_chunks(ts, sequence_length)
        logs = []
        for epoch in range(epochs):
            batch = []
            for chunk_index, chunk in enumerate(chunks):
                batch.append(chunk)
                if sum(map(len, batch)) < minibatch and chunk_index + 1 < len(chunks):
                    continue
                snaps, zos, selected, indices = [], [], [], []
                for sequence in batch:
                    hidden = prefix_hidden(self.policy, ts[sequence[0]].prefix)
                    for i in sequence:
                        t = ts[i]
                        hidden = self.policy.advance_hidden(t.snapshot.base_input, hidden)
                        snaps.append(t.snapshot)
                        zos.append(hidden)
                        selected.append(t.selected_candidate_id)
                        indices.append(i)
                lp, vs, qs, ent = batched_outputs(self.policy, self.policy.row_kind, snaps, torch.stack(zos), selected)
                old = torch.tensor([ts[i].old_logp for i in indices], device=device)
                weights = torch.tensor([ts[i].weight for i in indices], device=device)
                losses = ppo_losses(lp, old, targets[0][indices], weights, vs, targets[1][indices], qs, targets[2][indices], ent, self.policy.q_coefficient)
                if not torch.isfinite(losses["total"]):
                    raise DataIntegrityError("Nonfinite PPO objective")
                self.optimizer.zero_grad()
                losses["total"].backward()
                params = tuple(self.policy.parameters())
                pre = global_grad_norm(params)
                torch.nn.utils.clip_grad_norm_(params, 0.5, error_if_nonfinite=True)
                post = global_grad_norm(params)
                self.optimizer.step()
                self.optimizer_step_id += 1
                logs.append({"epoch": epoch, "valid_transitions": len(indices), "pre_clip_global_grad_norm": pre, "post_clip_global_grad_norm": post, "clip_threshold": 0.5, "optimizer_step_id": self.optimizer_step_id,
                             **{k: (float(x.detach()) if torch.is_tensor(x) else float(x)) for k, x in losses.items()}})
                batch = []
        rollout.clear()
        return logs


class Trainer:
    """PPO on the Blocksworld cases (episodes continue across update boundaries, as in the production collector)."""

    def __init__(self, policy, train_cases, device, seed=0):
        self.policy = policy
        self.order = interleave_by_size(train_cases)
        self.device = device
        self.case_cursor = 0
        self.episode_counter = 0
        self.episode = None
        self.prefix = []
        self.weight = 1.0
        self.hidden = None
        self.ppo = BatchedPPO(policy, lr=3e-4)
        self.rollout = Rollout()
        self.N = 0
        self.train_success_episodes = 0
        self.episodes_finished = 0
        self.nan_events = 0

    def _new_episode(self):
        case = self.order[self.case_cursor % len(self.order)]
        self.case_cursor += 1
        self.episode_counter += 1
        self.episode = BwEpisode(case, episode_id="ep-%d" % self.episode_counter)
        self.prefix = []
        self.weight = 1.0
        self.hidden = self.policy.initial_hidden()

    def collect(self, n):
        gam = gamma(1.0)
        stats = Counter()
        self.policy.eval()
        with torch.no_grad():
            while len(self.rollout.transitions) < n:
                if self.episode is None or self.episode.done:
                    self._new_episode()
                snap = self.episode.snapshot()
                hidden = prefix_hidden(self.policy, self.prefix) if self.hidden is None else self.hidden
                out = self.policy(snap, hidden)
                if not torch.isfinite(out.logits[out.mask]).all() or not torch.isfinite(out.value):
                    self.nan_events += 1
                    raise RuntimeError("non-finite policy output")
                idx = int(out.distribution.sample())
                selected = snap.candidate_ids[idx]
                logp = float(out.distribution.log_prob(torch.tensor(idx, device=out.logits.device)))
                reward, terminated, truncated, reason = self.episode.step(selected)
                next_snap = self.episode.snapshot()
                next_v = 0.0 if terminated else float(self.policy(next_snap, out.hidden).value)
                self.rollout.append(Transition(snap, next_snap, selected, logp, float(out.value), next_v, reward, 1.0, self.weight, terminated, truncated, reason, tuple(self.prefix)))
                self.prefix.append(snap)
                self.weight *= gam
                self.hidden = out.hidden
                self.N += 1
                stats["transitions"] += 1
                if terminated or truncated:
                    self.episodes_finished += 1
                    stats["episodes"] += 1
                    if reason == "TASK_SUCCESS":
                        self.train_success_episodes += 1
                        stats["successes"] += 1
        return stats

    def update(self):
        self.policy.train()
        logs = self.ppo.update(self.rollout, epochs=4, minibatch=64, sequence_length=16)
        return logs


def run_training(cfg, log=print):
    """Full protocol of runbook 12.2 for one method/seed. Returns the accounting dict (also written to the run directory)."""
    from ..c1_blocksworld_policies import make_policy
    out = Path(cfg["out_dir"])
    (out / "checkpoints").mkdir(parents=True, exist_ok=False)
    device = torch.device(cfg.get("device", "cpu"))
    train_cases, doc = load_cases(cfg["train_dev_split"], "train")
    dev_cases, _ = load_cases(cfg["train_dev_split"], "dev")
    set_suite_half_life(float(doc["half_life"]))
    torch.manual_seed(int(cfg["seed"]))
    policy = make_policy(cfg["method"], device, seed=int(cfg["seed"]))
    trainer = Trainer(policy, train_cases, device, seed=int(cfg["seed"]))
    solver, hops = Solver(), Hops()
    eval_points = list(cfg["eval_points"])
    t0 = time.time()
    metrics_rows, eval_rows = [], []
    done_points = set()

    def do_eval(label):
        eps = evaluate(policy, dev_cases, solver, hops, record_decisions=False)
        s = summarize(eps)
        row = {"point": label, "N": trainer.N, "id_dev_success_n": s["success_n"], "id_dev_decision_perfect_n": s["decision_perfect_n"], "n": s["n"], "wall_seconds": round(time.time() - t0, 1)}
        eval_rows.append(row)
        ck = out / "checkpoints" / ("n_%07d.pt" % trainer.N)
        if not ck.exists():
            save_checkpoint(ck, policy, trainer.ppo.optimizer, {"N": trainer.N, "method": cfg["method"], "seed": cfg["seed"], "label": label})
        (out / ("eval_id_n_%07d.json" % trainer.N)).write_text(json.dumps({"label": label, "N": trainer.N, "summary": s, "episodes": eps}, indent=1, default=str), encoding="utf-8")
        with (out / "eval_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0]))
            w.writeheader()
            w.writerows(eval_rows)
        log("[bw-train] %s N=%d ID dev success %d/%d perfect %d/%d" % (cfg["method"], trainer.N, s["success_n"], s["n"], s["decision_perfect_n"], s["n"]))
        return s

    do_eval("step0")
    done_points.add(0)
    update = 0
    stop_reason = None
    update_seconds = []
    while True:
        if trainer.N >= cfg["Ncap"]:
            stop_reason = "Ncap"
            break
        if time.time() - t0 > cfg["Tcap_wall_seconds"] or update >= cfg["max_updates"]:
            stop_reason = "Tcap" if time.time() - t0 > cfg["Tcap_wall_seconds"] else "max_updates"
            break
        u0 = time.time()
        stats = trainer.collect(cfg["rollout_n"])
        t_collect = time.time() - u0
        logs = trainer.update()
        update += 1
        update_seconds.append(time.time() - u0)
        row = {"update": update, "N": trainer.N, "episodes": stats["episodes"], "rollout_successes": stats["successes"], "rollout_success_rate": (stats["successes"] / stats["episodes"]) if stats["episodes"] else None,
               "collect_seconds": round(t_collect, 1), "update_seconds": round(update_seconds[-1], 1), "wall_seconds": round(time.time() - t0, 1), "loss_total": logs[-1]["total"], "grad_norm": logs[-1]["pre_clip_global_grad_norm"],
               "optimizer_steps": trainer.ppo.optimizer_step_id}
        metrics_rows.append(row)
        with (out / "train_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0]))
            w.writeheader()
            w.writerows(metrics_rows)
        log("[bw-train] %s update %d N=%d eps=%d succ=%d wall=%.0fs" % (cfg["method"], update, trainer.N, stats["episodes"], stats["successes"], time.time() - t0))
        for p in eval_points:
            if p not in done_points and p > 0 and trainer.N >= p:
                do_eval("N%d" % p)
                done_points.add(p)
    final = do_eval("final")
    final_ck = out / "checkpoints" / "final.pt"
    save_checkpoint(final_ck, policy, trainer.ppo.optimizer, {"N": trainer.N, "method": cfg["method"], "seed": cfg["seed"], "label": "final", "stop_reason": stop_reason})
    eps = evaluate(policy, dev_cases, solver, hops, record_decisions=True)
    (out / "eval_id_final.json").write_text(json.dumps({"summary": summarize(eps), "episodes": eps}, indent=1, default=str), encoding="utf-8")
    acct = {"method": cfg["method"], "seed": cfg["seed"], "N": trainer.N, "updates": update, "stop_reason": stop_reason, "wall_seconds": round(time.time() - t0, 1), "optimizer_steps": trainer.ppo.optimizer_step_id,
            "nan_events": trainer.nan_events, "train_success_episodes": trainer.train_success_episodes, "episodes_finished": trainer.episodes_finished, "final_id_dev_success_n": final["success_n"],
            "final_id_dev_decision_perfect_n": final["decision_perfect_n"], "mean_update_seconds": (sum(update_seconds) / len(update_seconds)) if update_seconds else None,
            "final_checkpoint": str(final_ck), "hard_fail": None}
    (out / "training_accounting.json").write_text(json.dumps(acct, indent=2) + "\n", encoding="utf-8")
    return acct
