"""T1-S (plan 6.1): cross-parent supervision. D0's rank part compares, for every decision (parent state), all strictly ordered pairs of legal successors (mean over pairs, per-decision weight unchanged). S replaces a FIXED 25 % of the pairs of
every decision by cross-parent pairs of the SAME problem and keeps the other 75 % as same-parent pairs; the number of pairs per decision (the denominator) is not increased, the action-set NLL, the architecture, the initialisation, the data, the
batches and the weights are those of D0.

Per decision (case, parent state s), all deterministic and fixed for the whole run:
  pairs      : strictly ordered (better a, worse b) successor pairs, sorted by sha256(case|state|a|b); n pairs;
  replaced   : the last round(0.25 n) of them (rounding half up); kept: the first n - round(0.25 n);
  cross pair : for a replaced pair (a, b) the better end x = succ(a) stays; the worse end y is the state with the smallest sha256(case|state|a|b|y) among the labelled states of the SAME problem that belong to a DIFFERENT decision
               (parents and successors of other decisions of that problem, d* known from the Train96 labels; siblings and the parent of this decision are excluded) and have a strictly larger exact distance than x.
               If no such state exists the slot is MASKED (contributes nothing): S never shows more pairs than D0.
The loss of a decision is nll + (sum over kept pairs softplus(V(x)-V(y)) + sum over cross pairs softplus(V(x)-V(y))) / n.  With fraction 0 it equals D0's loss exactly (fixture)."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import torch
import torch.nn.functional as F

from ...blocksworld import goal_progress as GP
from ...blocksworld import imitation as I
from ...blocksworld.method_serial import trainer as BST
from ..task import codes_for_state_general
from ..train import PddlTrainer


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


class SPlan:
    def __init__(self, cases, labels, fraction=0.25):
        self.fraction = fraction
        self.by_key = {}
        self.stats = defaultdict(int)
        byc = {c.case_id: c for c in cases}
        pool = defaultdict(dict)                                  # case -> {state mask: (d, decision key that owns it)}  (a state may belong to several decisions)
        owners = defaultdict(lambda: defaultdict(set))
        dec = {}
        for key, dist in labels.items():
            cid, st = key.split("|")
            task = byc[cid].task
            parent = task.state_from_list([int(x) for x in st.split(",")]) if st else 0
            succ = {aid: task.apply(parent, task.action_by_id[aid]) for aid in dist}
            dec[key] = (cid, parent, succ, dist)
            dmin = min(dist.values())
            for s, d in [(parent, dmin + 1)] + [(succ[a], dist[a]) for a in dist]:
                pool[cid][s] = d
                owners[cid][s].add(key)
        self.pool, self.owners = pool, owners
        for key, (cid, parent, succ, dist) in dec.items():
            pairs = sorted(((a, b) for a in dist for b in dist if dist[a] < dist[b]), key=lambda ab: sha("%s|%s|%s|%s" % (cid, key, ab[0], ab[1])))
            n = len(pairs)
            n_cross = int(self.fraction * n + 0.5)
            keep, repl = pairs[:n - n_cross], pairs[n - n_cross:]
            own = {parent} | set(succ.values())
            cross = []
            for a, b in repl:
                dx = dist[a]
                cands = [y for y, d in pool[cid].items() if d > dx and y not in own]              # a labelled state of another decision (parents and successors of other decisions), strictly farther from the goal
                if cands:
                    y = min(cands, key=lambda y: sha("%s|%s|%s|%s|%x" % (cid, key, a, b, y)))
                    cross.append((a, y))
                else:
                    cross.append((a, None))
            self.by_key[key] = {"n": n, "keep": keep, "cross": cross, "cid": cid}
            self.stats["decisions"] += 1
            self.stats["pairs"] += n
            self.stats["replaced_slots"] += n_cross
            self.stats["cross_constructed"] += sum(1 for _a, y in cross if y is not None)
            self.stats["masked_slots"] += sum(1 for _a, y in cross if y is None)
        self.extra_cache = {}

    def summary(self):
        s = dict(self.stats)
        s["fraction"] = self.fraction
        s["effective_pair_fraction_of_D0"] = (s["pairs"] - s["masked_slots"]) / s["pairs"] if s.get("pairs") else None
        return s


class PddlTrainerS(PddlTrainer):
    """PddlTrainer with the S rank part. Everything else (batches, weights, clipping, optimiser) is inherited unchanged."""

    def __init__(self, model, cases, rank_labels, device, plan, lr=3e-4, chunk=48):
        super().__init__(model, cases, rank_labels, device, lr=lr, chunk=chunk)
        self.plan = plan
        self.cross_used = 0
        self._codes_cache = {}

    def _extra_codes(self, st, task, key):
        got = self._codes_cache.get(key)
        if got is None:
            info = self.plan.by_key[key]
            ys = [y for _a, y in info["cross"] if y is not None]
            static_true = {"p:" + p + (":" + ":".join(a) if a else "") for p, a in task.static}
            got = torch.stack([codes_for_state_general(st, task, y, static_true) for y in ys]) if ys else None
            self._codes_cache[key] = got
        return got

    def group_loss_s(self, sub, astar, keys):
        m = self.model
        n = len(sub[0].candidate_ids)
        ids = sub[0].candidate_ids
        dev = self.device
        task = self.cases[keys[0].split("|")[0]].task
        st, _g = m.mg.goal_free_static(sub[0].template)
        legal_m = torch.tensor([s.mask for s in sub], device=dev, dtype=torch.bool)
        am = torch.zeros((len(sub), n), dtype=torch.bool, device=dev)
        for j in range(len(sub)):
            for a in astar[j]:
                am[j, ids.index(a)] = True
        extra = [self._extra_codes(st, task, k) for k in keys]
        res = m.forward_group(sub, extra=extra)[None]
        scores = res["logits"]
        nll = I.imitation_nll(scores, am)
        vn, vx = res["vn"], res["vx"]
        ranks = []
        used = 0
        for j, k in enumerate(keys):
            info = self.plan.by_key[k]
            terms = []
            for a, b in info["keep"]:
                terms.append(F.softplus(vn[j, ids.index(a)] - vn[j, ids.index(b)]))
            xi = 0
            for a, y in info["cross"]:
                if y is None:
                    continue
                terms.append(F.softplus(vn[j, ids.index(a)] - vx[j][xi]))
                xi += 1
                used += 1
            ranks.append(torch.stack(terms).sum() / info["n"] if terms else torch.zeros((), device=dev))
        rank = torch.stack(ranks)
        self.cross_used += used
        return nll + rank, nll, rank, scores.argmax(-1)

    def step(self, trajs):
        """``SerialTrainer.step`` with the S group loss (copy of the control flow; batching / weights / clipping identical)."""
        m = self.model
        m.train()
        per = [self.snapshots(t) for t in trajs]
        snaps = [s for p in per for s in p]
        astar = [a for t in trajs for a in t["astar"]]
        keys = [GP.rank_key(t["case_id"], s) for t in trajs for s in t["states"][:-1]]
        w = torch.tensor([x for ws in I.decision_weights(trajs) for x in ws], device=self.device)
        self.optimizer.zero_grad()
        tot = {"loss": 0.0, "nll": 0.0, "rank": 0.0, "cal": 0.0, "hit": 0.0}
        order = {}
        for i, sn in enumerate(snaps):
            order.setdefault(id(sn.template), []).append(i)
        self.cross_used = 0
        for idx in order.values():
            for a in range(0, len(idx), self.chunk):
                sel = idx[a:a + self.chunk]
                sub = [snaps[i] for i in sel]
                loss, nl, rk, top = self.group_loss_s(sub, [astar[i] for i in sel], [keys[i] for i in sel])
                ws = w[sel]
                part = (ws * loss).sum()
                if not torch.isfinite(part):
                    self.nan_events += 1
                    raise I.ImitationError("non-finite loss")
                part.backward()
                tot["loss"] += float(part)
                tot["nll"] += float((ws * nl).sum())
                tot["rank"] += float((ws * rk).sum())
                am_hit = torch.zeros((len(sub), len(sub[0].candidate_ids)), dtype=torch.bool, device=self.device)
                for j, i in enumerate(sel):
                    for aid in astar[i]:
                        am_hit[j, sub[0].candidate_ids.index(aid)] = True
                tot["hit"] += float((ws * am_hit.gather(1, top.unsqueeze(1)).squeeze(1).float()).sum())
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, BST.GRAD_CLIP, error_if_nonfinite=True))
        new_g = float(sum(float(p.grad.norm() ** 2) for p in self.model.new_params() if p.grad is not None) ** 0.5) if self.model.new_params() else 0.0
        self.optimizer.step()
        self.steps += 1
        return {**tot, "grad_norm": norm, "new_module_grad_norm": new_g, "decisions": len(snaps), "cross_pairs_used": self.cross_used}
