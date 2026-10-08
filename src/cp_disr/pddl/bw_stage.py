"""Blocksworld zero-training wrap-up of the public-Depots card (plan sections 5 and 6): selective (gated) look-ahead, fixed two-step with an h_add leaf, and the two invariance probes.

No weight is trained or changed. The scorers are the frozen M1-GOAL final model (MG) and the GOAL_REL checkpoint selected by the method-serial v3 card; thresholds of the gate come from TRAINING states only
(10th percentile of the top-2 logit margin over unique training decisions with >= 2 legal actions) and are frozen before any evaluation episode.

The choosers below run inside the frozen Blocksworld episode loop (``gp_attribution.run_episode_with`` / ``a04p_registry.run_cases``) so that their records are directly comparable with the
LOOK2_MG / REL_LOOK2 rows of the earlier cards; the fixed-tree code (``lookahead.choose_root``) is reused unchanged.
"""
from __future__ import annotations

import dataclasses
import json
import math
import statistics
from pathlib import Path

import torch

from ..blocksworld import a04p_controls as AC
from ..blocksworld import a04p_registry as R
from ..blocksworld import contracts as BC
from ..blocksworld import gp_attribution as A
from ..blocksworld import imitation as I
from ..blocksworld import planner as P
from ..blocksworld import state as S
from ..blocksworld.environment import BwEpisode, Case
from ..blocksworld.method_serial import lookahead as LA
from ..blocksworld.method_serial.model import SerialPolicy


class BwOps:
    """Adapter that lets the generic (grounded-PDDL) controllers of ``control.py`` run on Blocksworld; used by the consistency fixtures only."""

    def __init__(self, problem, model=None):
        self.problem, self.model = problem, model
        self.action_of = {S.action_id(problem.names, a): a for a in BC.all_actions(problem.n)}

    def legal_ids(self, state):
        return [S.action_id(self.problem.names, a) for a in S.legal_actions(state)]

    def apply_id(self, state, aid):
        return S.apply(state, self.action_of[aid])

    def is_goal(self, state):
        return S.goal_satisfied(self.problem, state)

    def unmet(self, state):
        return sum(1 for x, b in enumerate(self.problem.goal) if state[x] != b)

    def leaf_values(self, template, states, steps=None):
        return self.model.state_values(template, self.problem, states, steps)


class HAdd:
    """Additive delete-relaxation heuristic of the grounded Blocksworld contracts (unit cost) to the complete goal configuration. Leaf score of the h_add control (lower is better)."""

    def __init__(self, problem):
        self.problem = problem
        cs = BC.grounded_contracts(problem)
        self.actions = [([a.id for a in c.pre_pos], [a.id for a in c.effects.add]) for c in cs]
        self.goal = list(S.goal_atoms(problem))
        self.cache = {}

    def __call__(self, state):
        hit = self.cache.get(state)
        if hit is not None:
            return hit
        cost = {k: 0 for k, v in S.fact_values(self.problem, state).items() if v}
        changed = True
        while changed:
            changed = False
            for pre, add in self.actions:
                if all(p in cost for p in pre):
                    c = 1 + sum(cost[p] for p in pre)
                    for a in add:
                        if a not in cost or c < cost[a]:
                            cost[a] = c
                            changed = True
        h = sum(cost.get(g, 10 ** 6) for g in self.goal)
        self.cache[state] = h
        return h


class HAddModel:
    """Duck-typed leaf scorer for ``lookahead.choose_root`` (``state_values``): same tree, same terminal handling, same C3 and tie rules; only the leaf score differs."""

    def __init__(self):
        self._h = {}

    def state_values(self, template, problem, states, steps=None):
        key = (problem.names, problem.colors, problem.goal)
        h = self._h.get(key)
        if h is None:
            h = self._h[key] = HAdd(problem)
        return torch.tensor([float(h(s)) for s in states])


class GateCounters(LA.Counters):
    def reset(self):
        super().reset()
        self.total_decisions = self.gate_checks = self.gate_expansions = self.trap_unfolds = 0


class BwChooser:
    """Pipeline chooser with logging of the three selections (raw / C3 / final). mode 'fixed': expand every decision with a non-empty C3 set (identical to ``lookahead.make_chooser``);
    mode 'gated': expand only if |A_C| >= 2, the first choice does not reach the goal, and (top-2 margin <= tau or the first choice leads into a locally visible dead end)."""

    def __init__(self, leaf_model, mode, counters, tau=None):
        assert mode in ("fixed", "gated")
        self.leaf_model, self.mode, self.cnt, self.tau = leaf_model, mode, counters, tau
        self.log = []

    def __call__(self, _c, ep, snap, logits, mem):
        ids, problem, state = snap.candidate_ids, ep.problem, ep.state
        legal = [i for i, m in enumerate(snap.mask) if m]
        raw = AC.argmax_tie(logits, ids, legal)
        rec = {"raw": ids[raw], "c3": None, "final": None, "gate": None, "margin": None, "available": 0, "expanded": False}
        self.log.append(rec)
        info = {"raw": ids[raw], "intervened": False, "trigger": None}
        avail = [i for i in legal if S.apply(state, ep._action_of[ids[i]]) not in mem.visited]
        rec["available"] = len(avail)
        if not avail:
            rec["gate"] = "no_root"
            info["trigger"] = "NO_UNVISITED_SUCCESSOR"
            return None, info
        self.cnt.total_decisions += 1
        first = AC.argmax_tie(logits, ids, avail)
        rec["c3"] = ids[first]
        expand = True
        if self.mode == "gated":
            self.cnt.gate_checks += 1
            s1 = S.apply(state, ep._action_of[ids[first]])
            if len(avail) == 1:
                rec["gate"], expand = "single_root", False
            elif S.goal_satisfied(problem, s1):
                rec["gate"], expand = "goal_next", False
            else:
                ordered = sorted(avail, key=lambda i: (-float(logits[i]), ids[i]))
                margin = float(logits[ordered[0]] - logits[ordered[1]])
                rec["margin"] = margin
                ancestors = mem.visited | {state, s1}
                succ = [S.apply(s1, b) for b in S.legal_actions(s1)]
                self.cnt.trap_unfolds += len(succ)
                trap = all(t in ancestors for t in succ)
                small = self.tau is not None and margin <= self.tau
                expand = small or trap
                rec["gate"] = "both" if (small and trap) else "low_margin" if small else "local_trap" if trap else "keep_one_step"
        if not expand:
            sel = first
            info.update({"intervened": sel != raw, "trigger": "VISITED_SUCCESSOR" if sel != raw else None})
        else:
            rec["expanded"] = True
            if self.mode == "gated":
                self.cnt.gate_expansions += 1
            aid, _linfo = LA.choose_root(self.leaf_model, "MG", snap.template, problem, state, [ids[i] for i in avail], ep._action_of, mem.visited, self.cnt, None)
            if aid is None:
                self.cnt.fallback_decisions += 1
                sel = first
                info.update({"intervened": sel != raw, "trigger": "LOOKAHEAD_EMPTY_TREE_C3" if sel != raw else None})
            else:
                sel = ids.index(aid)
                info.update({"intervened": sel != raw, "trigger": "LOOKAHEAD" if sel != raw else None})
        rec["final"] = ids[sel]
        return sel, info


def run_condition(rr, name, scorer, chooser, groups, out_path, solver=None):
    """Ledgered, resumable evaluation of one condition on grouped cases with the frozen episode loop."""
    solver = solver or P.Solver()
    policy = SerialPolicy(scorer)

    def ev(case):
        chooser.log = []
        before = dict(chooser.cnt.as_dict())
        ep = A.run_episode_with(policy, case, solver, chooser)
        after = chooser.cnt.as_dict()
        ep["pipeline"] = chooser.log
        ep["counters"] = {k: (after[k] - before[k]) for k in before}
        for d in ep["decisions"]:
            d.pop("state", None)
        return ep
    return R.run_cases(rr, "%s@confirm" % name, groups, ev, out_path)


def training_margins(model, cases, d_train):
    """Top-2 logit margins over the unique (case, state) training decisions with >= 2 legal actions."""
    by_id = {c.case_id: c for c in cases}
    seen, margins = set(), []
    for t in d_train:
        for s in t["states"][:-1]:
            key = (t["case_id"], tuple(s))
            if key in seen:
                continue
            seen.add(key)
            sn = I.snapshot_at(by_id[t["case_id"]], tuple(s), 0)
            legal = [i for i, m in enumerate(sn.mask) if m]
            if len(legal) < 2:
                continue
            lg = sorted((float(x) for x in model.eval_logits(sn)[legal]), reverse=True)
            margins.append(lg[0] - lg[1])
    return margins


def percentile(xs, q):
    ys = sorted(xs)
    k = (len(ys) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


# ------------------------------------------------------------------------------------------------ probes
@torch.no_grad()
def probe_module_isolation(model, bank, cases, n_extra=3, seed=0):
    """Append goal vectors taken from OTHER states; under REL the old queries cannot read them (mask), under DENSE they can. Reports the change of the attention output of the OLD goals."""
    import random
    rng = random.Random(seed)
    items = []
    for r in bank:
        sn = I.snapshot_at(cases[r["case_id"]], tuple(r["state"]), 0)
        st, gprop, codes, _info, _n = model.layout([sn])
        _ro, H = model.mg.base._encode(st, codes[:1])
        z = model.mg.heads.phi(model._features(st, gprop, codes[:1], H))                       # goal vectors BEFORE the attention layer
        gs = model.attn._goal_static(st, gprop, sn.template)
        feats, allowed = model.attn.pair_features(gs, codes[:1], gprop)
        items.append((z, feats, allowed, model.attn(z, feats, allowed)))
    res = []
    for z, feats, allowed, base in items:
        other = items[rng.randrange(len(items))][0][0]
        extra = other[:n_extra]
        m, ng = extra.shape[0], z.shape[1]
        z2 = torch.cat((z, extra.unsqueeze(0)), 1)
        f2 = torch.zeros((1, ng + m, ng + m, feats.shape[-1]), device=z.device)
        f2[:, :ng, :ng] = feats
        a2 = torch.ones((1, ng + m, ng + m), dtype=torch.bool, device=z.device)
        a2[:, :ng, :ng] = allowed
        if model.attn.relational:
            a2[:, :ng, ng:] = False                                                           # old queries cannot read the appended goals
        out2 = model.attn(z2, f2, a2)[:, :ng]
        res.append(float((out2 - base).abs().max()))
    return {"states": len(res), "mean_max_abs_change": statistics.mean(res), "max_max_abs_change": max(res), "fraction_exactly_unchanged": sum(1 for x in res if x == 0.0) / len(res)}


def _template_with_goals(problem, goal_problem):
    from ..graph import Goal, build_template
    goals = tuple(Goal(a, 1) for a in S.goal_atoms(goal_problem))
    return build_template(BC.grounded_contracts(problem), goals, BC.PREDICATE_TYPES, BC.object_types(problem), BC.extra_atoms(problem))


def augmented_snapshot(case, state, variant):
    """variant: 'obj' (+1 block on the table, NO goal mark for it), 'obj_goal' (+1 block on the table with its satisfied OnTable goal), 'tower' (+2-block tower with satisfied goals).
    The appended material is physically separate from the original blocks."""
    p = case.problem
    n = p.n
    m = 1 if variant in ("obj", "obj_goal") else 2
    names = S.default_names(n + m)
    colors = tuple(p.colors) + tuple((i % 2) for i in range(m))
    new_state = tuple(state) + ((S.TABLE,) if m == 1 else (S.TABLE, n))
    new_goal = tuple(p.goal) + ((S.TABLE,) if m == 1 else (S.TABLE, n))
    pa = S.Problem(names, colors, new_state, new_goal)
    ca = Case(case.case_id + "+" + variant, "aug", pa, case.optimal_length, case.step_cap)
    ep = BwEpisode(ca)
    ep.state = new_state
    sn = ep.snapshot()
    if variant == "obj":
        sn = dataclasses.replace(sn, template=_template_with_goals(pa, p))
    return sn, pa


@torch.no_grad()
def probe_augmentation(model, bank, cases):
    """Scores of the ORIGINAL legal actions before / after appending physically separate, satisfied material; centred per state (no softmax over the enlarged candidate set)."""
    out = {}
    base = []
    for r in bank:
        sn = I.snapshot_at(cases[r["case_id"]], tuple(r["state"]), 0)
        base.append((sn, model.eval_logits(sn)))
    for variant in ("obj", "obj_goal", "tower"):
        dmax, same_top, rho, zrel, span = [], 0, [], [], []
        for (sn, lg0), r in zip(base, bank):
            case = cases[r["case_id"]]
            sa, _pa = augmented_snapshot(case, tuple(r["state"]), variant)
            lg1 = model.eval_logits(sa)
            ids0, ids1 = sn.candidate_ids, sa.candidate_ids
            pos1 = {a: i for i, a in enumerate(ids1)}
            leg = [i for i, ok in enumerate(sn.mask) if ok and sa.mask[pos1[ids0[i]]]]
            a0 = torch.stack([lg0[i] for i in leg])
            a1 = torch.stack([lg1[pos1[ids0[i]]] for i in leg])
            c0, c1 = a0 - a0.mean(), a1 - a1.mean()
            dmax.append(float((c0 - c1).abs().max()))
            span.append(dmax[-1] / max(float(c0.max() - c0.min()), 1e-9))
            same_top += int(int(a0.argmax()) == int(a1.argmax()))
            if len(leg) >= 3 and float(c0.std()) > 0 and float(c1.std()) > 0:
                rho.append(float(torch.corrcoef(torch.stack((a0.argsort().argsort().float(), a1.argsort().argsort().float())))[0, 1]))
            st0, gp0, cd0, _i0, _n0 = model.layout([sn])
            st1, gp1, cd1, _i1, _n1 = model.layout([sa])
            z0 = model.zv(st0, gp0, cd0[:1], (None,), sn.template)[None][0][0]
            z1 = model.zv(st1, gp1, cd1[:1], (None,), sa.template)[None][0][0]
            g0 = [g.fact_id for g in sn.template.goals]
            g1 = {g.fact_id: i for i, g in enumerate(sa.template.goals)}
            rel = [float((z1[g1[g]] - z0[i]).norm() / max(float(z0[i].norm()), 1e-9)) for i, g in enumerate(g0)]
            zrel.append(sum(rel) / len(rel))
        out[variant] = {"states": len(dmax), "mean_max_abs_centered_score_change": statistics.mean(dmax), "max_over_states": max(dmax), "mean_change_over_original_score_range": statistics.mean(span), "top1_unchanged": same_top / len(dmax),
                        "mean_spearman_rank_corr": statistics.mean(rho) if rho else None, "mean_relative_change_of_old_goal_vectors": statistics.mean(zrel)}
    return out
