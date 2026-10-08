"""Domain-independent relation-restricted goal scoring for grounded STRIPS tasks.

``GeneralGoalAttention`` keeps the parameters / initialisation / attention of the Blocksworld ``GoalAttention`` (4 heads, 128 channels, zero output projection) but derives its pair features and its relation
mask only from the public task description:

    M_ij(s) = [ i = j  or  share(g_i, g_j)  or  direct(s, g_i, g_j)  or  threat(g_i, g_j) ]
      share   : the two goal atoms have a common argument OBJECT (argument identity, not a name)
      direct  : some CURRENTLY TRUE binary atom that can change (it occurs in an ADD / DELETE list of a grounded action) links an object of g_i with an object of g_j
      threat  : some grounded action ADDs g_i and DELETEs g_j (or the reverse)
Pair features: same predicate, same leaf-type signature, same arity, truth of both goals, share, direct, threat, self. No On / OnTable or other domain-specific category is used. The dense variant uses the
identical features and parameters and lets every goal read every goal. With ``mode='rel'`` on a Blocksworld template the mask equals the Blocksworld mask (fixture).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from ..blocksworld.method_serial.model import NEW_SEED, GoalAttention, SerialModel, SerialPolicy
from .task import SCHEMA_ORDER, codes_for_state_general

PREDICATES = ("at", "available", "clear", "in", "lifting", "on")
TYPES = ("crate", "hoist", "pallet", "place", "truck")


class GeneralGoalAttention(GoalAttention):
    def _goal_static(self, st, gprop, template):
        hit = self._gs.get(id(st))
        if hit is not None and hit[0] is st:
            return hit[1]
        dev = st.prop_pos.device
        node = {n.id: n for n in template.nodes}
        goals = [node[g.fact_id] for g in template.goals]
        ng = len(goals)
        objs = [set(n.arguments) for n in goals]
        mat = lambda f: torch.tensor([[1.0 if f(i, j) else 0.0 for j in range(ng)] for i in range(ng)], device=dev)
        same_pred = mat(lambda i, j: goals[i].schema == goals[j].schema)
        same_sig = mat(lambda i, j: goals[i].argument_types == goals[j].argument_types)
        same_arity = mat(lambda i, j: len(goals[i].arguments) == len(goals[j].arguments))
        share = mat(lambda i, j: i != j and bool(objs[i] & objs[j]))
        gids = [g.fact_id for g in template.goals]
        adds = [{a.id for a in c.effects.add} for c in template.contracts]
        dels = [{a.id for a in c.effects.delete} for c in template.contracts]
        threat = torch.zeros((ng, ng), device=dev)
        for ad, de in zip(adds, dels):
            ai = [i for i, g in enumerate(gids) if g in ad]
            dj = [j for j, g in enumerate(gids) if g in de]
            for i in ai:
                for j in dj:
                    if i != j:
                        threat[i, j] = threat[j, i] = 1.0
        changing = set().union(*adds, *dels) if adds else set()
        on_idx, mats = [], []
        for pi, pid in enumerate(st.prop_ids):
            n = node[pid]
            if len(n.arguments) == 2 and pid in changing:
                a, b = n.arguments
                m = torch.zeros((ng, ng), device=dev)
                for i in range(ng):
                    for j in range(ng):
                        if i != j and ((a in objs[i] and b in objs[j]) or (a in objs[j] and b in objs[i])):
                            m[i, j] = 1.0
                on_idx.append(pi)
                mats.append(m.reshape(-1))
        M = torch.stack(mats) if mats else torch.zeros((0, ng * ng), device=dev)
        out = SimpleNamespace(ng=ng, same_pred=same_pred, same_sig=same_sig, same_arity=same_arity, share=share, threat=threat, on_idx=torch.tensor(on_idx, device=dev, dtype=torch.long), M=M, eye=torch.eye(ng, device=dev))
        self._gs[id(st)] = (st, out)
        return out

    def pair_features(self, gs, codes, gprop):
        G = codes.shape[0]
        ng = gs.ng
        truth = codes[:, gprop, 0]
        if len(gs.on_idx):
            conn = ((codes[:, gs.on_idx, 0] @ gs.M) > 0).reshape(G, ng, ng).to(codes.dtype)
        else:
            conn = codes.new_zeros((G, ng, ng))
        e = lambda t: t.unsqueeze(0).expand(G, -1, -1)
        feats = torch.stack((e(gs.same_pred), e(gs.same_sig), e(gs.same_arity), truth.unsqueeze(2).expand(G, ng, ng), truth.unsqueeze(1).expand(G, ng, ng), e(gs.share), conn, e(gs.threat), e(gs.eye)), dim=-1)
        allowed = (gs.eye.unsqueeze(0) + gs.share.unsqueeze(0) + conn + gs.threat.unsqueeze(0)) > 0
        if self.mode == "rand":
            allowed = self.matched_random_mask(allowed, codes, gs)
        return feats, allowed


class PddlSerialModel(SerialModel):
    """MG-G (mode 'mg'), DENSE-G ('dense') or REL-G ('rel') on top of one common base (graph encoder + phi / rho heads, identical initial tensors)."""

    def __init__(self, mg, mode, seed=NEW_SEED):
        assert mode in ("mg", "dense", "rel")
        super().__init__(mg, "MG", seed)
        self.mode = mode
        if mode != "mg":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.attn = GeneralGoalAttention(mode)
            self.kind = "GOAL_DENSE" if mode == "dense" else "GOAL_REL"
            self.to(next(mg.parameters()).device)

    def new_params(self):
        return list(self.attn.parameters()) if self.attn is not None else []

    EDGE_BUDGET = 6_000_000                                                        # graphs x template edges per encoder call (memory only; every graph is encoded independently)

    def zv(self, st, gprop, codes, steps=(None,), template=None):
        """Identical to ``SerialModel.zv`` but splits a batch of very large template graphs into encoder calls of bounded size (exact: graphs do not interact)."""
        ch = max(1, self.EDGE_BUDGET // max(int(st.ei.shape[1]), 1))
        if codes.shape[0] <= ch:
            return super().zv(st, gprop, codes, steps, template)
        parts = [super(PddlSerialModel, self).zv(st, gprop, codes[a:a + ch], steps, template) for a in range(0, codes.shape[0], ch)]
        return {t: tuple(torch.cat([p[t][k] for p in parts], 0) for k in range(3)) for t in parts[0]}

    @torch.no_grad()
    def state_values(self, template, task, states, steps=None, chunk=192):
        """V of arbitrary state bit masks of ``task`` (look-ahead leaves); same state function as the nominal-successor path."""
        st, gprop = self.mg.goal_free_static(template)
        static_true = {"p:" + p + (":" + ":".join(a) if a else "") for p, a in task.static}
        out = []
        for a in range(0, len(states), chunk):
            codes = torch.stack([codes_for_state_general(st, task, s, static_true) for s in states[a:a + chunk]])
            out.append(self.zv(st, gprop, codes, (steps,), template)[steps][1])
        return torch.cat(out) if out else torch.zeros(0)


def _policy_class():
    from .. import neural
    from ..c1_blocksworld_policies import B2CachedPolicy, TemplateStatic

    class PddlB2Policy(B2CachedPolicy):
        """B2CachedPolicy whose per-template tensors allow any arity (the Blocksworld version stores at most two argument roles). Identical arithmetic otherwise."""

        def _static(self, template):
            cache = self.__dict__.setdefault("_static_cache", {})
            hit = cache.get(id(template))
            if hit is not None and hit.template is template:
                return hit
            dev = next(self.parameters()).device
            feats = self.encoder.features
            nodes, ids = template.nodes, template.node_ids
            N = len(nodes)
            R = max([len(n.argument_types) for n in nodes] + [1])
            kind_idx, action_idx, pred_idx, is_action = [], [], [], []
            arg_idx = torch.zeros((N, R), dtype=torch.long)
            arg_w = torch.zeros((N, R))
            for i, n in enumerate(nodes):
                act = n.kind == "ACTION"
                kind_idx.append(0 if act else 1)
                action_idx.append(feats.actions[n.schema] if act else 0)
                pred_idx.append(0 if act else feats.predicates[n.schema])
                is_action.append(act)
                for role, t in enumerate(n.argument_types):
                    arg_idx[i, role] = feats.types[t]
                    arg_w[i, role] = 1.0 / (role + 1)
            prop_pos = [i for i, n in enumerate(nodes) if n.kind == "PROPOSITION"]
            action_pos = [i for i, n in enumerate(nodes) if n.kind == "ACTION"]
            signs = {g.fact_id: float(g.sign) for g in template.goals}
            prop_ids = tuple(ids[i] for i in prop_pos)
            prop_index = {p: j for j, p in enumerate(prop_ids)}
            pos = {v: i for i, v in enumerate(ids)}
            edges = template.edges
            ei = torch.tensor([[pos[a] for a, b, r in edges], [pos[b] for a, b, r in edges]], dtype=torch.long).reshape(2, -1)
            et = torch.tensor([neural.RELATIONS.index(r) for a, b, r in edges], dtype=torch.long)
            action_ids = tuple(ids[i] for i in action_pos)
            add_mask = torch.zeros((len(action_ids), len(prop_ids)), dtype=torch.bool)
            del_mask = torch.zeros((len(action_ids), len(prop_ids)), dtype=torch.bool)
            for r, c in enumerate(sorted(template.contracts, key=lambda c: c.id)):
                assert c.id == action_ids[r]
                for a in c.effects.add:
                    add_mask[r, prop_index[a.id]] = True
                for a in c.effects.delete:
                    del_mask[r, prop_index[a.id]] = True
            st = TemplateStatic(template=template, N=N, kind_idx=torch.tensor(kind_idx, device=dev), action_idx=torch.tensor(action_idx, device=dev), pred_idx=torch.tensor(pred_idx, device=dev),
                                is_action=torch.tensor(is_action, device=dev), arg_idx=arg_idx.to(dev), arg_w=arg_w.to(dev), prop_pos=torch.tensor(prop_pos, device=dev), action_pos=torch.tensor(action_pos, device=dev),
                                goal_pos=torch.tensor([ids.index(g.fact_id) for g in template.goals], device=dev), goal_sign=torch.tensor([[float(g.sign)] for g in template.goals], device=dev),
                                prop_goal_sign=torch.tensor([[signs.get(p, 0.0)] for p in prop_ids], device=dev), ei=ei.to(dev), et=et.to(dev), action_ids=action_ids,
                                action_row={a: r for r, a in enumerate(action_ids)}, add_mask=add_mask.to(dev), del_mask=del_mask.to(dev), prop_ids=prop_ids, prop_index=prop_index)
            cache[id(template)] = st
            return st
    return PddlB2Policy


def make_base(device, seed=20261008):
    """Fresh graph encoder with the Depots vocabulary (no Blocksworld weights); identical for every method that uses the same seed."""
    cls = _policy_class()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = cls(SCHEMA_ORDER, PREDICATES, TYPES, 48, 8)
    return base.to(device)


def make_model(mode, device, seed=20261008):
    from ..blocksworld import gp_attribution as A
    base = make_base(device, seed)
    mg = A.GoalProgressModelGoal(base, "global", "production", seed=seed).to(device)
    return PddlSerialModel(mg, mode)
