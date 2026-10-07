"""Matched trainer of the method suite: GPTrainer's batching / weighting / shuffling with a per-kind loss. Recurrent kinds read two supervised iteration counts (2 and 4)."""
from __future__ import annotations

import random

import torch

from .. import goal_progress as GP
from .. import imitation as I
from .. import state as S
from .heads_math import calibration_objectives, calibrated_logits, loss_terms_v3

EXISTING_LR, NEW_LR, ETA_LR = 1e-4, 3e-4, 1e-4
GRAD_CLIP, BATCH, CHUNK = 0.5, 32, 192
REC_CHUNK = 48
ETA_BOUND = 9.0
LAMBDA_CAL = 0.1


def _weighted_median(vals, weights):
    pairs = sorted(zip(vals, weights))
    half = sum(w for _v, w in pairs) / 2.0
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= half:
            return v
    return pairs[-1][0]


@torch.no_grad()
def calibration_constants(model, cases, rank_labels):
    """a0, c0 from the ORIGINAL training states only (no test label): median positive margin V(worse) - V(better) over strictly ordered candidate pairs -> a0 = clip(2/m);
    c0 = weighted median over training states of d*(t) - a0 V0(t). Common to CAL-ABS and CAL-REL."""
    import statistics
    by_id = {c.case_id: c for c in cases}
    margins, pts = [], []
    model.eval()
    for key, dist in rank_labels.items():
        cid, st = key.split("|")
        sn = I.snapshot_at(by_id[cid], tuple(int(x) for x in st.split(",")), 0)
        out = model.forward_group([sn])[None]
        ids = sn.candidate_ids
        legal = [j for j, m in enumerate(sn.mask) if m]
        vn = {ids[j]: float(out["vn"][0, j]) for j in legal}
        v0 = float(out["v0"][0])
        d0 = min(dist.values()) + 1
        for a in vn:
            for b in vn:
                if dist[a] < dist[b] and vn[b] - vn[a] > 0:
                    margins.append(vn[b] - vn[a])
        states = [(v0, d0)] + [(vn[a], dist[a]) for a in vn]
        pts += [(d - 0.0, v, 1.0 / len(states)) for v, d in states]
    fallback = len(margins) == 0
    m = 2.0 if fallback else statistics.median(margins)
    a0 = min(max(2.0 / m, 1e-3), 1e3)
    c0 = _weighted_median([d - a0 * v for d, v, _w in pts], [w for _d, _v, w in pts])
    return {"a0": a0, "c0": c0, "median_positive_margin": m, "n_margins": len(margins), "fallback_m_equals_2": fallback, "beta0": 1.0 / a0, "n_states": len(pts)}


class SerialTrainer:
    def __init__(self, model, kind_loss, cases, rank_labels, device, event_labels=None, cal_targets=None, cal_scale=1.0):
        self.model, self.loss_kind, self.device = model, kind_loss, device        # loss_kind: BASE FACT JOINT ABS REL ACTION REC
        self.cases = {c.case_id: c for c in cases}
        self.rank_labels, self.event_labels = rank_labels, event_labels
        self.cal_targets, self.cal_scale = cal_targets, cal_scale
        groups = [{"params": model.encoder_params() + model.existing_head_params(), "lr": EXISTING_LR}]
        if model.new_params():
            groups.append({"params": model.new_params(), "lr": NEW_LR})
        if model.eta_params():
            groups.append({"params": model.eta_params(), "lr": ETA_LR})
        self.optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.params = [p for g in groups for p in g["params"]]
        self.steps, self.nan_events, self.eta_clamped = 0, 0, 0
        self._snaps, self._extra = {}, {}
        self.chunk = REC_CHUNK if kind_loss == "REC" else CHUNK
        self.trace = []

    def snapshots(self, traj):
        got = self._snaps.get(traj["tid"])
        if got is None:
            c = self.cases[traj["case_id"]]
            got = [I.snapshot_at(c, s, t) for t, s in enumerate(traj["states"][:-1])]
            self._snaps[traj["tid"]] = got
        return got

    def extra_codes(self, traj, snaps):
        got = self._extra.get(traj["tid"])
        if got is None:
            c = self.cases[traj["case_id"]]
            m = self.model
            st, _g = m.mg.goal_free_static(snaps[0].template)
            from .model import codes_for_state
            got = []
            for i in range(len(snaps)):
                eps = self.cal_targets["%s|%d" % (traj["tid"], i)]["endpoints"]
                got.append(torch.stack([codes_for_state(st, c.problem, tuple(e["state"])) for e in eps]) if eps else None)
            self._extra[traj["tid"]] = got
        return got

    # ------------------------------------------------------------------ losses on one same-template group
    def group_loss(self, sub, astar, keys, extra_codes, d_extra):
        m = self.model
        n = len(sub[0].candidate_ids)
        ids = sub[0].candidate_ids
        dev = self.device
        legal_m = torch.tensor([s.mask for s in sub], device=dev, dtype=torch.bool)
        am = torch.zeros((len(sub), n), dtype=torch.bool, device=dev)
        dist = torch.zeros((len(sub), n), device=dev)
        for j in range(len(sub)):
            for a in astar[j]:
                am[j, ids.index(a)] = True
            for a, d in self.rank_labels[keys[j]].items():
                dist[j, ids.index(a)] = d
        k = self.loss_kind
        zero = torch.zeros((), device=dev)
        if k == "REC":
            res = m.forward_group(sub, steps=(2, 4))
            tot = 0.0
            for t in (2, 4):
                nl, rk, _c = GP.nll_and_rank(res[t]["logits"], am, dist, legal_m)
                tot = tot + 0.5 * (nl + rk)
            nl, rk, _c = GP.nll_and_rank(res[4]["logits"], am, dist, legal_m)
            return tot, nl, rk, res[4]["logits"].argmax(-1), zero
        res = m.forward_group(sub, extra=extra_codes)[None]
        if k == "ACTION":
            nl, rk, _c = GP.nll_and_rank(res["logits"], am, dist, legal_m)
            return nl + rk, nl, rk, res["logits"].argmax(-1), zero
        if k in ("ABS", "REL"):
            logits = calibrated_logits(res["v0"], res["vn"], res["legal"], float(m.a0), float(m.c0), m.log_beta)
            nl, rk, _c = GP.nll_and_rank(logits, am, dist, legal_m)
            cal = []
            for j in range(len(sub)):
                dj = d_extra[j]
                w_s = m.a0 * res["v0"][j] + m.c0
                lg = torch.nonzero(legal_m[j]).reshape(-1)
                w_n = m.a0 * res["vn"][j][lg] + m.c0
                w_e = m.a0 * res["vx"][j] + m.c0
                d_s = torch.tensor(float(dj["d0"]), device=dev)
                d_n = torch.stack([dist[j][i] for i in lg]) if len(lg) else dist.new_zeros(0)
                d_e = torch.tensor([float(e["d"]) for e in dj["endpoints"]], device=dev)
                w = torch.cat((w_s.reshape(1), w_n, w_e))
                d = torch.cat((d_s.reshape(1), d_n, d_e))
                if k == "ABS":
                    cal.append(calibration_objectives(w, d, torch.zeros((0, 2), dtype=torch.long, device=dev), self.cal_scale)["absolute"])
                else:
                    pairs = torch.tensor([[0, i] for i in range(1, len(w))], dtype=torch.long, device=dev)
                    cal.append(calibration_objectives(w, d, pairs, self.cal_scale)["relative"])
            cal = torch.stack(cal)
            return nl + rk + LAMBDA_CAL * cal, nl, rk, logits.argmax(-1), cal
        # EVENT: BASE / FACT / JOINT
        pending = res["t0"] < 0.5
        gv = torch.ones_like(pending)
        jo = m.event(res["z0"], res["zn"], res["t0"], res["tn"], res["logits"].masked_fill(~res["legal"], 0.0), gv, pending, res["legal"])
        G = pending.shape[1]
        glabel = [lab for lab in self._current_event]
        labels_t = torch.zeros((len(sub), G, n), dtype=torch.bool, device=dev)
        complete = torch.zeros(len(sub), dtype=torch.bool, device=dev)
        gids = [g.fact_id for g in sub[0].template.goals]
        for j, lab in enumerate(glabel):
            if lab["complete"]:
                complete[j] = True
                for g, a in lab["Y"]:
                    labels_t[j, gids.index(g), ids.index(a)] = True
        lt = loss_terms_v3(jo, am, dist, res["legal"], pending, labels_t, complete, k)
        return lt["loss"], lt["action_nll"], lt["rank"], jo.log_action.argmax(-1), lt["pair_specific"] * 0

    # ------------------------------------------------------------------ optimisation
    def step(self, trajs):
        m = self.model
        m.train()
        per = [self.snapshots(t) for t in trajs]
        snaps = [s for p in per for s in p]
        astar = [a for t in trajs for a in t["astar"]]
        keys = [GP.rank_key(t["case_id"], s) for t in trajs for s in t["states"][:-1]]
        w = torch.tensor([x for ws in I.decision_weights(trajs) for x in ws], device=self.device)
        extra = d_extra = ev = None
        if self.loss_kind in ("ABS", "REL"):
            extra = [e for t, p in zip(trajs, per) for e in self.extra_codes(t, p)]
            d_extra = [self.cal_targets["%s|%d" % (t["tid"], i)] for t in trajs for i in range(len(t["actions"]))]
        if self.loss_kind in ("BASE", "FACT", "JOINT"):
            ev = [self.event_labels[k] for k in keys]
        self.optimizer.zero_grad()
        tot = {"loss": 0.0, "nll": 0.0, "rank": 0.0, "cal": 0.0, "hit": 0.0}
        order = {}
        for i, sn in enumerate(snaps):
            order.setdefault(id(sn.template), []).append(i)
        for idx in order.values():
            for a in range(0, len(idx), self.chunk):
                sel = idx[a:a + self.chunk]
                sub = [snaps[i] for i in sel]
                self._current_event = [ev[i] for i in sel] if ev is not None else None
                loss, nl, rk, top, cal = self.group_loss(sub, [astar[i] for i in sel], [keys[i] for i in sel], [extra[i] for i in sel] if extra else None, [d_extra[i] for i in sel] if d_extra else None)
                ws = w[sel]
                part = (ws * loss).sum()
                if not torch.isfinite(part):
                    self.nan_events += 1
                    raise I.ImitationError("non-finite loss")
                part.backward()
                tot["loss"] += float(part)
                tot["nll"] += float((ws * nl).sum())
                tot["rank"] += float((ws * rk).sum())
                tot["cal"] += float((ws * cal).sum()) if torch.is_tensor(cal) and cal.ndim else 0.0
                am_hit = torch.zeros((len(sub), len(sub[0].candidate_ids)), dtype=torch.bool, device=self.device)
                for j, i in enumerate(sel):
                    for aid in astar[i]:
                        am_hit[j, sub[0].candidate_ids.index(aid)] = True
                tot["hit"] += float((ws * am_hit.gather(1, top.unsqueeze(1)).squeeze(1).float()).sum())
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, GRAD_CLIP, error_if_nonfinite=True))
        new_g = float(sum(float(p.grad.norm() ** 2) for p in self.model.new_params() if p.grad is not None) ** 0.5) if self.model.new_params() else 0.0
        self.optimizer.step()
        if m.log_beta is not None:
            with torch.no_grad():
                before = float(m.log_beta)
                m.log_beta.clamp_(-ETA_BOUND, ETA_BOUND)
                self.eta_clamped += float(m.log_beta) != before
        self.steps += 1
        return {**tot, "grad_norm": norm, "new_module_grad_norm": new_g, "decisions": len(snaps)}

    def epoch(self, trajs, seed):
        order = list(range(len(trajs)))
        random.Random(seed).shuffle(order)
        rows = [self.step([trajs[i] for i in order[k:k + BATCH]]) for k in range(0, len(order), BATCH)]
        n = len(rows)
        out = {"batches": n, "decisions": sum(r["decisions"] for r in rows), **{k: sum(r[k] for r in rows) / n for k in ("loss", "nll", "rank", "cal", "hit", "grad_norm", "new_module_grad_norm")}}
        if self.model.log_beta is not None:
            out["log_beta"] = float(self.model.log_beta)
            out["eta_clamped_total"] = self.eta_clamped
        return out
