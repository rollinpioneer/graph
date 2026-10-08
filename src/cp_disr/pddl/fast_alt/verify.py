"""Equivalence checks of FAST against the reference scorer (plan section 7.3). GPU arithmetic of the reference path is itself not bitwise repeatable (atomic scatter in message passing), so every
comparison is calibrated against the reference's own repeat (REF vs REF'), the plan's absolute / relative tolerance is reported next to it, and search prefixes are compared with the same baseline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .. import depots as DP
from .. import task as T
from ..search_match.budget import Budget
from ..search_match.engine import SuccessorIndex, gbfs
from .evaluators import FastNeuralEval, RefNeuralEval, TracingEval
from . import profiling as PF

PLAN_ATOL, PLAN_RTOL = 1e-5, 1e-6
NOISE_FACTOR, NOISE_FLOOR = 2.0, 1e-4              # registered pass rule: max|FAST-REF| <= NOISE_FACTOR * max|REF-REF'| + NOISE_FLOOR
PREFIX_SLACK = 2                                    # FAST may diverge from REF on at most (problems where REF' diverges from REF) + 2 fixture problems


def within_plan_tol(ref, x):
    return abs(ref - x) <= PLAN_ATOL + PLAN_RTOL * abs(ref)


def reference_codes(model, template, task, states, dev):
    st, _g = model.mg.goal_free_static(template)
    from ..task import codes_for_state_general
    static_true = {"p:" + p + (":" + ":".join(a) if a else "") for p, a in task.static}
    return torch.stack([codes_for_state_general(st, task, s, static_true) for s in states])


@torch.no_grad()
def check_tensors(old, model, dev, lib, extra_states=24):
    """Packed proposition codes and per-relation batched edge tensors against the reference construction (must be bitwise identical)."""
    out = {}
    for label, d in lib.items():
        dom = DP.DOMAIN_IPC if d["case_id"].startswith("ipc") else DP.DOMAIN_TYPED
        task = T.depots_task(dom, d["file"])
        tpl = task.template()
        ev = FastNeuralEval("FAST", model, dev)
        ev.prepare(task, {}, Budget(600, 10 ** 9, 64 * 2 ** 30))
        fv = ev.fv
        states = list(d["states"][:256])
        # high-bit and all-ones style extremes: highest dynamic atom set, lowest set, mixtures
        nbits = len(task.dyn_atoms)
        states += [1 << (nbits - 1), 1, (1 << nbits) - 1, (1 << (nbits // 2)) | 1]
        ref = reference_codes(model, tpl, task, states, dev)
        got = fv.pack(states)
        edges_ok = True
        st = fv.st
        feats = model.mg.base.encoder.features
        ref_static = feats.kind.weight[st.kind_idx] + torch.where(st.is_action.unsqueeze(-1), feats.action.weight[st.action_idx], feats.predicate.weight[st.pred_idx])
        ref_static = ref_static + (feats.arg.weight[st.arg_idx] * st.arg_w.unsqueeze(-1)).sum(1)
        static_ok = bool(torch.equal(ref_static, fv.static))
        for G in (1, 5, 13):
            offs = (torch.arange(G, device=dev) * fv.N).repeat_interleave(fv.E)
            ei = st.ei.repeat(1, G) + offs.unsqueeze(0)
            et = st.et.repeat(G)
            from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
            cached = fv.edges_for(G)
            edges_ok &= all(torch.equal(masked_edge_index(ei, et == r), cached[r]) for r in range(fv.R))
        out[label] = {"states": len(states), "state_bits": nbits, "codes_identical": bool(torch.equal(ref, got)), "codes_max_abs_diff": float((ref - got).abs().max()), "edge_tensors_identical": bool(edges_ok),
                      "static_embedding_identical": static_ok}
        del ev
    return out


@torch.no_grad()
def check_values(old, model, dev, lib):
    """FAST vs REF on natural batches and library chunks; REF vs REF' repeat as the noise baseline."""
    out, ok = {}, True
    for label, d in lib.items():
        dom = DP.DOMAIN_IPC if d["case_id"].startswith("ipc") else DP.DOMAIN_TYPED
        task = T.depots_task(dom, d["file"])
        ref, fast = RefNeuralEval("REF", model, dev), FastNeuralEval("FAST", model, dev)
        b = Budget(3600, 10 ** 9, 64 * 2 ** 30)
        ref.prepare(task, {}, b)
        fast.prepare(task, {}, b)
        batches = [bt for bt in d["batches"] if bt] + [d["states"][i:i + 48] for i in range(0, min(len(d["states"]), 192), 48)]
        singles = [[s] for s in d["states"][:16]]
        rows = []
        for group, items in (("natural_or_chunk", batches), ("single", singles)):
            max_fr = max_rr = 0.0
            tol_fast = tol_ref = tot = bit_fast = bit_ref = 0
            order_fast = order_ref = argmin_fast = argmin_ref = nb = 0
            for bt in items:
                r1 = ref.evaluate(bt, b)
                r2 = ref.evaluate(bt, b)
                f = fast.evaluate(bt, b)
                for x, y, z in zip(r1, r2, f):
                    tot += 1
                    max_fr, max_rr = max(max_fr, abs(x - z)), max(max_rr, abs(x - y))
                    tol_fast += within_plan_tol(x, z)
                    tol_ref += within_plan_tol(x, y)
                    bit_fast += (x == z)
                    bit_ref += (x == y)
                key = lambda v: sorted(range(len(v)), key=lambda i: (v[i], i))
                nb += 1
                order_fast += key(r1) == key(f)
                order_ref += key(r1) == key(r2)
                argmin_fast += key(r1)[0] == key(f)[0]
                argmin_ref += key(r1)[0] == key(r2)[0]
            passed = max_fr <= NOISE_FACTOR * max_rr + NOISE_FLOOR
            ok &= passed
            rows.append({"group": group, "batches": nb, "values": tot, "max_abs_diff_fast_vs_ref": max_fr, "max_abs_diff_ref_vs_ref_repeat": max_rr, "within_plan_tolerance_fast": tol_fast / max(tot, 1),
                         "within_plan_tolerance_ref_repeat": tol_ref / max(tot, 1), "bitwise_equal_fast": bit_fast / max(tot, 1), "bitwise_equal_ref_repeat": bit_ref / max(tot, 1),
                         "same_full_order_fast": order_fast / max(nb, 1), "same_full_order_ref_repeat": order_ref / max(nb, 1), "same_first_choice_fast": argmin_fast / max(nb, 1),
                         "same_first_choice_ref_repeat": argmin_ref / max(nb, 1), "pass_noise_rule": passed})
        out[label] = rows
    return out, ok


def fixture_problems(old):
    """Struct: the first two n4 and the first two n5 problems by id; Joint: per cell the first problem with and the first without transport (by id): 4 + 16 problems."""
    st = sorted(old.manifest["struct"], key=lambda c: c["case_id"])
    pick = [("struct", c) for n in ("n4", "n5") for c in [x for x in st if "_%s_" % n in x["case_id"]][:2]]
    jt = sorted(old.manifest["joint"], key=lambda c: c["case_id"])
    for cell in sorted({c["cell"] for c in jt}):
        for tr in (True, False):
            c = next(x for x in jt if x["cell"] == cell and bool(x["analysis"]["needs_transport"]) == tr)
            pick.append(("joint", c))
    return pick


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def check_prefixes(old, model, dev, wall=60.0, max_exp=1000):
    """REF, REF' and FAST on the fixture problems (<= 1000 expansions or the first solution, 60 s each); compares the generated-batch sequences, the plans and the first divergence."""
    rows = []
    for set_name, c in fixture_problems(old):
        dom = old.domain(set_name)
        task = T.depots_task(dom, c["file"])
        idx = SuccessorIndex(task)
        runs = {}
        for name, mk in (("REF", RefNeuralEval), ("REF_REPEAT", RefNeuralEval), ("FAST", FastNeuralEval)):
            tr = TracingEval(mk(name, model, dev))
            b = Budget(wall, max_exp, 8 * 2 ** 30)
            tr.prepare(task, {}, b)
            res = gbfs(task, idx, tr, b)
            runs[name] = {"batches": tr.batches, "plan": res.plan, "expanded": res.expanded, "status": res.status, "wall": b.elapsed()}
        flat = lambda r: [s for bt in r["batches"] for s in bt]
        d_self = first_divergence(runs["REF"]["batches"], runs["REF_REPEAT"]["batches"])
        d_fast = first_divergence(runs["REF"]["batches"], runs["FAST"]["batches"])
        rows.append({"set": set_name, "case_id": c["case_id"], "status_ref": runs["REF"]["status"], "expanded_ref": runs["REF"]["expanded"], "batches_ref": len(runs["REF"]["batches"]),
                     "first_divergence_ref_vs_repeat": d_self, "first_divergence_ref_vs_fast": d_fast, "plan_equal_ref_repeat": runs["REF"]["plan"] == runs["REF_REPEAT"]["plan"],
                     "plan_equal_ref_fast": runs["REF"]["plan"] == runs["FAST"]["plan"], "generation_order_equal_ref_fast": flat(runs["REF"]) == flat(runs["FAST"]), "status_fast": runs["FAST"]["status"],
                     "expanded_fast": runs["FAST"]["expanded"], "wall_ref_s": round(runs["REF"]["wall"], 3), "wall_fast_s": round(runs["FAST"]["wall"], 3)})
    div_self = sum(1 for r in rows if r["first_divergence_ref_vs_repeat"] is not None)
    div_fast = sum(1 for r in rows if r["first_divergence_ref_vs_fast"] is not None)
    summary = {"problems": len(rows), "ref_vs_repeat_divergent": div_self, "ref_vs_fast_divergent": div_fast, "pass_rule": "FAST divergent problems <= REF-repeat divergent problems + %d" % PREFIX_SLACK,
               "pass": div_fast <= div_self + PREFIX_SLACK, "plan_equal_ref_fast": sum(1 for r in rows if r["plan_equal_ref_fast"]), "plan_equal_ref_repeat": sum(1 for r in rows if r["plan_equal_ref_repeat"])}
    return rows, summary
