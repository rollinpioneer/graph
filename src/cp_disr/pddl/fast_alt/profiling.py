"""Bounded profiler of the neural scoring path (plan section 6): input library, split timings of the reference and the FAST path, micro-benchmarks and the call-cost description."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .. import depots as DP
from .. import task as T
from ..search_match import evaluators as EV
from ..search_match.budget import Budget
from ..search_match.engine import SuccessorIndex, gbfs
from .evaluators import FastNeuralEval, RefNeuralEval, TracingEval

BATCH_SIZES = (1, 4, 8, 16, 32, 48)
LOGICAL_BATCH = 64
WARMUP, REPEATS = 2, 3
IPC_PROBE = ("ipc_p15", "ipc_p21", "ipc_p22")


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def pick_templates(old):
    """Train96: smallest, median and largest template by edge count (ties by case id); plus IPC p15, p21, p22."""
    rows = []
    for c in sorted(old.manifest["train"], key=lambda x: x["case_id"]):
        task = T.depots_task(DP.DOMAIN_TYPED, c["file"])
        rows.append((len(task.template().edges), c["case_id"], c))
    rows.sort(key=lambda x: (x[0], x[1]))
    pick = [("train_small", rows[0][2]), ("train_median", rows[len(rows) // 2][2]), ("train_large", rows[-1][2])]
    ipc = {c["case_id"]: c for c in old.manifest["ipc"]}
    pick += [(cid, ipc[cid]) for cid in IPC_PROBE]
    return pick


def collect_inputs(old, ref_eval, pick, budget_seconds=60.0, max_expansions=32, max_states=1024):
    """Natural successor batches of the first (<= 32) valid expansions of the reference search + the distinct reachable states seen (<= 1024). Returns {label: dict}. Counted as new search activity."""
    out = {}
    for label, c in pick:
        dom = DP.DOMAIN_IPC if c["case_id"].startswith("ipc") else DP.DOMAIN_TYPED
        task = T.depots_task(dom, c["file"])
        idx = SuccessorIndex(task)
        tr = TracingEval(ref_eval)
        b = Budget(budget_seconds, max_expansions, 8 * 2 ** 30)
        tr.prepare(task, {}, b)
        res = gbfs(task, idx, tr, b)
        batches = [list(x) for x in tr.batches]
        distinct, seen = [], set()
        for bt in batches:
            for s in bt:
                if s not in seen and len(distinct) < max_states:
                    seen.add(s)
                    distinct.append(s)
        out[label] = {"case_id": c["case_id"], "domain": str(dom), "file": c["file"], "status": res.status, "expanded": res.expanded, "wall": b.elapsed(), "batches": batches, "states": distinct,
                      "template_edges": len(task.template().edges), "template_nodes": len(task.template().nodes), "goals": len(task.template().goals)}
    return out


def save_inputs(inputs, path):
    with open(path, "w", encoding="utf-8") as f:
        for label, d in inputs.items():
            row = {k: v for k, v in d.items() if k not in ("batches", "states")}
            row["label"] = label
            row["batches"] = [[hex(s) for s in b] for b in d["batches"]]
            row["batch_sizes"] = [len(b) for b in d["batches"]]
            row["n_distinct_states"] = len(d["states"])
            f.write(json.dumps(row) + "\n")


def load_state_library(path):
    lib = {}
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        seen, states = set(), []
        batches = [[int(h, 16) for h in b] for b in d["batches"]]
        for b in batches:
            for s in b:
                if s not in seen:
                    seen.add(s)
                    states.append(s)
        lib[d["label"]] = dict(d, batches=batches, states=states)
    return lib


# ------------------------------------------------------------------------------------------------ split timings
@torch.no_grad()
def ref_split(model, template, task, states, dev):
    """Instrumented copy of the reference path ``state_values`` -> ``zv`` -> ``_encode`` -> heads. Returns (values, {section: seconds}). The sections are separated by device synchronisations (profiling only)."""
    Tm = dict(host_pack=0.0, tensor_h2d=0.0, static_prep=0.0, relational=0.0, unused_readout=0.0, goal_value=0.0, readback=0.0)
    st, gprop = model.mg.goal_free_static(template)
    base = model.mg.base
    enc = base.encoder
    static_true = {"p:" + p + (":" + ":".join(a) if a else "") for p, a in task.static}
    t0 = time.perf_counter()
    rows = []
    for s in states:
        true = task.atom_true_ids(s) | static_true
        rows.append([[1.0, 0.0, 0.0] if p in true else [0.0, 1.0, 0.0] for p in st.prop_ids])
    Tm["host_pack"] += time.perf_counter() - t0
    sync(dev)
    t0 = time.perf_counter()
    codes = torch.stack([torch.tensor(r, device=dev) for r in rows])
    sync(dev)
    Tm["tensor_h2d"] += time.perf_counter() - t0
    E = int(st.ei.shape[1])
    ch = max(1, type(model).EDGE_BUDGET // max(E, 1))
    outs = []
    for a in range(0, codes.shape[0], ch):
        cc = codes[a:a + ch]
        sync(dev)
        t0 = time.perf_counter()
        feats = enc.features
        G, N = cc.shape[0], st.N
        static = feats.kind.weight[st.kind_idx] + torch.where(st.is_action.unsqueeze(-1), feats.action.weight[st.action_idx], feats.predicate.weight[st.pred_idx])
        static = static + (feats.arg.weight[st.arg_idx] * st.arg_w.unsqueeze(-1)).sum(1)
        dyn = feats.fact(torch.cat((cc, st.prop_goal_sign.unsqueeze(0).expand(G, -1, -1)), dim=-1))
        dyn_full = torch.zeros((G, N, dyn.shape[-1]), device=dyn.device, dtype=dyn.dtype)
        dyn_full[:, st.prop_pos] = dyn
        h = static.unsqueeze(0) + dyn_full
        offsets = (torch.arange(G, device=h.device) * N).repeat_interleave(E)
        ei = st.ei.repeat(1, G) + offsets.unsqueeze(0)
        et = st.et.repeat(G)
        hflat = h.reshape(G * N, -1)
        sync(dev)
        Tm["static_prep"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for layer, norm in zip(enc.layers, enc.norms):
            hflat = norm(torch.relu(layer(hflat, ei, et)))
        H = hflat.reshape(G, N, -1)
        sync(dev)
        Tm["relational"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        ro = enc.readout
        goals = ro.goal(torch.cat((H[:, st.goal_pos], st.goal_sign.unsqueeze(0).expand(G, -1, -1)), dim=-1))
        ro.global_readout(torch.cat((H[:, st.action_pos].mean(1), H[:, st.prop_pos].mean(1), goals.mean(1)), dim=-1))
        sync(dev)
        Tm["unused_readout"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        x = model._features(st, gprop, cc, H)
        z = model.mg.heads.phi(x)
        if model.attn is not None:
            gs = model.attn._goal_static(st, gprop, template)
            f, al = model.attn.pair_features(gs, cc, gprop)
            z = z + model.attn(z, f, al)
        v = model.mg.heads.rho(z.sum(1)).squeeze(-1)
        sync(dev)
        Tm["goal_value"] += time.perf_counter() - t0
        outs.append(v)
    t0 = time.perf_counter()
    vals = torch.cat(outs).double().cpu().tolist()
    Tm["readback"] += time.perf_counter() - t0
    return vals, Tm


@torch.no_grad()
def fast_split(fv, states, dev):
    """Instrumented FAST path (same operations as ``FastValue.values``)."""
    Tm = dict(host_pack=0.0, tensor_h2d=0.0, static_prep=0.0, relational=0.0, unused_readout=0.0, goal_value=0.0, readback=0.0)
    model, st = fv.model, fv.st
    ch = max(1, fv.edge_budget // max(fv.E, 1))
    outs = []
    G0 = len(states)
    t0 = time.perf_counter()
    buf = b"".join(s.to_bytes(fv.nbytes, "little") for s in states)
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8).reshape(G0, fv.nbytes), axis=1, bitorder="little")
    true = (bits[:, fv.bit_idx].astype(bool) & fv.has_bit) | fv.is_static
    Tm["host_pack"] += time.perf_counter() - t0
    sync(dev)
    t0 = time.perf_counter()
    t = torch.from_numpy(true).to(dev)
    codes = torch.zeros((G0, true.shape[1], 3), device=dev)
    codes[..., 0] = t
    codes[..., 1] = ~t
    sync(dev)
    Tm["tensor_h2d"] += time.perf_counter() - t0
    for a in range(0, G0, ch):
        cc = codes[a:a + ch]
        G, N = cc.shape[0], fv.N
        sync(dev)
        t0 = time.perf_counter()
        feats = fv.enc.features
        dyn = feats.fact(torch.cat((cc, st.prop_goal_sign.unsqueeze(0).expand(G, -1, -1)), dim=-1))
        h = fv.static.unsqueeze(0).repeat(G, 1, 1)
        h[:, st.prop_pos] = h[:, st.prop_pos] + dyn
        edges = fv.edges_for(G)
        hflat = h.reshape(G * N, -1)
        sync(dev)
        Tm["static_prep"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        size = (G * N, G * N)
        for li, (layer, norm) in enumerate(zip(fv.enc.layers, fv.enc.norms)):
            W = fv.weights[li]
            out = torch.zeros(G * N, layer.out_channels, device=dev)
            for r in range(fv.R):
                hh = layer.propagate(edges[r], x=hflat, edge_type_ptr=None, size=size)
                out = out + (hh @ W[r])
            out = out + hflat @ layer.root
            out = out + layer.bias
            hflat = norm(torch.relu(out))
        H = hflat.reshape(G, N, -1)
        sync(dev)
        Tm["relational"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        x = model._features(st, fv.gprop, cc, H)
        z = model.mg.heads.phi(x)
        if model.attn is not None:
            gs = model.attn._goal_static(st, fv.gprop, fv.template)
            f, al = model.attn.pair_features(gs, cc, fv.gprop)
            z = z + model.attn(z, f, al)
        v = model.mg.heads.rho(z.sum(1)).squeeze(-1)
        sync(dev)
        Tm["goal_value"] += time.perf_counter() - t0
        outs.append(v)
    t0 = time.perf_counter()
    vals = torch.cat(outs).double().cpu().tolist()
    Tm["readback"] += time.perf_counter() - t0
    return vals, Tm


# ------------------------------------------------------------------------------------------------ micro-benchmark
def library_batch(states, B):
    """B states from the library (repeated when the library is shorter: timing only, flagged)."""
    reps = -(-B // len(states))
    return (states * reps)[:B], B > len(states)


def interleave_ab(label, B, rep):
    """Deterministic AB / BA order per (template, batch, repeat)."""
    return int(hashlib.sha1(("%s|%d|%d" % (label, B, rep)).encode()).hexdigest(), 16) % 2 == 0


def microbench(old, model, dev, lib, log=print):
    """REF vs FAST end-to-end (``evaluate``) and split timings on the library states; 2 warm-ups + 3 interleaved recorded repeats per cell."""
    rows, models = [], {}
    for label, d in lib.items():
        dom = DP.DOMAIN_IPC if d["case_id"].startswith("ipc") else DP.DOMAIN_TYPED
        task = T.depots_task(dom, d["file"])
        ref, fast = RefNeuralEval("REF", model, dev), FastNeuralEval("FAST", model, dev)
        b = Budget(3600, 10 ** 9, 64 * 2 ** 30)
        ref.prepare(task, {}, b)
        fast.prepare(task, {}, b)
        template = task.template()
        E = len(template.edges)
        G = len(template.goals)
        ch = max(1, type(model).EDGE_BUDGET // max(E, 1))
        for B in list(BATCH_SIZES) + [LOGICAL_BATCH]:
            states, repeated = library_batch(d["states"], B)
            for impl in ("REF", "FAST"):                                        # warm-up
                ev = ref if impl == "REF" else fast
                for _ in range(WARMUP):
                    ev.evaluate(states, b)
            for rep in range(REPEATS):
                order = ("REF", "FAST") if interleave_ab(label, B, rep) else ("FAST", "REF")
                for impl in order:
                    ev = ref if impl == "REF" else fast
                    sync(dev)
                    t0 = time.perf_counter()
                    ev.evaluate(states, b)
                    sync(dev)
                    wall = time.perf_counter() - t0
                    # split timing of the same call
                    if impl == "REF":
                        _, tm = ref_split(model, template, task, states[:EV.NEURAL_CHUNK], dev)
                        if B > EV.NEURAL_CHUNK:
                            _, tm2 = ref_split(model, template, task, states[EV.NEURAL_CHUNK:], dev)
                            tm = {k: tm[k] + tm2[k] for k in tm}
                    else:
                        _, tm = fast_split(fast.fv, states[:EV.NEURAL_CHUNK], dev)
                        if B > EV.NEURAL_CHUNK:
                            _, tm2 = fast_split(fast.fv, states[EV.NEURAL_CHUNK:], dev)
                            tm = {k: tm[k] + tm2[k] for k in tm}
                    n_outer = -(-B // EV.NEURAL_CHUNK)
                    enc_forwards = sum(-(-len(states[i:i + EV.NEURAL_CHUNK]) // ch) for i in range(0, B, EV.NEURAL_CHUNK))
                    rows.append({"template": label, "case_id": d["case_id"], "impl": impl, "B": B, "repeat": rep, "order": "".join(o[0] for o in order), "wall_ms": round(wall * 1000, 3), "per_state_ms": round(wall * 1000 / B, 4),
                                 "outer_calls": n_outer, "encoder_forwards": enc_forwards, "template_edges": E, "template_nodes": len(template.nodes), "goals": G, "edge_chunk": ch, "repeated_states": repeated,
                                 **{"split_%s_ms" % k: round(v * 1000, 3) for k, v in tm.items()}})
        log("[profile] %s done (%d rows)" % (label, len(rows)))
    return rows


def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ cost description
def fit_cost_model(rows):
    """T(B, E, G) ~ alpha + beta B + gamma B E + delta B G^2 on measurements that fit a single encoder forward (B <= edge chunk); per-template alpha + beta B always reported."""
    out = {"definition": "wall_ms of one outer call ~ alpha + beta*B + gamma*B*E + delta*B*G^2 (single encoder forward only); descriptive, not causal", "per_impl": {}}
    for impl in ("REF", "FAST"):
        pts = [r for r in rows if r["impl"] == impl and r["B"] <= 48 and r["B"] <= r["edge_chunk"]]
        per_t = {}
        for label in sorted({r["template"] for r in pts}):
            xs = np.array([[1.0, r["B"]] for r in pts if r["template"] == label])
            ys = np.array([r["wall_ms"] for r in pts if r["template"] == label])
            if len(set(xs[:, 1])) >= 2:
                coef, *_ = np.linalg.lstsq(xs, ys, rcond=None)
                resid = ys - xs @ coef
                per_t[label] = {"alpha_ms": round(float(coef[0]), 4), "beta_ms_per_state": round(float(coef[1]), 5), "rmse_ms": round(float(np.sqrt((resid ** 2).mean())), 4), "points": len(ys), "B_range": [float(xs[:, 1].min()), float(xs[:, 1].max())]}
        entry = {"per_template_alpha_beta": per_t}
        if len({r["template"] for r in pts}) >= 4:
            X = np.array([[1.0, r["B"], r["B"] * r["template_edges"], r["B"] * r["goals"] ** 2] for r in pts])
            y = np.array([r["wall_ms"] for r in pts])
            cond = float(np.linalg.cond(X))
            if cond < 1e8:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                res = y - X @ coef
                ss = float(((y - y.mean()) ** 2).sum())
                entry["global"] = {"alpha_ms": float(coef[0]), "beta": float(coef[1]), "gamma_per_edge_state": float(coef[2]), "delta_per_goal2_state": float(coef[3]), "r2": round(1 - float((res ** 2).sum()) / ss, 4) if ss > 0 else None,
                                   "condition_number": cond, "points": len(y)}
            else:
                entry["global"] = {"status": "NOT_FITTED_COLLINEAR", "condition_number": cond}
        out["per_impl"][impl] = entry
    return out


def speedups(rows):
    """Paired REF / FAST speed-up per (template, B) from the medians of the interleaved repeats."""
    res = []
    for label in sorted({r["template"] for r in rows}):
        for B in sorted({r["B"] for r in rows if r["template"] == label}):
            g = lambda impl: sorted(r["wall_ms"] for r in rows if r["template"] == label and r["B"] == B and r["impl"] == impl)
            a, b = g("REF"), g("FAST")
            if a and b:
                ma, mb = a[len(a) // 2], b[len(b) // 2]
                res.append({"template": label, "B": B, "ref_ms": ma, "fast_ms": mb, "speedup": round(ma / mb, 3) if mb else None})
    return res
