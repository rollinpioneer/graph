"""Stage implementations of the public-Depots card (library; the CLI in scripts/c1_public_depots.py only parses arguments and writes receipts).

Data roles: train (exact labels, gradients) / dev (checkpoint selection) / struct + joint (Track B sealed tests) / ipc (Track A public instances). A model sees no test problem before ``lock``.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from . import data as DT
from . import depots as DP
from . import task as T

# ------------------------------------------------------------------------------------------------ frozen design constants (copied into the registration)
SEEDS = {"train": {3: 1_000_000, 4: 1_100_000, 5: 1_200_000}, "dev": {3: 2_000_000, 4: 2_100_000, 5: 2_200_000}, "struct": {4: 3_000_000, 5: 3_100_000}}
JOINT_SEED0 = 4_000_000
TRAIN_PER_N, DEV_PER_N, STRUCT_PER = 32, 8, 16
JOINT_PER_CELL = 16
MAX_ATTEMPTS = 150_000
LAMA_SECONDS, LMCUT_SECONDS, FD_MEMORY_MB = 300, 300, 8000
DEADLINE_SECONDS = 300                         # uniform wall-clock budget per problem for every time-limited method (neural controllers, WL-GOOSE, LAMA)
BUDGET = {"updates": 4100, "batch_trajectories": 32, "lr": 3e-4, "checkpoints": [820, 1640, 2460, 3280, 4100], "init_seed": 20261008, "chunk_decisions": 48, "grad_clip": 0.5, "rank_weight": 1.0}


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ data generation
def _fill(name, n, spec, count, seed0, outdir, seen, transport_quota=False, pallets=None):
    got, seed, attempts = [], seed0, 0
    quota = {True: (count + 1) // 2, False: count // 2} if transport_quota else None
    while len(got) < count:
        if attempts >= MAX_ATTEMPTS:
            break
        text, cmd = DP.generate(n, seed, pallets=pallets)
        seed += 1
        attempts += 1
        a = DP.analyse(text)
        if not (a["goal_complete"] and DP.shape_matches(a["goal_heights"], spec) and not a["init_satisfies_goal"]):
            continue
        h = DP.canonical_hash(text)
        if h in seen:
            continue
        if quota is not None and quota[a["needs_transport"]] <= 0:
            continue
        seen.add(h)
        if quota is not None:
            quota[a["needs_transport"]] -= 1
        cid = "%s_%03d" % (name, len(got))
        path = outdir / ("%s.pddl" % cid)
        path.write_text(text)
        got.append({"case_id": cid, "file": str(path), "sha256": sha_text(text), "generator_cmd": " ".join(cmd[1:]), "seed": seed - 1, "iso_hash": h, "analysis": a})
    return got, {"requested": count, "built": len(got), "attempts": attempts, "next_seed": seed}


def _balance(groups, report, family):
    """Transport coverage: every group (struct: crate count; joint: cell) is filled with half problems that need transport and half that do not. If any group cannot fill a condition, ALL groups keep the same
    (smaller) number of each condition (decided from generation counts only, before any model exists)."""
    k = min(min(sum(1 for c in g if c["analysis"]["needs_transport"]), sum(1 for c in g if not c["analysis"]["needs_transport"])) for g in groups.values())
    out = []
    for name, g in groups.items():
        keep = {True: k, False: k}
        for c in g:
            t = c["analysis"]["needs_transport"]
            if keep[t] > 0:
                keep[t] -= 1
                out.append(c)
            else:
                Path(c["file"]).unlink()
    report["%s_uniform_per_condition" % family] = k
    return out


def gen_data(rr):
    out = Path(rr) / "data" / "problems"
    seen, man, report = set(), {}, {}
    for fam, seeds, per in (("train", SEEDS["train"], TRAIN_PER_N), ("dev", SEEDS["dev"], DEV_PER_N)):
        cases = []
        for n in (3, 4, 5):
            d = out / fam
            d.mkdir(parents=True, exist_ok=True)
            got, rep = _fill("%s_n%d" % (fam, n), n, DP.one_tower({2, 3}), per, seeds[n], d, seen, transport_quota=True)
            cases += got
            report["%s_n%d" % (fam, n)] = rep
        man[fam] = cases
    groups = {}
    d = out / "struct"
    d.mkdir(parents=True, exist_ok=True)
    for n, spec in ((4, [2, 2]), (5, [3, 2])):
        got, rep = _fill("struct_n%d" % n, n, spec, STRUCT_PER, SEEDS["struct"][n], d, seen, transport_quota=True)
        groups[n] = got
        report["struct_n%d" % n] = rep
    man["struct"] = _balance(groups, report, "struct")
    groups = {}
    d = out / "joint"
    d.mkdir(parents=True, exist_ok=True)
    for ci, (cell, (n, spec)) in enumerate(DP.JOINT_CELLS.items()):
        got, rep = _fill("joint_%s" % cell, n, spec, JOINT_PER_CELL, JOINT_SEED0 + 100_000 * ci, d, seen, transport_quota=True)
        for g in got:
            g["cell"] = cell
        groups[cell] = got
        report["joint_%s" % cell] = rep
    man["joint"] = _balance(groups, report, "joint")
    ipc = []
    for i in range(1, 23):
        pf = DP.DOMAIN_IPC.parent / ("p%02d.pddl" % i)
        ipc.append({"case_id": "ipc_p%02d" % i, "file": str(pf), "sha256": sha_text(pf.read_text())})
    man["ipc"] = ipc
    man["domains"] = {"typed_generator_domain": {"path": str(DP.DOMAIN_TYPED), "sha256": sha_text(DP.DOMAIN_TYPED.read_text())}, "ipc_domain": {"path": str(DP.DOMAIN_IPC), "sha256": sha_text(DP.DOMAIN_IPC.read_text())}}
    man["generation_report"] = report
    return man


# ------------------------------------------------------------------------------------------------ exact labels (train / dev / struct)
def _label_worker(a):
    cid, domain, problem, light = a
    return DT.build_case_data(domain, problem, cid, light=light)


def exact_labels(man, workers=32):
    """Train / dev / struct: full label pool (optimal plans, trajectories, successor costs). Joint problems with 6 crates: exact optimum + destruction label only (2.6M states each); 8-crate and IPC problems have no exact label."""
    jobs = [(c["case_id"], str(DP.DOMAIN_TYPED), c["file"], False) for fam in ("train", "dev", "struct") for c in man[fam]]
    jobs += [(c["case_id"], str(DP.DOMAIN_TYPED), c["file"], True) for c in man["joint"] if c["analysis"]["n_crates"] == 6]
    with Pool(min(workers, len(jobs))) as p:
        res = p.map(_label_worker, jobs, chunksize=1)
    return {r["case_id"]: r for r in res}

# ------------------------------------------------------------------------------------------------ Fast Downward references
def _ref_worker(a):
    cid, domain, problem, workdir, lama_s, lmcut_s = a
    wd = Path(workdir) / cid
    lama = DT.run_fd(domain, problem, wd, "seq-sat-lama-2011", lama_s, FD_MEMORY_MB, tag="lama")
    first = DT.run_fd(domain, problem, wd, "lama-first", lama_s, FD_MEMORY_MB, tag="lamafirst")
    lm = DT.run_fd(domain, problem, wd, "seq-opt-lmcut", lmcut_s, FD_MEMORY_MB, tag="lmcut")
    return {"case_id": cid, "lama": lama, "lama_first": first, "lmcut": lm}


def fd_references(cases, domain, workdir, workers=40, lama_s=LAMA_SECONDS, lmcut_s=LMCUT_SECONDS):
    jobs = [(c["case_id"], str(domain), c["file"], str(workdir), lama_s, lmcut_s) for c in cases]
    with Pool(min(workers, len(jobs))) as p:
        res = p.map(_ref_worker, jobs, chunksize=1)
    return {r["case_id"]: r for r in res}


def reference_length(ref, exact=None):
    """(L_ref, kind): exact optimum > A*+LM-cut solution (optimal) > best LAMA plan (best known, not proved optimal)."""
    if exact is not None:
        return exact, "exact"
    lm = ref["lmcut"]
    if lm["solved"] and lm["best_length"] is not None:
        return lm["best_length"], "optimal_lmcut"
    best = [x["best_length"] for x in (ref["lama"], ref["lama_first"]) if x["best_length"] is not None]
    return (min(best), "best_known_lama") if best else (None, "unsolved")


# ------------------------------------------------------------------------------------------------ cases
_TASKS = {}


def get_task(domain, problem_file):
    key = (str(domain), str(problem_file))
    if key not in _TASKS:
        _TASKS[key] = T.depots_task(domain, problem_file)
    return _TASKS[key]


def make_case(c, split, domain, L, kind, **meta):
    task = get_task(domain, c["file"])
    return T.PddlCase(c["case_id"], split, task, int(L), T.step_cap_for(int(L)), dict(meta, length_kind=kind, **{k: c.get(k) for k in ("cell",) if c.get(k)}))


# ------------------------------------------------------------------------------------------------ training / selection / evaluation (appended to pddl_stages)
import torch

from . import control as CT


def episode_summary_row(res):
    L = max(1, res["optimal_length"])
    return {"case_id": res["case_id"], "success": res["success"], "steps": res["steps"], "L_ref": res["optimal_length"], "cap": res["step_cap"], "reason": res["reason"],
            "ratio": (res["steps"] / L) if res["success"] else None, "penalised_ratio": (res["steps"] / L) if res["success"] else ((res["step_cap"] + 1) / L),
            "interventions": res["interventions"], "wall_seconds": res["wall_seconds"]}


def summarise(results):
    rows = [episode_summary_row(r) for r in results]
    succ = [r for r in rows if r["success"]]
    return {"n": len(rows), "success": len(succ), "mean_penalised_ratio": round(statistics.mean(r["penalised_ratio"] for r in rows), 4) if rows else None,
            "mean_ratio_success": round(statistics.mean(r["ratio"] for r in succ), 4) if succ else None, "total_steps": sum(r["steps"] for r in rows),
            "perfect": sum(1 for r in results if r.get("decision_perfect")), "wall_seconds": round(sum(r["wall_seconds"] for r in rows), 2), "failures": dict(Counter(r["reason"] for r in rows if not r["success"]))}


def load_done(path):
    p = Path(path)
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["case_id"]] = d
    return out


class ExactLabeler:
    """Scoring-only: d* and optimal action ids of a visited state, from the complete state space of a small task."""

    def __init__(self, task):
        self.task, self.oracle = task, T.ExactOracle(task, 40_000_000)

    def __call__(self, state):
        if not self.oracle.ok:
            return -1, []
        d, succ = self.oracle.query(state)
        return d, [self.task.actions[a].aid for a, dd in succ if dd == d - 1]

    def close(self):
        self.oracle.close()


def run_set(out_path, cases, model, exec_name, tau=None, use_labels=False, steps=None):
    """Run every case once (resumable: finished case ids in ``out_path`` are skipped). Returns all result dicts."""
    from ..blocksworld.method_serial.model import SerialPolicy
    done = load_done(out_path)
    pol = SerialPolicy(model)
    for case in cases:
        if case.case_id in done:
            continue
        ops = CT.PddlOps(case.task, model)
        ep = T.PddlEpisode(case)
        cnt = CT.Counters()
        if exec_name == "one_step":
            chooser = CT.choose_one_step
        elif exec_name == "two_step":
            chooser = lambda o, e, s, lg, m: CT.choose_two_step(o, model, e, s, lg, m, cnt, "model", steps)
        elif exec_name == "gated":
            chooser = lambda o, e, s, lg, m: CT.choose_gated(o, model, e, s, lg, m, tau, cnt, steps)
        else:
            raise ValueError(exec_name)
        lab = ExactLabeler(case.task) if use_labels and case.meta.get("exact") else None
        try:
            res = CT.run_episode(pol, ep, ops, chooser, lab, deadline_seconds=DEADLINE_SECONDS)
        finally:
            if lab is not None:
                lab.close()
        res["counters"] = cnt.as_dict()
        res["meta"] = {k: v for k, v in case.meta.items() if isinstance(v, (str, int, float, bool))}
        for d in res["decisions"]:
            d.pop("state", None)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done[case.case_id] = res
    return [done[c.case_id] for c in cases]


def train_loop(out_dir, mode, device, cases, trajs, labels, log=print):
    """Train one model for BUDGET['updates'] optimizer updates (resumable at 100-update boundaries); checkpoints at BUDGET['checkpoints']."""
    from ..torch_rl import save_checkpoint
    from . import model as PM, train as PT
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    m = PM.make_model(mode, device, seed=BUDGET["init_seed"])
    tr = PT.PddlTrainer(m, cases, labels, device, lr=BUDGET["lr"], chunk=BUDGET["chunk_decisions"])
    resume = out_dir / "resume_last.pt"
    rows, update, attempts, wall_prev = [], 0, 1, 0.0
    if resume.is_file():
        st = torch.load(resume, map_location=device, weights_only=False)
        m.load_state_dict(st["model"])
        tr.optimizer.load_state_dict(st["optimizer"])
        rows, update, attempts, wall_prev = st["rows"], st["update"], st["attempts"] + 1, st["wall"]
        tr.steps = update
    t0 = time.time()
    ckpts = {}
    for u in BUDGET["checkpoints"]:
        p = out_dir / "checkpoints" / ("u%05d.pt" % u)
        if p.is_file():
            ckpts[str(u)] = {"path": str(p), "sha256": _sha(p), "bytes": p.stat().st_size}
    nb = len(tr.update_batches(trajs, 0))
    init_new = [p.detach().clone().cpu() for p in m.new_params()]
    while update < BUDGET["updates"]:
        ep, off = divmod(update, nb)
        batches = tr.update_batches(trajs, ep)
        r = tr.step(batches[off])
        update += 1
        r["update"], r["epoch"], r["wall_seconds"] = update, ep, round(wall_prev + time.time() - t0, 1)
        rows.append(r)
        if update % 100 == 0 or update == 1:
            log("[pddl-train] %s update %d %s" % (mode, update, {k: round(v, 5) for k, v in r.items() if isinstance(v, float)}))
        if update in BUDGET["checkpoints"]:
            p = out_dir / "checkpoints" / ("u%05d.pt" % update)
            save_checkpoint(p, m, tr.optimizer, {"mode": mode, "update": update})
            ckpts[str(update)] = {"path": str(p), "sha256": _sha(p), "bytes": p.stat().st_size}
        if update % 100 == 0 or update == BUDGET["updates"]:
            tmp = out_dir / "resume_last.pt.tmp"
            torch.save({"model": m.state_dict(), "optimizer": tr.optimizer.state_dict(), "rows": rows, "update": update, "attempts": attempts, "wall": wall_prev + time.time() - t0}, tmp)
            os.replace(tmp, resume)
    new_now = [p.detach().cpu() for p in m.new_params()]
    acct = {"mode": mode, "optimizer_steps": update, "decision_samples_shown": sum(r["decisions"] for r in rows), "nan_events": tr.nan_events, "attempts": attempts, "wall_seconds": round(wall_prev + time.time() - t0, 1),
            "parameters_trainable": sum(p.numel() for p in tr.params), "parameters_new_module": sum(p.numel() for p in m.new_params()), "checkpoints": ckpts, "batches_per_epoch": nb, "trajectories": len(trajs),
            "final_update_row": {k: v for k, v in rows[-1].items()}, "new_module_param_change_l2": float(sum(((a - b) ** 2).sum() for a, b in zip(new_now, init_new)) ** 0.5) if new_now else 0.0,
            "budget": BUDGET}
    return acct, rows


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select_checkpoint(rows_by_update):
    """Dev selection: most successes, then smaller failure-penalised cost ratio, then the earlier checkpoint."""
    return max(rows_by_update, key=lambda u: (rows_by_update[u]["success"], -rows_by_update[u]["mean_penalised_ratio"], -int(u)))


def gate_threshold(model, cases, trajs, q=10):
    """10th percentile of the top-2 logit margin over unique TRAINING decisions with >= 2 legal actions (per scorer, frozen before any test episode)."""
    from .task import snapshot_at
    by_id = {c.case_id: c for c in cases}
    seen, margins = set(), []
    for t in trajs:
        for s in t["states"][:-1]:
            key = (t["case_id"], tuple(s))
            if key in seen:
                continue
            seen.add(key)
            sn = snapshot_at(by_id[t["case_id"]], s, 0)
            legal = [i for i, m_ in enumerate(sn.mask) if m_]
            if len(legal) < 2:
                continue
            lg = sorted((float(x) for x in model.eval_logits(sn)[legal]), reverse=True)
            margins.append(lg[0] - lg[1])
    if not margins:
        return None, 0
    ys = sorted(margins)
    k = (len(ys) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo), len(ys)
