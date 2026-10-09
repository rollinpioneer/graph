"""Training / selection stage code of card CP-DISR-C1-COMPACT-DIAGNOSIS-V2 for D0 (original DENSE-G recipe) and C0 (non-attention control), and the deterministic schedule ledger shared by both arms (and by a possible T1).

Everything that defines the recipe is the code of the earlier public-Depots card, imported and not modified: ``PddlTrainer`` (ACTION loss = action-set NLL + mean pairwise rank loss over strictly ordered legal successors, per-decision weight
1 / (trajectory length x batch size)), Adam lr 3e-4 for every parameter, gradient clip 0.5, batches of 32 trajectories, 4,100 accepted updates, checkpoints 820 / 1640 / 2460 / 3280 / 4100, common initialisation seed 20261008.
The only differences of this module to ``stages.train_loop`` are the model factory (C0 mode) and extra accounting."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

from .. import depots as DP
from .. import stages as ST
from .. import train as PT
from ...blocksworld.method_serial import trainer as BST
from ...torch_rl import load_checkpoint, save_checkpoint
from .models import count_new, count_trainable, make_model_v2

BUDGET = dict(ST.BUDGET)


def rj(p):
    return json.loads(Path(p).read_text())


def sha_file(p):
    return ST._sha(p)


def load_old(root):
    root = Path(root)
    return rj(root / "data" / "manifest.json"), rj(root / "data" / "exact.json"), root


def train_inputs(root):
    man, exact, _ = load_old(root)
    cases, trajs, labels = [], [], {}
    for c in man["train"]:
        ex = exact[c["case_id"]]
        if ex["status"] != "OK":
            continue
        cases.append(ST.make_case(c, "train", DP.DOMAIN_TYPED, ex["optimal_length"], "exact"))
        trajs += ex["trajectories"]
        labels.update(ex["rank_labels"])
    return cases, trajs, labels


def dev_cases(root):
    man, exact, root = load_old(root)
    refs = rj(root / "data" / "refs_dev.json")
    out = []
    for c in man["dev"]:
        cid = c["case_id"]
        ex = exact.get(cid)
        if ex and ex.get("status") == "OK":
            L, kind = ex["optimal_length"], "exact"
        else:
            L, kind = ST.reference_length(refs[cid])
        meta = {"exact": bool(ex and ex.get("status") == "OK"), "length_kind": kind}
        a = c.get("analysis")
        if a:
            meta.update({"n_crates": a["n_crates"], "k_towers": a["k_goal_towers"], "max_goal_height": a["max_goal_height"], "needs_transport": a["needs_transport"], "max_init_height": a["max_init_height"]})
        if ex and ex.get("status") == "OK":
            meta["requires_goal_destruction"] = ex["requires_goal_destruction"]
        out.append(ST.make_case(c, "dev", DP.DOMAIN_TYPED, L, kind, **meta))
    return out


def traj_digest(trajs):
    return hashlib.sha256(json.dumps([[t["tid"], t["states"], t["actions"], t["astar"]] for t in trajs], sort_keys=True).encode()).hexdigest()


def update_batches(trajs, seed):
    """Identical to ``PddlTrainer.update_batches`` (a pure function of the trajectory list and the epoch seed)."""
    order = list(range(len(trajs)))
    random.Random(seed).shuffle(order)
    return [[trajs[i] for i in order[k:k + BST.BATCH]] for k in range(0, len(order), BST.BATCH)]


def schedule_ledger(trajs, updates=BUDGET["updates"]):
    """Deterministic account of what every update shows: trajectory ids, number of decisions, weight mass. Independent of the model: D0, C0 and a T1 of the same data see exactly this."""
    nb = len(update_batches(trajs, 0))
    rows = []
    for u in range(updates):
        ep, off = divmod(u, nb)
        b = update_batches(trajs, ep)[off]
        rows.append({"update": u + 1, "epoch": ep, "batch_trajectories": len(b), "decisions": sum(len(t["actions"]) for t in b), "tid_digest": hashlib.sha1(",".join(t["tid"] for t in b).encode()).hexdigest()[:12]})
    return rows


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def train_loop_v2(out_dir, mode, device, cases, trajs, labels, log=print, max_updates=None, trainer_factory=None):
    """``stages.train_loop`` with the model factory of this card. Resumable at 100-update boundaries. ``max_updates`` is only used by fixtures; ``trainer_factory`` (T1 only) builds the trainer, default = the original one."""
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    m = make_model_v2(mode, device, seed=BUDGET["init_seed"])
    tr = trainer_factory(m, cases, labels, device) if trainer_factory else PT.PddlTrainer(m, cases, labels, device, lr=BUDGET["lr"], chunk=BUDGET["chunk_decisions"])
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
            ckpts[str(u)] = {"path": str(p), "sha256": sha_file(p), "bytes": p.stat().st_size}
    nb = len(tr.update_batches(trajs, 0))
    init_new = [p.detach().clone().cpu() for p in m.new_params()]
    target = BUDGET["updates"] if max_updates is None else max_updates
    while update < target:
        ep, off = divmod(update, nb)
        batches = tr.update_batches(trajs, ep)
        r = tr.step(batches[off])
        update += 1
        r["update"], r["epoch"], r["wall_seconds"] = update, ep, round(wall_prev + time.time() - t0, 1)
        rows.append(r)
        if update % 100 == 0 or update == 1:
            log("[compact-train] %s update %d %s" % (mode, update, {k: round(v, 5) for k, v in r.items() if isinstance(v, float)}), flush=True)
        if update in BUDGET["checkpoints"] and max_updates is None:
            p = out_dir / "checkpoints" / ("u%05d.pt" % update)
            save_checkpoint(p, m, tr.optimizer, {"mode": mode, "update": update})
            ckpts[str(update)] = {"path": str(p), "sha256": sha_file(p), "bytes": p.stat().st_size}
        if (update % 100 == 0 or update == target) and max_updates is None:
            tmp = out_dir / "resume_last.pt.tmp"
            torch.save({"model": m.state_dict(), "optimizer": tr.optimizer.state_dict(), "rows": rows, "update": update, "attempts": attempts, "wall": wall_prev + time.time() - t0}, tmp)
            os.replace(tmp, resume)
    new_now = [p.detach().cpu() for p in m.new_params()]
    acct = {"mode": mode, "optimizer_steps": update, "decision_samples_shown": sum(r["decisions"] for r in rows), "nan_events": tr.nan_events, "attempts": attempts, "wall_seconds": round(wall_prev + time.time() - t0, 1),
            "parameters_trainable": count_trainable(m), "parameters_new_module": count_new(m), "checkpoints": ckpts, "batches_per_epoch": nb, "trajectories": len(trajs), "final_update_row": dict(rows[-1]),
            "new_module_param_change_l2": float(sum(((a - b) ** 2).sum() for a, b in zip(new_now, init_new)) ** 0.5) if new_now else 0.0, "budget": BUDGET, "init_seed": BUDGET["init_seed"],
            "train_digest": traj_digest(trajs), "device": str(device), "torch": torch.__version__}
    return acct, rows, m


def load_model_v2(mode, path, device):
    m = make_model_v2(mode, device, seed=BUDGET["init_seed"])
    load_checkpoint(path, m)
    m.eval()
    return m


def devsel(out_dir, mode, device, dev, log=print):
    """Original rule on Dev24 (policy rollouts, one-step executor): most successes, then smaller failure-penalised cost ratio, then the earlier checkpoint. Evaluations are resumable per checkpoint file."""
    out_dir = Path(out_dir)
    acct = rj(out_dir / "training_accounting.json")
    rows, table = {}, []
    for u, info in acct["checkpoints"].items():
        assert sha_file(info["path"]) == info["sha256"], "checkpoint hash mismatch"
        m = load_model_v2(mode, info["path"], device)
        res = ST.run_set(out_dir / ("dev_u%s.jsonl" % u), dev, m, "one_step")
        rows[u] = ST.summarise(res)
        table.append({"model": mode, "update": u, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in rows[u].items()}})
        del m
    best = ST.select_checkpoint(rows)
    sel = {"model": mode, "selected_update": best, "checkpoint": acct["checkpoints"][best], "dev_by_update": rows, "rule": "most dev successes, then smaller failure-penalised cost ratio, then earlier checkpoint", "dev_cases": len(dev)}
    (out_dir / "selection.json").write_text(json.dumps(sel, indent=1, sort_keys=True) + "\n")
    write_csv(out_dir / "dev_by_checkpoint.csv", table)
    return sel
