"""CP-DISR-C1-BW-IMITATION-A02: uniform exact-planner imitation of B2-CACHED, QMARK and ASNET-READOUT (Amendment A02, Option A).

The three frozen policies (``c1_blocksworld_policies``) are trained from scratch on the SAME planner-labelled datasets D0 / D1 / D2 with the SAME loss and the SAME fixed schedule;
only the model parameters, forward results and optimizer state differ between the methods.

    loss      L = - log sum_{a in A*(s,g)} pi(a | s)       illegal actions masked; A* = EVERY legal a with d*(T(s,a),g) = d*(s,g) - 1 (never a single tie-break action)
    schedule  100 epochs on D0 -> deterministic rollouts of the 144 train cases by all three methods -> D1 = D0 + B2 + QMARK + ASNET rollouts -> 50 epochs
              -> rollouts -> D2 = D1 + B2 + QMARK + ASNET rollouts -> 50 epochs; the last epoch of round 2 is the only final checkpoint. Nothing adapts to any metric.
    frozen    ``v_head`` and ``q_head`` (and every module that cannot influence a policy logit) are not trained; no value / Q target, reward, entropy bonus or PPO clipping exists here.

Nothing in this module reads dev36 or an evaluation split: training cases come from ``launch.open_training_split`` (the train rows only are used).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import torch

from . import state as S
from .environment import BwEpisode
from .planner import Solver

TAGS = ("B2", "QMARK", "ASNET")
TAG_OF = {"B2-CACHED": "B2", "B1-K+QMARK-BW": "QMARK", "ASNET-READOUT": "ASNET"}
ROW_KIND = {"B2": "b2", "QMARK": "qmark", "ASNET": "asnet"}
OPT = {"lr": 0.0003, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "grad_clip": 0.5, "trajectory_batch_size": 32}
STAGES = ({"name": "stage0", "epochs": 100, "shuffle_seed": 0, "dataset": "D0"}, {"name": "round1", "epochs": 50, "shuffle_seed": 1, "dataset": "D1"},
          {"name": "round2", "epochs": 50, "shuffle_seed": 2, "dataset": "D2"})
TRAINABLE_MODULES = ("encoder", "observation", "gru", "candidate", "contract", "base", "query_projection")
FROZEN_MODULES = ("v_head", "q_head", "prior", "set_encoder", "b0_fuse", "effect_readout", "effect_fuse", "cat_proj")
DECISION_CHUNK = 256                       # decisions per backward chunk (memory only; the loss is a fixed weighted sum, so chunking does not change it)
LOSS_NAME = "NEG_LOG_SUM_PROB_OVER_FULL_OPTIMAL_ACTION_SET"
FORBIDDEN_CHECKPOINT = re.compile(r"(blocksworld_main_v1/20\d{6}T\d{6}Z_[0-9a-f]{8}/)|(/n_\d{7}\.pt$)")


class ImitationError(RuntimeError):
    pass


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ guards (A02 section 13 negative tests)
def validate_schedule(stages, opt):
    """The schedule and optimiser are frozen constants: any other value (epochs, rounds, lr, ...) is refused."""
    if [dict(s) for s in stages] != [dict(s) for s in STAGES] or dict(opt) != OPT:
        raise ImitationError("the imitation schedule / optimiser is frozen (100+50+50 epochs, two aggregation rounds); it may not be tuned")


def refuse_checkpoint_init(cfg):
    """From scratch only: no PPO checkpoint, no N=32768 (or any other mid) checkpoint may be loaded."""
    for key, value in cfg.items():
        if key in ("init_checkpoint", "resume", "checkpoint", "init_from"):
            raise ImitationError("checkpoint loading is not allowed (config key %s)" % key)
        if isinstance(value, str) and (FORBIDDEN_CHECKPOINT.search(value) or value.endswith(".pt")):
            raise ImitationError("checkpoint loading is not allowed (config key %s = %s)" % (key, value))


def validate_identical_supervision(cfgs):
    """All three methods must receive the identical D0 hash, loss, optimiser, schedule and init seed (no planner supervision for a subset of the methods)."""
    keys = ("d0_sha256", "loss", "optimizer", "stages", "seed", "train_dev_split_sha256", "aggregation_rounds")
    ref = {k: cfgs[0].get(k) for k in keys}
    for c in cfgs[1:]:
        if {k: c.get(k) for k in keys} != ref:
            raise ImitationError("the three methods must use the same supervision, loss and schedule")
    if sorted(c["method"] for c in cfgs) != sorted(TAG_OF):
        raise ImitationError("exactly B2-CACHED, B1-K+QMARK-BW and ASNET-READOUT are trained")


def load_train_cases(path):
    """TRAIN rows of the train/dev file (the only file a training path may open; every A0 / A1 / A2 / B file is refused). The dev rows are not used."""
    from .launch import open_training_split
    from .train import case_from_json
    doc = open_training_split(path)
    return [case_from_json(c) for c in doc["train"]], float(doc["half_life"])


# ------------------------------------------------------------------------------------------------ labels and datasets
def snapshot_at(case, state, t):
    ep = BwEpisode(case)
    ep.state = tuple(state)
    ep.step_index = t
    ep.previous_ok = t > 0
    return ep.snapshot()


def label_state(solver, case, state):
    """(remaining optimal length, sorted ids of ALL optimal legal actions) at ``state``: A*(s,g) = {a legal : d*(T(s,a),g) = d*(s,g) - 1}."""
    L, acts = solver.optimal_actions(tuple(state), case.problem.goal)
    return L, [S.action_id(case.problem.names, a) for a in acts]


def _traj(source, case, states, actions, astar, remaining, success):
    return {"tid": "%s:%s" % (source, case.case_id), "source": source, "case_id": case.case_id, "n_blocks": case.problem.n, "step_cap": case.step_cap, "states": [list(s) for s in states],
            "actions": actions, "astar": astar, "remaining": remaining, "success": bool(success)}


def expert_trajectory(case, solver, source="expert"):
    """One canonical optimal trajectory (fixed-lexicographic first optimal action); every visited state carries the FULL optimal-action set."""
    s = case.problem.init
    states, actions, astar, remaining = [s], [], [], []
    while not S.goal_satisfied(case.problem, s):
        L, acts = solver.optimal_actions(s, case.problem.goal)
        astar.append([S.action_id(case.problem.names, a) for a in acts])
        remaining.append(L)
        actions.append(S.action_id(case.problem.names, acts[0]))
        s = S.apply(s, acts[0])
        states.append(s)
    return _traj(source, case, states, actions, astar, remaining, True)


@torch.no_grad()
def rollout_trajectory(policy, case, solver, source):
    """Deterministic argmax rollout of the CURRENT policy up to the frozen step cap; every visited decision state is labelled by the exact planner."""
    ep = BwEpisode(case)
    hidden = policy.initial_hidden()
    states, actions, astar, remaining = [ep.state], [], [], []
    while not ep.done:
        snap = ep.snapshot()
        L, ids = label_state(solver, case, ep.state)
        out = policy(snap, hidden)
        hidden = out.hidden
        sel, _i = out.select(True)
        astar.append(ids)
        remaining.append(L)
        actions.append(sel)
        ep.step(sel)
        states.append(ep.state)
    return _traj(source, case, states, actions, astar, remaining, ep.success)


def build_d0(cases):
    solver = Solver()
    return [expert_trajectory(c, solver) for c in cases]


def aggregate(base, rollouts_by_tag):
    """D_{r+1} = D_r + B2 trajectories + QMARK trajectories + ASNET trajectories, in this fixed order (identical for every method)."""
    out = list(base)
    for tag in TAGS:
        out.extend(rollouts_by_tag[tag])
    return out


def verify_labels(trajs, cases, solver=None):
    """Recompute A* of every stored decision with the exact planner (raises if a stored set differs, e.g. a single tie-break action replaced the full set)."""
    solver = solver or Solver()
    by_id = {c.case_id: c for c in cases}
    for t in trajs:
        case = by_id[t["case_id"]]
        if not (len(t["states"]) == len(t["actions"]) + 1 == len(t["astar"]) + 1) or not t["actions"]:
            raise ImitationError("malformed trajectory %s" % t["tid"])
        for s, ids in zip(t["states"][:-1], t["astar"]):
            _L, truth = label_state(solver, case, s)
            if ids != truth or not truth:
                raise ImitationError("A* of a stored decision differs from the exact planner set (%s)" % t["tid"])
    return True


def decision_weights(trajs, batch_trajectories=None):
    """Per-decision weights of a batch: 1 / (length * number of trajectories). Every trajectory has the SAME total weight whatever its length or whether it loops."""
    B = batch_trajectories or len(trajs)
    return [[1.0 / (len(t["actions"]) * B)] * len(t["actions"]) for t in trajs]


def dataset_identity(trajs):
    decisions = sum(len(t["actions"]) for t in trajs)
    lengths = [len(t["actions"]) for t in trajs]
    by_source = {}
    for t in trajs:
        by_source[t["source"]] = by_source.get(t["source"], 0) + 1
    uniq = {(t["case_id"], tuple(s)) for t in trajs for s in t["states"][:-1]}
    cycles = 0
    for t in trajs:
        seen = set()
        for s, a in zip(t["states"][:-1], t["actions"]):
            key = (tuple(s), a)
            if key in seen:
                cycles += 1
                break
            seen.add(key)
    sizes = [len(a) for t in trajs for a in t["astar"]]
    return {"sha256": digest([[t["tid"], t["states"], t["actions"], t["astar"]] for t in trajs]), "trajectories": len(trajs), "decisions": decisions, "unique_goal_state_pairs": len(uniq),
            "source_counts": by_source, "mean_trajectory_length": decisions / len(trajs) if trajs else 0, "max_trajectory_length": max(lengths) if lengths else 0, "cycle_containing_trajectories": cycles,
            "planner_label_coverage": 1.0 if sizes and min(sizes) >= 1 else 0.0, "mean_optimal_set_size": sum(sizes) / len(sizes) if sizes else 0,
            "decisions_with_multiple_optimal_actions": sum(1 for n in sizes if n > 1)}


# ------------------------------------------------------------------------------------------------ the policy-logit kernel (grouped; same arithmetic as the production heads)
def _group_logits(policy, kind, snaps, zos):
    """(S, n) masked logits of snapshots that share ONE template, from the policy's own modules (same arithmetic as ``batched_outputs``; V / Q heads are not evaluated)."""
    st = policy._static(snaps[0].template)
    dev = zos.device
    S_g = len(snaps)
    n = len(snaps[0].candidate_ids)
    codes_list, qnodes, legal_per, offsets, graph_off = [], [], [], [], 0
    for sn in snaps:
        legal = [i for i, m in enumerate(sn.mask) if m]
        legal_per.append(legal)
        codes0 = policy._codes(st, sn.facts.values)
        rows_idx = [st.action_row[sn.candidate_ids[i]] for i in legal]
        offsets.append(graph_off)
        if kind == "b2":
            codes_list.append(torch.cat((codes0.unsqueeze(0), policy._successor_codes(st, codes0, rows_idx)), 0))
        elif kind == "qmark":
            codes_list.append(codes0.unsqueeze(0).repeat(len(legal) + 1, 1, 1))
            qnodes += [-1] + [int(st.action_pos[r]) for r in rows_idx]
        else:
            codes_list.append(codes0.unsqueeze(0))
        graph_off += codes_list[-1].shape[0]
    z, H = policy._encode(st, torch.cat(codes_list, 0), torch.tensor(qnodes, device=dev) if kind == "qmark" else None)
    rows_all, zk_means, seg, flat_pos, feats = [], [], [], [], []
    for si, sn in enumerate(snaps):
        legal, off = legal_per[si], offsets[si]
        K = len(legal)
        zk = z[off]
        zk_means.append(zk.mean(0))
        if kind == "asnet":
            ridx = torch.tensor([int(st.action_pos[st.action_row[sn.candidate_ids[i]]]) for i in legal], device=dev)
            rows_all.append(H[off, ridx].unsqueeze(1))
        else:
            rows_all.append(z[off + 1: off + 1 + K].float() - zk.float().unsqueeze(0))
        seg += [si] * K
        flat_pos += [si * n + i for i in legal]
        feats += [sn.candidate_features[i] for i in legal]
    rows_all = torch.cat(rows_all, 0)
    seg_t = torch.tensor(seg, device=dev)
    zk_mean = torch.stack(zk_means)
    ca = policy.candidate(torch.tensor(feats, device=dev, dtype=zos.dtype))
    ctx = torch.cat((zos[seg_t], ca, zk_mean[seg_t]), dim=-1)
    uk_legal = policy._phi_batch(ctx, rows_all)
    uk_full = zos.new_zeros((S_g * n, 128)).index_copy(0, torch.tensor(flat_pos, device=dev), uk_legal).view(S_g, n, 128)
    mask = torch.tensor([sn.mask for sn in snaps], device=dev, dtype=torch.bool)
    return policy.base(uk_full).squeeze(-1).masked_fill(~mask, -torch.inf)


def imitation_nll(logits, astar_mask):
    """-log sum_{a in A*} pi(a|s) per row (``logits`` carry -inf at illegal actions; ``astar_mask`` marks every optimal action as positive)."""
    return torch.logsumexp(logits, dim=-1) - torch.logsumexp(logits.masked_fill(~astar_mask, -torch.inf), dim=-1)


def batch_losses(policy, kind, snaps, zos, astar_ids):
    """(per-decision NLL (D,), argmax-in-A* flags (D,)); snapshots are grouped by template for the encoder."""
    groups = {}
    for i, sn in enumerate(snaps):
        groups.setdefault(id(sn.template), []).append(i)
    nll, hit = [None] * len(snaps), [None] * len(snaps)
    for idx in groups.values():
        logits = _group_logits(policy, kind, [snaps[i] for i in idx], zos[idx])
        am = torch.zeros_like(logits, dtype=torch.bool)
        for j, i in enumerate(idx):
            ids = snaps[i].candidate_ids
            for a in astar_ids[i]:
                am[j, ids.index(a)] = True
        loss = imitation_nll(logits, am)
        flags = am.gather(1, logits.argmax(-1).unsqueeze(1)).squeeze(1)
        for j, i in enumerate(idx):
            nll[i], hit[i] = loss[j], flags[j]
    return torch.stack(nll), torch.stack(hit)


def trajectory_hidden(policy, per, device):
    """Hidden states z_o for a batch of trajectories: zero at every trajectory start, carried along it; padded steps are never read. Returns a list (per trajectory) of (len, 128)."""
    B = len(per)
    T = max(len(p) for p in per)
    obs = torch.zeros((T, B, 48), device=device)
    for i, p in enumerate(per):
        for t, sn in enumerate(p):
            obs[t, i] = torch.as_tensor(sn.base_input, device=device)
    h = torch.zeros((B, 128), device=device)
    hs = []
    for t in range(T):
        h = policy.gru(torch.relu(policy.observation(obs[t])), h)
        hs.append(h)
    return [torch.stack([hs[t][i] for t in range(len(per[i]))]) for i in range(B)]


def trainable_parameters(policy):
    """Parameters of the modules that influence a policy logit; everything else (V / Q heads, ...) is frozen. Every child module must be classified."""
    params = []
    for name, module in policy.named_children():
        if name in TRAINABLE_MODULES:
            train = True
        elif name in FROZEN_MODULES:
            train = False
        else:
            raise ImitationError("unclassified module %s" % name)
        for p in module.parameters():
            p.requires_grad_(train)
        if train:
            params += list(module.parameters())
    return params


def module_digest(policy, names):
    h = hashlib.sha256()
    for name, module in policy.named_children():
        if name in names:
            for k, v in module.state_dict().items():
                h.update(k.encode())
                h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def derived_seed(label, seed=0):
    return int(hashlib.sha256(("A02:%s:%d" % (label, seed)).encode()).hexdigest()[:8], 16)


def make_imitation_policy(method, device, seed=0):
    """From scratch: the shared production modules come from the same init seed (identical for the three methods); architecture-specific parameters (QMARK ``query_projection``)
    are re-initialised from a deterministic derived seed."""
    from ..c1_blocksworld_policies import make_policy
    policy = make_policy(method, device, seed=seed)
    if hasattr(policy, "query_projection"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derived_seed("query_projection", seed))
            policy.query_projection.reset_parameters()
        policy.query_projection.to(device)
    return policy


class ImitationTrainer:
    def __init__(self, policy, kind, cases, device):
        self.policy, self.kind, self.device = policy, kind, device
        self.cases = {c.case_id: c for c in cases}
        self.params = trainable_parameters(policy)
        self.optimizer = torch.optim.Adam(self.params, lr=OPT["lr"], betas=tuple(OPT["betas"]), eps=OPT["eps"], weight_decay=OPT["weight_decay"])
        self.steps = 0
        self.nan_events = 0
        self._snaps = {}

    def snapshots(self, traj):
        got = self._snaps.get(traj["tid"])
        if got is None:
            case = self.cases[traj["case_id"]]
            got = [snapshot_at(case, s, t) for t, s in enumerate(traj["states"][:-1])]
            self._snaps[traj["tid"]] = got
        return got

    def step(self, trajs):
        """One optimizer step on a batch of trajectories. Loss = mean over the batch's trajectories of the per-trajectory mean decision NLL (equal weight per trajectory)."""
        pol = self.policy
        pol.train()
        per = [self.snapshots(t) for t in trajs]
        zo_list = trajectory_hidden(pol, per, self.device)
        zos = torch.cat(zo_list, 0)
        snaps = [sn for p in per for sn in p]
        astar = [a for t in trajs for a in t["astar"]]
        w = torch.tensor([x for ws in decision_weights(trajs) for x in ws], device=self.device)
        zleaf = zos.detach().requires_grad_(True)
        self.optimizer.zero_grad()
        total, mass, hits = 0.0, 0.0, 0.0
        for a in range(0, len(snaps), DECISION_CHUNK):
            b = min(a + DECISION_CHUNK, len(snaps))
            nll, hit = batch_losses(pol, self.kind, snaps[a:b], zleaf[a:b], astar[a:b])
            part = (w[a:b] * nll).sum()
            if not torch.isfinite(part):
                self.nan_events += 1
                raise ImitationError("non-finite imitation loss")
            part.backward()
            total += float(part)
            mass += float((w[a:b] * torch.exp(-nll.detach())).sum())
            hits += float((w[a:b] * hit.float()).sum())
        zos.backward(zleaf.grad)
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, OPT["grad_clip"], error_if_nonfinite=True))
        self.optimizer.step()
        self.steps += 1
        return {"loss": total, "decisions": len(snaps), "mass": mass, "argmax_in_optimal": hits, "grad_norm": norm}

    def epoch(self, trajs, seed):
        order = list(range(len(trajs)))
        random.Random(seed).shuffle(order)
        bs = OPT["trajectory_batch_size"]
        rows = [self.step([trajs[i] for i in order[k:k + bs]]) for k in range(0, len(order), bs)]
        n = len(rows)
        return {"batches": n, "loss": sum(r["loss"] for r in rows) / n, "mass": sum(r["mass"] for r in rows) / n, "argmax_in_optimal": sum(r["argmax_in_optimal"] for r in rows) / n,
                "grad_norm": sum(r["grad_norm"] for r in rows) / n, "decisions": sum(r["decisions"] for r in rows)}


# ------------------------------------------------------------------------------------------------ the complete schedule of one method
def write_json_atomic(path, doc):
    path = Path(path)
    tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
    tmp.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def wait_for(paths, poll=10.0, log=print):
    """Barrier: block until every file exists (no timeout, no adaptive behaviour)."""
    announced = False
    while not all(Path(p).is_file() for p in paths):
        if not announced:
            log("[il] waiting for %s" % [str(p) for p in paths if not Path(p).is_file()])
            announced = True
        time.sleep(poll)


def check_shared_hash(ds, name, sha):
    """The first run to reach a dataset records its hash; every other run must reproduce it exactly (D0 / D1 / D2 are identical for the three methods)."""
    hash_file = Path(ds) / ("%s_sha256.txt" % name)
    if not hash_file.exists():
        tmp = hash_file.with_name(hash_file.name + ".%d.tmp" % os.getpid())
        tmp.write_text(sha + "\n")
        os.replace(tmp, hash_file)
    if hash_file.read_text().strip() != sha:
        raise ImitationError("%s differs between the method runs" % name)


def run_imitation(cfg, log=print):
    """One method's complete A02 schedule. The three parallel runs synchronise only through the rollout files in the shared ``dataset_dir``; each run rebuilds D1 / D2 from those
    files and refuses to continue if its dataset hash differs from the hash of another run."""
    from ..rl import set_suite_half_life
    from ..torch_rl import save_checkpoint
    validate_schedule(cfg["stages"], cfg["optimizer"])
    refuse_checkpoint_init({k: v for k, v in cfg.items() if k not in ("out_dir", "dataset_dir", "d0_path", "train_dev_split")})
    out, ds = Path(cfg["out_dir"]), Path(cfg["dataset_dir"])
    (out / "checkpoints").mkdir(parents=True, exist_ok=False)
    device = torch.device(cfg.get("device", "cpu"))
    cases, H = load_train_cases(cfg["train_dev_split"])
    set_suite_half_life(H)
    d0 = json.loads(Path(cfg["d0_path"]).read_text())
    d0_id = dataset_identity(d0)
    if d0_id["sha256"] != cfg["d0_sha256"]:
        raise ImitationError("D0 differs from the frozen D0")
    check_shared_hash(ds, "D0", d0_id["sha256"])
    tag = TAG_OF[cfg["method"]]
    policy = make_imitation_policy(cfg["method"], device, seed=int(cfg["seed"]))
    frozen_before = module_digest(policy, FROZEN_MODULES)
    trainer = ImitationTrainer(policy, ROW_KIND[tag], cases, device)
    solver = Solver()
    epoch_rows, stage_ids, ckpts = [], {"D0": d0_id}, {}
    t0 = time.time()
    data = d0
    for si, stage in enumerate(STAGES):
        if si > 0:
            policy.eval()
            mine = [rollout_trajectory(policy, c, solver, "%s_r%d" % (tag, si)) for c in cases]
            write_json_atomic(ds / ("rollouts_round%d_%s.json" % (si, tag)), mine)
            wait_for([ds / ("rollouts_round%d_%s.json" % (si, t)) for t in TAGS], log=log)
            roll = {t: json.loads((ds / ("rollouts_round%d_%s.json" % (si, t))).read_text()) for t in TAGS}
            data = aggregate(data, roll)
            ident = dataset_identity(data)
            check_shared_hash(ds, "D%d" % si, ident["sha256"])
            write_json_atomic(out / ("dataset_D%d_identity.json" % si), ident)
            stage_ids["D%d" % si] = ident
            log("[il] %s D%d: %d trajectories, %d decisions, sha %s" % (tag, si, ident["trajectories"], ident["decisions"], ident["sha256"][:12]))
        for ep in range(stage["epochs"]):
            row = trainer.epoch(data, stage["shuffle_seed"] * 100003 + ep)
            row.update({"stage": stage["name"], "epoch": ep + 1, "optimizer_steps": trainer.steps, "wall_seconds": round(time.time() - t0, 1)})
            epoch_rows.append(row)
            if (ep + 1) % 10 == 0 or ep == 0:
                log("[il] %s %s epoch %d loss %.4f mass %.3f argmax-in-A* %.3f wall %.0fs" % (tag, stage["name"], ep + 1, row["loss"], row["mass"], row["argmax_in_optimal"], time.time() - t0))
            with (out / "imitation_epochs.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
                w.writeheader()
                w.writerows(epoch_rows)
        name = "final" if si == len(STAGES) - 1 else stage["name"]
        path = out / "checkpoints" / ("%s.pt" % name)
        save_checkpoint(path, policy, trainer.optimizer, {"method": cfg["method"], "stage": stage["name"], "optimizer_steps": trainer.steps})
        ckpts[stage["name"]] = {"path": str(path), "sha256": sha256_file(path)}
    if module_digest(policy, FROZEN_MODULES) != frozen_before:
        raise ImitationError("a frozen module changed during training")
    acct = {"method": cfg["method"], "tag": tag, "seed": int(cfg["seed"]), "optimizer_steps": trainer.steps, "epochs": sum(s["epochs"] for s in STAGES), "aggregation_rounds": 2,
            "nan_events": trainer.nan_events, "wall_seconds": round(time.time() - t0, 1), "dataset_identity": stage_ids, "final_epoch": epoch_rows[-1], "frozen_modules_unchanged": True,
            "checkpoints": ckpts, "final_checkpoint": ckpts["round2"]["path"], "final_checkpoint_sha256": ckpts["round2"]["sha256"], "stop_reason": "FIXED_SCHEDULE_100_50_50", "hard_fail": None,
            "loss": LOSS_NAME, "stage_end_epoch_rows": {s["name"]: [r for r in epoch_rows if r["stage"] == s["name"]][-1] for s in STAGES}}
    (out / "imitation_training_accounting.json").write_text(json.dumps(acct, indent=1, default=str) + "\n", encoding="utf-8")
    return acct
