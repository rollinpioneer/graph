"""C1-BW-GP-MINIMAL-ATTRIBUTION-V1 (runbook v2): two matched trainings (B2-RANK, M1-GOAL), a destruction-stratified colour-twin confirmation set and nine zero-training / trained conditions.

* B2-RANK  = the old M0 training path (``imitation.ImitationTrainer``: GRU through the whole trajectory, B2 rows, 1e-4 for every trainable module) + the pairwise rank loss (rank_weight 0 -> 1).
* M1-GOAL  = the old M1 (global aggregation, 386-d per-goal features, phi/rho heads, successor value difference, no GRU/time) with the encoder's goal-mark channel RESTORED (production marks).
Everything else is taken unchanged from ``goal_progress`` / ``a04p_controls``.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch

from . import a04p_controls as AC
from . import eval_a03 as E
from . import goal_progress as GP
from . import imitation as I
from . import metrics as M
from . import state as S
from .environment import BwEpisode

CARD = "C1-BW-GP-MINIMAL-ATTRIBUTION-V1"
BASE_COMMIT = "b625cfc7f4c83f2c1cf054eef58538d77363c1d5"
BRANCH = "codex/cp-disr-c1-bw-gp-minimal-controls-v1"
RUNBOOK_REL = "docs/c1_blocksworld/CP_DISR_C1_GP_Minimal_Attribution_Runbook_v2.md"
GPA_REL = E.BASE_REL + "/gp_minimal_attribution_v1"
GP_ROOT = E.BASE_REL + "/goal_progress_v1/train_confirm/20261006T095758Z_90758ab1"
A04P_ROOT = E.BASE_REL + "/goal_progress_v1/a04p/20261006T094835Z_90758ab1"
OLD_ROOTS = E.OLD_ROOTS + (E.BASE_REL + "/evaluation_a03/20261006T070810Z_a8f9593a", GP_ROOT, A04P_ROOT)
CONFIRM_REL = "configs/splits/c1_bw_gp_attr_confirm112_v2.json"
NAMESPACE = "C1-BW-GPA-REVIEW-v2-20261006"
OLD = {"M0": {"run_id": "R-C1-GP-M0-0", "path": GP_ROOT + "/runs/R-C1-GP-M0-0/checkpoints/final.pt", "sha256": "973925b522af2ea53d621b68b96005d23fae45a11c45423d94c0a1a9012aa82f", "bytes": 19860291},
       "M1": {"run_id": "R-C1-GP-M1-0", "path": GP_ROOT + "/runs/R-C1-GP-M1-0/checkpoints/final.pt", "sha256": "8144d0ec6227d0eb8ae5403fe0bf72dacf351ff6a1bbf1051e5adb33ca2ae94f", "bytes": 17880803}}
D_TRAIN = {"file_sha256": "387e66309c51195ad017b2b90e9f2742090457d108a353f34159acd8f496bf78", "semantic_sha256": "2b3380200d8aaf719871a4b2ae446bcfb1ed992dfb821cd6fb2f0bbb12d31d69",
           "rank_labels_file_sha256": "5486cea84eeca5efa8cbd73722cd34aa2609b44a369b95f398b9d3cdd5be50f6", "rank_labels_semantic_sha256": "24b5e335a727ce8328947451a55f1b6bdb5c686a0715a74f8b17b77f8da90064"}
M1_HEAD_SHA = "57ab0cf2512b02f95dee91fe73a9980471e5310548e7e4d1840340758f887f37"
RUN_IDS = {"B2RANK": "R-C1-GPA-B2-RANK-0", "M1GOAL": "R-C1-GPA-M1-GOAL-0"}
CONDITIONS = ("C0", "M0", "M1", "B2RANK", "M1GOAL", "G1", "C3", "G1C3", "M1C3")
STRUCTURE_GROUPS = (("T6-2", 6, (3, 2, 1)), ("T6-3", 6, (2, 2, 2)), ("T7-2", 7, (3, 2, 1, 1)), ("T8-3", 8, (3, 2, 2, 1)))
PER_CELL = 8
CELL_MAX_ATTEMPTS = 2000
COLOR_MAX_ATTEMPTS = 10000
EPOCHS, BATCH, SHUFFLE_SEED, LR_ALL = 100, 32, 0, 1e-4
FROZEN = tuple(dict.fromkeys(E.FROZEN_SOURCES + ("src/cp_disr/blocksworld/eval_a03.py", "src/cp_disr/blocksworld/goal_progress.py", "src/cp_disr/blocksworld/a04p_controls.py", "src/cp_disr/blocksworld/goal_probe.py",
                                                 "src/cp_disr/blocksworld/a04p_registry.py", "scripts/c1_bw_goal_progress.py", "scripts/c1_bw_a04p.py")))


# ------------------------------------------------------------------------------------------------ B2-RANK
class RankTrainer(I.ImitationTrainer):
    """Old M0 trainer + rank term. ``rank_weight = 0`` must reproduce the old M0 step (checked by a fixture before training)."""

    def __init__(self, policy, kind, cases, device, rank_labels, rank_weight, lr=LR_ALL):
        super().__init__(policy, kind, cases, device)
        self.rank_labels, self.rank_weight = rank_labels, rank_weight
        self.optimizer = torch.optim.Adam(self.params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)          # optimizer reset, 1e-4 for every trainable parameter

    def _losses(self, snaps, zos, astar, keys):
        groups = {}
        for i, sn in enumerate(snaps):
            groups.setdefault(id(sn.template), []).append(i)
        nll, rank, hit = [None] * len(snaps), [None] * len(snaps), [None] * len(snaps)
        for idx in groups.values():
            sub = [snaps[i] for i in idx]
            logits = I._group_logits(self.policy, self.kind, sub, zos[idx])
            ids = sub[0].candidate_ids
            am = torch.zeros_like(logits, dtype=torch.bool)
            dist = torch.zeros_like(logits)
            lm = torch.tensor([s.mask for s in sub], device=logits.device, dtype=torch.bool)
            for j, i in enumerate(idx):
                for a in astar[i]:
                    am[j, ids.index(a)] = True
                for a, d in self.rank_labels[keys[i]].items():
                    dist[j, ids.index(a)] = d
            nl, rk, _c = GP.nll_and_rank(logits, am, dist, lm)
            flags = am.gather(1, logits.argmax(-1).unsqueeze(1)).squeeze(1)
            for j, i in enumerate(idx):
                nll[i], rank[i], hit[i] = nl[j], rk[j], flags[j]
        return torch.stack(nll), torch.stack(rank), torch.stack(hit)

    def step(self, trajs):
        pol = self.policy
        pol.train()
        per = [self.snapshots(t) for t in trajs]
        zo_list = I.trajectory_hidden(pol, per, self.device)
        zos = torch.cat(zo_list, 0)
        snaps = [sn for p in per for sn in p]
        astar = [a for t in trajs for a in t["astar"]]
        keys = [GP.rank_key(t["case_id"], s) for t in trajs for s in t["states"][:-1]]
        w = torch.tensor([x for ws in I.decision_weights(trajs) for x in ws], device=self.device)
        zleaf = zos.detach().requires_grad_(True)
        self.optimizer.zero_grad()
        tot = {"loss": 0.0, "nll": 0.0, "rank": 0.0, "mass": 0.0, "hit": 0.0}
        for a in range(0, len(snaps), I.DECISION_CHUNK):
            b = min(a + I.DECISION_CHUNK, len(snaps))
            nll, rank, hit = self._losses(snaps[a:b], zleaf[a:b], astar[a:b], keys[a:b])
            part = (w[a:b] * (nll + self.rank_weight * rank)).sum()
            if not torch.isfinite(part):
                self.nan_events += 1
                raise I.ImitationError("non-finite loss")
            part.backward()
            tot["loss"] += float(part)
            tot["nll"] += float((w[a:b] * nll).sum())
            tot["rank"] += float((w[a:b] * rank).sum())
            tot["mass"] += float((w[a:b] * torch.exp(-nll.detach())).sum())
            tot["hit"] += float((w[a:b] * hit.float()).sum())
        zos.backward(zleaf.grad)
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, I.OPT["grad_clip"], error_if_nonfinite=True))
        self.optimizer.step()
        self.steps += 1
        return {**tot, "grad_norm": norm, "decisions": len(snaps)}

    def epoch(self, trajs, seed):
        order = list(range(len(trajs)))
        __import__("random").Random(seed).shuffle(order)
        rows = [self.step([trajs[i] for i in order[k:k + BATCH]]) for k in range(0, len(order), BATCH)]
        n = len(rows)
        return {"batches": n, "decisions": sum(r["decisions"] for r in rows), **{k: sum(r[k] for r in rows) / n for k in ("loss", "nll", "rank", "mass", "hit", "grad_norm")}}


# ------------------------------------------------------------------------------------------------ M1-GOAL
class GoalProgressModelGoal(GP.GoalProgressModel):
    """Old M1 with the encoder's goal-mark channel restored: ``prop_goal_sign`` enters the message passing for the current state AND every nominal successor.
    ``goal_marks="zero"`` is the old behaviour (equivalence fixture)."""

    def __init__(self, base, mode="global", goal_marks="production", seed=GP.HEAD_SEED):
        super().__init__(base, mode, seed)
        assert goal_marks in ("zero", "production")
        self.goal_marks = goal_marks

    def goal_free_static(self, template):
        if self.goal_marks == "zero":
            return super().goal_free_static(template)
        st = self.base._static(template)
        gprop = torch.tensor([st.prop_index[g.fact_id] for g in template.goals], device=st.prop_pos.device)
        return st, gprop


# ------------------------------------------------------------------------------------------------ conditions
def combo_choose(controller, ep, snap, logits, mem):
    """G1+C3: exclude visited successors first, then the G1 rule on the remainder (exact order of runbook 9.3)."""
    ids = snap.candidate_ids
    legal = [i for i, m in enumerate(snap.mask) if m]
    raw = AC.argmax_tie(logits, ids, legal)
    info = {"raw": ids[raw], "intervened": False, "trigger": None}
    available = [i for i in legal if S.apply(ep.state, ep._action_of[ids[i]]) not in mem.visited]
    if not available:
        info["trigger"] = "NO_UNVISITED_SUCCESSOR"
        return None, info
    before = {k for k, v in S.goal_atom_truth(ep.problem, ep.state).items() if v}
    improving = [i for i in available if before < {k for k, v in S.goal_atom_truth(ep.problem, S.apply(ep.state, ep._action_of[ids[i]])).items() if v}]
    sel = AC.argmax_tie(logits, ids, improving if improving else available)
    info.update({"intervened": sel != raw, "trigger": ("GOAL_PROGRESS_SET" if improving else "VISITED_SUCCESSOR") if sel != raw else None})
    return sel, info


@torch.no_grad()
def run_episode_with(policy, case, solver, chooser):
    """Same episode loop and record format as ``a04p_controls.run_episode`` with an explicit action chooser."""
    ep = BwEpisode(case)
    problem = case.problem
    hidden = policy.initial_hidden()
    mem = AC.Memory(ep.state)
    decisions, trace, reason = [], [], None
    while not ep.done:
        snap = ep.snapshot()
        out = policy(snap, hidden)
        hidden = out.hidden
        L_before, opt = solver.optimal_actions(ep.state, problem.goal)
        opt_ids = [S.action_id(problem.names, a) for a in opt]
        ids = snap.candidate_ids
        sel_idx, info = chooser("", ep, snap, out.logits, mem)
        if sel_idx is None:
            reason = "NO_UNVISITED_SUCCESSOR"
            break
        selected = ids[sel_idx]
        legal_ids = [ids[i] for i, m in enumerate(snap.mask) if m]
        probs = out.distribution.probs
        before = S.goal_atom_truth(problem, ep.state)
        state = ep.state
        ep.step(selected)
        after = S.goal_atom_truth(problem, ep.state)
        trace.append((state, selected))
        mem.visits[(state, selected)] += 1
        mem.visited.add(ep.state)
        decisions.append({"decision_index": len(decisions), "state": list(state), "raw": info["raw"], "selected": selected, "intervened": info["intervened"], "trigger": info["trigger"],
                          "optimal_actions": opt_ids, "selected_is_optimal": selected in opt_ids, "raw_is_optimal": info["raw"] in opt_ids, "optimal_remaining": L_before,
                          "probs": {i: round(float(probs[ids.index(i)]), 6) for i in legal_ids}, "n_legal": len(legal_ids), "destroyed_satisfied_goal": M.goal_destroyed(before, after)})
    flags = [d["selected_is_optimal"] for d in decisions]
    success = bool(ep.success) and reason is None
    first_int = next((d["decision_index"] for d in decisions if d["intervened"]), None)
    return {"case_id": case.case_id, "n_blocks": problem.n, "success": success, "reason": reason or ep.reason, "steps": len(decisions), "optimal_length": case.optimal_length, "step_cap": case.step_cap,
            "decision_perfect": bool(success and all(flags)), "first_divergence": M.first_divergence(flags), "cycle": M.repeated_state_action_cycle(trace),
            "excess_steps": M.excess_steps(success, len(decisions), case.optimal_length), "interventions": sum(d["intervened"] for d in decisions), "first_intervention": first_int,
            "destroyed_satisfied_goal_count": sum(len(d["destroyed_satisfied_goal"]) for d in decisions),
            "necessary_destruction_steps": sum(1 for d in decisions if d["destroyed_satisfied_goal"] and d["selected_is_optimal"]),
            "avoidable_destruction_steps": sum(1 for d in decisions if d["destroyed_satisfied_goal"] and not d["selected_is_optimal"]), "decisions": decisions}


def chooser_for(cond):
    if cond in ("G1C3",):
        return combo_choose
    ctrl = {"G1": "G1", "C3": "C3", "M1C3": "C3"}.get(cond, "C0")
    return lambda _c, ep, snap, logits, mem: AC.choose(ctrl, ep, snap, logits, mem)


def load_condition(root, cond, device, run_root=None):
    from ..torch_rl import load_checkpoint
    root = Path(root)
    if cond in ("C0", "G1", "C3", "G1C3"):
        return E.load_model(root, "B2", device)
    if cond == "M0" or cond == "B2RANK":
        if cond == "M0":
            ck = root / OLD["M0"]["path"]
            assert E.sha256_file(ck) == OLD["M0"]["sha256"]
        else:
            ck = _new_ckpt(root, run_root, "B2RANK")
        pol = I.make_imitation_policy("B2-CACHED", device, 0)
        load_checkpoint(ck, pol)
        pol.eval()
        return pol
    if cond in ("M1", "M1C3", "M1GOAL"):
        if cond == "M1GOAL":
            model = GoalProgressModelGoal(GP.load_base(root, device), "global", "production").to(device)
            ck = _new_ckpt(root, run_root, "M1GOAL")
        else:
            model = GP.GoalProgressModel(GP.load_base(root, device), "global").to(device)
            ck = root / OLD["M1"]["path"]
            assert E.sha256_file(ck) == OLD["M1"]["sha256"]
        load_checkpoint(ck, model)
        model.eval()
        return GP.ScorePolicy(model)
    raise ValueError(cond)


def _new_ckpt(root, run_root, key):
    acct = json.loads((Path(run_root) / "runs" / RUN_IDS[key] / "training_accounting.json").read_text())
    ck = Path(root) / acct["checkpoints"]["final"]["path"]
    assert E.sha256_file(ck) == acct["checkpoints"]["final"]["sha256"]
    return ck


# ------------------------------------------------------------------------------------------------ identity
def source_identity(root, new_sources):
    root = Path(root)
    frozen = {}
    for rel in FROZEN:
        now = E.sha256_file(root / rel)
        base = hashlib.sha256(subprocess.check_output(["git", "-C", str(root), "show", "%s:%s" % (BASE_COMMIT, rel)])).hexdigest()
        frozen[rel] = {"sha256": now, "sha256_at_base_commit": base, "unchanged_since_base": now == base}
    return {"base_commit": BASE_COMMIT, "frozen_sources": frozen, "new_sources": {r: {"sha256": E.sha256_file(root / r)} for r in new_sources if (root / r).is_file()},
            "old_root_tree_ids_at_base": {rel: E.git(root, "rev-parse", "%s:%s" % (BASE_COMMIT, rel)) for rel in OLD_ROOTS}, "all_frozen_unchanged": all(v["unchanged_since_base"] for v in frozen.values())}


def check_identity(root, run_root):
    """Source identity only: the clean-tree test excludes the run root (ledger / logs / results are the explicit artifact whitelist)."""
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    own = str(run_root.relative_to(root))
    if E.git(root, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)%s" % own):
        raise E.A03Error("tracked source tree is dirty")
    reg = json.loads((run_root / "prep" / "source_identity.json").read_text())
    for rel, v in reg["frozen_sources"].items():
        if E.sha256_file(root / rel) != v["sha256"]:
            raise E.A03Error("frozen source changed: %s" % rel)
    for rel, v in reg["new_sources"].items():
        if E.sha256_file(root / rel) != v["sha256"]:
            raise E.A03Error("registered source changed: %s" % rel)
    for rel in OLD_ROOTS:
        if E.git(root, "rev-parse", "HEAD:%s" % rel) != reg["old_root_tree_ids_at_base"][rel]:
            raise E.A03Error("an old result directory changed: %s" % rel)
    return E.git(root, "rev-parse", "HEAD")


# ------------------------------------------------------------------------------------------------ destruction labels and colour twins
def destruction_flag(problem, solver):
    """True / False / None (UNKNOWN is never coerced to False)."""
    from .planner import PlannerLimit, destruction_labels
    try:
        return destruction_labels(problem, solver)[0]
    except PlannerLimit:
        return None


def colour_swapped(problem):
    """Same names, initial state and goal; every block's colour flipped (the static contract colour premises follow)."""
    return S.Problem(problem.names, tuple(1 - c for c in problem.colors), problem.init, problem.goal)


def seen_problem_hashes(root, d_train, cases):
    """Colour-preserving problem classes of every earlier set plus every hand-empty (colours, state, goal) problem in D_train."""
    from . import canonical as K
    root = Path(root)
    forbidden = set()
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    forbidden |= {c["problem_iso_hash"] for c in td["train"] + td["dev"]}
    for f in ("c1_bw_a0_iso_v1", "c1_bw_a1_color_reverse_v1", "c1_bw_a2_noniso_v1", "c1_bw_b_scale_v1", "c1_bw_a04p_pilot80_v1", "c1_bw_gp_confirm112_v1"):
        forbidden |= {c["problem_iso_hash"] for c in json.loads((root / ("configs/splits/%s.json" % f)).read_text())["cases"]}
    by_id = {c.case_id: c for c in cases}
    for t in d_train:
        p = by_id[t["case_id"]].problem
        for s in t["states"][:-1]:
            if S.held(tuple(s)) == -1 and tuple(s) != tuple(p.goal):
                forbidden.add(K.problem_iso_hash(S.Problem(p.names, p.colors, tuple(s), p.goal)))
    return forbidden


def _extra(p, q, **kw):
    from . import generator as Gen
    from . import contracts as C
    from . import goal_probe as GPB
    comps = [len(c) for c in GPB.goal_components(p.goal)]
    return {"k_nontrivial_towers": sum(1 for t in comps if t >= 2), "max_tower_height": max(comps), "n_candidates": len(C.template_for(p).contracts), "initial_goal_atoms_satisfied": sum(S.goal_atom_truth(p, p.init).values()),
            "namespace": NAMESPACE, "requires_goal_destruction_all_optimal_plans": q["labels"]["requires_goal_destruction"], **kw}


def gen_cell(builder, forbidden, key, n, heights, top, want_break, quota, p_stack, max_attempts, rejects):
    """First ``quota`` qualified, deduplicated problems of one (structure, destruction stratum) cell. Planner / label only: no model is ever called."""
    from . import canonical as K
    from . import generator as Gen
    out, attempt = [], 0
    while len(out) < quota and attempt < max_attempts:
        p = Gen.make_problem(key, n, len(out), heights, top, attempt, p_stack=p_stack)
        attempt += 1
        if p is None:
            rejects["generator_none"] += 1
            continue
        h = K.problem_iso_hash(p)
        if h in forbidden:
            rejects["seen_problem_class"] += 1
            continue
        q = builder.qualify(p, need_destruction=True)
        if q is None:
            rejects["unqualified_or_unsolved_or_label_unavailable"] += 1
            continue
        d = q["labels"]["requires_goal_destruction"]
        if d is None:
            rejects["destruction_label_unknown"] += 1
            continue
        if bool(d) != want_break:
            rejects["other_stratum"] += 1
            continue
        forbidden.add(h)
        out.append((p, q, h))
    return out, attempt


def gen_color_pairs(builder, forbidden, key, want_break, quota, max_attempts, rejects):
    """RED / BLUE twins of one physical skeleton (n=5, shape (3,1,1)): identical names, initial state, goal supports; every colour flipped. Both members are deduplicated and must agree on L* and D*."""
    from . import canonical as K
    from . import generator as Gen
    out, attempt = [], 0
    while len(out) < quota and attempt < max_attempts:
        p = Gen.make_problem(key, 5, len(out), (3, 1, 1), S.RED, attempt, p_stack=0.55)
        attempt += 1
        if p is None:
            rejects["generator_none"] += 1
            continue
        pb = colour_swapped(p)
        h1, h2 = K.problem_iso_hash(p), K.problem_iso_hash(pb)
        if h1 == h2:
            rejects["twins_isomorphic"] += 1
            continue
        if h1 in forbidden or h2 in forbidden:
            rejects["seen_problem_class"] += 1
            continue
        q1, q2 = builder.qualify(p, need_destruction=True), builder.qualify(pb, need_destruction=True)
        if q1 is None or q2 is None:
            rejects["unqualified_or_unsolved_or_label_unavailable"] += 1
            continue
        d1, d2 = q1["labels"]["requires_goal_destruction"], q2["labels"]["requires_goal_destruction"]
        if d1 is None or d2 is None:
            rejects["destruction_label_unknown"] += 1
            continue
        if d1 != d2 or q1["optimal_length"] != q2["optimal_length"]:
            rejects["twin_label_mismatch"] += 1
            continue
        if bool(d1) != want_break:
            rejects["other_stratum"] += 1
            continue
        forbidden.update((h1, h2))
        out.append(((p, q1, h1), (pb, q2, h2)))
    return out, attempt


# ------------------------------------------------------------------------------------------------ pre-training equivalence fixtures
def _grads(policy_params):
    return [None if p.grad is None else p.grad.detach().clone() for p in policy_params]


def _close(a, b, atol=1e-6, rtol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=rtol)


def check_rank_zero_equivalence(root, cases, d_train, labels, device, n_traj=6):
    """B2-RANK with rank_weight = 0 reproduces the old M0 training step (loss, gradients of every trainable parameter, parameters after the step), GRU gradients included."""
    import copy
    batch = d_train[:n_traj]
    # the trained A02 B2 already fits D_train (loss exactly 0, gradients ~ 1e-15, also after weight noise of 0.02), which would make the comparison vacuous: the fixture uses the untrained
    # B2-CACHED (same architecture, init seed 0) as the common starting point of the old and the new path
    p_old = I.make_imitation_policy("B2-CACHED", device, 0)
    p_new = copy.deepcopy(p_old)
    p_ctl = copy.deepcopy(p_old)
    t_old = I.ImitationTrainer(p_old, "b2", cases, device)
    t_old.optimizer = torch.optim.Adam(t_old.params, lr=LR_ALL, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    t_new = RankTrainer(p_new, "b2", cases, device, labels, 0.0)
    r_old, r_new = t_old.step(batch), t_new.step(batch)
    worst, worst_param = 0.0, 0.0
    ok = _close(torch.tensor(r_old["loss"]), torch.tensor(r_new["loss"]))
    for a, b in zip(t_old.params, t_new.params):
        ga, gb = a.grad, b.grad
        if (ga is None) != (gb is None):
            ok = False
            continue
        if ga is not None:
            worst = max(worst, float((ga - gb).abs().max()))
            ok &= _close(ga, gb)
        # parameters after the Adam step are informational only: Adam's first step is +-lr whatever the gradient size, so gradients at the 1e-15 noise level may flip sign on a GPU
        worst_param = max(worst_param, float((a.detach() - b.detach()).abs().max()))
    gru = float(sum(p.grad.abs().sum() for p in p_new.gru.parameters() if p.grad is not None))
    # control: weight 1 must change the gradient
    t_ctl = RankTrainer(p_ctl, "b2", cases, device, labels, 1.0)
    t_ctl.step(batch)
    differs = any((a.grad is not None and b.grad is not None and not _close(a.grad, b.grad)) for a, b in zip(t_old.params, t_ctl.params))
    return {"pass": bool(ok and gru > 0 and differs), "loss_old": r_old["loss"], "loss_new": r_new["loss"], "max_abs_grad_diff": worst, "max_abs_param_diff_after_adam_step_informational": worst_param, "gru_grad_abs_sum": gru, "rank_weight_one_changes_gradient": differs,
            "tolerance": {"atol": 1e-6, "rtol": 1e-5}, "batch_trajectories": n_traj}


def check_goal_switch_equivalence(root, cases, d_train, labels, device, n_traj=6):
    """M1-GOAL with the goal marks switched OFF equals the old M1 (scores, loss, gradients); with marks ON the production marks reach the message passing and make per-goal features depend on the other goals."""
    import copy
    from . import goal_probe as GPB
    base = GP.load_base(root, device)
    m_old = GP.GoalProgressModel(copy.deepcopy(base), "global").to(device)
    m_off = GoalProgressModelGoal(copy.deepcopy(base), "global", "zero").to(device)
    m_on = GoalProgressModelGoal(copy.deepcopy(base), "global", "production").to(device)
    batch = d_train[:n_traj]
    by = {c.case_id: c for c in cases}
    snaps = [I.snapshot_at(by[t["case_id"]], t["states"][0], 0) for t in batch]
    ok = True
    with torch.no_grad():
        for sn in snaps:
            a, b = m_old.group_scores([sn])[0], m_off.group_scores([sn])[0]
            ok &= bool(torch.equal(a == -torch.inf, b == -torch.inf)) and _close(torch.where(torch.isfinite(a), a, torch.zeros_like(a)), torch.where(torch.isfinite(b), b, torch.zeros_like(b)))
    t_old, t_off = GP.GPTrainer(m_old, cases, labels, device), GP.GPTrainer(m_off, cases, labels, device)
    r_old, r_off = t_old.step(batch), t_off.step(batch)
    ok &= _close(torch.tensor(r_old["loss"]), torch.tensor(r_off["loss"]))
    for a, b in zip(t_old.params, t_off.params):
        if (a.grad is None) != (b.grad is None):
            ok = False
        elif a.grad is not None:
            ok &= _close(a.grad, b.grad)
    # ON: marks are really used (encoder output differs) and per-goal features depend on the other goals
    sn = snaps[-1]
    with torch.no_grad():
        st_on, gp_on = m_on.goal_free_static(sn.template)
        st_off, gp_off = m_off.goal_free_static(sn.template)
        codes = m_on.base._codes(m_on.base._static(sn.template), sn.facts.values).unsqueeze(0)
        x_on, x_off = m_on.features(st_on, gp_on, codes), m_off.features(st_off, gp_off, codes)
        marks_reach = float(st_on.prop_goal_sign.abs().sum()) > 0 and float(st_off.prop_goal_sign.abs().sum()) == 0 and not _close(x_on, x_off)
        p = by[batch[-1]["case_id"]].problem
        comps = [c for c in GPB.goal_components(p.goal) if len(c) >= 2]
        flat = GPB.flatten_other_towers(p.goal, comps[0][-1]) if comps else p.goal
        p2 = S.Problem(p.names, p.colors, p.init, flat)
        from .environment import Case
        s2 = I.snapshot_at(Case("x", "t", p2, by[batch[-1]["case_id"]].optimal_length, 30), sn_state(batch[-1]), 0)

        def feats(model, snap):
            st, gp = model.goal_free_static(snap.template)
            x = model.features(st, gp, model.base._codes(model.base._static(snap.template), snap.facts.values).unsqueeze(0))[0]
            return {g.fact_id: x[i] for i, g in enumerate(snap.template.goals)}
        f_on1, f_on2, f_off1, f_off2 = feats(m_on, sn), feats(m_on, s2), feats(m_off, sn), feats(m_off, s2)
        common = set(f_on1) & set(f_on2)
        depends_on = any(not _close(f_on1[k], f_on2[k], 1e-6, 1e-5) for k in common) if flat != p.goal else True
        independent_off = all(_close(f_off1[k], f_off2[k], 1e-6, 1e-5) for k in common)
    head_sha = hashlib.sha256(b"".join(q.detach().cpu().numpy().tobytes() for q in m_on.heads.parameters())).hexdigest()
    return {"pass": bool(ok and marks_reach and depends_on and independent_off and head_sha == M1_HEAD_SHA), "switch_off_equals_old_M1": bool(ok), "marks_reach_message_passing": bool(marks_reach),
            "features_depend_on_other_goals_when_on": bool(depends_on), "features_independent_of_other_goals_when_off": bool(independent_off), "head_initial_sha256": head_sha,
            "head_sha_matches_M1": head_sha == M1_HEAD_SHA, "head_parameters": sum(q.numel() for q in m_on.heads.parameters()),
            "trainable_parameters": sum(q.numel() for q in t_off.params)}


def sn_state(traj):
    return tuple(traj["states"][0])


def net_threshold(n):
    """Registered direction threshold for a paired net difference over N problems / pairs (runbook 10.2)."""
    if n >= 32:
        return 5
    if n < 16:
        return None
    return math.ceil(5 * n / 32)


def direction(net, n):
    t = net_threshold(n)
    if t is None:
        return "NO_LABEL_N_LT_16"
    if net >= t:
        return "A_AHEAD"
    if net <= -t:
        return "B_AHEAD"
    return "NO_CLEAR_DIFFERENCE"
