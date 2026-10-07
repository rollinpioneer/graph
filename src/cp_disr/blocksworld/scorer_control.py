"""C1-BW-SCORER-CONTROL-V1: zero-training 2x2 pairing of {fixed A02 B2, fixed M1-GOAL final} x {C3, G1+C3} on a fresh 112-problem Blocksworld set.

Nothing here trains, opens an optimizer or changes a rule: the rules are ``a04p_controls.choose('C3')`` and ``gp_attribution.combo_choose`` (G1+C3) unchanged; the scorers are the frozen checkpoints
loaded by the frozen loaders. Only registration, fixtures, the fresh-set builder and the identity helpers are new.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import torch

from . import a04p_controls as AC
from . import eval_a03 as E
from . import gp_attribution as A
from . import state as S

CARD = "C1-BW-SCORER-CONTROL-V1"
BASE_COMMIT = "178df59e4d8e32cbeeffd4deee51225f326530c2"
BRANCH = "codex/cp-disr-c1-bw-scorer-control-v1"
PLAN_REL = "docs/c1_blocksworld/CP_DISR_C1_Attribution_Closeout_and_Execution_Control_Check_v1.md"
SC_REL = E.BASE_REL + "/scorer_control_v1"
GPA_RUN_REL = A.GPA_REL + "/20261006T143512Z_3cb37301"
FRESH_REL = "configs/splits/c1_bw_scorer_control_fresh112_v1.json"
NAMESPACE = "C1-BW-SCORER-CONTROL-FRESH-v1"
CONDITIONS = ("B_C3", "B_G1C3", "MG_C3", "MG_G1C3")
SCORER = {"B_C3": "B2", "B_G1C3": "B2", "MG_C3": "M1GOAL", "MG_G1C3": "M1GOAL"}
RULE = {"B_C3": "C3", "B_G1C3": "G1C3", "MG_C3": "C3", "MG_G1C3": "G1C3"}
M1GOAL = {"run_id": "R-C1-GPA-M1-GOAL-0", "path": GPA_RUN_REL + "/runs/R-C1-GPA-M1-GOAL-0/checkpoints/final.pt", "bytes": 17880803,
          "sha256": "b04b32daa53553dd1632e836fb881df2e3a86181ebba95a80157b56d2c1d145f"}
OLD_ROOTS = A.OLD_ROOTS + (GPA_RUN_REL,)
FROZEN = tuple(dict.fromkeys(A.FROZEN + ("src/cp_disr/blocksworld/gp_attribution.py", "src/cp_disr/blocksworld/a04p_registry.py")))
PRIOR_SPLITS = ("c1_bw_a0_iso_v1", "c1_bw_a1_color_reverse_v1", "c1_bw_a2_noniso_v1", "c1_bw_b_scale_v1", "c1_bw_a04p_pilot80_v1", "c1_bw_gp_confirm112_v1", "c1_bw_gp_attr_confirm112_v2")


# ------------------------------------------------------------------------------------------------ rules and scorers
def chooser_for(cond):
    """C3 and G1+C3 exactly as already implemented; the scorer never enters the rule."""
    if RULE[cond] == "G1C3":
        return A.combo_choose
    return lambda _c, ep, snap, logits, mem: AC.choose("C3", ep, snap, logits, mem)


def load_scorer(root, cond, device):
    if SCORER[cond] == "B2":
        return E.load_model(root, "B2", device)
    return A.load_condition(root, "M1GOAL", device, Path(root) / GPA_RUN_REL)


def weights_digest(policy):
    m = getattr(policy, "model", policy)
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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


# ------------------------------------------------------------------------------------------------ fresh set isolation
def excluded_hashes(root, d_train, train_cases):
    """Every colour-preserving problem class used for training, dev, qualification, smoke, pilot or confirmation so far, by source."""
    root = Path(root)
    base = A.seen_problem_hashes(root, d_train, train_cases)         # train/dev, A0-B, pilot80, confirm v1 and every hand-empty problem of D_train
    by_source = {"earlier_sets_and_D_train": len(base)}
    v2 = {c["problem_iso_hash"] for c in json.loads((root / "configs/splits/c1_bw_gp_attr_confirm112_v2.json").read_text())["cases"]}
    by_source["confirm112_v2"] = len(v2)
    return base | v2, by_source


# ------------------------------------------------------------------------------------------------ rule fixtures (train-split cases and artificial logits only; no scorer, no fresh case)
def _arg(logits, ids, pool):
    return min(pool, key=lambda i: (-float(logits[i]), ids[i]))


def reference_choice(ep, snap, logits, visited, with_g1):
    """Independent restatement of plan section 4.1 (C3 first, G1 on the remainder, original max-logit / smallest-id tie-break)."""
    ids = snap.candidate_ids
    legal = [i for i, m in enumerate(snap.mask) if m]
    avail = [i for i in legal if S.apply(ep.state, ep._action_of[ids[i]]) not in visited]
    if not avail:
        return None
    pool = avail
    if with_g1:
        before = {k for k, v in S.goal_atom_truth(ep.problem, ep.state).items() if v}
        improving = [i for i in avail if before < {k for k, v in S.goal_atom_truth(ep.problem, S.apply(ep.state, ep._action_of[ids[i]])).items() if v}]
        pool = improving or avail
    return _arg(logits, ids, pool)


def check_rule_fixtures(cases, n_cases=14, steps=30, seed=20261007):
    from .environment import BwEpisode
    from . import planner as P
    rng = random.Random(seed)
    out = {"decisions_compared": 0, "mismatch_c3": 0, "mismatch_g1c3": 0, "illegal_selected": 0, "ties_present": False, "chooser_is_original_combo": chooser_for("B_G1C3") is A.combo_choose and chooser_for("MG_G1C3") is A.combo_choose,
           "both_scorers_share_each_rule": chooser_for("B_C3").__code__ is chooser_for("MG_C3").__code__ and chooser_for("B_G1C3") is chooser_for("MG_G1C3")}
    sample = [cases[i] for i in sorted(rng.sample(range(len(cases)), n_cases))]
    for c in sample:
        ep = BwEpisode(c)
        mem = AC.Memory(ep.state)
        for _ in range(steps):
            if ep.done:
                break
            snap = ep.snapshot()
            ids = snap.candidate_ids
            raw = [round(rng.uniform(-3, 3), 1) for _ in ids]                 # one decimal: ties occur
            logits = torch.tensor([x if m else 9.9 for x, m in zip(raw, snap.mask)])         # the highest logits sit on ILLEGAL candidates: the hard mask must still win
            out["ties_present"] |= len({x for x, m in zip(raw, snap.mask) if m}) < sum(snap.mask)
            r3 = reference_choice(ep, snap, logits, mem.visited, False)
            rg = reference_choice(ep, snap, logits, mem.visited, True)
            s3, _ = chooser_for("B_C3")("", ep, snap, logits, mem)
            sg, _ = chooser_for("B_G1C3")("", ep, snap, logits, mem)
            out["decisions_compared"] += 1
            out["mismatch_c3"] += s3 != r3
            out["mismatch_g1c3"] += sg != rg
            for s in (s3, sg):
                if s is not None and not snap.mask[s]:
                    out["illegal_selected"] += 1
            if rg is None:
                break
            ep.step(ids[rg])
            mem.visited.add(ep.state)
    solver = P.Solver()
    # C3 must run before G1: an improving successor that is already visited is never chosen, however high its score
    blocked, none_case, preserved, preserved_n = False, False, True, 0
    for c in sample[:6]:
        plan = solver.one_optimal_plan(c.problem.init, c.problem.goal)
        ep = BwEpisode(c)
        for a in plan[:-1]:
            ep.step(S.action_id(c.problem.names, a))
        snap = ep.snapshot()
        ids = snap.candidate_ids
        last = S.action_id(c.problem.names, plan[-1])
        mem = AC.Memory(ep.state)
        mem.visited.add(S.apply(ep.state, plan[-1]))
        logits = torch.full((len(ids),), -1.0)
        logits[ids.index(last)] = 50.0
        sel, _ = chooser_for("B_G1C3")("", ep, snap, logits, mem)
        blocked |= (sel is None) or ids[sel] != last
        mem.visited |= {S.apply(ep.state, ep._action_of[ids[i]]) for i, m in enumerate(snap.mask) if m}
        sel, info = chooser_for("B_G1C3")("", ep, snap, logits, mem)
        none_case |= sel is None and info["trigger"] == "NO_UNVISITED_SUCCESSOR"
        # an optimal trajectory never repeats a state, so C3 alone leaves it untouched (plan 7.3)
        ep = BwEpisode(c)
        mem = AC.Memory(ep.state)
        for a in plan:
            snap = ep.snapshot()
            ids = snap.candidate_ids
            aid = S.action_id(c.problem.names, a)
            logits = torch.full((len(ids),), -1.0)
            logits[ids.index(aid)] = 5.0
            sel, info = chooser_for("B_C3")("", ep, snap, logits, mem)
            preserved &= sel is not None and ids[sel] == aid and not info["intervened"]
            preserved_n += 1
            ep.step(aid)
            mem.visited.add(ep.state)
        preserved &= bool(ep.success)
    out.update({"c3_before_g1_visited_improving_successor_not_chosen": bool(blocked), "empty_candidate_set_ends_NO_UNVISITED_SUCCESSOR": bool(none_case),
                "c3_leaves_optimal_trajectories_unchanged": bool(preserved), "optimal_steps_replayed": preserved_n})
    out["pass"] = bool(out["mismatch_c3"] == 0 and out["mismatch_g1c3"] == 0 and out["illegal_selected"] == 0 and out["decisions_compared"] > 100 and out["ties_present"] and out["chooser_is_original_combo"]
                       and out["both_scorers_share_each_rule"] and blocked and none_case and preserved)
    return out


def check_scorer_loading(root, cases, device):
    """Both frozen scorers load, give finite masked logits on a TRAIN case, and a forward pass leaves every weight untouched."""
    from .environment import BwEpisode
    res = {}
    for cond in ("B_C3", "MG_C3"):
        pol = load_scorer(root, cond, device)
        d0 = weights_digest(pol)
        ep = BwEpisode(cases[-1])
        snap = ep.snapshot()
        with torch.no_grad():
            out = pol(snap, pol.initial_hidden())
        legal = [i for i, m in enumerate(snap.mask) if m]
        m = getattr(pol, "model", pol)
        res[cond] = {"digest_before": d0, "digest_after_forward": weights_digest(pol), "finite_on_legal": bool(torch.isfinite(out.logits[legal]).all()), "n_candidates": len(snap.candidate_ids),
                     "n_parameters": sum(p.numel() for p in m.parameters()), "any_requires_grad_after_load": any(p.requires_grad for p in m.parameters()) if SCORER[cond] == "B2" else None}
    res["pass"] = all(res[c]["digest_before"] == res[c]["digest_after_forward"] and res[c]["finite_on_legal"] for c in ("B_C3", "MG_C3")) and res["B_C3"]["digest_before"] != res["MG_C3"]["digest_before"]
    return res
