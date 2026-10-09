"""W1 (plan 5.4), step 1 in the project environment: export the D0 supervision as WL-GOOSE training data.

Fixed L2 WL feature system of the original model (ILG, 2 iterations, a-m pruning, set hash, rank-SVM, typed track). The data are exactly what D0 consumes: the parent states of the 457 Train96 trajectories (5,979 decisions) and their legal
successors with exact successor distances. A pair (better, worse) is created for every strictly ordered pair of legal successors of one parent (equal distances create no preference; the optimal-action set is therefore compared with every
strictly worse candidate). No parent-child pair and no cross-parent pair is added. The weight of a pair follows the rank part of D0's loss: a decision carries 1 / (trajectory length) (the constant 1 / batch size is dropped) and its
strictly ordered pairs share that weight equally (mean over pairs); weights of repeated parents / identical pairs add up.
States are converted by the SAME adapter code as the deployed WL scorer (translation atoms of the typed Fast Downward translation), so the fit sees what the search will see."""
from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from .. import depots as DP
from .. import stages as ST
from ..search_match import evaluators as EV
from .train import load_old


def bits_of(m):
    return EV.bits(m)


class StateExporter:
    """Project bit-mask state -> list of (predicate, args) atoms of the author adapter (translation atoms in the model domain), plus objects and goals of the translated problem."""

    def __init__(self, task, domain, problem, preds):
        self.task, self.preds = task, preds
        with tempfile.TemporaryDirectory() as d:
            sas_path = Path(d) / "output.sas"
            EV.translate_to_sas(domain, problem, sas_path)
            sas = EV.parse_sas(sas_path)
        by_bit, objects = {}, set()
        for names in sas["variables"]:
            for nm in names:
                m = EV.ATOM_RE.match(nm)
                if not m:
                    continue
                pred = m.group(1)
                args = tuple(x.strip() for x in m.group(2).split(",")) if m.group(2).strip() else ()
                objects.update(args)
                bit = task.dyn_index.get((pred, args))
                if bit is not None and pred in preds:
                    by_bit[bit] = (pred, list(args))
        self.by_bit = by_bit
        self.kept_mask = sum(1 << b for b in by_bit)
        self.objects = sorted(objects)
        self.goals = []
        for var, val in sas["goals"]:
            m = EV.ATOM_RE.match(sas["variables"][var][val])
            if m and m.group(1) in preds:
                args = tuple(x.strip() for x in m.group(2).split(",")) if m.group(2).strip() else ()
                self.goals.append((m.group(1), list(args)))

    def atoms(self, state):
        return [self.by_bit[b] for b in bits_of(state & self.kept_mask)]


def decision_weights_by_parent(trajs):
    """{(case_id, state key): weight} where each decision of a trajectory weighs 1 / len(trajectory) (D0: 1 / (len * B))."""
    w = defaultdict(float)
    for t in trajs:
        n = len(t["actions"])
        for s in t["states"][:-1]:
            w[(t["case_id"], ",".join(map(str, s)))] += 1.0 / n
    return w


def export(root, params_path, out_path, max_cases=None):
    """Writes the JSON consumed by ``scripts/c1_compact_w1_fit.py``. Returns statistics."""
    man, exact, _ = load_old(root)
    preds = {n: a for n, a in json.loads(Path(params_path).read_text())["domain"]["predicates"]}
    trajs, labels = [], {}
    cases = [c for c in man["train"] if exact[c["case_id"]]["status"] == "OK"][:max_cases]
    for c in cases:
        trajs += exact[c["case_id"]]["trajectories"]
        labels.update(exact[c["case_id"]]["rank_labels"])
    wdec = decision_weights_by_parent(trajs)
    problems, stats = [], {"problems": 0, "parents": 0, "states": 0, "pairs_raw": 0, "decisions": sum(len(t["actions"]) for t in trajs), "trajectories": len(trajs), "parents_without_strict_pair": 0,
                           "optimal_set_sizes": defaultdict(int)}
    for c in cases:
        cid = c["case_id"]
        task = ST.get_task(DP.DOMAIN_TYPED, c["file"])
        ex = StateExporter(task, DP.DOMAIN_TYPED, c["file"], preds)
        index, states, pairs = {}, [], defaultdict(float)

        def sid(st):
            if st not in index:
                index[st] = len(states)
                states.append(st)
            return index[st]
        parents = sorted({k for (cc, k) in wdec if cc == cid})
        for k in parents:
            dist = labels["%s|%s" % (cid, k)]
            parent = task.state_from_list([int(x) for x in k.split(",")]) if k else 0
            succ = {aid: task.apply(parent, task.action_by_id[aid]) for aid in dist}
            dmin = min(dist.values())
            stats["optimal_set_sizes"][sum(1 for d in dist.values() if d == dmin)] += 1
            strict = [(a, b) for a in dist for b in dist if dist[a] < dist[b]]
            stats["parents"] += 1
            if not strict:
                stats["parents_without_strict_pair"] += 1
                continue
            w = wdec[(cid, k)] / len(strict)
            for a, b in strict:
                if succ[a] == succ[b]:
                    continue
                pairs[(sid(succ[a]), sid(succ[b]))] += w
                stats["pairs_raw"] += 1
        problems.append({"case_id": cid, "objects": ex.objects, "goals": ex.goals, "states": [ex.atoms(s) for s in states], "state_masks_hex": [format(s, "x") for s in states],
                         "pairs": [[i, j, wt] for (i, j), wt in sorted(pairs.items())]})
        stats["problems"] += 1
        stats["states"] += len(states)
    stats["optimal_set_sizes"] = dict(stats["optimal_set_sizes"])
    stats["pairs_unique"] = sum(len(p["pairs"]) for p in problems)
    stats["weight_mass"] = sum(w for p in problems for _i, _j, w in p["pairs"])
    Path(out_path).write_text(json.dumps({"predicates": [[n, a] for n, a in preds.items()], "problems": problems, "stats": stats}))
    return stats
