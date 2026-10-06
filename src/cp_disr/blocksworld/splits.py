"""Split construction and audits for the Blocksworld cross-task card (runbook 11, 9.3, 10, 8.2). Planner-qualified, hash-seeded, built before any training; no model result exists here."""
from __future__ import annotations

import statistics

from . import canonical as K
from . import contracts as C
from . import generator as Gen
from . import metrics as M
from . import state as S
from .environment import step_cap_for
from .planner import PlannerLimit, Solver, bfs_cost, destruction_labels

TRAIN_N = (3, 4, 5)
SPLIT_SIZES = {"train": 48, "dev": 12, "a1": {3: 10, 4: 11, 5: 11}}
B_SHAPES = {6: ((3, 3), (2, 2, 2), (4, 2), (3, 2, 1)), 7: ((3, 2, 2), (4, 3), (3, 3, 1), (4, 2, 1)), 8: ((3, 3, 2), (4, 2, 2), (4, 4), (3, 3, 1, 1))}
A2_COUNTS = {(4, (2, 2)): 12, (5, (3, 2)): 10, (5, (2, 2, 1)): 10}
A0_PER_N = 8


class Builder:
    def __init__(self):
        self.solver = Solver()
        self.hops = {}

    def hopindex(self, problem):
        t = C.template_for(problem)
        key = id(t)
        if key not in self.hops:
            self.hops[key] = (t, M.HopIndex(t))
        return self.hops[key][1]

    def qualify(self, problem, need_destruction=True):
        """Planner qualification and labels, or None when the instance does not qualify (limit, trivial, unsolved)."""
        try:
            L = self.solver.cost_to_go(problem.init, problem.goal)
        except PlannerLimit:
            return None
        if L is None or L < 2:
            return None
        bfs_ok = None
        if problem.n <= 5:
            bfs_ok = bfs_cost(problem.init, problem.goal) == L
            if not bfs_ok:
                raise AssertionError("A* and BFS disagree")
        try:
            destruction = destruction_labels(problem, self.solver) if need_destruction else (None, None, L)
        except PlannerLimit:
            return None
        if need_destruction and destruction[0] is None:
            return None
        labels = M.case_labels(problem, self.solver, self.hopindex(problem), destruction)
        return {"optimal_length": L, "step_cap": step_cap_for(L), "bfs_verified": bfs_ok, "labels": labels}


def case_record(case_id, split, problem, q, extra=None):
    labels = dict(q["labels"])
    labels["plan"] = labels["plan"]
    rec = {"case_id": case_id, "split": split, "n_blocks": problem.n, "names": list(problem.names), "colors": list(problem.colors), "init": list(problem.init), "goal": list(problem.goal),
           "optimal_length": q["optimal_length"], "step_cap": q["step_cap"], "bfs_verified": q["bfs_verified"], "goal_shape": list(K.goal_shape(problem)), "goal_iso_hash": K.goal_iso_hash(problem),
           "goal_shape_hash": K.goal_shape_hash(problem), "problem_iso_hash": K.problem_iso_hash(problem), "init_iso_hash": K.init_iso_hash(problem), "labels": _clean(labels)}
    if extra:
        rec.update(extra)
    return rec


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and x == float("inf"):
        return "INF"
    return x


TRAIN_REPEAT_CAP = {3: 2, 4: 1, 5: 1}   # n=3 has only 36 problem-isomorphism classes (12 dev + 24 train): each train class is used exactly twice


def enumerate_classes(n, shapes, top_color, b):
    """Exhaustive problem-isomorphism classes (representative instance, qualification) for small n: all colourings x all alternating-tower goals x all valid initial states."""
    import itertools
    states = [s for s in itertools.product([S.TABLE] + list(range(n)), repeat=n) if S.is_valid(s) and all(x != S.HELD for x in s)]
    reps = {}
    for colors in itertools.product([0, 1], repeat=n):
        for heights in shapes:
            tower_heights = [h for h in heights if h > 1]
            seqs = [Gen.tower_colors(h, top_color) for h in tower_heights]
            goals = set()
            for perm in itertools.permutations(range(n), sum(tower_heights)):
                pos, ok, goal = 0, True, [S.TABLE] * n
                for seq in seqs:
                    blocks = perm[pos:pos + len(seq)]
                    pos += len(seq)
                    if any(colors[bk] != c for bk, c in zip(blocks, seq)):
                        ok = False
                        break
                    for lo, hi in zip(blocks, blocks[1:]):
                        goal[hi] = lo
                if ok:
                    goals.add(tuple(goal))
            for g in sorted(goals):
                for s0 in states:
                    if s0 == g:
                        continue
                    p = S.Problem(S.default_names(n), colors, s0, g)
                    h = K.problem_iso_hash(p)
                    if h in reps:
                        continue
                    q = b.qualify(p, need_destruction=False)
                    if q is not None:
                        reps[h] = (p, q)
    return reps


def build_train_dev(b, log=print):
    """Dev is built first (all problem classes distinct); train then avoids every dev class. n=3 has too few classes for 60 distinct problems, so its train classes may repeat (disclosed)."""
    import hashlib
    dev_hashes, uses, train, dev = set(), {}, [], []
    exhaustive = enumerate_classes(3, Gen.TRAIN_SHAPES[3], S.RED, b)          # n=3: all classes known; dev and train are drawn from it without sampling
    # dev initial states must not occur in train: group the classes by initial-state isomorphism class and give whole groups to dev (exactly 12 classes)
    groups = {}
    for h, (p, q) in exhaustive.items():
        groups.setdefault(K.init_iso_hash(p), []).append(h)
    keys = sorted(groups, key=lambda k: hashlib.sha256(("dev3|" + k).encode()).hexdigest())

    def pick(i, need, chosen):
        if need == 0:
            return chosen
        for j in range(i, len(keys)):
            size = len(groups[keys[j]])
            if size <= need:
                r = pick(j + 1, need - size, chosen + [keys[j]])
                if r is not None:
                    return r
        return None
    dev_keys = pick(0, SPLIT_SIZES["dev"], [])
    if dev_keys is None:
        raise RuntimeError("no initial-state-disjoint dev/train partition of the n=3 classes")
    dev3 = sorted(h for k in dev_keys for h in groups[k])
    train3 = sorted(h for k in keys if k not in dev_keys for h in groups[k])
    dev_init_iso = {k for k in dev_keys}
    for split, per_n in (("dev", SPLIT_SIZES["dev"]), ("train", SPLIT_SIZES["train"])):
        for n in TRAIN_N:
            shapes = Gen.TRAIN_SHAPES[n]
            made = 0
            if n == 3:
                hs = dev3 if split == "dev" else [train3[i % len(train3)] for i in range(per_n)]
                for i, h in enumerate(hs):
                    p, q = exhaustive[h]
                    if split == "dev":
                        dev_hashes.add(h)
                    else:
                        uses[h] = uses.get(h, 0) + 1
                    cid = "BW_%s_n3_%03d" % (split, i)
                    (train if split == "train" else dev).append(case_record(cid, split, p, q, {"want_blue_on_red": None, "blue_on_red": Gen.blue_on_red(p.colors, p.init)}))
                continue
            while made < per_n:
                heights = shapes[made % len(shapes)]
                want = (made % 2 == 0)
                prob = None
                for attempt in range(800):
                    p = Gen.make_problem(split, n, made, heights, S.RED, attempt, want_blue_on_red=want and attempt < 400)
                    if p is None:
                        continue
                    q = b.qualify(p, need_destruction=False)
                    if q is None:
                        continue
                    h = K.problem_iso_hash(p)
                    ih = K.init_iso_hash(p)
                    if split == "dev":
                        if h in dev_hashes:
                            continue
                    else:
                        if h in dev_hashes or ih in dev_init_iso or uses.get(h, 0) >= TRAIN_REPEAT_CAP[n]:
                            continue
                    prob = (p, q, h)
                    break
                if prob is None:
                    raise RuntimeError("could not build %s n=%d #%d" % (split, n, made))
                p, q, h = prob
                if split == "dev":
                    dev_hashes.add(h)
                    dev_init_iso.add(K.init_iso_hash(p))
                else:
                    uses[h] = uses.get(h, 0) + 1
                cid = "BW_%s_n%d_%03d" % (split, n, made)
                (train if split == "train" else dev).append(case_record(cid, split, p, q, {"want_blue_on_red": want, "blue_on_red": Gen.blue_on_red(p.colors, p.init)}))
                made += 1
        log("%s built: %d" % (split, len(train) if split == "train" else len(dev)))
    return train, dev


def build_a0(dev_cases, b):
    """24 colour-preserving renamed copies of dev cases (positive control; goal_iso and problem_iso identical to the source)."""
    out = []
    for n in TRAIN_N:
        src = [c for c in dev_cases if c["n_blocks"] == n][:A0_PER_N]
        for i, c in enumerate(src):
            rng = Gen.rng_for("a0", n, i)
            order = list(range(n))
            rng.shuffle(order)
            names = tuple("k%d" % order[j] for j in range(n))
            p0 = S.Problem(tuple(c["names"]), tuple(c["colors"]), tuple(c["init"]), tuple(c["goal"]))
            p = K.rename(p0, names)
            q = b.qualify(p, need_destruction=False)
            out.append(case_record("BW_a0_n%d_%03d" % (n, i), "a0", p, q, {"source_case_id": c["case_id"]}))
    return out


def build_a1(b, forbidden_problem, forbidden_goal_iso, log=print):
    out, seen = [], set()
    for n in TRAIN_N:
        shapes = Gen.TRAIN_SHAPES[n]
        made = 0
        while made < SPLIT_SIZES["a1"][n]:
            heights = shapes[made % len(shapes)]
            found = None
            for attempt in range(600):
                p = Gen.make_problem("a1", n, made, heights, S.BLUE, attempt)
                if p is None:
                    continue
                if K.goal_iso_hash(p) in forbidden_goal_iso or K.problem_iso_hash(p) in forbidden_problem:
                    continue
                q = b.qualify(p, need_destruction=True)
                if q is None or not _hop_ok(q):
                    continue
                h = K.problem_iso_hash(p)
                if h in seen:
                    continue
                found = (p, q, h)
                break
            if found is None:
                raise RuntimeError("could not build a1 n=%d #%d" % (n, made))
            p, q, h = found
            seen.add(h)
            out.append(case_record("BW_a1_n%d_%03d" % (n, made), "a1", p, q))
            made += 1
    return out


def _hop_ok(q):
    lab = q["labels"]
    return lab["initial_decision_hop"] <= 4 and lab["hop_le4_fraction"] >= 0.8


def build_a2(b, forbidden_problem, forbidden_shapes, log=print):
    out, seen = [], set()
    for (n, heights), count in A2_COUNTS.items():
        pool = {True: [], False: []}
        attempt = 0
        need = count // 2
        while (len(pool[True]) < need * 3 or len(pool[False]) < need * 3) and attempt < 4000:
            p = Gen.make_problem("a2", n, attempt, heights, S.RED, 0)
            attempt += 1
            if p is None:
                continue
            if K.goal_shape_hash(p) in forbidden_shapes or K.problem_iso_hash(p) in forbidden_problem:
                continue
            q = b.qualify(p, need_destruction=True)
            if q is None or not _hop_ok(q):
                continue
            h = K.problem_iso_hash(p)
            if h in seen:
                continue
            seen.add(h)
            pool[bool(q["labels"]["coordination_slice"])].append((p, q, h))
        log("a2 pool n=%d %s: coordination=%d no=%d (attempts %d)" % (n, heights, len(pool[True]), len(pool[False]), attempt))
        for flag in (True, False):
            if len(pool[flag]) < need:
                raise RuntimeError("a2 shortage n=%d %s flag=%s: %d < %d" % (n, heights, flag, len(pool[flag]), need))
        # plan-length matching: pair every chosen coordination case with the nearest-length non-coordination case (smallest differences first, deterministic)
        trues = sorted(pool[True], key=lambda t: (t[1]["optimal_length"], t[2]))
        falses = sorted(pool[False], key=lambda t: (t[1]["optimal_length"], t[2]))
        pairs, used = [], set()
        for t in trues:
            best = min((f for f in falses if id(f) not in used), key=lambda f: (abs(f[1]["optimal_length"] - t[1]["optimal_length"]), f[2]))
            pairs.append((abs(best[1]["optimal_length"] - t[1]["optimal_length"]), t[2], t, best))
        pairs.sort(key=lambda x: (x[0], x[1]))
        chosen, taken_f = [], set()
        for _d, _h, t, f in pairs:
            if len(chosen) >= 2 * need:
                break
            if id(f) in taken_f:
                f = min((g for g in falses if id(g) not in taken_f), key=lambda g: (abs(g[1]["optimal_length"] - t[1]["optimal_length"]), g[2]))
            taken_f.add(id(f))
            chosen += [t, f]
        for i, (p, q, h) in enumerate(chosen):
            out.append(case_record("BW_a2_n%d_%s_%03d" % (n, "".join(map(str, heights)), i), "a2", p, q))
    return out


def build_b(b, log=print):
    out, seen = [], set()
    cover = {"le4": 0, "gt4": 0}
    for n in (6, 7, 8):
        shapes = B_SHAPES[n]
        made = 0
        attempt = 0
        want_dest = True
        while made < 16 and attempt < 6000:
            heights = shapes[made % len(shapes)]
            p = Gen.make_problem("b", n, attempt, heights, S.RED, 0, p_stack=0.88)
            attempt += 1
            if p is None:
                continue
            q = b.qualify(p, need_destruction=True)
            if q is None:
                continue
            lab = q["labels"]
            if not 8 <= q["optimal_length"] <= 22 or lab["decisions_hop_le4"] == 0:
                continue
            h = K.problem_iso_hash(p)
            if h in seen:
                continue
            # alternate interference / non-interference so both occur
            if bool(lab["requires_goal_destruction"]) != (made % 2 == 0) and attempt < 3000:
                continue
            seen.add(h)
            out.append(case_record("BW_b_n%d_%03d" % (n, made), "b", p, q))
            cover["le4"] += lab["decisions_hop_le4"]
            cover["gt4"] += lab["decisions_hop_gt4"]
            made += 1
        if made < 16:
            raise RuntimeError("could not build b n=%d (%d)" % (n, made))
        log("b n=%d built (attempts %d)" % (n, attempt))
    # Amendment A01: the hop quota is removed (hop > 4 is structurally empty); the B slice is a scale / plan-length / graph-size / interference / cost slice.
    return out, cover


def half_life(train):
    return float(statistics.median(c["optimal_length"] for c in train))
