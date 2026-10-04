"""Scene-level 70/15/15 split of ALFWorld train pick_and_place_simple and train-only static tables."""
import json
import random
from collections import defaultdict

from .observation import type_of

SPLIT_SEED = 20261004
ALPHA = 1.0  # frozen Laplace constant of the legacy mutually-exclusive class belief
TAU = 2.0    # frozen shrinkage strength of the multi-holder instance belief (towards the class marginal)


def load_catalog(path):
    rows = json.load(open(path))
    bad = [r for r in rows if "error" in r]
    if bad:
        raise ValueError("catalog contains %d failed games" % len(bad))
    return rows


def holders_are_feasible(row):
    """Dataset validity (not an outcome of any method): at least one receptacle that holds a target
    must satisfy the game's own canContain(receptacle type, target type); otherwise the
    canContain-masked CHECK interface can never reach a target and the game is unsolvable by contract."""
    by = {x["name"]: x["rtype"] for x in row["receptacles"]}
    can = {tuple(p) for p in row["cancontain"]}
    return any((by[t["receptacle"]], row["goal_otype"]) in can for t in row["oracle"]["targets"] if t["receptacle"])


def make_splits(rows, seed=SPLIT_SEED, dev_frac=0.15, test_frac=0.15):
    """Whole rooms (FloorPlan number) go to exactly one of train/dev/test.

    The scene assignment is computed on all train games; contract-unsolvable games (see
    holders_are_feasible) are then removed from whichever list they fell into and recorded.
    """
    excluded = sorted(r["gamefile"] for r in rows if not holders_are_feasible(r))
    train_rows = [r for r in rows if r["split_source"] == "train"]
    games_per_scene = defaultdict(list)
    for r in train_rows:
        games_per_scene[r["scene"]].append(r["gamefile"])
    scenes = sorted(games_per_scene)
    random.Random(seed).shuffle(scenes)
    total = len(train_rows)
    out = {"train": [], "dev": [], "test": []}
    scene_assign = {}
    for s in scenes:
        if len(out["test"]) < test_frac * total:
            name = "test"
        elif len(out["dev"]) < dev_frac * total:
            name = "dev"
        else:
            name = "train"
        out[name].extend(sorted(games_per_scene[s]))
        scene_assign[s] = name
    out["official_valid_unseen"] = sorted(r["gamefile"] for r in rows if r["split_source"] == "valid_unseen")
    out["official_valid_seen"] = sorted(r["gamefile"] for r in rows if r["split_source"] == "valid_seen")
    out["scenes"] = {str(k): v for k, v in sorted(scene_assign.items())}
    out["seed"] = seed
    out["excluded_contract_unsolvable"] = excluded
    for k in ("train", "dev", "test", "official_valid_unseen", "official_valid_seen"):
        out[k] = [g for g in out[k] if g not in set(excluded)]
    return out


def build_tables(rows, train_files):
    """Static tables from TRAIN games only (canContain, openable rate, empirical location counts)."""
    train_files = set(train_files)
    train = [r for r in rows if r["gamefile"] in train_files]
    # canContain is the ALFRED type-compatibility table: game independent (audited: no game disagrees
    # with the union), so it is taken from the static PDDL problem facts of every game. It carries no
    # placement information. Statistics (p_open, belief) below use TRAIN games only.
    can_contain = defaultdict(set)
    for r in rows:
        for a, b in r["cancontain"]:
            can_contain[a].add(b)
    open_n = defaultdict(lambda: [0, 0])
    belief = defaultdict(lambda: defaultdict(int))
    holder_k = defaultdict(lambda: defaultdict(int))   # holder instances by (target class, receptacle class)
    holder_n = defaultdict(lambda: defaultdict(int))   # feasible instances by (target class, receptacle class)
    rtypes = set()
    for r in train:
        own_can = {tuple(p) for p in r["cancontain"]}
        by_name = {x["name"]: x for x in r["receptacles"]}
        for x in r["receptacles"]:
            rtypes.add(x["rtype"])
            open_n[x["rtype"]][0] += int(x["openable"])
            open_n[x["rtype"]][1] += 1
        holders = {t["receptacle"] for t in r["oracle"]["targets"] if t["receptacle"]}
        for h in holders:
            belief[r["goal_otype"]][by_name[h]["rtype"]] += 1
        for x in r["receptacles"]:
            if (x["rtype"], r["goal_otype"]) in own_can:
                holder_n[r["goal_otype"]][x["rtype"]] += 1
                holder_k[r["goal_otype"]][x["rtype"]] += int(x["name"] in holders)
    overall = sum(v[0] for v in open_n.values()) / max(1, sum(v[1] for v in open_n.values()))
    return {
        "can_contain": {k: sorted(v) for k, v in sorted(can_contain.items())},
        "p_open": {k: v[0] / v[1] for k, v in sorted(open_n.items())},
        "p_open_default": overall,
        "belief_counts": {o: dict(sorted(c.items())) for o, c in sorted(belief.items())},
        "holder_k": {o: dict(sorted(c.items())) for o, c in sorted(holder_k.items())},
        "holder_n": {o: dict(sorted(c.items())) for o, c in sorted(holder_n.items())},
        "tau": TAU,
        "rtypes": sorted(rtypes),
        "alpha": ALPHA,
        "n_train_games": len(train),
    }


class Tables:
    """Query wrapper; every number comes from train games."""

    def __init__(self, tables):
        self.t = tables
        self.can = {k: set(v) for k, v in tables["can_contain"].items()}

    def can_contain(self, rtype, otype):
        return otype in self.can.get(rtype, ())

    def p_open(self, rtype):
        return self.t["p_open"].get(rtype, self.t["p_open_default"])

    def holder_rate(self, otype, rtype):
        """P(an instance of this receptacle class holds >=1 target-class object): train counts shrunk to the
        class marginal over all target classes, (k + tau*m_c) / (n + tau) with m_c = (K_c + 1) / (N_c + 2)."""
        K = sum(self.t["holder_k"].get(o, {}).get(rtype, 0) for o in self.t["holder_k"])
        N = sum(self.t["holder_n"].get(o, {}).get(rtype, 0) for o in self.t["holder_n"])
        m = (K + 1.0) / (N + 2.0)
        k = self.t["holder_k"].get(otype, {}).get(rtype, 0)
        n = self.t["holder_n"].get(otype, {}).get(rtype, 0)
        return (k + self.t["tau"] * m) / (n + self.t["tau"])

    def class_probs(self, otype, feasible_rtypes):
        """p_hat(c_r | c_o) with Laplace smoothing over all known receptacle classes, restricted+renormalised."""
        counts = self.t["belief_counts"].get(otype, {})
        classes = self.t["rtypes"]
        total = sum(counts.values()) + self.t["alpha"] * len(classes)
        raw = {c: (counts.get(c, 0) + self.t["alpha"]) / total for c in set(classes) | set(feasible_rtypes)}
        z = sum(raw[c] for c in feasible_rtypes)
        return {c: raw[c] / z for c in feasible_rtypes}


def instance_belief(tables, otype, feasible_unchecked):
    """Class mass split equally over remaining same-class instances, renormalised (spec 5.2.2)."""
    names = list(feasible_unchecked)
    if not names:
        return {}
    classes = sorted({type_of(n) for n in names})
    cp = tables.class_probs(otype, classes)
    per_class = defaultdict(int)
    for n in names:
        per_class[type_of(n)] += 1
    w = {n: cp[type_of(n)] / per_class[type_of(n)] for n in names}
    z = sum(w.values())
    return {n: v / z for n, v in w.items()}
