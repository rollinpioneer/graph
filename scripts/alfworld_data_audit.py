"""Per-game target/holder/goal-instance statistics and train->dev/test class-coverage audit (oracle side, analysis only)."""
import collections
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cp_disr.platforms.alfworld import data  # noqa: E402

RUN = "runs/alfworld_prior_reliance"


def dist(xs):
    c = collections.Counter(xs)
    return {"mean": round(st.mean(xs), 3), "max": max(xs), "hist": dict(sorted(c.items())), "frac_gt1": round(sum(v for k, v in c.items() if k > 1) / len(xs), 3)}


def main():
    rows = {r["gamefile"]: r for r in data.load_catalog(RUN + "/catalog.json")}
    splits = json.load(open(RUN + "/splits.json"))
    out = {"per_split": {}, "coverage": {}}
    names = ("train", "dev", "test", "official_valid_unseen")
    seen_train = {"otype": set(), "pair": set(), "rtype": set(), "holder_rtype_by_o": collections.defaultdict(set)}
    for g in splits["train"]:
        r = rows[g]
        by = {x["name"]: x["rtype"] for x in r["receptacles"]}
        seen_train["otype"].add(r["goal_otype"])
        seen_train["rtype"] |= set(by.values())
        for t in r["oracle"]["targets"]:
            if t["receptacle"]:
                seen_train["pair"].add((r["goal_otype"], by[t["receptacle"]]))
    for name in names:
        gs = splits[name]
        n_inst, n_hold, n_hold_feas, n_goal = [], [], [], []
        for g in gs:
            r = rows[g]
            by = {x["name"]: x["rtype"] for x in r["receptacles"]}
            can = {tuple(p) for p in r["cancontain"]}
            ts = [t for t in r["oracle"]["targets"] if t["receptacle"]]
            holders = {t["receptacle"] for t in ts}
            n_inst.append(len(ts))
            n_hold.append(len(holders))
            n_hold_feas.append(sum((by[h], r["goal_otype"]) in can for h in holders))
            n_goal.append(sum(1 for x in r["receptacles"] if x["rtype"] == r["goal_rtype"]))
        out["per_split"][name] = {"n_games": len(gs), "n_scenes": len({rows[g]["scene"] for g in gs}),
                                  "target_instances": dist(n_inst), "target_holder_receptacles": dist(n_hold),
                                  "feasible_holder_receptacles": dist(n_hold_feas), "goal_receptacle_instances": dist(n_goal)}
        if name == "train":
            continue
        pairs, miss_o, miss_pair, miss_r, goal_r_unseen, tot = 0, 0, 0, 0, 0, len(gs)
        unseen_o, unseen_r, unseen_pairs = collections.Counter(), collections.Counter(), collections.Counter()
        for g in gs:
            r = rows[g]
            by = {x["name"]: x["rtype"] for x in r["receptacles"]}
            if r["goal_otype"] not in seen_train["otype"]:
                miss_o += 1
                unseen_o[r["goal_otype"]] += 1
            hp = {(r["goal_otype"], by[t["receptacle"]]) for t in r["oracle"]["targets"] if t["receptacle"]}
            if hp - seen_train["pair"]:
                miss_pair += 1
                for p in hp - seen_train["pair"]:
                    unseen_pairs[p] += 1
            rs = set(by.values()) - seen_train["rtype"]
            if rs:
                miss_r += 1
                for x in rs:
                    unseen_r[x] += 1
            if r["goal_rtype"] not in seen_train["rtype"]:
                goal_r_unseen += 1
        out["coverage"][name] = {"games": tot, "target_class_not_in_train": miss_o, "unseen_target_classes": dict(unseen_o),
                                 "games_with_holder_(target,rtype)_pair_unseen_in_train": miss_pair, "unseen_pairs": {"%s|%s" % k: v for k, v in unseen_pairs.items()},
                                 "games_with_receptacle_class_not_in_train": miss_r, "unseen_receptacle_classes": dict(unseen_r),
                                 "goal_receptacle_class_not_in_train": goal_r_unseen}
    # same-public-start groups (room + goal) inside dev: how many hidden worlds share one initial observation
    groups = collections.defaultdict(list)
    for g in splits["dev"]:
        r = rows[g]
        groups[(tuple(sorted(x["name"] for x in r["receptacles"])), r["goal_otype"], r["goal_rtype"])].append(g)
    sizes = collections.Counter(len(v) for v in groups.values())
    out["dev_same_initial_observation_groups"] = {"n_groups": len(groups), "size_hist": dict(sorted(sizes.items())),
                                                  "games_in_groups_ge2": sum(len(v) for v in groups.values() if len(v) >= 2)}
    json.dump(out, open(RUN + "/data_audit.json", "w"), indent=1)
    for name, v in out["per_split"].items():
        print(name, v["n_games"], "games /", v["n_scenes"], "scenes | multi-target-instance %.2f | multi-holder %.2f | multi-feasible-holder %.2f | multi-goal-instance %.2f" % (
            v["target_instances"]["frac_gt1"], v["target_holder_receptacles"]["frac_gt1"], v["feasible_holder_receptacles"]["frac_gt1"], v["goal_receptacle_instances"]["frac_gt1"]))
    print(json.dumps(out["coverage"], indent=1))
    print(json.dumps(out["dev_same_initial_observation_groups"]))


if __name__ == "__main__":
    main()
