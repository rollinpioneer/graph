"""E0 setup: catalog -> scene-level 70/15/15 split -> train-only static tables (+ coverage audit)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cp_disr.platforms.alfworld import catalog, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/xushijie2/xsj2_alf/data")
    ap.add_argument("--out", default="runs/alfworld_prior_reliance")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cat_path = os.path.join(a.out, "catalog.json")
    rows = catalog.build_catalog(a.data_root, cat_path, a.workers)
    # make gamefile paths data-root independent
    for r in rows:
        r["gamefile"] = os.path.relpath(r["gamefile"], a.data_root)
    json.dump(rows, open(cat_path, "w"))
    splits = data.make_splits(rows)
    tables = data.build_tables(rows, splits["train"])
    json.dump(splits, open(os.path.join(a.out, "splits.json"), "w"), indent=1)
    json.dump(tables, open(os.path.join(a.out, "tables.json"), "w"), indent=1, sort_keys=True)
    T = data.Tables(tables)
    by = {r["gamefile"]: r for r in rows}
    summary = {"n_games": {k: len(v) for k, v in splits.items() if isinstance(v, list)},
               "n_scenes": {k: len({by[g]["scene"] for g in v}) for k, v in splits.items() if isinstance(v, list) and v and k in ("train", "dev", "test")}}
    # scene disjointness
    sc = {k: {by[g]["scene"] for g in splits[k]} for k in ("train", "dev", "test")}
    summary["scene_overlap"] = {"train&dev": len(sc["train"] & sc["dev"]), "train&test": len(sc["train"] & sc["test"]), "dev&test": len(sc["dev"] & sc["test"])}
    # static canContain audit: every game's own canContain(target) pairs must agree with the train table
    mism, uncovered, checked = 0, 0, 0
    for r in rows:
        own = {tuple(p) for p in r["cancontain"]}
        for x in r["receptacles"]:
            pair = (x["rtype"], r["goal_otype"])
            checked += 1
            if x["rtype"] not in tables["can_contain"]:
                uncovered += 1
            elif (pair in own) != T.can_contain(*pair):
                mism += 1
    summary["can_contain_audit"] = {"pairs_checked": checked, "mismatch_vs_train_table": mism, "rtype_not_in_train": uncovered}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
