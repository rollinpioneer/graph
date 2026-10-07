#!/usr/bin/env python
"""Post-hoc description of the NEW runs of C1-BW-SCALE-ORDER-PROTOTYPE-V1 (read-only on the logs; written after the results were seen, so it is descriptive, not registered).

    python scripts/c1_bw_scale_order_post.py --run-root R
Writes results/post_error_classes_new_runs.csv and results/post_colour_twins_fresh112.csv.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def wcsv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run-root", required=True)
    a = ap.parse_args()
    root, rr = Path(a.root).resolve(), Path(a.run_root).resolve()
    from cp_disr.blocksworld import scale_order as SO
    from cp_disr.blocksworld import scorer_control as SC
    from cp_disr.blocksworld import train as T
    sc_eval = root / SO.SC_RUN_REL / "eval"
    runs = (("board48", SO.BOARD_REL, "MG_C3", rr / "eval/MG_C3/board48.jsonl", True), ("board48", SO.BOARD_REL, "PAD_C3", rr / "eval/PAD_C3/board48.jsonl", True), ("board48", SO.BOARD_REL, "B_G1C3", rr / "eval/B_G1C3/board48.jsonl", True),
            ("fresh112", SC.FRESH_REL, "PAD_C3", rr / "eval/PAD_C3/fresh112.jsonl", True), ("fresh112", SC.FRESH_REL, "MG_C3", sc_eval / "MG_C3/episodes.jsonl", True), ("fresh112", SC.FRESH_REL, "B_G1C3", sc_eval / "B_G1C3/episodes.jsonl", True))
    rows = []
    for s, rel, cond, path, c3 in runs:
        meta = {c["case_id"]: c for c in json.loads((root / rel).read_text())["cases"]}
        probs = {k: T.case_from_json(v).problem for k, v in meta.items()}
        agg = {}
        for e in jl(path):
            m = meta[e["case_id"]]
            st = SO.episode_error_stats(probs[e["case_id"]], e, c3)
            for scope, cc in (("raw", st["raw_classes"]), ("executed", st["exec_classes"])):
                for cl, n in cc.items():
                    for gt, gv in (("ALL", "ALL"), ("h", m["max_tower_height"]), ("n", m["n_blocks"])):
                        agg[(scope, cl, gt, str(gv))] = agg.get((scope, cl, gt, str(gv)), 0) + n
            agg[("decisions", "-", "ALL", "ALL")] = agg.get(("decisions", "-", "ALL", "ALL"), 0) + st["decisions"]
        for (scope, cl, gt, gv), n in sorted(agg.items()):
            rows.append({"set": s, "condition": cond, "scope": scope, "class": cl, "group_type": gt, "group_value": gv, "count": n})
    wcsv(rr / "results" / "post_error_classes_new_runs.csv", rows)
    meta = {c["case_id"]: c for c in json.loads((root / SC.FRESH_REL).read_text())["cases"]}
    tw = []
    for cond, path in (("PAD_C3", rr / "eval/PAD_C3/fresh112.jsonl"), ("MG_C3", sc_eval / "MG_C3/episodes.jsonl"), ("B_G1C3", sc_eval / "B_G1C3/episodes.jsonl")):
        ep = {e["case_id"]: e for e in jl(path)}
        for field in ("success", "decision_perfect"):
            c = Counter()
            for pid in sorted({m["pair_id"] for m in meta.values() if m.get("pair_id")}):
                r = next(i for i, m in meta.items() if m.get("pair_id") == pid and m["color_variant"] == "RED")
                b = next(i for i, m in meta.items() if m.get("pair_id") == pid and m["color_variant"] == "BLUE")
                x, y = ep[r][field], ep[b][field]
                c["both" if x and y else "RED_only" if x else "BLUE_only" if y else "neither"] += 1
            tw.append({"condition": cond, "metric": field, "pairs": sum(c.values()), **{k: c[k] for k in ("both", "RED_only", "BLUE_only", "neither")}})
    wcsv(rr / "results" / "post_colour_twins_fresh112.csv", tw)
    for r in tw:
        print(r)
    for cond in ("MG_C3", "PAD_C3", "B_G1C3"):
        for s in ("board48", "fresh112"):
            d = {r["class"]: r["count"] for r in rows if r["condition"] == cond and r["set"] == s and r["scope"] == "raw" and r["group_type"] == "ALL"}
            print(s, cond, sum(d.values()), d)


if __name__ == "__main__":
    main()
