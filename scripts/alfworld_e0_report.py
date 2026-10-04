"""E0 outputs: candidate-count and macro-duration distributions + human-readable traces."""
import json
import os
import statistics as st
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cp_disr.baselines import alfworld_scripts as S  # noqa: E402
from cp_disr.platforms.alfworld import data  # noqa: E402
from cp_disr.platforms.alfworld.adapter import AlfEpisode  # noqa: E402

RUN = "runs/alfworld_prior_reliance"
ROOT = "/home/xushijie2/xsj2_alf/data"
TABLES = None


def init():
    global TABLES
    TABLES = data.Tables(json.load(open(RUN + "/tables.json")))


def job(args):
    rel, seed = args
    ep = AlfEpisode(os.path.join(ROOT, rel), TABLES)
    import random

    rng = random.Random(seed)
    rows = []
    try:
        while not ep.done:
            pub = ep.public()
            n_check = sum(1 for a in pub.legal if a[0] == "CHECK")
            a = rng.choice(sorted(pub.legal))
            res = ep.execute(*a)
            rows.append((len(pub.legal), n_check, len(pub.unchecked), a[0], res.duration, res.ok, len(res.commands)))
    finally:
        ep.close()
    return rows


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    splits = json.load(open(RUN + "/splits.json"))
    games = splits["train"] + splits["dev"]
    init()
    with Pool(24, initializer=init) as p:
        out = p.map(job, [(g, 5) for g in games])
    flat = [r for o in out for r in o]
    legal = [r[0] for r in flat]
    checks = [r[1] for r in flat if r[1]]
    nu = [r[2] for r in flat if r[1]]
    summary = {
        "games": len(games), "decisions": len(flat),
        "legal_candidates_per_decision": {"mean": st.mean(legal), "p50": q(legal, .5), "p95": q(legal, .95), "max": max(legal)},
        "legal_CHECK_per_search_decision": {"mean": st.mean(checks), "p50": q(checks, .5), "p95": q(checks, .95), "max": max(checks)},
        "unchecked_feasible_per_search_decision": {"mean": st.mean(nu), "p50": q(nu, .5), "p95": q(nu, .95), "max": max(nu)},
        "max_legal_over_N_u_plus_2": max(r[0] - (r[2] + 2) for r in flat),
        "duration_by_macro": {k: {"mean": st.mean(r[4] for r in flat if r[3] == k), "min": min(r[4] for r in flat if r[3] == k), "max": max(r[4] for r in flat if r[3] == k), "n": sum(1 for r in flat if r[3] == k)} for k in ("CHECK", "TAKE", "PUT")},
        "macro_failures": sum(1 for r in flat if not r[5]),
    }
    # human-readable traces (belief-optimal, three dev games)
    traces = []
    for rel in splits["dev"][:3]:
        r = S.run_episode(os.path.join(ROOT, rel), TABLES, S.belief_optimal)
        ep = AlfEpisode(os.path.join(ROOT, rel), TABLES)
        lines = [ep.initial_feedback.strip().replace("\n", " ")]
        for kind, tgt in r["macros"]:
            res = ep.execute(kind, tgt)
            lines.append("%s(%s): %s -> %s" % (kind, tgt, " | ".join(res.commands), res.feedback[-1].strip()[:110]))
        lines.append("won=%s raw_steps=%d" % (ep.won, ep.raw_steps))
        ep.close()
        traces.append({"game": rel.split("/")[-3], "trace": lines})
    json.dump({"summary": summary, "traces": traces}, open(RUN + "/e0_outputs.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for t in traces:
        print("\n== " + t["game"])
        print("\n".join(t["trace"]))


if __name__ == "__main__":
    main()
