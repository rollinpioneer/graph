"""Shared search panel of the new models (Joint32, IPC22) and of the paired task changes (old DENSE, old WL): one budgeted run per (arm, problem) with the shared eager GBFS of the earlier cards and the reference scorer path.
Arms of one problem run back to back in one process (one GPU), order by sha1(problem id), so paired timings share GPU and time block."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .. import depots as DP
from ..search_match.runner import run_case
from .train import load_old, rj

SEARCH = {"wall_seconds": 300, "max_expansions": 100000, "max_rss_growth_bytes": 8 * 2 ** 30, "grace_seconds": 20.0}


def panel_cases(rr, root, set_name):
    man, _exact, _ = load_old(root)
    panels = rj(Path(rr) / "registration" / "panels.json")
    if set_name == "joint32":
        ids = set(panels["joint32"])
        return sorted((c for c in man["joint"] if c["case_id"] in ids), key=lambda c: c["case_id"]), DP.DOMAIN_TYPED
    if set_name == "ipc22":
        return sorted(man["ipc"], key=lambda c: c["case_id"]), DP.DOMAIN_IPC
    if set_name == "variants":
        rows = rj(Path(rr) / "diagnostics" / "problem_transform_manifest.json")["rows"]
        return sorted(({"case_id": r["problem_id"], "file": r["file"], "sha256": r["sha256"]} for r in rows if r.get("feasible") and r.get("file")), key=lambda c: c["case_id"]), DP.DOMAIN_TYPED
    raise ValueError(set_name)


def order_arms(case_id, arms):
    k = int(hashlib.sha1(case_id.encode()).hexdigest(), 16) % len(arms)
    return arms[k:] + arms[:k]


def done_keys(path):
    out = set()
    p = Path(path)
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out.add((d["scorer"], d["case_id"]))
    return out


def run_panel(rr, root, arms, set_name, shard, nshards, make_eval, tag, log=print, plans_name="plans"):
    cases, domain = panel_cases(rr, root, set_name)
    cases = cases[shard::nshards]
    out = Path(rr) / "evaluation" / "runs" / ("%s_%s_%d.jsonl" % (tag, set_name, shard))
    out.parent.mkdir(parents=True, exist_ok=True)
    plans = Path(rr) / "evaluation" / plans_name
    evals = {a: make_eval(a) for a in arms}
    done = done_keys(out)
    for c in cases:
        for a in order_arms(c["case_id"], list(arms)):
            if (a, c["case_id"]) in done:
                continue
            rec = run_case(evals[a], a, set_name, c, domain, SEARCH, plans)
            rec["arm_order"] = ">".join(order_arms(c["case_id"], list(arms)))
            rec["shard"] = [shard, nshards]
            try:
                rec["load_avg_1m"] = os.getloadavg()[0]
            except OSError:
                pass
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            log("[panel] %s %s %s -> %s expanded %s wall %.1fs" % (a, set_name, c["case_id"], rec["status"], rec.get("expanded"), rec["wall_total"]), flush=True)
    return str(out)
