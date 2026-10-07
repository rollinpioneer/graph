"""Shared evaluation protocol of the method suite: case sets, one-shot ledgered episode runs, Board48 checkpoint selection, state-bank ranking, paired tables."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .. import a04p_controls as AC
from .. import a04p_registry as R
from .. import eval_a03 as E
from .. import gp_attribution as A
from .. import planner as P
from .. import scale_order as SO
from .. import scorer_control as SC
from .. import state as S
from .. import train as T

SELECT_KEY = ("success", "h4_perfect", "neg_penalty", "perfect", "neg_epoch")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def wj(p, doc):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha_file(p):
    return E.sha256_file(p)


def load_cases(root, rel):
    """(groups {cell: [Case]}, meta {case_id: record}) of a registered split json (cases keyed by their 'cell')."""
    doc = json.loads((Path(root) / rel).read_text())
    by, meta = defaultdict(list), {}
    for c in doc["cases"]:
        by[c["cell"]].append(T.case_from_json(c))
        meta[c["case_id"]] = c
    return by, meta


def load_dev36(root):
    td = json.loads((Path(root) / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    return [("dev", [T.case_from_json(c) for c in td["dev"]])], td


def c3_chooser():
    return SC.chooser_for("MG_C3")


def raw_chooser():
    return lambda _c, ep, snap, logits, mem: AC.choose("C0", ep, snap, logits, mem)


def run_set(run_root, actor, groups, policy, chooser, out_path, solver=None):
    solver = solver or P.Solver()
    return R.run_cases(run_root, actor, groups, lambda c: A.run_episode_with(policy, c, solver, chooser), out_path)


def penalty_ratio(e):
    L = max(1, e["optimal_length"])
    return (e["steps"] / L) if e["success"] else ((e["step_cap"] + 1) / L)


def summarize(eps, meta=None):
    es = list(eps)
    succ = [e for e in es if e["success"]]
    out = {"n": len(es), "success": len(succ), "perfect": sum(e["decision_perfect"] for e in es), "mean_penalty_ratio": round(statistics.mean(penalty_ratio(e) for e in es), 4) if es else None,
           "total_actions": sum(e["steps"] for e in es), "excess_success": sum(e["excess_steps"] for e in succ), "failures": len(es) - len(succ),
           "failure_reasons": dict(Counter(e["reason"] for e in es if not e["success"])), "interventions": sum(e["interventions"] for e in es),
           "avoidable_destruction_steps": sum(e["avoidable_destruction_steps"] for e in es), "wall_seconds": round(sum(e.get("wall_seconds", 0.0) for e in es), 2)}
    if meta is not None:
        h4 = [e for e in es if meta[e["case_id"]]["max_tower_height"] == 4]
        out.update({"h4_n": len(h4), "h4_success": sum(e["success"] for e in h4), "h4_perfect": sum(e["decision_perfect"] for e in h4),
                    "h2_n": len(es) - len(h4), "h2_perfect": sum(e["decision_perfect"] for e in es if meta[e["case_id"]]["max_tower_height"] != 4)})
    return out


def select_key(row, epoch):
    return (row["success"], row["h4_perfect"], -row["mean_penalty_ratio"], row["perfect"], -epoch)


def choose_checkpoint(rows):
    """rows {epoch: summary}; plan 5.2 key: success, h4 decision-perfect, smaller failure-penalised length ratio, decision-perfect, earlier epoch."""
    return max(rows, key=lambda ep: select_key(rows[ep], ep))


def pair_counts(a, b, field):
    """a, b: {case_id: episode}; returns shared-correct / only-a / only-b / shared-wrong."""
    ids = sorted(set(a) & set(b))
    both = sum(a[i][field] and b[i][field] for i in ids)
    oa = sum(a[i][field] and not b[i][field] for i in ids)
    ob = sum(b[i][field] and not a[i][field] for i in ids)
    return {"n": len(ids), "both": both, "only_first": oa, "only_second": ob, "neither": len(ids) - both - oa - ob, "net": oa - ob}


# ------------------------------------------------------------------------------------------------ state bank
def bank_cases(root):
    _by, meta = load_cases(root, SO.BOARD_REL)
    return {cid: T.case_from_json(c) for cid, c in meta.items()}, meta


def bank_records(run_root_so):
    return json.loads((Path(run_root_so) / "prep" / "state_bank.json").read_text())


def bank_eval(policy, bank, cases):
    rows = []
    for r in bank:
        s = SO.score_state(policy, cases[r["case_id"]], r)
        rows.append({"bank_id": r["bank_id"], "case_id": r["case_id"], "kind": r["kind"], "all_optimal_neutral": r["all_optimal_neutral"], "must_destroy": r["must_destroy"],
                     "direct_progress_available": r["direct_progress_available"], "top1": s["top1"], "top1_optimal": s["top1_optimal"], "optimal_mass": s["optimal_mass"], "margin": s["margin"], "local_excess": s["local_excess"]})
    return rows


def bank_summary(rows, ref=None):
    def one(pred):
        rs = [r for r in rows if pred(r)]
        if not rs:
            return None
        o = {"states": len(rs), "top1_optimal": sum(r["top1_optimal"] for r in rs), "mean_optimal_mass": round(statistics.mean(r["optimal_mass"] for r in rs), 4),
             "mean_local_excess": round(statistics.mean(r["local_excess"] for r in rs), 4)}
        if ref is not None:
            rm = {x["bank_id"]: x for x in ref}
            o["repaired_vs_ref"] = sum((not rm[r["bank_id"]]["top1_optimal"]) and r["top1_optimal"] for r in rs)
            o["newly_wrong_vs_ref"] = sum(rm[r["bank_id"]]["top1_optimal"] and not r["top1_optimal"] for r in rs)
        return o
    return {"ALL": one(lambda r: True), "neutral_preparation": one(lambda r: r["all_optimal_neutral"]), "must_destroy": one(lambda r: r["must_destroy"]),
            "direct_progress_available": one(lambda r: r["direct_progress_available"]), "deviation_states": one(lambda r: r["kind"] == "deviation")}
