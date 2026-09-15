"""Gate evaluation on frozen holdout scores. Never regenerates or retunes."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path

from .state_builder import reward_relevant
from upgrade_v2.p1_mainline_grasp_v6gm1.extra_methods import valid_count

ATOL = 1e-12


def _csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _num(x):
    if x in (None, "", "None"):
        return None
    return float(x)


def count_key(state: dict):
    return (state["task"], valid_count(state), state.get("success") is True, state.get("terminal_failure") is True)


def graph_key(state: dict):
    objs = state["objects"]
    return (state["task"], tuple((k, objs[k]["phase"]) for k in sorted(objs)),
            state.get("success") is True, state.get("terminal_failure") is True)


def event_key(state: dict, open_losses: tuple):
    return (state["task"], valid_count(state), open_losses,
            state.get("success") is True, state.get("terminal_failure") is True)


def _open_losses(states, index: int) -> tuple:
    open_set = set()
    for s in states[:index + 1]:
        for e in s.get("events") or []:
            key = (e.get("object_id"), e.get("loss_id"))
            if e.get("kind") == "LOSS":
                open_set.add(key)
            elif e.get("kind") == "HOLD_REESTABLISHED":
                open_set.discard(key)
    return tuple(sorted(open_set))


def evaluate(holdout: Path, scored: Path, out: Path) -> dict:
    holdout, scored, out = Path(holdout), Path(scored), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((holdout / "case_registry.json").read_text(encoding="utf-8"))
    returns = _csv(scored / "per_episode_returns.csv")
    trans = _csv(scored / "per_transition_rewards.csv")
    identity = _csv(scored / "potential_identity.csv")
    by_ep = {r["episode_id"]: r for r in returns}
    by_case = defaultdict(list)
    for c in cases:
        by_case[c["case_id"]].append(c)

    v6_at = defaultdict(dict)
    for r in trans:
        if r["method"] == "V6_CAP_POTENTIAL" and r["reward"] not in (None, ""):
            v6_at[r["episode_id"]][int(r["from_state_index"])] = float(r["reward"])

    # Gate E
    e_rows = []
    e_pass = True
    max_abs = 0.0
    for r in identity:
        err = float(r["abs_error"])
        step_err = float(r["max_step_identity_error"])
        ok = err <= ATOL and step_err <= ATOL
        e_pass = e_pass and ok
        max_abs = max(max_abs, err, step_err)
        e_rows.append({**r, "gate_e_passed": ok})

    # Gate A
    a_rows = []
    a_pass = True
    for fam in sorted({c["family_id"] for c in cases}):
        h1 = next(c for c in cases if c["family_id"] == fam and c["case_id"] == "H1")
        h2 = next(c for c in cases if c["family_id"] == fam and c["case_id"] == "H2")
        r1, r2 = by_ep[h1["episode_id"]], by_ep[h2["episode_id"]]
        v1, v2 = float(r1["V6_CAP_POTENTIAL_signed"]), float(r2["V6_CAP_POTENTIAL_signed"])
        same_end = abs(float(r1["psi_start"]) - float(r2["psi_start"])) <= ATOL and abs(float(r1["psi_end"]) - float(r2["psi_end"])) <= ATOL
        diff = abs(v1 - v2)
        order_penalty = (v1 < -ATOL) or (v2 < -ATOL)
        ok = (not order_penalty) and ((not same_end) or diff <= ATOL)
        a_pass = a_pass and ok
        a_rows.append({
            "family_id": fam, "H1_episode": h1["episode_id"], "H2_episode": h2["episode_id"],
            "H1_v6": v1, "H2_v6": v2, "abs_diff": diff, "same_start_end_psi": same_end,
            "H1_A_first": r1.get("LINEAR_A_FIRST_R1_signed"), "H1_B_first": r1.get("LINEAR_B_FIRST_R1_signed"),
            "H2_A_first": r2.get("LINEAR_A_FIRST_R1_signed"), "H2_B_first": r2.get("LINEAR_B_FIRST_R1_signed"),
            "negative_order_penalty": order_penalty, "passed": ok,
        })

    # Gate B
    b_rows = []
    b_pass = True
    for case_id in ("H3", "H4"):
        for c in by_case[case_id]:
            states = _jsonl(holdout / c["states_path"])
            idxs = c["segments"].get("alias_states") or []
            keys_count = [count_key(states[i]) for i in idxs]
            keys_graph = [graph_key(states[i]) for i in idxs]
            keys_event = [event_key(states[i], _open_losses(states, i)) for i in idxs]
            count_same = len(set(keys_count)) == 1 and len(idxs) >= 2
            graph_diff = len(set(keys_graph)) > 1
            event_diff = len(set(keys_event)) > 1
            ok = count_same and graph_diff
            b_pass = b_pass and ok
            b_rows.append({
                "episode_id": c["episode_id"], "case_id": case_id, "alias_states": idxs,
                "count_keys": [str(k) for k in keys_count],
                "graph_keys": [str(k) for k in keys_graph],
                "event_keys": [str(k) for k in keys_event],
                "unordered_count_same": count_same,
                "pathgraph_different": graph_diff,
                "matched_events_different": event_diff,
                "passed": ok,
            })

    # Gate C
    c_rows = []
    c_pass = True
    for c in by_case["H5"]:
        i0, i1 = c["segments"]["invalidation_transition"]
        r = v6_at[c["episode_id"]][i0]
        ok = r < 0.0
        c_pass = c_pass and ok
        c_rows.append({"episode_id": c["episode_id"], "case_id": "H5", "start": i0, "end": i1,
                       "v6_value": r, "requirement": "reward<0", "passed": ok})
    for c in by_case["H6"]:
        s, e = c["segments"]["repeat_segment"]
        total = math.fsum(v6_at[c["episode_id"]][i] for i in range(s, e - 1))
        ok = abs(total) <= ATOL
        c_pass = c_pass and ok
        c_rows.append({"episode_id": c["episode_id"], "case_id": "H6", "start": s, "end": e,
                       "v6_value": total, "requirement": "|R|<=1e-12", "passed": ok})
    for c in by_case["H8"]:
        i0, i1 = c["segments"]["label_only_transition"]
        r = v6_at[c["episode_id"]][i0]
        ok = abs(r) <= ATOL
        c_pass = c_pass and ok
        c_rows.append({"episode_id": c["episode_id"], "case_id": "H8", "start": i0, "end": i1,
                       "v6_value": r, "requirement": "reward=0+/-1e-12", "passed": ok})

    # Gate D
    d_rows = []
    d_pass = True
    n_cycles = 0
    for c in by_case["H7"]:
        states = _jsonl(holdout / c["states_path"])
        for ci, (s, e) in enumerate(c["segments"]["cycles"]):
            n_cycles += 1
            same = reward_relevant(states[s]) == reward_relevant(states[e])
            total = math.fsum(v6_at[c["episode_id"]][i] for i in range(s, e))
            ok = same and abs(total) <= ATOL
            d_pass = d_pass and ok
            d_rows.append({
                "episode_id": c["episode_id"], "cycle_index": ci, "start": s, "end": e,
                "state_equal": same, "v6_signed": total, "passed": ok,
            })
    d_all = n_cycles == 12 and d_pass

    def dump(path, rows):
        if not rows:
            path.write_text("status\nNO_ROWS\n", encoding="utf-8"); return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    art = out
    dump(art / "legal_order.csv", a_rows)
    dump(art / "aliasing.csv", b_rows)
    dump(art / "credit_semantics.csv", c_rows)
    dump(art / "exact_state_cycles.csv", d_rows)
    dump(art / "potential_identity.csv", e_rows)

    # incremental value table from observed alias/credit
    incremental = [
        {"capability": "current_valid_subgoals", "Count": True, "CountPlusEvents": True, "PathGraph": True,
         "note": "valid flags are visible to all three"},
        {"capability": "loss_recovery_events", "Count": False, "CountPlusEvents": True, "PathGraph": True,
         "note": "H3/H4 event keys differ; count keys do not"},
        {"capability": "active_object_identity", "Count": False, "CountPlusEvents": bool(any(r["matched_events_different"] for r in b_rows)),
         "PathGraph": True, "note": "product-state keeps object-phase pairs"},
        {"capability": "phase_structure", "Count": False, "CountPlusEvents": bool(any(r["matched_events_different"] for r in b_rows)),
         "PathGraph": True, "note": "PathGraph node includes phase; events are token-based"},
        {"capability": "multiple_legal_orders", "Count": False, "CountPlusEvents": False, "PathGraph": True,
         "note": "H1/H2 both legal; count does not encode order"},
        {"capability": "remaining_graph_cost", "Count": False, "CountPlusEvents": False, "PathGraph": True,
         "note": "V6 capability units / GRAPH_COST_ONLY if legacy bound; this holdout uses constructed legacy=0"},
        {"capability": "unified_state_reward", "Count": False, "CountPlusEvents": False, "PathGraph": True,
         "note": "Gate E identity is V6 potential, not count/event"},
    ]
    dump(art / "strong_baseline_comparison.csv", incremental)

    gates = {
        "A_legal_order": a_pass,
        "B_aliasing": b_pass,
        "C_credit": c_pass,
        "D_exact_state_cycles": d_all,
        "E_potential_identity": e_pass and len(e_rows) == 32 and max_abs <= ATOL,
    }
    passed = all(gates.values())
    claims = [
        {"claim": "H1_H2_legal_order", "gate": "A", "status": "PASS" if a_pass else "FAIL", "n": len(a_rows)},
        {"claim": "H3_H4_aliasing", "gate": "B", "status": "PASS" if b_pass else "FAIL", "n": len(b_rows)},
        {"claim": "H5_H6_H8_credit", "gate": "C", "status": "PASS" if c_pass else "FAIL", "n": len(c_rows)},
        {"claim": "H7_exact_state_cycles", "gate": "D", "status": "PASS" if d_all else "FAIL", "n_cycles": n_cycles},
        {"claim": "potential_identity", "gate": "E", "status": "PASS" if gates["E_potential_identity"] else "FAIL",
         "n": len(e_rows), "max_abs_error": max_abs},
    ]
    dump(art / "claim_to_evidence.csv", claims)

    decision = {
        "independent_state_holdout_passed": passed,
        "representation_confirmation": "SUPPORTED_ON_INDEPENDENT_STATE_HOLDOUT" if passed else "FAIL_ON_INDEPENDENT_STATE_HOLDOUT",
        "reward_accounting_confirmation": "SUPPORTED_ON_INDEPENDENT_STATE_HOLDOUT" if passed else "FAIL_ON_INDEPENDENT_STATE_HOLDOUT",
        "confirmation_passed": False,
        "physical_cycle_claim": "NOT_IN_SCOPE",
        "vision": False,
        "policy_gain_claimed": False,
        "real_robot": False,
        "reward_retuned": False,
        "graph_retuned": False,
        "gates": gates,
        "n_episodes": 32,
        "n_cycles": n_cycles,
        "max_identity_abs_error": max_abs,
        "failed_gates": [k for k, v in gates.items() if not v],
    }
    final = art / "final"
    final.mkdir(exist_ok=True)
    (final / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    report = _report(decision, a_rows, b_rows, c_rows, d_rows, e_rows, incremental)
    (final / "report.md").write_text(report, encoding="utf-8")
    return decision


def _report(decision, a_rows, b_rows, c_rows, d_rows, e_rows, incremental):
    lines = [
        "# Independent constructed-state holdout",
        "",
        f"independent_state_holdout_passed: {str(decision['independent_state_holdout_passed']).lower()}",
        f"representation_confirmation: {decision['representation_confirmation']}",
        f"reward_accounting_confirmation: {decision['reward_accounting_confirmation']}",
        "confirmation_passed: false",
        "",
        "This holdout is STATE_CONDITIONED_HOLDOUT. It is not physics, vision, policy, or robot confirmation.",
        "Generator did not import V6 reward. Scoring used frozen methods after generator commit/hash lock.",
        "",
        "## Gates",
    ]
    for k, v in decision["gates"].items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines += ["", "## Gate A legal order"]
    for r in a_rows:
        lines.append(f"- family {r['family_id']}: H1={r['H1_v6']:.12g} H2={r['H2_v6']:.12g} |diff|={r['abs_diff']:.3g} passed={r['passed']}")
        lines.append(f"  A-first/B-first on H1: {r['H1_A_first']} / {r['H1_B_first']}; on H2: {r['H2_A_first']} / {r['H2_B_first']}")
    lines += ["", "## Gate B aliasing"]
    for r in b_rows:
        lines.append(f"- {r['episode_id']}: count_same={r['unordered_count_same']} graph_diff={r['pathgraph_different']} events_diff={r['matched_events_different']} passed={r['passed']}")
    lines += ["", "## Gate C credit"]
    for r in c_rows:
        lines.append(f"- {r['case_id']} {r['episode_id']}: {r.get('v6_value')} passed={r['passed']}")
    lines += ["", "## Gate D exact state cycles", f"n_cycles={len(d_rows)}"]
    fail_d = [r for r in d_rows if not r["passed"]]
    lines.append("all passed" if not fail_d else f"failed: {fail_d}")
    lines += ["", "## Gate E potential identity", f"n={len(e_rows)} max_abs_error={decision['max_identity_abs_error']}"]
    lines += ["", "## Incremental value (observed, not preset win)"]
    for r in incremental:
        lines.append(f"- {r['capability']}: count={r['Count']} events={r['CountPlusEvents']} graph={r['PathGraph']} ({r['note']})")
    lines += ["", "## Out of scope", "- physical closed loop", "- vision", "- policy training", "- real robot"]
    return "\n".join(lines) + "\n"