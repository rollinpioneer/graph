"""Fill semantic_gates from real files and executed tests. No hand-true."""
from __future__ import annotations
import json, math, sys
from copy import deepcopy
from pathlib import Path
from .util import require_new, write_json, write_csv


def _tools(pkg: Path):
    sys.path.insert(0, str(Path(pkg)/"tools"))
    from probes import loops, dual_frames, suite, frame
    from reward_v6 import score_episode
    from reeval_core import EventBalance, progress
    return loops, dual_frames, suite, frame, score_episode, EventBalance, progress


def _rewards(seq, method, score_episode):
    d,v,_ = score_episode(seq)
    return [r["reward"] for r in d if r["method"]==method], v


def evaluate(pkg: Path, art: Path, csv_prefix_summary: dict | None = None) -> dict:
    loops, dual_frames, suite, frame, score_episode, EventBalance, progress = _tools(pkg)
    out_dir = Path(art)/"03_semantics_v1"
    # may already exist if prefix audit wrote into it; allow subdir files
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = []

    # Q1 phi clawback: constructed loops
    q1_rows = []
    for p in (0.2, 0.4, 0.8):
        seq = loops(p, 1, False)
        rs, v = _rewards(seq, "V6_CAP_POTENTIAL", score_episode)
        # index 1: TRANSPORT p -> LOST p
        expected = v[1]["cost_cap"] - v[2]["cost_cap"] - v[1]["local_credit"]
        q1_rows.append({"p": p, "r_fail": rs[1], "expected": expected, "passed": abs(rs[1]-expected)<=1e-12,
                        "old_missing": True})
        old,_ = _rewards(seq, "FULL_FROZEN", score_episode)
        q1_rows[-1]["old_fail"] = old[1]
        # no extra double penalty vs potential telescope
        q1_rows[-1]["telescope"] = abs(math.fsum(rs) - (v[-1]["psi"]-v[0]["psi"]))<=1e-12
    write_csv(out_dir/"q1_phi_clawback.csv", q1_rows)
    evidence.append(str(out_dir/"q1_phi_clawback.csv"))
    phi_ok = all(r["passed"] and r["telescope"] for r in q1_rows)

    # Q2 label neutrality
    seq = loops(0.4)
    rs,_ = _rewards(seq, "V6_CAP_POTENTIAL", score_episode)
    old,_ = _rewards(seq, "FULL_FROZEN", score_episode)
    label_row = {"v6_lost_to_recovering": rs[2], "old": old[2], "passed": rs[2]==0.0, "old_positive": old[2]>0}
    # command name / case id
    mut = deepcopy(seq)
    for s in mut:
        s["command_name"] = "recover"
        s["case_id"] = "success"
    rs2,_ = _rewards(mut, "V6_CAP_POTENTIAL", score_episode)
    label_row["command_case_invariant"] = rs==rs2
    write_json(out_dir/"q2_label_neutrality.json", label_row)
    evidence.append(str(out_dir/"q2_label_neutrality.json"))
    label_ok = label_row["passed"] and label_row["command_case_invariant"]

    # genuine progress not negative
    seq = [frame("n",0,"WAIT",0), frame("n",1,"HELD",0), frame("n",2,"TRANSPORT",0),
           frame("n",3,"TRANSPORT",1), frame("n",4,"PLACED",1), frame("n",5,"VALID",1)]
    rs,_ = _rewards(seq, "V6_CAP_POTENTIAL", score_episode)
    gp = {"rewards": rs, "passed": all(x>=-1e-12 for x in rs)}
    write_json(out_dir/"q1_genuine_progress.json", gp)
    evidence.append(str(out_dir/"q1_genuine_progress.json"))

    # event pairing
    b = EventBalance()
    pairing = []
    pairing.append({"op":"loss","r": b.step("A", loss_id="l")[0], "expect": -1})
    pairing.append({"op":"dup_loss","r": b.step("A", loss_id="l")[0], "expect": 0})
    pairing.append({"op":"wrong_obj","r": b.step("B", completed_loss_id="l")[0], "expect": 0})
    pairing.append({"op":"complete","r": b.step("A", completed_loss_id="l")[0], "expect": 1})
    pairing.append({"op":"dup_complete","r": b.step("A", completed_loss_id="l")[0], "expect": 0})
    rs_loop,_ = _rewards(loops(0.4), "V6_CAP_POTENTIAL", score_episode)
    rs_evt,_ = _rewards(loops(0.4), "PAIRED_EVENT_BALANCED", score_episode)
    pairing_ok = all(abs(r["r"]-r["expect"])<1e-12 for r in pairing) and rs_evt[2]==0 and rs_evt[3]==1
    # candidate itself is not the event method
    write_json(out_dir/"event_pairing.json", {"rows": pairing, "candidate_is_not_event_bonus": True,
                                             "v6_on_recovery_label": rs_loop[2], "passed": pairing_ok})
    evidence.append(str(out_dir/"event_pairing.json"))

    # baselines
    a_only = progress(True, False, False, "A_FIRST")
    b_only = progress(True, False, False, "B_FIRST")
    rs_ab,_ = _rewards(dual_frames(("B","A")), "LINEAR_A_FIRST_R1", score_episode)
    base = {"A_FIRST_A_only": a_only, "B_FIRST_A_only": b_only, "different": a_only != b_only,
            "reverse_order_not_forced_negative": all(r>=0 for r in rs_ab),
            "success_both": progress(True,True,True,"A_FIRST")==1 and progress(True,True,True,"B_FIRST")==1}
    base["passed"] = base["different"] and base["reverse_order_not_forced_negative"] and base["success_both"]
    write_json(out_dir/"baseline_fairness.json", base)
    evidence.append(str(out_dir/"baseline_fairness.json"))

    # raw provenance from development run
    dev = Path(art)/"02_development_v1"
    summary = json.loads((dev/"run_summary.json").read_text(encoding="utf-8"))
    unresolved = json.loads((dev/"unresolved.json").read_text(encoding="utf-8"))
    read_fail = json.loads((dev/"read_failures.json").read_text(encoding="utf-8")) if (dev/"read_failures.json").is_file() else []
    prov = json.loads((dev/"input_provenance.json").read_text(encoding="utf-8"))
    tiers = {p.get("evidence_tier") for p in prov}
    raw_ok = summary.get("episodes_scored")==112 and not unresolved and not read_fail and "EXISTING_SIMPLIFIED_STATE_SIMULATION_REPLAY" in tiers
    raw_doc = {"episodes_scored": summary.get("episodes_scored"), "unresolved": len(unresolved),
               "read_failures": len(read_fail), "tiers": sorted(tiers), "passed": raw_ok,
               "run_summary": str(dev/"run_summary.json")}
    write_json(out_dir/"raw_provenance.json", raw_doc)
    evidence.append(str(out_dir/"raw_provenance.json"))

    # input leakage: packaged prefix + csv rebuild
    pref_rows = list((dev/"prefix_audit.csv").read_text(encoding="utf-8").splitlines())
    # also constructed future mutation
    seq = loops(0.4, 3)
    mut = deepcopy(seq); mut[-1]["objects"]["obj"]["pos"][0] = 0.1
    a,_ = _rewards(seq, "V6_CAP_POTENTIAL", score_episode)
    b,_ = _rewards(mut, "V6_CAP_POTENTIAL", score_episode)
    csv_ok = bool(csv_prefix_summary and csv_prefix_summary.get("passed"))
    leak = {
        "packaged_prefix_file": str(dev/"prefix_audit.csv"),
        "constructed_future_mutation_passed": a[:-1]==b[:-1],
        "csv_prefix_audit_passed": csv_ok,
        "csv_prefix_summary": csv_prefix_summary,
        "passed": a[:-1]==b[:-1] and csv_ok,
    }
    write_json(out_dir/"input_leakage.json", leak)
    evidence.append(str(out_dir/"input_leakage.json"))

    gates = {
        "phi_loss_clawback": {"passed": phi_ok, "evidence_files": [str(out_dir/"q1_phi_clawback.csv")],
                              "numerator": sum(r["passed"] for r in q1_rows), "denominator": len(q1_rows), "unresolved": 0},
        "label_neutrality": {"passed": label_ok, "evidence_files": [str(out_dir/"q2_label_neutrality.json")],
                             "numerator": int(label_ok), "denominator": 1, "unresolved": 0},
        "genuine_progress_not_negative": {"passed": gp["passed"], "evidence_files": [str(out_dir/"q1_genuine_progress.json")],
                                          "numerator": int(gp["passed"]), "denominator": 1, "unresolved": 0},
        "loss_completion_event_pairing": {"passed": pairing_ok, "evidence_files": [str(out_dir/"event_pairing.json")],
                                          "numerator": int(pairing_ok), "denominator": 1, "unresolved": 0},
        "baseline_fairness": {"passed": base["passed"], "evidence_files": [str(out_dir/"baseline_fairness.json")],
                              "numerator": int(base["passed"]), "denominator": 1, "unresolved": 0},
        "raw_provenance": {"passed": raw_ok, "evidence_files": [str(out_dir/"raw_provenance.json"), str(dev/"input_provenance.json")],
                           "numerator": summary.get("episodes_scored"), "denominator": 112, "unresolved": len(unresolved)+len(read_fail)},
        "input_leakage": {"passed": leak["passed"], "evidence_files": [str(out_dir/"input_leakage.json"), str(dev/"prefix_audit.csv")],
                          "numerator": int(leak["passed"]), "denominator": 1, "unresolved": 0 if csv_ok else "csv_prefix_audit_required"},
    }
    write_json(out_dir/"semantic_gates.json", gates)
    return gates
