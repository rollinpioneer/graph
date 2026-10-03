"""CP-DISR-TP-EF-SCRIPTED-WITNESS-1: post-OPEN fixed-script physical consequence witnesses (max 8 branches).

Two saved, R2-validated post-OPEN snapshots (T_B_dev_03, T_B_dev_05); two frozen 4-skill routes (T, S); 2 repeats.
No provider, RL, optimizer, test; no OPEN; no stepwise planner; no retries; no replacement state.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cp_disr.analysis import tp_ef_post_open_capture as cap
from cp_disr.analysis import tp_ef_post_open_restore_r2 as r2
from cp_disr.analysis import tp_ef_protocol_review as pr

CARD = "CP-DISR-TP-EF-SCRIPTED-WITNESS-1"
BASELINE = "b8af10315985a25c915eed37c15c4f15869f8d02"
CAPTURE_RUN = "runs/final_master/S4/tp_ef_post_open_capture/20261003T014215Z_88f2a168"
R2_RUN = "runs/final_master/S4/tp_ef_post_open_restore_r2/20261003T015733Z_3c09b627"
DEV03_DIR = "/home/xushijie2/graph_cp_disr_snapshots/tp_ef_post_open_capture/20261003T014215Z/T_B_dev_03"
DEV05_DIR = "/home/xushijie2/graph_cp_disr_snapshots/tp_ef_post_open_restore_r2/20261003T015733Z/snap/T_B_dev_05"
GPU_INDEX = 0
GPU_UUID = "GPU-4120c94f-c9c2-6653-a0ec-41957aa85822"
H = 23.1
DEADLINE = 60.0
EPS_G_FLOOR = 0.02
EPS_T_FLOOR = 2.1
ROUTES = {"T": ["pt", "plt", "ps", "pbs"], "S": ["ps", "pbs", "pt", "plt"]}
ROUTE_IDS = {r: [pr.IDS[n] for n in names] for r, names in ROUTES.items()}
ORDER = [("A", "A1", "T_B_dev_03", "T", 0), ("A", "A2", "T_B_dev_03", "S", 0), ("A", "A3", "T_B_dev_03", "T", 1), ("A", "A4", "T_B_dev_03", "S", 1),
         ("B", "B1", "T_B_dev_05", "T", 0), ("B", "B2", "T_B_dev_05", "S", 0), ("B", "B3", "T_B_dev_05", "T", 1), ("B", "B4", "T_B_dev_05", "S", 1)]
CAPS = {"physical_branch_attempts": 8, "environment_constructions": 8, "start_case_calls": 8, "scripted_skill_calls": 32, "skill_retries": 0, "provider_requests": 0,
        "rl_transitions": 0, "optimizer_steps": 0, "test_episodes": 0, "elastic": 0, "open_skill_calls": 0}
OUT_DIR_REL = "runs/final_master/S4/tp_ef_scripted_witness"
PRIOR_DIRS = r2.PRIOR_DIRS + ("runs/final_master/S4/tp_ef_post_open_restore_r2",)


def branch_id_of(case, route, repeat):
    return hashlib.sha256(f"{CARD}|{case}|{route}|{repeat}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------- snapshot identity
def snapshot_specs(root):
    root = Path(root)
    cap_hash = json.loads((root / CAPTURE_RUN / "snapshot_file_hashes.json").read_text())["files"]["T_B_dev_03"]
    r2_cap = json.loads((root / R2_RUN / "dev05_capture_result.json").read_text())["snapshot_files"]
    return {"T_B_dev_03": {"dir": DEV03_DIR, "files": [{"path": f["path"], "bytes": f["bytes"], "sha256": f["sha256_at_capture"]} for f in cap_hash], "source": CAPTURE_RUN + "/snapshot_file_hashes.json"},
            "T_B_dev_05": {"dir": DEV05_DIR, "files": [{"path": f["path"], "bytes": f["bytes"], "sha256": f["sha256"]} for f in r2_cap], "source": R2_RUN + "/dev05_capture_result.json"}}


def snapshot_integrity(specs):
    out = {}
    for case, s in specs.items():
        rows = []
        for f in s["files"]:
            p = Path(f["path"])
            rows.append({"path": f["path"], "bytes_expected": f["bytes"], "bytes_now": p.stat().st_size if p.is_file() else None, "sha256_expected": f["sha256"],
                         "sha256_now": cap.sha_file(p) if p.is_file() else None})
        ok = len(rows) == 6 and all(r["sha256_now"] == r["sha256_expected"] and r["bytes_now"] == r["bytes_expected"] for r in rows) and all(str(Path(r["path"]).parent) == s["dir"] for r in rows)
        out[case] = {"dir": s["dir"], "files": rows, "all_match": ok}
    return out


# ------------------------------------------------------------------------------------- pure analysis
def decision_times(trace_clock_end, decision_start):
    return {"decision_relative_completion_time": trace_clock_end - decision_start, "episode_relative_completion_time": trace_clock_end}


def q_values(decision_rel, episode_rel):
    return {"Q_ref_decision": 2.0 ** (-decision_rel / H), "G_episode": 2.0 ** (-episode_rel / H)}


def epsilons(rows):
    """Within-(state, route) repeat ranges only; computed before any cross-route comparison."""
    groups = {}
    for r in rows:
        if r.get("valid"):
            groups.setdefault((r["case_id"], r["route"]), []).append(r)
    rg, rt, used = 0.0, 0.0, []
    for k, items in sorted(groups.items()):
        if len(items) < 2:
            continue
        qs = [i["Q_ref_decision"] for i in items]
        ts = [i["decision_relative_completion_time"] for i in items]
        rg, rt = max(rg, max(qs) - min(qs)), max(rt, max(ts) - min(ts))
        used.append({"case_id": k[0], "route": k[1], "n": len(items), "Q_range": max(qs) - min(qs), "T_range": max(ts) - min(ts)})
    return {"epsilon_G": max(EPS_G_FLOOR, rg), "epsilon_T": max(EPS_T_FLOOR, rt), "repeat_Q_range_max": rg, "repeat_T_range_max": rt, "groups": used}


def judge_state(case, rows, eps):
    """RELIABLE_PHYSICAL_WITNESS only when all seven frozen conditions hold."""
    mine = [r for r in rows if r["case_id"] == case]
    by = {(r["route"], r["repeat"]): r for r in mine}
    cond = {}
    cond["1_both_routes_2of2_task_success"] = all(by.get((rt, rp)) and by[(rt, rp)].get("valid") and by[(rt, rp)]["task_success"] for rt in "TS" for rp in (0, 1))
    cond["2_exactly_four_skills_each"] = all(by.get((rt, rp)) and by[(rt, rp)]["skill_count"] == 4 for rt in "TS" for rp in (0, 1))
    cond["3_no_anomaly"] = all(by.get((rt, rp)) and by[(rt, rp)].get("valid") and not by[(rt, rp)].get("anomalies") for rt in "TS" for rp in (0, 1))
    out = {"case_id": case, "conditions": cond, "epsilon": {"epsilon_G": eps["epsilon_G"], "epsilon_T": eps["epsilon_T"]}}
    if not (cond["1_both_routes_2of2_task_success"] and cond["2_exactly_four_skills_each"] and cond["3_no_anomaly"]):
        out.update(reliable=False, status="NOT_RELIABLE_EXECUTION_CONDITIONS", winner="NONE", delta_T=None, delta_G=None)
        return out
    winners = []
    for rp in (0, 1):
        t, s = by[("T", rp)], by[("S", rp)]
        dt = t["decision_relative_completion_time"] - s["decision_relative_completion_time"]
        winners.append("T" if dt < 0 else ("S" if dt > 0 else "TIE"))
    mean = lambda rt, k: sum(by[(rt, rp)][k] for rp in (0, 1)) / 2.0
    dT = mean("T", "decision_relative_completion_time") - mean("S", "decision_relative_completion_time")
    dG = mean("T", "Q_ref_decision") - mean("S", "Q_ref_decision")
    cond["4_repeat_winner_direction_consistent"] = len(set(winners)) == 1 and winners[0] != "TIE"
    cond["5_abs_delta_T_gt_epsilon_T"] = abs(dT) > eps["epsilon_T"]
    cond["6_abs_delta_G_gt_epsilon_G"] = abs(dG) > eps["epsilon_G"]
    wt = "T" if dT < 0 else ("S" if dT > 0 else "TIE")
    wg = "T" if dG > 0 else ("S" if dG < 0 else "TIE")
    cond["7_shorter_time_and_higher_Q_same_route"] = wt == wg and wt != "TIE"
    reliable = all(cond.values())
    out.update(reliable=reliable, status="RELIABLE_PHYSICAL_WITNESS" if reliable else "NOT_RELIABLE", winner=wt if reliable else "NONE", per_repeat_winner=winners, delta_T=dT, delta_G=dG,
               mean_time={"T": mean("T", "decision_relative_completion_time"), "S": mean("S", "decision_relative_completion_time")},
               mean_Q={"T": mean("T", "Q_ref_decision"), "S": mean("S", "Q_ref_decision")})
    return out


def mechanism_gate(j03, j05, rows, execution_failure, wave_a_only):
    if execution_failure:
        return {"status": "ENGINEERING_OR_EXECUTION_FAILURE"}
    if not j03["reliable"]:
        return {"status": "INSUFFICIENT_FIRST_WITNESS"}
    if wave_a_only or j05 is None:
        return {"status": "WAVE_B_NOT_RELEASED"}
    if not j05["reliable"]:
        return {"status": "INSUFFICIENT_TWO_WITNESSES"}
    if not (len(rows) == 8 and all(r.get("valid") and r["task_success"] for r in rows)):
        return {"status": "ENGINEERING_OR_EXECUTION_FAILURE"}
    if j03["winner"] == j05["winner"]:
        return {"status": "FIXED_ACTION_BIAS", "winner_both": j03["winner"]}
    return {"status": "PASS", "directions": {"T_B_dev_03": j03["winner"], "T_B_dev_05": j05["winner"]}}


# ------------------------------------------------------------------------------------- route execution
def expected_add_facts(contract):
    return [a.id for a in contract.effects.add]


def fact_map(snap):
    return {k: v.value for k, v in snap.facts.values.items()}


def execute_route(bundle, snap, route_ids, route_name, obs_hash_fn, decision_start, episode_start, contracts):
    """Run exactly the frozen skills in order. Returns (skills list, anomalies list, final_success)."""
    from cp_disr.adapters import EvaluationInput
    skills, anomalies = [], []
    final_success = False
    for i, aid in enumerate(route_ids):
        idx = {c: j for j, c in enumerate(snap.candidate_ids)}
        mask_ok = aid in idx and bool(snap.mask[idx[aid]])
        pre = {"index": i, "action": aid, "facts": fact_map(snap), "candidate_ids": list(snap.candidate_ids), "mask": [bool(m) for m in snap.mask], "mask_for_action": mask_ok,
               "public_obs_hash": obs_hash_fn(), "clock": float(bundle.clock.now_seconds())}
        if not mask_ok:
            anomalies.append(f"MASK_FALSE_BEFORE_SKILL_{i}:{aid}")
            skills.append({"pre": pre, "post": None})
            break
        contract = contracts[aid]
        t0 = float(bundle.clock.now_seconds())
        ex = bundle.executor.execute(aid, float(contract.timeout_seconds))
        raw = dict(getattr(bundle.executor, "last", None) or {})
        obs_obj = bundle.observations.observe()
        measured = bundle.perception.infer(obs_obj)
        records = bundle.verifier.verify(measured, ex)
        end = float(bundle.clock.now_seconds())
        snap = bundle.snapshot_builder.build(snap, records, obs_obj, ex, end)
        task = bundle.evaluator.evaluate(EvaluationInput(task_id=bundle.task_id, env_id=snap.env_id, episode_id=snap.episode_id, evidence_refs=tuple(ex.evidence_ids),
                                                         elapsed_seconds=end - episode_start, interval_start_seconds=float(ex.start_seconds) - episode_start, interval_end_seconds=end - episode_start))
        facts_post = fact_map(snap)
        want = expected_add_facts(contract)
        missing = [f for f in want if facts_post.get(f) != "TRUE"]
        nxt_ok = True
        if i < len(route_ids) - 1:
            idx2 = {c: j for j, c in enumerate(snap.candidate_ids)}
            nxt_ok = route_ids[i + 1] in idx2 and bool(snap.mask[idx2[route_ids[i + 1]]])
        post = {"controller_exit": ex.controller_exit, "duration": end - t0, "clock_end": end, "sim_duration": raw.get("sim_duration"), "steps": raw.get("steps"), "states": raw.get("states"),
                "verifier_records": cap.facts_to_list(records), "facts": facts_post, "candidate_ids": list(snap.candidate_ids), "mask": [bool(m) for m in snap.mask],
                "public_obs_hash": obs_hash_fn(), "evaluator": {"success": bool(task.success), "terminated": bool(task.terminated), "truncated": bool(task.truncated), "reason": task.reason,
                                                               "reward_events": [list(e) for e in task.reward_events]},
                "expected_add_facts": want, "expected_postcondition_missing": missing, "next_action_mask_true": nxt_ok}
        skills.append({"pre": pre, "post": post})
        if ex.controller_exit != "NORMAL_TERMINATION":
            anomalies.append(f"CONTROLLER_EXIT_SKILL_{i}:{ex.controller_exit}")
            break
        if missing:
            anomalies.append(f"POSTCONDITION_NOT_CONFIRMED_SKILL_{i}:{missing}")
            break
        if not nxt_ok:
            anomalies.append(f"NEXT_MASK_FALSE_AFTER_SKILL_{i}")
            break
        last = i == len(route_ids) - 1
        if not last and task.success:
            anomalies.append(f"EARLY_SUCCESS_AFTER_SKILL_{i}")
            break
        if not last and (task.terminated or task.truncated):
            anomalies.append(f"EARLY_TERMINATION_AFTER_SKILL_{i}:{task.reason}")
            break
        if last:
            final_success = bool(task.success)
            if not final_success:
                anomalies.append(f"NO_SUCCESS_AFTER_FINAL_SKILL:{task.reason}")
    if len(skills) != len(route_ids) and not anomalies:
        anomalies.append("SKILL_COUNT_MISMATCH")
    return skills, anomalies, final_success


# --------------------------------------------------------------------------------------- restore gate
def restore_and_gate(bundle, ledger, snap0, case_id, snap_dir, profile):
    """R2 restore criteria before the first skill; leaves the env at the exact restored snapshot state (caches re-applied after the render check)."""
    env = bundle.environment
    snap_dir = Path(snap_dir)
    man = json.loads((snap_dir / "manifest.json").read_text())
    sim_saved = {k: np.array(v) for k, v in np.load(snap_dir / "sim_state.npz").items()}
    py_saved = cap.deserialize_state(man["python_state_meta"], np.load(snap_dir / "python_state.npz"))
    pub_saved = {k: np.array(v) for k, v in np.load(snap_dir / "obs_public_cached.npz").items()}
    forced_saved = {k: np.array(v) for k, v in np.load(snap_dir / "obs_forced_render.npz").items()}
    rng_saved = {"numpy_global": dict(man["rng"]["numpy_global"], key=np.array(np.load(snap_dir / "rng_numpy.npz")["key"])), "python_random": man["rng"]["python_random"]}
    rt = man["runtime_state"]
    checks, fp = {}, {}
    checks["camera_live_vs_frozen"] = r2.camera_live_vs_frozen(env, profile)

    def apply_all(prev_key):
        cap.apply_sim(env, sim_saved)
        miss = cap.load_state(env, {p: v for p, v in py_saved.items() if not (p and p[0][1].startswith("@"))})
        for name, obj in (("robot0", env.robots[0]), ("robot0.controller", env.robots[0].controller), ("robot0.gripper", env.robots[0].gripper)):
            miss += cap.load_state(obj, {p[1:]: v for p, v in py_saved.items() if p and p[0][1] == "@" + name})
        cap.apply_rng(rng_saved)
        cap.apply_py_runtime_state(bundle, dict(rt, verifier_prev=rt[prev_key]))
        return miss

    missing = apply_all("verifier_prev_before_verify")
    raw = cap.compare_sim(sim_saved, cap.capture_sim(env))
    raw["pass"] = bool(raw["max_abs"] <= r2.THRESH["sim_arrays"]["max_abs_diff"])
    checks["sim_arrays_raw"] = raw
    env.sim.forward()
    sn = cap.compare_sim(sim_saved, cap.capture_sim(env), skip=("qacc", "qacc_warmstart"))
    sn["pass"] = bool(sn["max_abs"] <= r2.THRESH["sim_arrays"]["max_abs_diff"])
    checks["sim_arrays"] = sn
    now_clock = float(bundle.clock.now_seconds())
    checks["clock"] = {"pass": bool(now_clock == man["clock"]["now"] and float(env.sim.data.time) == man["clock"]["sim_time"] and float(bundle.clock._origin) == man["clock"]["origin"]), "restored_now": now_clock}
    checks["rng"] = {"pass": cap.rng_hash(cap.rng_snapshot()) == man["hashes"]["rng"]}

    def dump_all():
        d = cap.dump_state(env)
        for name, obj in (("robot0", env.robots[0]), ("robot0.controller", env.robots[0].controller), ("robot0.gripper", env.robots[0].gripper)):
            for p, v in cap.dump_state(obj, skip=cap.SKIP_KEYS - {"robots"}).items():
                d[(("a", "@" + name),) + p] = v
        return d
    diffs = cap.diff_states(py_saved, dump_all())
    checks["controller_gripper_state"] = {"entries": len(py_saved), "diffs_total": len(diffs), "first_diffs": diffs[:10], "pass": not diffs and not missing}
    pub_now = cap.public_obs_arrays(env)
    checks["public_obs_cached_exact"] = {"pass": all(np.array_equal(pub_saved[k], pub_now[k]) and pub_saved[k].dtype == pub_now[k].dtype for k in pub_saved)}
    obs_obj = bundle.observations.observe()
    ex = SimpleNamespace(execution_id="witness-restore-gate", controller_exit="NORMAL_TERMINATION", start_seconds=0.0, end_seconds=0.0, evidence_ids=())
    records = bundle.verifier.verify(bundle.perception.infer(obs_obj), ex)
    facts_now = cap.facts_to_list(records)
    checks["facts_cached_obs"] = {"pass": cap.facts_comparable(facts_now) == cap.facts_comparable(man["facts"])}
    snap_dec = bundle.snapshot_builder.build(snap0, records, obs_obj, ex, float(bundle.clock.now_seconds()))
    goal_now = [[g.fact_id, g.sign] for g in snap_dec.template.goals]
    cand_now = [list(snap_dec.candidate_ids), [bool(m) for m in snap_dec.mask]]
    fm = {f["fact_id"]: f["value"] for f in facts_now}
    idx = {c: i for i, c in enumerate(snap_dec.candidate_ids)}
    req = {"facts": {k: fm.get(k) for k in cap.FACT_REQUIRED}, "pick_masks": {c: bool(snap_dec.mask[idx[c]]) for c in cap.PICK_IDS}}
    req["pass"] = bool(all(fm.get(k) == v for k, v in cap.FACT_REQUIRED.items()) and all(req["pick_masks"].values()))
    checks["goal_candidates_mask"] = {"pass": goal_now == man["goal"] and cand_now == [man["candidate_ids"], man["candidate_mask"]]}
    checks["post_open_requirements"] = req
    # render gate (one fresh render of the restored state), then re-apply the exact saved caches
    env.sim.forward()
    r1 = cap.forced_obs_arrays(env)
    prev_after = cap.prev_from_json(rt["verifier_prev"])

    def facts_of(obs_dict):
        v = type(bundle.verifier)(env)
        v.prev = dict(prev_after)
        pres = bundle.perception.infer(obs_dict)
        return pres, cap.facts_to_list(v.verify(pres, None))
    pres1, facts1 = facts_of(env.public_observation())
    pres_s, facts_s = facts_of(r2.obs_dict_from_arrays(forced_saved))
    m = {"rgb": r2.rgb_metrics(forced_saved["rgb"], r1["rgb"]), "depth_raw": r2.depth_metrics(forced_saved["depth"], r1["depth"], "raw"),
         "depth_metric": r2.depth_metrics(r2.cap_metric_depth(env, forced_saved["depth"]), r2.cap_metric_depth(env, r1["depth"]), "metric"),
         "proprio": r2.vector_metrics(forced_saved["proprio"], r1["proprio"], r2.THRESH["proprio"]["max_abs_diff"])}
    m["perception"] = r2.compare_perception(r2.perception_summary(pres_s), r2.perception_summary(pres1))
    m["facts"] = {"saved_vs_restore_equal": cap.facts_comparable(facts_s) == cap.facts_comparable(facts1), "restore_vs_manifest_forced_equal": cap.facts_comparable(facts1) == cap.facts_comparable(man["facts_forced_render"])}
    m["pass_"] = bool(m["rgb"]["pass_"] and m["depth_raw"]["pass_"] and m["depth_metric"]["pass_"] and m["proprio"]["pass_"] and m["perception"]["pass_"] and all(m["facts"].values()))
    checks["render_gate"] = {"pass": m["pass_"], "metrics": m}
    fp["render1_array_hashes"] = {k: cap.arr_hash(v) for k, v in sorted(r1.items())}
    # re-apply saved state so the first skill starts from the exact snapshot (caches included)
    missing2 = apply_all("verifier_prev")
    env.sim.forward()
    sn2 = cap.compare_sim(sim_saved, cap.capture_sim(env), skip=("qacc", "qacc_warmstart"))
    diffs2 = cap.diff_states(py_saved, dump_all())
    pub2 = cap.public_obs_arrays(env)
    checks["state_after_gate_reapply"] = {"sim_max_abs": sn2["max_abs"], "py_diffs": len(diffs2), "cached_obs_exact": all(np.array_equal(pub_saved[k], pub2[k]) for k in pub_saved), "missing": len(missing2)}
    checks["state_after_gate_reapply"]["pass"] = bool(sn2["max_abs"] <= r2.THRESH["sim_arrays"]["max_abs_diff"] and not diffs2 and checks["state_after_gate_reapply"]["cached_obs_exact"] and not missing2)
    fp.update({"sim": cap.digest({k: cap.arr_hash(v) for k, v in sorted(cap.capture_sim(env).items())}), "py_state": cap.state_hash(dump_all()), "facts": cap.digest(cap.facts_comparable(facts_now)),
               "cached_obs": cap.digest({k: cap.arr_hash(v) for k, v in sorted(pub2.items())}), "state_identity_sha256_saved": man["state_identity_sha256"]})
    failures = [k for k, v in checks.items() if not v.get("pass")]
    return {"checks": checks, "failures": failures, "pass": not failures, "fingerprint": fp, "decision_snapshot": snap_dec,
            "decision_start_clock": float(bundle.clock.now_seconds()), "episode_start": float(bundle.episode_start_seconds)}


# --------------------------------------------------------------------------------------- branch worker
def public_obs_hash(env):
    a = cap.public_obs_arrays(env)
    return cap.digest({k: cap.arr_hash(v) for k, v in sorted(a.items())})


def run_branch(root, out, branch_id):
    root, out = Path(root).resolve(), Path(out).resolve()
    reg = pr.read_json(out / "branch_registration.json")
    b = next(x for x in reg["branches"] if x["branch_id"] == branch_id)
    claims = out / "branch_receipts"
    claims.mkdir(exist_ok=True)
    try:
        os.close(os.open(claims / f"{branch_id}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise RuntimeError("STOPPED_BUDGET: branch already attempted; no retry")
    res = {"card": CARD, "branch_id": branch_id, "order": b["order"], "wave": b["wave"], "case_id": b["case_id"], "route": b["route"], "repeat": b["repeat"], "valid": False,
           "status": "FAIL", "anomalies": [], "task_success": False, "skill_count": 0, "pid": os.getpid()}
    ledger = cap.Ledger()
    try:
        specs = snapshot_specs(root)
        integ = snapshot_integrity({b["case_id"]: specs[b["case_id"]]})
        res["snapshot_integrity_at_start"] = integ[b["case_id"]]["all_match"]
        if not integ[b["case_id"]]["all_match"] or specs[b["case_id"]]["dir"] != b["snapshot_dir"]:
            res["anomalies"].append("SNAPSHOT_HASH_MISMATCH_AT_START")
            return res
        profile = json.loads((root / r2.PROFILE_REL).read_text())
        res["gpu"] = r2.gpu_identity()
        if not res["gpu"] or res["gpu"]["uuid"] != GPU_UUID:
            res["anomalies"].append("GPU_UUID_MISMATCH")
            return res
        bundle, manifest = cap.build_runtime(root, out, b["case_id"], ledger, "wit_" + branch_id)
        snap0, real_env = cap.start_case_reusing_bootstrap(bundle, ledger, b["case_id"])
        env = bundle.environment
        res["renderer"] = r2.gl_renderer_info()
        orig_exec = bundle.executor.execute
        allowed = set(ROUTE_IDS[b["route"]])
        count = {"n": 0}

        def guarded(cid, timeout):
            if cid not in allowed or count["n"] >= 4 or cid == pr.IDS["open"]:
                raise RuntimeError("PROTOCOL_VIOLATION: action outside the frozen route or more than 4 skills")
            count["n"] += 1
            ledger.skill_calls += 1
            return orig_exec(cid, timeout)
        bundle.executor.execute = guarded
        gate = restore_and_gate(bundle, ledger, snap0, b["case_id"], b["snapshot_dir"], profile)
        res["restore_gate"] = {"pass": gate["pass"], "failures": gate["failures"], "checks": gate["checks"], "fingerprint": gate["fingerprint"]}
        if ledger.skill_calls != 0:
            gate["pass"] = False
        if not gate["pass"]:
            res["anomalies"].append("RESTORE_GATE_FAILED:" + ",".join(gate["failures"]))
            return res
        contracts = {c.id: c for c in gate["decision_snapshot"].template.contracts}
        decision_start, episode_start = gate["decision_start_clock"], gate["episode_start"]
        res["timing_start"] = {"decision_start_clock": decision_start, "remaining_deadline": DEADLINE - (decision_start - episode_start)}
        skills, anomalies, success = execute_route(bundle, gate["decision_snapshot"], ROUTE_IDS[b["route"]], b["route"], lambda: public_obs_hash(env), decision_start, episode_start, contracts)
        res["skills"] = skills
        res["anomalies"] += anomalies
        res["skill_count"] = sum(1 for s in skills if s.get("post") is not None)
        res["task_success"] = bool(success)
        if success and not anomalies:
            end = skills[-1]["post"]["clock_end"]
            ep_rel = end - episode_start
            dec_rel = end - decision_start
            res.update({"first_confirmed_success_clock": end, "decision_relative_completion_time": dec_rel, "episode_relative_completion_time": ep_rel, **q_values(dec_rel, ep_rel)})
            res["valid"] = res["skill_count"] == 4
            res["status"] = "PASS" if res["valid"] else "FAIL"
        return res
    except Exception as exc:
        import traceback
        res["anomalies"].append(f"EXCEPTION:{type(exc).__name__}:{exc}")
        res["traceback"] = traceback.format_exc()
        res["status"] = "STOPPED_ENGINEERING"
        return res
    finally:
        res.update(ledger.as_dict())
        res["skill_calls_executed"] = ledger.skill_calls
        try:
            ledger.bootstrap_env.close()
        except Exception:
            pass
        pr.write_json(out / "branches" / f"{branch_id}.json", res)


# ------------------------------------------------------------------------------------ registration / gate
def gpu_state():
    g = r2.gpu_identity(GPU_INDEX)
    return g


def protected_hashes(root):
    root = Path(root)
    out = {}
    for sub in PRIOR_DIRS:
        for p in sorted((root / sub).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(root))] = cap.sha_file(p)
    return out


def register(root, out):
    """Zero-environment frozen registration (committed before any branch runs)."""
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    specs = snapshot_specs(root)
    integ = snapshot_integrity(specs)
    for c, v in integ.items():
        if not v["all_match"]:
            problems.append("SNAPSHOT_HASH_MISMATCH:" + c)
    ident = {}
    for c, s in specs.items():
        man = json.loads((Path(s["dir"]) / "manifest.json").read_text())
        ident[c] = {"snapshot_dir": s["dir"], "hash_source": s["source"], "files": integ[c]["files"], "manifest_sha256": cap.sha_file(Path(s["dir"]) / "manifest.json"),
                    "state_identity_sha256": man["state_identity_sha256"], "state_hashes": man["hashes"], "restore_seed": man["restore_seed"], "clock": man["clock"],
                    "facts": {f["fact_id"]: f["value"] for f in man["facts"]}, "candidate_mask": dict(zip(man["candidate_ids"], man["candidate_mask"]))}
    pr.write_json(out / "snapshot_identity_manifest.json", {"card": CARD, "read_only": True, "snapshots": ident})
    gpu = gpu_state()
    if gpu is None or gpu["uuid"] != GPU_UUID:
        problems.append("GPU_UUID_MISMATCH")
    compat = r2.compat_gate(root)
    pr.write_json(out / "renderer_contract_binding.json", {"card": CARD, "contract": r2.CONTRACT_REL, "evidence_of_applicability": R2_RUN, **compat})
    if not compat["applicable"]:
        problems.append("RENDERER_CONTRACT_NOT_APPLICABLE")
    branches = []
    for i, (wave, label, case, route, rep) in enumerate(ORDER):
        branches.append({"order": label, "index": i, "wave": wave, "branch_id": branch_id_of(case, route, rep), "case_id": case, "route": route, "repeat": rep, "route_actions": ROUTE_IDS[route],
                         "snapshot_dir": specs[case]["dir"], "snapshot_state_identity_sha256": ident[case]["state_identity_sha256"], "snapshot_manifest_sha256": ident[case]["manifest_sha256"],
                         "snapshot_file_sha256": {Path(f["path"]).name: f["sha256_expected"] for f in ident[case]["files"]}, "gpu_index": GPU_INDEX, "gpu_uuid": GPU_UUID})
    assert len({b["branch_id"] for b in branches}) == 8
    pr.write_json(out / "branch_registration.json", {"card": CARD, "frozen_before_first_branch": True, "branches": branches, "caps": CAPS}, mode=0o444)
    auth = {"card": CARD, "baseline": BASELINE, "authorized_by": "user chat instruction approving CP-DISR-TP-EF-SCRIPTED-WITNESS-1", "status": "REGISTERED_FROZEN" if not problems else "STOPPED_REGISTRATION",
            "problems": problems, "caps": CAPS, "states": list(specs), "routes": {r: {"candidate": ids[0], "full_route": ids} for r, ids in ROUTE_IDS.items()}, "order": [o[1] for o in ORDER],
            "restore_gate_thresholds": r2.THRESH, "time_and_return": {"H_seconds": H, "deadline_seconds": DEADLINE, "clock": "original episode clock; no fresh 60 s for the post-OPEN state",
                                                                      "primary": ["decision_relative_completion_time", "Q_ref_decision = 2^(-decision_relative/H)"], "additional": "G_episode = 2^(-episode_relative/H)"},
            "epsilon_rule": {"epsilon_G": f"max({EPS_G_FLOOR}, same-route repeat range of Q_ref_decision)", "epsilon_T": f"max({EPS_T_FLOOR} s, same-route repeat range of decision-relative time)",
                             "computed_before_cross_route_winner": True},
            "reliable_witness_conditions": ["both routes 2/2 TASK_SUCCESS", "exactly 4 skills per branch", "no restore/perception/mask/controller/recorder anomaly", "repeat winner directions agree",
                                            "abs(delta_T) > epsilon_T", "abs(delta_G) > epsilon_G", "shorter time and higher Q_ref point to the same route"],
            "wave_rule": "Wave B is released only if dev_03 is RELIABLE_PHYSICAL_WITNESS (epsilons from dev_03 repeats); final gate uses epsilons over all repeats",
            "mechanism_gate": "PASS only if both states reliable, 8/8 TASK_SUCCESS, opposite winners; same winner = FIXED_ACTION_BIAS; second state insufficient = INSUFFICIENT_TWO_WITNESSES; any execution failure = ENGINEERING_OR_EXECUTION_FAILURE",
            "forbidden": ["OPEN", "stepwise B_PLAN", "skill retry", "recovery action", "state or candidate replacement", "provider", "RL", "optimizer", "test"],
            "physical_gpu": gpu}
    pr.write_json(out / "authorization.json", auth)
    return auth


def gate0(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    problems = []
    auth = pr.read_json(out / "authorization.json")
    if auth["status"] != "REGISTERED_FROZEN":
        problems.append("REGISTRATION_NOT_FROZEN")
    ident = {"user": os.environ.get("USER", ""), "python": sys.executable}
    if ident["user"] != cap.USER or sys.executable != cap.PYTHON:
        problems.append("WRONG_ACCOUNT_OR_PYTHON")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], cwd=root).returncode != 0:
        problems.append("BASELINE_NOT_ANCESTOR")
    if dirty:
        problems.append("DIRTY_TRACKED_TREE")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str((out / "branch_registration.json").relative_to(root))], cwd=root, capture_output=True).returncode == 0
    if not tracked:
        problems.append("REGISTRATION_NOT_COMMITTED")
    for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE"):
        if key in os.environ:
            problems.append("PROVIDER_CREDENTIAL_PRESENT:" + key)
    integ = snapshot_integrity(snapshot_specs(root))
    for c, v in integ.items():
        if not v["all_match"]:
            problems.append("SNAPSHOT_HASH_MISMATCH:" + c)
    gpu = gpu_state()
    if gpu is None or gpu["uuid"] != GPU_UUID or gpu["compute_processes"]:
        problems.append("STOPPED_CAPTURE_GPU_UNAVAILABLE")
    pr.write_json(out / "source_identity.json", {"card": CARD, "baseline": BASELINE, "execution_commit": commit, "dirty_tracked": dirty, "identity": ident, "gpu": gpu,
                                                  "runtime_identity": cap.runtime_identity(root), "registration_sha256": cap.sha_file(out / "branch_registration.json"),
                                                  "authorization_sha256": cap.sha_file(out / "authorization.json"), "snapshot_integrity_at_gate": integ, "problems": problems})
    pr.write_json(out / "budget_ledger.json", {"caps": CAPS, "used": {k: 0 for k in CAPS}, "processes": [], "env_reset_calls_total": 0})
    pr.write_json(out / "protected_hashes_before.json", protected_hashes(root))
    return {"problems": problems}


def _spawn(root, out, branch_id):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(GPU_INDEX), "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(k, None)
    log = Path(out) / "logs" / f"{branch_id}.log"
    log.parent.mkdir(exist_ok=True)
    return subprocess.run([sys.executable, str(Path(root) / "scripts/tp_ef_scripted_witness.py"), "branch", "--root", str(root), "--output", str(out), "--branch-id", branch_id], cwd=root, env=env,
                          stdout=log.open("ab"), stderr=subprocess.STDOUT, timeout=1800).returncode


def row_of(res):
    keys = ("branch_id", "order", "wave", "case_id", "route", "repeat", "valid", "status", "task_success", "skill_count", "decision_relative_completion_time", "episode_relative_completion_time",
            "first_confirmed_success_clock", "Q_ref_decision", "G_episode")
    r = {k: res.get(k) for k in keys}
    r["anomalies"] = res.get("anomalies", [])
    return r


def run_all(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    reg = pr.read_json(out / "branch_registration.json")["branches"]
    ledger = pr.read_json(out / "budget_ledger.json")
    rows, results, seq, stopped = [], {}, [], ""
    execution_failure = False
    wave_a_only = False
    j03 = None
    for wave in ("A", "B"):
        if wave == "B":
            eps_a = epsilons(rows)
            j03 = judge_state("T_B_dev_03", rows, eps_a)
            pr.write_json(out / "wave_a_gate.json", {"epsilon_from_dev03_repeats": eps_a, "judgement": j03})
            if not j03["reliable"]:
                wave_a_only = True
                break
        for b in [x for x in reg if x["wave"] == wave]:
            if ledger["used"]["physical_branch_attempts"] >= CAPS["physical_branch_attempts"]:
                stopped = "STOPPED_BUDGET"
                break
            code = _spawn(root, out, b["branch_id"])
            rf = out / "branches" / f"{b['branch_id']}.json"
            res = pr.read_json(rf) if rf.is_file() else {"status": "FAIL", "anomalies": ["NO_RESULT_FILE"], "valid": False}
            ledger["used"]["physical_branch_attempts"] += 1
            ledger["used"]["environment_constructions"] += res.get("environment_constructions", 0)
            ledger["used"]["start_case_calls"] += res.get("start_case_calls", 0)
            ledger["used"]["scripted_skill_calls"] += res.get("skill_calls_executed", 0)
            ledger["env_reset_calls_total"] += res.get("env_reset_calls_total", 0)
            ledger["processes"].append({"order": b["order"], "branch_id": b["branch_id"], "status": res["status"], "constructions": res.get("environment_constructions"),
                                        "start_case": res.get("start_case_calls"), "env_reset_calls": res.get("env_reset_calls_total"), "skills": res.get("skill_calls_executed")})
            pr.write_json(out / "budget_ledger.json", ledger)
            results[b["branch_id"]] = res
            rows.append(row_of(res))
            seq.append({"order": b["order"], "status": res["status"], "exit": code})
            if res["status"] != "PASS":
                stopped = f"STOPPED_{b['order']}:{res['status']}:{res.get('anomalies')}"
                execution_failure = True
                break
        if stopped:
            break
    over = [k for k, v in CAPS.items() if ledger["used"][k] > v]
    return {"stopped": stopped, "sequence": seq, "caps_exceeded": over, "execution_failure": execution_failure, "wave_a_only": wave_a_only, "rows": rows}


# ------------------------------------------------------------------------------------- finalization
def finalize(root, out, summary):
    root, out = Path(root).resolve(), Path(out).resolve()
    reg = pr.read_json(out / "branch_registration.json")["branches"]
    results = {b["branch_id"]: pr.read_json(out / "branches" / f"{b['branch_id']}.json") for b in reg if (out / "branches" / f"{b['branch_id']}.json").is_file()}
    rows = [row_of(r) for b in reg for r in [results.get(b["branch_id"])] if r]
    with (out / "per_branch_restore_metrics.jsonl").open("w") as f:
        for b in reg:
            r = results.get(b["branch_id"])
            if r:
                f.write(json.dumps({"order": r["order"], "branch_id": r["branch_id"], "case_id": r["case_id"], "restore_gate": r.get("restore_gate"), "gpu": r.get("gpu"), "renderer": r.get("renderer"),
                                    "snapshot_integrity_at_start": r.get("snapshot_integrity_at_start")}, sort_keys=True, default=str) + "\n")
    with (out / "per_skill_facts_and_masks.jsonl").open("w") as f:
        for b in reg:
            r = results.get(b["branch_id"])
            for s in (r or {}).get("skills", []):
                f.write(json.dumps({"order": r["order"], "branch_id": r["branch_id"], "case_id": r["case_id"], "route": r["route"], "repeat": r["repeat"], **s}, sort_keys=True, default=str) + "\n")
    fields = ["order", "wave", "branch_id", "case_id", "route", "repeat", "status", "valid", "task_success", "skill_count", "decision_relative_completion_time", "episode_relative_completion_time",
              "first_confirmed_success_clock", "Q_ref_decision", "G_episode", "anomalies"]
    with (out / "branch_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r.get(k)) if k == "anomalies" else r.get(k)) for k in fields})
    # paired restore: restored-state fingerprints must agree across all branches of one state
    paired = {}
    for case in ("T_B_dev_03", "T_B_dev_05"):
        fps = [(r["order"], r["restore_gate"]["fingerprint"]) for r in results.values() if r["case_id"] == case and r.get("restore_gate")]
        keys = ("sim", "py_state", "facts", "cached_obs", "state_identity_sha256_saved")
        paired[case] = {"branches": [o for o, _ in fps], "identical_across_branches": {k: len({fp.get(k) for _, fp in fps}) <= 1 for k in keys},
                        "render1_rgb_hash_distinct_values": len({fp.get("render1_array_hashes", {}).get("rgb") for _, fp in fps}) if fps else 0,
                        "render_gate_rgb_max_abs": [r["restore_gate"]["checks"]["render_gate"]["metrics"]["rgb"].get("max_abs") for r in results.values() if r["case_id"] == case and r.get("restore_gate") and "render_gate" in r["restore_gate"]["checks"]]}
    pr.write_json(out / "paired_restore.json", {"card": CARD, "per_state": paired, "all_restore_gates_pass": all(r.get("restore_gate", {}).get("pass") for r in results.values()) and bool(results)})
    valid_rows = [r for r in rows if r["valid"]]
    eps_final = epsilons(valid_rows)
    pr.write_json(out / "repeatability.json", {"card": CARD, "final_epsilon": eps_final, "wave_a_epsilon": (pr.read_json(out / "wave_a_gate.json")["epsilon_from_dev03_repeats"] if (out / "wave_a_gate.json").is_file() else None),
                                               "same_route_repeat_details": eps_final["groups"]})
    j = {c: judge_state(c, valid_rows, eps_final) for c in ("T_B_dev_03", "T_B_dev_05") if any(r["case_id"] == c for r in rows)}
    pr.write_json(out / "physical_utility.json", {"card": CARD, "H": H, "per_state": j, "rows": rows, "epsilon_final": {"epsilon_G": eps_final["epsilon_G"], "epsilon_T": eps_final["epsilon_T"]}})
    gate = mechanism_gate(j.get("T_B_dev_03", {"reliable": False}), j.get("T_B_dev_05"), rows, summary["execution_failure"], summary["wave_a_only"])
    if gate["status"] == "WAVE_B_NOT_RELEASED":
        gate = {"status": "INSUFFICIENT_FIRST_WITNESS"}
    gate.update({"wave_a_only": summary["wave_a_only"], "stopped": summary["stopped"], "caps_exceeded": summary["caps_exceeded"], "branches_run": len(rows)})
    pr.write_json(out / "mechanism_gate.json", gate)
    (out / "provider_revision_request.md").write_text(provider_md(gate, j), encoding="utf-8")
    (out / "final_summary.md").write_text(summary_md(gate, j, rows, eps_final, pr.read_json(out / "budget_ledger.json")), encoding="utf-8")
    return gate


def provider_md(gate, j):
    if gate["status"] != "PASS":
        return f"# Provider revision request — {CARD}\n\nStatus: **NOT_REQUESTED**\n\nMechanism gate: `{gate['status']}`. No provider call is requested; nothing was called.\n"
    return "\n".join([f"# Provider revision request — {CARD}", "", "Status: **REQUESTED_NOT_EXECUTED**", "",
                      f"Mechanism gate PASS: winners reverse across states ({gate['directions']}), both states are RELIABLE_PHYSICAL_WITNESS, 8/8 TASK_SUCCESS.", "",
                      "Requested: ONE_EVIDENCE_BASED_PROVIDER_PROMPT_REVISION (credential and explicit authorization required; none was used by this card).", "",
                      "Evidence for the revision: `physical_utility.json` (per-state winners, deltas, epsilons) and `branch_results.csv`.", ""])


def summary_md(gate, j, rows, eps, ledger):
    L = [f"# {CARD} — final summary", "", f"Mechanism gate: **{gate['status']}**", f"Stopped: {gate.get('stopped') or 'no'}; branches run: {gate['branches_run']}; caps exceeded: {gate.get('caps_exceeded') or 'none'}", ""]
    for c, v in j.items():
        L += [f"## {c}: {v.get('status')}  winner={v.get('winner')}", f"- conditions: {v.get('conditions')}", f"- delta_T={v.get('delta_T')} delta_G={v.get('delta_G')} per-repeat winners={v.get('per_repeat_winner')}",
              f"- mean decision-relative time: {v.get('mean_time')}; mean Q_ref_decision: {v.get('mean_Q')}", ""]
    L += ["## Branches", ""] + [f"- {r['order']} {r['case_id']} route {r['route']} rep{r['repeat']}: {r['status']} success={r['task_success']} skills={r['skill_count']} decision_rel={r['decision_relative_completion_time']} Q={r['Q_ref_decision']} anomalies={r['anomalies']}" for r in rows]
    L += ["", f"epsilon_G={eps['epsilon_G']} epsilon_T={eps['epsilon_T']}", "", f"Budget used: {ledger['used']}", f"env.reset() invocations (incl. bootstrap resets): {ledger.get('env_reset_calls_total')}", ""]
    return "\n".join(L)


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["authorization.json", "source_identity.json", "snapshot_identity_manifest.json", "branch_registration.json", "budget_ledger.json", "per_branch_restore_metrics.jsonl",
            "per_skill_facts_and_masks.jsonl", "branch_results.csv", "paired_restore.json", "repeatability.json", "physical_utility.json", "mechanism_gate.json", "provider_revision_request.md", "final_summary.md"]
    ledger = pr.read_json(out / "budget_ledger.json")
    before = pr.read_json(out / "protected_hashes_before.json")
    integ_after = snapshot_integrity(snapshot_specs(root))
    gate = pr.read_json(out / "mechanism_gate.json")
    prov = (out / "provider_revision_request.md").read_text()
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".log", ".jsonl", ".csv"} and bool(__import__("re").search(r"sk-[A-Za-z0-9]{16,}|Authorization:|Bearer [A-Za-z0-9._-]{16,}|DASHSCOPE_API_KEY=", p.read_text(errors="ignore"))))
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "caps_respected": all(ledger["used"][k] <= v for k, v in CAPS.items()),
              "zero_provider_rl_optimizer_test_open": all(ledger["used"][k] == 0 for k in ("provider_requests", "rl_transitions", "optimizer_steps", "test_episodes", "open_skill_calls", "skill_retries")),
              "snapshots_hash_unchanged_after_card": all(v["all_match"] for v in integ_after.values()), "prior_cards_unchanged": before == protected_hashes(root),
              "provider_request_label_consistent": ("REQUESTED_NOT_EXECUTED" in prov) == (gate["status"] == "PASS"), "no_secret_shaped_content": secrets == 0,
              "registration_unchanged": pr.read_json(out / "source_identity.json")["registration_sha256"] == cap.sha_file(out / "branch_registration.json")}
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k != "outputs_present") and all(checks["outputs_present"].values()) else "FAIL"
    pr.write_json(out / "verify.json", checks)
    return checks
