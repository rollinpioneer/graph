#!/usr/bin/env python3
"""Plan v1.1 Stage 0D: T_B/T_C uniform-masked random reward exposure. Not PPO."""
from __future__ import annotations
import csv, json, math, os, random, statistics, sys, time, traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import yaml

from cp_disr.common import BindingError, digest
from cp_disr.graph import four_views
from cp_disr.neural import Policy
from cp_disr.platforms.libero.runtime_factory import create_task_runtime, PREDICATES, TASK_OBJECTS
from cp_disr.adapters import EvaluationInput
from cp_disr.rl import set_suite_half_life, gamma
from cp_disr.stage2a_p0 import write_json, _log, _read

OUT = ROOT / "runs" / "stage_0d"
REP = ROOT / "reports"
ST = ROOT / "status"
OBS_DIM = 48
CAND_DIM = 8

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def git_hash():
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()

def rms(t):
    if t is None:
        return None
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x * x)))

def contract_by_id(snapshot, cid):
    return next(c for c in snapshot.template.contracts if c.id == cid)

def nominal_nonempty(snapshot, cid):
    contract = contract_by_id(snapshot, cid)
    k, h, ki, hi = four_views(snapshot.template, snapshot.facts.values, snapshot.prior_edges, contract)
    return ki.values != k.values

def write_runtime_manifest():
    cal = json.loads((ROOT / "runs/stage_0a/reference_execution_manifest.json").read_text(encoding="utf-8"))
    d0 = _read(ROOT / "experiments/manifests/runtime_manifest.yaml")
    rt = dict(d0)
    runtime = dict(d0.get("runtime") or {})
    timeouts = dict(runtime.get("skill_timeouts") or {})
    deadlines = dict(runtime.get("task_deadlines") or {})
    drefs = dict(runtime.get("reference_skill_seconds_by_task") or {})
    splits = dict(runtime.get("task_splits") or {})
    tmap = {"OPEN": 16.0, "PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}
    timeouts["D0"] = timeouts.get("D0") or tmap
    timeouts["T_B"] = tmap
    timeouts["T_C"] = tmap
    deadlines["D0"] = float(deadlines.get("D0") or 43.0)
    deadlines["T_B"] = 60.0
    deadlines["T_C"] = 60.0
    drefs["D0"] = float(cal["families"]["D0"]["d_ref_median"])
    drefs["T_B"] = float(cal["families"]["T_B"]["d_ref_median"])
    drefs["T_C"] = float(cal["families"]["T_C"]["d_ref_median"])
    splits["D0"] = "configs/splits/D0_stage_1a.json"
    splits["T_B"] = "configs/splits/T_B_stage_0a.json"
    splits["T_C"] = "configs/splits/T_C_stage_2a.json"
    runtime["skill_timeouts"] = timeouts
    runtime["task_deadlines"] = deadlines
    runtime["reference_skill_seconds_by_task"] = drefs
    runtime["task_splits"] = splits
    runtime["environment_version"] = "plan-v1.1-method-2.1.1"
    runtime["suite_H_seconds"] = cal["H"]
    assets = dict(runtime.get("task_assets") or {})
    assets["T_B"] = "src/cp_disr/platforms/libero/d0_env.py"
    assets["T_C"] = assets.get("T_C") or "src/cp_disr/platforms/libero/d0_env.py"
    runtime["task_assets"] = assets
    runtime["stage_2a_contract_path"] = "configs/runtime/stage_2a_contract_registry.yaml"
    rt["runtime"] = runtime
    rt["manifest_status"] = "PLAN_V1_1_PART0_RUNTIME"
    rt["document_version"] = "3.1"
    rt["method_version"] = "2.1.1"
    rt["plan_version"] = "1.1"
    rt["vlm_runtime"] = d0.get("vlm_runtime")
    path = ROOT / "experiments/manifests/runtime_manifest_v211.yaml"
    path.write_text(yaml.safe_dump(rt, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path, cal["H"], drefs

def model_kwargs(template, method):
    actions = sorted({c.name for c in template.contracts})
    predicates = sorted(PREDICATES)
    types = sorted(set(TASK_OBJECTS["T_B"].values()))
    return dict(actions=actions, predicates=predicates, types=types, observation_dim=OBS_DIM, candidate_dim=CAND_DIM, method=method, B=0.5)

def sample_action(snapshot, rng, contract_aware):
    legal = [i for i, m in enumerate(snapshot.mask) if m]
    if not legal:
        return None, None, legal, []
    nonempty = [i for i in legal if nominal_nonempty(snapshot, snapshot.candidate_ids[i])]
    pool = nonempty if (contract_aware and nonempty) else legal
    idx = pool[rng.randrange(len(pool))]
    return snapshot.candidate_ids[idx], idx, legal, nonempty

def diagnostic_forward(policy, snapshot):
    if policy is None:
        return None
    with torch.no_grad():
        out = policy(snapshot)
    rows = []
    diffs = (out.diagnostics or {}).get("differences") or {}
    delta_map = (out.diagnostics or {}).get("delta") or {}
    for i, cid in enumerate(snapshot.candidate_ids):
        if not snapshot.mask[i]:
            continue
        if not snapshot.prior_edges:
            continue
        if not nominal_nonempty(snapshot, cid):
            continue
        diff = diffs.get(cid)
        residual = delta_map.get(cid)
        rows.append({
            "candidate_id": cid,
            "dp_rms": None if diff is None else rms(diff.dp),
            "residual": None if residual is None else float(residual.detach()),
        })
    return rows

def run_episode(bundle, case_id, rng, timeouts, contract_aware, diag_policy):
    snap = bundle.start_case(case_id)
    collector_start = bundle.clock.now_seconds()
    decisions = []
    n_force = 0
    n_empty_patch = 0
    n_ge2 = 0
    branches = []
    prior_n = len(snap.prior_edges or ())
    prior_empty = prior_n == 0
    last_reason = "NONE"
    success = False
    dp_rows = []
    loop_guard = Counter()
    while True:
        now = float(bundle.clock.now_seconds())
        deadline = float(bundle.evaluator.deadline)
        remaining = deadline - now
        if remaining <= 0:
            actual = bundle.evaluator.evaluate(EvaluationInput(bundle.task_id, snap.env_id, snap.episode_id, (), max(now, deadline), now, max(now, deadline)))
            last_reason = actual.reason
            success = bool(actual.success)
            break
        cid, idx, legal, nonempty = sample_action(snap, rng, contract_aware)
        n_legal = len(legal)
        branches.append(n_legal)
        if n_legal >= 2:
            n_ge2 += 1
        if n_legal == 1:
            n_force += 1
        if cid is None:
            reason = bundle.safety.end_no_candidates(snap.env_id, snap.episode_id)
            last_reason = reason
            break
        if cid not in nonempty:
            n_empty_patch += 1
        loop_guard[cid] += 1
        observation = bundle.observations.observe()
        if not bundle.safety.can_execute(cid, observation):
            raise BindingError("Safety/mask disagreement before execution")
        contract = contract_by_id(snap, cid)
        skill_timeout = float(contract.timeout_seconds)
        capped = min(skill_timeout, max(remaining, 1.0 / 20.0))
        start = bundle.clock.now_seconds()
        execution = bundle.executor.execute(cid, capped)
        observation = bundle.observations.observe()
        measured = bundle.perception.infer(observation)
        facts = bundle.verifier.verify(measured, execution)
        end = bundle.clock.now_seconds()
        duration = bundle.clock.duration_seconds(start, end)
        gamma(duration)
        elapsed = end - bundle.episode_start_seconds
        actual = bundle.evaluator.evaluate(EvaluationInput(bundle.task_id, snap.env_id, snap.episode_id, execution.evidence_ids, elapsed, start, end))
        last_reason = actual.reason
        success = bool(actual.success)
        drows = diagnostic_forward(diag_policy, snap)
        if drows:
            dp_rows.extend(drows)
        decisions.append({
            "candidate_id": cid,
            "n_legal": n_legal,
            "n_nonempty_patch": len(nonempty),
            "duration": duration,
            "reason": actual.reason,
            "success": success,
        })
        if actual.terminated or actual.success:
            break
        snap = bundle.snapshot_builder.build(snap, facts, observation, execution, end)
        if len(decisions) >= 40:
            last_reason = "MAX_DECISIONS"
            break
    end_ep = bundle.clock.now_seconds()
    return {
        "case_id": case_id,
        "success": success,
        "reason": last_reason,
        "n_decisions": len(decisions),
        "n_ge2": n_ge2,
        "n_force": n_force,
        "n_empty_patch": n_empty_patch,
        "mean_branch": statistics.mean(branches) if branches else 0.0,
        "prior_edge_count": prior_n,
        "prior_empty": prior_empty,
        "episode_seconds": float(end_ep - collector_start),
        "loop_max": max(loop_guard.values()) if loop_guard else 0,
        "dp_rows": dp_rows,
        "decisions": decisions,
    }

def family_verdict(successes, success_cases):
    n = len(successes)
    ncase = len(success_cases)
    if n >= 3 and ncase >= 2:
        return "PASS"
    if 1 <= n <= 2 or (n >= 3 and ncase < 2):
        return "PASS_WITH_NOTES"
    if n == 0:
        return "NEEDS_RERUN"
    return "PASS_WITH_NOTES"

def run_family(task_id, manifest, H, seed, log_path):
    set_suite_half_life(H)
    man = yaml.safe_load(Path(manifest).read_text(encoding="utf-8"))
    man["runtime"]["active_task_id"] = task_id
    bundle = create_task_runtime(man, task_id)
    split = json.loads((ROOT / man["runtime"]["task_splits"][task_id]).read_text(encoding="utf-8"))
    pool = [r["case_id"] for r in split["dev"]]
    rng = random.Random(seed)
    template = bundle.template
    diag = Policy(**model_kwargs(template, "Full"))
    diag.eval()
    episodes = []
    def run_n(n, contract_aware, tag, start_i):
        local = []
        for i in range(n):
            case_id = pool[rng.randrange(len(pool))]
            rec = run_episode(bundle, case_id, rng, man["runtime"]["skill_timeouts"][task_id], contract_aware, diag)
            rec.update({"task_id": task_id, "index": start_i + i, "policy": tag, "split": "dev_preflight_with_replacement"})
            local.append(rec)
            with log_path.open("a", encoding="utf-8") as f:
                slim = {k: rec[k] for k in rec if k not in ("decisions", "dp_rows")}
                slim["n_dp_rows"] = len(rec.get("dp_rows") or [])
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
            _log("%s %s %s success=%s reason=%s T=%.2f" % (task_id, tag, rec["case_id"], rec["success"], rec["reason"], rec["episode_seconds"]))
        return local
    try:
        batch = run_n(100, False, "UniformMaskedRandom", 0)
        episodes.extend(batch)
        succ = [e for e in episodes if e["success"]]
        cases = {e["case_id"] for e in succ}
        expanded = False
        contract_eps = []
        if len(succ) <= 2:
            extra = run_n(100, False, "UniformMaskedRandom", 100)
            episodes.extend(extra)
            expanded = True
            succ = [e for e in episodes if e["success"]]
            cases = {e["case_id"] for e in succ}
            if len(succ) <= 2:
                contract_eps = run_n(50, True, "ContractAwareRandom_diagnostic", 200)
        verdict = family_verdict([e for e in episodes if e["success"]], {e["case_id"] for e in episodes if e["success"]})
        if verdict == "NEEDS_RERUN" and contract_eps and any(e["success"] for e in contract_eps):
            # contract-aware is diagnostic only; uniform still 0
            pass
        summary = {
            "task_id": task_id,
            "planned_uniform": 100,
            "expanded_uniform": expanded,
            "uniform_n": len(episodes),
            "uniform_success": sum(1 for e in episodes if e["success"]),
            "uniform_success_cases": sorted({e["case_id"] for e in episodes if e["success"]}),
            "infra_exceptions": 0,
            "contract_aware_n": len(contract_eps),
            "contract_aware_success": sum(1 for e in contract_eps if e["success"]),
            "mean_decisions": statistics.mean(e["n_decisions"] for e in episodes) if episodes else 0,
            "mean_branch": statistics.mean(e["mean_branch"] for e in episodes) if episodes else 0,
            "ge2_rate": (sum(e["n_ge2"] for e in episodes) / max(1, sum(e["n_decisions"] for e in episodes))),
            "force_rate": (sum(e["n_force"] for e in episodes) / max(1, sum(e["n_decisions"] for e in episodes))),
            "empty_patch_rate": (sum(e["n_empty_patch"] for e in episodes) / max(1, sum(e["n_decisions"] for e in episodes))),
            "prior_nonempty_rate": sum(1 for e in episodes if not e["prior_empty"]) / max(1, len(episodes)),
            "failure_reasons": dict(Counter(e["reason"] for e in episodes if not e["success"])),
            "success_seconds": [e["episode_seconds"] for e in episodes if e["success"]],
            "fail_seconds": [e["episode_seconds"] for e in episodes if not e["success"]],
            "dp_n": sum(len(e.get("dp_rows") or []) for e in episodes + contract_eps),
            "verdict": verdict,
            "contract_aware_episodes_logged": len(contract_eps),
        }
        if summary["uniform_success"] == 0 and summary["contract_aware_n"] == 50:
            summary["verdict"] = "NEEDS_RERUN"
            summary["execution_reason"] = "NEEDS_TASK_REVIEW"
        return summary, episodes, contract_eps
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    ST.mkdir(parents=True, exist_ok=True)
    started = utc()
    man_path, H, drefs = write_runtime_manifest()
    set_suite_half_life(H)
    log_path = OUT / "random_episode_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    families = {}
    all_eps = []
    for i, task_id in enumerate(("T_B", "T_C")):
        _log("0D start %s" % task_id)
        summary, eps, ceps = run_family(task_id, man_path, H, seed=9100 + i, log_path=log_path)
        families[task_id] = summary
        all_eps.extend(eps)
        all_eps.extend(ceps)
    verdicts = [families[t]["verdict"] for t in ("T_B", "T_C")]
    if "BLOCKED" in verdicts:
        status = "BLOCKED"
    elif "NEEDS_RERUN" in verdicts and all(v in ("NEEDS_RERUN", "PASS_WITH_NOTES", "PASS") for v in verdicts):
        # family-level NEEDS_RERUN maps to stage NEEDS_RERUN
        status = "NEEDS_RERUN" if any(v == "NEEDS_RERUN" for v in verdicts) and not any(v == "PASS" for v in verdicts) else (
            "PASS_WITH_NOTES" if any(v == "PASS_WITH_NOTES" for v in verdicts) or "NEEDS_RERUN" in verdicts else "PASS"
        )
        if "NEEDS_RERUN" in verdicts and "PASS" in verdicts:
            status = "PASS_WITH_NOTES"
        elif set(verdicts) == {"NEEDS_RERUN"}:
            status = "NEEDS_RERUN"
    elif all(v == "PASS" for v in verdicts):
        status = "PASS"
    else:
        status = "PASS_WITH_NOTES"
    exec_reason = None
    if status == "NEEDS_RERUN":
        exec_reason = "NEEDS_TASK_REVIEW"
    # exposure CSV
    with (OUT / "reward_exposure.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "uniform_n", "uniform_success", "n_success_cases", "verdict", "expanded", "contract_aware_n", "contract_aware_success"])
        w.writeheader()
        for t, s in families.items():
            w.writerow({
                "task_id": t,
                "uniform_n": s["uniform_n"],
                "uniform_success": s["uniform_success"],
                "n_success_cases": len(s["uniform_success_cases"]),
                "verdict": s["verdict"],
                "expanded": s["expanded_uniform"],
                "contract_aware_n": s["contract_aware_n"],
                "contract_aware_success": s["contract_aware_success"],
            })
    with (OUT / "decision_structure.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "mean_decisions", "mean_branch", "ge2_rate", "force_rate", "empty_patch_rate", "prior_nonempty_rate", "dp_n"])
        w.writeheader()
        for t, s in families.items():
            w.writerow({k: s.get(k) for k in ["task_id", "mean_decisions", "mean_branch", "ge2_rate", "force_rate", "empty_patch_rate", "prior_nonempty_rate", "dp_n"]})
    (OUT / "task_revision_log.jsonl").write_text("", encoding="utf-8")
    doc = {
        "stage": "0D",
        "plan_version": "1.1",
        "method_version": "2.1.1",
        "document_version": "3.1",
        "status": status,
        "execution_reason": exec_reason,
        "git_hash": git_hash(),
        "started_at": started,
        "completed_at": utc(),
        "new_rl_runs": 0,
        "H_seconds": H,
        "families": families,
        "random_success_5pct_gate_used": False,
        "ppo_or_bc": False,
        "runtime_manifest": str(man_path.relative_to(ROOT)),
    }
    write_json(OUT / "stage_0d_result.json", doc)
    write_json(ST / "stage_0d.json", doc)
    md = [
        "# Stage 0D — Reward Exposure and Decision Structure (Plan v1.1 / Method 2.1.1)",
        "",
        "Status: `" + status + "`.",
        "",
        "Random success is an exposure proxy, not a 5% learnability theorem. No PPO/BC. No VLM ranking / Q / hidden-correct-action sampling.",
        "",
    ]
    for t, s in families.items():
        md.append("## %s" % t)
        md.append("")
        md.append("- uniform successes: %s / %s across cases %s" % (s["uniform_success"], s["uniform_n"], s["uniform_success_cases"]))
        md.append("- family verdict: `%s`" % s["verdict"])
        md.append("- expanded to 200: %s; contract-aware extra: %s (success %s)" % (s["expanded_uniform"], s["contract_aware_n"], s["contract_aware_success"]))
        md.append("- mean decisions=%s mean branch=%s ge2_rate=%s force_rate=%s empty_patch_rate=%s prior_nonempty_rate=%s" % (
            s["mean_decisions"], s["mean_branch"], s["ge2_rate"], s["force_rate"], s["empty_patch_rate"], s["prior_nonempty_rate"]))
        md.append("- failure reasons: %s" % s["failure_reasons"])
        md.append("")
    md.append("New-profile RL training runs this stage: **0**.")
    md.append("")
    (REP / "stage_0d_summary.md").write_text("\n".join(md) + "\n")
    (REP / "stage_0d_summary.md")  # noqa
    print(json.dumps({"status": status, "families": {t: {"verdict": families[t]["verdict"], "success": families[t]["uniform_success"], "n": families[t]["uniform_n"]} for t in families}}, ensure_ascii=False))

if __name__ == "__main__":
    main()
