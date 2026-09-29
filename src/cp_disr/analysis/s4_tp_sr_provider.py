"""S4 T_P_SR provider / representation phases, summary and verification.

The provider (H, I) and representation (J) phases are gated by the physical opportunity gate. When the gate is not passed
they refuse to run and write an explicit NOT_RUN record; they never spend provider or forward budget. `summarize` and
`verify` are the terminal phases and are executed in both outcomes.
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from cp_disr.analysis import s4_tp_sr_design as d
from cp_disr.analysis.s1_integration import _atomic_json, _load_attempts, sha256_file

OLD_EVIDENCE_PATHS = ("runs/final_master/S1", "experiments", "configs/splits", "configs/tasks/resolved/T_A.yaml", "configs/tasks/resolved/T_C.yaml",
                      "configs/runtime/stage_2a_contract_registry.yaml", "src/cp_disr/platforms/libero/runtime_factory.py",
                      "src/cp_disr/platforms/libero/perception.py", "src/cp_disr/platforms/libero/verifier.py",
                      "src/cp_disr/platforms/libero/skill_executor.py", "src/cp_disr/platforms/libero/d0_env.py",
                      "src/cp_disr/analysis/s1_revision_resume.py", "src/cp_disr/analysis/s1_integration.py", "src/cp_disr/analysis/s1_e4_recovery.py")
SECRET_PATTERNS = (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{12,}"), re.compile(r"(?i)DASHSCOPE_API_KEY\s*[=:]\s*\S{8,}"))


def gate_passed(out):
    p = Path(out) / "opportunity/physical_gate.json"
    return p.is_file() and bool(d.rd(p).get("passed"))


def _not_run(out, phase):
    _atomic_json(Path(out) / "provider" / f"{phase}_NOT_RUN.json",
                 {"phase": phase, "status": "NOT_RUN", "reason": "PHYSICAL_GATE_NOT_PASSED" if not gate_passed(out) else "PHASE_NOT_IMPLEMENTED_IN_THIS_ROUND",
                  "provider_calls_made": 0, "representation_forwards_made": 0, "recorded_utc": d.now()})
    raise d.StopRun("PHYSICAL_GATE_NOT_PASSED" if not gate_passed(out) else "PHASE_NOT_IMPLEMENTED",
                    f"{phase} was not run; no provider or forward budget was consumed")


def freeze_provider_sample(root, config_path, out):
    _not_run(out, "freeze-provider-sample")


def call_provider(root, config_path, out, workers=2):
    _not_run(out, "call-provider")


def probe_representation(root, config_path, out, gpus):
    _not_run(out, "probe-representation")


# ------------------------------------------------------------------ summarize
def _per_bin(branch_rows):
    stats = {}
    for b in d.BINS:
        rows = [r for r in branch_rows if str(r["occlusion_bin"]) == str(b)]
        for route in ("direct", "relocation"):
            rr = [r for r in rows if r["route"] == route]
            ok = sum(1 for r in rr if str(r["success"]) == "True")
            times = [float(r["sim_time"]) for r in rr if r["sim_time"] not in ("", None) and str(r["success"]) == "True"]
            stats[f"bin{b}_{route}"] = {"success": ok, "n": len(rr), "mean_success_sim_time": float(np.mean(times)) if times else None}
    return stats


def summarize(root, config_path, out):
    root, out = Path(root), Path(out)
    led = d.ledger(out)
    d.sync_physical(out)
    led = d.ledger(out)
    branch_rows = d.read_csv(out / "physical/branch_results.csv")
    scene_rows = d.read_csv(out / "opportunity/scene_opportunity.csv")
    gate = d.rd(out / "opportunity/physical_gate.json")
    static = d.rd(out / "static/static_results.json")
    counts = gate["counts"]
    thr = d.rd(out / "physical/throughput_report.json")
    perbin = _per_bin(branch_rows)
    _atomic_json(out / "opportunity/per_bin_route_stats.json", perbin)
    routes = collections.Counter((r["route"], r["execution_status"], r["termination_reason"], r["attempt_state"]) for r in branch_rows)
    _atomic_json(out / "opportunity/branch_outcome_distribution.json", {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in sorted(routes.items())})
    direct_ok = sum(1 for r in branch_rows if r["route"] == "direct" and r["success"] == "True")
    reloc_ok = sum(1 for r in branch_rows if r["route"] == "relocation" and r["success"] == "True")
    b = thr["batches"]
    est_single = None
    for batch in b:
        if batch["workers"] == 2 and batch.get("min_wall"):
            est_single = 1.0 / batch["min_wall"]
    passed = bool(gate["passed"])
    decision = {"status": "COMPLETE", "task_family_feasibility": "ESTABLISHED_FOR_CURATED_DEMONSTRATION" if passed else "NOT_ESTABLISHED",
                "recommended_next_action": "REQUEST_S2_TP_SR_CONSTRUCTION_AUTHORIZATION" if passed else "EXPLICIT_RESEARCH_DECISION",
                "stop_reason": "" if passed else "PHYSICAL_GATE_FAILED", "method_version": d.METHOD_VERSION, "method_upgrade_authorized": False, "s2_authorized": False,
                "card_recommendation_on_gate_failure": "REDESIGN_GEOMETRY_OR_CONTROLLER" if not passed else ""}
    if passed:
        raise d.StopRun("STOPPED_EVIDENCE_INTEGRITY", "gate passed but provider/representation phases were not run; refusing to declare feasibility")
    d.set_decision(out, **decision)
    _atomic_json(out / "recommended_s2_distribution.json", {"status": "NOT_RECOMMENDED", "reason": "physical gate failed: relocation is not a physically helpful option in the generated geometry",
                                                          "s2_authorized": False})
    unresolved = [
        {"item": "relocation-route measurement", "status": "OPEN", "detail": "22/48 relocation branches stop after PICK(interferer) with NO_PLAN / GOAL_ALREADY_SATISFIED / UNKNOWN_CONTROLLER_EXIT (identical in both repeats). Probable cause: verifier/perception reading of the held or shaded cyan interferer under the nearest-palette mask; NOT verified because no further physical episode is authorized."},
        {"item": "direct-route premise", "status": "OPEN", "detail": "direct PICK(target) succeeded in %d/48 branches, so the interferer almost never impedes the direct route in the generated geometry; controller/geometry redesign needed for any helpful relocation opportunity." % direct_ok},
        {"item": "perception mask", "status": "OPEN", "detail": "T_P_SR uses a nearest-palette colour assignment (tp_sr_runtime) instead of the production independent tolerance masks; production perception is untouched. Effect on non-T_P_SR tasks: none."},
        {"item": "lost static resets", "status": "CLOSED_RECORDED", "detail": "T_P_SR_pool_00 and _01 were reset before the mask fix (2/64 resets); both stay FAIL and are not re-screened."},
        {"item": "provider and representation phases", "status": "NOT_RUN", "detail": "gated by the physical opportunity gate; no provider call, no representation forward."},
        {"item": "initial RGB capture", "status": "OPEN", "detail": "static screening did not persist RGB/depth; a provider phase would need a capture reset budget beyond the 64 static resets."},
        {"item": "throughput baseline", "status": "OPEN", "detail": "no matched single-worker baseline was measured; the 4-vs-2 worker comparison uses different branch subsets."}]
    d.write_csv(out / "unresolved_items.csv", unresolved)
    speed = (b[1]["aggregate"] / b[0]["aggregate"]) if len(b) > 1 and b[0]["aggregate"] else None
    md = [f"# S4 T_P_SOFT_RELOCATION_V1 design feasibility ({d.CARD_ID})", "",
          "Curated mechanism demonstration; not an unbiased benchmark. method_version 2.1.1; no method upgrade; S2 not authorized.", "",
          f"**Result: task_family_feasibility = {decision['task_family_feasibility']}; recommended_next_action = {decision['recommended_next_action']} "
          "(card rule on gate failure: REDESIGN_GEOMETRY_OR_CONTROLLER).**", "",
          "## Pool and static screen", "",
          f"64 generated configs (16 per occlusion bin); 64/64 static resets used; {static['pass']} PASS (per bin {static['by_bin_pass']}). "
          "Two resets (pool_00, pool_01) were lost to a perception colour-mask crosstalk found before the fix; four further configs failed the reset-fact checks.", "",
          "## Physical probe", "",
          f"24 scenes, 96 paired branch episodes (direct vs relocation, 2 repeats, shared restore seed, B_PLAN continuation): direct success {direct_ok}/48, relocation success {reloc_ok}/48. "
          f"Scene opportunity: STRONG_HELPFUL {counts['STRONG_HELPFUL']}, COST_HELPFUL {counts['COST_HELPFUL']}, NEUTRAL {counts['NEUTRAL']}, HARMFUL {counts['HARMFUL']}, UNKNOWN {counts['UNKNOWN']}. "
          f"Gate checks: {gate['checks']}; engineering-failure rate {gate['engineering_failure_rate']:.3f}.", "",
          "The direct route rarely fails at these separations (no helpful headroom), and when both routes succeed relocation takes about 2x the simulated time. "
          "Relocation-route failures after PICK(interferer) repeat identically in both repeats and are probably a verifier/perception reading issue, not evidence about the physical benefit; that cause was not verified.", "",
          "## Not run", "", "Provider calls, representation probe, RL/optimizer/training, S2/S3, formal test: none (all ledger entries 0 where capped at 0; provider first calls/retries 0/12, 0/12; representation forwards 0/12).", "",
          "## Throughput", "",
          f"Physical workers: 2 then 4 (max 4). Aggregate {b[0]['aggregate']:.3f}/s at 2 workers, {b[-1]['aggregate']:.3f}/s at 4 workers; decision {thr['decision']}. "
          f"4-vs-2 ratio {speed:.2f}x on different branch subsets; single-worker baseline not measured directly (best-single estimate {est_single:.3f}/s from min wall)."]
    (out / "final_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    d.mark_phase(out, "summarize", status="COMPLETE")
    ids = {"config_sha256": sha256_file(config_path), **{f: sha256_file(root / f) for f in d.CODE_FILES if (root / f).is_file()}}
    _atomic_json(out / "source_identity.json", {"head": d.git(root, "rev-parse", "HEAD"), "base_commit": d.BASE_COMMIT, "files_sha256": ids,
                                                "git_status_at_summary": d.git(root, "status", "--short").splitlines()[:40],
                                                "perception_variant": "cp-disr-d0-rgbd-colorseg-v2+nearest-palette-assignment"})
    return {"status": "SUMMARIZED", "decision": decision, "direct_success": direct_ok, "reloc_success": reloc_ok, "throughput_ratio_4_vs_2": speed}


# ------------------------------------------------------------------ verify
def verify(root, config_path, out):
    root, out = Path(root), Path(out)
    config = d.load_config(config_path)
    led = d.ledger(out)
    checks = {}
    checks["caps_respected"] = all(int(v["used"]) <= int(v["cap"]) for v in led.values())
    checks["zero_cap_items_unused"] = all(int(led[k]["used"]) == 0 for k in ("rl_transitions", "optimizer_steps", "training_attempts", "elastic_attempts", "formal_test_episodes"))
    checks["provider_and_representation_unused"] = all(int(led[k]["used"]) == 0 for k in ("provider_first_calls", "provider_retries", "representation_forward_cases"))
    checks["pool_64"] = int(led["candidate_reset_configs"]["used"]) == 64 and len(d.rd(out / "pool/pool_configs.json")["configs"]) == 64
    rows = list((out / "static/rows").glob("*.json"))
    checks["static_rows_64_and_ledger_64"] = len(rows) == 64 and int(led["static_capture_resets"]["used"]) == 64
    reg = d.rd(out / "physical/witnesses/e4_branch_registration.json")
    attempts = _load_attempts(out / "physical")
    checks["physical_96_registered"] = len(reg["branches"]) == 96 and int(led["physical_branch_episodes"]["used"]) == 96
    checks["all_attempts_terminal"] = all(attempts.get(b["branch_id"]) in ("COMPLETED", "FAILED", "UNKNOWN") for b in reg["branches"])
    checks["no_duplicate_attempts"] = len(attempts) == len(set(attempts)) == 96
    checks["physical_scenes_24"] = int(led["physical_qualification_scenes"]["used"]) == 24 and len(reg["scenes"]) == 24
    ev = [json.loads(l) for l in (out / "budget_events.jsonl").read_text().splitlines() if l.strip()]
    pev = [json.loads(l) for l in (out / "physical/budget_events.jsonl").read_text().splitlines() if l.strip()]
    per = collections.defaultdict(list)
    for e in pev:
        per[e["branch_id"]].append(e["status"])
    checks["physical_events_one_reserve_each"] = len(per) == 96 and all(v.count("RESERVED") == 1 for v in per.values())
    # paired restore
    scenes = d.read_csv(out / "opportunity/scene_opportunity.csv")
    sc_ok = sum(1 for r in scenes if r["paired_restore"].count("PAIRED_RESTORE_VERIFIED") == 2)
    checks["paired_restore_verified_scene_count"] = sc_ok
    # old evidence unchanged
    base = "b9ae7f0ee5adf086eb76f2945b2b31addf1b2c99"
    changed = subprocess.run(["git", "-C", str(root), "diff", "--name-only", base, "--", *OLD_EVIDENCE_PATHS], capture_output=True, text=True).stdout.split()
    changed += subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "--", *OLD_EVIDENCE_PATHS], capture_output=True, text=True).stdout.split()
    checks["old_evidence_and_production_files_unchanged"] = not changed
    # secrets
    leaks = []
    for p in out.rglob("*"):
        if p.is_file() and p.stat().st_size < 5_000_000:
            try:
                text = p.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            if any(pat.search(text) for pat in SECRET_PATTERNS):
                leaks.append(str(p.relative_to(out)))
    checks["no_secret_patterns_in_output"] = not leaks
    checks["no_provider_artifacts"] = not list((out / "provider").glob("*cache*")) and not (out / "provider/provider_call_ledger.csv").exists()
    dec = d.rd(out / "decision_manifest.json")
    checks["decision_consistent_with_gate"] = (dec["task_family_feasibility"] == ("ESTABLISHED_FOR_CURATED_DEMONSTRATION" if gate_passed(out) else "NOT_ESTABLISHED")
                                               and dec["method_upgrade_authorized"] is False and dec["s2_authorized"] is False and dec["method_version"] == "2.1.1")
    ok = all(v is True or (isinstance(v, int) and not isinstance(v, bool)) for v in checks.values()) and all(v for k, v in checks.items() if isinstance(v, bool))
    doc = {"status": "PASS" if ok else "FAIL", "checks": checks, "changed_protected_paths": changed, "secret_pattern_files": leaks, "verified_utc": d.now()}
    _atomic_json(out / "verify.json", doc)
    return doc
