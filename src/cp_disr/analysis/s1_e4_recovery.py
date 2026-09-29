"""S1-REV1 E4 recovery: budget-bound preparation, paired-seed registration,
single-branch worker, slot scheduler and paired-restore QA.

Reuses the production reserve/claim/finish primitives and
``execute_registered_branch`` unchanged. This module adds only: a separate
recovery ledger, a corrected (candidate-independent) paired restore seed,
separate restore-receipt / initial-state sidecars, and result aggregation.
It never calls a provider, trains, or touches the original ledger.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cp_disr.common import digest
from cp_disr.analysis.s1_integration import (
    IntegrationError, _append_jsonl, _atomic_json, _load_attempts,
    finish_branch_attempt, reserve_branch_attempt, sha256_file,
)

BASE_COMMIT = "23fab0ca0e454e63ee21d0be5f1011cf40ac1074"
SOURCE_REL = "runs/final_master/S1/20260928T154311Z_bea4bf0d"
RECOVERY_REL = SOURCE_REL + "/e4_recovery_23fab0c"
TAG = "e4_recovery_23fab0c"
CAP = 8
SEED_RULE = "restore_seed = int(sha256(f'{case_id}|{repeat}')[:8], 16); candidate_id is NOT an input"
APPROVAL_TEXT = ("批准在同一个 S1-REV1 内追加最多8个E4物理恢复attempt。原8次已用预算保留，累计上限16次；"
                 "新增provider、RL、optimizer和elastic均为0。先以2个并行分支检查真实执行链，技术检查通过后在同一8次额度内完成剩余分支。"
                 "保持原场景、候选对和方法，不启动S2/S3或正式test。")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _rd(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paths(root):
    root = Path(root).resolve()
    return root, root / SOURCE_REL, root / RECOVERY_REL


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


# --------------------------------------------------------------- prepare
def prepare(root):
    root, src, out = paths(root)
    head = _git(root, "rev-parse", "HEAD")
    if subprocess.call(["git", "-C", str(root), "merge-base", "--is-ancestor", BASE_COMMIT, head]) != 0:
        raise RuntimeError("HEAD does not descend from baseline commit")
    dirty = [l for l in _git(root, "status", "--porcelain").splitlines() if RECOVERY_REL not in l]
    if dirty:
        raise RuntimeError("working tree not clean: " + "; ".join(dirty[:5]))
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("recovery directory already exists; refusing to re-prepare")
    old_reg_path = src / "witnesses/e4_branch_registration.json"
    old_ledger_path = src / "budget_ledger.json"
    old_reg, old_ledger = _rd(old_reg_path), _rd(old_ledger_path)
    item = old_ledger["physical_witness_episodes"]
    if not (int(item["used"]) == int(item["cap"]) == 8):
        raise RuntimeError("original ledger is not 8/8")
    old = old_reg["branches"]
    cases = sorted({b["case_id"] for b in old})
    cands = sorted({b["candidate_id"] for b in old})
    keys = {(b["case_id"], b["candidate_id"], int(b["repeat"])) for b in old}
    want = {(c, k, r) for c in cases for k in cands for r in (0, 1)}
    if len(old) != 8 or len(cases) != 2 or len(cands) != 2 or keys != want:
        raise RuntimeError("historical registration is not the 2x2x2 structure")
    manifest_path = src / "input_binding/T_A_s1_rev1_runtime_manifest.yaml"
    old_by_key = {(b["case_id"], b["candidate_id"], int(b["repeat"])): b for b in old}
    branches = []
    for case in cases:
        for repeat in (0, 1):
            for cand in cands:
                ob = old_by_key[(case, cand, repeat)]
                ident = f"{TAG}|{case}|{cand}|{repeat}"
                seed = int(hashlib.sha256(f"{case}|{repeat}".encode()).hexdigest()[:8], 16)
                branches.append({
                    "branch_id": hashlib.sha256(ident.encode()).hexdigest()[:16],
                    "attempt_id": hashlib.sha256(ident.encode()).hexdigest()[:16],
                    "case_id": case, "candidate_id": cand, "repeat": repeat,
                    "wave": 1 if (case == cases[0] and repeat == 0) else 2,
                    "restore_seed": seed, "restore_seed_rule": SEED_RULE,
                    "supersedes_protocol_attempt": ob["branch_id"],
                    "superseded_restore_seed": ob["restore_seed"],
                    "seed_correction_reason": "original seed was candidate-dependent; paired candidates in the same case/repeat must share one seed",
                    "selection_hash": hashlib.sha256(ident.encode()).hexdigest(),
                    "manifest_path": str(manifest_path),
                    "authorized": True, "execute_now": True, "status": "REGISTERED",
                })
    # pairing sanity: same (case, repeat) -> one seed, different seeds across (case, repeat) allowed
    seeds = {}
    for b in branches:
        seeds.setdefault((b["case_id"], b["repeat"]), set()).add(b["restore_seed"])
    if any(len(v) != 1 for v in seeds.values()):
        raise RuntimeError("paired seed rule violated")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("witnesses/branch_receipts", "witnesses/restore_receipts", "witnesses/initial_state_checks", "branch_results", "worker_logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    reg = {"status": "REGISTERED", "recovery_tag": TAG, "physical_witness_episodes_cap": CAP,
           "source_registration_path": str(old_reg_path), "source_registration_sha256": sha256_file(old_reg_path),
           "branches": branches}
    _atomic_json(out / "witnesses/e4_branch_registration.json", reg)
    (out / "witnesses/e4_branch_registration.json").chmod(0o444)
    ledger = {k: {"cap": 0, "used": 0} for k in ("discovery_scenes", "elastic_attempts", "optimizer_steps", "planner_environment_episodes", "provider_first_calls", "provider_retries", "rl_transitions")}
    ledger["physical_witness_episodes"] = {"cap": CAP, "used": 0}
    ledger["references"] = {"original_ledger_path": str(old_ledger_path), "original_ledger_sha256": sha256_file(old_ledger_path),
                            "original_physical_witness_used": 8, "original_physical_witness_cap": 8,
                            "budget_amendment": "budget_amendment.json", "cumulative_max_original_plus_recovery": 16,
                            "note": "separate recovery ledger; original ledger and all earlier records stay read-only"}
    _atomic_json(out / "budget_ledger.json", ledger)
    (out / "budget_events.jsonl").touch()
    _atomic_json(out / "attempt_registry.json", {})
    auth = {"authorized": True, "execute_now": True, "approved_additional_physical_episodes": CAP,
            "approval_source": "user reply in the working session (multiple-choice: 批准，最多8个)",
            "approval_text_adopted": APPROVAL_TEXT, "approved_at_utc": _now(),
            "scope": "same S1-REV1, same branch; E4 recovery only",
            "new_provider_calls": 0, "new_rl": 0, "new_optimizer": 0, "new_elastic": 0,
            "s2_s3_formal_test": "NOT_AUTHORIZED", "baseline_commit": BASE_COMMIT, "execution_commit": head}
    _atomic_json(out / "authorization.json", auth)
    _atomic_json(out / "budget_amendment.json", {
        "amendment_id": TAG, "original_physical_witness": {"cap": 8, "used": 8, "ledger_unchanged": True},
        "recovery_physical_witness": {"cap": CAP, "used_at_creation": 0}, "cumulative_max": 16,
        "not_a_refund": True, "not_mixed_with_elastic_rl": True, "failures_and_unknown_are_kept": True,
        "authorization_sha256": sha256_file(out / "authorization.json")})
    code_files = ["src/cp_disr/analysis/s1_revision_resume.py", "src/cp_disr/analysis/s1_integration.py",
                  "src/cp_disr/analysis/s1_e4_recovery.py", "scripts/s1_e4_recovery.py",
                  "src/cp_disr/platforms/libero/runtime_factory.py"]
    _atomic_json(out / "runtime_manifest.json", {
        "baseline_commit": BASE_COMMIT, "execution_commit": head,
        "source_manifest_path": str(manifest_path), "source_manifest_sha256": sha256_file(manifest_path),
        "code_sha256": {f: sha256_file(root / f) for f in code_files},
        "python": sys.version, "repository_path": str(root)})
    _event(out, "prepared", execution_commit=head, branches=len(branches))
    return {"status": "PREPARED", "branches": len(branches), "execution_commit": head}


def _event(out, kind, **fields):
    _append_jsonl(Path(out) / "scheduling_events.jsonl", {"event": kind, "timestamp_utc": _now(), "monotonic": time.monotonic(), **fields})


# --------------------------------------------------------------- worker
class ProbedBundle:
    """Transparent proxy that persists restore receipt and initial-state QA after each reset."""

    def __init__(self, inner, out, branch):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_out", Path(out))
        object.__setattr__(self, "_branch", branch)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def start_case(self, case_id, restore_seed=None):
        snap = self._inner.start_case(case_id, restore_seed=restore_seed)
        try:
            self._write_sidecars()
        except Exception as exc:  # QA sidecar failure must be visible, never silent
            _atomic_json(self._out / "witnesses/initial_state_checks" / f"{self._branch['branch_id']}.json",
                         {"status": "SIDECAR_ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return snap

    def _write_sidecars(self):
        b, out, inner = self._branch, self._out, self._inner
        bid = b["branch_id"]
        rec = _jsonable(inner.restore_receipt)
        budget_receipt = out / "witnesses/branch_receipts" / f"{bid}.json"
        _atomic_json(out / "witnesses/restore_receipts" / f"{bid}.json", {
            "branch_id": bid, "case_id": b["case_id"], "candidate_id": b["candidate_id"], "repeat": b["repeat"],
            "restore_receipt": rec, "restore_receipt_sha256": digest(rec),
            "budget_receipt_sha256": sha256_file(budget_receipt) if budget_receipt.is_file() else "",
            "state_identity_kind": rec.get("state_identity_kind")})
        env = inner.environment
        sim = env.sim
        qpos = np.asarray(sim.data.qpos, dtype=float).copy()
        qvel = np.asarray(sim.data.qvel, dtype=float).copy()
        try:
            hidden = _jsonable(env.hidden_truth())
        except Exception as exc:
            hidden = {"error": f"{type(exc).__name__}: {exc}"}
        snap = inner.current_snapshot
        _atomic_json(out / "witnesses/initial_state_checks" / f"{bid}.json", {
            "status": "OK", "qa_only": True, "used_by_policy_planner_provider_or_relation_admission": False,
            "branch_id": bid, "case_id": b["case_id"], "candidate_id": b["candidate_id"], "repeat": b["repeat"],
            "compare_key": {
                "case_id": b["case_id"], "applied_restore_seed": rec.get("applied_restore_seed"),
                "normalized_reset_config_sha256": rec.get("normalized_reset_config_sha256"),
                "public_facts_sha256": rec.get("public_facts_sha256"),
                "candidate_ids_sha256": rec.get("candidate_ids_sha256"),
                "candidate_mask_sha256": rec.get("candidate_mask_sha256")},
            "candidate_ids": list(snap.candidate_ids), "candidate_mask": [bool(x) for x in snap.mask],
            "physical": {"qpos": qpos.tolist(), "qvel": qvel.tolist(),
                         "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
                         "qvel_sha256": hashlib.sha256(qvel.tobytes()).hexdigest(),
                         "hidden_truth": hidden},
            "identity_fields_excluded_from_compare": {"episode_id": snap.episode_id, "wall_clock_utc": _now(),
                                                       "worker_pid": os.getpid(),
                                                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                                       "sim_time": float(sim.data.time)}})


def worker(root, branch_id):
    root, src, out = paths(root)
    from cp_disr.analysis.s1_revision_resume import execute_registered_branch
    from cp_disr.runtime import load_runtime

    def factory(manifest, branch):
        return ProbedBundle(load_runtime(manifest), out, branch)

    started = time.time()
    res = execute_registered_branch(root, branch_id, out, bundle_factory=factory)
    res = _jsonable(res)
    res.update({"worker_pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "worker_wall_seconds": time.time() - started})
    _atomic_json(out / "branch_results" / f"{branch_id}.json", res)
    return res


# --------------------------------------------------------------- scheduler
def _spawn(root, out, branch_id, gpu):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "CP_DISR_PHYSICAL_GPU_INDEX": str(gpu),
                "PYTHONPATH": f"{root}/src:{root}"})
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(k, None)
    log = open(Path(out) / "worker_logs" / f"{branch_id}.log", "ab")
    proc = subprocess.Popen([sys.executable, str(Path(root) / "scripts/s1_e4_recovery.py"), "worker", "--root", str(root), "--branch-id", branch_id],
                            env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(root), start_new_session=True)
    return proc


def run_wave(root, wave, gpus, max_workers=None):
    """Run every branch of a wave. Slots are fixed to GPUs; a free slot immediately takes the next branch.
    Stops launching new branches after any engineering fault; running branches finish naturally."""
    root, src, out = paths(root)
    reg = _rd(out / "witnesses/e4_branch_registration.json")
    todo = [b["branch_id"] for b in reg["branches"] if b["wave"] in (wave if isinstance(wave, (list, tuple, set)) else [wave])
            and _load_attempts(out).get(b["branch_id"]) is None]
    slots = list(gpus)[: (max_workers or len(gpus))]
    running, faults, done = {}, [], []
    _event(out, "wave_start", wave=wave, branches=todo, gpus=slots)
    while todo or running:
        for gpu in slots:
            if gpu in running or not todo or faults:
                continue
            bid = todo.pop(0)
            try:
                reserve_branch_attempt(out, bid)
            except IntegrationError as exc:
                faults.append({"branch_id": bid, "fault": f"RESERVE:{exc}"})
                _event(out, "fault", branch_id=bid, fault=str(exc))
                continue
            proc = _spawn(root, out, bid, gpu)
            running[gpu] = (bid, proc, time.time())
            _event(out, "branch_launch", branch_id=bid, gpu=gpu, pid=proc.pid, slots_in_use=len(running))
        for gpu, (bid, proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            res_path = out / "branch_results" / f"{bid}.json"
            state = _load_attempts(out).get(bid)
            if rc != 0 or not res_path.is_file():
                faults.append({"branch_id": bid, "fault": f"WORKER_EXIT:{rc}", "attempt_state": state})
                if state in ("RESERVED", "STARTED"):
                    finish_branch_attempt(out, bid, "UNKNOWN", error_phase="worker_process_exit", exit_code=rc)
            else:
                r = _rd(res_path)
                if r.get("execution_status") == "EXCEPTION":
                    faults.append({"branch_id": bid, "fault": "EXCEPTION:" + str(r.get("termination_reason"))})
            done.append(bid)
            _event(out, "branch_finish", branch_id=bid, gpu=gpu, exit_code=rc, wall_seconds=time.time() - t0, slots_in_use=len(running) - 1)
            del running[gpu]
        time.sleep(1.0)
    _event(out, "wave_end", wave=wave, done=done, faults=faults)
    return {"done": done, "faults": faults}


# --------------------------------------------------------------- checks / summary
def _journal(out, bid):
    p = Path(out) / "witnesses" / f"branch_{bid}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def paired_restore(out, case, repeat, reg):
    ids = [b["branch_id"] for b in reg["branches"] if b["case_id"] == case and b["repeat"] == repeat]
    docs = []
    for bid in ids:
        p = Path(out) / "witnesses/initial_state_checks" / f"{bid}.json"
        if not p.is_file():
            return {"case_id": case, "repeat": repeat, "status": "PAIRED_RESTORE_NOT_ESTABLISHED", "reason": "MISSING_SIDECAR", "branch_ids": ids}
        docs.append(_rd(p))
    if len(docs) != 2 or any(d.get("status") != "OK" for d in docs):
        return {"case_id": case, "repeat": repeat, "status": "PAIRED_RESTORE_NOT_ESTABLISHED", "reason": "SIDECAR_NOT_OK", "branch_ids": ids}
    a, b = docs
    public_equal = a["compare_key"] == b["compare_key"] and a["candidate_ids"] == b["candidate_ids"] and a["candidate_mask"] == b["candidate_mask"]
    qa, qb = np.array(a["physical"]["qpos"]), np.array(b["physical"]["qpos"])
    va, vb = np.array(a["physical"]["qvel"]), np.array(b["physical"]["qvel"])
    shape_ok = qa.shape == qb.shape and va.shape == vb.shape
    qmax = float(np.max(np.abs(qa - qb))) if shape_ok and qa.size else None
    vmax = float(np.max(np.abs(va - vb))) if shape_ok and va.size else None
    ha, hb = a["physical"].get("hidden_truth"), b["physical"].get("hidden_truth")
    hidden_equal = (ha == hb) and not (isinstance(ha, dict) and "error" in ha)
    physical_equal = bool(shape_ok and qmax is not None and qmax <= 1e-9 and vmax is not None and vmax <= 1e-9)
    ok = public_equal and physical_equal
    return {"case_id": case, "repeat": repeat, "branch_ids": ids,
            "status": "PAIRED_RESTORE_VERIFIED_PUBLIC_AND_SIM_STATE" if ok else "PAIRED_RESTORE_NOT_ESTABLISHED",
            "public_compare_key_equal": public_equal, "qpos_max_abs_diff": qmax, "qvel_max_abs_diff": vmax,
            "qpos_sha256_equal": a["physical"]["qpos_sha256"] == b["physical"]["qpos_sha256"],
            "hidden_truth_qa_equal": hidden_equal,
            "note": "controller internal state and RNG stream are not directly compared"}


def technical_check(root, wave=1):
    root, src, out = paths(root)
    reg = _rd(out / "witnesses/e4_branch_registration.json")
    wave_branches = [b for b in reg["branches"] if b["wave"] == wave]
    attempts, ledger = _load_attempts(out), _rd(out / "budget_ledger.json")
    checks, problems = {}, []
    runtime_manifest = _rd(out / "runtime_manifest.json")
    checks["source_manifest_hash_matches"] = sha256_file(runtime_manifest["source_manifest_path"]) == runtime_manifest["source_manifest_sha256"]
    checks["registration_hash_matches"] = all(
        _rd(out / "witnesses/branch_receipts" / f"{b['branch_id']}.json").get("registration_sha256") == sha256_file(out / "witnesses/e4_branch_registration.json")
        for b in wave_branches if (out / "witnesses/branch_receipts" / f"{b['branch_id']}.json").is_file())
    per = {}
    pids, gpus = set(), set()
    for b in wave_branches:
        bid = b["branch_id"]
        rp = out / "branch_results" / f"{bid}.json"
        res = _rd(rp) if rp.is_file() else {}
        j = _journal(out, bid)
        phases = [e["phase"] for e in j]
        has_action = "action_start" in phases
        chain_ok = (not has_action) or all(p in phases for p in ("action_complete", "verifier_complete", "evaluator_complete")) or any(e["phase"] in ("terminated", "error") for e in j)
        seqs = [e["sequence"] for e in j]
        per[bid] = {"case_id": b["case_id"], "candidate_id": b["candidate_id"], "attempt_state": attempts.get(bid),
                    "journal_events": len(j), "sequence_monotonic": seqs == list(range(len(seqs))),
                    "has_result": bool(res), "execution_status": res.get("execution_status"), "protocol_complete": res.get("protocol_complete"),
                    "has_error_event": "error" in phases, "closed": "closed" in phases, "action_chain_recorded": chain_ok,
                    "first_action_recorded": "action_start" in phases or any(e.get("termination_reason") for e in j),
                    "restore_receipt_file": (out / "witnesses/restore_receipts" / f"{bid}.json").is_file(),
                    "initial_state_file": (out / "witnesses/initial_state_checks" / f"{bid}.json").is_file(),
                    "worker_pid": res.get("worker_pid"), "cuda_visible_devices": res.get("cuda_visible_devices")}
        pids.add(res.get("worker_pid")); gpus.add(res.get("cuda_visible_devices"))
        rr = out / "witnesses/restore_receipts" / f"{bid}.json"
        if rr.is_file() and j:
            ev = next((e for e in j if e["phase"] == "restore_complete"), None)
            per[bid]["journal_restore_hash_is_restore_receipt"] = bool(ev) and ev.get("restore_receipt_sha256") == _rd(rr)["restore_receipt_sha256"]
        for k, v in per[bid].items():
            if k in ("has_error_event",) and v:
                problems.append(f"{bid}:{k}")
            if k in ("has_result", "sequence_monotonic", "action_chain_recorded", "restore_receipt_file", "initial_state_file", "closed") and not v:
                problems.append(f"{bid}:{k}")
            if k == "journal_restore_hash_is_restore_receipt" and not v:
                problems.append(f"{bid}:{k}")
    checks["independent_runtime_processes"] = len(pids - {None}) == len(wave_branches)
    checks["per_worker_gpu_recorded"] = None not in gpus
    pr = paired_restore(out, wave_branches[0]["case_id"], wave_branches[0]["repeat"], reg)
    checks["paired_restore_comparable"] = pr["status"].startswith("PAIRED_RESTORE_VERIFIED")
    used = ledger["physical_witness_episodes"]["used"]
    checks["ledger_used_matches_attempts"] = used == len(attempts) and used == len(wave_branches) if wave == 1 else used == len(attempts)
    checks["no_duplicate_consumption"] = len(attempts) == len(set(attempts))
    checks["no_attempt_left_reserved_or_started"] = all(attempts.get(b["branch_id"]) in ("COMPLETED", "FAILED", "UNKNOWN") for b in wave_branches)
    ev_lines = [json.loads(l) for l in (out / "budget_events.jsonl").read_text().splitlines() if l.strip()]
    per_branch_counts = {}
    for e in ev_lines:
        per_branch_counts.setdefault(e["branch_id"], []).append(e["status"])
    checks["budget_events_one_reserve_one_start"] = all(v.count("RESERVED") == 1 and v.count("STARTED") <= 1 for v in per_branch_counts.values())
    verdict = "PASS" if all(checks.values()) and not problems else "FAIL"
    doc = {"wave": wave, "verdict": verdict, "checks": checks, "problems": problems, "per_branch": per, "paired_restore": pr,
           "release_criterion": "technical validity only; success / prior benefit / E4 difference are NOT release conditions", "checked_at_utc": _now()}
    _atomic_json(out / f"wave{wave}_technical_check.json", doc)
    return doc


def _outcome(r):
    return (r.get("execution_status"), r.get("termination_reason"), bool(r.get("task_success")),
            (r.get("trace") or [{}])[0].get("candidate_id") if r.get("trace") else None,
            [t.get("candidate_id") for t in r.get("trace", [])])


def summarize(root):
    root, src, out = paths(root)
    reg = _rd(out / "witnesses/e4_branch_registration.json")
    attempts, ledger = _load_attempts(out), _rd(out / "budget_ledger.json")
    rows, results = [], {}
    for b in reg["branches"]:
        rp = out / "branch_results" / f"{b['branch_id']}.json"
        r = _rd(rp) if rp.is_file() else {}
        results[b["branch_id"]] = r
        rows.append({"branch_id": b["branch_id"], "case_id": b["case_id"], "candidate_id": b["candidate_id"], "repeat": b["repeat"],
                     "attempt_state": attempts.get(b["branch_id"], "NOT_STARTED"), "controller_exit": r.get("controller_exit", ""),
                     "execution_status": r.get("execution_status", ""), "termination_reason": r.get("termination_reason", ""),
                     "task_success": r.get("task_success", ""), "protocol_complete": r.get("protocol_complete", ""),
                     "decision_count": r.get("decision_count", ""), "executed_actions": "|".join(t.get("candidate_id", "") for t in r.get("trace", []))})
    with (out / "branch_results.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    cases = sorted({b["case_id"] for b in reg["branches"]})
    paired, case_status = [], {}
    for case in cases:
        per_rep = []
        for repeat in (0, 1):
            pr = paired_restore(out, case, repeat, reg)
            bs = [b for b in reg["branches"] if b["case_id"] == case and b["repeat"] == repeat]
            rs = [results.get(b["branch_id"], {}) for b in bs]
            complete = all(r.get("protocol_complete") is True for r in rs) and len(rs) == 2
            outs = [(_outcome(r)[0], _outcome(r)[1], _outcome(r)[2], tuple(_outcome(r)[4])) for r in rs]
            if not complete or not pr["status"].startswith("PAIRED_RESTORE_VERIFIED"):
                eff = "NOT_ESTABLISHED"
            else:
                eff = "OUTCOME_DIFFERENCE" if outs[0] != outs[1] else "NO_OUTCOME_DIFFERENCE"
            per_rep.append({"repeat": repeat, "paired_restore": pr, "candidates": [b["candidate_id"] for b in bs], "outcomes": [list(o) for o in outs],
                            "protocol_complete": complete, "paired_effect_status": eff})
            paired.append({"case_id": case, **per_rep[-1]})
        effs = {p["paired_effect_status"] for p in per_rep}
        case_status[case] = ("NOT_ESTABLISHED" if "NOT_ESTABLISHED" in effs else
                             ("REPEATS_AGREE_" + next(iter(effs)) if len(effs) == 1 else "REPEATS_DISAGREE"))
    _atomic_json(out / "paired_comparisons.json", {"per_case_repeat": paired, "per_case": case_status,
        "note": "engineering paired-outcome record; not a scientific E4 conclusion. Two repeats are not two independent configurations."})
    # throughput
    events = [json.loads(l) for l in (out / "scheduling_events.jsonl").read_text().splitlines() if l.strip()]
    fin = [e for e in events if e["event"] == "branch_finish"]
    launches = {e["branch_id"]: e for e in events if e["event"] == "branch_launch"}
    valid_actions = sum(len(r.get("trace", [])) for r in results.values() if r.get("protocol_complete") is True)
    span = (max(e["monotonic"] for e in fin) - min(e["monotonic"] for e in launches.values())) if fin and launches else None
    _atomic_json(out / "throughput_report.json", {
        "workers_max_concurrent": max([e.get("slots_in_use", 0) for e in events if e["event"] == "branch_launch"] or [0]),
        "branches_finished": len(fin), "wall_seconds_first_launch_to_last_finish": span,
        "valid_skill_completions_in_protocol_complete_branches": valid_actions,
        "skills_per_wall_second": (valid_actions / span) if span else None,
        "per_branch_wall_seconds": {e["branch_id"]: e["wall_seconds"] for e in fin},
        "single_worker_baseline": "NOT_MEASURED", "speedup_claim": "NONE (no comparable single-worker baseline; no extra episodes spent to measure)"})
    dist = {}
    for r in rows:
        k = (r["controller_exit"], r["execution_status"], r["termination_reason"], r["task_success"])
        dist[str(k)] = dist.get(str(k), 0) + 1
    states = {s: sum(1 for b in reg["branches"] if attempts.get(b["branch_id"]) == s) for s in ("RESERVED", "STARTED", "COMPLETED", "FAILED", "UNKNOWN")}
    used = ledger["physical_witness_episodes"]["used"]
    summary = {"reserved_total": used, "attempt_states": states, "cumulative_original_plus_recovery": 8 + used, "distribution": dist,
               "case_status": case_status, "paired_restore": {f"{p['case_id']}#{p['repeat']}": p["paired_restore"]["status"] for p in paired}}
    _atomic_json(out / "summary.json", summary)
    lines = ["# E4 恢复运行摘要（工程记录，非科学结论）", "",
             f"- 恢复 attempt 已预留 {used}/{CAP}；状态 {states}；累计（原8+恢复）{8 + used}/16。",
             "- 新 provider / RL / optimizer / elastic：0 / 0 / 0 / 0。S2、S3、正式 test：NOT_RUN。",
             "- E1–E6 中六类证据表未在本轮补齐，均标 NOT_ESTABLISHED；tp_training_authorized=false。", "",
             "## 分支结果", "", "| case | candidate | repeat | state | exit | status | reason | success |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['case_id']} | {r['candidate_id']} | {r['repeat']} | {r['attempt_state']} | {r['controller_exit']} | {r['execution_status']} | {r['termination_reason']} | {r['task_success']} |")
    lines += ["", "## 配对恢复与配对结果", ""]
    for p in paired:
        lines.append(f"- {p['case_id']} repeat {p['repeat']}: restore={p['paired_restore']['status']}；paired_effect={p['paired_effect_status']}")
    lines += ["", f"每 case 汇总：{case_status}", "", "吞吐见 throughput_report.json（单 worker 基线 NOT_MEASURED，不声称加速）。"]
    (out / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
