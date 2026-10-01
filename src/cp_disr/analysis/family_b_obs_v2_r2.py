"""CP-DISR-S4-FAMILY-B-OBS-V2-R2-VALIDATION-1: staged canary -> remainder execution of the FROZEN
FAMILY_B_PUBLIC_OBS_V2 technical validation.

This module only controls identity, budget, phase ordering and gate evaluation.  It never edits the frozen
observation profile, camera set, fusion rule, thresholds, layouts, controller or task code, and it reuses the
v2 worker/paired-restore/timeline code unchanged (family_b_obs_v2 analysis module stays byte-identical).
"""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cp_disr.analysis import family_b_obs_v2 as m
from cp_disr.platforms.libero.perception import THRESHOLDS

CARD = "CP-DISR-S4-FAMILY-B-OBS-V2-R2-VALIDATION-1"
REPAIR_BASE = "4a84c6a7c6a545c8d74968d3a5c8f6a186de111d"
PROFILE_ORIGIN = "55668197091e4d9d24c3025fcc3ec522febb211e"
FAILED_R1_COMMIT = "1887c305f7bfa9ac489cb355cab2b9e236da4cdd"
FROZEN_SHA = "3c98a09649a73a3d9cee1db7e7a9f67063e515bfe5c3e689cabf4cf09900d917"
PY = "/home/xushijie2/envs/lerobotpi0-xfs/bin/python"
CFG_R2 = "configs/final_master/family_b_obs_v2_r2.yaml"
R1_DIR = "runs/final_master/S4/family_b_observation_v2/20261001T040031Z_3c98a096"
REPAIR_DIR = "runs/final_master/S4/family_b_observation_v2/engineering_repair_1887c305f"
FROZEN_FILES = tuple(f"{m.CFG_DIR}/{n}" for n in (
    "observation_profile_v2.json", "camera_manifest.json", "fusion_contract.json", "pair_equivalence_contract.json"))
SOURCE_FILES_R2 = m.SOURCE_FILES + (
    "src/cp_disr/analysis/family_b_obs_v2_r2.py", "scripts/family_b_obs_v2_r2.py", CFG_R2)
CANARY = ("layout_0", "B_PENDING", 0, "pad_v")
REMAINDER = (("layout_0", "B_PENDING", 0, "pad_u"), ("layout_1", "C_PENDING", 0, "pad_v"),
             ("layout_1", "C_PENDING", 0, "pad_u"))
CAPS = {"attempts": 4, "constructions": 4, "resets": 4, "skills": 24, "canary": 1, "remainder": 3}
ATTEMPT_ID = "FAMILY_B_OBS_V2_R2_ATTEMPT_1:{pad}"
PASS_CANARY, FAIL_CANARY = "CANARY_PASS", "CANARY_FAIL"
GATE_PASS = "OBS_V2_R2_TECHNICAL_GATE_PASS_READY_FOR_REMAINDER_REVIEW"
GATE_FAIL = "OBS_V2_R2_TECHNICAL_GATE_FAIL"
WATCHDOG_SECONDS = 1800
read, write, sha = m.read, m.write, m.sha


# --------------------------------------------------------------------------------- identity
def branch_key(b):
    return (b["layout"], b["context"], b["repeat"], b["candidate"])


def source_hashes(root):
    return {p: sha(Path(root) / p) for p in SOURCE_FILES_R2}


def r2_branches(profile_sha, commit, source_hash, contexts, old_ids, manifest_path, ref_ids=None):
    """Four R2 branches; id binds logical identity, profile version/hash, prep commit, source hash, R2 attempt."""
    out = []
    for layout, context, repeat in m.TECHNICAL:
        pair_key = m.digest({"layout": layout, "context": context, "repeat": repeat, "profile": profile_sha,
                             "commit": commit, "card": CARD})[:16]
        seed = int(m.hashlib.sha256(f"obsv2r2:{pair_key}".encode()).hexdigest()[:8], 16)
        for pad in m.PAD_PAIRS:
            logical = {"layout": layout, "context": context, "repeat": repeat, "candidate": pad}
            attempt_identity = ATTEMPT_ID.format(pad=pad)
            bid = m.branch_identity({**logical, "card": CARD}, profile_sha, commit, source_hash, attempt_identity)
            is_canary = (layout, context, repeat, pad) == CANARY
            out.append({
                **logical, "branch_id": bid, "case_id": layout, "restore_seed": seed,
                "candidate_id": f"a:PLACE_BUFFER:carrier:{pad}:v1", "prefix": contexts[context]["prefix"],
                "wave": "obs_v2_r2_canary" if is_canary else "obs_v2_r2_remainder",
                "phase": "canary" if is_canary else "remainder", "manifest_path": str(manifest_path),
                "authorized": True, "execute_now": True, "plan_id": bid, "attempt_id": bid,
                "attempt_identity": attempt_identity, "card_id": CARD,
                "v1_logical_branch_id_reference_only": (ref_ids or {}).get((layout, context, repeat, pad)),
                "profile_version": m.PROFILE_VERSION, "observation_profile_sha256": profile_sha,
                "cameras": list(m.CAMERAS), "source_commit": commit, "source_hash": source_hash})
    ids = [b["branch_id"] for b in out]
    if len(set(ids)) != 4 or set(ids) & set(old_ids):
        raise ValueError("STOPPED_SOURCE_IDENTITY:branch identities collide with a historical card")
    return out


def canary_of(branches):
    hit = [b for b in branches if branch_key(b) == CANARY]
    if len(hit) != 1:
        raise ValueError("STOPPED_CANARY:identity")
    return hit[0]


def frozen_bytes_ok(root):
    """The four frozen files must equal the freeze commit byte for byte; returns the list of differing files."""
    bad = []
    for rel in FROZEN_FILES:
        want = subprocess.run(["git", "-C", str(root), "show", f"{PROFILE_ORIGIN}:{rel}"], capture_output=True,
                              check=True).stdout
        if (Path(root) / rel).read_bytes() != want:
            bad.append(rel)
    return bad


def protected_paths(root):
    root = Path(root)
    paths = list(m.protected_paths())
    for rel in (m.OLD_EVIDENCE, m.OLD_REVIEW, R1_DIR, REPAIR_DIR, m.CFG_DIR):
        if (root / rel).exists():
            paths.append(root / rel)
    return paths


def register(root, out):
    import yaml
    from cp_disr.analysis import family_b_pilot as v1
    from cp_disr.runtime import require_runtime
    root, out = Path(root).resolve(), Path(out)
    if m.git(root, "status", "--porcelain", "-uno"):
        raise ValueError("STOPPED_SOURCE_IDENTITY:tracked worktree not clean")
    commit = m.git(root, "rev-parse", "HEAD")
    if m.git(root, "merge-base", "--is-ancestor", REPAIR_BASE, "HEAD") != "":
        raise ValueError("STOPPED_SOURCE_IDENTITY:repair base is not an ancestor")
    if frozen_bytes_ok(root):
        raise ValueError("STOPPED_OBSERVATION_PROFILE:frozen file differs from freeze commit")
    profile = m.check_profile(root)
    if profile["profile_sha256"] != FROZEN_SHA:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:hash")
    cfg = yaml.safe_load((root / CFG_R2).read_text())
    base_cfg = yaml.safe_load((root / m.CFG_DIR / "config.yaml").read_text())
    if (cfg["card_id"], cfg["observation_profile_sha256"], cfg["repair_base_commit"]) != (CARD, FROZEN_SHA, REPAIR_BASE):
        raise ValueError("STOPPED_SOURCE_IDENTITY:r2 config identity")
    if (out / "physical/witnesses/e4_branch_registration.json").exists():
        raise ValueError("STOPPED_SOURCE_IDENTITY:already registered")
    hashes = source_hashes(root)
    source_hash = m.digest(hashes)
    prior = yaml.safe_load(v1.SOURCE_MANIFEST.read_text())
    ref = float(prior["runtime"]["reference_skill_seconds_by_task"]["T_P_SO_MVP"])
    runtime = dict(prior["runtime"])
    rt = base_cfg["runtime"]
    runtime.update({
        "repository_path": str(root), "active_task_id": "T_P_FB", "contract_path": rt["contract_path"],
        "layouts_path": rt["layouts_path"], "observation_profile_path": rt["observation_profile_path"],
        "reference_skill_seconds_by_task": {"T_P_FB": ref}, "task_deadlines": {"T_P_FB": 60.0},
        "skill_timeouts": {"T_P_FB": {"PICK": 9., "PLACE": 8., "PLACE_BUFFER": 8.}},
        "task_assets": {"T_P_FB": "src/cp_disr/platforms/libero/family_b_env.py",
                        "license": "assets/cp_disr/LICENSE", "owned_by": "cp_disr_project"},
        "task_evaluator_version": {"T_P_FB": "family-b-evaluator-v1"},
        "task_splits": {"T_P_FB": rt["layouts_path"]}, "decision_cap_by_task": {"T_P_FB": 6}})
    src = root / "src/cp_disr/platforms/libero/family_b_runtime_v2.py"
    manifest = dict(prior)
    manifest["runtime"] = runtime
    manifest["runtime_factory"] = {"factory": "create_family_b_obs_v2_runtime",
                                   "module": "cp_disr.platforms.libero.family_b_runtime_v2",
                                   "source_path": str(src), "sha256": sha(src)}
    require_runtime(manifest)
    manifest_path = out / "spec/runtime_manifest_T_P_FB_obs_v2_r2.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    old_v1 = {(b["layout"], b["context"], b["repeat"], b["candidate"]): b["branch_id"]
              for b in read(root / m.OLD_EVIDENCE / "physical/registration.json")["branches"]}
    r1_ids = [b["branch_id"] for b in read(root / R1_DIR / "registration_v2.json")["branches"]]
    branches = r2_branches(profile["profile_sha256"], commit, source_hash, base_cfg["contexts"],
                           set(old_v1.values()) | set(r1_ids), manifest_path, old_v1)
    # other historical cards: every branch id found in any committed registration under runs/
    seen = set()
    for p in (root / "runs").rglob("registration*.json"):
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("branches"), list):
            seen |= {b.get("branch_id") for b in doc["branches"] if isinstance(b, dict)}
    if {b["branch_id"] for b in branches} & seen:
        raise ValueError("STOPPED_SOURCE_IDENTITY:branch id exists in a historical registration")
    registration = {"card_id": CARD, "branches": branches, "manifest_sha256": sha(manifest_path),
                    "reference_skill_seconds": ref}
    problems = m.check_registration_uniform(registration)
    if problems:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:" + ";".join(problems))
    phys = out / "physical"
    write(phys / "witnesses/e4_branch_registration.json", registration)
    write(phys / "registration.json", registration)
    write(phys / "budget_ledger.json", {"physical_witness_episodes": {"cap": CAPS["attempts"], "used": 0}})
    write(out / "registration_r2.json", registration)
    write(out / "canary_registration.json", {"card_id": CARD, "canary": canary_of(branches),
          "rule": "layout_0/B_PENDING/repeat0/pad_v", "remainder_order": [list(k) for k in REMAINDER],
          "remainder_branch_ids": [b["branch_id"] for b in branches if branch_key(b) in REMAINDER]})
    write(out / "budget_ledger_r2.json", {
        "card_id": CARD,
        "history": {"v1": {"attempts": 4, "resets": 4, "skill_calls": 20, "status": "PERMANENTLY_CONSUMED",
                           "original_remaining_20": "NOT_RELEASED"},
                    "v2_r1": {"attempt_slots_charged": 2, "unused_slots": 2,
                              "unused_slots_status": "RETIRED_NOT_TRANSFERABLE",
                              "environment_construction_attempts": 2, "successful_environment_constructors": 0,
                              "actual_explicit_reset_calls": "NOT_REACHED", "observations": 0, "skill_calls": 0}},
        "attempts": {"cap": CAPS["attempts"], "used": 0},
        "environment_constructions": {"cap": CAPS["constructions"], "used": 0},
        "explicit_resets": {"cap": CAPS["resets"], "used": 0},
        "live_skill_calls": {"cap": CAPS["skills"], "used": 0},
        "skill_retries": {"cap": 0, "used": 0}, "standalone_capture_resets": {"cap": 0, "used": 0},
        "provider_calls": {"cap": 0, "used": 0}, "rl_transitions": {"cap": 0, "used": 0},
        "optimizer_steps": {"cap": 0, "used": 0}, "elastic": {"cap": 0, "used": 0},
        "phases": {"canary": {"cap": CAPS["canary"], "used": 0}, "remainder": {"cap": CAPS["remainder"], "used": 0}},
        "reserved_branches": [], "remainder_slots_status": "PENDING_CANARY",
        "original_remaining_20": "NOT_RELEASED",
        "actual": {"constructions": None, "explicit_resets": None, "internal_resets": None, "skill_calls": None}})
    _event(out, "REGISTERED", branches=[b["branch_id"] for b in branches], commit=commit)
    write(out / "bindings/reference_cost.json", {"reference_skill_seconds": ref})
    write(out / "authorization_r2.json", {
        "card_id": CARD, "authorized_by": "user message approving CP-DISR-S4-FAMILY-B-OBS-V2-R2-VALIDATION-1",
        "v2_r2_branch_attempts_max": 4, "environment_construction_attempts_max": 4,
        "successful_explicit_resets_max": 4, "live_skill_calls_max": 24, "skill_retries": 0,
        "standalone_capture_resets": 0, "provider": 0, "rl_transitions": 0, "optimizer_steps": 0, "elastic": 0,
        "formal_test": 0, "s2": False, "s3": False, "tp_training": False, "original_remaining_20_released": False,
        "canary_attempts": 1, "conditional_remainder_attempts": 3,
        "r1_unused_slots_transferred": False, "no_config_change_after_failure": True})
    write(out / "source_identity.json", {
        "card_id": CARD, "branch": m.git(root, "branch", "--show-current"), "commit": commit,
        "repair_base_commit": REPAIR_BASE, "profile_origin_commit": PROFILE_ORIGIN,
        "failed_r1_commit": FAILED_R1_COMMIT, "source_hash": source_hash, "sources": hashes,
        "observation_profile_sha256": profile["profile_sha256"], "manifest_sha256": sha(manifest_path),
        "frozen_file_sha256": {rel: sha(root / rel) for rel in FROZEN_FILES}})
    base_ref = {k: base_cfg[k] for k in ("task_id", "method_version", "observation_profile_version", "contexts",
                                         "technical_wave", "runtime", "parallel") if k in base_cfg}
    (out / "r2_config_resolved.yaml").write_text(yaml.safe_dump(
        {"r2": cfg, "runtime_and_contexts_from_frozen_v2_config": base_ref}, sort_keys=True))
    for rel in (m.PROFILE_PATH, f"{m.CFG_DIR}/camera_manifest.json", f"{m.CFG_DIR}/fusion_contract.json"):
        write(out / Path(rel).name, read(root / rel))
    return {"status": "REGISTERED", "commit": commit, "canary": canary_of(branches)["branch_id"],
            "branches": [b["branch_id"] for b in branches]}


# ------------------------------------------------------------------------- ledger / events
def _event(out, kind, **fields):
    path = Path(out) / "budget_events_r2.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"event": kind, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields},
                                sort_keys=True) + "\n")
        handle.flush()


@contextmanager
def _lock(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    with (Path(out) / ".r2_ledger.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def reserve_and_charge(out, b, phase):
    """Atomic: reserve the physical slot, then charge the R2 ledger.  Raises before any change on violation."""
    from cp_disr.analysis.s1_integration import reserve_branch_attempt
    out = Path(out)
    with _lock(out):
        ledger = read(out / "budget_ledger_r2.json")
        bid = b["branch_id"]
        if bid in ledger["reserved_branches"]:
            raise ValueError("STOPPED_DUPLICATE_ATTEMPT")
        if b["phase"] != phase:
            raise ValueError("STOPPED_PHASE_MISMATCH")
        for key in ("attempts", "environment_constructions", "explicit_resets"):
            if ledger[key]["used"] + 1 > ledger[key]["cap"]:
                raise ValueError("STOPPED_BUDGET_EXHAUSTED:" + key)
        if ledger["phases"][phase]["used"] + 1 > ledger["phases"][phase]["cap"]:
            raise ValueError("STOPPED_BUDGET_EXHAUSTED:phase_" + phase)
        reserve_branch_attempt(out / "physical", bid)
        for key in ("attempts", "environment_constructions", "explicit_resets"):
            ledger[key]["used"] += 1
        ledger["phases"][phase]["used"] += 1
        ledger["reserved_branches"].append(bid)
        write(out / "budget_ledger_r2.json", ledger)
        _event(out, "RESERVED_CHARGED", branch_id=bid, phase=phase)
    return ledger


def r_ok(c):
    return c.get("reset_calls", -1) >= 1


def finalize_counts(out):
    out = Path(out)
    with _lock(out):
        ledger = read(out / "budget_ledger_r2.json")
        res = [read(p) for p in sorted((out / "physical/branch_results").glob("*.json"))]
        counts = [r.get("env_counts", {}) for r in res]
        ledger["live_skill_calls"]["used"] = sum(len(r.get("actions", [])) for r in res)
        ledger["actual"] = {
            "constructions_attempted": len(res),
            "constructor_successful": sum(1 for c in counts if c.get("constructions") == 1 and r_ok(c)),
            "explicit_resets": sum(max(int(c.get("reset_calls", 0)), 0) for c in counts),
            "internal_resets": sum(max(int(c.get("internal_resets", 0)), 0) for c in counts),
            "skill_calls": ledger["live_skill_calls"]["used"]}
        write(out / "budget_ledger_r2.json", ledger)
    return ledger


def mark_remainder_not_released(out):
    with _lock(out):
        ledger = read(Path(out) / "budget_ledger_r2.json")
        ledger["remainder_slots_status"] = "NOT_RELEASED_AFTER_CANARY_FAILURE"
        write(Path(out) / "budget_ledger_r2.json", ledger)
        _event(out, "REMAINDER_NOT_RELEASED_AFTER_CANARY_FAILURE")


# ------------------------------------------------------------------------------- identity
def verify_identity(root, out):
    """Raised before any reservation; HEAD/sources/frozen files must be exactly what was registered."""
    root, out = Path(root).resolve(), Path(out).resolve()
    ident = read(out / "source_identity.json")
    if m.git(root, "rev-parse", "HEAD") != ident["commit"]:
        raise ValueError("STOPPED_SOURCE_IDENTITY:HEAD moved since registration")
    if m.git(root, "status", "--porcelain", "-uno"):
        raise ValueError("STOPPED_SOURCE_IDENTITY:tracked worktree dirty")
    if ident["sources"] != source_hashes(root):
        raise ValueError("STOPPED_SOURCE_IDENTITY:source drift")
    if frozen_bytes_ok(root):
        raise ValueError("STOPPED_OBSERVATION_PROFILE:frozen file changed")
    reg = read(out / "physical/registration.json")
    problems = m.check_registration_uniform(reg)
    if problems or len(reg["branches"]) != 4 or reg["card_id"] != CARD:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:" + ";".join(problems or ["registration"]))
    if Path(sys.executable).resolve() != Path(PY).resolve() or not str(m.__file__).startswith(str(root / "src")):
        raise ValueError("STOPPED_SOURCE_IDENTITY:interpreter or cp_disr import path")
    return ident


# -------------------------------------------------------------------------------- launcher
def default_launcher(root, out, b, gpu):
    from cp_disr.analysis.family_b_pilot import _worker_env
    logs = Path(out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = (logs / f"{b['branch_id']}.log").open("ab")
    cmd = [sys.executable, str(Path(root) / "scripts/family_b_obs_v2_r2.py"), "worker", "--root", str(root),
           "--output", str(out), "--branch-id", b["branch_id"]]
    proc = subprocess.Popen(cmd, cwd=root, env=_worker_env(gpu), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    proc.log_handle = log
    return proc


def _wait(proc, timeout=WATCHDOG_SECONDS, sleep=time.sleep):
    t0 = time.monotonic()
    while proc.poll() is None:
        if time.monotonic() - t0 > timeout:
            return "WATCHDOG_TIMEOUT"
        sleep(0.5)
    return proc.poll()


def _close(proc):
    handle = getattr(proc, "log_handle", None)
    if handle:
        handle.close()


def _kill(proc):
    import os
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass


def _no_prior_results(out, branches):
    if any((Path(out) / "physical/branch_results" / f"{b['branch_id']}.json").exists() for b in branches):
        raise ValueError("STOPPED_DUPLICATE_ATTEMPT")


def run_canary(root, out, gpus, launcher=default_launcher, identity_check=verify_identity, sleep=time.sleep):
    """Dispatch exactly ONE branch (the canary).  The remainder is never dispatched from here."""
    root, out = Path(root).resolve(), Path(out).resolve()
    identity_check(root, out)
    reg = read(out / "physical/registration.json")
    canary_b = canary_of(reg["branches"])
    ledger = read(out / "budget_ledger_r2.json")
    if ledger["attempts"]["used"] != 0 or ledger["reserved_branches"]:
        raise ValueError("STOPPED_CANARY_ALREADY_RUN")
    _no_prior_results(out, reg["branches"])
    gpu = list(gpus)[0]
    reserve_and_charge(out, canary_b, "canary")
    started = time.monotonic()
    proc = launcher(root, out, canary_b, gpu)
    code = _wait(proc, sleep=sleep)
    if code == "WATCHDOG_TIMEOUT":
        _kill(proc)
        from cp_disr.analysis.s1_integration import finish_branch_attempt
        finish_branch_attempt(out / "physical", canary_b["branch_id"], "UNKNOWN", execution_status="WATCHDOG_TIMEOUT")
    _close(proc)
    span = time.monotonic() - started
    finalize_counts(out)
    write(out / "physical/throughput_canary.json", {"phase": "canary", "gpu": gpu, "workers": 1,
                                                   "span_seconds": span, "exit_code": code,
                                                   "single_worker_baseline": "NOT_MEASURED",
                                                   "speedup_claim": "NONE (no matched single-worker baseline)"})
    res_path = out / "physical/branch_results" / f"{canary_b['branch_id']}.json"
    result = read(res_path) if res_path.is_file() else {"status": "NO_RESULT"}
    write(out / "canary_result.json", {"branch_id": canary_b["branch_id"], "gpu": gpu, "exit_code": code,
                                       "status": result.get("status"), "task_success": result.get("task_success"),
                                       "actions": len(result.get("actions", [])),
                                       "env_counts": result.get("env_counts"), "worker_wall_seconds": span})
    gate = check_canary(root, out)
    if gate["status"] != PASS_CANARY:
        mark_remainder_not_released(out)
    return {"canary": canary_b["branch_id"], "gpu": gpu, "gate": gate["status"], "dispatched": 1}


def remainder_queues(branches, canary_gpu, gpus):
    """layout_0 pad_u on the canary GPU; layout_1 pad_v then pad_u sequentially on one second GPU."""
    by = {branch_key(b): b for b in branches}
    l0u, l1v, l1u = (by[k] for k in REMAINDER)
    others = [g for g in gpus if g != canary_gpu]
    if not others:
        return {canary_gpu: [l0u, l1v, l1u]}
    return {canary_gpu: [l0u], others[0]: [l1v, l1u]}


def run_remainder(root, out, gpus, launcher=default_launcher, identity_check=verify_identity, sleep=time.sleep):
    root, out = Path(root).resolve(), Path(out).resolve()
    gate = check_canary(root, out, write_file=False)
    recorded = read(out / "canary_gate.json") if (out / "canary_gate.json").is_file() else {}
    if gate["status"] != PASS_CANARY or recorded.get("status") != PASS_CANARY:
        raise ValueError("STOPPED_CANARY_NOT_PASS")
    identity_check(root, out)
    reg = read(out / "physical/registration.json")
    ledger = read(out / "budget_ledger_r2.json")
    if ledger["attempts"]["used"] != 1 or ledger["phases"]["remainder"]["used"] != 0:
        raise ValueError("STOPPED_REMAINDER_STATE")
    canary_gpu = read(out / "canary_result.json")["gpu"]
    rest = [b for b in reg["branches"] if branch_key(b) in REMAINDER]
    if len(rest) != 3 or any(b["phase"] != "remainder" for b in rest):
        raise ValueError("STOPPED_REMAINDER_CARDINALITY")
    _no_prior_results(out, rest)
    pool = [canary_gpu] + [g for g in gpus if g != canary_gpu][:1]  # at most 2 workers
    todo = remainder_queues(rest, canary_gpu, pool)
    slots = list(todo)
    active, faults, dispatched, started = {}, [], [], time.monotonic()
    while any(todo.values()) or active:
        for gpu in slots:
            if gpu in active or not todo[gpu] or faults:
                continue
            b = todo[gpu].pop(0)
            reserve_and_charge(out, b, "remainder")
            proc = launcher(root, out, b, gpu)
            active[gpu] = (proc, b["branch_id"])
            dispatched.append({"branch_id": b["branch_id"], "gpu": gpu, "t_start": time.monotonic() - started})
        if not active:
            break
        sleep(0.5)
        for gpu, (proc, bid) in list(active.items()):
            code = proc.poll()
            if code is None:
                if time.monotonic() - started > WATCHDOG_SECONDS * 3:
                    _kill(proc)
                    faults.append({"branch_id": bid, "reason": "WATCHDOG_TIMEOUT"})
                    from cp_disr.analysis.s1_integration import finish_branch_attempt
                    finish_branch_attempt(out / "physical", bid, "UNKNOWN", execution_status="WATCHDOG_TIMEOUT")
                    _close(proc)
                    del active[gpu]
                continue
            _close(proc)
            del active[gpu]
            path = out / "physical/branch_results" / f"{bid}.json"
            if not path.is_file():
                faults.append({"branch_id": bid, "exit_code": code, "reason": "RESULT_MISSING"})
            elif read(path)["status"] == "EXCEPTION":
                faults.append({"branch_id": bid, "exit_code": code, "reason": read(path).get("detail")})
    span = time.monotonic() - started
    finalize_counts(out)
    write(out / "physical/throughput_remainder.json", {
        "phase": "remainder", "workers": len(slots), "gpus": slots, "dispatched": dispatched, "faults": faults,
        "span_seconds": span, "single_worker_baseline": "NOT_MEASURED",
        "speedup_claim": "NONE (no matched single-worker baseline)"})
    return {"dispatched": [d["branch_id"] for d in dispatched], "faults": faults}


# ---------------------------------------------------------------------------------- worker
def _camera_record(env, profile):
    """Live vs frozen camera table (pure reads of the constructed model; nothing is written back)."""
    from cp_disr.platforms.libero import family_b_obs_v2 as obs
    refs = obs.selected_camera_refs(profile)
    rows, ok_all = {}, True
    names = list(getattr(env, "camera_names", obs.CAMERAS))
    for name in obs.CAMERAS:
        live, ref = obs.live_camera(env, name), refs[name]
        i = names.index(name) if name in names else None
        width = getattr(env, "camera_widths", [None] * 9)[i] if i is not None else None
        height = getattr(env, "camera_heights", [None] * 9)[i] if i is not None else None
        d_pos = float(np.max(np.abs(np.asarray(live["pos"]) - np.asarray(ref["pos"]))))
        d_quat = float(np.max(np.abs(np.asarray(live["quat"]) - np.asarray(ref["quat"]))))
        d_fov = abs(live["fovy"] - ref["fovy"])
        tol = obs.FIXED_ATOL
        fixed_live = live["mode"] == 0 and live["parent_body"] == 0
        checks = {"mode": live["mode"] == ref["mode"], "pos": d_pos <= tol, "quat": d_quat <= tol,
                  "fovy": d_fov <= tol, "width": width == ref["width"], "height": height == ref["height"],
                  "fixed_in_world": bool(fixed_live and ref["fixed_in_world"] is True)}
        ok = all(checks.values())
        ok_all &= ok
        rows[name] = {"frozen": {k: ref[k] for k in ("mode", "pos", "quat", "fovy", "width", "height", "fixed_in_world")},
                      "live": {"mode": live["mode"], "parent_body": live["parent_body"], "pos": live["pos"],
                               "quat": live["quat"], "fovy": live["fovy"], "width": width, "height": height,
                               "fixed_in_world": bool(fixed_live)},
                      "abs_diff": {"pos_max": d_pos, "quat_max": d_quat, "fovy": d_fov},
                      "tolerance": tol, "checks": checks, "pass": bool(ok)}
    return {"profile_sha256": profile["profile_sha256"], "cameras": rows, "status": "PASS" if ok_all else "FAIL",
            "fallback_used": False, "live_values_written_back": False}


def worker(root, out, branch_id):
    """R2 worker: coordinator reservation required; reuses the v2 worker unchanged otherwise."""
    from cp_disr.platforms.libero import family_b_obs_v2 as obs
    root, out = Path(root), Path(out)
    ledger = read(out / "budget_ledger_r2.json")
    if branch_id not in ledger["reserved_branches"]:
        raise RuntimeError("STOPPED_BUDGET:branch was not reserved by the coordinator")
    m.SOURCE_FILES = SOURCE_FILES_R2  # worker-side source identity covers the R2 files too
    real = obs.verify_frozen_cameras
    cap = out / "captures" / branch_id
    cap.mkdir(parents=True, exist_ok=True)

    def recording_verify(env, profile):
        try:
            write(cap / "camera_live_vs_frozen.json", _camera_record(env, profile))
        except Exception as exc:  # evidence only; the production check below still fails closed
            write(cap / "camera_live_vs_frozen.json", {"status": "RECORD_ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return real(env, profile)

    obs.verify_frozen_cameras = recording_verify
    return m.worker(root, out, branch_id)


# ------------------------------------------------------------------------- evaluation / gates
GROUPS = {
    "A_source_config": ("registered_identity", "profile_identity", "camera_live_vs_frozen"),
    "B_environment_restore": ("no_exception", "constructor_and_single_reset", "restore_receipt",
                              "recorder_errors_empty"),
    "C_decision_state": ("setup_complete", "held_carrier_true", "completed_target_confirmed", "both_pads_legal",
                         "candidate_ids_stable", "two_fixed_views_public", "hidden_flags_clean"),
    "D_observation_chain": ("chain_files_saved", "remaining_publicly_supported", "selected_view_present",
                            "on_table_true", "held_false", "pick_mask_true"),
    "E_execution": ("place_buffer_normal", "task_success_expected_sequence", "evaluator_terminal_success",
                    "planner_semantics", "planner_plan_found_after_candidate", "no_skill_retry",
                    "no_planner_after_success"),
}


def _chain_files(cap, last):
    d = Path(cap) / "action_03"
    need = [f"{last['frame_id']}_{c}_{k}" for c in m.CAMERAS for k in ("rgb.png", "depth_metric.npy")]
    got = all(any(d.rglob(n)) for n in need)
    ba = all(any(d.rglob(f"{s}_{c}_rgb.png")) for s in ("before", "after") for c in m.CAMERAS)
    return got and ba


def _sideview_selected(cap):
    n = 0
    for vp in Path(cap).glob("action_*/views_perception.json"):
        for frame in read(vp):
            n += sum(1 for f in frame["fusion"].values() if f["selected_view"] == "sideview")
    return n


def evaluate_branch(out, b, attempts=None):
    """All per-branch checks (shared by canary and full gate).  Never reads hidden truth for a verdict."""
    out = Path(out)
    bid = b["branch_id"]
    cap = out / "captures" / bid
    attempts = attempts if attempts is not None else (
        read(out / "physical/attempt_registry.json") if (out / "physical/attempt_registry.json").is_file() else {})
    row = {"branch_id": bid, "layout": b["layout"], "context": b["context"], "candidate": b["candidate"],
           "phase": b["phase"]}
    checks, problems = {}, []

    def chk(name, ok, why=""):
        checks[name] = bool(ok)
        if not ok:
            problems.append(f"{bid}:{name}" + (f":{why}" if why else ""))

    res_path = out / "physical/branch_results" / f"{bid}.json"
    if not res_path.is_file():
        chk("terminal_receipt", False, "no terminal result")
        return {**row, "checks": checks, "problems": problems, "terminal_receipt": False}
    r = read(res_path)
    acts = r.get("actions", [])
    row["status"] = r.get("status")
    chk("terminal_receipt", attempts.get(bid) in ("COMPLETED", "FAILED", "UNKNOWN"))
    chk("registered_identity", r.get("attempt_identity") == b["attempt_identity"] and
        r.get("branch_id") == bid and b.get("card_id") == CARD and "R2" in b["attempt_identity"])
    chk("profile_identity", r.get("observation_profile_sha256") == FROZEN_SHA == b["observation_profile_sha256"])
    cam_path = cap / "camera_live_vs_frozen.json"
    cam = read(cam_path) if cam_path.is_file() else {}
    chk("camera_live_vs_frozen", cam.get("status") == "PASS" and cam.get("fallback_used") is False and
        list(cam.get("cameras", {})) == list(m.CAMERAS))
    row["camera_live_vs_frozen_status"] = cam.get("status")
    chk("no_exception", r.get("status") != "EXCEPTION" and "traceback" not in r, str(r.get("detail", ""))[:200])
    counts = r.get("env_counts", {})
    chk("constructor_and_single_reset", counts.get("constructions") == 1 and counts.get("reset_calls") == 1 and
        counts.get("bootstrap_constructions", 0) == 0, str(counts))
    row["env_counts"] = counts
    chk("restore_receipt", bool(r.get("restore_receipt")))
    row["recorder_errors"] = r.get("recorder_errors", [])
    chk("recorder_errors_empty", not row["recorder_errors"])
    setup = acts[:3]
    chk("setup_complete", len(setup) == 3 and [a["action_id"] for a in setup] == b["prefix"] and
        all(a["controller_exit"] == "NORMAL_TERMINATION" for a in setup) and (cap / "boundary/public.json").is_file())
    pub = read(cap / "boundary/public.json") if (cap / "boundary/public.json").is_file() else None
    done = {"B_PENDING": "obj_c", "C_PENDING": "obj_b"}[b["context"]]
    facts = pub["fact_values"] if pub else {}
    chk("held_carrier_true", facts.get("p:Held:carrier") == "TRUE")
    chk("completed_target_confirmed", facts.get(f"p:Inside:{done}:receiver") == "TRUE")
    ids = pub["candidate_ids"] if pub else []
    mask = pub["candidate_mask"] if pub else []
    chk("both_pads_legal", bool(pub) and all(c in ids and mask[ids.index(c)] for c in
                                             (f"a:PLACE_BUFFER:carrier:{p}:v1" for p in m.PAD_PAIRS)))
    snap3 = read(cap / "action_03/snapshot.json") if (cap / "action_03/snapshot.json").is_file() else None
    chk("candidate_ids_stable", bool(pub) and snap3 is not None and snap3["candidate_ids"] == ids)
    vb = read(cap / "boundary/views_boundary.json") if (cap / "boundary/views_boundary.json").is_file() else {}
    chk("two_fixed_views_public", vb.get("cameras") == list(m.CAMERAS) and set(vb.get("views", {})) == set(m.CAMERAS))
    qa_flags, vflags = [], []
    for qa in cap.glob("action_*/qa_state.jsonl"):
        for line in qa.read_text().splitlines():
            rec = json.loads(line)
            qa_flags += [rec.get(k) for k in ("used_by_planner", "used_by_policy", "used_by_provider",
                                              "used_by_verifier")]
    for vp in cap.glob("action_*/views_perception.json"):
        vflags += [(f["hidden_truth_used"], f["qpos_qvel_used"]) for f in read(vp)]
    chk("hidden_flags_clean", bool(qa_flags) and not any(qa_flags) and bool(vflags) and not any(any(f) for f in vflags))
    rem = m.remaining_object(b["context"])
    if len(acts) > 3 and snap3 is not None and (cap / "action_03/views_perception.json").is_file():
        support = m.remaining_support(cap, b["context"], THRESHOLDS["min_pixels"])
        last = read(cap / "action_03/views_perception.json")[-1]
        row["remaining_public_support"] = support
        row["per_view"] = {v: {"mask_pixels": last["cameras"][v]["objects"][rem]["mask_pixels"],
                               "depth_support": last["cameras"][v]["objects"][rem]["depth_support"],
                               "blob_present": last["cameras"][v]["objects"][rem]["blob_present"]} for v in m.CAMERAS}
        chk("chain_files_saved", _chain_files(cap, last))
        chk("remaining_publicly_supported", support.get("supported") is True)
        chk("selected_view_present", support.get("selected_view") in m.CAMERAS)
        chk("on_table_true", support.get("on_table") == "TRUE")
        chk("held_false", support.get("held") == "FALSE")
        chk("pick_mask_true", support.get("pick_mask") is True)
    else:
        row["remaining_public_support"] = {"ok": False, "reason": "candidate action not executed or record missing"}
        for n in GROUPS["D_observation_chain"]:
            chk(n, False, "no post-candidate chain")
    seq = [a["action_id"] for a in acts]
    expected = b["prefix"] + [b["candidate_id"], f"a:PICK:{rem}:v1", f"a:PLACE:{rem}:receiver:v1"]
    chk("place_buffer_normal", len(acts) > 3 and acts[3]["controller_exit"] == "NORMAL_TERMINATION")
    chk("task_success_expected_sequence", r.get("status") == "TASK_SUCCESS" and r.get("task_success") is True and
        seq == expected)
    evaluators = {i: read(cap / f"action_{i:02d}/evaluator.json") for i in range(len(acts))
                  if (cap / f"action_{i:02d}/evaluator.json").is_file()}
    chk("evaluator_terminal_success", bool(evaluators) and evaluators[max(evaluators)]["task_success"] is True and
        max(evaluators) == len(acts) - 1)
    psem = m.planner_semantics(cap, acts, evaluators)
    row["planner_semantics_problems"] = psem
    chk("planner_semantics", not psem, ";".join(psem))
    cand_plan = read(cap / "action_03/planner.json") if (cap / "action_03/planner.json").is_file() else {}
    chk("planner_plan_found_after_candidate", cand_plan.get("status") == "PLAN_FOUND" and
        cand_plan.get("plan", [None])[:1] == [f"a:PICK:{rem}:v1"])
    chk("no_skill_retry", len(seq) == len(set(seq)) and len(seq) == 6)
    chk("no_planner_after_success", bool(acts) and not (cap / f"action_{len(acts) - 1:02d}/planner.json").is_file())
    row.update(checks=checks, problems=problems, terminal_receipt=checks["terminal_receipt"],
               setup_complete=checks["setup_complete"], task_success=checks["task_success_expected_sequence"],
               sideview_selected_any=_sideview_selected(cap))
    return row


def check_canary(root, out, write_file=True):
    out = Path(out)
    reg = read(out / "physical/registration.json")
    b = canary_of(reg["branches"])
    row = evaluate_branch(out, b)
    groups = {g: {n: row["checks"].get(n, False) for n in names} for g, names in GROUPS.items()}
    ok = all(all(v.values()) for v in groups.values()) and not row["problems"]
    gate = {"card_id": CARD, "status": PASS_CANARY if ok else FAIL_CANARY, "branch_id": b["branch_id"],
            "canary": list(CANARY), "groups": groups, "problems": row["problems"], "row": row,
            "remainder_released": bool(ok),
            "unused_remainder_slots_status": "RELEASED_TO_REMAINDER_PHASE" if ok else
            "NOT_RELEASED_AFTER_CANARY_FAILURE"}
    if write_file:
        write(out / "canary_gate.json", gate)
    return gate


def _read_ledger(out):
    return read(Path(out) / "budget_ledger_r2.json")


def aggregate_camera(out):
    out = Path(out)
    reg = read(out / "physical/registration.json")
    doc = {"profile_sha256": FROZEN_SHA, "branches": {}}
    for b in reg["branches"]:
        p = out / "captures" / b["branch_id"] / "camera_live_vs_frozen.json"
        doc["branches"][b["branch_id"]] = read(p) if p.is_file() else {"status": "NOT_REACHED"}
    doc["all_pass"] = all(v.get("status") == "PASS" for v in doc["branches"].values())
    write(out / "camera_live_vs_frozen.json", doc)
    return doc


def merge_throughput(out):
    out = Path(out)
    phases = {p.stem.split("_", 1)[1]: read(p) for p in sorted((out / "physical").glob("throughput_*.json"))}
    write(out / "throughput_report.json", {"phases": phases, "single_worker_baseline": "NOT_MEASURED",
                                           "speedup_claim": "NONE (no matched single-worker baseline)"})


def gate_r2(root, out):
    out = Path(out)
    reg = read(out / "physical/registration.json")
    attempts = read(out / "physical/attempt_registry.json") if (out / "physical/attempt_registry.json").is_file() else {}
    ledger = finalize_counts(out)
    problems, rows = [], {}
    for b in reg["branches"]:
        row = evaluate_branch(out, b, attempts)
        rows[b["branch_id"]] = row
        problems += row["problems"]
    have = [b for b in reg["branches"] if (out / "physical/branch_results" / f"{b['branch_id']}.json").is_file()]
    if len(have) != 4:
        problems.append("not 4/4 terminal receipts")
    pair = {"status": "FAIL", "reason": "not all four branches ran"}
    if len(have) == 4:
        pair = m.paired_restore(out)
        src = out / "paired_restore_v2.json"
        if src.is_file():
            write(out / "paired_restore_v2_r2.json", read(src))
            src.unlink()
        if pair["status"] != "PASS":
            problems.append("paired restore equivalence not established")
    else:
        write(out / "paired_restore_v2_r2.json", pair)
    scan = m._scan_forbidden()
    if scan:
        problems.append("perception module references hidden/QA state: " + ",".join(scan))
    cams = aggregate_camera(out)
    if not cams["all_pass"]:
        problems.append("camera live-vs-frozen not PASS for every branch")
    pad_checks = {f"{b['layout']}/{b['candidate']}": rows[b["branch_id"]]["checks"].get("remaining_publicly_supported")
                  for b in reg["branches"]}
    counts_ok = (ledger["attempts"]["used"] <= 4 and ledger["environment_constructions"]["used"] <= 4 and
                 ledger["explicit_resets"]["used"] <= 4 and ledger["live_skill_calls"]["used"] <= 24 and
                 all(ledger[k]["used"] == 0 for k in ("provider_calls", "rl_transitions", "optimizer_steps", "elastic",
                                                       "skill_retries", "standalone_capture_resets")))
    if not counts_ok:
        problems.append("budget counts exceed authorization")
    if ledger["original_remaining_20"] != "NOT_RELEASED":
        problems.append("original remaining 20 not NOT_RELEASED")
    canary_gate = read(out / "canary_gate.json") if (out / "canary_gate.json").is_file() else {"status": "NOT_RUN"}
    if canary_gate["status"] != PASS_CANARY:
        problems.append("canary not PASS")
    ok = not problems
    gate = {"card_id": CARD, "status": "PASS" if ok else "FAIL", "label": GATE_PASS if ok else GATE_FAIL,
            "problems": problems, "branches": rows, "paired_restore_status": pair["status"],
            "camera_live_vs_frozen_all_pass": cams["all_pass"], "canary_status": canary_gate["status"],
            "hidden_scan_hits": scan, "v2_r2_attempts_used": ledger["attempts"]["used"],
            "remaining_20_released": False, "compares_u_vs_v_cost": False,
            "remaining_target_supported_by_candidate": pad_checks}
    write(out / "technical_gate_r2.json", gate)
    m.branch_results_csv(out)
    merge_throughput(out)
    return gate


def summarize(root, out):
    out = Path(out)
    gate = read(out / "technical_gate_r2.json")
    ledger = _read_ledger(out)
    reg = read(out / "physical/registration.json")
    canary = read(out / "canary_gate.json")
    act = ledger["actual"]
    used = ledger["attempts"]["used"]
    lines = [
        "# Family B observation v2-r2 technical validation", "",
        f"- Card: {CARD}; frozen profile `{m.PROFILE_VERSION}` sha256 {FROZEN_SHA} (unchanged, cameras agentview+sideview).",
        "- v1 consumed attempts: 4 (permanent); v2-r1 consumed attempts: 2 (+2 unused slots RETIRED_NOT_TRANSFERABLE).",
        f"- v2-r2 attempts used: {used} (cap 4); cumulative attempts used: {4 + 2 + used}.",
        f"- Successful constructors (actual): {act.get('constructor_successful')}; constructor attempts: "
        f"{act.get('constructions_attempted')}; explicit reset calls (actual): {act.get('explicit_resets')}; "
        f"internal constructor resets: {act.get('internal_resets')}; live skill calls (actual): {act.get('skill_calls')}.",
        "- original remaining 20: NOT_RELEASED; provider/RL/optimizer = 0; S2/S3/formal test/TP training = false.",
        f"- Canary: **{canary['status']}** (layout_0/B_PENDING/repeat0/pad_v); remainder slots: "
        f"{ledger['remainder_slots_status']}.",
        f"- Full R2 technical gate: **{gate['status']}** ({gate['label']}).", "",
        "| branch | layout/context | candidate | task end | remaining target (per-view mask/depth support) | selected view | OnTable | PICK mask |",
        "|---|---|---|---|---|---|---|---|"]
    side = 0
    for b in reg["branches"]:
        r = gate["branches"][b["branch_id"]]
        sup = r.get("remaining_public_support", {})
        pv = r.get("per_view", {})
        pvs = "; ".join(f"{v}: {d['mask_pixels']}/{d['depth_support']}" for v, d in pv.items()) or "n/a"
        side += r.get("sideview_selected_any", 0)
        lines.append(f"| {b['branch_id']} | {b['layout']}/{b['context']} | {b['candidate']} | {r.get('status')} | {pvs} | "
                     f"{sup.get('selected_view')} | {sup.get('on_table')} | {sup.get('pick_mask')} |")
    lines += ["", "- Post-pad_v public observability of the remaining target: " + ", ".join(
        f"{b['layout']}={gate['branches'][b['branch_id']].get('remaining_public_support', {}).get('ok')}"
        for b in reg["branches"] if b["candidate"] == "pad_v"),
        f"- sideview actually selected (fusion selections across all frames/objects): {side}",
        f"- Paired restore: {gate['paired_restore_status']}; camera live-vs-frozen all pass: "
        f"{gate['camera_live_vs_frozen_all_pass']}.",
        "- Gate problems: " + ("none" if not gate["problems"] else "; ".join(gate["problems"])),
        "- U vs V cost is not compared by this card; speedup NOT_MEASURED.",
        "- Next step requires explicit user authorization (remainder review / any release of the original 20)."]
    (out / "final_summary.md").write_text("\n".join(lines) + "\n")


def inventory_before(root, out):
    inv = m.inventory(protected_paths(root))
    write(Path(out) / "protected_before.json", {"sha256": inv, "files": len(inv),
                                                "capture_count": sum("/captures/" in k for k in inv)})
    return len(inv)


def verify(root, out):
    out = Path(out)
    root = Path(root)
    before = read(out / "protected_before.json")
    after_inv = m.inventory(protected_paths(root))
    write(out / "protected_after.json", {"sha256": after_inv, "files": len(after_inv),
                                         "capture_count": sum("/captures/" in k for k in after_inv)})
    diff = m.diff_inventory(before["sha256"], after_inv)
    ledger = _read_ledger(out)
    reg = read(out / "physical/registration.json")
    results = list((out / "physical/branch_results").glob("*.json"))
    checks = {
        "protected_unchanged": not (diff["changed"] or diff["removed"] or diff["added"]),
        "frozen_profile_files_equal_freeze_commit": not frozen_bytes_ok(root),
        "attempts_le_4": ledger["attempts"]["used"] <= 4 and len(results) <= 4,
        "constructions_le_4": ledger["environment_constructions"]["used"] <= 4,
        "resets_le_4": ledger["explicit_resets"]["used"] <= 4,
        "skills_le_24": ledger["live_skill_calls"]["used"] <= 24,
        "no_fifth_attempt": len(reg["branches"]) == 4 and
        len(set(ledger["reserved_branches"])) == len(ledger["reserved_branches"]),
        "provider_rl_optimizer_zero": all(ledger[k]["used"] == 0 for k in
                                          ("provider_calls", "rl_transitions", "optimizer_steps", "elastic")),
        "remaining_20_not_released": ledger["original_remaining_20"] == "NOT_RELEASED",
        "gate_file_present": (out / "technical_gate_r2.json").is_file(),
        "r1_unused_slots_not_transferred": ledger["history"]["v2_r1"]["unused_slots_status"] == "RETIRED_NOT_TRANSFERABLE"}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "protected_diff": diff,
              "protected_files": len(after_inv)}
    write(out / "verify.json", result)
    return result
