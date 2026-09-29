"""S4 T_P_SOFT_RELOCATION_V1 design feasibility: ledger, freeze-spec, pool, static screen, physical probe, opportunity.

Card CP-DISR-S4-TASK-REDESIGN-1 (method_version 2.1.1). Curated mechanism demonstration, not an unbiased benchmark.
Everything here reuses production runtime, branch executor, reserve/claim/finish receipts and paired-restore QA.
No RL, optimizer, training, checkpoint, S2/S3, formal test or Method 2.2 code path exists in this module.
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

from cp_disr.analysis.s1_integration import (IntegrationError, _append_jsonl, _atomic_json, _load_attempts, _transaction_lock,
                                             finish_branch_attempt, reserve_branch_attempt, sha256_file)

CARD_ID = "CP-DISR-S4-TASK-REDESIGN-1"
FAMILY = "T_P_SOFT_RELOCATION_V1"
TASK_ID = "T_P_SR"
METHOD_VERSION = "2.1.1"
BASE_COMMIT = "7dc6965cc909cb0a79087e433006d56d6415096c"
PHYS_SALT = "TP_SR_PHYSICAL_V1"
DIRECT = "a:PICK:target:v1"
RELOC = "a:PICK:interferer:v1"
DIRECT_SEQ = ("a:PICK:target:v1", "a:PLACE:target:container:v1")
RELOC_SEQ = ("a:PICK:interferer:v1", "a:PLACE_BUFFER:interferer:buffer:v1", "a:PICK:target:v1", "a:PLACE:target:container:v1")
LEDGER_KEYS = ("candidate_reset_configs", "static_capture_resets", "physical_qualification_scenes", "physical_branch_episodes",
               "provider_first_calls", "provider_retries", "representation_forward_cases", "rl_transitions", "optimizer_steps",
               "training_attempts", "elastic_attempts", "formal_test_episodes")
CODE_FILES = ("src/cp_disr/analysis/s4_tp_sr_design.py", "src/cp_disr/analysis/s4_tp_sr_provider.py", "scripts/s4_tp_sr_design.py",
              "src/cp_disr/platforms/libero/tp_sr_generator.py", "src/cp_disr/platforms/libero/tp_sr_runtime.py",
              "configs/final_master/s4_tp_sr_design.yaml", "configs/runtime/tp_sr_contract_registry.yaml", "configs/tasks/resolved/T_P_SR.yaml")
INITIAL_FACTS = {"p:GripperEmpty": 1, "p:Held:target": 0, "p:Held:interferer": 0, "p:OnTable:target": 1, "p:OnTable:interferer": 1,
                 "p:Open:container": 1, "p:Inside:target:container": 0, "p:Inside:interferer:container": 0,
                 "p:AtBuffer:target:buffer": 0, "p:AtBuffer:interferer:buffer": 0}
REQUIRED_TRUE = ("p:GripperEmpty", "p:OnTable:target", "p:OnTable:interferer", "p:Open:container")
REQUIRED_FALSE = ("p:Inside:target:container", "p:AtBuffer:interferer:buffer")
BINS = (0, 1, 2, 3)
BIN_QUOTA = {3: 6, 2: 6, 1: 6, 0: 6}   # high-occlusion 12 (bins 3+2), mid 6, low 6
NORMAL_EXITS = {"NORMAL_TERMINATION", "SUCCESS"}
ENGINEERING_STATUS = {"EXCEPTION", "UNKNOWN_CONTROLLER_EXIT", "REJECTED_MASK", "ENGINEERING_STOP", "DECISION_CAP", "SYMBOLIC_GOAL"}


class StopRun(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rd(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path):
    with Path(path).open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ ledger
def init_ledger(out, config):
    caps = config["budgets"]
    ledger = {k: {"cap": int(caps.get(k, 0)), "used": 0} for k in LEDGER_KEYS}
    _atomic_json(Path(out) / "budget_ledger.json", ledger)
    (Path(out) / "budget_events.jsonl").touch()
    return ledger


def charge(out, key, n=1, ref=""):
    """Atomically consume n units of a capped budget; raises StopRun('STOPPED_BUDGET_EXHAUSTED') and consumes nothing on overflow."""
    out = Path(out)
    with _transaction_lock(out):
        ledger = rd(out / "budget_ledger.json")
        item = ledger[key]
        if int(item["used"]) + n > int(item["cap"]):
            raise StopRun("STOPPED_BUDGET_EXHAUSTED", f"{key} {item['used']}+{n}>{item['cap']}")
        item["used"] = int(item["used"]) + n
        _atomic_json(out / "budget_ledger.json", ledger)
        _append_jsonl(out / "budget_events.jsonl", {"event": "charge", "key": key, "n": n, "used": item["used"], "cap": item["cap"], "ref": ref, "ts": now()})
        return item["used"]


def ledger(out):
    return rd(Path(out) / "budget_ledger.json")


def sync_physical(out):
    """Mirror the physical branch ledger (production primitive key) into the S4 ledger; only ever increases."""
    out = Path(out)
    pl = rd(out / "physical/budget_ledger.json")["physical_witness_episodes"]
    with _transaction_lock(out):
        led = rd(out / "budget_ledger.json")
        if int(pl["used"]) > int(led["physical_branch_episodes"]["used"]):
            led["physical_branch_episodes"]["used"] = int(pl["used"])
            _atomic_json(out / "budget_ledger.json", led)
            _append_jsonl(out / "budget_events.jsonl", {"event": "sync_physical", "used": int(pl["used"]), "ts": now()})
    return int(pl["used"])


def event(out, kind, **fields):
    _append_jsonl(Path(out) / "scheduling_events.jsonl", {"event": kind, "timestamp_utc": now(), "monotonic": time.monotonic(), **fields})


# ------------------------------------------------------------------ manifests
def authority_check(root):
    root = Path(root)
    head = git(root, "rev-parse", "HEAD")
    if subprocess.call(["git", "-C", str(root), "merge-base", "--is-ancestor", BASE_COMMIT, head]) != 0:
        raise StopRun("STOPPED_AUTHORITY_RECONCILIATION_REQUIRED", "HEAD does not descend from the S1 closeout commit")
    am = root / "docs/authoritative/authority_manifest.json"
    need = [am, root / "docs/authoritative/CP_DISR_Final_Experimental_Plan_v3.md", root / "docs/authoritative/CP_DISR_Final_Research_Content_v5.md"]
    if not all(p.is_file() for p in need):
        raise StopRun("STOPPED_AUTHORITY_RECONCILIATION_REQUIRED", "authority manifest/docs missing")
    return {"head": head, "authority_manifest_sha256": sha256_file(am), "base_commit": BASE_COMMIT}


def out_dir_for(root, config_path):
    sha8 = sha256_file(config_path)[:8]
    return Path(root) / "runs/final_master/S4/tp_soft_relocation_design" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{sha8}"


def runtime_manifest(root, out, force=False):
    """T_P_SR runtime manifest bound to the tp_sr_runtime source hash. Rewritten (with an event) if the source changed."""
    root, out = Path(root), Path(out)
    src = root / "src/cp_disr/platforms/libero/tp_sr_runtime.py"
    path = out / "spec/runtime_manifest.yaml"
    sha = sha256_file(src)
    if path.is_file() and not force:
        cur = yaml.safe_load(path.read_text())
        if cur["runtime_factory"]["sha256"] == sha:
            return path
        event(out, "runtime_manifest_rebound", old=cur["runtime_factory"]["sha256"], new=sha)
    base = json.loads(json.dumps(yaml.safe_load((root / "runs/final_master/S1/20260928T154311Z_bea4bf0d/input_binding/T_A_s1_rev1_runtime_manifest.yaml").read_text())))
    rt = base["runtime"]
    rt["active_task_id"] = TASK_ID
    rt["repository_path"] = str(root)
    rt["experiment_root"] = str(root / "experiments")
    rt["stage_2a_contract_path"] = "configs/runtime/tp_sr_contract_registry.yaml"
    rt["reference_skill_seconds_by_task"] = {TASK_ID: 3.649999999999709}
    rt["skill_timeouts"] = {TASK_ID: {"PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}}
    rt["task_deadlines"] = {TASK_ID: 60.0}
    rt["task_evaluator_version"] = {TASK_ID: "cp-disr-tpsr-task-evaluator-v1"}
    rt["task_assets"] = {TASK_ID: "src/cp_disr/platforms/libero/d0_env.py", "license": "assets/cp_disr/LICENSE", "owned_by": "cp_disr_project"}
    rt["task_splits"] = {TASK_ID: str((out / "pool/T_P_SR_pool_split.json").relative_to(root))}
    rt["decision_cap_by_task"] = {TASK_ID: 8}
    base["runtime_factory"] = {"module": "cp_disr.platforms.libero.tp_sr_runtime", "factory": "create", "source_path": str(src), "sha256": sha}
    base["manifest_status"] = "S4_TP_SR_DESIGN_RUNTIME_BOUND_TRAINING_NOT_STARTED"
    base["note"] = "S4 T_P_SR design-feasibility runtime; no training, no formal test."
    base["vlm_runtime"] = {"api_account_authorized": "MUST_VERIFY_MODEL_ACCESS", "automatic_fallback_allowed": False,
                           "base_http_api_url": "https://dashscope.aliyuncs.com/api/v1", "region": "cn-beijing"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
    return path


def _truth(v):
    from cp_disr.facts import Truth
    return {1: Truth.TRUE, 0: Truth.FALSE, -1: Truth.UNKNOWN}[v]


def route_search(contracts, start, goal_id, max_depth=6):
    """Exhaustive nominal simple-path search over T/F/U vectors using the production precondition/overlay functions."""
    from cp_disr.common import ContractError
    from cp_disr.contracts import nominal_overlay, precondition_value
    from cp_disr.facts import Truth
    order = sorted(start)
    key = lambda v: tuple(v[k].value for k in order)
    found = []
    stack = [(dict(start), (), {key(start)})]
    while stack:
        vals, seq, seen = stack.pop()
        if vals[goal_id] == Truth.TRUE:
            found.append(seq)
            continue
        if len(seq) >= max_depth:
            continue
        for c in sorted(contracts, key=lambda x: x.id):
            if precondition_value(c, vals) != Truth.TRUE:
                continue
            try:
                nxt = dict(nominal_overlay(c, vals))
            except ContractError:
                continue
            k = key(nxt)
            if k in seen:
                continue
            stack.append((nxt, seq + (c.id,), seen | {k}))
    return sorted(found)


def contract_reachability(root, facts=None):
    from cp_disr.platforms.libero import tp_sr_runtime as rt
    root = Path(root)
    timeouts = {"PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}
    contracts = rt.ground_task_contracts(root / "configs/runtime/tp_sr_contract_registry.yaml", timeouts)
    template = rt.build_task_template(contracts)
    start = {k: _truth(v) for k, v in INITIAL_FACTS.items()} if facts is None else dict(facts)
    routes = route_search(contracts, start, "p:Inside:target:container")
    return {"routes": [list(r) for r in routes], "direct_reachable": tuple(DIRECT_SEQ) in routes, "relocation_reachable": tuple(RELOC_SEQ) in routes,
            "action_ids": [c.id for c in contracts], "template_propositions": sorted(n.id for n in template.nodes if n.kind == "PROPOSITION")}


# ------------------------------------------------------------------ A: freeze-spec
RELATIONS = [
    {"relation_id": "R1", "type": "SOFT_RELEVANT_TO_GOAL", "source": "a:PICK:interferer:v1", "effect": "p:Held:interferer", "target": "p:Inside:target:container",
     "scope": "soft relevance of moving the interferer to the goal; not a contract precondition"},
    {"relation_id": "R2", "type": "SOFT_SUPPORTS", "source": "a:PLACE_BUFFER:interferer:buffer:v1", "effect": "p:AtBuffer:interferer:buffer", "target": "a:PICK:target:v1",
     "scope": "relocating the interferer softly supports a later PICK(target)"},
    {"relation_id": "R3", "type": "SOFT_RELEVANT_TO_GOAL", "source": "a:PLACE_BUFFER:interferer:buffer:v1", "effect": "p:AtBuffer:interferer:buffer", "target": "p:Inside:target:container",
     "scope": "relocation is softly relevant to the goal"},
]
REJECTED_RELATIONS = [{"source": "a:PICK:interferer:v1", "target": "a:PLACE_BUFFER:interferer:buffer:v1", "reason": "CONTRACT_REDUNDANCY (direct contract precondition Held(interferer))"}]


def freeze_spec(root, config_path, out):
    root, out = Path(root), Path(out)
    auth = authority_check(root)
    config = load_config(config_path)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("spec", "pool", "static", "physical", "opportunity", "provider", "representation", "raw"):
        (out / sub).mkdir(exist_ok=True)
    init_ledger(out, config)
    reach = contract_reachability(root)
    if not (reach["direct_reachable"] and reach["relocation_reachable"]):
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "both nominal routes must be contract-reachable")
    if any(":OPEN:" in a for a in reach["action_ids"]) or set(reach["action_ids"]) != set(config["allowed_action_ids"]):
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "action set mismatch")
    task = yaml.safe_load((root / "configs/tasks/resolved/T_P_SR.yaml").read_text())
    _atomic_json(out / "spec/task_family_spec.json", {
        "card_id": CARD_ID, "task_family_id": FAMILY, "task_id": TASK_ID, "method_version": METHOD_VERSION,
        "dataset_role": config["dataset_role"], "publication_claim": "NOT_AN_UNBIASED_BENCHMARK",
        "goal": config["goal"], "initial_required_facts": config["initial_required_facts"], "initial_required_candidates": config["initial_required_candidates"],
        "allowed_action_ids": config["allowed_action_ids"], "masked_or_absent_groundings": task["masked_or_absent_groundings"],
        "forbidden_hard_facts": ["ClearPath", "Occluded", "Blocked", "EasyGrasp", "SafeToPick"],
        "geometry_fields_scope": "generator/QA only; never policy input, provider payload, relation admission or reward"})
    _atomic_json(out / "spec/relation_universe.json", {"accepted_types": ["SOFT_SUPPORTS", "SOFT_RELEVANT_TO_GOAL"], "relations": RELATIONS,
                                                       "explicitly_rejected": REJECTED_RELATIONS, "hard_relations_admitted": 0})
    _atomic_json(out / "spec/contract_reachability.json", {**reach, "direct_route": list(DIRECT_SEQ), "relocation_route": list(RELOC_SEQ)})
    _atomic_json(out / "spec/opportunity_rules.json", {
        "STRONG_HELPFUL": "relocation 2/2 and direct <=1/2", "COST_HELPFUL": "both 2/2 and reloc mean sim time <= 0.85 x direct",
        "NEUTRAL": "both 2/2, no controller-failure difference, |delta time| < 20%",
        "HARMFUL": "direct 2/2 and reloc <=1/2, or both 2/2 and reloc >= 1.5 x direct", "UNKNOWN": "everything else",
        "action_sequence_difference_alone": "insufficient", "engineering_failures": "retained in denominator"})
    _atomic_json(out / "spec/physical_gate.json", {"helpful_scenes_min": 8, "neutral_or_harmful_scenes_min": 4, "both_routes_reachable_scenes_min": 12,
                                                   "helpful_occlusion_bins_min": 2, "engineering_failure_rate_max": 0.20})
    _atomic_json(out / "stage_manifest.json", {"card_id": CARD_ID, "task_family_id": FAMILY, "status": "SPEC_FROZEN", "created_utc": now(),
                                              "phases_done": ["freeze-spec"], **auth})
    _atomic_json(out / "decision_manifest.json", {"status": "IN_PROGRESS", "task_family_feasibility": "NOT_DETERMINED", "method_version": METHOD_VERSION,
                                                  "method_upgrade_authorized": False, "s2_authorized": False})
    ids = {"config_sha256": sha256_file(config_path), **{f: sha256_file(root / f) for f in CODE_FILES if (root / f).is_file()}}
    _atomic_json(out / "source_identity.json", {"head": auth["head"], "base_commit": BASE_COMMIT, "files_sha256": ids, "git_status_at_freeze": git(root, "status", "--short")})
    _atomic_json(out / "environment_snapshot.json", {"python": sys.version, "host": os.uname().nodename, "cwd": str(root), "time_utc": now(),
                                                     "env": {k: os.environ.get(k, "") for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}})
    runtime_manifest(root, out)  # split file is created later; manifest only records its path
    return {"status": "SPEC_FROZEN", "out": str(out), **reach}


def mark_phase(out, phase, **fields):
    p = Path(out) / "stage_manifest.json"
    doc = rd(p)
    doc.setdefault("phases_done", [])
    if phase not in doc["phases_done"]:
        doc["phases_done"].append(phase)
    doc.update(fields)
    _atomic_json(p, doc)


def set_decision(out, **fields):
    p = Path(out) / "decision_manifest.json"
    doc = rd(p)
    doc.update(fields)
    _atomic_json(p, doc)


# ------------------------------------------------------------------ C: pool
def generate_pool_phase(root, config_path, out):
    from cp_disr.platforms.libero import tp_sr_generator as gen
    root, out = Path(root), Path(out)
    config = load_config(config_path)
    base_sha = sha256_file(config_path)
    bounds = gen.production_bounds(root)
    cfgs = gen.generate_pool(base_sha, bounds)
    if len(cfgs) != gen.POOL_SIZE or len({c["case_id"] for c in cfgs}) != gen.POOL_SIZE:
        raise StopRun("STOPPED_GENERATOR_INVALID", "pool size/uniqueness")
    per_bin = {b: sum(1 for c in cfgs if c["occlusion_bin"] == b) for b in BINS}
    if any(v != 16 for v in per_bin.values()):
        raise StopRun("STOPPED_GENERATOR_INVALID", f"per-bin counts {per_bin}")
    charge(out, "candidate_reset_configs", len(cfgs), "generate-pool")
    _atomic_json(out / "pool/pool_configs.json", {"generator_version": gen.GENERATOR_VERSION, "base_sha256": base_sha, "bounds": bounds, "configs": cfgs})
    split = {"task_id": TASK_ID, "task_family_id": FAMILY, "train": [], "dev": [gen.split_row(c) for c in cfgs], "test": []}
    _atomic_json(out / "pool/T_P_SR_pool_split.json", split)
    write_csv(out / "pool/pool_index.csv", [{"case_id": c["case_id"], "occlusion_bin": c["occlusion_bin"], "axis": c["axis"], "seed": c["seed"],
                                             **{k: c["features"][k] for k in sorted(c["features"])}} for c in cfgs])
    runtime_manifest(root, out)
    mark_phase(out, "generate-pool", pool_sha256=sha256_file(out / "pool/pool_configs.json"))
    return {"status": "POOL_GENERATED", "configs": len(cfgs), "per_bin": per_bin}


# ------------------------------------------------------------------ D: static screen
def _shard_ids(cfgs, shard):
    if not shard:
        return cfgs
    k, n = (int(x) for x in shard.split("/"))
    return [c for i, c in enumerate(cfgs) if i % n == k]


def static_check_one(bundle, cfg):
    """One reset of one pool config; returns a JSON-able row. QA-only hidden truth is compared with the requested layout."""
    snap = bundle.start_case(cfg["case_id"])
    facts = {k: str(getattr(v, "value", v)) for k, v in snap.facts.values.items()}
    ids = list(snap.candidate_ids)
    mask = [bool(m) for m in snap.mask]
    legal = [i for i, m in zip(ids, mask) if m]
    env = bundle.environment
    hidden = env.hidden_truth()
    obs = env.public_observation()
    meas = bundle.perception.infer(obs)
    blobs = meas.measurements.get("blobs", {})
    row = {"case_id": cfg["case_id"], "occlusion_bin": cfg["occlusion_bin"], "facts": facts,
           "legal_candidates": legal, "candidate_ids": ids}
    reasons = []
    xy = lambda n: np.asarray(hidden[n])[:2]
    checks = {"target_xy": np.allclose(xy("target"), cfg["target_xy"], atol=1e-5), "interferer_xy": np.allclose(xy("interferer"), cfg["second_xy"], atol=1e-5)
              if "interferer" in hidden else False, "container_xy": np.allclose(xy("container"), cfg["container_xy"], atol=1e-5),
              "buffer_xy": np.allclose(xy("buffer"), cfg["buffer_xy"], atol=1e-5)}
    lid = np.asarray(hidden["lid"])[:2]
    checks["lid_parked_open"] = bool(np.allclose(lid, [cfg["container_xy"][0] + 0.26, cfg["container_xy"][1]], atol=1e-5))
    row["hidden_qa"] = {k: bool(v) for k, v in checks.items()}
    row["qa_blob_xyz"] = {k: (v.get("xyz") if v else None) for k, v in blobs.items()}
    row["qa_hidden_xyz"] = {k: np.asarray(hidden[k]).tolist() for k in ("target", "interferer", "container", "buffer", "lid") if k in hidden}
    row["qa_blob_pixels"] = {k: (int(v["pixels"]) if v else 0) for k, v in blobs.items()}
    reasons += [f"HIDDEN_QA_{k}" for k, v in checks.items() if not v]
    tvals = {k: snap.facts.values.get(k) for k in INITIAL_FACTS}
    for k in REQUIRED_TRUE:
        if tvals.get(k) != _truth(1):
            reasons.append(f"FACT_NOT_TRUE:{k}")
    for k in REQUIRED_FALSE:
        if tvals.get(k) != _truth(0):
            reasons.append(f"FACT_NOT_FALSE:{k}")
    if set(legal) != set(bundle_required_candidates()):
        reasons.append("LEGAL_CANDIDATES_NOT_EXACTLY_TWO_PICKS")
    tblob, iblob = blobs.get("target"), blobs.get("interferer")
    row["target_pixels"] = int(tblob["pixels"]) if tblob else 0
    row["interferer_pixels"] = int(iblob["pixels"]) if iblob else 0
    if row["target_pixels"] < 8:
        reasons.append("TARGET_NOT_READABLE")
    if row["interferer_pixels"] < 8:
        reasons.append("INTERFERER_NOT_READABLE")
    start = {k: v for k, v in snap.facts.values.items()}
    try:
        from cp_disr.platforms.libero import tp_sr_runtime as rt
        routes = route_search(list(snap.template.contracts), start, "p:Inside:target:container")
    except Exception as exc:  # noqa: BLE001
        routes = []
        reasons.append(f"ROUTE_SEARCH_ERROR:{type(exc).__name__}")
    row["direct_reachable"] = tuple(DIRECT_SEQ) in routes
    row["relocation_reachable"] = tuple(RELOC_SEQ) in routes
    if not (row["direct_reachable"] and row["relocation_reachable"]):
        reasons.append("ROUTE_NOT_CONTRACT_REACHABLE")
    row["reasons"] = reasons
    row["status"] = "PASS" if not reasons else "FAIL"
    return jsonable(row)


def bundle_required_candidates():
    return (DIRECT, RELOC)


def static_worker(root, config_path, out, shard, gpu=None):
    from cp_disr.runtime import load_runtime
    root, out = Path(root), Path(out)
    pool = rd(out / "pool/pool_configs.json")["configs"]
    mine = _shard_ids(pool, shard)
    done_dir = out / "static/rows"
    done_dir.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load(runtime_manifest(root, out).read_text())
    bundle = load_runtime(manifest)
    rows = []
    try:
        for cfg in mine:
            p = done_dir / f"{cfg['case_id']}.json"
            if p.is_file():  # never re-reset (budget)
                rows.append(rd(p))
                continue
            charge(out, "static_capture_resets", 1, cfg["case_id"])
            try:
                row = static_check_one(bundle, cfg)
            except Exception as exc:  # noqa: BLE001 - failures are results, and the reset stays charged
                row = {"case_id": cfg["case_id"], "occlusion_bin": cfg["occlusion_bin"], "status": "FAIL",
                       "reasons": [f"STATIC_EXCEPTION:{type(exc).__name__}:{str(exc)[:200]}"]}
            _atomic_json(p, row)
            rows.append(row)
    finally:
        try:
            bundle.environment.close()
        except Exception:  # noqa: BLE001
            pass
    return {"shard": shard, "rows": len(rows)}


def _spawn_env(root, gpu):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "CP_DISR_PHYSICAL_GPU_INDEX": str(gpu), "PYTHONPATH": f"{root}/src:{root}"})
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(k, None)
    return env


def static_screen(root, config_path, out, gpus, max_workers, shard=None):
    root, out = Path(root), Path(out)
    if shard:
        return static_worker(root, config_path, out, shard)
    n = max(1, min(int(max_workers), 8))
    gpus = list(gpus) or [0]
    procs = []
    (out / "static/logs").mkdir(parents=True, exist_ok=True)
    for k in range(n):
        env = _spawn_env(root, gpus[k % len(gpus)])
        log = open(out / f"static/logs/shard_{k}.log", "ab")
        cmd = [sys.executable, str(root / "scripts/s4_tp_sr_design.py"), "static-screen", "--root", str(root), "--config", str(config_path),
               "--output", str(out), "--scene-shard", f"{k}/{n}"]
        procs.append((k, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(root), start_new_session=True)))
    faults = []
    for k, p in procs:
        rc = p.wait()
        if rc != 0:
            faults.append({"shard": k, "returncode": rc})
    return merge_static(root, out, faults)


def merge_static(root, out, faults=()):
    out = Path(out)
    pool = rd(out / "pool/pool_configs.json")["configs"]
    rows = []
    for cfg in pool:
        p = out / "static/rows" / f"{cfg['case_id']}.json"
        rows.append(rd(p) if p.is_file() else {"case_id": cfg["case_id"], "occlusion_bin": cfg["occlusion_bin"], "status": "NOT_RUN", "reasons": ["NOT_RUN"]})
    npass = sum(1 for r in rows if r["status"] == "PASS")
    by_bin = {b: sum(1 for r in rows if r["status"] == "PASS" and r["occlusion_bin"] == b) for b in BINS}
    _atomic_json(out / "static/static_results.json", {"rows": rows, "pass": npass, "by_bin_pass": by_bin, "shard_faults": list(faults)})
    write_csv(out / "static/static_screen.csv", [{"case_id": r["case_id"], "occlusion_bin": r["occlusion_bin"], "status": r["status"],
                                                   "legal_candidates": "|".join(r.get("legal_candidates", [])), "target_pixels": r.get("target_pixels", ""),
                                                   "interferer_pixels": r.get("interferer_pixels", ""), "direct_reachable": r.get("direct_reachable", ""),
                                                   "relocation_reachable": r.get("relocation_reachable", ""), "reasons": "|".join(r.get("reasons", []))} for r in rows])
    mark_phase(out, "static-screen", static_pass=npass)
    if npass < 12:
        set_decision(out, task_family_feasibility="NOT_ESTABLISHED", recommended_next_action="REDESIGN_GEOMETRY_OR_CONTROLLER", stop_reason="STOPPED_GENERATOR_INVALID")
    return {"status": "STATIC_DONE", "pass": npass, "by_bin_pass": by_bin, "faults": list(faults), "not_run": sum(1 for r in rows if r["status"] == "NOT_RUN")}


# ------------------------------------------------------------------ E: prepare physical probe
def select_physical(static_rows):
    """Deterministic, prior-blind selection using only contract/geometry/reset QA (never provider, policy or route-outcome results)."""
    passing = [r for r in static_rows if r["status"] == "PASS"]
    key = lambda r: hashlib.sha256(f"{r['case_id']}:{PHYS_SALT}".encode()).hexdigest()
    by_bin = {b: sorted([r for r in passing if r["occlusion_bin"] == b], key=key) for b in BINS}
    chosen, shortfall = [], {}
    for b, q in BIN_QUOTA.items():
        take = by_bin[b][:q]
        chosen += take
        if len(take) < q:
            shortfall[b] = q - len(take)
    if shortfall:
        left = sorted([r for r in passing if r not in chosen], key=key)
        need = sum(shortfall.values())
        chosen += left[:need]
    return [r["case_id"] for r in chosen[:24]], shortfall


def prepare_physical_probe(root, config_path, out):
    root, out = Path(root), Path(out)
    res = rd(out / "static/static_results.json")
    if res["pass"] < 12:
        raise StopRun("STOPPED_GENERATOR_INVALID", "fewer than 12 static PASS scenes")
    if (out / "physical/witnesses/e4_branch_registration.json").is_file():
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "physical registration already exists")
    ids, shortfall = select_physical(res["rows"])
    charge(out, "physical_qualification_scenes", len(ids), "prepare-physical-probe")
    manifest_path = runtime_manifest(root, out)
    head = git(root, "rev-parse", "HEAD")
    bins = {r["case_id"]: r["occlusion_bin"] for r in res["rows"]}
    branches = []
    for case in ids:
        for repeat in (0, 1):
            for route, cand in (("direct", DIRECT), ("relocation", RELOC)):
                ident = f"s4_tp_sr|{case}|{cand}|{repeat}"
                seed = int(hashlib.sha256(f"{case}|{repeat}".encode()).hexdigest()[:8], 16)
                branches.append({"branch_id": hashlib.sha256(ident.encode()).hexdigest()[:16], "attempt_id": hashlib.sha256(ident.encode()).hexdigest()[:16],
                                 "case_id": case, "candidate_id": cand, "route": route, "repeat": repeat, "occlusion_bin": bins[case],
                                 "restore_seed": seed, "selection_hash": hashlib.sha256(ident.encode()).hexdigest(), "manifest_path": str(manifest_path),
                                 "authorized": True, "execute_now": True, "status": "REGISTERED"})
    seeds = {}
    for b in branches:
        seeds.setdefault((b["case_id"], b["repeat"]), set()).add(b["restore_seed"])
    if any(len(v) != 1 for v in seeds.values()):
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "paired seed rule violated")
    cap = int(load_config(config_path)["budgets"]["physical_branch_episodes"])
    if len(branches) > cap:
        raise StopRun("STOPPED_BUDGET_EXHAUSTED", "registration exceeds physical cap")
    phys = out / "physical"
    for sub in ("witnesses/branch_receipts", "witnesses/restore_receipts", "witnesses/initial_state_checks", "branch_results", "worker_logs"):
        (phys / sub).mkdir(parents=True, exist_ok=True)
    reg = {"status": "REGISTERED", "physical_witness_episodes_cap": cap, "selection_rule": f"per-bin quota {BIN_QUOTA}, order sha256(scene_id:{PHYS_SALT}); shortfall filled from remaining PASS scenes in the same order",
           "shortfall": shortfall, "scenes": ids, "branches": branches}
    _atomic_json(phys / "witnesses/e4_branch_registration.json", reg)
    (phys / "witnesses/e4_branch_registration.json").chmod(0o444)
    pledger = {"physical_witness_episodes": {"cap": cap, "used": 0}}
    _atomic_json(phys / "budget_ledger.json", pledger)
    (phys / "budget_events.jsonl").touch()
    _atomic_json(phys / "attempt_registry.json", {})
    write_csv(out / "physical/physical_scene_selection.csv", [{"case_id": c, "occlusion_bin": bins[c], "selection_hash": hashlib.sha256(f"{c}:{PHYS_SALT}".encode()).hexdigest()} for c in ids])
    mark_phase(out, "prepare-physical-probe", physical_scenes=len(ids), execution_head=head)
    return {"status": "PREPARED", "scenes": len(ids), "branches": len(branches), "shortfall": shortfall}


# ------------------------------------------------------------------ F: physical probe
def physical_worker(root, out, branch_id):
    from cp_disr.analysis.s1_e4_recovery import ProbedBundle
    from cp_disr.analysis.s1_revision_resume import execute_registered_branch
    from cp_disr.runtime import load_runtime
    root, out = Path(root), Path(out)
    phys = out / "physical"

    def factory(manifest, branch):
        return ProbedBundle(load_runtime(manifest), phys, branch)

    t0 = time.time()
    res = jsonable(execute_registered_branch(root, branch_id, phys, bundle_factory=factory))
    res.update({"worker_pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "worker_wall_seconds": time.time() - t0})
    _atomic_json(phys / "branch_results" / f"{branch_id}.json", res)
    return res


def worker_dry_run(root, out, branch_id):
    """Zero-cost: hash gate + grounding + registration lookup. Reserves nothing and creates no environment."""
    from cp_disr.runtime import require_runtime
    root, out = Path(root), Path(out)
    reg = rd(out / "physical/witnesses/e4_branch_registration.json")
    branch = next((b for b in reg["branches"] if b["branch_id"] == branch_id), None)
    if branch is None:
        raise StopRun("STOPPED_RUNTIME_BINDING", "branch not registered")
    manifest = yaml.safe_load(Path(branch["manifest_path"]).read_text())
    spec = require_runtime(manifest)
    import importlib
    mod = importlib.import_module(spec["module"])
    if Path(mod.__file__).resolve() != Path(spec["source_path"]).resolve() or not callable(getattr(mod, spec["factory"], None)):
        raise StopRun("STOPPED_RUNTIME_BINDING", "module/source mismatch")
    rtm = manifest["runtime"]
    mod.ground_task_contracts(Path(rtm["repository_path"]) / rtm["stage_2a_contract_path"], rtm["skill_timeouts"][TASK_ID])
    split = rd(Path(rtm["repository_path"]) / rtm["task_splits"][TASK_ID])
    if branch["case_id"] not in {r["case_id"] for r in split["dev"]}:
        raise StopRun("STOPPED_RUNTIME_BINDING", "case missing from split")
    return {"dry_run": "OK", "branch_id": branch_id}


def _spawn_worker(root, out, branch_id, gpu):
    phys = Path(out) / "physical"
    log = open(phys / "worker_logs" / f"{branch_id}.log", "ab")
    cmd = [sys.executable, str(Path(root) / "scripts/s4_tp_sr_design.py"), "physical-worker", "--root", str(root), "--output", str(out), "--branch-id", branch_id]
    return subprocess.Popen(cmd, env=_spawn_env(root, gpu), stdout=log, stderr=subprocess.STDOUT, cwd=str(root), start_new_session=True)


def _branch_fault(out, bid):
    p = Path(out) / "physical/branch_results" / f"{bid}.json"
    return rd(p) if p.is_file() else {}


def run_batch(root, out, branch_ids, gpus, workers):
    """Run branches with `workers` concurrent slots (slot i -> gpus[i]). Returns wall span, per-branch walls and faults."""
    phys = Path(out) / "physical"
    slots = list(gpus)[:workers]
    todo, running, faults, walls = list(branch_ids), {}, [], {}
    t_first, t_last = None, None
    event(out, "batch_start", workers=len(slots), gpus=slots, branches=len(todo))
    consecutive_exc = 0
    while todo or running:
        for gpu in slots:
            if gpu in running or not todo or faults:
                continue
            bid = todo.pop(0)
            try:
                reserve_branch_attempt(phys, bid)
            except IntegrationError as exc:
                faults.append({"branch_id": bid, "fault": f"RESERVE:{exc}"})
                continue
            proc = _spawn_worker(root, out, bid, gpu)
            t0 = time.monotonic()
            t_first = t0 if t_first is None else t_first
            running[gpu] = (bid, proc, t0)
            event(out, "branch_launch", branch_id=bid, gpu=gpu, pid=proc.pid, slots_in_use=len(running))
        for gpu, (bid, proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            state = _load_attempts(phys).get(bid)
            res = _branch_fault(out, bid)
            if rc != 0 or not res:
                faults.append({"branch_id": bid, "fault": f"WORKER_EXIT:{rc}", "attempt_state": state})
                if state in ("RESERVED", "STARTED"):
                    finish_branch_attempt(phys, bid, "UNKNOWN", error_phase="worker_process_exit", exit_code=rc)
            elif res.get("execution_status") == "EXCEPTION":
                consecutive_exc += 1
                if consecutive_exc >= 2:
                    faults.append({"branch_id": bid, "fault": "CONSECUTIVE_EXCEPTIONS:" + str(res.get("termination_reason"))})
            else:
                consecutive_exc = 0
            wall = time.monotonic() - t0
            walls[bid] = wall
            t_last = time.monotonic()
            event(out, "branch_finish", branch_id=bid, gpu=gpu, exit_code=rc, wall_seconds=wall, slots_in_use=len(running) - 1)
            del running[gpu]
        time.sleep(1.0)
    sync_physical(out)
    span = (t_last - t_first) if t_first is not None and t_last is not None else None
    return {"workers": len(slots), "n": len(walls), "span": span, "aggregate": (len(walls) / span) if span else None,
            "median_wall": float(np.median(list(walls.values()))) if walls else None, "min_wall": min(walls.values()) if walls else None, "faults": faults, "walls": walls}


def order_branches(reg):
    """Deterministic interleave so a measurement batch mixes scenes, bins and both routes (no scene-outcome dependence)."""
    return [b["branch_id"] for b in sorted(reg["branches"], key=lambda b: b["selection_hash"])]


def decide_workers(stats2, stats4):
    """Pure decision rule (tested): returns workers to keep."""
    single_best = (1.0 / stats2["min_wall"]) if stats2.get("min_wall") else None
    if single_best and stats2.get("aggregate") is not None and stats2["aggregate"] <= 1.05 * single_best:
        return 1, "TWO_WORKER_AGGREGATE_LE_1.05X_BEST_SINGLE"
    if stats4 is None:
        return 2, "FOUR_NOT_TRIED"
    ok = (stats4["aggregate"] >= 1.15 * stats2["aggregate"] and stats4["median_wall"] <= 1.5 * stats2["median_wall"] and not stats4["faults"])
    return (4, "FOUR_KEPT") if ok else (2, "FOUR_REJECTED_THROUGHPUT_OR_CONTENTION")


def run_physical_probe(root, config_path, out, gpus, max_workers):
    root, out = Path(root), Path(out)
    phys = out / "physical"
    reg = rd(phys / "witnesses/e4_branch_registration.json")
    attempts = _load_attempts(phys)
    todo = [b for b in order_branches(reg) if attempts.get(b) is None]
    if not todo:
        return {"status": "NOTHING_TO_RUN"}
    gpus = list(gpus)
    try:
        worker_dry_run(root, out, todo[0])
    except Exception as exc:  # noqa: BLE001
        event(out, "dry_run_failed_before_any_reservation", error=str(exc))
        raise StopRun("STOPPED_RUNTIME_BINDING", str(exc)) from exc
    cap = min(int(max_workers), 4, len(gpus))
    report = {"batches": [], "single_worker_baseline": "NOT_MEASURED_DIRECTLY", "note": "no extra episodes spent on a solo baseline; best-single estimate = 1/min branch wall"}
    stats2 = stats4 = None
    workers = min(2, cap)
    b1 = todo[:8]
    stats2 = run_batch(root, out, b1, gpus, workers)
    report["batches"].append({k: v for k, v in stats2.items() if k != "walls"})
    todo = todo[len(b1):]
    if stats2["faults"]:
        report["faults"] = stats2["faults"]
    elif todo:
        if cap >= 4 and workers == 2:
            keep, why = decide_workers(stats2, None)
            if keep == 2:
                b2 = todo[:16]
                stats4 = run_batch(root, out, b2, gpus, 4)
                report["batches"].append({k: v for k, v in stats4.items() if k != "walls"})
                todo = todo[len(b2):]
        keep, why = decide_workers(stats2, stats4)
        keep = min(keep, cap)
        report["decision"] = {"workers_kept": keep, "reason": why}
        if todo and not (stats4 and stats4["faults"]):
            rest = run_batch(root, out, todo, gpus, keep)
            report["batches"].append({k: v for k, v in rest.items() if k != "walls"})
            if rest["faults"]:
                report["faults"] = rest["faults"]
    _atomic_json(phys / "throughput_report.json", report)
    mark_phase(out, "run-physical-probe")
    return report


# ------------------------------------------------------------------ G: opportunity
def _rate(res):
    return sum(1 for r in res if r["success"]), len(res)


def branch_row(out, b, attempts):
    r = _branch_fault(out, b["branch_id"])
    trace = r.get("trace", [])
    state = attempts.get(b["branch_id"], "NOT_STARTED")
    eng = state in ("FAILED", "UNKNOWN", "NOT_STARTED", "RESERVED", "STARTED") or not r or r.get("execution_status") in ENGINEERING_STATUS or not r.get("protocol_complete")
    ctrl_fail = sum(1 for t in trace if t.get("controller_exit") not in NORMAL_EXITS)
    return {"branch_id": b["branch_id"], "case_id": b["case_id"], "route": b["route"], "repeat": b["repeat"], "occlusion_bin": b.get("occlusion_bin"),
            "attempt_state": state, "execution_status": r.get("execution_status", ""), "termination_reason": r.get("termination_reason", ""),
            "success": bool(r.get("task_success")) and not eng, "engineering_failure": bool(eng), "controller_failures": ctrl_fail,
            "sim_time": (trace[-1]["elapsed_seconds"] if trace else None), "actions": "|".join(t["candidate_id"] for t in trace)}


def classify_scene(direct, reloc):
    """direct/reloc: 2 branch rows each. Pure function (tested)."""
    d_ok, n_d = _rate(direct)
    r_ok, n_r = _rate(reloc)
    if any(x["engineering_failure"] for x in direct + reloc) or n_d != 2 or n_r != 2:
        return "UNKNOWN", "ENGINEERING_FAILURE_OR_INCOMPLETE"
    dt = [x["sim_time"] for x in direct if x["success"]]
    rt_ = [x["sim_time"] for x in reloc if x["success"]]
    if r_ok == 2 and d_ok <= 1:
        return "STRONG_HELPFUL", "relocation 2/2, direct <=1/2"
    if d_ok == 2 and r_ok <= 1:
        return "HARMFUL", "direct 2/2, relocation <=1/2"
    if d_ok == 2 and r_ok == 2:
        md, mr = float(np.mean(dt)), float(np.mean(rt_))
        if mr <= 0.85 * md:
            return "COST_HELPFUL", f"reloc time {mr:.2f} <= 0.85 x {md:.2f}"
        if mr >= 1.5 * md:
            return "HARMFUL", f"reloc time {mr:.2f} >= 1.5 x {md:.2f}"
        cf_d, cf_r = sum(x["controller_failures"] for x in direct), sum(x["controller_failures"] for x in reloc)
        if cf_d != cf_r:
            return "UNKNOWN", "controller failure difference"
        if abs(mr - md) / md < 0.20:
            return "NEUTRAL", f"|delta time| {abs(mr - md) / md:.2%} < 20%"
    return "UNKNOWN", "no rule matched"


def physical_gate(scene_rows, branch_rows, static_rows):
    helpful = [s for s in scene_rows if s["opportunity"] in ("STRONG_HELPFUL", "COST_HELPFUL")]
    nh = [s for s in scene_rows if s["opportunity"] in ("NEUTRAL", "HARMFUL")]
    reach = {r["case_id"] for r in static_rows if r.get("direct_reachable") and r.get("relocation_reachable")}
    scenes_reach = [s for s in scene_rows if s["case_id"] in reach]
    bins = sorted({s["occlusion_bin"] for s in helpful})
    eng = sum(1 for b in branch_rows if b["engineering_failure"]) / max(1, len(branch_rows))
    checks = {"helpful_ge_8": len(helpful) >= 8, "neutral_or_harmful_ge_4": len(nh) >= 4, "both_reachable_ge_12": len(scenes_reach) >= 12,
              "helpful_bins_ge_2": len(bins) >= 2, "engineering_failure_le_20pct": eng <= 0.20}
    return {"checks": checks, "helpful": len(helpful), "neutral_or_harmful": len(nh), "reachable_scenes": len(scenes_reach), "helpful_bins": bins,
            "engineering_failure_rate": eng, "passed": all(checks.values())}


def classify_opportunity(root, config_path, out):
    from cp_disr.analysis.s1_e4_recovery import paired_restore
    out = Path(out)
    phys = out / "physical"
    reg = rd(phys / "witnesses/e4_branch_registration.json")
    attempts = _load_attempts(phys)
    rows = [branch_row(out, b, attempts) for b in reg["branches"]]
    write_csv(out / "physical/branch_results.csv", rows)
    static_rows = rd(out / "static/static_results.json")["rows"]
    scene_rows = []
    for case in reg["scenes"]:
        d = [r for r in rows if r["case_id"] == case and r["route"] == "direct"]
        rl = [r for r in rows if r["case_id"] == case and r["route"] == "relocation"]
        label, why = classify_scene(d, rl)
        prs = [paired_restore(phys, case, rep, reg)["status"] for rep in (0, 1)]
        scene_rows.append({"case_id": case, "occlusion_bin": d[0]["occlusion_bin"] if d else "", "opportunity": label, "reason": why,
                           "direct_success": f"{_rate(d)[0]}/{len(d)}", "reloc_success": f"{_rate(rl)[0]}/{len(rl)}",
                           "direct_mean_time": float(np.mean([x["sim_time"] for x in d if x["sim_time"] is not None])) if any(x["sim_time"] is not None for x in d) else "",
                           "reloc_mean_time": float(np.mean([x["sim_time"] for x in rl if x["sim_time"] is not None])) if any(x["sim_time"] is not None for x in rl) else "",
                           "paired_restore": "|".join(prs)})
    write_csv(out / "opportunity/scene_opportunity.csv", scene_rows)
    gate = physical_gate(scene_rows, rows, static_rows)
    counts = {k: sum(1 for s in scene_rows if s["opportunity"] == k) for k in ("STRONG_HELPFUL", "COST_HELPFUL", "NEUTRAL", "HARMFUL", "UNKNOWN")}
    _atomic_json(out / "opportunity/physical_gate.json", {**gate, "counts": counts})
    mark_phase(out, "classify-opportunity", physical_gate_passed=gate["passed"])
    if not gate["passed"]:
        set_decision(out, status="COMPLETE", task_family_feasibility="NOT_ESTABLISHED", recommended_next_action="REDESIGN_GEOMETRY_OR_CONTROLLER",
                     stop_reason="PHYSICAL_GATE_FAILED", method_upgrade_authorized=False, s2_authorized=False)
    return {"status": "CLASSIFIED", "counts": counts, "gate": gate}
