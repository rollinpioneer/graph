#!/usr/bin/env python
"""CP-DISR-TB-INDEP-HOLDOUT-01 runner: frozen T_B prematerialized independent holdout (label PREMATERIALIZED_INDEPENDENT_HOLDOUT).

Subcommands
  prerelease : every hard check of section 3, builds the no-prior test split and the release manifest; builds NO environment, runs NO episode.
  worker     : one execution class on one GPU; model x case planned slots in the frozen order, one deterministic pass each.
  collect    : raw evidence -> accounting / results / summary (derived from rows only).
  verify     : final integrity checks.

This runner never trains, creates no optimizer, calls no provider, opens no test VLM cache / relation truth / old test result,
and never retries.  The recorder (src/cp_disr/final_tb_eval_record.py) is used byte-identically to the verified commit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
V2_1 = Path("/home/xushijie2/graph_cp_disr_v2_1")
CARD = "CP-DISR-TB-INDEP-HOLDOUT-01"
LABEL = "PREMATERIALIZED_INDEPENDENT_HOLDOUT"
RELEASE_BRANCH = "codex/cp-disr-tb-indep-holdout"
BASELINE = "241159c1ceca3b53f0ba8423879524c6b2668e07"
PREP_HEAD = "d64d59c65daac61bc1633276b3e1b4439068c259"
RECORDER_COMMIT = "f297f538f9f813ed7a8df39cb710b00b50068406"
PREP_DIR = ROOT / "runs/final_master/2.1.1/launch/20261002T150615Z_tb_eval_prep_241159c1"
E1_LAUNCH = ROOT / "runs/final_master/2.1.1/launch/20261002T031938Z_e1_elastic01_e4dc34ae"
RECORDER_REL = "src/cp_disr/final_tb_eval_record.py"
CURRENT, ARCHIVED = "CURRENT_2_1_1_EXECUTION_CLASS", "ARCHIVED_V13_EXECUTION_CLASS"
CLASS_ORDER = {CURRENT: ["TB-M3", "TB-M5", "TB-M6", "TB-C1"], ARCHIVED: ["TB-M2", "TB-M4", "TB-M1"]}
TIER_OF = {"TB-M3": "CORE", "TB-M5": "CORE", "TB-M6": "CORE", "TB-C1": "CORE", "TB-M2": "HISTORICAL_EXTENSION", "TB-M4": "HISTORICAL_EXTENSION", "TB-M1": "SYSTEM_REFERENCE"}
PLANNED = 210
FORBIDDEN_OPEN = (r"(^|/)vlm_cache(/|$)", r"(^|/)relation_truth", r"(^|/)accepted_relations", r"(^|/)final_edges\.json", r"(^|/)test_metrics", r"(^|/)final_test_summary",
                  r"(^|/)selection_manifest", r"stage_2a_inputs/T_B/test", r"T_B_stage_2a_v11\.json")
CACHE_KEYS = ("cache_dir", "cache_key", "cache_status")


# ----------------------------------------------------------------------------- helpers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a, cwd=ROOT):
    return subprocess.check_output(["git", *a], cwd=str(cwd), text=True).strip()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p, d):
    Path(p).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_jsonl(path, obj):
    """Durable append: flush + fsync so a crash keeps every completed record."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def fsync_probe(directory):
    """Output dir must be creatable, writable, and survive write + fsync + rename + read-back."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    tmp, dst = d / ".probe.tmp", d / ".probe"
    with open(tmp, "w") as f:
        f.write("probe-" + now())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    ok = dst.read_text().startswith("probe-")
    dst.unlink()
    return ok


def derive_no_prior_test_split(derived_dev_split, test30_active):
    """Pure transform: no-prior train/dev rows (cache references removed) + the 30 test30 rows.  Raises if any cache pointer would survive."""
    for r in test30_active:  # test rows must ARRIVE without any cache pointer; they are never silently cleaned
        if any(k in r for k in CACHE_KEYS):
            raise ValueError("test row carries a cache pointer: %s" % r.get("case_id", "?"))

    def clean(row):
        r = {k: v for k, v in row.items() if k not in CACHE_KEYS and k != "source_cache_ref_audit_only"}
        return r
    split = {
        "task_id": "T_B", "version": "tb-indep-holdout-no-prior-v1", "second_role": "second_object",
        "note": "R = absent for every evaluated model. train/dev rows exist only to bootstrap the first environment; test rows come from T_B_stage_2a_test30.json active (no cache pointer keys).",
        "train": [clean(r) for r in derived_dev_split["train"]], "dev": [clean(r) for r in derived_dev_split["dev"]],
        "test": [clean(r) for r in test30_active], "test_isolated": True,
    }
    text = json.dumps(split)
    for needle in CACHE_KEYS + ("vlm_cache", "source_cache_ref"):
        if needle in text:
            raise ValueError("derived split would carry a cache reference: " + needle)
    for part in ("train", "dev", "test"):
        for r in split[part]:
            if any(k in r for k in CACHE_KEYS):
                raise ValueError("cache pointer in row " + r.get("case_id", "?"))
    return split


def pick_gpus(n=2):
    """Two distinct GPUs with the least memory in use / utilisation (shared host); recorded, not negotiated later."""
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    info = []
    for r in rows:
        i, m, u = [int(x) for x in r.split(",")]
        info.append((m + 200 * u, i, m, u))
    info.sort()
    return [{"index": i, "memory_used_mib": m, "utilization_pct": u} for _, i, m, u in info[:n]]


# ----------------------------------------------------------------------------- prerelease
def cmd_prerelease(a):
    out = Path(a.out).resolve()
    checks, detail = {}, {}

    def chk(name, ok, info=None):
        checks[name] = bool(ok)
        if info is not None:
            detail[name] = info

    cand = load(PREP_DIR / "candidate_holdout_release_manifest.json")
    freeze = load(PREP_DIR / "test_source_freeze.json")
    srcid = load(PREP_DIR / "evaluation_module_source_identity.json")
    prep_verify = load(PREP_DIR / "verify.json")
    matrix = load(PREP_DIR / "historical_model_load_matrix.json")
    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    chk("branch_is_release_branch", branch == RELEASE_BRANCH, branch)
    chk("prep_head_is_ancestor", subprocess.run(["git", "merge-base", "--is-ancestor", PREP_HEAD, "HEAD"], cwd=str(ROOT)).returncode == 0)
    chk("recorder_commit_is_ancestor", subprocess.run(["git", "merge-base", "--is-ancestor", RECORDER_COMMIT, "HEAD"], cwd=str(ROOT)).returncode == 0)
    chk("tracked_tree_clean", dirty == "", dirty)
    changed = [l for l in git("diff", "--name-only", PREP_HEAD, "HEAD").splitlines()]
    chk("changes_since_prep_head_are_only_runner_files", all(c.startswith(("scripts/final_tb_holdout_runner", "tests/test_final_tb_holdout_runner")) for c in changed), changed)
    changed_src = git("diff", "--name-only", PREP_HEAD, "HEAD", "--", "src", "configs", "experiments").splitlines()
    chk("no_src_configs_experiments_change_since_prep", changed_src == [], changed_src)
    rec_v = git("rev-parse", "%s:%s" % (RECORDER_COMMIT, RECORDER_REL))
    chk("recorder_blob_equals_verified_f297f538", rec_v == git("hash-object", RECORDER_REL) == git("rev-parse", "HEAD:" + RECORDER_REL), rec_v)
    prot = {r: git("rev-parse", "%s:%s" % (BASELINE, r)) == git("hash-object", r) == v["baseline_blob"] for r, v in srcid["protected_production_files"].items()}
    chk("protected_production_files_match_prep_record", all(prot.values()) and prep_verify["checks"]["all_checks_pass"] and len(prot) == 15, {"n": len(prot), "bad": [k for k, v in prot.items() if not v]})
    models = cand["models"]
    chk("seven_models_in_candidate_manifest", sorted(m["slot"] for m in models) == sorted(TIER_OF), [m["slot"] for m in models])
    chk("inclusion_matches_frozen_prep", all(TIER_OF[m["slot"]] == m["tier"] for m in models) and cand["episode_budget"]["maximum_if_all_included"] == PLANNED)
    chk("classes_match_card", all(m["execution_class"] == (CURRENT if m["slot"] in CLASS_ORDER[CURRENT] else ARCHIVED) for m in models))
    exist, shaok = {}, {}
    for m in models:
        p = Path(m["final_checkpoint"]["path"])
        exist[m["slot"]] = p.is_file()
        shaok[m["slot"]] = p.is_file() and sha256_file(p) == m["final_checkpoint"]["sha256"] and p.stat().st_size == m["final_checkpoint"]["bytes"]
    chk("seven_final_checkpoints_exist", all(exist.values()), exist)
    chk("seven_sha256_match_candidate_manifest", all(shaok.values()), shaok)
    chk("sidecar_json_exists", all(Path(m["final_checkpoint"]["path"]).with_suffix(".json").is_file() for m in models))
    # execution-class source identity
    cur_hash_ok = all(sha256_file(ROOT / rel) == cand["execution_classes"][CURRENT]["execution_hashes"][k] for k, rel in
                      {"stage2a_v11": "src/cp_disr/stage2a_v11.py", "collector": "src/cp_disr/collector.py", "neural": "src/cp_disr/neural.py", "torch_rl": "src/cp_disr/torch_rl.py",
                       "persistence": "src/cp_disr/persistence.py", "runtime_factory": "src/cp_disr/platforms/libero/runtime_factory.py", "clock": "src/cp_disr/platforms/libero/clock.py",
                       "safety": "src/cp_disr/platforms/libero/safety.py", "skill_executor": "src/cp_disr/platforms/libero/skill_executor.py"}.items())
    chk("current_class_execution_hashes_match_manifest", cur_hash_ok)
    rec_arch = cand["execution_classes"][ARCHIVED]["execution_hashes_recorded_in_run_manifests"]
    arch_map = {"stage2a_v11": "src/cp_disr/stage2a_v11.py", "collector": "src/cp_disr/collector.py", "neural": "src/cp_disr/neural.py", "torch_rl": "src/cp_disr/torch_rl.py",
                "persistence": "src/cp_disr/persistence.py", "runtime_factory": "src/cp_disr/platforms/libero/runtime_factory.py"}
    arch_bad = [k for k, rel in arch_map.items() if sha256_file(V2_1 / rel) != rec_arch.get(k)]
    chk("archived_class_source_hashes_match_run_manifests", arch_bad == [], arch_bad)
    # strict-load checks (zero environment): current 4 under the current tree, archived 3 under the archived tree
    py = sys.executable
    models_in = load(PREP_DIR / "phase_a/frozen_models_input.json")
    sub_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE"):
        sub_env.pop(k, None)
    loads = {}
    for cls, tree in ((CURRENT, ROOT), (ARCHIVED, V2_1)):
        o = out / ("strict_load_%s.json" % ("current" if cls == CURRENT else "archived"))
        out.mkdir(parents=True, exist_ok=True)
        env = dict(sub_env, PYTHONPATH="%s:%s" % (tree / "src", tree))
        r = subprocess.run([py, str(ROOT / "scripts/final_tb_eval_prep_load_check.py"), "--models", str(PREP_DIR / "phase_a/frozen_models_input.json"), "--out", str(o),
                            "--label", "prerelease_" + cls, "--only", *CLASS_ORDER[cls]], cwd=str(ROOT), env=env, capture_output=True, text=True)
        rows = {x["slot"]: x for x in load(o)["rows"]} if o.is_file() else {}
        loads[cls] = {s: bool(rows.get(s, {}).get("strict_load")) and rows.get(s, {}).get("sha256_matches_frozen_list") is True for s in CLASS_ORDER[cls]}
        if r.returncode != 0:
            detail["strict_load_stderr_" + cls] = r.stderr[-600:]
    chk("current_four_strict_load_pass_current_tree", all(loads[CURRENT].values()), loads[CURRENT])
    chk("archived_three_strict_load_pass_archived_tree", all(loads[ARCHIVED].values()), loads[ARCHIVED])
    # test30 identity
    t30p, idsp = ROOT / "configs/splits/T_B_stage_2a_test30.json", ROOT / "configs/splits/T_B_stage_2a_test_ids.json"
    t30 = load(t30p)["active"]
    order = [r["case_id"] for r in t30]
    chk("test30_files_sha_equal_frozen", sha256_file(t30p) == freeze["split_configs_sha256"]["configs/splits/T_B_stage_2a_test30.json"] and sha256_file(idsp) == freeze["split_configs_sha256"]["configs/splits/T_B_stage_2a_test_ids.json"])
    chk("test30_id_order_equals_frozen", order == freeze["active_cases"]["order"] == cand["test_cases"]["order"] and len(order) == 30)
    chk("reset_seed_order_equals_frozen", [r["seed"] for r in t30] == [x["reset_seed"] for x in freeze["active_cases"]["rows"]])
    chk("reset_config_row_hashes_equal_frozen", [canon_sha({k: r[k] for k in sorted(r)}) for r in t30] == [x["reset_config_row_sha256"] for x in freeze["active_cases"]["rows"]])
    chk("test30_rows_have_no_cache_pointer_keys", all(not any(k in r for k in CACHE_KEYS) for r in t30))
    # derived no-prior split
    try:
        split = derive_no_prior_test_split(load(E1_LAUNCH / "train_split_tb_noprior.json"), t30)
        split_path = out / "no_prior_test_split_T_B.json"
        dump(split_path, split)
        split_sha = sha256_file(split_path)
        chk("derived_no_prior_test_split_built_without_cache_pointers", True, {"path": str(split_path), "sha256": split_sha, "n_test": len(split["test"])})
        chk("derived_split_test_order_equals_frozen", [r["case_id"] for r in split["test"]] == order and [r["seed"] for r in split["test"]] == [r["seed"] for r in t30])
    except Exception as exc:
        split_sha, split_path = None, None
        chk("derived_no_prior_test_split_built_without_cache_pointers", False, str(exc))
    chk("provider_zero_no_api_key_env", not any(os.environ.get(k) for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE")))
    chk("output_path_absolute_and_fsync_ok", out.is_absolute() and fsync_probe(out), str(out))
    gpus = pick_gpus(2)
    chk("two_distinct_gpus_selected", len({g["index"] for g in gpus}) == 2, gpus)
    du = os.statvfs(str(out))
    chk("disk_free_over_20GiB", du.f_bavail * du.f_frsize > 20 * 1024 ** 3, du.f_bavail * du.f_frsize)
    ok = all(checks.values())
    dump(out / "prerelease_checks.json", {"card": CARD, "created": now(), "all_pass": ok, "checks": checks, "detail": detail, "head": head,
                                          "test_episodes_run": 0, "environments_constructed": 0})
    print(json.dumps({"all_pass": ok, "failed": [k for k, v in checks.items() if not v]}, indent=1))
    if not ok:
        print("PRERELEASE FAILED: 0 test episode, stop.")
        return 2
    cand_sha = sha256_file(PREP_DIR / "candidate_holdout_release_manifest.json")
    classes = {CURRENT: {"source_tree": str(ROOT), "recorder_loaded_by_file_path": str(ROOT / RECORDER_REL),
                         "runtime_manifest": cand["execution_classes"][CURRENT]["runtime_manifest"], "runtime_manifest_sha256": cand["execution_classes"][CURRENT]["runtime_manifest_sha256"],
                         "binding": cand["execution_classes"][CURRENT]["binding"], "models": CLASS_ORDER[CURRENT], "worker": "A", "gpu": gpus[0]},
               ARCHIVED: {"source_tree": str(V2_1), "v2_1_head": cand["execution_classes"][ARCHIVED]["v2_1_head"], "recorder_loaded_by_file_path": str(ROOT / RECORDER_REL),
                          "execution_hashes_recorded_in_run_manifests": rec_arch, "binding": cand["execution_classes"][ARCHIVED]["binding"], "models": CLASS_ORDER[ARCHIVED], "worker": "B",
                          "gpu": gpus[1], "result_label": "ARCHIVED_EXECUTION_CLASS"}}
    manifest = {
        "document": "test_release_manifest", "card": CARD, "created": now(), "task": "T_B", "test_label": LABEL, "not_strict_blind_test": True,
        "scope_statement": "GLOBAL_S4 = NOT_COMPLETE; GLOBAL_S5 = NOT_COMPLETE; this is not GLOBAL S6",
        "bindings": {"release_branch": RELEASE_BRANCH, "release_head": head, "prep_evidence_head": PREP_HEAD, "verified_recorder_source_commit": RECORDER_COMMIT,
                     "baseline": BASELINE, "note": "verify.json of the prep branch records head f297f538 (recorder / offline-test identity); d64d59c65 is the later branch HEAD with Phase C / release evidence; files are kept as written, this release binds both",
                     "candidate_manifest_sha256": cand_sha, "recorder_blob": rec_v, "no_prior_test_split_sha256": split_sha, "no_prior_test_split": str(split_path)},
        "authorization": {"formal_holdout_authorized_by": CARD, "planned_episodes": PLANNED, "inclusion_frozen_before_any_test_score_by": "CP-DISR-TB-EVAL-PREP-LITE-01",
                          "inclusion_not_redecided_here": True},
        "tiers": {"CORE": {"models": 4, "episodes": 120}, "HISTORICAL_EXTENSION": {"models": 2, "episodes": 60}, "SYSTEM_REFERENCE": {"models": 1, "episodes": 30}},
        "models": [dict(m, tier=TIER_OF[m["slot"]]) for m in models], "evaluation_model_rule": "bound final checkpoint + SHA256 only; selected/best-dev never substitutes final",
        "execution_classes": classes, "test_cases": {"label": LABEL, "n": 30, "order": order, "reset_seeds": [r["seed"] for r in t30], "source": "configs/splits/T_B_stage_2a_test30.json",
                                                     "scene_set_sha256": freeze["scene_set"]["scene_set_sha256"]},
        "evaluator_settings": cand["evaluator_settings"], "invalid_episode_rule": cand["invalid_episode_rule"],
        "planned_slots": {"total": PLANNED, "per_model": 30, "slot_id": "<model slot>|<case id>", "statuses": ["MEASURED", "TECHNICAL_NOT_MEASURED", "NOT_EXECUTED_RELEASE_STOPPED"]},
        "parallel": {"max_workers": 2, "workers": [{"id": "A", "class": CURRENT, "gpu": gpus[0], "out": str(out / "worker_current")}, {"id": "B", "class": ARCHIVED, "gpu": gpus[1], "out": str(out / "worker_archived")}],
                     "note": "parallelism shortens wall time only; models, cases, seeds and order are unchanged; a technical error stops the whole release (no automatic resource change / resume)"},
        "visibility_rule": "no model / checkpoint / protocol / seed / control decision before all 210 planned slots are complete; health monitoring only",
        "interpretation_boundary_C1": "final holdout performance is not the development learning process (0/4096/8192/final); it cannot by itself show that B2 early-learning advantage generalizes; development results are not deleted or replaced by holdout results; checkpoints are never swapped after test",
        "forbidden": ["training", "optimizer", "provider", "old test VLM cache", "relation truth", "tuning", "checkpoint change after test", "model add/remove after test", "automatic retry"],
        "outputs": ["test_release_manifest.json", "holdout_accounting.json", "core_results.json", "historical_results.json", "system_reference_results.json", "per_episode_eval_row_v2.jsonl",
                    "model_hash_receipts.json", "technical_events.jsonl", "holdout_summary.md", "verify.json"],
    }
    dump(out / "test_release_manifest.json", manifest)
    lock = {"manifest_sha256": sha256_file(out / "test_release_manifest.json"), "runner_sha256": sha256_file(ROOT / "scripts/final_tb_holdout_runner.py"),
            "recorder_sha256": sha256_file(ROOT / RECORDER_REL), "split_sha256": split_sha, "release_head": head, "created": now()}
    dump(out / "release_lock.json", lock)
    print("PRERELEASE OK; manifest + lock written; gpus:", gpus)
    return 0


# ----------------------------------------------------------------------------- worker
class ReleaseStopRequested(BaseException):
    """BaseException on purpose: the recorder's `except Exception` must not turn a stop request into an infrastructure failure of a slot."""


def load_recorder(path):
    spec = importlib.util.spec_from_file_location("final_tb_eval_record_by_path", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def is_forbidden_path(s):
    """Data paths only: python source files (e.g. src/cp_disr/vlm_cache_pipeline.py) are code, never a test cache."""
    if s.endswith((".py", ".pyc", ".so")):
        return False
    return any(re.search(n, s) for n in FORBIDDEN_OPEN)


def install_open_guard():
    """Hard guard: any attempt to open a test VLM cache / relation truth / old test result / v11 split (test cache rows) is refused."""
    def hook(event, args):
        if event == "open" and args:
            p = args[0]
            if isinstance(p, (str, bytes, os.PathLike)):
                s = os.fsdecode(p)
                if is_forbidden_path(s):
                    raise PermissionError("holdout open-guard refused: " + s)
    sys.addaudithook(hook)


def build_plan(release_dir, cls):
    man = load(Path(release_dir) / "test_release_manifest.json")
    c = man["execution_classes"][cls]
    models = [m for m in man["models"] if m["slot"] in c["models"]]
    models.sort(key=lambda m: c["models"].index(m["slot"]))
    return {"class": cls, "worker": c["worker"], "root": c["source_tree"], "gpu": c["gpu"]["index"], "models": models, "cases": man["test_cases"]["order"],
            "split": man["bindings"]["no_prior_test_split"], "split_sha256": man["bindings"]["no_prior_test_split_sha256"],
            "runtime_manifest": c.get("runtime_manifest"), "runtime_manifest_sha256": c.get("runtime_manifest_sha256")}


def verify_release_lock(release_dir):
    lock = load(Path(release_dir) / "release_lock.json")
    bad = []
    if sha256_file(Path(release_dir) / "test_release_manifest.json") != lock["manifest_sha256"]:
        bad.append("manifest_sha256")
    if sha256_file(ROOT / "scripts/final_tb_holdout_runner.py") != lock["runner_sha256"]:
        bad.append("runner_sha256")
    if sha256_file(ROOT / RECORDER_REL) != lock["recorder_sha256"]:
        bad.append("recorder_sha256")
    if git("hash-object", RECORDER_REL) != git("rev-parse", "%s:%s" % (RECORDER_COMMIT, RECORDER_REL)):
        bad.append("recorder_blob")
    man = load(Path(release_dir) / "test_release_manifest.json")
    if sha256_file(man["bindings"]["no_prior_test_split"]) != lock["split_sha256"]:
        bad.append("split_sha256")
    if git("rev-parse", "HEAD") != lock["release_head"]:
        bad.append("release_head")
    if git("status", "--porcelain", "--untracked-files=no") != "":
        bad.append("tracked_tree_dirty")
    return bad


def run_plan(plan, out, stop_flag, release_dir=None, smoke=False):
    """Execute the planned slots of one class, in the frozen order, one deterministic pass each.  Never retries."""
    import threading
    out = Path(out).resolve()
    stop_flag = Path(stop_flag).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert fsync_probe(out), "output dir failed the fsync probe"
    rows_path, events_path = out / "per_episode_eval_row_v2.jsonl", out / "technical_events.jsonl"
    prog_path, state_path, receipts_path = out / "progress.json", out / "worker_state.json", out / "model_hash_receipts.json"
    planned = {m["slot"]: len(plan["cases"]) for m in plan["models"]}
    prog = {"worker": plan["worker"], "class": plan["class"], "state": "STARTING", "pid": os.getpid(), "planned_total": sum(planned.values()), "planned": planned,
            "completed": {m["slot"]: 0 for m in plan["models"]}, "current_slot": None, "current_case_index": None, "heartbeat": now(), "started": now()}
    lock = threading.Lock()
    receipts = {}

    def flush_progress():
        with lock:
            prog["heartbeat"] = now()
            prog["completed_total"] = sum(prog["completed"].values())
            tmp = prog_path.with_name("progress.json.tmp")
            tmp.write_text(json.dumps(prog, indent=1) + "\n")
            os.replace(tmp, prog_path)

    def heartbeat():
        while prog["state"] in ("STARTING", "RUNNING"):
            try:
                flush_progress()
            except Exception:
                pass
            time.sleep(20)

    def set_state(s, **kw):
        with lock:
            prog["state"] = s
            prog.update(kw)
        dump(state_path, {"state": s, "pid": os.getpid(), "time": now(), **kw})
        flush_progress()

    def event(kind, **kw):
        append_jsonl(events_path, {"time": now(), "worker": plan["worker"], "class": plan["class"], "kind": kind, **kw})

    set_state("STARTING")
    os.environ["MUJOCO_GL"] = "egl"
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        os.environ.pop(k, None)
    root = Path(plan["root"]).resolve()
    split_path = Path(plan["split"]).resolve()
    os.chdir(str(root))
    rec = load_recorder(ROOT / RECORDER_REL)
    final_status = "FINISHED_ALL_PLANNED"
    try:
        install_open_guard()
        import torch
        from cp_disr import stage2a_v11 as v11
        assert Path(v11.__file__).resolve().is_relative_to(root), "stage2a_v11 not imported from the class source tree: %s" % v11.__file__
        assert torch.cuda.device_count() == 1, "worker must see exactly one GPU"
        device = torch.device("cuda", 0)
        split_doc = load(split_path)
        assert sha256_file(split_path) == plan["split_sha256"]
        index = {r["case_id"]: r for r in split_doc["train"] + split_doc["dev"] + list(split_doc.get("test") or [])}
        assert all(c in index for c in plan["cases"])
        v11.ENABLED_SPLITS = dict(v11.ENABLED_SPLITS)
        v11.ENABLED_SPLITS["T_B"] = split_path
        gate = rec.no_prior_case_gate
        if plan["class"] == CURRENT:
            rm = str(Path(plan["runtime_manifest"]).resolve())
            assert sha256_file(rm) == plan["runtime_manifest_sha256"]
            v11.bind_run_context(SimpleNamespace(runtime_manifest=rm, train_split=str(split_path), render_gpu_device_id=0, prior_mode="absent"))
            rt_used = rm
        else:
            assert getattr(v11, "RUN_CONTEXT", None) is None and not hasattr(v11, "bind_run_context"), "archived class must be the pre-run-context source"
            rt_used = str(root / v11.RUNTIME_REL)
        files = {"recorder": ROOT / RECORDER_REL, "stage2a_v11": root / "src/cp_disr/stage2a_v11.py", "collector": root / "src/cp_disr/collector.py", "neural": root / "src/cp_disr/neural.py",
                 "torch_rl": root / "src/cp_disr/torch_rl.py", "runtime_factory": root / "src/cp_disr/platforms/libero/runtime_factory.py", "rl": root / "src/cp_disr/rl.py",
                 "skill_executor": root / "src/cp_disr/platforms/libero/skill_executor.py", "split_used": split_path, "runtime_manifest_used": rt_used}
        hashes = rec.source_identity(files)
        orig_record = rec.record_episode
        cur = {"slot": None, "k": 0}

        def wrapped(bundle, collector, snap, case_id, source_n, prior_mode):
            if stop_flag.exists():
                raise ReleaseStopRequested()
            with lock:
                prog["current_case_index"] = cur["k"]
            t0 = time.time()
            row = orig_record(bundle, collector, snap, case_id, source_n, prior_mode)
            append_jsonl(rows_path, {"slot_id": "%s|%s" % (cur["slot"], case_id), "slot": cur["slot"], "case_id": case_id, "case_index": cur["k"], "worker": plan["worker"],
                                     "class": plan["class"], "recorded_at": now(), "wall_seconds": round(time.time() - t0, 2), "row": row})
            cur["k"] += 1
            with lock:
                prog["completed"][cur["slot"]] += 1
            flush_progress()
            return row

        rec.record_episode = wrapped
        set_state("RUNNING")
        threading.Thread(target=heartbeat, daemon=True).start()
        for m in plan["models"]:
            slot = m["slot"]
            if stop_flag.exists():
                final_status = "STOPPED_BY_RELEASE_FLAG"
                event("RELEASE_STOP_OBSERVED", before_slot=slot)
                break
            cur["slot"], cur["k"] = slot, 0
            with lock:
                prog["current_slot"] = slot
            out_eval = out / ("eval_row_v2_%s.json" % slot)
            ck = m["final_checkpoint"]
            receipts[slot] = {"slot": slot, "method": m["method"], "path": ck["path"], "expected_sha256": ck["sha256"], "bytes": ck["bytes"], "execution_class": plan["class"], "status": "STARTED", "started": now()}
            try:
                payload = rec.evaluate_checkpoint_recorded(root, "T_B", m["method"], ck["path"], plan["cases"], index, device, hashes, out_eval, label=LABEL, v11=v11, expected_sha256=ck["sha256"],
                                                           case_gate=gate, extra_identity={"execution_class": plan["class"], "slot": slot, "root": str(root), "release_card": CARD, "smoke": bool(smoke)})
                receipts[slot].update({"status": payload["status"], "sha256_before_first_episode": payload["evaluated_model"]["sha256"], "sha256_after_last_episode": payload.get("model_sha256_after_last_episode"),
                                       "evaluated_generation": payload["evaluated_generation"], "eval_file": str(out_eval), "finished": now()})
            except ReleaseStopRequested:
                receipts[slot].update({"status": "STOPPED_BY_RELEASE_FLAG", "finished": now()})
                event("RELEASE_STOP_OBSERVED", slot=slot, next_case_index=cur["k"])
                final_status = "STOPPED_BY_RELEASE_FLAG"
                break
            except rec.InfrastructureFailure as exc:
                info = {}
                try:
                    info = load(exc.partial_path).get("infrastructure_failure", {}) if getattr(exc, "partial_path", None) else {}
                except Exception:
                    pass
                case = info.get("case_id") or plan["cases"][min(cur["k"], len(plan["cases"]) - 1)]
                event("INFRASTRUCTURE_FAILURE", slot=slot, case_id=case, slot_id="%s|%s" % (slot, case), error=str(exc)[:1500], detail=info)
                receipts[slot].update({"status": "INFRASTRUCTURE_FAILURE_STOPPED", "finished": now()})
                final_status = "INFRASTRUCTURE_FAILURE"
                break
            except rec.ModelIdentityError as exc:
                event("MODEL_HASH_MISMATCH", slot=slot, error=str(exc)[:800], episodes_recorded_for_model=cur["k"])
                receipts[slot].update({"status": "MODEL_HASH_MISMATCH", "finished": now()})
                final_status = "MODEL_HASH_MISMATCH"
                break
            except Exception as exc:  # setup / load / environment construction error before or outside an episode
                case = plan["cases"][min(cur["k"], len(plan["cases"]) - 1)]
                event("INFRASTRUCTURE_FAILURE", slot=slot, case_id=case, slot_id="%s|%s" % (slot, case), error=("%s: %s" % (type(exc).__name__, exc))[:1500], trace=traceback.format_exc()[-1800:])
                receipts[slot].update({"status": "INFRASTRUCTURE_FAILURE_STOPPED", "finished": now()})
                final_status = "INFRASTRUCTURE_FAILURE"
                break
            finally:
                dump(receipts_path, receipts)
            if cur["k"] != len(plan["cases"]):
                event("INTERNAL_COUNT_MISMATCH", slot=slot, recorded=cur["k"], planned=len(plan["cases"]))
                final_status = "INFRASTRUCTURE_FAILURE"
                break
    except BaseException as exc:  # outside any slot: still a stop, never a retry
        final_status = "INFRASTRUCTURE_FAILURE"
        event("INFRASTRUCTURE_FAILURE", slot=None, case_id=None, error=("%s: %s" % (type(exc).__name__, exc))[:1500], trace=traceback.format_exc()[-1800:])
    finally:
        if final_status != "FINISHED_ALL_PLANNED" and not smoke:
            try:
                stop_flag.parent.mkdir(parents=True, exist_ok=True)
                if not stop_flag.exists():
                    stop_flag.write_text(json.dumps({"by": plan["worker"], "reason": final_status, "time": now()}) + "\n")
            except Exception:
                pass
        dump(receipts_path, receipts)
        set_state(final_status)
    return 0 if final_status == "FINISHED_ALL_PLANNED" else 4


def cmd_worker(a):
    rel = Path(a.release).resolve()
    cls = CURRENT if a.cls == "current" else ARCHIVED
    stop_flag = rel / "STOP_RELEASE"
    if stop_flag.exists():
        print("STOP_RELEASE exists: refusing to start (0 episode).")
        return 3
    bad = verify_release_lock(rel)
    if bad:
        print("release lock mismatch %s: refusing to start (0 episode)." % bad)
        return 3
    plan = build_plan(rel, cls)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(plan["gpu"]):
        print("CUDA_VISIBLE_DEVICES must equal the manifest GPU %s: refusing to start (0 episode)." % plan["gpu"])
        return 3
    out = rel / ("worker_current" if cls == CURRENT else "worker_archived")
    if (out / "per_episode_eval_row_v2.jsonl").exists():
        print("worker output already exists: no automatic resume / retry (0 episode).")
        return 3
    return run_plan(plan, out, stop_flag, release_dir=rel)


def cmd_smoke(a):
    """Engineering smoke of the worker mechanics on ONE frozen DEV case (T_B_dev_00) per class; never reads the test split; output is scratch and removed."""
    cls = CURRENT if a.cls == "current" else ARCHIVED
    cand = load(PREP_DIR / "candidate_holdout_release_manifest.json")
    models = {m["slot"]: m for m in cand["models"]}
    slot = "TB-M3" if cls == CURRENT else "TB-M2"
    split = E1_LAUNCH / "train_split_tb_noprior.json"
    c = cand["execution_classes"][cls]
    plan = {"class": cls, "worker": "SMOKE-" + a.cls, "root": c["source_tree"], "gpu": 0, "models": [models[slot]], "cases": ["T_B_dev_00"], "split": str(split), "split_sha256": sha256_file(split),
            "runtime_manifest": c.get("runtime_manifest"), "runtime_manifest_sha256": c.get("runtime_manifest_sha256")}
    out = Path(a.out).resolve()
    rc = run_plan(plan, out, out / "STOP_SMOKE_UNUSED", smoke=True)
    print("smoke rc", rc, "state", load(out / "worker_state.json")["state"])
    return rc


# ----------------------------------------------------------------------------- collect (derived from raw rows only)
FINISHED_STATES = ("FINISHED_ALL_PLANNED",)
WORKER_DIRS = (("A", "worker_current", CURRENT), ("B", "worker_archived", ARCHIVED))
PAIRS = (("TB-M5", "TB-M3", "B2 s1 vs B1-K s1"), ("TB-M5", "TB-M6", "B2 s1 vs B1-K+E s0"), ("TB-M5", "TB-C1", "B2 s1 vs B1-K+E s1"), ("TB-M6", "TB-C1", "B1-K+E s0 vs s1"))
TIER_ORDER = ("CORE", "HISTORICAL_EXTENSION", "SYSTEM_REFERENCE")


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.is_file() else []


def _q(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _dist(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "min": min(xs), "q25": _q(xs, .25), "q75": _q(xs, .75), "max": max(xs)}


def model_stats(slot, meta, rows, n_planned, n_technical):
    succ = [r for r in rows if r["success"]]
    ttf = [r["time_to_first_confirmed_success_s"] for r in succ if r.get("time_to_first_confirmed_success_s") is not None]
    seqs = Counter(tuple(a["candidate_id"] for a in r["actions"]) for r in rows)
    first = Counter((r["actions"][0]["candidate_id"] if r["actions"] else None) for r in rows)
    return {
        "slot": slot, "method": meta["method"], "seed": meta["seed"], "plan_run": meta["plan_run"], "tier": meta["tier"], "execution_class": meta["execution_class"],
        "result_label": ("ARCHIVED_EXECUTION_CLASS" if meta["execution_class"] == ARCHIVED else "CURRENT_EXECUTION_CLASS"),
        "counts": {"planned": n_planned, "measured_valid": len(rows), "technical_not_measured": n_technical, "not_executed": n_planned - len(rows) - n_technical},
        "success": {"n": len(succ), "of": len(rows), "rate": (len(succ) / len(rows)) if rows else None, "text": "%d/%d" % (len(succ), len(rows))},
        "mean_start_discounted_return_G": statistics.fmean(r["G"] for r in rows) if rows else None,
        "G_distribution": _dist([r["G"] for r in rows]),
        "time_to_first_confirmed_success_s": {"denominator": "successes only", "n_success": len(succ), "n_with_time": len(ttf), "mean": statistics.fmean(ttf) if ttf else None,
                                              "median": statistics.median(ttf) if ttf else None, "label": "EPISODE_START_TO_FIRST_INDEPENDENT_CONFIRMED_SUCCESS_SIM_SECONDS"},
        "skill_count": {"mean": statistics.fmean(r["steps"] for r in rows) if rows else None, "median": statistics.median(r["steps"] for r in rows) if rows else None,
                        "distribution": dict(sorted(Counter(r["steps"] for r in rows).items()))},
        "reason_distribution_all": dict(Counter(str(r["reason"]) for r in rows)),
        "failure_reason_distribution": dict(Counter(str(r["reason"]) for r in rows if not r["success"])),
        "task_duration_s": _dist([r["episode_end_elapsed_s"] for r in rows]),
        "action_sequence_summary": {"distinct_sequences": len(seqs), "top_sequences": [{"sequence": list(k), "count": v} for k, v in seqs.most_common(8)],
                                    "first_action_distribution": {str(k): v for k, v in first.most_common()},
                                    "controller_exit_distribution": dict(Counter(a["controller_exit"] for r in rows for a in r["actions"]))},
        "per_case": [{"case_id": r["case_id"], "success": r["success"], "G": r["G"], "steps": r["steps"], "reason": r["reason"], "time_to_first_confirmed_success_s": r["time_to_first_confirmed_success_s"]} for r in rows],
    }


def exact_sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def paired(rows_a, rows_b, la, lb, label):
    A, B = {r["case_id"]: r for r in rows_a}, {r["case_id"]: r for r in rows_b}
    ids = [c for c in A if c in B]
    a_only = sum(1 for c in ids if A[c]["success"] and not B[c]["success"])
    b_only = sum(1 for c in ids if B[c]["success"] and not A[c]["success"])
    both = sum(1 for c in ids if A[c]["success"] and B[c]["success"])
    neither = sum(1 for c in ids if not A[c]["success"] and not B[c]["success"])
    d = [A[c]["G"] - B[c]["G"] for c in ids]
    return {"comparison": label, "a": la, "b": lb, "n_paired_cases": len(ids), "a_success_b_fail": a_only, "a_fail_b_success": b_only, "both_success": both, "both_fail": neither,
            "success_diff_a_minus_b": (sum(1 for c in ids if A[c]["success"]) - sum(1 for c in ids if B[c]["success"])) / len(ids) if ids else None,
            "mean_delta_G_a_minus_b": statistics.fmean(d) if d else None, "sd_delta_G": statistics.stdev(d) if len(d) > 1 else None,
            "se_delta_G": (statistics.stdev(d) / math.sqrt(len(d))) if len(d) > 1 else None, "exact_two_sided_sign_test_p_on_discordant_success": exact_sign_p(a_only, b_only),
            "note": "descriptive; one training seed per cell, 30 same-case pairs; not a seed-level generalisation test"}


def gather(rel):
    rel = Path(rel)
    rows, events, receipts, states = [], [], {}, {}
    for w, d, cls in WORKER_DIRS:
        wd = rel / d
        rows += read_jsonl(wd / "per_episode_eval_row_v2.jsonl")
        events += read_jsonl(wd / "technical_events.jsonl")
        if (wd / "model_hash_receipts.json").is_file():
            receipts.update(load(wd / "model_hash_receipts.json"))
        states[w] = load(wd / "worker_state.json")["state"] if (wd / "worker_state.json").is_file() else "NOT_STARTED"
    return rows, events, receipts, states


def account(rel, man, rows, events, states):
    order = man["test_cases"]["order"]
    metas = {m["slot"]: m for m in man["models"]}
    slot_order = [s for t in TIER_ORDER for s in [m["slot"] for m in man["models"] if m["tier"] == t]]
    by_id = {}
    for r in rows:
        by_id.setdefault(r["slot_id"], []).append(r)
    ev_slot = {}
    for e in events:
        if e.get("kind") in ("INFRASTRUCTURE_FAILURE", "MODEL_HASH_MISMATCH") and e.get("slot_id"):
            ev_slot.setdefault(e["slot_id"], e)
    synthesized = []
    for w, d, cls in WORKER_DIRS:  # a worker that vanished without a closing state: first missing slot is TECHNICAL (process died), never a policy failure
        if states[w] in ("RUNNING", "STARTING"):
            for s in CLASS_ORDER[cls]:
                miss = [c for c in order if "%s|%s" % (s, c) not in by_id]
                if miss:
                    sid = "%s|%s" % (s, miss[0])
                    ev = {"time": now(), "worker": w, "class": cls, "kind": "INFRASTRUCTURE_FAILURE", "subkind": "WORKER_PROCESS_DIED_UNRECORDED", "slot": s, "case_id": miss[0], "slot_id": sid}
                    ev_slot.setdefault(sid, ev)
                    synthesized.append(ev)
                    break
    slots, per_model_n_tech = [], Counter()
    for s in slot_order:
        for c in order:
            sid = "%s|%s" % (s, c)
            if sid in by_id:
                st = "MEASURED"
            elif sid in ev_slot:
                st = "TECHNICAL_NOT_MEASURED"
                per_model_n_tech[s] += 1
            else:
                st = "NOT_EXECUTED_RELEASE_STOPPED"
            slots.append({"slot_id": sid, "slot": s, "case_id": c, "tier": metas[s]["tier"], "status": st, "duplicate_rows": max(0, len(by_id.get(sid, [])) - 1)})
    cnt = Counter(x["status"] for x in slots)
    acc = {"document": "holdout_accounting", "card": CARD, "label": LABEL, "planned_slots": len(slots), "status_counts": dict(cnt),
           "all_planned_slots_measured": cnt.get("MEASURED", 0) == PLANNED, "per_model": {s: {"planned": len(order), "measured": sum(1 for x in slots if x["slot"] == s and x["status"] == "MEASURED"),
                                                                                           "technical_not_measured": per_model_n_tech[s]} for s in slot_order},
           "worker_states": states, "release_stopped": (rel / "STOP_RELEASE").is_file(), "technical_events": len(events) + len(synthesized),
           "policy": "NORMAL_POLICY_OUTCOME counts in the denominator and is never re-run; INFRASTRUCTURE_FAILURE = TECHNICAL_NOT_MEASURED (not a policy failure), no scene swap, no backfill, no auto-retry",
           "slots": slots}
    return acc, synthesized, per_model_n_tech, slot_order


def cmd_collect(a):
    rel = Path(a.release).resolve()
    man = load(rel / "test_release_manifest.json")
    rows, events, receipts, states = gather(rel)
    acc, synth, ntech, slot_order = account(rel, man, rows, events, states)
    metas = {m["slot"]: m for m in man["models"]}
    order = man["test_cases"]["order"]
    # merged raw evidence
    with open(rel / "per_episode_eval_row_v2.jsonl", "w", encoding="utf-8") as f:
        idx = {s: i for i, s in enumerate(slot_order)}
        for r in sorted(rows, key=lambda r: (idx[r["slot"]], r["case_index"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    with open(rel / "technical_events.jsonl", "w", encoding="utf-8") as f:
        for e in events + synth:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    dump(rel / "model_hash_receipts.json", {"document": "model_hash_receipts", "card": CARD, "receipts": receipts,
                                            "expected": {s: metas[s]["final_checkpoint"]["sha256"] for s in slot_order},
                                            "rule": "sha256 checked before the first episode and after the model's 30th; mismatch stops the release"})
    dump(rel / "holdout_accounting.json", acc)
    by_slot = {s: sorted([r["row"] for r in rows if r["slot"] == s], key=lambda r: order.index(r["case_id"])) for s in slot_order}
    stats = {s: model_stats(s, metas[s], by_slot[s], len(order), ntech[s]) for s in slot_order}
    head = {"card": CARD, "label": LABEL, "not_strict_blind_test": True, "GLOBAL_S4": "NOT_COMPLETE", "GLOBAL_S5": "NOT_COMPLETE", "not_GLOBAL_S6": True,
            "all_planned_slots_measured": acc["all_planned_slots_measured"], "r_prior": "absent", "evaluator": "deterministic argmax, isolated RNG, optimizer_steps 0, provider 0",
            "no_best_test_checkpoint_selected": True, "no_selected_dev_checkpoint_in_this_table": True}
    core = [s for s in slot_order if metas[s]["tier"] == "CORE"]
    dump(rel / "core_results.json", dict(head, tier="CORE", planned_episodes=120, models={s: stats[s] for s in core}, paired_same_case=[paired(by_slot[x], by_slot[y], x, y, lbl) for x, y, lbl in PAIRS]))
    dump(rel / "historical_results.json", dict(head, tier="HISTORICAL_EXTENSION", planned_episodes=60, execution_class_label="ARCHIVED_EXECUTION_CLASS",
                                               note="archived v1.3 execution class; not converted to the current Policy; not a same-source ranking with CORE",
                                               models={s: stats[s] for s in slot_order if metas[s]["tier"] == "HISTORICAL_EXTENSION"}))
    dump(rel / "system_reference_results.json", dict(head, tier="SYSTEM_REFERENCE", planned_episodes=30, execution_class_label="ARCHIVED_EXECUTION_CLASS",
                                                     models={s: stats[s] for s in slot_order if metas[s]["tier"] == "SYSTEM_REFERENCE"}))
    write_summary(rel, man, acc, stats, core)
    print(json.dumps({"collected": True, "status_counts": acc["status_counts"], "all_planned_slots_measured": acc["all_planned_slots_measured"], "worker_states": states}))
    return 0


def _f(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def write_summary(rel, man, acc, stats, core):
    L = []
    L += ["# CP-DISR-TB-INDEP-HOLDOUT-01 — frozen T_B prematerialized independent holdout", "",
          "- 测试标签：`%s`（不是 strict blind test）。`GLOBAL_S4 = NOT_COMPLETE`，`GLOBAL_S5 = NOT_COMPLETE`；本次不是 GLOBAL S6。" % LABEL,
          "- R = absent；deterministic argmax；isolated RNG；optimizer_steps = 0；provider = 0；每个 model×case 一次评测。",
          "- planned slots：%d；MEASURED：%d；TECHNICAL_NOT_MEASURED：%d；NOT_EXECUTED_RELEASE_STOPPED：%d。" % (
              acc["planned_slots"], acc["status_counts"].get("MEASURED", 0), acc["status_counts"].get("TECHNICAL_NOT_MEASURED", 0), acc["status_counts"].get("NOT_EXECUTED_RELEASE_STOPPED", 0)),
          "- 全部 210 个 planned slot 正常完成：**%s**。" % ("是" if acc["all_planned_slots_measured"] else "否（见 holdout_accounting.json / technical_events.jsonl）"), ""]
    for tier, title in (("CORE", "A. CORE（current 2.1.1 执行类，4×30=120）"), ("HISTORICAL_EXTENSION", "B. HISTORICAL_EXTENSION（ARCHIVED_EXECUTION_CLASS，2×30=60）"),
                        ("SYSTEM_REFERENCE", "C. SYSTEM_REFERENCE（ARCHIVED_EXECUTION_CLASS，1×30=30）")):
        L += ["## " + title, "", "| slot | method | seed | success | mean G | time-to-first-success mean / median (s, 成功局) | skills mean / median | duration median (s) | valid / planned / technical |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s, st in stats.items():
            if st["tier"] != tier:
                continue
            c = st["counts"]
            L.append("| %s | %s | %s | %s | %s | %s / %s | %s / %s | %s | %d / %d / %d |" % (
                s, st["method"], st["seed"], st["success"]["text"], _f(st["mean_start_discounted_return_G"]), _f(st["time_to_first_confirmed_success_s"]["mean"], 2),
                _f(st["time_to_first_confirmed_success_s"]["median"], 2), _f(st["skill_count"]["mean"], 2), _f(st["skill_count"]["median"], 1), _f(st["task_duration_s"].get("median"), 2),
                c["measured_valid"], c["planned"], c["technical_not_measured"]))
        L += [""]
        for s, st in stats.items():
            if st["tier"] == tier:
                L.append("- %s 失败原因分布：%s" % (s, json.dumps(st["failure_reason_distribution"], ensure_ascii=False)))
        L += [""]
    if True:
        cr = load(Path(rel) / "core_results.json")
        L += ["## CORE 同 case 配对差异（描述性；每格一个训练 seed，30 个配对 case）", "",
              "| 比较 | 配对 n | a成功b失败 | a失败b成功 | 均值 ΔG (a−b) | SE | 精确双侧 sign test p |", "|---|---|---|---|---|---|---|"]
        for p in cr["paired_same_case"]:
            L.append("| %s | %d | %d | %d | %s | %s | %s |" % (p["comparison"], p["n_paired_cases"], p["a_success_b_fail"], p["a_fail_b_success"], _f(p["mean_delta_G_a_minus_b"]), _f(p["se_delta_G"]), _f(p["exact_two_sided_sign_test_p_on_discordant_success"], 4)))
        L += [""]
    L += ["## 解释边界（C1）", "",
          "- 开发期学习过程（0 / 4096 / 8192 / final）与独立 holdout 的 final 模型表现是两个不同的证据层。",
          "- 本 holdout 不能直接证明“B2 的 early-learning 优势可泛化”；它只回答 final 模型在这 30 个冻结 test case 上的表现。",
          "- 若 B2 与 +E 的 final holdout 都高，不删除 early-learning 开发期结果；若 +E 或 B2 的 holdout 下降，如实报告 endpoint generalization gap，不更换 checkpoint。",
          "- 没有选择 best-test checkpoint，没有调参；selected-dev checkpoint 分数不在主 test 表中。", "",
          "## 执行类与限制", "",
          "- CORE 为 CURRENT_2_1_1 执行类；HISTORICAL_EXTENSION / SYSTEM_REFERENCE 为 ARCHIVED_V13 执行类，结果始终标注 ARCHIVED_EXECUTION_CLASS。7 个模型不是同源排名，历史模型未转换进当前 Policy，未改 state_dict。",
          "- 沿用 prep 卡的未决缺口 UC2 / UC3 / UC4 / UC5 / UC7（带入，不在此修复）。",
          "- `success_seconds` 仅是最终 skill 时长（FINAL_SKILL_DURATION_ONLY）；episode 级时间使用 `time_to_first_confirmed_success_s`。", ""]
    (Path(rel) / "holdout_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- verify
def row_internal_ok(r):
    acts = r["actions"]
    ok = len(acts) == r["steps"]
    ok &= all(abs(b["clock_start"] - a["clock_end"]) < 1e-9 for a, b in zip(acts, acts[1:]))
    ok &= (not acts) or abs(acts[0]["clock_start"] - r["episode_start_clock_s"]) < 1e-9
    if r["success"]:
        i = r["first_confirmed_success_decision_index"]
        ok &= i is not None and r["time_to_first_confirmed_success_s"] is not None
    ok &= r["G"] == r["G"] and r["G"] not in (float("inf"), float("-inf"))
    return bool(ok)


def cmd_verify(a):
    rel = Path(a.release).resolve()
    man, lock = load(rel / "test_release_manifest.json"), load(rel / "release_lock.json")
    acc = load(rel / "holdout_accounting.json")
    rows = read_jsonl(rel / "per_episode_eval_row_v2.jsonl")
    events = read_jsonl(rel / "technical_events.jsonl")
    rc = load(rel / "model_hash_receipts.json")
    order = man["test_cases"]["order"]
    metas = {m["slot"]: m for m in man["models"]}
    freeze = load(PREP_DIR / "test_source_freeze.json")
    chk = {}
    chk["release_lock_still_valid"] = verify_release_lock(rel) == []
    chk["planned_210_slots_accounted"] = len(acc["slots"]) == PLANNED and sum(acc["status_counts"].values()) == PLANNED
    chk["no_duplicate_slot_rows"] = len({r["slot_id"] for r in rows}) == len(rows)
    chk["measured_count_equals_rows"] = acc["status_counts"].get("MEASURED", 0) == len(rows)
    chk["every_row_slot_is_planned"] = all(r["slot"] in metas and r["case_id"] in order for r in rows)
    chk["rows_follow_frozen_case_order_per_model"] = all([r["case_id"] for r in sorted([x for x in rows if x["slot"] == s], key=lambda x: x["case_index"])] == order[:sum(1 for x in rows if x["slot"] == s)] for s in metas)
    chk["rows_have_eval_row_v2_fields"] = all(all(k in r["row"] for k in ("case_id", "success", "G", "steps", "reason", "success_seconds", "source_n", "prior_mode", "actions", "time_to_first_confirmed_success_s", "episode_end_elapsed_s", "outcome_class")) for r in rows)
    chk["prior_mode_absent_every_row"] = all(r["row"]["prior_mode"] in ("absent", "empty", "none") or "absent" in str(r["row"]["prior_mode"]) for r in rows)
    chk["every_outcome_is_normal_policy_outcome"] = all(r["row"]["outcome_class"] == "NORMAL_POLICY_OUTCOME" for r in rows)
    chk["row_internal_consistency"] = all(row_internal_ok(r["row"]) for r in rows)
    chk["worker_class_matches_manifest"] = all(r["class"] == metas[r["slot"]]["execution_class"] for r in rows)
    complete = [s for s in metas if sum(1 for r in rows if r["slot"] == s) == len(order)]
    rec_ok = {}
    for s in complete:
        x = rc["receipts"].get(s, {})
        rec_ok[s] = x.get("status") == "COMPLETE" and x.get("sha256_before_first_episode") == x.get("sha256_after_last_episode") == metas[s]["final_checkpoint"]["sha256"]
    chk["complete_models_hash_before_equals_after_equals_manifest"] = all(rec_ok.values()) if rec_ok else True
    chk["checkpoint_files_unchanged_now"] = all(sha256_file(m["final_checkpoint"]["path"]) == m["final_checkpoint"]["sha256"] for m in man["models"])
    chk["eval_payload_optimizer_steps_zero_and_rows_match"] = True
    for w, d, cls in WORKER_DIRS:
        for s in complete:
            pf = rel / d / ("eval_row_v2_%s.json" % s)
            if metas[s]["execution_class"] != cls:
                continue
            pl = load(pf)
            chk["eval_payload_optimizer_steps_zero_and_rows_match"] &= pl["optimizer_steps"] == 0 and pl["eval_action"] == "deterministic_argmax" and \
                [r["row"] for r in sorted([x for x in rows if x["slot"] == s], key=lambda x: x["case_index"])] == pl["rows"]
    t30p = ROOT / "configs/splits/T_B_stage_2a_test30.json"
    chk["test30_identity_unchanged"] = sha256_file(t30p) == freeze["split_configs_sha256"]["configs/splits/T_B_stage_2a_test30.json"] and [r["case_id"] for r in load(t30p)["active"]] == order
    chk["no_prior_split_has_no_cache_pointer"] = not any(k in json.dumps(load(man["bindings"]["no_prior_test_split"])) for k in CACHE_KEYS)
    chk["provider_zero_in_verifier_env"] = not any(os.environ.get(k) for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE"))
    chk["no_open_guard_refusals_in_events"] = not any("open-guard" in json.dumps(e) for e in events)
    chk["technical_events_consistent_with_accounting"] = (len([e for e in events if e.get("kind") in ("INFRASTRUCTURE_FAILURE", "MODEL_HASH_MISMATCH")]) > 0) == (acc["status_counts"].get("TECHNICAL_NOT_MEASURED", 0) > 0 or any(e.get("slot_id") is None for e in events))
    all_done = acc["all_planned_slots_measured"] and all(v == "FINISHED_ALL_PLANNED" for v in acc["worker_states"].values())
    chk["stop_flag_consistent_with_completion"] = (rel / "STOP_RELEASE").is_file() == (not all_done)
    ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout.splitlines()
    left = [l for l in ps if re.search(r"python\S*\s+\S*final_tb_holdout_runner\.py\s+(worker|smoke)", l)]
    chk["no_leftover_worker_processes"] = left == []
    chk["tracked_tree_clean"] = git("status", "--porcelain", "--untracked-files=no") == ""
    chk["all_planned_slots_measured_normal_completion"] = acc["all_planned_slots_measured"] and all(s in ("FINISHED_ALL_PLANNED",) for s in acc["worker_states"].values())
    integrity = {k: v for k, v in chk.items() if k != "all_planned_slots_measured_normal_completion"}
    res = {"document": "verify", "card": CARD, "created": now(), "checks": chk, "integrity_all_pass": all(integrity.values()), "complete_normal": chk["all_planned_slots_measured_normal_completion"],
           "failed": [k for k, v in chk.items() if not v], "head": git("rev-parse", "HEAD"), "model_hash_complete_models": rec_ok, "leftover_processes": left}
    dump(rel / "verify.json", res)
    print(json.dumps({"integrity_all_pass": res["integrity_all_pass"], "complete_normal": res["complete_normal"], "failed": res["failed"]}))
    return 0 if res["integrity_all_pass"] else 5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prerelease"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_prerelease)
    p = sub.add_parser("worker"); p.add_argument("--release", required=True); p.add_argument("--cls", choices=("current", "archived"), required=True); p.set_defaults(fn=cmd_worker)
    p = sub.add_parser("smoke"); p.add_argument("--cls", choices=("current", "archived"), required=True); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("collect"); p.add_argument("--release", required=True); p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("verify"); p.add_argument("--release", required=True); p.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
