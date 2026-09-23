"""Plan v1.1 Stage 1A final test-ID evaluation only. No PPO / no retrain."""
from __future__ import annotations

import csv, hashlib, json, os, time, traceback
from pathlib import Path

from .common import BindingError, canonical, digest
from .stage1a_v11 import (
    BASE_COMMIT, N_CAP, REPORT_PATH, RUNTIME_REL, SPLIT_REL, STATUS_PATH, STOP_REL,
    apply_episode_prior, bind_profile, git_commit, make_bundle, make_policy,
    require_case_cache, sha256_text, utc_now, utc_stamp, write_json, s1,
)
from .platforms.libero.d0_env import CaseSpec
from .stage0c import file_hash
from .torch_rl import load_checkpoint

TEST_SPLIT_REL = Path("configs/splits/D0_stage_1a_test_ids.json")
ACTIVE_REL = Path("configs/splits/D0_stage_1a_test30.json")
NAMESPACE = "cp_disr_exp_v1_1"
POOL_N = 50
ACTIVE_N = 30
B2_CKPT = Path("runs/stage_1a/D0/B2/seed_0/20260922T164616Z_1a2768d8/checkpoints/n_008192.pt")
FULL_CKPT = Path("runs/stage_1a/D0/Full/seed_0/20260922T164616Z_1a2768d8/checkpoints/n_008192.pt")
TARGET_BOX = (-0.22, 0.00, -0.18, -0.02)
SECOND_BOX = (0.04, 0.22, -0.18, -0.02)
MIN_DIST = 0.09
CONTAINER = [0.18, 0.12]
BUFFER = [-0.18, 0.12]
BASELINE = "820a8df6c08f18edf10564891362532c1413264e"


def log(msg):
    print("[stage1a_final_eval] %s %s" % (utc_now(), msg), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unit_interval(payload):
    digest_bytes = hashlib.sha256(canonical(payload).encode("utf-8")).digest()
    return int.from_bytes(digest_bytes[:8], "big") / float(2 ** 64)


def derived_u32(payload):
    digest_bytes = hashlib.sha256(canonical(payload).encode("utf-8")).digest()
    return int.from_bytes(digest_bytes[:4], "big")


def sample_xy(base, tag, box):
    x_lo, x_hi, y_lo, y_hi = box
    x = x_lo + (x_hi - x_lo) * unit_interval({**base, "axis": tag + "_x"})
    y = y_lo + (y_hi - y_lo) * unit_interval({**base, "axis": tag + "_y"})
    return [float(x), float(y)]


def dist(a, b):
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def pose_key(row):
    return (
        round(float(row["target_xy"][0]), 12),
        round(float(row["target_xy"][1]), 12),
        round(float(row["second_xy"][0]), 12),
        round(float(row["second_xy"][1]), 12),
        round(float(row.get("container_xy", CONTAINER)[0]), 12),
        round(float(row.get("container_xy", CONTAINER)[1]), 12),
        round(float(row.get("buffer_xy", BUFFER)[0]), 12),
        round(float(row.get("buffer_xy", BUFFER)[1]), 12),
        bool(row.get("lid_closed", True)),
    )


def collect_used_inits(root):
    used = []
    split = json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))
    for row in list(split.get("train") or []) + list(split.get("dev") or []) + list(split.get("test") or []):
        used.append({"source": "D0_stage_1a.json:" + row["case_id"], "row": row})
    for path in [
        root / "experiments/stage_1a_inputs",
        root / "experiments/stage_0c_inputs",
        root / "experiments/part_0_validation",
        root / "runs/stage_0a",
        root / "runs/stage_0d",
    ]:
        if not path.exists():
            continue
        for fp in path.rglob("reset_config.json"):
            try:
                rec = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(rec, dict) and "target_xy" in rec and "second_xy" in rec:
                used.append({"source": str(fp.relative_to(root)), "row": rec})
    seeds, poses = set(), set()
    for item in used:
        row = item["row"]
        if "seed" in row:
            try:
                seeds.add(int(row["seed"]))
            except Exception:
                pass
        if "target_xy" in row and "second_xy" in row:
            poses.add(pose_key(row))
    return used, seeds, poses


def legal_pose(t, s):
    if not (TARGET_BOX[0] <= t[0] <= TARGET_BOX[1] and TARGET_BOX[2] <= t[1] <= TARGET_BOX[3]):
        return False, "target_out_of_box"
    if not (SECOND_BOX[0] <= s[0] <= SECOND_BOX[1] and SECOND_BOX[2] <= s[1] <= SECOND_BOX[3]):
        return False, "second_out_of_box"
    if dist(t, s) < MIN_DIST:
        return False, "min_distance"
    return True, None
def sample_case(index, used_seeds, used_poses):
    reject_index = 0
    while reject_index < 64:
        base = {
            "namespace": NAMESPACE,
            "task": "D0",
            "split": "test",
            "case_or_episode": int(index),
            "stream": "case_init",
            "reject_index": int(reject_index),
        }
        t = sample_xy(base, "target", TARGET_BOX)
        s = sample_xy(base, "second", SECOND_BOX)
        ok, reason = legal_pose(t, s)
        seed = derived_u32({
            "namespace": NAMESPACE,
            "task": "D0",
            "split": "test",
            "case_or_episode": int(index),
            "stream": "env_reset",
            "reject_index": int(reject_index),
        })
        row = {
            "case_id": "D0_test_%02d" % index,
            "split": "test",
            "pool_index": int(index),
            "seed": int(seed),
            "target_xy": t,
            "second_xy": s,
            "container_xy": list(CONTAINER),
            "buffer_xy": list(BUFFER),
            "lid_closed": True,
            "stream": {
                "namespace": NAMESPACE,
                "task": "D0",
                "split": "test",
                "case_or_episode": int(index),
                "streams": ["case_init", "env_reset"],
                "reject_index": int(reject_index),
                "note": "test case stream omits method and training_seed",
            },
        }
        if not ok:
            reject_index += 1
            continue
        if seed in used_seeds:
            reject_index += 1
            continue
        if pose_key(row) in used_poses:
            reject_index += 1
            continue
        return row
    raise BindingError("unable to sample legal independent test case %s" % index)


def verify_checkpoint(root, rel, method):
    path = root / rel
    sidecar = path.with_suffix(".json")
    rng = path.with_suffix(".rng.json")
    if not path.is_file() or not sidecar.is_file() or not rng.is_file():
        raise BindingError("untrusted or missing checkpoint artifacts for %s" % method)
    file_sha = sha256_file(path)
    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    man = doc.get("manifest") or {}
    if doc.get("sha256") != file_sha:
        raise BindingError("%s sidecar sha256 %s != file %s" % (method, doc.get("sha256"), file_sha))
    if man.get("method") != method:
        raise BindingError("%s checkpoint method mismatch" % method)
    if int(man.get("N") or 0) != 8192:
        raise BindingError("%s checkpoint N is %s" % (method, man.get("N")))
    if abs(float(man.get("H")) - 23.09999999999752) > 1e-12:
        raise BindingError("%s H mismatch" % method)
    if abs(float(man.get("d_ref")) - 4.5500000000015195) > 1e-12:
        raise BindingError("%s d_ref mismatch" % method)
    if man.get("actor_episode_discount_weight") is not False:
        raise BindingError("%s actor_episode_discount_weight not false" % method)
    if man.get("split_hash") != "26e82ba484a2f49bbfba2aa82abaa0cee5c991fdd49514be7311cfeae2b73c1c":
        raise BindingError("%s split_hash changed" % method)
    rng_rel = str(rng.relative_to(root)).replace("\\", "/")
    return {
        "method": method,
        "path": str(rel).replace("\\", "/"),
        "abs_path": str(path),
        "sha256": file_sha,
        "sidecar_sha256": doc.get("sha256"),
        "manifest_hash": doc.get("manifest_hash"),
        "N": man.get("N"),
        "T": man.get("T"),
        "H": man.get("H"),
        "d_ref": man.get("d_ref"),
        "Tcap": man.get("Tcap"),
        "complete_updates_field": man.get("complete_updates"),
        "authorized_complete_updates": 8,
        "git_commit": man.get("git_commit"),
        "base_commit": man.get("base_commit"),
        "runtime_manifest": man.get("runtime_manifest"),
        "runtime_hash": man.get("runtime_hash"),
        "config_hash": man.get("config_hash"),
        "cache_hash": man.get("cache_hash"),
        "split_hash": man.get("split_hash"),
        "rng_sidecar": rng_rel,
        "bytes": path.stat().st_size,
    }


def work_dir(root, stamp, configsha8):
    return root / "runs" / "stage_1a" / "final_test_id" / ("%s_%s" % (stamp, configsha8))


def load_or_init_work(root):
    pointer = root / "runs" / "stage_1a" / "final_test_id" / "CURRENT.json"
    if pointer.is_file():
        cur = json.loads(pointer.read_text(encoding="utf-8"))
        out = root / cur["work_dir"]
        if out.is_dir() and (out / "selection_manifest.json").is_file():
            return out, cur
    prof = bind_profile(root)
    b2 = verify_checkpoint(root, B2_CKPT, "B2")
    full = verify_checkpoint(root, FULL_CKPT, "Full")
    stamp = utc_stamp()
    payload = {
        "scope": "STAGE_1A_FINAL_TEST_ONLY",
        "H": prof["H"],
        "d_ref": prof["d_ref"],
        "Tcap": prof["Tcap"],
        "Ncap": N_CAP,
        "actor_episode_discount_weight": False,
        "runtime_path": prof["runtime_path"],
        "split_path": prof["split_path"],
        "task_deadline_seconds": prof["task_deadline_seconds"],
        "base_commit": BASELINE,
        "selected_b2_sha256": b2["sha256"],
        "selected_full_sha256": full["sha256"],
        "profile": "method-2.1.1",
    }
    configsha8 = sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))[:8]
    out = work_dir(root, stamp, configsha8)
    out.mkdir(parents=True, exist_ok=True)
    cur = {"stamp": stamp, "configsha8": configsha8, "work_dir": str(out.relative_to(root)).replace("\\", "/")}
    write_json(pointer, cur)
    return out, cur
def freeze_selection(root, out, cur):
    if (out / "selection_manifest.json").is_file():
        log("selection_manifest already frozen")
        return json.loads((out / "selection_manifest.json").read_text(encoding="utf-8"))
    import yaml
    prof = bind_profile(root)
    b2 = verify_checkpoint(root, B2_CKPT, "B2")
    full = verify_checkpoint(root, FULL_CKPT, "Full")
    hist = root / "status/history/stage_1a.smoke_pass.820a8df.json"
    smoke = json.loads(hist.read_text(encoding="utf-8")) if hist.is_file() else json.loads((root / STATUS_PATH).read_text(encoding="utf-8"))
    doc = {
        "stage": "1A",
        "execution_scope": "STAGE_1A_FINAL_TEST_ONLY",
        "training_complete": True,
        "smoke_passed": True,
        "final_test_complete": False,
        "selection_frozen_at": utc_now(),
        "selection_frozen_before_any_test_result": True,
        "reselection_forbidden": True,
        "stamp": cur["stamp"],
        "configsha8": cur["configsha8"],
        "baseline_commit": BASELINE,
        "git_head_at_freeze": git_commit(root),
        "H": prof["H"],
        "d_ref": prof["d_ref"],
        "Tcap": prof["Tcap"],
        "task_deadline_seconds": prof["task_deadline_seconds"],
        "actor_episode_discount_weight": False,
        "selected": {"B2": b2, "Full": full},
        "selection_rule": "locked from completed smoke; n_004096 vs n_008192 is not compared again",
        "smoke_jobs": smoke.get("jobs"),
        "old_stage2a_stop_guard": str(STOP_REL).replace("\\", "/"),
        "stop_guard_present": (root / STOP_REL).is_file(),
        "note": "No test-ID metric was computed before this freeze.",
    }
    write_json(out / "selection_manifest.json", doc)
    (out / "selection_manifest.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(out / "source_hashes.json", {
        "split_hash": sha256_file(root / SPLIT_REL),
        "runtime_hash": sha256_file(root / RUNTIME_REL),
        "b2_ckpt": b2["sha256"],
        "full_ckpt": full["sha256"],
        "git_head": git_commit(root),
        "baseline": BASELINE,
    })
    log("froze selection_manifest B2=%s Full=%s" % (b2["sha256"][:12], full["sha256"][:12]))
    return doc


def register_test_ids(root, out):
    if (out / "active_test30_manifest.json").is_file() and TEST_SPLIT_REL.is_file():
        log("test-ID manifests already frozen")
        return json.loads((out / "active_test30_manifest.json").read_text(encoding="utf-8"))
    existing = json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))
    if existing.get("test"):
        raise BindingError("unexpected test list already inside D0_stage_1a.json")
    used, used_seeds, used_poses = collect_used_inits(root)
    pool = []
    for i in range(POOL_N):
        row = sample_case(i, used_seeds, used_poses)
        used_seeds.add(int(row["seed"]))
        used_poses.add(pose_key(row))
        pool.append(row)
    active = pool[:ACTIVE_N]
    overlap_rows = []
    for row in pool:
        hits = []
        key = pose_key(row)
        for item in used:
            src = item["row"]
            if "target_xy" in src and "second_xy" in src and pose_key(src) == key:
                hits.append(item["source"])
        overlap_rows.append({"case_id": row["case_id"], "overlap": hits})
    if any(r["overlap"] for r in overlap_rows):
        raise BindingError("generated test poses overlap used inits")
    registration = {
        "registration_time": utc_now(),
        "registration_timing": "post-training, pre-evaluation freeze; NOT a pre-training registration",
        "task_id": "D0",
        "version": "d0-runtime-v2.1-p0-test-id-final-eval",
        "policy": "independent_test_pool_after_smoke",
        "namespace": NAMESPACE,
        "stream_fields": ["namespace", "task", "split", "case_or_episode", "stream"],
        "stream_excludes": ["method", "training_seed"],
        "pool_n": POOL_N,
        "active_n": ACTIVE_N,
        "reserved_n": POOL_N - ACTIVE_N,
        "distribution": {
            "target_xy_box": TARGET_BOX,
            "second_xy_box": SECOND_BOX,
            "min_distance": MIN_DIST,
            "container_xy": CONTAINER,
            "buffer_xy": BUFFER,
            "lid_closed": True,
            "source": "scripts/stage_1a/build_d0_split.py frozen D0 init distribution",
        },
        "do_not_rewrite": str(SPLIT_REL).replace("\\", "/"),
        "train_dev_split_hash": sha256_file(root / SPLIT_REL),
        "test": pool,
        "active": active,
        "reserved": pool[ACTIVE_N:],
    }
    write_json(root / TEST_SPLIT_REL, registration)
    write_json(root / ACTIVE_REL, {"active": active, "n": len(active), "shared_by_methods": ["B2", "Full"], "same_order": True, "same_reset_seed": True})
    write_json(out / "test_id_pool_manifest.json", registration)
    write_json(out / "active_test30_manifest.json", {"active": active, "n": len(active), "case_ids": [r["case_id"] for r in active], "frozen_at": utc_now(), "methods_share_identical_cases": True})
    write_json(out / "split_overlap_audit.json", {
        "n_used_sources": len(used),
        "n_used_seeds": len(used_seeds),
        "n_used_poses": len(used_poses),
        "active_overlap": overlap_rows[:ACTIVE_N],
        "any_overlap": False,
        "filtered_by_model_performance": False,
        "filtered_by_scripted_success": False,
        "filtered_by_nonempty_prior": False,
        "independence_rule": "reject exact pose/seed collision with used inits; reject out-of-box and min-distance only",
    })
    log("registered test pool 50 / active 30; train/dev split file not rewritten")
    return {"active": active}
def capture_test_images(root, rows, gpu):
    from PIL import Image
    import numpy as np
    from .platforms.libero.d0_env import make_env
    mapping = []
    for row in rows:
        dest = root / "experiments" / "stage_1a_inputs" / "test" / "D0" / row["case_id"]
        dest.mkdir(parents=True, exist_ok=True)
        rgb_path = dest / "rgb.png"
        if rgb_path.is_file() and (dest / "CAPTURE_COMPLETE").is_file():
            mapping.append({**row, "image_ref": str(rgb_path.relative_to(root)).replace("\\", "/"), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(root)).replace("\\", "/")})
            continue
        spec = CaseSpec(
            case_id=row["case_id"], split="test", seed=int(row["seed"]),
            target_xy=tuple(row["target_xy"]), second_xy=tuple(row["second_xy"]),
            container_xy=tuple(row["container_xy"]), buffer_xy=tuple(row["buffer_xy"]),
            lid_closed=True, task_id="D0", deadline=43.0,
        )
        env = make_env(spec, gpu=gpu)
        try:
            env.reset()
            env._open_gripper_reset()
            obs = env.public_observation()
            Image.fromarray(np.asarray(obs["rgb"]).astype("uint8")).save(rgb_path)
            np.save(dest / "depth.npy", np.asarray(obs["depth"]))
            write_json(dest / "reset_config.json", row)
            (dest / "CAPTURE_COMPLETE").write_text("complete\n", encoding="utf-8")
        finally:
            env.close()
        mapping.append({**row, "image_ref": str(rgb_path.relative_to(root)).replace("\\", "/"), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(root)).replace("\\", "/")})
        log("captured %s" % row["case_id"])
    return mapping


def write_cost_ledger(out, audit_rows, new_calls):
    path = out / "cost_ledger.csv"
    fields = ["case_id", "status", "attempts", "input_tokens", "output_tokens", "image_tokens", "total_tokens", "api_calls_billed"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in audit_rows:
            usage = r.get("usage_tokens") or {}
            billed = 0 if r.get("status") == "REUSED" else int(r.get("attempts") or 1)
            w.writerow({
                "case_id": r["case_id"], "status": r.get("status"), "attempts": r.get("attempts"),
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "image_tokens": usage.get("image_tokens"), "total_tokens": usage.get("total_tokens"),
                "api_calls_billed": billed,
            })
    write_json(out / "cost_ledger_summary.json", {"new_vlm_requests": new_calls, "note": "token usage recorded; credentials not written"})


def materialize_caches(root, out, gpu):
    from .stage1a import generate_d0_caches
    from .platforms.libero.snapshot import load_cache_edges
    active = json.loads((out / "active_test30_manifest.json").read_text(encoding="utf-8"))["active"]
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise BindingError("DASHSCOPE_API_KEY missing from process environment; BLOCKED without model fallback")
    mapping = capture_test_images(root, active, gpu=gpu)
    existing_rgb = {}
    for split in ("train", "dev"):
        base = root / "experiments" / "stage_1a_inputs" / split / "D0"
        if not base.is_dir():
            continue
        for case_dir in base.iterdir():
            rgb = case_dir / "rgb.png"
            if rgb.is_file():
                existing_rgb.setdefault(file_hash(rgb), []).append(str(rgb.relative_to(root)))
    few = root / "experiments/stage_0c_inputs/fewshots"
    if few.is_dir():
        for rgb in few.rglob("rgb.png"):
            existing_rgb.setdefault(file_hash(rgb), []).append(str(rgb.relative_to(root)))
    rgb_overlap = [{"case_id": rec["case_id"], "hits": existing_rgb.get(rec["image_sha256"]) } for rec in mapping if existing_rgb.get(rec["image_sha256"])]
    if rgb_overlap:
        raise BindingError("test RGB identity overlaps used inputs: %s" % rgb_overlap)
    log("generating VLM caches for %s active test cases" % len(mapping))
    caches = generate_d0_caches(root, mapping, gpu=gpu)
    cache_map = {r.get("case_id"): r for r in caches if r.get("case_id")}
    updated, audit_rows = [], []
    new_calls = reused = nonempty = 0
    for row in mapping:
        rec = cache_map.get(row["case_id"]) or {}
        if not rec.get("cache_dir"):
            raise BindingError("cache generation did not bind %s" % row["case_id"])
        folder = root / rec["cache_dir"]
        if not (folder / "COMPLETE").is_file():
            raise BindingError("missing COMPLETE for %s; missing cache is not EMPTY_PRIOR" % row["case_id"])
        edges = load_cache_edges(folder)
        status = rec.get("status")
        if status == "REUSED":
            reused += 1
        else:
            new_calls += 1
        if edges:
            nonempty += 1
        merged = dict(row)
        merged.update({"cache_dir": rec.get("cache_dir"), "cache_key": rec.get("cache_key"), "cache_status": status, "source_relation_count": len(edges)})
        updated.append(merged)
        usage = None
        attempts = rec.get("attempts")
        raw_path = folder / "raw_response.json"
        if raw_path.is_file():
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if isinstance(raw, list) and raw:
                usage = raw[-1].get("usage")
                attempts = attempts or len(raw)
        audit_rows.append({
            "case_id": row["case_id"], "cache_dir": rec.get("cache_dir"), "cache_key": rec.get("cache_key"),
            "status": status, "complete": True, "source_relation_count": len(edges),
            "legal_empty": len(edges) == 0, "missing_cache": False, "attempts": attempts,
            "usage_tokens": None if not usage else {
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "image_tokens": usage.get("image_tokens"), "total_tokens": usage.get("total_tokens"),
            },
        })
    write_json(out / "active_test30_manifest.json", {"active": updated, "n": len(updated), "case_ids": [r["case_id"] for r in updated], "frozen_at": utc_now(), "methods_share_identical_cases": True, "cache_bound": True})
    pool = json.loads((out / "test_id_pool_manifest.json").read_text(encoding="utf-8"))
    by_id = {r["case_id"]: r for r in updated}
    pool["test"] = [by_id.get(r["case_id"], r) for r in pool["test"]]
    pool["active"] = updated
    write_json(out / "test_id_pool_manifest.json", pool)
    write_json(root / TEST_SPLIT_REL, pool)
    write_json(root / ACTIVE_REL, {"active": updated, "n": len(updated), "shared_by_methods": ["B2", "Full"], "same_order": True, "same_reset_seed": True})
    write_json(out / "cache_audit.json", {
        "n_enabled": len(updated), "n_missing": 0, "n_reused": reused, "n_new_requests": new_calls,
        "n_nonempty_source": nonempty, "n_legal_empty": sum(1 for r in audit_rows if r["legal_empty"]),
        "rows": audit_rows, "note": "B2 does not generate a second prior set; missing cache is not EMPTY_PRIOR",
    })
    write_cost_ledger(out, audit_rows, new_calls)
    log("cache audit enabled=%s reused=%s new=%s nonempty=%s" % (len(updated), reused, new_calls, nonempty))
    return updated
def fingerprint_policy(policy):
    h = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def attach_test_cases(bundle, root, rows, deadline):
    for row in rows:
        spec = CaseSpec(
            case_id=row["case_id"], split="test", seed=int(row["seed"]),
            target_xy=tuple(row["target_xy"]), second_xy=tuple(row["second_xy"]),
            container_xy=tuple(row["container_xy"]), buffer_xy=tuple(row["buffer_xy"]),
            lid_closed=True, task_id="D0", deadline=float(deadline),
        )
        spec.deadline = float(deadline)
        bundle.cases[row["case_id"]] = spec
        bundle.caches[row["case_id"]] = root / row["cache_dir"]


def method_eval_dir(root, method):
    return root / "runs" / "stage_1a" / "D0" / method / "seed_0" / "20260922T164616Z_1a2768d8" / "evaluation" / "final_test_id"


def evaluate_method(root, out, method, gpu):
    import torch
    from .collector import Collector
    sel = json.loads((out / "selection_manifest.json").read_text(encoding="utf-8"))
    active = json.loads((out / "active_test30_manifest.json").read_text(encoding="utf-8"))["active"]
    if len(active) != ACTIVE_N:
        raise BindingError("active test set is %s, required %s" % (len(active), ACTIVE_N))
    ckpt_info = sel["selected"][method]
    ckpt_path = root / ckpt_info["path"]
    live_sha = sha256_file(ckpt_path)
    if live_sha != ckpt_info["sha256"]:
        raise BindingError("%s checkpoint bytes changed after freeze" % method)
    prof = bind_profile(root)
    hashes_doc = json.loads((out / "source_hashes.json").read_text(encoding="utf-8"))
    hashes_doc.update({"H": prof["H"], "d_ref": prof["d_ref"], "method": method, "checkpoint_sha256": live_sha})
    eval_root = method_eval_dir(root, method)
    eval_root.mkdir(parents=True, exist_ok=True)
    combined_eval = out / method
    combined_eval.mkdir(parents=True, exist_ok=True)
    episode_path = eval_root / "test_episode.jsonl"
    decision_path = eval_root / "test_decision_log.jsonl"
    fail_path = eval_root / "failures.jsonl"
    done, rows = set(), []
    if episode_path.is_file():
        for line in episode_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == "valid":
                done.add(rec["case_id"])
                rows.append(rec)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu) if str(os.environ.get("CUDA_VISIBLE_DEVICES") or "") == "" else 0)
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    torch.set_grad_enabled(False)
    s1.seed_all(0)
    bundle = make_bundle(root)
    attach_test_cases(bundle, root, active, prof["task_deadline_seconds"])
    policy = make_policy(bundle.template, method, device)
    load_checkpoint(ckpt_path, policy, optimizer=None)
    policy.eval()
    fp_before = fingerprint_policy(policy)
    write_json(eval_root / "weight_fingerprint_before.json", {"sha256": fp_before, "checkpoint_file_sha256": live_sha, "optimizer_loaded": False})
    collector = Collector(bundle, policy)
    started = time.time()
    trans_rows, infra = [], 0
    if decision_path.is_file():
        trans_rows = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if episode_path.is_file():
        infra = sum(1 for line in episode_path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("status") == "infrastructure_failed")
    try:
        for rec in active:
            case = rec["case_id"]
            if case in done:
                log("%s skip completed %s" % (method, case))
                continue
            require_case_cache(root, rec)
            ep_started = time.time()
            try:
                snap = bundle.start_case(case)
                snap, prior, source_n = apply_episode_prior(method, bundle, snap, sampler=None, eval_original=True)
                collector.reset_episode(snap.env_id, snap.episode_id)
                G, steps, success, reason, success_seconds = 0.0, 0, False, None, None
                reward_events, skills = [], []
                while True:
                    t, result = collector.step(snap, deterministic=True)
                    if t is None:
                        reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
                        success = bool(result.get("success") if isinstance(result, dict) else getattr(result, "success", False))
                        break
                    compact = s1.compact_transition(
                        method, 0, case, t, result, collector.last_output, collector.last_execution,
                        prior.audit_mode, prior.original_hash, source_n, rec.get("cache_key"),
                        "v11_1A_D0_%s_s0_final_test" % method,
                    )
                    trans_rows.append(compact)
                    s1.append_jsonl(decision_path, compact)
                    G += float(t.reward) * float(t.weight)
                    steps += 1
                    snap = t.next_snapshot
                    actual = result if isinstance(result, dict) else {"success": result.success, "reason": result.reason, "reward_events": list(result.reward_events)}
                    skills.append({"skill": t.selected_candidate_id, "duration": t.duration, "reward": t.reward, "reason": actual.get("reason")})
                    if actual.get("reward_events"):
                        reward_events.extend(list(actual["reward_events"]))
                    success = bool(actual.get("success"))
                    reason = actual.get("reason")
                    if success and success_seconds is None:
                        success_seconds = float(bundle.clock.now_seconds() - bundle.episode_start_seconds)
                    if t.terminated or t.truncated:
                        break
                duration = float(bundle.clock.now_seconds() - bundle.episode_start_seconds)
                if not success:
                    success_seconds = None
                ep = {
                    "status": "valid", "case_id": case, "split": "test", "method": method,
                    "source_run": "v11_1A_D0_%s_s0_20260922T164616Z_1a2768d8" % method,
                    "checkpoint": str(ckpt_path), "checkpoint_sha256": live_sha, "reset_seed": rec["seed"],
                    "success": bool(success), "reward_events": reward_events, "discounted_return": G,
                    "completion_time": success_seconds, "episode_duration_seconds": duration, "skills": steps,
                    "skill_log": skills, "terminal_reason": reason,
                    "prior_source_relation_count": source_n, "prior_effective_relation_count": len(prior.edges),
                    "prior_mode": prior.audit_mode, "cache_key": rec.get("cache_key"), "cache_dir": rec.get("cache_dir"),
                    "H": prof["H"], "d_ref": prof["d_ref"], "task_deadline_seconds": prof["task_deadline_seconds"],
                    "hashes": hashes_doc, "wall_seconds": time.time() - ep_started, "optimizer_steps": 0,
                }
                s1.append_jsonl(episode_path, ep)
                rows.append(ep)
                if not success:
                    s1.append_jsonl(fail_path, {"case_id": case, "method": method, "reason": reason, "infrastructure": False})
                log("%s %s success=%s G=%.4f skills=%s reason=%s" % (method, case, success, G, steps, reason))
            except Exception as exc:
                infra += 1
                fail = {"status": "infrastructure_failed", "case_id": case, "method": method, "error": type(exc).__name__, "detail": str(exc)[:500], "traceback": traceback.format_exc()[-2000:], "infrastructure": True}
                s1.append_jsonl(fail_path, fail)
                s1.append_jsonl(episode_path, fail)
                log("%s INFRA %s %s" % (method, case, type(exc).__name__))
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
    fp_after = fingerprint_policy(policy)
    if fp_after != fp_before:
        raise BindingError("%s policy weights changed during eval" % method)
    if sha256_file(ckpt_path) != live_sha:
        raise BindingError("%s checkpoint file changed during eval" % method)
    write_json(eval_root / "weight_fingerprint_after.json", {"sha256": fp_after, "checkpoint_file_sha256": live_sha, "unchanged": True})
    valid = [r for r in rows if r.get("status") == "valid"]
    succ = [r for r in valid if r.get("success")]
    agg = s1.aggregate_struct(trans_rows) if trans_rows else {}
    if method == "B2":
        dp_opp, dp_rate, delta_rate = 0, "N/A", "N/A"
    else:
        qual = [r for r in trans_rows if r.get("qualifying_for_dp")]
        dp_opp = len(qual)
        if dp_opp == 0:
            dp_rate, delta_rate = "N/A", "N/A"
        else:
            dp_rate = sum(1 for r in qual if r["struct"]["dp_nonzero"]) / float(dp_opp)
            delta_rate = sum(1 for r in qual if r["struct"]["residual_nonzero"]) / float(dp_opp)
    metrics = {
        "method": method, "planned_episodes": ACTIVE_N, "valid_episodes": len(valid),
        "infrastructure_failed": infra, "success_count": len(succ), "success_denominator": len(valid),
        "success_rate": (len(succ) / len(valid)) if valid else None,
        "mean_discounted_return": (sum(r["discounted_return"] for r in valid) / len(valid)) if valid else None,
        "success_completion_times": [r["completion_time"] for r in succ],
        "n_success_completion_times": len(succ),
        "mean_success_completion_time": (sum(r["completion_time"] for r in succ) / len(succ)) if succ else "NA",
        "episode_durations": [r.get("episode_duration_seconds") for r in valid],
        "failure_reasons": {},
        "full_nonempty_source_prior_n": sum(1 for r in valid if int(r.get("prior_source_relation_count") or 0) > 0) if method == "Full" else 0,
        "dp_opportunity_n": dp_opp, "dp_nonzero_rate": dp_rate, "delta_nonzero_rate": delta_rate,
        "eval_wall_seconds": time.time() - started, "optimizer_steps": 0, "new_rl_updates": 0,
        "H": prof["H"], "checkpoint_sha256": live_sha, "weight_fingerprint": fp_before,
        "deterministic": True, "eval_action": "deterministic_argmax",
        "prior_mode": "absent" if method == "B2" else "original", "struct_aggregate": agg,
    }
    for r in valid:
        if not r.get("success"):
            k = str(r.get("terminal_reason"))
            metrics["failure_reasons"][k] = metrics["failure_reasons"].get(k, 0) + 1
    write_json(eval_root / "test_metrics.json", metrics)
    with (eval_root / "test_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "success", "discounted_return", "completion_time", "episode_duration_seconds", "skills", "terminal_reason", "prior_source_relation_count", "prior_effective_relation_count"])
        w.writeheader()
        for r in valid:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    write_json(eval_root / "evaluation_manifest.json", {"method": method, "execution_scope": "STAGE_1A_FINAL_TEST_ONLY", "checkpoint": ckpt_info, "n_planned": ACTIVE_N, "n_valid": len(valid), "n_infra": infra, "hashes": hashes_doc, "gpu": gpu, "model_eval": True, "optimizer_steps": 0, "started_at": utc_now()})
    for name in ("test_episode.jsonl", "test_decision_log.jsonl", "failures.jsonl", "test_metrics.json", "test_metrics.csv", "evaluation_manifest.json"):
        src = eval_root / name
        if src.is_file():
            (combined_eval / name).write_bytes(src.read_bytes())
    write_json(combined_eval / "complete.json", {"method": method, "valid": len(valid), "infra": infra, "complete": len(valid) == ACTIVE_N})
    log("%s eval done valid=%s/%s infra=%s" % (method, len(valid), ACTIVE_N, infra))
    return metrics
def summarize(root, out):
    b2 = json.loads((method_eval_dir(root, "B2") / "test_metrics.json").read_text(encoding="utf-8"))
    full = json.loads((method_eval_dir(root, "Full") / "test_metrics.json").read_text(encoding="utf-8"))
    cache_audit = json.loads((out / "cache_audit.json").read_text(encoding="utf-8"))
    rows = []
    for method, met in (("B2", b2), ("Full", full)):
        rows.append({
            "method": method, "valid_episodes": met["valid_episodes"], "success_count": met["success_count"],
            "success_rate": met["success_rate"], "mean_discounted_return": met["mean_discounted_return"],
            "mean_success_completion_time": met["mean_success_completion_time"],
            "dp_opportunity_n": met["dp_opportunity_n"], "dp_nonzero_rate": met["dp_nonzero_rate"],
            "delta_nonzero_rate": met["delta_nonzero_rate"], "nonempty_source_prior_n": met.get("full_nonempty_source_prior_n"),
            "optimizer_steps": 0,
        })
    with (out / "final_test_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (out / "evaluation_run_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "case_id", "success", "discounted_return", "terminal_reason"])
        w.writeheader()
        for method in ("B2", "Full"):
            for line in (method_eval_dir(root, method) / "test_episode.jsonl").read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("status") != "valid":
                    continue
                w.writerow({"method": method, "case_id": rec["case_id"], "success": rec["success"], "discounted_return": rec["discounted_return"], "terminal_reason": rec.get("terminal_reason")})
    complete = b2["valid_episodes"] == ACTIVE_N and full["valid_episodes"] == ACTIVE_N
    summary = {
        "execution_scope": "STAGE_1A_FINAL_TEST_ONLY", "training_complete": True, "smoke_passed": True,
        "final_test_complete": complete, "B2": b2, "Full": full,
        "cache_audit": {k: cache_audit[k] for k in cache_audit if k != "rows"},
        "new_rl_updates": 0, "optimizer_steps": 0,
    }
    write_json(out / "final_test_summary.json", summary)
    md = [
        "# Stage 1A final test-ID evaluation", "",
        "execution_scope: STAGE_1A_FINAL_TEST_ONLY",
        "training_complete: true (unchanged 8 updates / N=8192)",
        "smoke_passed: true",
        "final_test_complete: %s" % str(complete).lower(),
        "test-ID registration: post-training, frozen before any test result", "",
        "## Checkpoints", "",
        "- B2 n_008192 sha256: %s" % b2["checkpoint_sha256"],
        "- Full n_008192 sha256: %s" % full["checkpoint_sha256"],
        "- weights unchanged during eval: true", "",
        "## Results", "",
        "| method | valid/planned | success | mean discounted return | mean success time | nonempty source prior | DP opp | DP rate | Delta rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| B2 | %s/%s | %s | %s | %s | 0 | %s | %s | %s |" % (b2["valid_episodes"], ACTIVE_N, b2["success_count"], b2["mean_discounted_return"], b2["mean_success_completion_time"], b2["dp_opportunity_n"], b2["dp_nonzero_rate"], b2["delta_nonzero_rate"]),
        "| Full | %s/%s | %s | %s | %s | %s | %s | %s | %s |" % (full["valid_episodes"], ACTIVE_N, full["success_count"], full["mean_discounted_return"], full["mean_success_completion_time"], full["full_nonempty_source_prior_n"], full["dp_opportunity_n"], full["dp_nonzero_rate"], full["delta_nonzero_rate"]),
        "", "Low test success or Full not leading is not an implementation failure.",
        "New RL updates: 0. Optimizer steps: 0.", "",
    ]
    (out / "final_test_summary.md").write_text("\n".join(md), encoding="utf-8")
    return summary, complete


def update_stage_outputs(root, out, summary, complete):
    hist_report = (root / "reports/history/stage_1a_summary.smoke_pass.820a8df.md").read_text(encoding="utf-8")
    smoke = json.loads((root / "status/history/stage_1a.smoke_pass.820a8df.json").read_text(encoding="utf-8"))
    extra = ["", "## Final test-ID completion", "", "This section is appended after smoke PASS. Training numbers above are unchanged.", "", (out / "final_test_summary.md").read_text(encoding="utf-8"), "", "Artifacts: %s" % str(out.relative_to(root)).replace("\\", "/"), ""]
    (root / REPORT_PATH).write_text(hist_report.rstrip() + "\n" + "\n".join(extra), encoding="utf-8")
    status = dict(smoke)
    status.update({
        "execution_scope": "STAGE_1A_FINAL_TEST_ONLY", "training_complete": True, "smoke_passed": True,
        "final_test_complete": complete, "status": "PASS" if complete else "RUNNING",
        "final_test_dir": str(out.relative_to(root)).replace("\\", "/"),
        "final_test": {
            "B2": {"valid": summary["B2"]["valid_episodes"], "success": summary["B2"]["success_count"], "mean_G": summary["B2"]["mean_discounted_return"]},
            "Full": {"valid": summary["Full"]["valid_episodes"], "success": summary["Full"]["success_count"], "mean_G": summary["Full"]["mean_discounted_return"]},
        },
        "new_rl_updates": 0, "completed_at": utc_now() if complete else status.get("completed_at"),
        "next_stage": "wait for explicit 2A authorization" if complete else "STAGE_1A_FINAL_TEST_ONLY running",
    })
    write_json(root / STATUS_PATH, status)
    return status


def mark_running(root, out):
    smoke = json.loads((root / "status/history/stage_1a.smoke_pass.820a8df.json").read_text(encoding="utf-8"))
    smoke.update({
        "execution_scope": "STAGE_1A_FINAL_TEST_ONLY", "training_complete": True, "smoke_passed": True,
        "final_test_complete": False, "status": "RUNNING",
        "final_test_dir": str(out.relative_to(root)).replace("\\", "/"), "issues": [],
        "next_stage": "STAGE_1A_FINAL_TEST_ONLY",
    })
    write_json(root / STATUS_PATH, smoke)
    report = (root / "reports/history/stage_1a_summary.smoke_pass.820a8df.md").read_text(encoding="utf-8")
    (root / REPORT_PATH).write_text(report.rstrip() + "\n\n## Final test-ID completion\n\nStatus: RUNNING. execution_scope=STAGE_1A_FINAL_TEST_ONLY. training_complete=true. smoke_passed=true. final_test_complete=false.\n", encoding="utf-8")


def cmd_stage_1a_v11_final_eval(root, phase="all", method=None, gpu=0):
    root = Path(root)
    os.chdir(root)
    if not (root / STOP_REL).is_file():
        raise BindingError("old Stage 2A STOP guard missing")
    out, cur = load_or_init_work(root)
    if phase in ("freeze", "all"):
        freeze_selection(root, out, cur)
    if phase in ("register", "all"):
        register_test_ids(root, out)
    if phase in ("materialize", "all"):
        materialize_caches(root, out, gpu=gpu)
        mark_running(root, out)
    if phase == "eval":
        if method not in ("B2", "Full"):
            raise BindingError("--method required for eval")
        evaluate_method(root, out, method, gpu=gpu)
        return {"status": "RUNNING", "method": method}
    if phase in ("report",):
        summary, complete = summarize(root, out)
        update_stage_outputs(root, out, summary, complete)
        return {"status": "PASS" if complete else "RUNNING", "final_test_complete": complete, "work_dir": str(out)}
    return {"status": "RUNNING", "work_dir": str(out), "stamp": cur["stamp"], "configsha8": cur["configsha8"]}
