#!/usr/bin/env python
"""Phase C worker (CP-DISR-TB-EVAL-PREP-LITE-01): ONE frozen dev episode (T_B_dev_00) through the new recorder.

--mode current   : current-source execution class (this worktree), derived no-prior dev10 split + derived runtime manifest.
--mode archived  : archived execution class = the v2_1 worktree source recorded in the historical manifests
                   (the recorder is loaded by file path from this worktree; no archived file is modified).
Never touches the test split, a test cache or relation truth.  provider = 0, optimizer = 0.
The old-record comparison covers ONLY fields that exist in the old eval row: success, G, steps, reason, success_seconds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

CASE = "T_B_dev_00"  # frozen; never changed
COMPARE = ("success", "G", "steps", "reason", "success_seconds")


def load_recorder(path):
    spec = importlib.util.spec_from_file_location("final_tb_eval_record_by_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def internal_checks(row):
    acts = row["actions"]
    chk = {}
    chk["n_actions_equals_steps"] = len(acts) == row["steps"]
    chk["action_clock_chain_continuous"] = all(abs(b["clock_start"] - a["clock_end"]) < 1e-9 for a, b in zip(acts, acts[1:]))
    chk["first_action_starts_at_episode_start"] = bool(acts) and abs(acts[0]["clock_start"] - row["episode_start_clock_s"]) < 1e-9
    if row["success"]:
        i = row["first_confirmed_success_decision_index"]
        covered = sum(a["duration"] for a in acts if a["decision_index"] <= i)
        chk["time_to_success_equals_durations_plus_gap"] = abs(covered + (row["inter_step_clock_gap_s"] or 0.0) - row["time_to_first_confirmed_success_s"]) < 1e-9
        chk["time_to_success_equals_last_confirmed_clock_end_minus_start"] = abs(acts[i]["clock_end"] - row["episode_start_clock_s"] - row["time_to_first_confirmed_success_s"]) < 1e-9
        chk["last_action_duration_equals_old_success_seconds"] = abs(acts[-1]["duration"] - row["success_seconds"]) < 1e-9
        chk["exactly_one_first_success_index"] = sum(1 for a in acts if a["success_confirmed"]) >= 1 and acts[i]["success_confirmed"] and not any(a["success_confirmed"] for a in acts[:i])
        chk["time_to_success_gt_old_success_seconds_when_multistep"] = (row["steps"] == 1) or row["time_to_first_confirmed_success_s"] > row["success_seconds"]
    chk["episode_end_equals_last_clock_end_minus_start"] = bool(acts) and abs(acts[-1]["clock_end"] - row["episode_start_clock_s"] - row["episode_end_elapsed_s"]) < 1e-9
    chk["all_internal_checks_pass"] = all(chk.values())
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("current", "archived"), required=True)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--old-eval", required=True, help="existing eval_final.json of this run (old record, dev10)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--recorder", required=True)
    ap.add_argument("--split", required=True, help="no-prior derived dev10 split (no cache pointers)")
    ap.add_argument("--runtime-manifest", default=None, help="current mode only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # resolve every path BEFORE chdir (an earlier archived attempt lost its result to a relative --out after chdir)
    for k in ("checkpoint", "old_eval", "root", "recorder", "split", "runtime_manifest", "out"):
        if getattr(a, k):
            setattr(a, k, str(Path(getattr(a, k)).resolve()))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    probe = Path(a.out) / ".write_probe"
    probe.write_text("ok")  # pre-flight: the result file must be writable before any environment is built
    probe.unlink()
    t0 = time.time()
    os.environ["MUJOCO_GL"] = "egl"
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        os.environ.pop(k, None)
    root = Path(a.root).resolve()
    os.chdir(str(root))
    rec = load_recorder(a.recorder)
    import torch
    from cp_disr import stage2a_v11 as v11
    assert torch.cuda.device_count() == 1, "worker must see exactly one GPU"
    device = torch.device("cuda", 0)
    split_doc = json.loads(Path(a.split).read_text(encoding="utf-8"))
    assert not split_doc.get("test"), "dev check must not use a test split"
    index = {r["case_id"]: r for r in split_doc["train"] + split_doc["dev"]}
    assert CASE in index and CASE in [r["case_id"] for r in split_doc["dev"]]
    v11.ENABLED_SPLITS = dict(v11.ENABLED_SPLITS)
    v11.ENABLED_SPLITS["T_B"] = Path(a.split)
    gate = rec.no_prior_case_gate
    if a.mode == "current":
        ctx = SimpleNamespace(runtime_manifest=a.runtime_manifest, train_split=a.split, render_gpu_device_id=0, prior_mode="absent")
        v11.bind_run_context(ctx)
    else:
        assert getattr(v11, "RUN_CONTEXT", None) is None and not hasattr(v11, "bind_run_context"), "archived class must be the pre-run-context source"
    files = {
        "recorder": a.recorder, "stage2a_v11": root / "src/cp_disr/stage2a_v11.py", "collector": root / "src/cp_disr/collector.py",
        "neural": root / "src/cp_disr/neural.py", "torch_rl": root / "src/cp_disr/torch_rl.py",
        "runtime_factory": root / "src/cp_disr/platforms/libero/runtime_factory.py", "rl": root / "src/cp_disr/rl.py",
        "skill_executor": root / "src/cp_disr/platforms/libero/skill_executor.py", "split_used": a.split,
        "runtime_manifest_used": a.runtime_manifest or (root / v11.RUNTIME_REL),
    }
    hashes = rec.source_identity(files)
    out_eval = Path(a.out) / ("eval_row_v2_%s_%s.json" % (a.mode, CASE))
    payload = rec.evaluate_checkpoint_recorded(
        root, "T_B", a.method, a.checkpoint, [CASE], index, device, hashes, out_eval, label="dev_consistency_T_B_dev_00",
        v11=v11, expected_sha256=a.expected_sha256, case_gate=gate,
        extra_identity={"execution_class": a.mode, "slot": a.slot, "root": str(root)})
    new = payload["rows"][0]
    old_doc = json.loads(Path(a.old_eval).read_text(encoding="utf-8"))
    old = next(r for r in old_doc["rows"] if r["case_id"] == CASE)
    cmp = {k: {"old": old.get(k), "new": new.get(k), "equal": old.get(k) == new.get(k)} for k in COMPARE}
    info = {"source_n": {"old": old.get("source_n"), "new": new.get("source_n"), "note": "informational; not in the comparable set"},
            "prior_mode": {"old": old.get("prior_mode"), "new": new.get("prior_mode")}}
    res = {
        "mode": a.mode, "slot": a.slot, "method": a.method, "case": CASE, "episodes_run": 1, "environment_constructions": 1,
        "provider_calls": 0, "optimizer_steps": payload["optimizer_steps"], "test_episodes": 0,
        "old_eval_file": a.old_eval, "old_eval_sha256": rec.sha256_file(a.old_eval),
        "old_record_comparison": cmp, "all_old_fields_equal": all(v["equal"] for v in cmp.values()), "informational": info,
        "internal_consistency": internal_checks(new), "new_fields": {k: new[k] for k in rec.NEW_ROW_FIELDS},
        "eval_row_v2_file": str(out_eval), "payload_status": payload["status"], "evaluated_model": payload["evaluated_model"],
        "evaluated_generation": payload["evaluated_generation"], "source_hashes": hashes,
        "seconds": round(time.time() - t0, 1), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    Path(a.out, "dev_check_%s.json" % a.mode).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"mode": a.mode, "old_equal": res["all_old_fields_equal"], "cmp": cmp, "internal": res["internal_consistency"]["all_internal_checks_pass"]}, indent=1))
    return 0 if res["all_old_fields_equal"] and res["internal_consistency"]["all_internal_checks_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
