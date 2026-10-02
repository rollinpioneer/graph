#!/usr/bin/env python3
"""Build the R-TB-E-1 (ELASTIC-01) result record from files only.

No environment, no GPU, no torch, no re-evaluation. It reads what the run wrote, re-hashes the checkpoint
files, and writes `e1_run_record.json` and `e1_run_record.md`.  Labels required by the authorization:
  * recorded `success_seconds`  -> FINAL_SKILL_DURATION_ONLY
  * episode total success time  -> UNVERIFIED
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

SUCCESS_SECONDS_LABEL = "FINAL_SKILL_DURATION_ONLY"
EPISODE_TOTAL_LABEL = "UNVERIFIED"
EVAL_POINTS = ("eval_n_000000", "eval_n_004096", "eval_n_008192", "eval_final")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def jlines(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def parse_utc(text):
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def _normalized_reasons(rows):
    """Reason families (the part before the first ':'), so per-episode ids do not hide the counts."""
    out = {}
    for x in rows:
        fam = str(x.get("reason")).split(":", 1)[0]
        out[fam] = out.get(fam, 0) + 1
    return dict(sorted(out.items()))


def eval_point(run, name):
    doc = jload(run / (name + ".json"))
    if doc is None:
        return {"point": name, "present": False}
    rows = doc.get("rows", [])
    ckpt = Path(doc["checkpoint"])
    local = run / "checkpoints" / ckpt.name
    return {"point": name, "present": True, "success_n": doc["success_n"], "n": doc["n"], "success_rate": doc["success_rate"],
            "mean_discounted_return": doc["mean_discounted_return"], "eval_action": doc.get("eval_action"),
            "isolated_rng": doc.get("isolated_rng"), "optimizer_steps_during_eval": doc.get("optimizer_steps"),
            "checkpoint": ckpt.name, "checkpoint_sha256": sha256_file(local) if local.is_file() else None,
            "exit_reasons": {r: sum(1 for x in rows if x.get("reason") == r) for r in sorted({x.get("reason") for x in rows})},
            "exit_reasons_normalized": _normalized_reasons(rows),
            "steps_per_case": [x.get("steps") for x in rows],
            "success_seconds": [x.get("success_seconds") for x in rows if x.get("success")],
            "success_seconds_label": SUCCESS_SECONDS_LABEL, "episode_total_success_time": EPISODE_TOTAL_LABEL,
            "eval_json_sha256": sha256_file(run / (name + ".json"))}


def generations(run):
    out = []
    gdir = run / "persistence" / "generations"
    for g in sorted(gdir.iterdir()) if gdir.is_dir() else []:
        hashes = jload(g / "HASHES.json") or {}
        files = hashes.get("files", {})
        row = {"generation": g.name, "complete_marker": (g / "COMPLETE").exists(), "files_recorded": len(files), "files_match": True,
               "model_pt_sha256": files.get("model.pt")}
        for fname, want in files.items():
            p = g / fname
            if not p.is_file() or sha256_file(p) != want:
                row["files_match"] = False
        flat = run / "checkpoints" / (g.name + ".pt")
        row["checkpoints_dir_pt_sha256"] = sha256_file(flat) if flat.is_file() else None
        row["flat_pt_equals_generation_model_pt"] = (row["checkpoints_dir_pt_sha256"] == files.get("model.pt")) if flat.is_file() else None
        out.append(row)
    return out


def storage(launch):
    events = jlines(launch / "storage_guard.jsonl")
    counts = {}
    for e in events:
        counts[e["event"]] = counts.get(e["event"], 0) + 1
    samples = [e["free_bytes"] for e in events if "free_bytes" in e]
    return {"events": counts, "pause_events": [e for e in events if e["event"] == "SIGSTOP"],
            "resume_events": [e for e in events if e["event"] == "SIGCONT"],
            "hard_margin_breach_events": [e for e in events if e["event"] == "HARD_MARGIN_BREACH"],
            "min_free_bytes_over_logged_samples": min(samples) if samples else None,
            "min_free_gib_over_logged_samples": round(min(samples) / 2 ** 30, 3) if samples else None,
            "guard_summary": jload(launch / "storage_guard_summary.json"),
            "start_gate": (jload(launch / "launch_evidence.json") or {}).get("start_gate"),
            "thresholds": {"start_free_gib": 6, "hard_reserved_margin_gib": 2, "pause_below_gib": 3, "resume_at_gib": 5, "sampling_seconds": 5},
            "limits": ["the minimum is over the guard's own samples; free space between 5 s samples is not measured",
                       "the SAMPLE log is thinned (every 12th tick, or every tick while free < resume threshold)",
                       "other users share this filesystem; quota was NOT_MEASURED"]}


def summarize(run, launch, label):
    js = jload(run / "job_summary.json")
    manifest = jload(run / "manifest.json") or {}
    evidence = jload(launch / "launch_evidence.json") if launch else None
    out = {"label": label, "run_dir": str(run), "job_summary_present": js is not None}
    if js is None:
        return out
    gens = generations(run)
    fresh = {p.name: jload(p) for p in sorted(run.glob("fresh_load_*.json"))}
    events = jlines(run / "persistence" / "raw_events.jsonl")
    sel = jload(run / "checkpoint_selection.json")
    first = jload(run / "first_update_selfcheck.json") or {}
    wall = None
    if evidence and (run / "job_summary.json").is_file():
        wall = (dt.datetime.fromtimestamp((run / "job_summary.json").stat().st_mtime, dt.timezone.utc) - parse_utc(evidence["time"])).total_seconds()
    ts = [parse_utc(e["time"]) for e in events if e.get("time")]
    out.update({
        "plan_id": manifest.get("planned_id"), "run_id": manifest.get("run_id"), "task": js["task"], "method": js["method"],
        "training_seed": manifest.get("training_seed"), "prep_commit": (manifest.get("hashes") or {}).get("git_commit"),
        "hashes": manifest.get("hashes"),
        "actual_N_valid_transitions": js["valid_transitions"], "actual_T_interaction_seconds": js["interaction_seconds"],
        "Ncap": 16384, "Tcap": js.get("Tcap"), "H": js.get("H"), "d_ref": js.get("d_ref"), "stop_reason": js["stop_reason"],
        "complete_updates": js["complete_updates"], "fragment_updates": js["fragment_updates"],
        "complete_plus_fragment_updates": js["complete_updates"] + js["fragment_updates"],
        "optimizer_steps": js["optimizer_steps"], "train_success_episodes": js["train_success_episodes"],
        "NaN_n": js["NaN_n"], "hard_fail": js["hard_fail"], "first_update_ok": js["first_update_ok"],
        "first_update_selfcheck": {k: first.get(k) for k in ("n_transitions", "optimizer_steps_this_update", "param_changed", "init_param_changed")},
        "eval_points": [eval_point(run, p) for p in EVAL_POINTS],
        "selected_checkpoint": Path(js["selected_checkpoint"]).name, "selected_checkpoint_sha256":
            sha256_file(run / "checkpoints" / Path(js["selected_checkpoint"]).name),
        "checkpoint_selection_rule": (sel or {}).get("selection_rule"),
        "generations": gens, "generations_all_match": all(g["files_match"] and g["complete_marker"] for g in gens),
        "fresh_load": fresh, "fresh_load_all_ok": bool(fresh) and all(v and v.get("model") and v.get("adam") and v.get("fresh_process") for v in fresh.values()),
        "wall_seconds_launch_to_job_summary": wall, "wall_hours": round(wall / 3600, 3) if wall else None,
        "raw_events_first_utc": min(ts).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None,
        "raw_events_last_utc": max(ts).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None,
        "success_seconds_label": SUCCESS_SECONDS_LABEL, "episode_total_success_time": EPISODE_TOTAL_LABEL,
    })
    if launch:
        out["storage"] = storage(launch)
        state = jload(launch / "launch_state.json") or {}
        out["ledger"] = {"plans": {k: v.get("status") for k, v in (state.get("plans") or {}).items()},
                         "new_rl_attempts_used": state.get("new_rl_attempts_used"), "new_rl_attempts_cap": state.get("new_rl_attempts_cap"),
                         "elastic_slots": {k: (v or {}).get("status") for k, v in (state.get("elastic_slots") or {}).items()}}
        out["launch_evidence"] = evidence
    return out


EXECUTION_PATH_HASH_KEYS = ("stage2a_v11", "phase_a_v12_persistence", "collector", "torch_rl", "neural", "persistence", "runtime_factory",
                            "runtime_manifest_derived", "train_split_derived", "dev10_split_original")


def manifest_difference(launch):
    """The two resolved runtime manifests may differ only in the line naming the train-split file (a different launch dir)."""
    plan = jload(Path(launch) / "launch_plan.json") or {}
    base = Path(plan.get("base_out", "")) / "runtime_manifest_tb_resolved.yaml"
    new = Path(launch) / "runtime_manifest_tb_resolved.yaml"
    if not base.is_file() or not new.is_file():
        return {"checked": False}
    a, b = base.read_text(encoding="utf-8").splitlines(), new.read_text(encoding="utf-8").splitlines()
    diff = [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
    only_split_path = len(a) == len(b) and bool(diff) and all("train_split_tb_noprior.json" in a[i - 1] and "train_split_tb_noprior.json" in b[i - 1] for i in diff)
    identity = (jload(Path(launch) / "derived_inputs_identity.json") or {}).get("rows", {}).get("runtime_manifest_derived_normalized", {})
    return {"checked": True, "differing_lines": diff, "differs_only_in_train_split_path": only_split_path,
            "normalized_hash_identical_to_E0": identity.get("identical"), "note": "raw file hashes differ only because the train-split file lives in a different launch directory"}


def same_profile_check(e1, e0, launch=None):
    """Independent init and identical execution path/profile versus R-TB-E-0 (seed and plan are the only intended differences)."""
    h1, h0 = e1.get("hashes") or {}, e0.get("hashes") or {}
    init1 = next((p for p in e1["eval_points"] if p["point"] == "eval_n_000000"), {}).get("checkpoint_sha256")
    init0 = next((p for p in e0["eval_points"] if p["point"] == "eval_n_000000"), {}).get("checkpoint_sha256")
    return {"execution_path_hashes_equal": {k: h1.get(k) == h0.get(k) for k in EXECUTION_PATH_HASH_KEYS},
            "all_execution_path_hashes_equal": all(h1.get(k) == h0.get(k) for k in EXECUTION_PATH_HASH_KEYS),
            "runtime_manifest_derived": manifest_difference(launch) if launch else None,
            "all_equal_after_train_split_path_normalization": (
                all(h1.get(k) == h0.get(k) for k in EXECUTION_PATH_HASH_KEYS if k != "runtime_manifest_derived")
                and bool(launch) and bool(manifest_difference(launch).get("differs_only_in_train_split_path"))
                and manifest_difference(launch).get("normalized_hash_identical_to_E0") is True),
            "final_tb_hash_differs_as_intended": h1.get("final_tb") != h0.get("final_tb"),
            "git_commit_e1_vs_e0": [h1.get("git_commit"), h0.get("git_commit")],
            "frozen_values_equal": {k: e1.get(k) == e0.get(k) for k in ("H", "d_ref", "Tcap", "Ncap")},
            "initial_checkpoint_sha256": {"E1": init1, "E0": init0}, "initial_checkpoint_differs_from_E0": bool(init1 and init0 and init1 != init0),
            "training_seed": {"E1": e1.get("training_seed"), "E0": e0.get("training_seed")}}


def e0_row(e0):
    return {k: e0.get(k) for k in ("run_id", "training_seed", "actual_N_valid_transitions", "actual_T_interaction_seconds", "complete_updates",
                                   "fragment_updates", "optimizer_steps", "train_success_episodes", "NaN_n", "stop_reason")} | {
        "dev10": [(p["point"], "%s/%s" % (p.get("success_n"), p.get("n")), p.get("mean_discounted_return"), p.get("exit_reasons_normalized")) for p in e0["eval_points"]]}


def markdown(rec):
    r = rec["R-TB-E-1"]
    L = ["# R-TB-E-1 (B1-K+E, seed 1, ELASTIC-01) run record", ""]
    if not r.get("job_summary_present"):
        return "\n".join(L + ["job_summary.json is not present: the run did not finish.", ""])
    L += ["Generated from files only (no environment, no GPU). Frozen T_B profile, empty R, provider 0, T_P 0, formal test 0.", "",
          "## Training", "",
          "actual N = %d valid transitions, actual T = %.2f s (Ncap 16384, Tcap %.2f); stop reason %s." % (
              r["actual_N_valid_transitions"], r["actual_T_interaction_seconds"], r["Tcap"], r["stop_reason"]),
          "complete updates %d + fragment updates %d; optimizer steps %d; training success episodes %d; NaN_n %d; hard_fail %s; first_update_ok %s." % (
              r["complete_updates"], r["fragment_updates"], r["optimizer_steps"], r["train_success_episodes"], r["NaN_n"], r["hard_fail"], r["first_update_ok"]),
          "wall time (launch to job_summary): %s h." % r.get("wall_hours"), "",
          "## Frozen dev10 (deterministic argmax)", "", "| point | success | mean discounted return | checkpoint sha256 |", "|---|---|---|---|"]
    for p in r["eval_points"]:
        L.append("| %s | %s/%s | %.6f | %s |" % (p["point"], p.get("success_n"), p.get("n"), p.get("mean_discounted_return", float("nan")),
                                                   (p.get("checkpoint_sha256") or "")[:16] + "..."))
    L += ["", "success_seconds in the eval rows: %s (final-skill duration only); episode total success time: %s." % (SUCCESS_SECONDS_LABEL, EPISODE_TOTAL_LABEL), "",
          "## Checkpoints", "", "selected checkpoint %s sha256 %s (rule %s)." % (r["selected_checkpoint"], r["selected_checkpoint_sha256"], r["checkpoint_selection_rule"]),
          "%d generations; all file hashes and COMPLETE markers match: %s. Fresh-process loads all ok: %s (%s)." % (
              len(r["generations"]), r["generations_all_match"], r["fresh_load_all_ok"], ", ".join(sorted(r["fresh_load"]))), ""]
    s = r.get("storage")
    if s:
        L += ["## Storage guard", "", "events: %s." % json.dumps(s["events"]), "pause (SIGSTOP) events: %d; resume (SIGCONT) events: %d; HARD_MARGIN_BREACH events: %d." % (
            len(s["pause_events"]), len(s["resume_events"]), len(s["hard_margin_breach_events"])),
              "minimum free over logged samples: %s GiB. %s" % (s["min_free_gib_over_logged_samples"], "; ".join(s["limits"])), ""]
    if r.get("ledger"):
        L += ["## Ledger", "", json.dumps(r["ledger"]), ""]
    sp = rec["same_profile_check_vs_E0"]
    L += ["## Same profile / independent initialisation versus R-TB-E-0", "",
          "execution-path hashes equal to R-TB-E-0 (raw): %s; equal after normalizing only the train-split path in the resolved runtime manifest: %s (%s); frozen H/d_ref/Tcap/Ncap equal: %s; final_tb.py hash differs (intended: registration row and cap): %s." % (
              sp["all_execution_path_hashes_equal"], sp["all_equal_after_train_split_path_normalization"], json.dumps(sp.get("runtime_manifest_derived")),
              all(sp["frozen_values_equal"].values()), sp["final_tb_hash_differs_as_intended"]),
          "initial checkpoint (n_000000) differs from R-TB-E-0's: %s (E1 %s, E0 %s); training seed E1 %s, E0 %s." % (
              sp["initial_checkpoint_differs_from_E0"], (sp["initial_checkpoint_sha256"]["E1"] or "")[:16], (sp["initial_checkpoint_sha256"]["E0"] or "")[:16],
              sp["training_seed"]["E1"], sp["training_seed"]["E0"]), ""]
    L += ["## Why the 0/10 points failed (exit-reason families, normalized)", ""]
    for p in r["eval_points"]:
        L.append("- %s: %s" % (p["point"], json.dumps(p["exit_reasons_normalized"])))
    L += [""]
    L += ["## Descriptive context only", "",
          "Base run R-TB-E-0 (same method, seed 0): " + json.dumps(rec["R-TB-E-0_descriptive"], ensure_ascii=False),
          "Two seeds are not a reliability estimate; no comparison with B2 and no change to Method 2.1.1 follows from this record.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--launch", required=True)
    ap.add_argument("--e0-run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"document": "e1_run_record", "card": "CP-DISR-TB-E1-ELASTIC-01", "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "R-TB-E-1": summarize(Path(a.run), Path(a.launch), "R-TB-E-1"),
           "labels": {"success_seconds": SUCCESS_SECONDS_LABEL, "episode_total_success_time": EPISODE_TOTAL_LABEL}}
    e0_full = summarize(Path(a.e0_run), None, "R-TB-E-0")
    rec["R-TB-E-0_descriptive"] = e0_row(e0_full)
    rec["same_profile_check_vs_E0"] = same_profile_check(rec["R-TB-E-1"], e0_full, a.launch)
    (out / "e1_run_record.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "e1_run_record.md").write_text(markdown(rec), encoding="utf-8")
    print(json.dumps({"written": [str(out / "e1_run_record.json"), str(out / "e1_run_record.md")],
                      "job_summary_present": rec["R-TB-E-1"]["job_summary_present"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
