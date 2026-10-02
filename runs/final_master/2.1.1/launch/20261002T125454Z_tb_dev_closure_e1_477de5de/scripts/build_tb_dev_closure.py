#!/usr/bin/env python3
"""T_B development-evidence closure builder (card: T_B dev closure with R-TB-E-1).

Zero environment: reads files only. No episode is run, no model is loaded (torch is not imported),
no test cache / relation truth / test result is opened, no scene is generated. Intermediate checkpoints are not
re-hashed (their recorded hashes are cited); only the final checkpoint files are re-hashed (bytes, not loaded).
Writes new summary material into --out and never touches the original run records.
"""
import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

LABEL_SS = "FINAL_SKILL_DURATION_ONLY"
LABEL_EPISODE_TOTAL = "UNVERIFIED"
PTS = (("eval_n_000000", "0"), ("eval_n_004096", "4096"), ("eval_n_008192", "8192"), ("eval_final", "final"))
NOT_CLAIMED_BEGIN = "<!-- NOT_CLAIMED_BEGIN -->"
NOT_CLAIMED_END = "<!-- NOT_CLAIMED_END -->"
EXEC_HASH_KEYS = ("collector", "persistence", "neural", "torch_rl", "runtime_factory", "stage2a_v11")


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


def one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit("expected exactly one match for %s, got %s" % (pattern, hits))
    return Path(hits[0])


def fam(reason):
    return str(reason).split(":", 1)[0]


def reasons_norm(rows):
    out = {}
    for r in rows:
        k = fam(r.get("reason"))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gen_info(run, stem):
    g = Path(run) / "persistence" / "generations" / stem
    man = jload(g / "manifest.json") or {}
    hs = (jload(g / "HASHES.json") or {}).get("files", {})
    return {"generation": stem, "complete_marker": (g / "COMPLETE").exists(), "N": man.get("N"), "T": man.get("T"),
            "update": man.get("update"), "complete_updates": man.get("complete_updates"), "fragment_updates": man.get("fragment_updates"),
            "model_pt_sha256_recorded": hs.get("model.pt"), "files_recorded": sorted(hs)}


def point_record(run, name, label):
    doc = jload(Path(run) / (name + ".json"))
    ck = Path(doc["checkpoint"]).name
    stem = ck[:-3] if ck.endswith(".pt") else ck
    rows = doc.get("rows", [])
    ok = [x["success_seconds"] for x in rows if x.get("success") and x.get("success_seconds") is not None]
    gi = gen_info(run, stem)
    return {"point": label, "eval_file": name + ".json", "eval_label": doc.get("label"), "update": gi["update"], "N_actual": gi["N"], "T_actual_s": gi["T"],
            "success_n": doc["success_n"], "n": doc["n"], "mean_discounted_return": doc["mean_discounted_return"],
            "exit_reason_families": reasons_norm(rows), "checkpoint": ck, "checkpoint_sha256_recorded": gi["model_pt_sha256_recorded"],
            "generation_complete": gi["complete_marker"], "recorded_checkpoint_path": doc["checkpoint"],
            "mean_recorded_success_seconds": (sum(ok) / len(ok)) if ok else None, "success_seconds_label": LABEL_SS,
            "case_ids": [x["case_id"] for x in rows], "eval_json_sha256": sha256_file(Path(run) / (name + ".json"))}


def collect(spec, rehash_final=True):
    run = Path(spec["dir"])
    js = jload(run / "job_summary.json")
    man = jload(run / "manifest.json") or {}
    pts = [point_record(run, n, lab) for n, lab in PTS]
    finals = sorted((run / "checkpoints").glob("final_n_*_u*.pt"))
    if len(finals) != 1:
        raise SystemExit("%s: expected one final checkpoint, got %s" % (spec["id"], finals))
    fin = finals[0]
    stem = fin.name[:-3]
    gi = gen_info(run, stem)
    latest = (run / "persistence" / "LATEST").read_text().strip() if (run / "persistence" / "LATEST").is_file() else None
    fresh = jload(run / ("fresh_load_%s.json" % stem))
    sel_name = Path(js["selected_checkpoint"]).name
    sel = jload(run / "checkpoint_selection.json") or {}
    dev20 = jload(run / "common_dev20.json")
    final_sha = sha256_file(fin) if rehash_final else None
    c = {
        "id": spec["id"], "model_slot": spec.get("slot"), "origin": spec["origin"], "method": js["method"], "task": js["task"], "seed": man.get("training_seed", man.get("seed")),
        "planned_id": js.get("planned_id", man.get("planned_id")), "run_id": js.get("run_id", man.get("run_id")), "run_dir": str(run), "recorded_job_dir": js.get("job_dir"),
        "manifest_hashes": man.get("hashes"), "H": js.get("H"), "d_ref": js.get("d_ref"), "Tcap": js.get("Tcap"),
        "actual_N": js["valid_transitions"], "actual_T_s": js["interaction_seconds"], "stop_reason": js["stop_reason"],
        "complete_updates": js["complete_updates"], "fragment_updates": js["fragment_updates"], "optimizer_steps": js["optimizer_steps"],
        "train_success_episodes": js["train_success_episodes"], "NaN_n": js["NaN_n"], "hard_fail": js["hard_fail"],
        "points": pts,
        "final": {"checkpoint_file": str(fin), "checkpoint_name": fin.name, "bytes": fin.stat().st_size, "sha256_recomputed_now": final_sha,
                  "generation": gi, "persistence_LATEST": latest, "fresh_load_record": fresh,
                  "checks": {
                      "latest_is_final_generation": latest == stem,
                      "generation_complete_marker": gi["complete_marker"],
                      "generation_N_equals_job_summary": gi["N"] == js["valid_transitions"],
                      "generation_T_equals_job_summary": gi["T"] == js["interaction_seconds"],
                      "generation_update_equals_complete_updates": gi["update"] == js["complete_updates"],
                      "recorded_model_sha_equals_recomputed": (gi["model_pt_sha256_recorded"] == final_sha) if rehash_final else None,
                      "eval_final_checkpoint_is_final": pts[3]["checkpoint"] == fin.name,
                      "fresh_load_ok": bool(fresh and fresh.get("model") and fresh.get("adam") and fresh.get("fresh_process") and fresh.get("generation") == stem),
                  }},
        "selected_checkpoint_by_job_rule": {"name": sel_name, "rule": sel.get("selection_rule"), "is_final": sel_name == fin.name,
                                           "role": "SEPARATE_SELECTED_COLUMN_NOT_A_SUBSTITUTE_FOR_FINAL"},
        "common_dev20_selection": ({"checkpoint": Path(dev20["checkpoint"]).name, "success_n": dev20["success_n"], "n": dev20["n"],
                                    "mean_discounted_return": dev20["mean_discounted_return"], "label": dev20.get("label")} if dev20 else None),
    }
    c["final"]["all_checks_true"] = all(v is True for v in c["final"]["checks"].values() if v is not None)
    return c


def registry(fm, hist):
    fm, hist = Path(fm), Path(hist)
    return [
        {"id": "R2-B0-s0", "slot": "TB-M1", "origin": "historical R2 (v13_r2), original old-account run, copy under graph_cp_disr_v2_1",
         "dir": str(hist / "v13_r2/T_B/B0/seed_0/20260927T070500Z_826fe5c5"), "base": True},
        {"id": "R1-B1K-s0", "slot": "TB-M2", "origin": "historical R1 (v13_r1)", "dir": str(hist / "v13_r1/T_B/B1-K/seed_0/20260926T112613Z_9e64dd09"), "base": True},
        {"id": "R-TB-K-1", "slot": "TB-M3", "origin": "final_master 2.1.1 (prep cac2fa5b)", "dir": str(one(str(fm / "T_B/B1-K/seed_1/R-TB-K-1-*"))), "base": True},
        {"id": "R1-B2-s0", "slot": "TB-M4", "origin": "historical R1 (v13_r1)", "dir": str(hist / "v13_r1/T_B/B2/seed_0/20260926T112613Z_9e64dd09"), "base": True},
        {"id": "R-TB-DK-1", "slot": "TB-M5", "origin": "final_master 2.1.1 (prep cac2fa5b)", "dir": str(one(str(fm / "T_B/B2/seed_1/R-TB-DK-1-*"))), "base": True},
        {"id": "R-TB-E-0", "slot": "TB-M6", "origin": "final_master 2.1.1 (prep cac2fa5b)", "dir": str(one(str(fm / "T_B/B1-K+E/seed_0/R-TB-E-0-*"))), "base": True},
        {"id": "R-TB-E-1", "slot": "TB-C1", "origin": "final_master 2.1.1 (prep e4dc34ae, ELASTIC-01, conditional)", "dir": str(one(str(fm / "T_B/B1-K+E/seed_1/R-TB-E-1-*"))), "base": False},
    ]


CURRENT = ("R-TB-E-0", "R-TB-DK-1", "R-TB-K-1", "R-TB-E-1")
HIST_IDS = ("R1-B1K-s0", "R1-B2-s0", "R2-B0-s0")


def label_of(c):
    return "%s %s s%s" % (c["id"], c["method"], c["seed"])


def crosscheck_base_csv(base_csv, runs):
    """Existing comparison base (E-0, DK-1, K-1): success, return and checkpoint hash must equal the re-read values."""
    rows = list(csv.DictReader(open(base_csv, encoding="utf-8", newline="")))
    out, ok = [], True
    for r in rows:
        c = runs.get(r["plan"])
        if c is None:
            continue
        p = next(x for x in c["points"] if x["eval_file"] == r["eval_file"])
        same = (int(r["success_n"]) == p["success_n"] and abs(float(r["mean_discounted_return"]) - p["mean_discounted_return"]) < 1e-12
                and r["checkpoint_sha256"] == p["checkpoint_sha256_recorded"] and int(r["skill_transitions_N"]) == p["N_actual"])
        ok = ok and same
        out.append({"plan": r["plan"], "eval_file": r["eval_file"], "equal": same})
    return {"rows_compared": len(out), "all_equal": ok and len(out) == 12, "detail": out}


def fmt_ret(x):
    return "%.6f" % x


def fmt_n(x):
    return "%d" % x


def fmt_t(x):
    return "%.2f" % x


def md_tables(runs, order, title_prefix):
    L = []
    hdr = "| run | 0 | 4096 | 8192 | final |"
    sep = "|---|---|---|---|---|"
    L += ["**%s success (n of 10, frozen dev10, deterministic argmax)**" % title_prefix, "", hdr, sep]
    for k in order:
        c = runs[k]
        L.append("| %s | %s |" % (label_of(c), " | ".join("%d/%d" % (p["success_n"], p["n"]) for p in c["points"])))
    L += ["", "**%s mean discounted return (frozen H = 23.1 s)**" % title_prefix, "", hdr, sep]
    for k in order:
        c = runs[k]
        L.append("| %s | %s |" % (label_of(c), " | ".join(fmt_ret(p["mean_discounted_return"]) for p in c["points"])))
    L += ["", "**%s actual N / T at each evaluation point** (N = valid skill transitions, T = interaction seconds recorded in the generation manifest; update = update index of the evaluated generation)" % title_prefix, "",
          hdr, sep]
    for k in order:
        c = runs[k]
        L.append("| %s | %s |" % (label_of(c), " | ".join("u%s: N %s / T %s" % (p["update"], fmt_n(p["N_actual"]), fmt_t(p["T_actual_s"])) for p in c["points"])))
    return L


def md_long(runs, order):
    L = ["| run | point | update | N actual | T actual (s) | success | mean disc. return | recorded success_seconds mean [%s] | exit-reason families | checkpoint (sha256 first 12, recorded) |" % LABEL_SS,
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        c = runs[k]
        for p in c["points"]:
            ss = "NA" if p["mean_recorded_success_seconds"] is None else "%.3f" % p["mean_recorded_success_seconds"]
            L.append("| %s | %s | %s | %s | %s | %d/%d | %.6f | %s | %s | %s (%s) |" % (
                c["id"], p["point"], p["update"], fmt_n(p["N_actual"]), fmt_t(p["T_actual_s"]), p["success_n"], p["n"], p["mean_discounted_return"], ss,
                json.dumps(p["exit_reason_families"]), p["checkpoint"], (p["checkpoint_sha256_recorded"] or "")[:12]))
    return L


def md_budget(runs, order):
    L = ["| run | N actual (Ncap 16384) | T actual s (Tcap 68812.8) | stop | complete + fragment updates | optimizer steps | train success episodes | NaN_n | hard_fail |", "|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        c = runs[k]
        L.append("| %s | %d | %.2f | %s | %d + %d | %d | %d | %d | %s |" % (label_of(c), c["actual_N"], c["actual_T_s"], c["stop_reason"], c["complete_updates"], c["fragment_updates"],
                                                                          c["optimizer_steps"], c["train_success_episodes"], c["NaN_n"], c["hard_fail"]))
    return L


def descriptive_statements(runs):
    """Every statement is asserted against the data before it is written; a failed assertion aborts the build."""
    def succ(k):
        return [p["success_n"] for p in runs[k]["points"]]
    s = []
    dk, e0, e1, k1 = succ("R-TB-DK-1"), succ("R-TB-E-0"), succ("R-TB-E-1"), succ("R-TB-K-1")
    assert dk[1] == 10 and dk[2] == 10 and dk[3] == 10 and dk[0] == 0
    s.append("B2 seed 1 (R-TB-DK-1) was 10/10 on frozen dev10 at N = 4096 and N = 8192 (and at final).")
    assert e0 == [0, 0, 0, 10] and e1 == [0, 0, 0, 10]
    s.append("B1-K+E seed 0 (R-TB-E-0) and seed 1 (R-TB-E-1) were 0/10 at N = 4096 and N = 8192 and 10/10 at final.")
    assert k1 == [0, 0, 0, 0]
    s.append("B1-K seed 1 (R-TB-K-1) was 0/10 at all four evaluation points.")
    b2h = succ("R1-B2-s0")
    assert b2h == [0, 0, 10, 10]
    s.append("Historical B2 seed 0 (R1, original execution identity, listed apart) was 0/10 at N = 4096 and 10/10 at N = 8192 and at final.")
    assert succ("R1-B1K-s0") == [0, 0, 0, 0] and succ("R2-B0-s0") == [0, 0, 0, 0]
    s.append("Historical B1-K seed 0 (R1) and B0 seed 0 (R2) were 0/10 at all four evaluation points.")
    return s


NOT_CLAIMED = [
    "No learning-speed ratio or multiple is stated.",
    "No statistical-significance statement is made (one or two seeds per method, ten dev cases).",
    "No statement of method equivalence and none of general superiority of any method is made.",
    "The 4096 / 8192 points are pre-registered learning-process evidence on dev10; the registered primary endpoint stays the final point.",
    "No change to Method 2.1.1, the training configuration, the profile or any budget follows from these numbers.",
]


PLAN_V3_HIST_VALUES = {  # Plan v3 section 1 table: N, complete+fragment updates
    "R1-B1K-s0": (13623, 13, 1), "R1-B2-s0": (14464, 14, 1), "R2-B0-s0": (14705, 14, 1)}


def git_out(repo, *args):
    try:
        return subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return "ERROR: %s" % exc


def probe(path):
    try:
        os.stat(path)
        return "EXISTS_AND_READABLE"
    except PermissionError:
        return "PERMISSION_DENIED"
    except FileNotFoundError:
        return "MISSING"


def historical_identity(runs, hist_repo):
    cur = runs["R-TB-E-1"]["manifest_hashes"]
    cur_dev10_eval = runs["R-TB-E-1"]["points"][3]["case_ids"]
    out = {}
    for k in HIST_IDS:
        c = runs[k]
        mh = c["manifest_hashes"] or {}
        rel = str(Path(c["run_dir"]).relative_to(hist_repo))
        eq = {key: (mh.get(key) == cur.get(key)) for key in EXEC_HASH_KEYS if mh.get(key) is not None and cur.get(key) is not None}
        N, cu, fu = PLAN_V3_HIST_VALUES[k]
        out[k] = {
            "slot": c["model_slot"], "method": c["method"], "seed": c["seed"], "planned_id": c["planned_id"], "run_id": c["run_id"],
            "local_run_dir": c["run_dir"], "job_dir_recorded_in_job_summary": c["recorded_job_dir"],
            "recorded_location_differs_from_local": c["recorded_job_dir"] != c["run_dir"],
            "original_location_probe": ({"path": c["recorded_job_dir"], "result": probe(c["recorded_job_dir"] + "/job_summary.json")} if c["recorded_job_dir"] != c["run_dir"] else None),
            "recorded_source_commits": {kk: mh.get(kk) for kk in ("git_commit", "base_commit", "baseline_commit") if mh.get(kk)},
            "source_hashes_recorded_in_manifest": {kk: mh.get(kk) for kk in EXEC_HASH_KEYS + ("split", "split_dev10", "runtime_manifest", "clock", "safety", "skill_executor") if mh.get(kk)},
            "versus_current_2.1.1_execution_path_E1": {"equal": sorted(x for x, v in eq.items() if v), "differ": sorted(x for x, v in eq.items() if not v)},
            "frozen_values_equal_to_current": {"H": c["H"] == runs["R-TB-E-1"]["H"], "d_ref": c["d_ref"] == runs["R-TB-E-1"]["d_ref"], "Tcap": c["Tcap"] == runs["R-TB-E-1"]["Tcap"]},
            "dev10_split_hash_equal_to_current": mh.get("split_dev10") == cur.get("dev10_split_original"),
            "dev10_case_ids_equal_to_current": c["points"][3]["case_ids"] == cur_dev10_eval,
            "plan_v3_section1_values": {"expected_N": N, "expected_complete": cu, "expected_fragment": fu,
                                        "matches": (c["actual_N"], c["complete_updates"], c["fragment_updates"]) == (N, cu, fu)},
            "actual": {"N": c["actual_N"], "T_s": c["actual_T_s"], "complete": c["complete_updates"], "fragment": c["fragment_updates"], "stop": c["stop_reason"],
                       "optimizer_steps": c["optimizer_steps"], "train_success_episodes": c["train_success_episodes"]},
            "final_checkpoint_checks": c["final"]["checks"],
            "git_tracking_in_historical_repo": {"tracked_files_under_run_dir": len([x for x in git_out(hist_repo, "ls-files", rel).splitlines() if x]),
                                                  "last_commit_touching_run_dir": git_out(hist_repo, "log", "-1", "--format=%H %s", "--", rel)},
        }
    return out


COMPAT_EVIDENCE_CITED = [
    {"id": "CE1", "source": "docs/authoritative/CP_DISR_Final_Experimental_Plan_v3.md section 1 (rows T_B R1 B1-K/B2, T_B R2 B0)",
     "says": "keep original N/T and model identity; B0 keeps the actual old-account deviation; do not write it as an xushijie2 run, do not discard or auto-retrain"},
    {"id": "CE2", "source": "Plan v3 section 1.2 and section 12.2",
     "says": "no automatic resumption of historical obligations; when historical weights/inputs cannot be shown usable, no sidecar and no automatic retraining; the gap stays recorded and goes to the elastic / manual decision"},
    {"id": "CE3", "source": "docs/authoritative/CP_DISR_Final_Research_Content_v5.md lines 824-825 ([E1], [E2])",
     "says": "R1 ref 55a3b7ce..., runs/v13_r1/20260926T112613Z/; R2 ref eaa43e97..., runs/v13_r2/20260927T070500Z/, actual old-account run"},
    {"id": "CE4", "source": "R-TB-E-1 launch dir: smoke_compatibility_rebind.json and e1_run_record.json (same_profile_check_vs_E0)",
     "says": "the new runs R-TB-E-0/K-1/DK-1 (prep cac2fa5b) and R-TB-E-1 (prep e4dc34ae) share one execution path; E-1 differs from E-0 only in the train-split path line of the resolved manifest (normalized hash identical)"},
    {"id": "CE5", "source": "this record (file-level, no model loaded)",
     "says": "historical and current runs share H, d_ref, Tcap, Ncap, the dev10 split hash and the dev10 case ids, and the collector and persistence source hashes; they differ in neural, torch_rl, runtime_factory and stage2a_v11 source hashes"},
]

UNCONFIRMED_HISTORICAL = [
    {"id": "UC1", "item": "Whether the three historical final weights load and run under the current (2.1.1) neural/torch_rl/runtime code",
     "why": "source hashes of neural, torch_rl, runtime_factory and stage2a_v11 differ from the current execution path; each historical run only has its own fresh-load record from the code of its time",
     "status": "NOT_CHECKED", "handling": "listed apart; not averaged with new runs; no retraining; a load-only check (no episode) is requested in the sub-card, not run here"},
    {"id": "UC2", "item": "Whether the historical evaluation function (stage2a_v11 543cb10c...) and the current one (451fdbd9...) score identically",
     "why": "different source hash; no differential audit exists and a full-repository audit is out of scope",
     "status": "UNVERIFIED", "handling": "historical dev numbers stay as recorded under their own identity; test would use the single unified evaluator, which is the point of the unified evaluation card"},
    {"id": "UC3", "item": "R2 B0 s0 original location and byte-level provenance of the local copy",
     "why": "job_summary records job_dir under /home/__compress_data/xushijie/graph_cp_disr_v2_1_r2_b0/ (old account); the files read here are a copy under graph_cp_disr_v2_1/runs/v13_r2; the original was not readable from this account when this record was built (probe result in the JSON: permission denied)",
     "status": "ORIGINAL_NOT_ACCESSIBLE", "handling": "internal consistency only (generation HASHES.json vs recomputed final hash, fresh-load record); recorded as an actual old-account run, not rewritten as an xushijie2 run"},
    {"id": "UC4", "item": "Commit pointers differ between run manifests and the ledger text",
     "why": "R1 manifests record git_commit 0eb3776d... (base 66614acb...), R2 records git_commit 228517ed... (baseline 55a3b7ce...); Research Content v5 cites 55a3b7ce... for R1 and eaa43e97... for R2",
     "status": "NOT_RECONCILED", "handling": "each run keeps the identity recorded in its own manifest; no re-derivation"},
    {"id": "UC5", "item": "Weights-to-input chain for test release (Adam/model index, input hashes) of the historical runs",
     "why": "generation folders hold model.pt/model.json/episode.pkl/rng.json with HASHES.json, but the test_release_manifest (Plan 12.1 item 3) has not been built",
     "status": "NOT_BUILT", "handling": "built at release time from the frozen list; any item that cannot be proven stays a recorded gap"},
    {"id": "UC6", "item": "R2 B0 first start folder 20260927T065500Z_ac37e5e4",
     "why": "contains only n_000000 and no job_summary: an aborted start, not the run that produced N14705",
     "status": "EXCLUDED", "handling": "not part of any list; named only so it is not mistaken for a model"},
    {"id": "UC7", "item": "R2 B0 s0 `common_dev20.json` holds 10 cases, not 20",
     "why": "the file is labelled common_dev20 but records n = 10 (the same ten frozen dev10 case ids as eval_final.json), whereas the R1 B1-K / B2 files record n = 20 and Plan v3 section 1 reports a selected dev20 of 0/20 for B0",
     "status": "RECORDED_DIFFERENCE_NOT_RECONCILED", "handling": "the separate selected column shows the file as recorded (0/10 for B0); it is not used for any main-evaluation decision and no number is rewritten"},
]


def freeze_list(runs, repo_head):
    models = []
    for k in ("R2-B0-s0", "R1-B1K-s0", "R-TB-K-1", "R1-B2-s0", "R-TB-DK-1", "R-TB-E-0", "R-TB-E-1"):
        c = runs[k]
        f = c["final"]
        hist = k in HIST_IDS
        models.append({
            "slot": c["model_slot"], "plan_run": k, "task": c["task"], "method": c["method"], "seed": c["seed"],
            "list": "BASE_6x30" if k != "R-TB-E-1" else "CONDITIONAL_OUTSIDE_628",
            "origin": "historical" if hist else "final_master_2.1.1",
            "main_eval_checkpoint": {"rule": "last valid post-update checkpoint (final generation, after the fragment update)",
                                     "file": f["checkpoint_file"], "name": f["checkpoint_name"], "bytes": f["bytes"],
                                     "sha256": f["sha256_recomputed_now"], "generation": f["generation"]["generation"],
                                     "N": f["generation"]["N"], "T_s": f["generation"]["T"], "update_index": f["generation"]["update"],
                                     "complete_updates": c["complete_updates"], "fragment_updates": c["fragment_updates"],
                                     "recorded_model_pt_sha256": f["generation"]["model_pt_sha256_recorded"], "checks": f["checks"], "all_checks_true": f["all_checks_true"]},
            "selected_by_job_rule_separate_column": c["selected_checkpoint_by_job_rule"],
            "common_dev20_selection_separate_column": c["common_dev20_selection"],
            "status": {"file_and_hash_consistency": "OK" if f["all_checks_true"] else "CHECK_FAILED",
                       "load_under_current_2.1.1_code": "TRAINED_AND_FRESH_LOADED_ON_THE_2.1.1_PATH" if not hist else "NOT_CHECKED (see UC1)",
                       "identity_gaps": [] if not hist else (["UC1", "UC2", "UC4", "UC5"] + (["UC3", "UC7"] if k == "R2-B0-s0" else []))},
            "execution_identity": {"planned_id": c["planned_id"], "run_id": c["run_id"], "source_commits": (c["manifest_hashes"] or {}).get("git_commit"),
                                   "run_dir": c["run_dir"]},
        })
    return {"document": "frozen_main_evaluation_model_list_T_B", "frozen_at_utc": utc_now(), "repo_head_at_freeze": repo_head,
            "rules": [
                "R1 Main-evaluation model of every run = its last valid post-update checkpoint (the final_n_*_u* generation), the registered primary endpoint.",
                "R2 The checkpoint selected by the job rule (max_mean3_then_max_worst_then_earlier_N) and any best-dev / common-dev20 selection are listed in separate columns; they are never a substitute for the final model. A method whose selected checkpoint is n_000000 is evaluated at its final model, never at N = 0.",
                "R3 Intermediate evaluation points (N = 4096, 8192) are dev learning-process evidence, not members of this list.",
                "R4 Existing hash verification is reused: intermediate checkpoint hashes are cited from the runs' own generation HASHES.json and the earlier checkpoint_identity records; no intermediate model is loaded or re-hashed. Only the final checkpoint files are re-hashed (bytes only, no model load) as the freeze-time stat check.",
                "R5 A model whose usability cannot be shown stays listed with its gap; no sidecar stands in for it and no automatic retraining follows (Plan v3 12.2).",
                "R6 The list is frozen as a candidate list for the later release; it does not release any test, and the test_release_manifest is not built here."],
            "models": models,
            "excluded": [{"item": "R2 B0 first start folder 20260927T065500Z_ac37e5e4", "reason": "n_000000 only, no job_summary (UC6)"},
                         {"item": "all n_000000 / n_004096 / n_008192 checkpoints and all update_complete_* / update_fragment_1 generations", "reason": "R3, not main-evaluation models"}],
            "counts": {"base_models": 6, "base_episodes_if_released": 6 * 30, "conditional_models": 1, "conditional_episodes_if_approved": 30,
                       "plan_v3_note": "base T_B 6x30 = 180 is part of the 628 cap; conditional +E s1 is counted outside it (Plan v3 12.2)"}}


def provenance(repo, test_dir, vlm_cache_test, hist_repo, runs):
    repo, test_dir = Path(repo), Path(test_dir)
    scenes = sorted(test_dir.glob("T_B_test_*"))
    per, names_multiset = [], {}
    h_all = hashlib.sha256()
    for s in scenes:
        names = sorted(p.name for p in s.iterdir())
        for n in names:
            names_multiset[n] = names_multiset.get(n, 0) + 1
        hs = {n: sha256_file(s / n) for n in names}
        for n in sorted(hs):
            h_all.update(("%s/%s:%s\n" % (s.name, n, hs[n])).encode())
        per.append({"scene": s.name, "files": names, "sha256": hs})
    cfg = {}
    for rel in ("configs/splits/T_B_stage_2a_test30.json", "configs/splits/T_B_stage_2a_test_ids.json", "configs/splits/T_B_stage_2a_v11.json"):
        cfg[rel] = {"sha256": sha256_file(repo / rel)}
    ids = jload(repo / "configs/splits/T_B_stage_2a_test_ids.json") or {}
    active_ids = [x["case_id"] for x in ids.get("active", [])]
    cfg["declared_by_test_ids_file"] = {k: ids.get(k) for k in (
        "registration_timing", "frozen_before_main_training", "frozen_before_any_stage2a_test_model_result", "reserved_pool_count", "active_case_count",
        "shared_by_methods", "same_order", "same_reset_seed", "stream_omits_method_and_training_seed", "filtered_by_model_performance",
        "filtered_by_scripted_success", "filtered_by_nonempty_prior")}
    cfg["declared_by_test_ids_file"]["note"] = "declared by the file itself; not independently verified here"
    cfg["active_case_ids_first_last_count"] = [active_ids[0] if active_ids else None, active_ids[-1] if active_ids else None, len(active_ids)]
    cfg["active_ids_equal_scene_dirs"] = active_ids == [s.name for s in scenes]
    res = [x["case_id"] for x in ids.get("reserved", [])]
    cfg["reserved_unused_case_ids_first_last_count"] = [res[0] if res else None, res[-1] if res else None, len(res)]
    vlm = sorted(p.name for p in Path(vlm_cache_test).iterdir()) if Path(vlm_cache_test).is_dir() else []
    use = {}
    for k, c in runs.items():
        try:
            r = subprocess.run(["grep", "-rIl", "--exclude=*.pt", "--exclude=*.pkl", "--exclude=*.npy", "T_B_test", c["run_dir"]], capture_output=True, text=True, timeout=120)
            files = [Path(x).name for x in r.stdout.split()]
        except Exception as exc:  # pragma: no cover
            files = ["ERROR %s" % exc]
        use[k] = {"files_mentioning_T_B_test": files, "count": len(files)}
    split_tb = {}
    for rel in sorted(glob.glob(str(repo / "runs/final_master/2.1.1/launch/*/train_split_tb_noprior.json"))):
        d = jload(rel) or {}
        split_tb[Path(rel).parent.name] = {"test_field": d.get("test"), "train_n": len(d.get("train", [])), "dev_n": len(d.get("dev", []))}
    pat = ["test_metrics*", "final_test_summary*", "test_summary.json", "selection_manifest.json"]
    roots = sorted(glob.glob(str(Path(hist_repo).parent / "graph_cp_disr_*" / "runs")))
    cmd = ["find"] + roots + ["-type", "f", "("]
    for i, p in enumerate(pat):
        cmd += (["-o"] if i else []) + ["-name", p]
    cmd += [")"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=110)
        found = [x for x in r.stdout.splitlines() if x]
        complete = True
    except subprocess.TimeoutExpired:
        found, complete = [], False
    shape = {}
    for x in found:
        y = re.sub(r"^.*?/graph_cp_disr_[^/]+/", "", x)
        y = re.sub(r"[0-9]{8}T[0-9]{6}Z_[0-9a-f]+", "<stamp>", y)
        y = re.sub(r"/(B2|Full|B0|B1-K|B1-K\+E)/", "/<method>/", y)
        shape[y] = shape.get(y, 0) + 1
    tasks_hit = sorted({("T_B" if "T_B" in x else "D0" if ("/D0/" in x or "final_test_id" in x) else "other") for x in found})
    t_b_paths = [x for x in found if "T_B" in x]
    unused_in_runs = all(v["count"] == 0 for v in use.values()) and all(v["test_field"] == [] for v in split_tb.values())
    return {"document": "tb_test_source_provenance_check", "created": utc_now(),
            "method": "file names, byte hashes of scene capture files and split-config hashes only; no test result, cache content or relation truth was opened; no scene was generated",
            "test_scene_dir": str(test_dir), "scene_count": len(scenes), "file_name_multiset_across_scenes": names_multiset,
            "scene_set_sha256": h_all.hexdigest(), "scenes": per, "split_configs": cfg,
            "vlm_cache_test_entry_count": len(vlm), "vlm_cache_test_note": "30 entries exist (materialized by the old stage_2a flow, request ledger dated 2026-09-23); names counted only, contents not read; the no-prior run context carries no cache pointers and never reads them (src/cp_disr/stage2a_v11.py require_case_cache)",
            "final_master_T_B_split_files": split_tb,
            "runs_small_files_mentioning_T_B_test": use,
            "test_result_file_names_found_in_accessible_worktrees": {"search_complete": complete, "worktree_run_roots_searched": len(roots), "name_patterns": pat,
                                                                    "distinct_path_shapes": shape, "tasks_by_path": tasks_hit, "T_B_paths": t_b_paths[:5], "opened": False},
            "assessment": {
                "T_B_test_used_in_final_master_training_or_dev": (False if unused_in_runs else "CHECK"),
                "T_B_test_result_files_present_in_accessible_worktrees": bool(t_b_paths),
                "test_scenes_and_caches_materialized_before_release": True,
                "reading": "inside the accessible worktrees no T_B test result and no development use of the T_B test scenes was found; the scenes and 30 prior-cache entries were materialized earlier by the old stage_2a flow (before the S6 release step of Plan v3 12.1). Whether any process outside the accessible worktrees ever evaluated models on them cannot be shown from files.",
                "decision_needed": "Source-qualification decision for the T_B test30 set (accept as independent test with the old materialization documented, or re-draw from the reserved pool entries T_B_test_30..49 under a new freeze) - not made here."}}


def md_comparison(runs, statements, base_csv_check):
    L = ["# T_B development evidence: comparison table with R-TB-E-1", "",
         "Built from existing files only (no episode, no environment, no model load, no test). The comparison base is the earlier three-run table "
         "(R-TB-E-0, R-TB-DK-1, R-TB-K-1); R-TB-E-1 is added without re-auditing the repository. Re-read values of the three base runs equal the base CSV "
         "(success, return, N, checkpoint hash): %s (%d rows)." % (base_csv_check["all_equal"], base_csv_check["rows_compared"]), "",
         "Frozen dev10, deterministic argmax, frozen H = 23.1 s, empty R. **Final is the registered primary endpoint**; the 0 / 4096 / 8192 points are "
         "pre-registered learning-process evidence only.", "",
         "## Part 1. Current-profile runs (execution path of prep cac2fa5b / e4dc34ae)", ""]
    order = ["R-TB-E-0", "R-TB-E-1", "R-TB-DK-1", "R-TB-K-1"]
    L += md_tables(runs, order, "Current runs:")
    L += ["", "T at the same N differs between runs because T is the simulated interaction time actually consumed (skill durations differ); N and T are the recorded values, not the nominal 4096 / 8192.", "",
          "### Training budget actually used", ""] + md_budget(runs, order)
    L += ["", "### Every evaluation point (long form)", ""] + md_long(runs, order)
    L += ["", "## Part 2. Historical seed-0 runs (original execution identity, listed apart, not pooled with Part 1)", "",
          "R1 B1-K / B2 and R2 B0 keep the N, T and identity they were run with. They are not averaged with, and not ranked against, the Part 1 runs; the unconfirmed items are in `02_historical_identity_and_compatibility.md`.", ""]
    horder = ["R1-B1K-s0", "R1-B2-s0", "R2-B0-s0"]
    L += md_tables(runs, horder, "Historical runs:")
    L += ["", "### Training budget actually used", ""] + md_budget(runs, horder)
    L += ["", "### Every evaluation point (long form)", ""] + md_long(runs, horder)
    L += ["", "## Part 3. What may be said", ""] + ["- " + s for s in statements]
    L += ["", "## Part 4. What is not said", "", NOT_CLAIMED_BEGIN] + ["- " + s for s in NOT_CLAIMED] + [NOT_CLAIMED_END]
    L += ["", "## Caveats", "",
          "- C1 `success_seconds` is the duration of the final successful skill only (label %s); the episode-level time to success is %s in all existing eval rows (code: `stage2a_v11.py` line 1044 falls back to `t.duration` because `rl.Snapshot` has `clock_seconds`, not `elapsed_seconds`). The frozen discounted return is the time-sensitive quantity here. The revision design is in `04_evaluation_record_revision_design.md`." % (LABEL_SS, LABEL_EPISODE_TOTAL),
          "- C2 Scope: dev10 only; one or two training seeds per method; the three Part 1 base runs and R-TB-E-1 ran on a shared server with 1-3 concurrent workers, so wall-clock is not compared here.",
          "- C3 Selection columns (job-rule selected checkpoint, common-dev20 selection) are kept in `03_frozen_model_list.md` and are not used in any table above.",
          "- C4 Exit-reason families are copied from the eval rows (reason text before the first colon); no mechanism is inferred."]
    return "\n".join(L) + "\n"


def md_historical(hid, runs):
    L = ["# Historical T_B seed-0 runs: identity, compatibility evidence and unconfirmed items", "",
         "Read-only inspection of the three runs found under `graph_cp_disr_v2_1/runs` (v13_r1 and v13_r2). Nothing was retrained, resumed, converted or copied; no checkpoint was loaded; "
         "no episode was run. They keep their original execution identity and are listed apart from the new runs (no pooled mean).", "",
         "## Identity as recorded", ""]
    L += ["| slot | run | planned_id | N / T (s) | updates | stop | recorded job_dir differs from local copy | source commits recorded in manifest |", "|---|---|---|---|---|---|---|---|"]
    for k in HIST_IDS:
        h = hid[k]
        a = h["actual"]
        L.append("| %s | %s | %s | %d / %.2f | %d + %d | %s | %s | %s |" % (h["slot"], k, h["planned_id"], a["N"], a["T_s"], a["complete"], a["fragment"], a["stop"],
                                                                      h["recorded_location_differs_from_local"], json.dumps(h["recorded_source_commits"])))
    L += ["", "Plan v3 section 1 values reproduced from the run files (N and complete + fragment updates): " + ", ".join("%s %s" % (k, hid[k]["plan_v3_section1_values"]["matches"]) for k in HIST_IDS) + ".", "",
          "## File-level comparison with the current execution path (E-1 manifest)", "",
          "| run | equal | differ | H / d_ref / Tcap equal | dev10 split hash equal | dev10 case ids equal |", "|---|---|---|---|---|---|"]
    for k in HIST_IDS:
        h = hid[k]
        v = h["versus_current_2.1.1_execution_path_E1"]
        L.append("| %s | %s | %s | %s | %s | %s |" % (k, ", ".join(v["equal"]), ", ".join(v["differ"]), all(h["frozen_values_equal_to_current"].values()), h["dev10_split_hash_equal_to_current"], h["dev10_case_ids_equal_to_current"]))
    L += ["", "## Internal consistency of each historical final checkpoint (bytes re-hashed, not loaded)", "",
          "| run | final file | sha256 | recorded model.pt sha equals recomputed | latest = final | N and T equal job_summary | eval_final uses it | fresh-load record |", "|---|---|---|---|---|---|---|---|"]
    for k in HIST_IDS:
        c = runs[k]
        f = c["final"]
        ch = f["checks"]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (k, f["checkpoint_name"], f["sha256_recomputed_now"], ch["recorded_model_sha_equals_recomputed"], ch["latest_is_final_generation"],
                                                         ch["generation_N_equals_job_summary"] and ch["generation_T_equals_job_summary"], ch["eval_final_checkpoint_is_final"], ch["fresh_load_ok"]))
    L += ["", "## Existing compatibility evidence cited", ""]
    for e in COMPAT_EVIDENCE_CITED:
        L.append("- %s. %s - %s" % (e["id"], e["source"], e["says"]))
    L += ["", "## Unconfirmed items (listed apart; no pooled mean; no automatic retraining)", "", "| id | item | why | status | handling |", "|---|---|---|---|---|"]
    for u in UNCONFIRMED_HISTORICAL:
        L.append("| %s | %s | %s | %s | %s |" % (u["id"], u["item"], u["why"], u["status"], u["handling"]))
    L += ["", "Correction to an earlier note: an earlier message said these historical artifacts could not be found; that came from a depth-limited file search. They are present, and this record is based on the files.", ""]
    return "\n".join(L) + "\n"


def md_freeze(fl):
    L = ["# Frozen main-evaluation model list for the later unified T_B evaluation", "", "Frozen at %s (UTC), repository head %s. This freezes the candidate list only; no test is released." % (fl["frozen_at_utc"], fl["repo_head_at_freeze"]), "", "## Rules", ""]
    L += ["- " + r for r in fl["rules"]]
    L += ["", "## Main-evaluation checkpoints (last valid post-update generation)", "",
          "| slot | run | method | seed | list | final checkpoint | N | T (s) | update | sha256 (recomputed now) | recorded = recomputed | file/hash checks | load under 2.1.1 code |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in fl["models"]:
        c = m["main_eval_checkpoint"]
        L.append("| %s | %s | %s | %s | %s | %s | %d | %.2f | %s | %s | %s | %s | %s |" % (m["slot"], m["plan_run"], m["method"], m["seed"], m["list"], c["name"], c["N"], c["T_s"], c["update_index"],
                                                                                         c["sha256"], c["checks"]["recorded_model_sha_equals_recomputed"], m["status"]["file_and_hash_consistency"], m["status"]["load_under_current_2.1.1_code"]))
    L += ["", "## Selected / best-dev checkpoints (separate columns; not substitutes for final)", "",
          "| run | job-rule selected checkpoint | equals final | rule | common-dev20 selection (historical runs) |", "|---|---|---|---|---|"]
    for m in fl["models"]:
        s = m["selected_by_job_rule_separate_column"]
        d = m["common_dev20_selection_separate_column"]
        L.append("| %s | %s | %s | %s | %s |" % (m["plan_run"], s["name"], s["is_final"], s["rule"], ("%s %d/%d (return %.4f)" % (d["checkpoint"], d["success_n"], d["n"], d["mean_discounted_return"])) if d else "not recorded"))
    L += ["", "Runs whose selected checkpoint is n_000000 are evaluated at their final model, never at N = 0.", "", "## Counts", "",
          "Base: 6 models x 30 = %d episodes if released (inside the Plan v3 cap of 628). Conditional: B1-K+E seed 1 adds %d (outside the cap, Plan v3 12.2)." % (fl["counts"]["base_episodes_if_released"], fl["counts"]["conditional_episodes_if_approved"]), "",
          "## Not in the list", ""]
    L += ["- %s: %s" % (x["item"], x["reason"]) for x in fl["excluded"]]
    L += ["", "## Identity gaps carried with the historical entries", "", "See `02_historical_identity_and_compatibility.md` (UC1-UC7). A gap is carried, not repaired.", ""]
    return "\n".join(L) + "\n"


def md_design(ex):
    return """# Design: unified evaluation record revision (observation and recording only)

Status: DESIGN ONLY. No source file is changed by this record, no environment is started, no episode is run. Implementation, offline tests and any replay need a separate approval.

## 1. What the existing record can and cannot say (verified in the source and the eval files)

- `eval_episodes` (`src/cp_disr/stage2a_v11.py`, line 1009 onward) writes one row per case with: `case_id, success, G, steps, reason, success_seconds, source_n, prior_mode`.
- `success_seconds` (line 1044) is `t.snapshot.elapsed_seconds + t.duration` only if the snapshot has `elapsed_seconds`; `rl.Snapshot` (`src/cp_disr/rl.py`, line 74) has `clock_seconds` and no `elapsed_seconds`, so the fallback `t.duration` is always taken. The recorded value is therefore the duration of the last (successful) skill only. Example from the existing R-TB-E-1 final evaluation: %(steps)s skill steps per successful case, recorded success_seconds %(ss)s, while the episode contains several skills. Label kept for the old field: %(lab)s. Episode-level time to success: %(ep)s in every existing file.
- The action sequence is not stored: rows keep `steps` (a count). `Transition.selected_candidate_id` exists in memory (`rl.py`) but `eval_episodes` does not write it.
- The evaluated model is identified only by the checkpoint path; the eval payload's `hashes` block holds source-code hashes, not the checkpoint file hash.
- The quantity that is needed already exists in the loop: `Collector.step` (`src/cp_disr/collector.py`, line 187) computes `elapsed = clock_end - bundle.episode_start_seconds` and hands it to the independent evaluator in `EvaluationInput(..., elapsed, start, end)`; `actual.success` is that evaluator's confirmation (the guard at line 195 rejects a reward without independent success). Precedent for the right definition: `phase_a_v13_r3.py` (line 439) and `stage1a_v11_final_eval.py` (line 626) record `bundle.clock.now_seconds() - bundle.episode_start_seconds`.

## 2. Target record (schema `eval_row_v2`, additive)

Per episode row, in addition to every existing field (unchanged, same values):

| field | definition | source (existing objects only) |
|---|---|---|
| `episode_start_clock_s` | simulation clock at episode start | `bundle.episode_start_seconds`, read after `start_case` |
| `time_to_first_confirmed_success_s` | simulated seconds from episode start to the end of the first interval in which the independent evaluator confirmed success; null if none | `t.next_snapshot.clock_seconds - bundle.episode_start_seconds` at the first step with `result.success` (equals Collector's `elapsed` for that step) |
| `time_to_first_confirmed_success_label` | `EPISODE_START_TO_FIRST_INDEPENDENT_CONFIRMED_SUCCESS_SIM_SECONDS` | constant |
| `episode_end_elapsed_s` | simulated seconds from episode start to the last interval end | same, at the last step |
| `actions` | ordered list, one entry per executed decision: `decision_index`, `selected_candidate_id`, `clock_start_s`, `clock_end_s`, `duration_s`, `reward`, `weight`, `exit_reason`, `terminated`, `truncated` | `t.snapshot.decision_id`, `t.selected_candidate_id`, `t.snapshot.clock_seconds`, `t.next_snapshot.clock_seconds`, `t.duration`, `t.reward`, `t.weight`, `t.reason` |
| `success_seconds` | unchanged | old computation |
| `success_seconds_label` | `FINAL_SKILL_DURATION_ONLY` | constant, also written for old-style rows by any reader |
| `no_transition_exit` | reason and elapsed when the loop ends through `t is None` (for example INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE) | the returned result dict |

Payload level: `schema_version = eval_row_v2`; `evaluated_model = {path, sha256, bytes}` with the sha256 computed from the file bytes before loading and checked again after the last episode; `evaluated_generation` with N, T and update from the generation manifest; the existing `hashes` (source) block; `case_order`; `eval_rng_isolation` flag; `label` (dev10 / common_dev20 / test).

## 3. Not changed

Reward, discounted return G and weights, action selection (deterministic argmax), termination and deadline logic (T_B deadline 60 s), safety / verifier / evaluator code, the training loop, checkpoint format, case order and the evaluation RNG isolation. `Collector.step` keeps its signature and return values.

## 4. Where it would live (proposal, not applied)

- Option A (preferred): a new evaluation-only module (for example `src/cp_disr/final_tb_eval_record.py`) that makes the same calls in the same order as `eval_episodes` (`seed_all(0)`, `make_policy`, `load_checkpoint`, `start_case`, `apply_episode_prior`, `reset_episode`, `step(..., deterministic=True)`) and only adds the recording above. `stage2a_v11.py` and `collector.py` stay byte-identical, so the execution-path hashes carried by every training manifest (`stage2a_v11`, `collector`) do not move and nothing in a possible further training run changes identity.
- Option B (not recommended): add a write-only attribute inside `Collector.step` or edit `eval_episodes`. It changes the `collector` / `stage2a_v11` source hashes shared with training.

## 5. Offline test plan (no environment, no episode)

1. Field arithmetic with scripted fake transitions and a fake bundle clock: 5-step success, success on the last allowed step, deadline exit, no-candidate exit, INSUFFICIENT_REMAINING exit.
2. Old-field invariance: `success`, `G`, `steps`, `reason` and the old `success_seconds` computed by the new module equal the old computation on the same scripted input.
3. Cross-check: sum of `duration_s` over actions versus `time_to_first_confirmed_success_s`; any difference is written as `inter_step_clock_gap_s`, never silently assumed zero.
4. The last action's `duration_s` equals the old `success_seconds` on successful rows.
5. Model hash: sha256 before load equals sha256 after the last episode; mismatch aborts the evaluation.
6. Static check (AST / hash): `collector.py`, `rl.py` (gamma, interval_reward) and `stage2a_v11.py` hashes unchanged; the new module imports no optimizer and writes no training artifact.
7. Reader test: rows without the new fields load and are labelled %(lab)s; they never enter a time-to-success column.

## 6. Order of work (all separate approvals)

(a) implement the module and tests, no episodes; (b) optional consistency replay of the frozen final models on the already-used dev10 to compare success, G, steps and the old success_seconds with the existing rows (this runs episodes and is not part of this record); (c) only then a test release. Existing evaluation files are not rewritten and cannot be completed retroactively: the action sequences were never stored, so their episode-level time stays %(ep)s.
""" % ex


def build_subcard(fl, prov):
    prereq = [
        {"id": "P1", "text": "S4 freeze of method, inputs, optimisation and endpoints, and S5 completion of the prescribed seeds, for T_B (Plan v3 12.1 item 1)", "state": "NOT CONFIRMED by this record - needs an explicit statement"},
        {"id": "P2", "text": "Unified main-evaluation model list frozen (final generation per run) with hashes", "state": "DONE as a candidate list in `03_frozen_model_list.md`; historical load check open (P8)"},
        {"id": "P3", "text": "Evaluation-record revision implemented, offline-tested and committed before any test run", "state": "NOT DONE - design only (`04_...`)"},
        {"id": "P4", "text": "`test_release_manifest`: model and Adam index, source hashes, input hashes, scene order, evaluator and RNG settings, invalid-episode rule", "state": "NOT BUILT - built at release from the frozen list"},
        {"id": "P5", "text": "Source qualification of the T_B test30 set (a T_B test used in development cannot be called blind test)", "state": "UNRESOLVED - evidence in `06_...`; decision U1"},
        {"id": "P6", "text": "A runner that takes the frozen list: the existing `evaluate_final_tests` (`stage2a_v11.py`, line 1589) reads an 8-checkpoint `selection_manifest` of the old stage and raises unless exactly 8 are selected", "state": "NOT DONE - cannot be reused as is"},
        {"id": "P7", "text": "Derived no-prior test split for the 30 active ids (records without cache pointers, as for the dev10 derivation), built without opening any cache or relation truth", "state": "NOT DONE"},
        {"id": "P8", "text": "Load-only compatibility check of the three historical final weights under the current code (no episode), to decide UC1", "state": "NOT RUN - approval requested (D4)"},
        {"id": "P9", "text": "Compute, disk and GPU plan for 180 (+30) episodes and the storage guard", "state": "NOT ASSESSED"},
        {"id": "P10", "text": "Approval of the conditional +E seed 1 model (30 episodes outside the 628)", "state": "PENDING (D3)"}]
    unresolved = [
        {"id": "U1", "item": "T_B test30 qualification: scenes and 30 prior-cache entries were materialized earlier by the old stage_2a flow, before the S6 release step; no T_B test result or development use was found in the accessible worktrees, but use outside them cannot be shown from files. Accept with the old materialization documented, or draw from the reserved pool under a new freeze.", "decision": "user"},
        {"id": "U2", "item": "Historical models (UC1-UC7): loadability under the current code, evaluator-source differences, R2 B0 original location, commit pointers, the R2 B0 dev20 file holding 10 cases. Carried as gaps; no sidecar, no retraining.", "decision": "user (after P8)"},
        {"id": "U3", "item": "Invalid-episode rule (infrastructure failure versus policy outcome) must be fixed in the release manifest before the first episode; no repeat to satisfaction.", "decision": "user / sub-card"},
        {"id": "U4", "item": "Whether the conditional B1-K+E seed 1 enters the same release or a later one.", "decision": "user"},
        {"id": "U5", "item": "S4 / S5 status for T_B (P1).", "decision": "user"},
        {"id": "U6", "item": "Whether a dev10 consistency replay of the frozen finals with the revised recorder is wanted before test (it runs episodes, separate card).", "decision": "user"}]
    decisions = [
        "D1 Confirm the model list of `03_frozen_model_list.md` (6 base + 1 conditional) and the rule that the final generation is the main-evaluation model.",
        "D2 Confirm the episode budget: 180 base, maximum 210 with the conditional model.",
        "D3 Decide whether the conditional B1-K+E seed 1 model is released with the base set.",
        "D4 Approve the load-only compatibility check (no episode) for the three historical final weights, or state that they stay as recorded gaps.",
        "D5 Approve a separate implementation card for the evaluation-record revision (source change plus offline tests, no episodes).",
        "D6 Decide the source-qualification path for T_B test30 (U1) and state the S4 / S5 status (U5)."]
    not_done = [
        "No test score, test cache, relation truth or test result file was read or opened. The test scene capture files (rgb.png, depth.npy, reset_config.json, CAPTURE_COMPLETE) were byte-hashed only, to fix scene identity; they were not parsed or rendered.",
        "No evaluation or episode was run; no model was loaded; no scene was generated or regenerated.",
        "No RL, environment, provider, optimizer or elastic slot was added or used; ELASTIC-02..04 are not allocated; the formal test was not started; Family B and RoboCasa were not resumed.",
        "No method, training configuration, profile, reward, action or termination setting was changed; no source file was edited."]
    return {"card_id": "CP-DISR-TB-INDEP-EVAL-01", "status": "REQUEST_NOT_EXECUTED", "models": fl["models"], "prov": prov, "prerequisites": prereq,
            "unresolved": unresolved, "decisions_requested": decisions, "not_done": not_done,
            "episode_budget": {"base_models": 6, "cases_per_model": 30, "base_episodes": 180, "conditional_models": 1, "conditional_episodes": 30, "max_episodes_if_conditional_approved": 210,
                               "plan_v3_cap_all_tasks": 628, "conditional_counted_outside_cap": True, "episode_deadline_s": 60.0}}


def md_subcard(sc):
    pv = sc["prov"]
    cfgs = pv["split_configs"]
    L = ["# Sub-card request: T_B independent evaluation (CP-DISR-TB-INDEP-EVAL-01)", "",
         "Status: **REQUEST - NOT EXECUTED**. This request reads no test result, runs no evaluation, loads no model and generates no scene. Everything below is taken from files and from Plan v3 section 12.", "",
         "## 1. Purpose and limits", "",
         "One unified independent evaluation of the frozen final T_B models on the T_B test30 scenes, with the revised evaluation record of `04_evaluation_record_revision_design.md`. It does not tune anything, does not select a model on test, does not train, and does not release or read test material for any task other than T_B.", "",
         "## 2. Model list (frozen in `03_frozen_model_list.md`)", "",
         "| slot | run | method | seed | list | final checkpoint | N | sha256 | open gaps |", "|---|---|---|---|---|---|---|---|---|"]
    for m in sc["models"]:
        c = m["main_eval_checkpoint"]
        L.append("| %s | %s | %s | %s | %s | %s | %d | %s | %s |" % (m["slot"], m["plan_run"], m["method"], m["seed"], m["list"], c["name"], c["N"], c["sha256"], ", ".join(m["status"]["identity_gaps"]) or "none"))
    L += ["", "Main rule: last valid post-update checkpoint of each run. Selected / best-dev columns are separate; no failing method is evaluated at N = 0 in place of its final.", "",
          "## 3. Data source", "",
          "- Scenes: `%s` (T_B_test_00 .. T_B_test_29; each holds %s); scene-set sha256 `%s`." % (pv["test_scene_dir"], ", ".join(sorted(pv["file_name_multiset_across_scenes"])), pv["scene_set_sha256"]),
          "- Split definition: `configs/splits/T_B_stage_2a_test30.json` (sha256 %s), `T_B_stage_2a_test_ids.json` (sha256 %s), `T_B_stage_2a_v11.json` (sha256 %s). 30 active cases in a fixed order with fixed reset seeds shared by the methods; 20 reserved pool entries (T_B_test_30..49) stay unused." % tuple(cfgs[k]["sha256"] for k in ("configs/splits/T_B_stage_2a_test30.json", "configs/splits/T_B_stage_2a_test_ids.json", "configs/splits/T_B_stage_2a_v11.json")),
          "- Prior / relation material: these methods run with empty R. The no-prior run context carries no cache pointers and `require_case_cache` returns without reading; the %d `vlm_cache/test` entries therefore are neither needed nor to be opened." % pv["vlm_cache_test_entry_count"],
          "- Provenance evidence (file names, hashes, config declarations only): `06_test_source_provenance_check.json`. In the accessible worktrees no T_B test result file exists (tasks with test-result file names: %s), the final-master T_B train/dev splits carry an empty `test` list, and no small run file of the seven models mentions a test scene. The only test-result files found belong to D0 (cited by identity per Plan v3 12.2, not opened)." % ", ".join(pv["test_result_file_names_found_in_accessible_worktrees"]["tasks_by_path"]), "",
          "## 4. Episode budget", "",
          "| item | episodes |", "|---|---|",
          "| base: 6 models, 30 test cases each | 180 (inside the Plan v3 cap of 628) |",
          "| conditional: B1-K+E seed 1 (R-TB-E-1), 30 cases | 30 (outside the 628; needs its own approval) |",
          "| maximum if the conditional model is approved | 210 |", "",
          "One pass per model on the fixed scene order, deterministic argmax, evaluation RNG isolated, no repeat to satisfaction. T_B episode deadline 60 s (simulated).", "",
          "## 5. Test prerequisites (Plan v3 12.1) and their present state", "", "| id | prerequisite | state |", "|---|---|---|"]
    for p in sc["prerequisites"]:
        L.append("| %s | %s | %s |" % (p["id"], p["text"], p["state"]))
    L += ["", "## 6. Unresolved items", "", "| id | item | decision by |", "|---|---|---|"]
    for u in sc["unresolved"]:
        L.append("| %s | %s | %s |" % (u["id"], u["item"], u["decision"]))
    L += ["", "## 7. Decisions requested", ""] + ["- %s" % d for d in sc["decisions_requested"]]
    L += ["", "## 8. What this request did not do", ""] + ["- %s" % d for d in sc["not_done"]]
    return "\n".join(L) + "\n"


WORDING_PATTERNS = [r"\b\d+(\.\d+)?\s?(x|×)\b", r"times\s+(faster|as fast)", r"\bfaster\b", r"\bslower\b", r"speed-?up", r"significan", r"\bp\s?[<=]\s?0?\.\d", r"equivalen", r"superior",
                    r"outperform", r"better than", r"generaliz", r"dominat", r"\bproves?\b"]


def wording_scan(texts):
    hits = []
    for name, t in texts.items():
        body = re.sub(re.escape(NOT_CLAIMED_BEGIN) + r".*?" + re.escape(NOT_CLAIMED_END), "", t, flags=re.S)
        for pat in WORDING_PATTERNS:
            for m in re.finditer(pat, body, flags=re.I):
                a = max(0, m.start() - 40)
                hits.append({"file": name, "pattern": pat, "context": body[a:m.end() + 40].replace("\n", " ")})
    return hits


def write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def wjson(path, obj):
    write(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def readme(out_name, checks):
    return """# T_B development-evidence closure with R-TB-E-1 and unified-evaluation preparation

Directory: `%s`. New summary material only; no original run record is overwritten (R-TB-E-1 results stay in commit 477de5dec2d59e22eaa0f6a840bb0d1b80f0a687).

| file | content |
|---|---|
| `01_comparison_with_E1.md/.json/_points.csv` | comparison table with R-TB-E-1: success, discounted return, actual N / T at 0 / 4096 / 8192 / final; permitted statements; what is not said |
| `02_historical_identity_and_compatibility.md/.json` | R1 B1-K / B2 seed 0 and R2 B0 seed 0: original identity, cited compatibility evidence, unconfirmed items listed apart |
| `03_frozen_model_list.md/.json` | frozen main-evaluation list (last valid post-update checkpoint), selected / best-dev in separate columns, hash reuse |
| `04_evaluation_record_revision_design.md` | design only: episode start to first independent confirmed success, action sequence, evaluated-model hash; old success_seconds kept as FINAL_SKILL_DURATION_ONLY |
| `05_tb_independent_eval_subcard_request.md/.json` | T_B independent-evaluation sub-card request: models, data source, episode budget, prerequisites, unresolved items, decisions requested |
| `06_test_source_provenance_check.json` | test source evidence from file names, hashes and config declarations only |
| `build_summary.json`, `MANIFEST_SHA256.json`, `scripts/build_tb_dev_closure.py` | build checks, file hashes, the zero-environment builder |

Not done in this step: no RL, environment, episode, provider or optimizer was added or used; no formal test was started; no test score, cache or relation truth was read; no model was loaded; no scene was generated; ELASTIC-02..04 are not allocated; Family B and RoboCasa are not resumed; no method or training configuration was changed.

Build checks: %s
""" % (out_name, json.dumps(checks, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hist-repo", required=True)
    ap.add_argument("--base-csv", required=True)
    ap.add_argument("--e1-record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    repo, hist_repo, out = Path(a.repo), Path(a.hist_repo), Path(a.out)
    fm = repo / "runs/final_master/2.1.1"
    protected = [Path(a.base_csv), Path(a.e1_record)]
    pre = {str(p): sha256_file(p) for p in protected}
    reg = registry(fm, hist_repo / "runs")
    runs = {s["id"]: collect(s) for s in reg}
    head = git_out(repo, "rev-parse", "HEAD")
    chk = {}
    chk["base_csv_crosscheck"] = crosscheck_base_csv(a.base_csv, runs)
    chk["base_csv_crosscheck_all_equal"] = chk["base_csv_crosscheck"]["all_equal"]
    e1rec = jload(a.e1_record)["R-TB-E-1"]
    e1 = runs["R-TB-E-1"]
    chk["e1_matches_committed_record"] = (e1["actual_N"] == e1rec["actual_N_valid_transitions"] and e1["actual_T_s"] == e1rec["actual_T_interaction_seconds"]
                                          and [p["success_n"] for p in e1["points"]] == [x["success_n"] for x in e1rec["eval_points"]]
                                          and [p["mean_discounted_return"] for p in e1["points"]] == [x["mean_discounted_return"] for x in e1rec["eval_points"]]
                                          and e1["final"]["sha256_recomputed_now"] == e1rec["selected_checkpoint_sha256"])
    chk["all_final_checkpoint_checks_true"] = {k: v["final"]["all_checks_true"] for k, v in runs.items()}
    chk["head"] = head
    assert chk["base_csv_crosscheck_all_equal"] and chk["e1_matches_committed_record"] and all(chk["all_final_checkpoint_checks_true"].values()), json.dumps(chk)[:2000]
    statements = descriptive_statements(runs)
    hid = historical_identity(runs, hist_repo)
    assert hid["R2-B0-s0"]["original_location_probe"]["result"] in ("PERMISSION_DENIED", "MISSING"), hid["R2-B0-s0"]["original_location_probe"]
    fl = freeze_list(runs, head)
    vlm_test = repo / "experiments/vlm_cache/test"
    prov = provenance(repo, hist_repo / "experiments/stage_2a_inputs/T_B/test", vlm_test, hist_repo, runs)
    sc = build_subcard(fl, prov)
    rows = jload(Path(e1["run_dir"]) / "eval_final.json")["rows"]
    ss = [x["success_seconds"] for x in rows if x.get("success_seconds") is not None]
    smin, smax = min(x["steps"] for x in rows), max(x["steps"] for x in rows)
    ex = {"steps": ("%d" % smin) if smin == smax else ("%d to %d" % (smin, smax)), "ss": "%.2f to %.2f s" % (min(ss), max(ss)), "lab": LABEL_SS, "ep": LABEL_EPISODE_TOTAL}
    texts = {"01_comparison_with_E1.md": md_comparison(runs, statements, chk["base_csv_crosscheck"]),
             "02_historical_identity_and_compatibility.md": md_historical(hid, runs),
             "03_frozen_model_list.md": md_freeze(fl),
             "04_evaluation_record_revision_design.md": md_design(ex),
             "05_tb_independent_eval_subcard_request.md": md_subcard(sc)}
    hits = wording_scan(texts)
    chk["forbidden_wording_hits"] = hits
    assert not hits, json.dumps(hits)[:2000]
    out.mkdir(parents=True, exist_ok=True)
    for name, t in texts.items():
        write(out / name, t)
    # long-form csv
    with open(out / "01_comparison_with_E1_points.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["plan", "method", "seed", "group", "point", "update", "N_actual", "T_actual_s", "success_n", "n", "mean_discounted_return",
                    "mean_recorded_success_seconds_" + LABEL_SS, "exit_reason_families", "checkpoint", "checkpoint_sha256_recorded"])
        for k in ("R-TB-E-0", "R-TB-E-1", "R-TB-DK-1", "R-TB-K-1", "R1-B1K-s0", "R1-B2-s0", "R2-B0-s0"):
            c = runs[k]
            for p in c["points"]:
                w.writerow([k, c["method"], c["seed"], "historical" if k in HIST_IDS else "current", p["point"], p["update"], p["N_actual"], p["T_actual_s"], p["success_n"], p["n"],
                            repr(p["mean_discounted_return"]), "" if p["mean_recorded_success_seconds"] is None else p["mean_recorded_success_seconds"],
                            json.dumps(p["exit_reason_families"]), p["checkpoint"], p["checkpoint_sha256_recorded"]])
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "points"} | {"points": [{kk: vv for kk, vv in p.items() if kk != "case_ids"} for p in v["points"]]} for k, v in runs.items()}
    wjson(out / "01_comparison_with_E1.json", {"document": "comparison_with_E1", "created": utc_now(), "labels": {"success_seconds": LABEL_SS, "episode_total_success_time": LABEL_EPISODE_TOTAL},
                                               "statements_permitted": statements, "not_claimed": NOT_CLAIMED, "base_csv_crosscheck": chk["base_csv_crosscheck"], "runs": slim})
    wjson(out / "02_historical_identity_and_compatibility.json", {"document": "historical_identity", "created": utc_now(), "runs": hid, "compatibility_evidence_cited": COMPAT_EVIDENCE_CITED, "unconfirmed": UNCONFIRMED_HISTORICAL})
    wjson(out / "03_frozen_model_list.json", fl)
    sc_json = dict(sc)
    sc_json["prov"] = {"summary_only": True, "see": "06_test_source_provenance_check.json", "scene_set_sha256": prov["scene_set_sha256"], "assessment": prov["assessment"]}
    wjson(out / "05_tb_independent_eval_subcard_request.json", sc_json)
    wjson(out / "06_test_source_provenance_check.json", prov)
    post = {str(p): sha256_file(p) for p in protected}
    chk["protected_inputs_unchanged"] = (pre == post)
    chk["created"] = utc_now()
    chk["zero_env"] = {"torch_imported": "torch" in sys.modules, "episodes_run": 0, "models_loaded": 0, "test_result_files_opened": 0, "test_cache_or_relation_truth_opened": 0,
                       "test_scene_capture_files_byte_hashed_not_parsed": sum(len(s["sha256"]) for s in prov["scenes"])}
    assert chk["protected_inputs_unchanged"] and not chk["zero_env"]["torch_imported"]
    write(out / "README.md", readme(out.name, {k: v for k, v in chk.items() if k not in ("base_csv_crosscheck",)}))
    wjson(out / "build_summary.json", chk)
    man = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST_SHA256.json":
            man[str(p.relative_to(out))] = sha256_file(p)
    wjson(out / "MANIFEST_SHA256.json", man)
    print(json.dumps({"written": len(man), "out": str(out), "checks": {k: v for k, v in chk.items() if k in ("base_csv_crosscheck_all_equal", "e1_matches_committed_record", "protected_inputs_unchanged", "zero_env")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
