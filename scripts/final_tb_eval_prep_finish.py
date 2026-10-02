#!/usr/bin/env python
"""Second builder for CP-DISR-TB-EVAL-PREP-LITE-01: dev-consistency summary, test-source freeze, candidate release
manifest, summary and verify.  Reads ONLY: committed split configs (ids / seeds / row hashes), committed provenance
JSON, run/eval json of earlier cards, checkpoint bytes (sha256 only).  Opens no test RGB/depth, no VLM cache, no relation
truth, no test result; builds no environment; runs no episode.

Subcommands: dev-consistency | test-source-freeze | release-manifest | summary | verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BASELINE = "241159c1ceca3b53f0ba8423879524c6b2668e07"
ROOT = Path(__file__).resolve().parents[1]
V2_1 = Path("/home/xushijie2/graph_cp_disr_v2_1")
CLOSURE = ROOT / "runs/final_master/2.1.1/launch/20261002T125454Z_tb_dev_closure_e1_477de5de"
E1_LAUNCH = ROOT / "runs/final_master/2.1.1/launch/20261002T031938Z_e1_elastic01_e4dc34ae"
LABEL = "PREMATERIALIZED_INDEPENDENT_HOLDOUT"
FILES_READ = []  # every input file this script opens for the test-source freeze


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a, cwd=ROOT):
    return subprocess.check_output(["git", *a], cwd=str(cwd), text=True).strip()


def load(p, record=False):
    if record:
        FILES_READ.append(str(p))
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p, d):
    Path(p).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------- dev consistency
def cmd_dev(out):
    out = Path(out)
    cur, arc = load(out / "phase_c/dev_check_current.json"), load(out / "phase_c/dev_check_archived.json")

    def view(d):
        nf = d["new_fields"]
        return {"slot": d["slot"], "method": d["method"], "case": d["case"], "status": d["payload_status"],
                "old_record_comparison": d["old_record_comparison"], "all_old_fields_equal": d["all_old_fields_equal"],
                "internal_consistency": d["internal_consistency"], "informational": d["informational"],
                "new_fields_summary": {**{k: nf[k] for k in ("episode_start_clock_s", "time_to_first_confirmed_success_s", "first_confirmed_success_decision_index",
                                                             "episode_end_elapsed_s", "inter_step_clock_gap_s", "no_transition_exit")},
                                       "n_actions": len(nf["actions"]), "action_sequence": [a["candidate_id"] for a in nf["actions"]]},
                "evaluated_model": d["evaluated_model"], "evaluated_generation": d["evaluated_generation"], "source_hashes": d["source_hashes"],
                "result_file": d["eval_row_v2_file"], "seconds": d["seconds"]}

    cur_ok = cur["all_old_fields_equal"] and cur["internal_consistency"]["all_internal_checks_pass"]
    arc_ok = arc["all_old_fields_equal"] and arc["internal_consistency"]["all_internal_checks_pass"]
    doc = {
        "card": "CP-DISR-TB-EVAL-PREP-LITE-01", "phase": "C", "fixed_case": "T_B_dev_00 (not changed)",
        "gate": "Phase A and B passed before any episode",
        "budget": {"episode_cap": 3, "episodes_executed": 3, "environment_constructions": 3, "environment_construction_cap": 3, "provider_calls": 0,
                   "optimizer_steps": 0, "test_episodes": 0, "third_execution_class_episode": "not needed: B0 s0 shares the archived execution class (identical execution hashes in its run manifest)"},
        "episodes": [
            {"id": "C-1", "class": "CURRENT_2_1_1_EXECUTION_CLASS", "slot": cur["slot"], "outcome": "COMPLETE", "result_persisted": True},
            {"id": "C-2a", "class": "ARCHIVED_V13_EXECUTION_CLASS", "slot": arc["slot"], "outcome": "EXECUTED_RESULT_NOT_PERSISTED",
             "result_persisted": False, "counted_against_budget": True,
             "reason": "the worker chdir-ed into the archived root and then wrote a RELATIVE --out path; the recorder's atomic write failed after the episode had run. "
                       "No value of this attempt was read or used. The worker now resolves every path before chdir and probes the output directory before building the environment.",
             "log": "logs/dev_check_archived_attempt1_lost_result.log"},
            {"id": "C-2b", "class": "ARCHIVED_V13_EXECUTION_CLASS", "slot": arc["slot"], "outcome": "COMPLETE", "result_persisted": True,
             "note": "re-run of C-2a with the path fix; this is the third and last episode"}],
        "current_class": view(cur), "archived_class": view(arc),
        "class_verdicts": {"CURRENT_2_1_1_EXECUTION_CLASS": "CONSISTENT" if cur_ok else "INCONSISTENT",
                           "ARCHIVED_V13_EXECUTION_CLASS": "CONSISTENT" if arc_ok else "INCONSISTENT"},
        "coverage_notes": [
            "C-1 used R-TB-K-1 (B1-K seed 1), a FAILING dev case (11 actions, INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE, last action controller exit TASK_DEADLINE): it exercises the no-transition exit and deadline-capped action recording, not the success path.",
            "C-2b used R1-B2-s0 (historical B2 seed 0), a SUCCESS dev case (5 actions): it exercises first-confirmed-success time, the action chain and the old/new time contrast on a real environment.",
            "The archived class is evidenced by one B2 episode; B1-K s0 and B0 s0 belong to the same class by identical recorded execution hashes (not by their own episode).",
            "Tool versions: C-1 ran with the dev-check worker as committed in the engineering commit; C-2b ran with the same worker after the path-resolution / output-probe fix (no change to episode logic, recorder bytes identical in both: see source_hashes.recorder).",
            "Only fields present in the old eval row were compared with the old record: success, G, steps, reason, success_seconds. actions / episode total time are checked for internal consistency only and were NOT written back into any old eval file."],
        "old_vs_new_time_contrast_C-2b": {"old_success_seconds_FINAL_SKILL_DURATION_ONLY": arc["old_record_comparison"]["success_seconds"]["new"],
                                          "time_to_first_confirmed_success_s": arc["new_fields"]["time_to_first_confirmed_success_s"]},
        "decision": {"ARCHIVED_V13_EXECUTION_CLASS": "Phase A class B CONFIRMED by Phase C" if arc_ok else "STOP: class excluded, test not read, model not repaired",
                     "CURRENT_2_1_1_EXECUTION_CLASS": "CONFIRMED" if cur_ok else "STOP"},
    }
    dump(out / "dev_consistency_results.json", doc)
    print(json.dumps(doc["class_verdicts"]))


# ----------------------------------------------------------------------------- test source freeze
def cmd_freeze(out):
    out = Path(out)
    t30p, idsp = ROOT / "configs/splits/T_B_stage_2a_test30.json", ROOT / "configs/splits/T_B_stage_2a_test_ids.json"
    provp = CLOSURE / "06_test_source_provenance_check.json"
    t30, ids, prov = load(t30p, True), load(idsp, True), load(provp, True)
    active = t30["active"]
    order = [r["case_id"] for r in active]
    assert order == [r["case_id"] for r in ids["active"]], "active order differs between the two registered lists"
    assert len(order) == 30 and len(set(order)) == 30
    scene_names = [s["scene"] for s in prov["scenes"]]
    cache_keys = sorted({k for r in active for k in r if k.startswith("cache")})
    rows = [{"case_id": r["case_id"], "pool_index": r.get("pool_index"), "reset_seed": r["seed"], "stream": r.get("stream"),
             "reset_config_row_sha256": canon_sha({k: r[k] for k in sorted(r)}),
             "scene_files_sha256_from_committed_provenance": next(s["sha256"] for s in prov["scenes"] if s["scene"] == r["case_id"])} for r in active]
    doc = {
        "card": "CP-DISR-TB-EVAL-PREP-LITE-01", "label": LABEL, "not_strict_blind_test": True,
        "label_note": "scenes and 30 VLM-cache entries were materialized by the old stage_2a flow before any release step; no T_B test result or development use was found in accessible worktrees (committed provenance), but use outside them cannot be shown from files",
        "allowed_reads_only": ["test id list", "reset seed / reset config identity", "scene file hashes (committed provenance)", "committed provenance"],
        "files_read": sorted(set(FILES_READ)),
        "split_configs_sha256": {"configs/splits/T_B_stage_2a_test30.json": sha256_file(t30p), "configs/splits/T_B_stage_2a_test_ids.json": sha256_file(idsp)},
        "not_done": {"test_rgb_or_depth_opened": False, "old_test_vlm_cache_opened": False, "relation_truth_read": False, "test_episode_run": False,
                     "test_score_generated": False, "scene_generated_or_regenerated": False, "test_result_file_opened": False},
        "scene_set": {"dir": prov["test_scene_dir"], "scene_count": prov["scene_count"], "scene_set_sha256": prov["scene_set_sha256"],
                      "source": "committed 06_test_source_provenance_check.json (byte hashes; scene contents never parsed)",
                      "scene_names_equal_active_ids": scene_names == order},
        "active_cases": {"n": len(order), "order": order, "same_order_in_both_registered_lists": True,
                         "reserved_pool_unused": [r["case_id"] for r in ids["reserved"]], "rows": rows},
        "registered_properties_quoted": {k: ids[k] for k in ("registration_timing", "frozen_before_main_training", "frozen_before_any_stage2a_test_model_result",
                                                              "same_order", "same_reset_seed", "stream_omits_method_and_training_seed", "filtered_by_model_performance",
                                                              "filtered_by_scripted_success", "filtered_by_nonempty_prior")},
        "no_prior_requirement": {
            "cache_pointer_keys_present_in_test30_active_rows": cache_keys,
            "forbidden_source": "configs/splits/T_B_stage_2a_v11.json test rows (they carry cache_dir/cache_key/cache_status)",
            "future_runner_split_rule": "case rows come from configs/splits/T_B_stage_2a_test30.json `active` (no cache pointer keys); the runner's split_index = the derived no-prior dev10 rows (needed only to bootstrap the first environment) + these 30 rows; "
                                        "every row passes no_prior_case_gate (rejects cache_dir/cache_key/cache_status) and no cache path is ever constructed",
            "split_not_built_by_this_card": True},
        "vlm_cache_test": {"entries_counted_by_name_only": prov["vlm_cache_test_entry_count"], "contents_read": False},
        "provenance_assessment_quoted": prov["assessment"],
    }
    dump(out / "test_source_freeze.json", doc)
    print(json.dumps({"n": len(order), "label": LABEL, "files_read": doc["files_read"], "cache_keys_in_active": cache_keys}, indent=1))


# ----------------------------------------------------------------------------- candidate release manifest
def cmd_release(out):
    out = Path(out)
    mat, core = load(out / "historical_model_load_matrix.json"), load(out / "core_model_identity.json")
    frozen = load(CLOSURE / "03_frozen_model_list.json")
    dev = load(out / "dev_consistency_results.json")
    frz = load(out / "test_source_freeze.json")
    sel = {m["plan_run"]: m for m in frozen["models"]}
    hist = {e["slot"]: e for e in mat["historical"]}
    corem = {m["slot"]: m for m in core["models"]}
    arch_ok = dev["class_verdicts"]["ARCHIVED_V13_EXECUTION_CLASS"] == "CONSISTENT"
    cur_ok = dev["class_verdicts"]["CURRENT_2_1_1_EXECUTION_CLASS"] == "CONSISTENT"

    def entry(slot, tier, run, method, seed, cls, include, gaps):
        src = corem.get(slot) or hist[slot]
        ck = src["checkpoint"]
        return {"slot": slot, "plan_run": run, "method": method, "seed": seed, "tier": tier, "execution_class": cls, "phase_a_class": src["phase_a_class"],
                "final_checkpoint": {"path": ck["path"], "bytes": ck["bytes"], "sha256": ck["sha256"], "N": ck["N"], "T_s": ck["T_s"], "generation": ck["generation"]},
                "evaluation_model_rule": "last valid post-update final checkpoint", "selected_best_dev_separate_column": sel[run]["selected_by_job_rule_separate_column"],
                "episodes": 30, "inclusion": include, "open_gaps": gaps}

    inc_h = "INCLUDED_BY_THIS_CARD_COMPATIBILITY_PASS" if arch_ok else "EXCLUDED_CLASS_INCONSISTENT"
    inc_r = "INCLUDED_NO_NEW_ENGINEERING_BRANCH" if arch_ok else "EXCLUDED_CLASS_INCONSISTENT"
    cur_cls, arc_cls = "CURRENT_2_1_1_EXECUTION_CLASS", "ARCHIVED_V13_EXECUTION_CLASS"
    models = [
        entry("TB-M3", "CORE", "R-TB-K-1", "B1-K", 1, cur_cls, "MUST", []),
        entry("TB-M5", "CORE", "R-TB-DK-1", "B2", 1, cur_cls, "MUST", []),
        entry("TB-M6", "CORE", "R-TB-E-0", "B1-K+E", 0, cur_cls, "MUST", []),
        entry("TB-C1", "CORE", "R-TB-E-1", "B1-K+E", 1, cur_cls, "MUST", []),
        entry("TB-M2", "HISTORICAL_EXTENSION", "R1-B1K-s0", "B1-K", 0, arc_cls, inc_h, ["UC2", "UC4", "UC5"]),
        entry("TB-M4", "HISTORICAL_EXTENSION", "R1-B2-s0", "B2", 0, arc_cls, inc_h, ["UC2", "UC4", "UC5"]),
        entry("TB-M1", "SYSTEM_REFERENCE", "R2-B0-s0", "B0", 0, arc_cls, inc_r, ["UC2", "UC3", "UC4", "UC5", "UC7"]),
    ]
    tiers = {
        "CORE": {"models": 4, "episodes_per_model": 30, "episodes": 120, "status": "MUST", "execution_class": cur_cls, "consistent": cur_ok},
        "HISTORICAL_EXTENSION": {"models": 2, "episodes": 60, "status": "INCLUDED (decided by this card, before any test result)" if arch_ok else "EXCLUDED",
                                 "condition": "compatibility PASS only; never retrain, convert weights or replace a checkpoint to obtain it", "execution_class": arc_cls},
        "SYSTEM_REFERENCE": {"models": 1, "episodes": 30, "status": "INCLUDED (no new engineering branch)" if arch_ok else "EXCLUDED", "execution_class": arc_cls},
    }
    max_eps = 120 + (90 if arch_ok else 0)
    hashes_rec = {k: v["recorded_in_run_manifest"] for k, v in hist["TB-M2"]["archived_identity"]["recorded_vs_source"].items()}
    cur_rm = E1_LAUNCH / "runtime_manifest_tb_resolved.yaml"
    doc = {
        "document": "candidate_holdout_release_manifest", "card": "CP-DISR-TB-EVAL-PREP-LITE-01", "status": "CANDIDATE_NOT_EXECUTED_NOT_AUTHORIZED",
        "baseline": BASELINE, "task": "T_B", "test_label": LABEL, "not_strict_blind_test": True,
        "scope_statement": "T_B-specific evaluation preparation; GLOBAL_S4 = NOT_COMPLETE; GLOBAL_S5 = NOT_COMPLETE; this is not the project-wide S6",
        "authorization": {"formal_holdout_authorized": False, "no_episode_launched": True,
                          "requires": "a separate formal holdout release card stating the S4 / S5 status for T_B and approving this manifest"},
        "plan_revision": {
            "original_plan_v3": "6 x 30 base + conditional B1-K+E seed1 x 30 (210 maximum; the conditional model is outside the 628 cap)",
            "current_local_priority_revision": "CORE 120 first (4 models x 30); historical models are a conditional extension decided by the compatibility classification of this card, never by CORE test results",
            "disclosure": "This is a revision of the local evaluation priority made in CP-DISR-TB-EVAL-PREP-LITE-01. It is NOT what Plan v3 originally said and must not be reported as such.",
            "forbidden": "retraining, weight conversion, checkpoint replacement or a complex historical adapter in order to reach 210"},
        "tiers": tiers,
        "episode_budget": {"core": 120, "historical_extension_max": 60, "system_reference_max": 30, "maximum_if_all_included": max_eps,
                           "retraining_or_conversion_to_reach_210": "forbidden"},
        "extension_decision": {"decided_by": "compatibility classification (Phase A) + dev consistency (Phase C) of this card", "decided_before_any_test_result": True,
                               "must_not_be_changed_after_seeing_CORE_test_results": True},
        "models": models,
        "evaluation_model_rule": "each run's last valid post-update final checkpoint; selected / best-dev appear only in a separate column and never replace final",
        "execution_classes": {
            cur_cls: {"source_tree": str(ROOT), "execution_hashes": mat["current_execution_class"]["execution_hashes"],
                      "runtime_manifest": str(cur_rm), "runtime_manifest_sha256": sha256_file(cur_rm),
                      "binding": "stage2a_v11.bind_run_context(duck-typed context: runtime_manifest, render_gpu_device_id=0, prior_mode=absent) + ENABLED_SPLITS[T_B] set; recorder imported normally",
                      "dev_consistency": dev["class_verdicts"][cur_cls]},
            arc_cls: {"source_tree": str(V2_1), "v2_1_head": mat["archived_execution_class"]["v2_1_head"], "commits": mat["archived_execution_class"]["commits"],
                      "execution_hashes_recorded_in_run_manifests": hashes_rec, "runtime_manifest_in_tree": "experiments/manifests/runtime_manifest_v211.yaml",
                      "binding": "root = the archived tree (read-only use); PYTHONPATH = its src; ENABLED_SPLITS[T_B] set to an absolute no-prior split; the recorder is loaded BY FILE PATH from the new worktree; "
                                 "no archived file is modified and no run context exists in that source",
                      "dev_consistency": dev["class_verdicts"][arc_cls]}},
        "reporting_rules": ["results are reported per tier and per execution class; seed-0 historical models are listed apart from the seed-1 / current models and never pooled without an explicit statement",
                            "time columns: time_to_first_confirmed_success_s is the only episode-level success time; success_seconds is FINAL_SKILL_DURATION_ONLY",
                            "label every test table " + LABEL],
        "evaluator_settings": {"action": "deterministic argmax", "episode_deadline_sim_seconds": 60, "passes_per_model": 1, "case_order": "fixed (test_cases.order), identical for every model",
                               "rng": "isolated (captured, seed_all(0), restored)", "optimizer_steps": 0, "provider_calls": 0,
                               "recorder": "src/cp_disr/final_tb_eval_record.py (schema eval_row_v2)", "model_hash": "sha256 before load and after the last episode; mismatch stops",
                               "prior": "absent for every model (effective R empty); no-prior test split without cache pointers"},
        "test_cases": {"label": LABEL, "n": 30, "order": frz["active_cases"]["order"], "source": "configs/splits/T_B_stage_2a_test30.json",
                       "scene_set_sha256": frz["scene_set"]["scene_set_sha256"], "freeze_record": "test_source_freeze.json"},
        "invalid_episode_rule": {
            "NORMAL_POLICY_OUTCOME": {"counted": "as failure", "rerun": False,
                                      "examples": ["DEADLINE", "NO_CANDIDATE_SAFE_TERMINATION (NO_SAFE_CANDIDATES)", "legal skill failure", "verifier-confirmed task failure", "INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE"]},
            "INFRASTRUCTURE_FAILURE": {"examples": ["CUDA / EGL / process crash", "filesystem / checkpoint read error", "evaluation code exception", "environment construction failure"],
                                       "action": "stop that release and report", "forbidden": ["automatic scene swap", "rerun until satisfied", "partial-result cherry-picking"],
                                       "implemented_in_recorder": "any exception escaping an episode writes a partial payload (status INFRASTRUCTURE_FAILURE_STOPPED) and raises InfrastructureFailure"}},
        "prerequisites_before_release": [
            {"id": "P1", "item": "S4 / S5 status for T_B stated explicitly by the user", "state": "NOT_CONFIRMED (GLOBAL_S4 / GLOBAL_S5 = NOT_COMPLETE)"},
            {"id": "P3", "item": "recorder implemented, offline-tested, committed", "state": "DONE in this card (engineering commit)"},
            {"id": "P4", "item": "test_release_manifest with model/Adam index, input hashes, evaluator and RNG settings", "state": "THIS FILE is the candidate; the release card must bind it to a committed source and fill UC5 items"},
            {"id": "P6", "item": "runner that takes the frozen list (old evaluate_final_tests needs an 8-checkpoint selection manifest)", "state": "NOT DONE - a small runner around evaluate_checkpoint_recorded is the only new code; no production module edit"},
            {"id": "P7", "item": "no-prior test split without cache pointers", "state": "RULE FROZEN, split not built (see test_source_freeze.json)"},
            {"id": "P9", "item": "compute / GPU / disk plan", "state": "measured here: ~15-35 s per dev episode incl. environment build; GPU chosen at release time (shared host)"},
            {"id": "U1", "item": "source qualification of test30 (prematerialized)", "state": "ACCEPTED AS LABEL " + LABEL + ", not strict blind test"}],
        "recorder_hardening_for_release_card": "evaluate_checkpoint_recorded does not pre-check that its output path is writable before running episodes (one dev attempt lost its result this way); add a pre-flight probe in the release runner or recorder (changes the module hash; re-test)",
        "open_gaps_carried": ["UC2 (two stage2a_v11 evaluators; reduced by one-case dev consistency per class, not eliminated)", "UC3 (R2 B0 original location inaccessible; local copy internally consistent)",
                              "UC4 (commit pointers differ between manifests and ledger text)", "UC5 (weights-to-input chain for historical runs, built at release)", "UC7 (R2 B0 common_dev20 file holds 10 cases)"],
        "hard_stops": ["any model hash change", "any infrastructure failure", "any execution-class inconsistency", "any attempt to read test VLM cache / relation truth", "no selection, tuning or retraining on test"],
    }
    dump(out / "candidate_holdout_release_manifest.json", doc)
    print(json.dumps({"max_episodes": max_eps, "tiers": {k: v["episodes"] for k, v in tiers.items()}}))


# ----------------------------------------------------------------------------- summary
def cmd_summary(out):
    out = Path(out)
    off, dev = load(out / "offline_test_receipt.json"), load(out / "dev_consistency_results.json")
    frz = load(out / "test_source_freeze.json")
    arc = dev["archived_class"]
    g = arc["old_record_comparison"]["G"]["new"]
    ss = arc["old_record_comparison"]["success_seconds"]["new"]
    tt = arc["new_fields_summary"]["time_to_first_confirmed_success_s"]
    md = f"""# CP-DISR-TB-EVAL-PREP-LITE-01 - T_B unified evaluation preparation (mainline first)

Baseline `{BASELINE[:9]}`, branch `codex/cp-disr-tb-eval-prep`. Preparation only: no formal holdout, no test episode, no test score, no RL / provider / optimizer / training.
`GLOBAL_S4 = NOT_COMPLETE`, `GLOBAL_S5 = NOT_COMPLETE`; this is T_B-specific evaluation preparation, not the project-wide S6.

## Frozen priority (disclosed revision)

CORE (MUST): B1-K seed1 (R-TB-K-1), B2 seed1 (R-TB-DK-1), B1-K+E seed0 (R-TB-E-0), B1-K+E seed1 (R-TB-E-1): 4 x 30 = **120** episodes.
HISTORICAL_EXTENSION: B1-K s0 and B2 s0, up to +60, decided here by compatibility, not by test results. SYSTEM_REFERENCE: B0 s0, up to +30, only because it needs no new engineering. Maximum 210; never reached by retraining, conversion or a complex adapter.
Original Plan v3 was 6 x 30 base + conditional E1 x 30. The CORE-120-first ordering is a **local priority revision made in this card**, not what Plan v3 said.
Main-evaluation model = each run's last valid post-update final checkpoint; selected / best-dev only in a separate column. Test label: `{LABEL}` (not a strict blind test).

## Phase A - historical model compatibility (0 environment, 0 episode, 0 optimizer)

All seven final checkpoints re-hashed equal to the frozen list and their sidecars. The four CORE models load strictly under the current code (class A). The three historical models fail strict load under the current Policy (14 `effect_readout.*` keys absent) and were **not** converted; they load strictly under the unmodified archived source whose hashes equal their run manifests (class B). B1-K s0, B2 s0 (0eb3776d) and B0 s0 (228517ed) share identical execution-file hashes, so there is one archived execution class and no third class. No model fell into class C.

## Phase B - recorder `eval_row_v2` (offline)

New evaluation-only module `src/cp_disr/final_tb_eval_record.py`; stage2a_v11 / collector / neural / torch_rl / runtime_factory / controller / Verifier / Evaluator / reward / deadline / termination / checkpoint files are byte-identical to the baseline. {off['n_passed']}/{off['n_tests']} offline tests pass, including an old-field comparison against the real `stage2a_v11.eval_episodes`, and all {len(off['mutation_checks'])} injected recorder defects are caught.

## Phase C - consistency on frozen dev case T_B_dev_00 (3 episodes, cap 3)

| class | model | old fields (success, G, steps, reason, success_seconds) | internal consistency |
|---|---|---|---|
| current 2.1.1 | R-TB-K-1 (fails: 11 actions, deadline-capped) | identical | pass |
| archived v13 | R1 B2 seed0 (succeeds: 5 actions) | identical (G = {g}) | pass |

On the success case the old `success_seconds` is {ss:.2f} s (final skill only) while the episode took {tt:.2f} s simulated to the first independently confirmed success, the quantity the old record could not give.
Budget disclosure: the first archived attempt ran its episode but its result file was lost (relative output path after `chdir`); nothing from it was used, it counts against the cap, and the re-run was the third and last episode. The worker now resolves paths first and probes the output directory before building an environment. The recorder itself does not pre-check its output path; adding that would change its hash and need re-testing, so it is a release-card hardening item.
Caveats: the current-class episode is a failing case (no real-environment success-path timing on the current class); the archived-class evidence is one B2 episode, and B1-K / B0 share the class by recorded hashes.

## Decision (made before any test result)

Both execution classes are consistent, so HISTORICAL_EXTENSION (+60) and SYSTEM_REFERENCE (+30) are **included** in the candidate manifest, reported per execution class. Gaps carried, not repaired: UC2 (two evaluator versions; one-case differential only), UC3 (B0 original location inaccessible), UC4, UC5, UC7.

## Test source (frozen, nothing opened)

30 active ids in fixed order, reset seeds and row hashes recorded from the committed split configs; scene-set sha `{frz['scene_set']['scene_set_sha256'][:16]}...` from committed provenance. No test RGB/depth, VLM cache, relation truth or test result was opened. The future runner must build a no-prior split from `T_B_stage_2a_test30.json` rows (no cache-pointer keys) and never use `T_B_stage_2a_v11.json` test rows.

## Not done / next

No formal holdout was authorised or started. A separate holdout release card must state the S4 / S5 status for T_B, bind the candidate manifest to a committed source, add the small frozen-list runner around `evaluate_checkpoint_recorded`, and build the no-prior test split.
"""
    (out / "final_prep_summary.md").write_text(md, encoding="utf-8")
    print("summary written")


# ----------------------------------------------------------------------------- verify
def cmd_verify(out):
    out = Path(out)
    checks = {}
    src_id, dev = load(out / "evaluation_module_source_identity.json"), load(out / "dev_consistency_results.json")
    off, rel = load(out / "offline_test_receipt.json"), load(out / "candidate_holdout_release_manifest.json")
    frz, mat, core = load(out / "test_source_freeze.json"), load(out / "historical_model_load_matrix.json"), load(out / "core_model_identity.json")
    required = ["historical_model_load_matrix.json", "historical_model_load_report.md", "core_model_identity.json", "eval_row_v2_schema.json",
                "evaluation_module_source_identity.json", "offline_test_receipt.json", "dev_consistency_results.json", "test_source_freeze.json",
                "candidate_holdout_release_manifest.json", "final_prep_summary.md"]
    checks["all_required_outputs_present"] = all((out / f).is_file() and (out / f).stat().st_size > 0 for f in required)
    for f in required:
        if f.endswith(".json"):
            json.loads((out / f).read_text(encoding="utf-8"))
    checks["all_json_parse"] = True
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    checks["branch_is_codex_cp_disr_tb_eval_prep"] = branch == "codex/cp-disr-tb-eval-prep"
    checks["baseline_is_ancestor"] = subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], cwd=str(ROOT)).returncode == 0
    changed = git("diff", "--name-only", BASELINE, "--", "src", "configs", "experiments").splitlines()
    checks["tracked_changes_src_configs_experiments_vs_baseline"] = changed
    checks["only_new_recorder_module_added_under_src"] = changed == ["src/cp_disr/final_tb_eval_record.py"]
    prot = {r: git("rev-parse", "%s:%s" % (BASELINE, r)) == git("hash-object", r) for r in src_id["protected_production_files"]}
    checks["protected_production_files_unchanged"] = all(prot.values())
    checks["protected_file_count"] = len(prot)
    checks["src_and_tests_equal_engineering_commit"] = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "tests"], cwd=str(ROOT)).returncode == 0
    checks["scripts_have_no_unstaged_changes"] = subprocess.run(["git", "diff", "--quiet", "--", "scripts"], cwd=str(ROOT)).returncode == 0
    checks["recorder_blob_in_HEAD_equals_worktree"] = git("rev-parse", "HEAD:src/cp_disr/final_tb_eval_record.py") == git("hash-object", "src/cp_disr/final_tb_eval_record.py")
    rec_hashes = {dev["current_class"]["source_hashes"]["recorder"], dev["archived_class"]["source_hashes"]["recorder"]}
    checks["both_dev_episodes_used_the_committed_recorder_bytes"] = rec_hashes == {sha256_file(ROOT / "src/cp_disr/final_tb_eval_record.py")}
    checks["offline_tests_all_pass"] = off["pytest_returncode"] == 0 and off["n_failed"] == 0 and off["all_required_checks_pass"]
    checks["all_mutations_detected"] = off["all_mutations_detected"]
    rehash = {e["slot"]: sha256_file(e["checkpoint"]["path"]) == e["checkpoint"]["sha256"] for e in mat["historical"] + mat["core"]}
    checks["all_seven_final_checkpoints_unchanged"] = all(rehash.values())
    checks["historical_classes"] = {e["slot"]: e["phase_a_class"] for e in mat["historical"]}
    checks["core_all_exact_load"] = core["all_core_exact_load"]
    checks["no_class_C"] = not any(e["phase_a_class"].startswith("C") for e in mat["historical"])
    b = dev["budget"]
    checks["phase_c_budget_within_cap"] = (b["episodes_executed"] <= 3 and b["environment_constructions"] <= 3 and b["provider_calls"] == 0
                                           and b["optimizer_steps"] == 0 and b["test_episodes"] == 0)
    checks["dev_case_is_frozen_T_B_dev_00"] = dev["current_class"]["case"] == "T_B_dev_00" and dev["archived_class"]["case"] == "T_B_dev_00"
    checks["both_classes_consistent"] = all(v == "CONSISTENT" for v in dev["class_verdicts"].values())
    checks["no_test_material_opened"] = all(v is False for v in frz["not_done"].values())
    allowed = {str(ROOT / "configs/splits/T_B_stage_2a_test30.json"), str(ROOT / "configs/splits/T_B_stage_2a_test_ids.json"),
               str(CLOSURE / "06_test_source_provenance_check.json")}
    checks["test_freeze_files_read_are_only_allowed_ones"] = set(frz["files_read"]) <= allowed and len(frz["files_read"]) == 3
    checks["test_label_is_prematerialized_independent_holdout"] = frz["label"] == LABEL and rel["test_label"] == LABEL and frz["not_strict_blind_test"] is True
    t = rel["tiers"]
    checks["tier_counts_120_60_30_max_210"] = (t["CORE"]["episodes"], t["HISTORICAL_EXTENSION"]["episodes"], t["SYSTEM_REFERENCE"]["episodes"],
                                               rel["episode_budget"]["maximum_if_all_included"]) == (120, 60, 30, 210)
    checks["release_not_authorized_not_executed"] = rel["status"] == "CANDIDATE_NOT_EXECUTED_NOT_AUTHORIZED" and rel["authorization"]["formal_holdout_authorized"] is False
    checks["plan_revision_disclosed"] = "NOT what Plan v3 originally said" in rel["plan_revision"]["disclosure"]
    checks["global_s4_s5_not_complete_stated"] = "GLOBAL_S4 = NOT_COMPLETE" in rel["scope_statement"] and "GLOBAL_S5 = NOT_COMPLETE" in rel["scope_statement"]
    import re
    psout = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,args="], capture_output=True, text=True).stdout.splitlines()
    pat = re.compile(r"^\s*(\d+)\s+\S*python\S*\s+(?:-\S+\s+)*\S*final_tb_eval_prep_(?:dev_check|load_check)\.py")
    left = [l for l in psout if pat.match(l)]
    checks["no_leftover_worker_processes"] = left == []
    checks["archived_scratch_removed"] = not Path("/home/xushijie2/tmp/tb_eval_prep_arch").exists()
    newer = subprocess.run("find %s -type f -not -path '*/.git/*' -newer %s 2>/dev/null | head -5" % (V2_1, out / "phase_a/frozen_models_input.json"),
                           shell=True, capture_output=True, text=True).stdout.split()
    checks["v2_1_worktree_untouched_since_prep_start"] = newer == []
    bools = {k: v for k, v in checks.items() if isinstance(v, bool)}
    checks["failed_checks"] = [k for k, v in bools.items() if not v]
    checks["all_checks_pass"] = not checks["failed_checks"]
    dump(out / "verify.json", {"card": "CP-DISR-TB-EVAL-PREP-LITE-01", "head": git("rev-parse", "HEAD"), "baseline": BASELINE, "branch": branch,
                               "checks": checks, "protected_files": prot, "final_checkpoint_rehash": rehash})
    print(json.dumps({"all_checks_pass": checks["all_checks_pass"], "failed": checks["failed_checks"]}, indent=1))
    return 0 if checks["all_checks_pass"] else 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dev-consistency", "test-source-freeze", "release-manifest", "summary", "verify"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"dev-consistency": cmd_dev, "test-source-freeze": cmd_freeze, "release-manifest": cmd_release, "summary": cmd_summary, "verify": cmd_verify}[a.cmd](a.out) or 0


if __name__ == "__main__":
    sys.exit(main())
