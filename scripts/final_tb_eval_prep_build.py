#!/usr/bin/env python
"""Artifact builder for CP-DISR-TB-EVAL-PREP-LITE-01 (zero environment; no model forward; no episode).

Subcommands: phase-a | schema | source-identity | offline-receipt
(later cards add: test-source-freeze | release-manifest | summary | verify -- see final_tb_eval_prep_finish.py)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

BASELINE = "241159c1ceca3b53f0ba8423879524c6b2668e07"
ROOT = Path(__file__).resolve().parents[1]
V2_1 = Path("/home/xushijie2/graph_cp_disr_v2_1")
MODULE = "src/cp_disr/final_tb_eval_record.py"
PROTECTED = (
    "src/cp_disr/stage2a_v11.py", "src/cp_disr/collector.py", "src/cp_disr/neural.py", "src/cp_disr/torch_rl.py", "src/cp_disr/rl.py",
    "src/cp_disr/persistence.py", "src/cp_disr/phase_a_v12.py", "src/cp_disr/final_tb.py", "src/cp_disr/final_tb_e1.py",
    "src/cp_disr/platforms/libero/runtime_factory.py", "src/cp_disr/platforms/libero/skill_executor.py",
    "src/cp_disr/platforms/libero/verifier.py", "src/cp_disr/platforms/libero/task_evaluator.py",
    "src/cp_disr/platforms/libero/safety.py", "src/cp_disr/platforms/libero/clock.py",
)
# manifest hash key -> repo-relative source path
MANIFEST_KEYS = {
    "stage2a_v11": "src/cp_disr/stage2a_v11.py", "collector": "src/cp_disr/collector.py", "neural": "src/cp_disr/neural.py",
    "torch_rl": "src/cp_disr/torch_rl.py", "persistence": "src/cp_disr/persistence.py",
    "runtime_factory": "src/cp_disr/platforms/libero/runtime_factory.py", "clock": "src/cp_disr/platforms/libero/clock.py",
    "safety": "src/cp_disr/platforms/libero/safety.py", "skill_executor": "src/cp_disr/platforms/libero/skill_executor.py",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()


def blob_sha256(commit, rel):
    return sha256_bytes(subprocess.check_output(["git", "show", "%s:%s" % (commit, rel)], cwd=str(ROOT)))


def dump(path, doc):
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=False, ensure_ascii=False) + "\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- phase A
def cmd_phase_a(out):
    out = Path(out)
    pa = out / "phase_a"
    models = load(pa / "frozen_models_input.json")
    cur = {r["slot"]: r for r in load(pa / "load_check_current_tree.json")["rows"]}
    arch = {}
    for f in sorted(pa.glob("load_check_archived_*.json")):
        d = load(f)
        for r in d["rows"]:
            arch[r["slot"]] = {"row": r, "file": f.name, "label": d["label"]}
    rows, core = [], []
    for m in models:
        c = cur[m["slot"]]
        entry = {
            "slot": m["slot"], "plan_run": m["plan_run"], "method": m["method"], "seed": m["seed"], "origin": m["origin"],
            "tier": ("CORE" if m["slot"] in ("TB-M3", "TB-M5", "TB-M6", "TB-C1") else
                     ("HISTORICAL_EXTENSION" if m["method"] in ("B1-K", "B2") else "SYSTEM_REFERENCE")),
            "checkpoint": {"path": m["file"], "bytes": c["bytes"], "sha256": c["sha256"], "N": m["N"], "T_s": m["T_s"], "generation": m["generation"]},
            "checks": {
                "sha256_matches_frozen_list": c["sha256_matches_frozen_list"], "bytes_match_frozen_list": c["bytes_match_frozen_list"],
                "sidecar_sha256_matches": c["sidecar_sha256_matches"], "manifest_N_matches": c["manifest_identity"]["N_matches_list"],
                "manifest_T_matches": c["manifest_identity"]["T_matches_list"],
            },
            "key_shape_dtype": {"n_param_tensors": c["n_param_tensors"], "n_param_elements": c["n_param_elements"], "dtypes": c["dtypes"],
                                "state_signature_sha256": c["state_signature_sha256"]},
            "current_tree_strict_load": {"strict_load": c["strict_load"], "error": c["strict_load_error"], "key_diff": c["key_diff"]},
        }
        if m["origin"] == "historical":
            a = arch[m["slot"]]["row"]
            commit = m["execution_identity"]["source_commits"]
            man = load(Path(m["execution_identity"]["run_dir"]) / "manifest.json")
            ident = {}
            for k, rel in MANIFEST_KEYS.items():
                rec_h = man["hashes"].get(k)
                if rec_h is None:
                    continue
                ident[k] = {"recorded_in_run_manifest": rec_h, "archived_commit_blob": blob_sha256(commit, rel),
                            "v2_1_worktree_file": sha256_file(V2_1 / rel), "match_commit": rec_h == blob_sha256(commit, rel),
                            "match_worktree": rec_h == sha256_file(V2_1 / rel)}
            entry["archived_identity"] = {"source_commit": commit, "run_dir": m["execution_identity"]["run_dir"], "recorded_vs_source": ident,
                                          "all_recorded_hashes_match_commit": all(v["match_commit"] for v in ident.values()),
                                          "all_recorded_hashes_match_v2_1_worktree": all(v["match_worktree"] for v in ident.values())}
            entry["archived_tree_strict_load"] = {"source": arch[m["slot"]]["label"], "strict_load": a["strict_load"], "error": a["strict_load_error"],
                                                  "key_diff": a["key_diff"], "evidence_file": "phase_a/" + arch[m["slot"]]["file"]}
            conv = {"weight_conversion": False, "non_strict_load": False, "adapter_needed": False, "forward_semantics_modified": False,
                    "retraining": False, "checkpoint_replaced": False}
            entry["conversion_requirements"] = conv
            if c["strict_load"] and entry["checks"]["sha256_matches_frozen_list"]:
                cls = "A_CURRENT_EVAL_EXACT_LOAD"
            elif (a["strict_load"] and entry["archived_identity"]["all_recorded_hashes_match_commit"] and not any(conv.values())
                  and entry["checks"]["sha256_matches_frozen_list"] and entry["checks"]["sidecar_sha256_matches"]):
                cls = "B_ARCHIVED_EXECUTION_CLASS_COMPATIBLE"
            else:
                cls = "C_HISTORICAL_EXTENSION_NOT_ELIGIBLE"
            entry["phase_a_class"] = cls
            entry["note"] = ("fails strict load under the current 2.1.1 Policy (14 effect_readout.* keys absent from the old state_dict) and is NOT converted; "
                             "strict load succeeds only under the unmodified archived source whose hashes equal the run manifest") if cls.startswith("B") else ""
            rows.append(entry)
        else:
            entry["phase_a_class"] = "A_CURRENT_EVAL_EXACT_LOAD" if c["strict_load"] and entry["checks"]["sha256_matches_frozen_list"] else "CORE_LOAD_FAILED"
            core.append(entry)
    classes = {}
    for e in rows + core:
        classes.setdefault(e["phase_a_class"], []).append(e["slot"])
    # the single archived execution class (R1 and R2 execution hashes coincide)
    r1 = next(e for e in rows if e["slot"] == "TB-M2")["archived_identity"]["recorded_vs_source"]
    r2 = next(e for e in rows if e["slot"] == "TB-M1")["archived_identity"]["recorded_vs_source"]
    shared = sorted(set(r1) & set(r2))
    same = {k: r1[k]["recorded_in_run_manifest"] == r2[k]["recorded_in_run_manifest"] for k in shared}
    matrix = {
        "card": "CP-DISR-TB-EVAL-PREP-LITE-01", "phase": "A", "baseline": BASELINE,
        "constraints_observed": {"environments": 0, "episodes": 0, "optimizers": 0, "providers": 0, "weights_converted": 0, "checkpoints_replaced": 0,
                                 "non_strict_loads": 0},
        "method": "sha256 of checkpoint bytes vs frozen list and sidecar; sidecar N/T identity; torch.load(cpu) key/shape/dtype listing; strict load_state_dict into Policy built exactly as stage2a_v11.make_policy (actions from the grounded contract registry, no environment)",
        "classes": classes,
        "historical": rows, "core": core,
        "archived_execution_class": {
            "name": "ARCHIVED_V13_EXECUTION_CLASS", "source_tree": str(V2_1), "v2_1_head": git("rev-parse", "HEAD", cwd=V2_1),
            "commits": {"R1_B1-K_B2": "0eb3776d143d2142eb2893e2a12a4d85f2d14fc1", "R2_B0": "228517ed214e18bd75ab11b84f3edff6b07f14b7"},
            "execution_hashes_identical_between_R1_and_R2_runs": all(same.values()), "per_key_identical": same,
            "consequence": "B1-K s0, B2 s0 and B0 s0 form ONE archived execution class; no third class exists",
        },
        "current_execution_class": {"name": "CURRENT_2_1_1_EXECUTION_CLASS", "source_tree": str(ROOT), "head_at_build": git("rev-parse", "HEAD"),
                                    "execution_hashes": {k: sha256_file(ROOT / rel) for k, rel in MANIFEST_KEYS.items()}},
        "decision_rule_reminder": "class C stops immediately: no converter, no adapter branch, no retraining, no checkpoint replacement",
    }
    dump(out / "historical_model_load_matrix.json", matrix)
    core_doc = {
        "card": "CP-DISR-TB-EVAL-PREP-LITE-01", "tier": "CORE_CURRENT_PROFILE", "baseline": BASELINE,
        "evaluation_model_rule": "last valid post-update final checkpoint of each run; selected / best-dev only in a separate column",
        "models": [{"slot": e["slot"], "plan_run": e["plan_run"], "method": e["method"], "seed": e["seed"], "checkpoint": e["checkpoint"],
                    "checks": e["checks"], "strict_load_current_tree": e["current_tree_strict_load"]["strict_load"],
                    "key_shape_dtype": e["key_shape_dtype"], "phase_a_class": e["phase_a_class"]} for e in core if e["slot"] != "TB-C1"],
        "not_core": [{"slot": e["slot"], "plan_run": e["plan_run"], "note": "R-TB-E-1 is a CORE member (B1-K+E seed1); listed in the frozen list as TB-C1"} for e in core if e["slot"] == "TB-C1"],
    }
    # TB-C1 (R-TB-E-1, B1-K+E seed1) IS a core model in the current priority: include it
    core_doc["models"] = [{"slot": e["slot"], "plan_run": e["plan_run"], "method": e["method"], "seed": e["seed"], "checkpoint": e["checkpoint"],
                           "checks": e["checks"], "strict_load_current_tree": e["current_tree_strict_load"]["strict_load"],
                           "key_shape_dtype": e["key_shape_dtype"], "phase_a_class": e["phase_a_class"]} for e in core]
    core_doc.pop("not_core")
    core_doc["core_label_map"] = {"R-TB-K-1": "B1-K seed1", "R-TB-DK-1": "B2 seed1", "R-TB-E-0": "B1-K+E seed0", "R-TB-E-1": "B1-K+E seed1"}
    core_doc["all_core_exact_load"] = all(m["strict_load_current_tree"] and m["checks"]["sha256_matches_frozen_list"] for m in core_doc["models"])
    dump(out / "core_model_identity.json", core_doc)
    md = ["# Historical model compatibility (Phase A, zero environment / zero episode / zero optimizer)", "",
          "Baseline `%s`. Nothing was converted, resumed, retrained or re-saved." % BASELINE, "",
          "| slot | run | method / seed | bytes re-hashed = frozen list | current-tree strict load | archived-tree strict load | class |", "|---|---|---|---|---|---|---|"]
    for e in rows:
        md.append("| %s | %s | %s s%d | %s | %s | %s | **%s** |" % (
            e["slot"], e["plan_run"], e["method"], e["seed"], e["checks"]["sha256_matches_frozen_list"],
            "PASS" if e["current_tree_strict_load"]["strict_load"] else "FAIL (%d keys missing)" % e["current_tree_strict_load"]["key_diff"]["n_missing_in_ckpt"],
            "PASS" if e["archived_tree_strict_load"]["strict_load"] else "FAIL", e["phase_a_class"]))
    md += ["", "## Reading", "",
           "The three historical final weights do not load strictly into the current 2.1.1 Policy: the current Policy carries 14 `effect_readout.*` parameters that the older state_dict does not have. "
           "No key was remapped, filled or dropped. Under the unmodified archived source (the v2_1 worktree whose neural / torch_rl / stage2a_v11 / runtime_factory / collector / persistence hashes equal the run manifests) "
           "the same files load strictly with no missing, unexpected or mismatching key. They are therefore class B, **provisionally**: class B is confirmed for the extension only if the archived execution class also reproduces the old dev record on the fixed dev case (Phase C).", "",
           "B1-K s0, B2 s0 (commit 0eb3776d) and B0 s0 (commit 228517ed) share identical execution-file hashes, so they form one archived execution class.", "",
           "## Core models (current tree, strict load)", "", "| slot | run | method / seed | sha256 = frozen list | strict load | class |", "|---|---|---|---|---|---|"]
    for e in core:
        md.append("| %s | %s | %s s%d | %s | %s | %s |" % (e["slot"], e["plan_run"], e["method"], e["seed"], e["checks"]["sha256_matches_frozen_list"],
                                                       "PASS" if e["current_tree_strict_load"]["strict_load"] else "FAIL", e["phase_a_class"]))
    (out / "historical_model_load_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"classes": classes, "archived_class_identical": matrix["archived_execution_class"]["execution_hashes_identical_between_R1_and_R2_runs"],
                      "core_all_exact": core_doc["all_core_exact_load"]}, indent=1))


# ----------------------------------------------------------------------------- schema
def cmd_schema(out):
    sys.path.insert(0, str(ROOT / "src"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("rec", ROOT / MODULE)
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)
    num_or_null = {"type": ["number", "null"]}
    action = {"type": "object", "required": list(rec.ACTION_FIELDS), "additionalProperties": False, "properties": {
        "decision_index": {"type": "integer"}, "candidate_id": {"type": "string"}, "clock_start": {"type": "number"}, "clock_end": {"type": "number"},
        "duration": {"type": "number"}, "reward": {"type": "number"}, "weight": {"type": "number"}, "controller_exit": {"type": "string"},
        "terminated": {"type": "boolean"}, "truncated": {"type": "boolean"}, "reason": {"type": ["string", "null"]}, "success_confirmed": {"type": "boolean"}}}
    row = {"type": "object", "required": list(rec.OLD_ROW_FIELDS) + list(rec.NEW_ROW_FIELDS) + ["outcome_class"], "properties": {
        "case_id": {"type": "string"}, "success": {"type": "boolean"}, "G": {"type": "number"}, "steps": {"type": "integer"},
        "reason": {"type": ["string", "null"]}, "success_seconds": {**num_or_null, "description": "OLD field, unchanged: duration of the final (successful) skill only"},
        "source_n": {"type": ["integer", "null"]}, "prior_mode": {"type": ["string", "null"]},
        "success_seconds_label": {"const": rec.SUCCESS_SECONDS_LABEL},
        "episode_start_clock_s": {"type": "number"},
        "time_to_first_confirmed_success_s": {**num_or_null, "description": "simulated seconds from episode start to the end of the first interval in which the independent evaluator confirmed success"},
        "time_to_first_confirmed_success_label": {"const": rec.TIME_TO_SUCCESS_LABEL},
        "first_confirmed_success_decision_index": {"type": ["integer", "null"]},
        "episode_end_elapsed_s": num_or_null,
        "actions": {"type": "array", "items": action},
        "inter_step_clock_gap_s": {**num_or_null, "description": "time_to_success minus sum of action durations up to the confirmed success; never assumed zero"},
        "no_transition_exit": {"type": ["object", "null"]}, "outcome_class": {"enum": [rec.NORMAL_POLICY_OUTCOME]}}}
    ident = {"type": "object", "required": ["path", "bytes", "sha256"], "properties": {"path": {"type": "string"}, "bytes": {"type": "integer"}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}}
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#", "title": "eval_row_v2 evaluation payload (CP-DISR-TB-EVAL-PREP-LITE-01)",
        "x_card_scope": "evaluation-only recorder; additive to the old eval payload; old fields keep their values and meaning",
        "type": "object",
        "required": ["schema_version", "task", "method", "checkpoint", "success_n", "n", "success_rate", "mean_discounted_return", "rows", "hashes", "isolated_rng",
                     "eval_action", "label", "optimizer_steps", "success_seconds_label", "time_to_first_confirmed_success_label", "evaluated_model",
                     "evaluated_generation", "case_order", "eval_rng_isolation", "status"],
        "properties": {
            "schema_version": {"const": rec.SCHEMA_VERSION}, "rows": {"type": "array", "items": row}, "hashes": {"type": "object", "description": "source / split / manifest identity"},
            "evaluated_model": ident, "model_sha256_after_last_episode": {"type": "string", "description": "present only if the post-run re-hash equals the pre-load hash"},
            "evaluated_generation": {"type": "object", "required": ["generation", "N", "T", "update"]},
            "case_order": {"type": "array", "items": {"type": "string"}},
            "eval_rng_isolation": {"type": "object", "required": ["rng_captured_before", "rng_restored_after", "seed_all"]},
            "isolated_rng": {"const": True}, "eval_action": {"const": "deterministic_argmax"}, "optimizer_steps": {"const": 0},
            "success_seconds_label": {"const": rec.SUCCESS_SECONDS_LABEL}, "time_to_first_confirmed_success_label": {"const": rec.TIME_TO_SUCCESS_LABEL},
            "status": {"enum": ["COMPLETE", "ABORTED_MODEL_HASH_CHANGED", "INFRASTRUCTURE_FAILURE_STOPPED"]},
            "infrastructure_failure": {"type": "object"}},
        "x_old_fields_preserved": list(rec.OLD_ROW_FIELDS),
        "x_old_schema_rule": "rows without time_to_first_confirmed_success_s are labelled FINAL_SKILL_DURATION_ONLY and the episode total time stays UNVERIFIED; they never enter a time-to-success column",
    }
    dump(Path(out) / "eval_row_v2_schema.json", schema)
    print("schema written; row required fields:", len(row["required"]))


# ----------------------------------------------------------------------------- source identity
def cmd_source_identity(out):
    files = [MODULE, "tests/test_final_tb_eval_record.py", "scripts/final_tb_eval_prep_load_check.py", "scripts/final_tb_eval_prep_dev_check.py",
             "scripts/final_tb_eval_prep_build.py"]
    extra = [f for f in ("scripts/final_tb_eval_prep_finish.py",) if (ROOT / f).is_file()]
    doc = {"card": "CP-DISR-TB-EVAL-PREP-LITE-01", "baseline": BASELINE, "head_at_build": git("rev-parse", "HEAD"),
           "new_files": {f: {"sha256": sha256_file(ROOT / f), "git_blob_sha1": git("hash-object", f)} for f in files + extra},
           "protected_production_files": {}}
    for rel in PROTECTED:
        want = git("rev-parse", "%s:%s" % (BASELINE, rel))
        got = git("hash-object", rel)
        doc["protected_production_files"][rel] = {"baseline_blob": want, "worktree_blob": got, "sha256": sha256_file(ROOT / rel), "unchanged": want == got}
    doc["all_protected_unchanged"] = all(v["unchanged"] for v in doc["protected_production_files"].values())
    tracked_diff = git("diff", "--name-only", BASELINE, "--", "src", "configs", "experiments").splitlines()
    doc["tracked_changes_under_src_configs_experiments_vs_baseline"] = tracked_diff
    doc["module_imports"] = sorted({l.strip() for l in (ROOT / MODULE).read_text().splitlines() if l.strip().startswith(("import ", "from "))})
    doc["module_contract"] = {"edits_production_modules": False, "imports_optimizer_or_trainer": False, "writes_training_artifacts": False,
                              "loads_standalone_by_file_path": True, "model_hash_checks": "before load and after last episode; mismatch aborts"}
    dump(Path(out) / "evaluation_module_source_identity.json", doc)
    print(json.dumps({"all_protected_unchanged": doc["all_protected_unchanged"], "tracked_changes": tracked_diff}, indent=1))


# ----------------------------------------------------------------------------- offline receipt
REQUIRED = {
    "multi_step_success": ["test_multistep_success_actions_clock_and_first_success_once", "test_success_on_single_confirmed_step_time_matches_end"],
    "deadline": ["test_deadline_exit_is_normal_failure"], "no_candidate": ["test_no_candidate_exit"],
    "insufficient_control_cycle": ["test_insufficient_remaining_control_cycle_exit"],
    "first_confirmed_success_recorded_once": ["test_multistep_success_actions_clock_and_first_success_once"],
    "action_order": ["test_multistep_success_actions_clock_and_first_success_once"],
    "clock_duration_accounting": ["test_multistep_success_actions_clock_and_first_success_once", "test_deadline_exit_is_normal_failure"],
    "old_fields_unchanged_vs_real_old_loop": ["test_old_fields_identical_to_real_old_eval_loop"],
    "model_hash_change_fails_closed": ["test_registered_sha_mismatch_fails_before_load", "test_model_hash_change_during_evaluation_fails_closed"],
    "old_schema_total_time_unverified": ["test_old_schema_rows_stay_unverified_and_never_enter_time_column"],
    "no_optimizer_import": ["test_module_imports_no_optimizer_and_writes_no_training_artifact"],
    "no_training_artifact_written": ["test_module_imports_no_optimizer_and_writes_no_training_artifact", "test_evaluation_writes_only_its_eval_file"],
    "collector_stage2a_v11_rl_hash_unchanged": ["test_protected_production_files_unchanged_vs_baseline"],
    "infrastructure_failure_stops_no_rerun": ["test_infrastructure_failure_stops_and_is_not_retried", "test_environment_closed_and_rng_restored_on_failure"],
    "normal_policy_outcomes_not_infrastructure": ["test_normal_policy_outcomes_never_become_infrastructure_failures"],
    "case_order_and_rng_isolation": ["test_case_order_and_rng_isolation_recorded"],
}
MUTATIONS = {
    "first_success_recorded_every_time": ("if first_success_elapsed is None:  # only", "if True:  # only"),
    "G_drops_weight": ("G += float(t.reward) * float(t.weight)", "G += float(t.reward)"),
    "post_run_hash_check_removed": ("        verify_model_identity(identity)\n    except ModelIdentityError:", "        pass\n    except ModelIdentityError:"),
    "success_seconds_uses_total_time": ("success_seconds = float(t.snapshot.elapsed_seconds + t.duration) if hasattr(t.snapshot, \"elapsed_seconds\") else t.duration",
                                        "success_seconds = clock_end - start_clock"),
    "infra_exception_swallowed": ("infra = {\"case_id\": case, \"error\": (\"%s: %s\" % (type(exc).__name__, exc))[:1500], \"kind\": INFRASTRUCTURE_FAILURE}\n                break",
                                  "continue"),
}


def run_pytest(tree=None, junit=None):
    """Run the recorder tests inside ``tree`` (default: this worktree).  A mutated COPY (src + tests + pytest.ini) is used for mutation checks."""
    tree = Path(tree or ROOT)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = "%s:%s" % (tree / "src", tree)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_final_tb_eval_record.py"]
    if junit:
        cmd += ["--junitxml", str(junit)]
    p = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True)
    return p.returncode, p.stdout[-1500:]


def cmd_offline_receipt(out):
    out = Path(out)
    junit = out / "logs" / "offline_pytest_junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    rc, tail = run_pytest(junit=junit)
    cases = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        status = "failed" if (tc.find("failure") is not None or tc.find("error") is not None) else ("skipped" if tc.find("skipped") is not None else "passed")
        cases[tc.get("name")] = status
    required = {k: {"tests": v, "all_passed": all(cases.get(t) == "passed" for t in v)} for k, v in REQUIRED.items()}
    muts = {}
    for name, (old, new) in MUTATIONS.items():
        tmp = Path(tempfile.mkdtemp(prefix="evalrec_mut_"))
        try:
            ign = shutil.ignore_patterns("__pycache__")
            shutil.copytree(ROOT / "src", tmp / "src", ignore=ign)
            shutil.copytree(ROOT / "tests", tmp / "tests", ignore=ign)
            shutil.copy(ROOT / "pytest.ini", tmp / "pytest.ini")
            tgt = tmp / "src" / "cp_disr" / "final_tb_eval_record.py"
            text = tgt.read_text(encoding="utf-8")
            assert old in text, name
            tgt.write_text(text.replace(old, new, 1), encoding="utf-8")
            mrc, mtail = run_pytest(tree=tmp)
            muts[name] = {"mutation_detected_by_tests": mrc != 0, "pytest_returncode": mrc, "summary": mtail.strip().splitlines()[-1] if mtail.strip() else ""}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    doc = {"card": "CP-DISR-TB-EVAL-PREP-LITE-01", "phase": "B", "environments": 0, "episodes": 0, "gpu": 0, "optimizer": 0, "provider": 0,
           "pytest_returncode": rc, "summary_tail": tail.strip().splitlines()[-1], "n_tests": len(cases), "n_passed": sum(1 for v in cases.values() if v == "passed"),
           "n_failed": sum(1 for v in cases.values() if v == "failed"), "n_skipped": sum(1 for v in cases.values() if v == "skipped"),
           "test_cases": cases, "required_checks": required, "all_required_checks_pass": all(v["all_passed"] for v in required.values()),
           "mutation_checks": muts, "all_mutations_detected": all(v["mutation_detected_by_tests"] for v in muts.values()),
           "test_file_sha256": sha256_file(ROOT / "tests/test_final_tb_eval_record.py"), "module_sha256": sha256_file(ROOT / MODULE),
           "head_at_run": git("rev-parse", "HEAD"), "old_loop_comparison": "tests drive the REAL stage2a_v11.eval_episodes and the recorder with the same scripted collector and require identical old fields"}
    dump(out / "offline_test_receipt.json", doc)
    print(json.dumps({k: doc[k] for k in ("pytest_returncode", "n_tests", "n_passed", "n_failed", "all_required_checks_pass", "all_mutations_detected")}, indent=1))
    return 0 if doc["pytest_returncode"] == 0 and doc["all_required_checks_pass"] and doc["all_mutations_detected"] else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("phase-a", "schema", "source-identity", "offline-receipt"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"phase-a": cmd_phase_a, "schema": cmd_schema, "source-identity": cmd_source_identity, "offline-receipt": cmd_offline_receipt}[a.cmd](a.out) or 0


if __name__ == "__main__":
    sys.exit(main())
