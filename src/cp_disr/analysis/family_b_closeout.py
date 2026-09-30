"""Evidence assembly and independent artifact/protocol verification for Family B."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import yaml

from .family_b_pilot import (
    BASE, CARD, _registered, freeze_source_hashes, read, sha, write)

OLD_ROOT = Path("/home/xushijie2/graph_cp_disr_s4_soft_ordering_repair")
OLD_RUN = OLD_ROOT / "runs/final_master/S4/family_a_soft_ordering_repair/20260930T035311Z_2f80581b"
OLD_GATE_HASH = "eb17121eef63865ed346e98f4aa83814fe631bcff4f1661ced7131ab4ef6fd01"
OLD_STOP_HASH = "fa610aa82417d6be51d7d8889070f11df17ba2c4d4c3cce2595bd6bc8d994e52"
TABLES = (
    "relation_adjudication.csv", "representation_entry.csv",
    "candidate_discriminability.csv", "paired_consequences.csv",
    "contract_insufficiency.csv", "prior_utility.csv",
)


def _csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
            [*fields, *(key for row in rows for key in row)])))
        writer.writeheader()
        writer.writerows(rows)


def _rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def protected_inventory(root, out, stage):
    root, out = Path(root), Path(out)
    files = {
        "authoritative_plan": root / "docs/authoritative/CP_DISR_Final_Experimental_Plan_v3.md",
        "authoritative_content": root / "docs/authoritative/CP_DISR_Final_Research_Content_v5.md",
        "family_a_original_gate": OLD_RUN / "physical/technical_wave_check.json",
        "family_a_stop_state": OLD_RUN / "decision/stop_state.json",
        "family_a_amended_gate": OLD_RUN / "physical/technical_wave_check_amended.json",
    }
    values = {key: {"path": str(path), "sha256": sha(path), "size": path.stat().st_size}
              for key, path in files.items()}
    if values["family_a_original_gate"]["sha256"] != OLD_GATE_HASH or values["family_a_stop_state"]["sha256"] != OLD_STOP_HASH:
        raise ValueError("STOPPED_SOURCE_IDENTITY:old evidence drift")
    write(out / "inventory" / f"protected_{stage}.json", values)
    return values


def _relation_rows(out):
    path = Path(out) / "provider/relation_request_ledger.json"
    if not path.is_file():
        return [{"status": "NOT_RUN", "scene_id": "", "relation_id": "",
                 "schema_admitted": "", "relation_truth": "UNKNOWN",
                 "current_opportunity": "UNKNOWN", "utility_under_shared_continuation": "UNKNOWN",
                 "evidence_refs": ""}]
    rows = []
    for entry in read(path)["entries"]:
        if entry.get("state") != "SUCCESS":
            rows.append({"status": entry["state"], "scene_id": entry["scene_id"],
                         "relation_id": "", "schema_admitted": "",
                         "relation_truth": "UNKNOWN", "current_opportunity": "UNKNOWN",
                         "utility_under_shared_continuation": "UNKNOWN", "evidence_refs": ""})
            continue
        cache = Path(entry["cache_path"])
        accepted = read(cache / "accepted_relations.json")
        for r in accepted:
            rows.append({"status": "NATURAL_ACCEPTED_NOT_INDEPENDENTLY_ADJUDICATED",
                         "scene_id": entry["scene_id"], "relation_id": r["relation_id"],
                         "schema_admitted": True, "relation_truth": "UNKNOWN",
                         "current_opportunity": "STATE_DEPENDENT_NOT_YET_ADJUDICATED",
                         "utility_under_shared_continuation": "UNKNOWN",
                         "evidence_refs": str(cache)})
        if not accepted:
            rows.append({"status": "NATURAL_EMPTY_OR_ISOLATED", "scene_id": entry["scene_id"],
                         "relation_id": "", "schema_admitted": False,
                         "relation_truth": "UNKNOWN", "current_opportunity": "UNKNOWN",
                         "utility_under_shared_continuation": "UNKNOWN",
                         "evidence_refs": str(cache)})
    return rows


def _representation_rows(out):
    path = Path(out) / "representation/input_dependency.jsonl"
    if not path.is_file():
        return [{"status": "NOT_RUN", "layout": "", "context": "", "method": "",
                 "condition": "", "gradient_to_contract_successor_DK": "",
                 "gradient_to_prior_patch_DP": "", "relative_logit_u_minus_v": ""}]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prior_utility_rows(bank):
    grouped = {}
    for r in bank:
        key = (r["layout"], r["context"], r["repeat"])
        grouped.setdefault(key, {})[r["method"]] = r
    result = []
    for key, methods in grouped.items():
        a, b = methods.get("B_PLAN"), methods.get("B_PLAN+R")
        if not a or not b:
            continue
        reg_a, reg_b = a.get("regret_sim_seconds"), b.get("regret_sim_seconds")
        result.append({"layout": key[0], "context": key[1], "repeat": key[2],
                       "b_plan_regret": reg_a, "b_plan_r_regret": reg_b,
                       "shared_continuation_delta_regret": (
                           float(reg_b)-float(reg_a) if reg_a not in ("", None)
                           and reg_b not in ("", None) else ""),
                       "status": "MEASURED" if reg_a not in ("", None)
                       and reg_b not in ("", None) else "NOT_MEASURABLE"})
    return result or [{"layout": "", "context": "", "repeat": "",
                       "b_plan_regret": "", "b_plan_r_regret": "",
                       "shared_continuation_delta_regret": "", "status": "NOT_RUN"}]


def assemble(root, cfg, out):
    root, out = Path(root), Path(out)
    gate = read(out / "physical/mechanism_gate.json")
    evidence = out / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    relation = _relation_rows(out)
    _csv(evidence / TABLES[0], relation, list(relation[0]))
    representation = _representation_rows(out)
    _csv(evidence / TABLES[1], representation, list(representation[0]))
    candidate = [
        {k: r.get(k) for k in (
            "status", "layout", "context", "capacity_seed", "method", "condition",
            "natural_relation_count", "legal_candidate_count",
            "changed_patch_candidate_count", "relative_logit_u_minus_v")}
        for r in representation]
    _csv(evidence / TABLES[2], candidate, list(candidate[0]))
    physical = out / "physical/paired_consequences.csv"
    if physical.is_file():
        shutil.copyfile(physical, evidence / TABLES[3])
    else:
        _csv(evidence / TABLES[3], [{"status": "NOT_RUN"}], ["status"])
    bank_path = out / "baselines/first_decision_reference_bank.csv"
    bank = _rows(bank_path) if bank_path.is_file() else []
    contract = [
        {"layout": r["layout"], "context": r["context"], "repeat": r["repeat"],
         "nominal_cost_equal": r.get("nominal_cost_equal"),
         "paired_success_complete": r.get("paired_success_complete"),
         "current_registered_contract_scope": True,
         "public_geometry_can_sort": "REPORTED_SEPARATELY",
         "status": r.get("method_status")}
        for r in bank if r["method"] == "B_PLAN"]
    contract = contract or [{"layout": "", "context": "", "repeat": "",
                             "nominal_cost_equal": "", "paired_success_complete": "",
                             "current_registered_contract_scope": True,
                             "public_geometry_can_sort": "UNKNOWN",
                             "status": "NOT_RUN"}]
    _csv(evidence / TABLES[4], contract, list(contract[0]))
    utility = _prior_utility_rows(bank)
    _csv(evidence / TABLES[5], utility, list(utility[0]))
    loaded = {name: _rows(evidence / name) for name in TABLES}
    manifest = {
        "card_id": CARD,
        "files": {name: {"path": str(evidence / name),
                         "sha256": sha(evidence / name),
                         "rows": len(loaded[name]),
                         "status": "MEASURED" if loaded[name] and
                         loaded[name][0].get("status") not in ("NOT_RUN", "NOT_RUN_PHYSICAL_GATE")
                         else "NOT_RUN"}
                  for name in TABLES},
        "E1": "PENDING_INDEPENDENT_RELATION_TRUTH",
        "E2": "STRUCTURAL_ONLY" if any(r.get("status") == "MEASURED" for r in representation) else "PENDING",
        "E3": "STRUCTURAL_ONLY" if any(r.get("status") == "MEASURED" for r in candidate) else "PENDING",
        "E4": "MEASURED" if gate["physical_mechanism_status"] == "ESTABLISHED" else "NOT_ESTABLISHED_OR_UNRESOLVED",
        "E5": "CURRENT_CONTRACT_SCOPE_ONLY" if bank else "PENDING",
        "E6": "PENDING_INDEPENDENT_CLOSED_LOOP_BASELINES",
        "method_incremental_value": "NOT_TESTED",
    }
    write(evidence / "evidence_manifest.json", manifest)
    physical_ok = gate["physical_mechanism_status"] == "ESTABLISHED"
    review = {
        "artifact_integrity_status": gate["artifact_integrity_status"],
        "execution_protocol_status": gate["execution_protocol_status"],
        "physical_mechanism_status": gate["physical_mechanism_status"],
        "natural_prior_status": "UNKNOWN",
        "representation_status": "STRUCTURAL_ONLY" if manifest["E2"] == "STRUCTURAL_ONLY" else "PENDING",
        "method_incremental_value": "NOT_TESTED",
        "s1_equivalence_status": "PARTIAL_WITH_EXPLICIT_SCOPE",
        "recommended_next_action": (
            "REQUEST_S2_FAMILY_B_CONSTRUCTION_REVIEW" if physical_ok else
            "DO_NOT_PROMOTE_FAMILY_B_WITHOUT_NEW_REVIEW"),
        "tp_training_authorized": False, "s2_authorized": False, "s3_authorized": False,
    }
    write(out / "decision/eligibility_review.json", review)
    handoff = {
        **review, "card_id": CARD, "base_commit": BASE,
        "main_performance_claim": "NOT_TESTED",
        "two_layout_pilot_not_generalization": True,
        "shared_continuation_bank_not_closed_loop": True,
        "natural_prior_truth_not_inferred_from_route_cost": True,
    }
    (out / "decision/main_experiment_handoff.yaml").write_text(yaml.safe_dump(handoff, sort_keys=True))
    (out / "final_summary.md").write_text(
        "# Family B staging qualification\n\n"
        f"Artifact integrity: {review['artifact_integrity_status']}; "
        f"protocol: {review['execution_protocol_status']}; "
        f"physical mechanism: {review['physical_mechanism_status']}.\n\n"
        "Method incremental performance: NOT_TESTED. S2/S3 training and formal test remain unauthorized.\n")
    return review


def verify(root, cfg, out, scope):
    root, out = Path(root), Path(out)
    if scope not in ("artifacts", "protocol", "all"):
        raise ValueError("unknown verification scope")
    errors = []
    source = read(out / "source_identity.json")
    if source["runtime_sources"] != freeze_source_hashes(root):
        errors.append("runtime source hashes drifted")
    before = read(out / "inventory/protected_before.json")
    after = protected_inventory(root, out, "after")
    if before != after:
        errors.append("protected evidence changed")
    manifest = read(out / "evidence/evidence_manifest.json")
    for name, item in manifest["files"].items():
        path = Path(item["path"])
        if not path.is_file() or sha(path) != item["sha256"]:
            errors.append("evidence hash mismatch:" + name)
    branches = _registered(out)
    budget = read(out / "physical/budget_ledger.json")["physical_witness_episodes"]
    root_budget = read(out / "budget_ledger.json")
    if budget["used"] > 24 or root_budget["branch_attempts"]["used"] != budget["used"]:
        errors.append("branch budget mismatch")
    if root_budget["explicit_resets"]["used"] > 24 or root_budget["live_skill_calls"]["used"] > 144:
        errors.append("execution budget exceeded")
    results = [b for b in branches if
               (out / "physical/branch_results" / f'{b["branch_id"]}.json').is_file()]
    gate_path = out / "physical/technical_gate.json"
    tech = read(gate_path) if gate_path.is_file() else {"status": "NOT_RUN"}
    if tech["status"] == "FAIL" and not (0 < budget["used"] <= 4):
        errors.append("technical fail dispatched beyond four")
    if tech["status"] == "PASS" and budget["used"] not in (4, 24):
        errors.append("unexpected release count")
    if len(results) != budget["used"]:
        errors.append("missing attempted branch result")
    mech = read(out / "physical/mechanism_gate.json")
    if mech["physical_mechanism_status"] != "ESTABLISHED":
        for name in ("relation_request_ledger.json", "action_request_ledger.json"):
            if (out / "provider" / name).is_file():
                errors.append("provider called without physical gate")
    if root_budget["rl_transitions"]["used"] != 0 or root_budget["optimizer_steps"]["used"] != 0:
        errors.append("unauthorized learning work")
    report = {"scope": scope, "status": "PASS" if not errors else "FAIL",
              "errors": errors, "branch_attempts": budget["used"],
              "observed_results": len(results), "technical_gate": tech["status"],
              "physical_mechanism": mech["physical_mechanism_status"],
              "old_gate_sha256": after["family_a_original_gate"]["sha256"],
              "old_stop_sha256": after["family_a_stop_state"]["sha256"]}
    write(out / f"verify_{scope}.json", report)
    return report
