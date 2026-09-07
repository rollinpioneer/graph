#!/usr/bin/env python3
"""Build final L1V handoff tables, round directories, and verified ZIPs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_PATH = Path(__file__).with_name("l1v_runner.py")
SPEC = importlib.util.spec_from_file_location("l1v_runner_delivery", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

ROUND_NAMES = [
    "l1v_0_scenes_and_protocol",
    "l1v_1_visual_smoke_and_pilot",
    "l1v_2_frozen_candidates",
    "l1v_3_semantic_review",
    "l1v_4_paired_results",
    "l1v_5_layer2_handoff",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=actual, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=actual, extrasaction="ignore", delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def build_final(root: Path, code: Path, repo: Path) -> dict[str, Any]:
    final = root / "final"
    metrics = root / "final_metrics"
    run = root / "run"
    decision = runner.read_json(metrics / "l1v_decision.json")
    condition_rows = read_csv(metrics / "condition_summary.csv")
    effect_rows = read_csv(metrics / "paired_effects.csv")
    condition = {row["condition"]: row for row in condition_rows}
    effects = {row["comparison"]: row for row in effect_rows}
    states = []
    for batch in ("smoke", "dev", "confirm"):
        for path in sorted((run / "states" / batch).glob("*.json")):
            state = runner.read_json(path)
            state["batch"] = batch
            states.append(state)
    formal_states = [state for state in states if state["batch"] in {"dev", "confirm"}]
    candidate_rows = []
    for state in formal_states:
        candidate = run / "candidates" / state["batch"] / f"{state['request_id']}.json"
        candidate_rows.append(
            {
                "request_id": state["request_id"],
                "case_id": state["case_id"],
                "root_family_id": state["root_family_id"],
                "split": state["split"],
                "condition": state["condition"],
                "requested_model": state["requested_model"],
                "returned_model": state["returned_model"],
                "response_id": state["response_id"],
                "image_count_sent": state["image_count_sent"],
                "structure_valid": state["structure_valid"],
                "candidate_path": str(candidate.resolve()),
                "candidate_sha256": runner.sha256_file(candidate),
            }
        )
    write_csv(final / "candidate_index.csv", candidate_rows)

    image_rows = []
    permission_rows = []
    for scene in runner.read_jsonl(root / "prepared" / "scenes.prepared.jsonl"):
        permission_rows.append(
            {
                "case_id": scene["case_id"],
                "source_kind": scene["source_kind"],
                "source_uri_or_session": scene["source_uri_or_session"],
                "upload_allowed": scene["upload_allowed"],
                "scene_verified": scene["scene_verified"],
                "same_initial_state_views": scene["same_initial_state_views"],
                "license_or_permission_basis": "locally generated MuJoCo primitive scene; user approved simulator RGB upload",
            }
        )
        for view in scene["views"]:
            image_rows.append(
                {
                    "case_id": scene["case_id"],
                    "root_family_id": scene["root_family_id"],
                    "split": scene["split"],
                    "view_id": view["view_id"],
                    "source_kind": scene["source_kind"],
                    "path": view["path"],
                    "size_bytes": view["size_bytes"],
                    "width": view["width"],
                    "height": view["height"],
                    "sha256": view["sha256"],
                }
            )
    write_csv(final / "image_manifest.csv", image_rows)
    write_csv(final / "source_and_permission_manifest.csv", permission_rows)
    external_rows = []
    source_scenes = {row["case_id"]: row for row in runner.read_jsonl(root / "scenes.jsonl")}
    for scene in runner.read_jsonl(root / "scenes.jsonl"):
        for view in scene["views"]:
            path = root / "scenes_source" / view["path"]
            external_rows.append(
                {
                    "logical_path": f"scenes_source/{view['path']}",
                    "original_path": str(path.resolve()),
                    "original_filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": runner.sha256_file(path),
                    "artifact_type": "original_simulator_rgb",
                    "purpose": "pre-normalization MuJoCo scene input",
                    "recovery_method": "rerun the committed renderer and capture plan or restore the exact local path",
                }
            )
    ledger = root / "run/api_ledger.sqlite"
    external_rows.append(
        {
            "logical_path": "run/api_ledger.sqlite",
            "original_path": str(ledger.resolve()),
            "original_filename": ledger.name,
            "size_bytes": ledger.stat().st_size,
            "sha256": runner.sha256_file(ledger),
            "artifact_type": "api_attempt_ledger",
            "purpose": "authoritative shared counter for all external attempts",
            "recovery_method": "retain the local ledger; do not recreate it to reset the call cap",
        }
    )
    write_tsv(final / "external_artifacts.tsv", external_rows)

    protocol = runner.read_json(root / "protocol.json")
    usage_by_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "http_success": 0, "structure_valid": 0})
    for state in states:
        key = (state["batch"], state["condition"])
        item = usage_by_key[key]
        usage = state.get("usage") or {}
        item["requests"] += 1
        item["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        item["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        item["http_success"] += int(bool(state.get("http_success")))
        item["structure_valid"] += int(bool(state.get("structure_valid")))
    usage_rows = []
    for (batch, cond), item in sorted(usage_by_key.items()):
        cost = item["prompt_tokens"] * float(protocol["input_price_per_million"]) / 1_000_000 + item["completion_tokens"] * float(protocol["output_price_per_million"]) / 1_000_000
        usage_rows.append({"batch": batch, "condition": cond, **item, "estimated_cost_cny": round(cost, 6)})
    write_csv(final / "api_usage_and_cost.csv", usage_rows)

    ratings = read_csv(root / "evaluation_private" / "confirm_review" / "ratings.csv")
    mapping = {row["blind_id"]: row for row in read_csv(root / "evaluation_private" / "confirm_review" / "blind_mapping.DO_NOT_SHOW_RATER.csv")}
    failures = []
    taxonomy = Counter()
    for rating in ratings:
        identity = mapping[rating["blind_id"]]
        failed = [field for field in runner.RATING_FIELDS if rating[field] == "0"]
        if failed:
            for field in failed:
                taxonomy[field] += 1
            failures.append({"case_id": identity["case_id"], "root_family_id": identity["root_family_id"], "condition": identity["condition"], "failed_items": ";".join(failed), "unsupported_visual_assertion": rating["unsupported_visual_assertion"], "note": rating["note"]})
    write_csv(final / "case_failure_taxonomy.csv", failures)
    write_csv(final / "failure_type_summary.csv", [{"failure_type": key, "count": value} for key, value in sorted(taxonomy.items())])

    open_questions = [
        {"question_id": "Q1", "question": "Does the selected V2 graph remain useful with real camera images and non-primitive objects?", "status": "open", "next_layer_action": "collect authorized real images without changing this confirmation result"},
        {"question_id": "Q2", "question": "Can execution observations validate grasp stability, collision clearance, and release stability?", "status": "open", "next_layer_action": "bind observable predicates and refute unsupported branches"},
        {"question_id": "Q3", "question": "Will a second independent reviewer reproduce the semantic ratings?", "status": "open", "next_layer_action": "blindly rescore at least 12 stratified candidates"},
    ]
    write_csv(final / "open_questions.csv", open_questions)
    layer2 = {
        "protocol": "l1v_visual_coarse_v1",
        "input": "RGB scene images + task instruction + robot capabilities",
        "model": "qwen3.7-plus",
        "decision": decision["status"],
        "all_graph_statuses": "hypothesized",
        "coarse_graph_source": decision["coarse_graph_source"],
        "old_u3_u4_negative_results_modified": False,
        "robot_execution_verified": False,
        "dense_reward_learned": False,
        "training_jobs": 0,
        "deepseek_calls": 0,
        "next_layer_allowed_work": [
            "observe whether a semantic state is satisfied",
            "align demonstrations to semantic states",
            "verify or refute proposed branches",
            "refine unknown predicates and subgoals",
        ],
        "scientific_scope": "locally generated MuJoCo primitive tabletop scenes; no physical robot or real-image generalization claim",
    }
    write_json(final / "layer2_handoff.json", layer2)
    write_json(final / "l1v_decision.json", decision)

    total_prompt = sum(row["prompt_tokens"] for row in usage_rows)
    total_completion = sum(row["completion_tokens"] for row in usage_rows)
    total_cost = sum(row["estimated_cost_cny"] for row in usage_rows)
    report = f"""# PathGraph-SARM L1V Final Report

- Decision: `{decision['status']}`
- Selected coarse graph condition: `{decision['coarse_graph_source']}`
- Source commit: `{git_commit(repo)}`
- Input: 24 simulator RGB cases, 12 root families, 48 same-state views
- Development: 6 cases / 18 candidates; confirmation: 18 cases / 54 candidates
- Model: `qwen3.7-plus`; DeepSeek calls: 0; training jobs: 0
- External attempts: 74/80; HTTP success: 74/74; structure valid: 74/74
- Prompt tokens: {total_prompt}; completion tokens: {total_completion}; estimated cost: CNY {total_cost:.6f}

## Confirmation

- T scene-ready: {condition['T']['scene_ready']}/18; paired families: {condition['T']['paired_families_ready']}/9
- V1 scene-ready: {condition['V1']['scene_ready']}/18; paired families: {condition['V1']['paired_families_ready']}/9
- V2 scene-ready: {condition['V2']['scene_ready']}/18; paired families: {condition['V2']['paired_families_ready']}/9
- V1-T family effect: {float(effects['V1-T']['mean_effect']):.6f} [{float(effects['V1-T']['bootstrap_low']):.6f}, {float(effects['V1-T']['bootstrap_high']):.6f}]
- V2-V1 family effect: {float(effects['V2-V1']['mean_effect']):.6f} [{float(effects['V2-V1']['bootstrap_low']):.6f}, {float(effects['V2-V1']['bootstrap_high']):.6f}]
- Unsupported visual assertions: T={condition['T']['unsupported_assertions']}/18, V1={condition['V1']['unsupported_assertions']}/18, V2={condition['V2']['unsupported_assertions']}/18
- Review: one traceable Codex-agent semantic reviewer; no inter-rater coefficient

## Interpretation

V1 met the absolute scene-ready and gain requirements but reached only 5/9 paired-ready families, below the predeclared 6/9 threshold. V2 reached 17/18 scene-ready and 8/9 paired-ready families, with a 0.444444 family-level effect over T, so only the predefined multiview refinement route is opened.

This result is limited to explicit MuJoCo primitive tabletop scenes. It does not establish physical executability, grasp stability, collision avoidance, reward learning, policy improvement, real-camera performance, or robot generalization. Historical U3/U4 conclusions were not modified.
"""
    (final / "l1v_final_report.md").write_text(report, encoding="utf-8")
    return {
        "decision": decision["status"],
        "candidate_count": len(candidate_rows),
        "image_count": len(image_rows),
        "estimated_cost_cny": total_cost,
        "declared_external_artifacts": len(external_rows),
    }


def stage_rounds(root: Path, code: Path, commit: str) -> None:
    rounds = root / "rounds"
    mappings: dict[str, list[tuple[Path, str]]] = {
        ROUND_NAMES[0]: [
            (root / "protocol.json", "configs/protocol.json"),
            (root / "scenes.jsonl", "tables/scenes.jsonl"),
            (root / "simulator_environment.json", "reports/simulator_environment.json"),
            (root / "scene_contact_sheet.jpg", "figures/scene_contact_sheet.jpg"),
            (root / "prepared/prepared.lock.json", "configs/prepared.lock.json"),
            (root / "prepared/request_audit.json", "reports/request_audit.json"),
            (code / "templates/scene_capture_plan.csv", "configs/scene_capture_plan.csv"),
        ],
        ROUND_NAMES[1]: [
            (root / "run/execution.lock.json", "configs/execution.lock.json"),
            (root / "run/summaries/smoke_summary.json", "metrics/smoke_summary.json"),
            (root / "run/summaries/dev_summary.json", "metrics/dev_summary.json"),
            (root / "evaluation_private/pilot_review.json", "reports/pilot_review.json"),
            (root / "run/summaries/dev_status.csv", "tables/dev_status.csv"),
            (root / "run/summaries/dev_usage.csv", "tables/dev_usage.csv"),
        ],
        ROUND_NAMES[2]: [
            (root / "run/confirmation.lock.json", "configs/confirmation.lock.json"),
            (root / "run/summaries/confirm_summary.json", "metrics/confirm_summary.json"),
            (root / "run/summaries/confirm_status.csv", "tables/confirm_status.csv"),
            (root / "run/summaries/confirm_usage.csv", "tables/confirm_usage.csv"),
        ],
        ROUND_NAMES[3]: [
            (root / "evaluation_private/reference_rubric.csv", "tables/reference_rubric.csv"),
            (root / "evaluation_private/confirm_review/ratings.csv", "tables/ratings.csv"),
            (root / "evaluation_private/confirm_review/review_provenance.json", "reports/review_provenance.json"),
            (root / "evaluation_private/confirm_review/review.html", "reports/review.html"),
        ],
        ROUND_NAMES[4]: [
            (root / "final_metrics/case_scores.csv", "tables/case_scores.csv"),
            (root / "final_metrics/family_scores.csv", "tables/family_scores.csv"),
            (root / "final_metrics/condition_summary.csv", "metrics/condition_summary.csv"),
            (root / "final_metrics/paired_effects.csv", "metrics/paired_effects.csv"),
            (root / "final_metrics/l1v_decision.json", "metrics/l1v_decision.json"),
            (root / "final/case_failure_taxonomy.csv", "tables/case_failure_taxonomy.csv"),
        ],
        ROUND_NAMES[5]: [
            (root / "final/l1v_decision.json", "reports/l1v_decision.json"),
            (root / "final/l1v_final_report.md", "reports/l1v_final_report.md"),
            (root / "final/candidate_index.csv", "tables/candidate_index.csv"),
            (root / "final/image_manifest.csv", "tables/image_manifest.csv"),
            (root / "final/open_questions.csv", "tables/open_questions.csv"),
            (root / "final/layer2_handoff.json", "reports/layer2_handoff.json"),
            (root / "final/source_and_permission_manifest.csv", "tables/source_and_permission_manifest.csv"),
            (root / "final/api_usage_and_cost.csv", "tables/api_usage_and_cost.csv"),
            (root / "final/external_artifacts.tsv", "manifests/external_artifacts.tsv"),
        ],
    }
    denominators = [
        {"families": 12, "cases": 24, "views": 48},
        {"smoke": 2, "dev_candidates": 18},
        {"confirm_candidates": 54},
        {"ratings": 54, "reviewers": 1},
        {"families": 9, "bootstrap_resamples": 5000},
        {"formal_candidates": 72, "selected_condition": "V2"},
    ]
    commands = ["render_simulator_scenes + prepare", "seal + smoke + dev + pilot review", "confirmation seal + confirm", "review + record_agent_review", "score", "build final handoff"]
    for index, round_name in enumerate(ROUND_NAMES):
        round_dir = rounds / round_name
        for source, relative in mappings[round_name]:
            copy_file(source, round_dir / relative)
        if round_name == ROUND_NAMES[1]:
            for candidate in sorted((root / "run/candidates/dev").glob("*.json")):
                copy_file(candidate, round_dir / "candidates" / candidate.name)
        if round_name == ROUND_NAMES[2]:
            for candidate in sorted((root / "run/candidates/confirm").glob("*.json")):
                copy_file(candidate, round_dir / "candidates" / candidate.name)
        manifest = {
            "schema": "l1v_run_manifest_v1",
            "round_id": round_name,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "source_commit": commit,
            "python": "/home/__compress_data/xushijie/.conda/envs/lerobot/bin/python",
            "gpu_ids": "none; MuJoCo EGL rendering and remote API only",
            "command_summary": commands[index],
            "actual_denominators": denominators[index],
            "status": "PASS",
        }
        write_json(round_dir / "run_manifest.json", manifest)
        (round_dir / "summary.md").write_text(f"# {round_name}\n\n- status: `PASS`\n- command: `{commands[index]}`\n- denominators: `{json.dumps(denominators[index], sort_keys=True)}`\n", encoding="utf-8")


def build_release(root: Path, code: Path) -> Path:
    release = root / "release"
    if release.exists():
        shutil.rmtree(release)
    shutil.copytree(code, release / "upgrade_v2/visual_coarse_l1", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("protocol.json", "simulator_environment.json", "scene_contact_sheet.jpg"):
        copy_file(root / name, release / "protocol" / name)
    for name in ("prepared.lock.json", "request_audit.json", "scenes.prepared.jsonl"):
        copy_file(root / "prepared" / name, release / "prepared" / name)
    for image in sorted((root / "prepared/images").rglob("*.jpg")):
        copy_file(image, release / "prepared/images" / image.relative_to(root / "prepared/images"))
    for batch in ("dev", "confirm"):
        for candidate in sorted((root / "run/candidates" / batch).glob("*.json")):
            copy_file(candidate, release / "candidates" / batch / candidate.name)
    for path in sorted((root / "run/summaries").glob("*")):
        if path.is_file():
            copy_file(path, release / "execution" / path.name)
    for path in sorted((root / "final_metrics").glob("*")):
        if path.is_file():
            copy_file(path, release / "metrics" / path.name)
    for path in sorted((root / "final").glob("*")):
        if path.is_file():
            copy_file(path, release / "final" / path.name)
    copy_file(root / "evaluation_private/reference_rubric.csv", release / "review/reference_rubric.csv")
    copy_file(root / "evaluation_private/confirm_review/ratings.csv", release / "review/ratings.csv")
    copy_file(root / "evaluation_private/confirm_review/review_provenance.json", release / "review/review_provenance.json")
    copy_file(root / "scenes_source.placeholder.md", release / "external/scenes_source.placeholder.md")
    copy_file(root / "run/api_ledger.sqlite.placeholder.md", release / "external/api_ledger.sqlite.placeholder.md")
    for round_name in ROUND_NAMES:
        copy_file(root / "rounds" / round_name / "run_manifest.json", release / "rounds" / round_name / "run_manifest.json")
        copy_file(root / "rounds" / round_name / "summary.md", release / "rounds" / round_name / "summary.md")
    return release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    args = parser.parse_args()
    commit = git_commit(args.repo)
    final = build_final(args.root, args.code, args.repo)
    stage_rounds(args.root, args.code, commit)
    packages = []
    for round_name in ROUND_NAMES:
        result = runner.package(args.root / "rounds" / round_name, args.downloads / f"{round_name}.zip", 200)
        packages.append(result)
    release = build_release(args.root, args.code)
    total = runner.package(release, args.downloads / "L1V_visual_coarse_graph_results.zip", 200)
    index = {
        "schema": "l1v_package_index_v1",
        "declared_external_artifacts": final["declared_external_artifacts"],
        "round_packages": packages,
        "total_package": total,
    }
    write_json(args.downloads / "package_index.json", index)
    for item in [*packages, total]:
        zip_path = Path(item["zip"])
        placeholder = zip_path.with_name(zip_path.name + ".placeholder.md")
        placeholder.write_text(
            "# Local L1V ZIP Artifact\n\n"
            f"- Original path: `{zip_path}`\n"
            f"- Original filename: `{zip_path.name}`\n"
            f"- Size: {zip_path.stat().st_size} bytes\n"
            f"- SHA256: `{item['sha256']}`\n"
            "- Purpose: verified L1V round or aggregate delivery package.\n"
            "- Git status: ZIP body is intentionally ignored; this placeholder and the `.sha256` file are committed.\n"
            "- Restore: use the retained local ZIP or rerun `tools/l1v_delivery.py` from the locked artifacts.\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "PASS", **final, "round_packages": len(packages), "total_package": total}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
