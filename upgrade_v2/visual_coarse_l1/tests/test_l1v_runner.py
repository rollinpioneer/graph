from __future__ import annotations

import csv
import importlib.util
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "l1v_runner.py"
SPEC = importlib.util.spec_from_file_location("l1v_runner", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_manifest(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    rows = []
    for family_index in range(12):
        split = "dev" if family_index < 3 else "confirm"
        family = f"F{family_index+1:02d}"
        for state in ("A", "B"):
            views = []
            for view_index in (1, 2):
                relative = Path(family) / f"{family}_{state}_view_{view_index}.jpg"
                target = images / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (40, 30), (family_index * 10, ord(state), view_index * 50))
                draw = ImageDraw.Draw(image)
                draw.rectangle((family_index, view_index, family_index + 8, (10 if state == "A" else 20)), fill=(255, 255, 255))
                image.save(target, quality=95)
                views.append({"view_id": f"view_{view_index}", "path": relative.as_posix()})
            rows.append(
                {
                    "case_id": f"{family}_{state}",
                    "root_family_id": family,
                    "split": split,
                    "state_label_local_only": state,
                    "task_instruction": f"task {family}",
                    "source_kind": "simulator_rgb",
                    "source_uri_or_session": "test",
                    "object_set_id": family,
                    "scene_verified": True,
                    "upload_allowed": True,
                    "same_initial_state_views": True,
                    "views": views,
                }
            )
    manifest = tmp_path / "scenes.jsonl"
    write_jsonl(manifest, rows)
    return manifest, images


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_repository_templates_are_complete_and_unapproved() -> None:
    scenes = runner.read_jsonl(code_root() / "templates" / "scenes.template.jsonl")
    references = runner.read_csv(code_root() / "templates" / "reference_rubric.csv")
    assert len(scenes) == 24
    assert len({row["root_family_id"] for row in scenes}) == 12
    assert all(len(row["views"]) == 2 for row in scenes)
    assert all(row["scene_verified"] is False for row in scenes)
    assert all(row["upload_allowed"] is False for row in scenes)
    assert all(row["same_initial_state_views"] is False for row in scenes)
    assert len(references) == 24
    assert all(row["ready_reference"] == "0" for row in references)


def test_prepare_enforces_split_and_image_contract(tmp_path: Path) -> None:
    manifest, images = make_manifest(tmp_path)
    output = tmp_path / "prepared"
    result = runner.prepare(manifest, images, output, code_root())
    assert result["status"] == "L1V_VISUAL_INPUTS_READY"
    assert result["requests"] == {"smoke": 2, "dev": 18, "confirm": 54}
    requests = runner.read_jsonl(output / "requests" / "confirm.jsonl")
    assert {condition: {row["image_count"] for row in requests if row["condition"] == condition} for condition in runner.CONDITIONS} == {"T": {0}, "V1": {1}, "V2": {2}}
    serialized = json.dumps(requests)
    assert "state_label_local_only" not in serialized
    assert "expected_behavior" not in serialized
    assert runner.verify_lock(output / "prepared.lock.json")["status"] == "PASS"


def valid_graph() -> dict:
    return {
        "schema_version": "l1v_coarse_graph_v1",
        "goal_interpretation": "move target to destination",
        "objects": [
            {"id": "obj", "description": "target", "role": "manipulated_object", "observability": "observed", "status": "hypothesized"}
        ],
        "nodes": [
            {"id": "start", "description": "start", "role": "start", "observable_conditions": ["visible"], "unknown_conditions": [], "status": "hypothesized"},
            {"id": "goal", "description": "goal", "role": "goal", "observable_conditions": ["placed"], "unknown_conditions": ["stable"], "status": "hypothesized"},
        ],
        "edges": [
            {"id": "move", "src": "start", "dst": "goal", "action": "move", "preconditions": ["visible"], "expected_effects": ["placed"], "trigger": None, "status": "hypothesized"}
        ],
        "physical_checks": ["verify stable"],
        "needs_clarification": False,
        "clarification_question": None,
    }


def test_graph_validation_rejects_dangling_and_verified_status() -> None:
    schema = runner.read_json(code_root() / "configs" / "coarse_graph.schema.json")
    graph = valid_graph()
    assert runner.graph_errors(graph, schema) == []
    graph["edges"][0]["dst"] = "missing"
    assert any("dangling" in error for error in runner.graph_errors(graph, schema))
    graph = valid_graph()
    graph["nodes"][0]["status"] = "verified"
    assert runner.graph_errors(graph, schema)


def test_confirmation_gate_requires_complete_reference(tmp_path: Path) -> None:
    manifest, images = make_manifest(tmp_path)
    prepared = tmp_path / "prepared"
    runner.prepare(manifest, images, prepared, code_root())
    protocol = runner.read_json(code_root() / "configs" / "protocol.template.json")
    protocol_path = tmp_path / "protocol.json"
    runner.write_json(protocol_path, protocol)
    run_root = tmp_path / "run"
    runner.seal(prepared, protocol_path, run_root, False, None, None)
    reference = tmp_path / "reference.csv"
    runner.write_csv(reference, [{"case_id": f"C{i}", "ready_reference": "1" if i else "0"} for i in range(24)])
    pilot = tmp_path / "pilot.json"
    runner.write_json(pilot, {"allow_confirmation": True, "actual_image_transport_checked": True, "reference_complete": True, "prompt_changed_after_pilot": False})
    try:
        runner.seal(prepared, protocol_path, run_root, True, reference, pilot)
    except ValueError as exc:
        assert "reference rubric is incomplete" in str(exc)
    else:
        raise AssertionError("incomplete reference unexpectedly passed")


def test_package_has_internal_hash_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.md").write_text("ok\n", encoding="utf-8")
    (source / "api_ledger.sqlite").write_bytes(b"external")
    output = tmp_path / "result.zip"
    result = runner.package(source, output, 1)
    assert result["status"] == "PASS"
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert "PACKAGE_SHA256SUMS.txt" in archive.namelist()
        assert "api_ledger.sqlite" not in archive.namelist()
        assert b"api_ledger.sqlite" in archive.read("external_artifacts.tsv")


def test_score_uses_family_pairs_and_invalid_output_is_zero(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    run_root = tmp_path / "run"
    ratings = []
    mapping = []
    index = 0
    for family_index in range(9):
        family = f"F{family_index+1:02d}"
        for state in ("A", "B"):
            for condition in runner.CONDITIONS:
                index += 1
                blind = f"R{index:03d}"
                request_id = f"confirm_{family}_{state}_{condition}"
                mapping.append({"blind_id": blind, "request_id": request_id, "case_id": f"{family}_{state}", "root_family_id": family, "condition": condition, "split": "confirm"})
                ready = "1" if condition != "T" else "0"
                ratings.append({"blind_id": blind, **{field: ready for field in runner.RATING_FIELDS}, "unsupported_visual_assertion": "0", "recovery_branch_conditional": "not_proposed", "review_status": "complete", "reviewer": "test", "note": ""})
                runner.write_json(run_root / "states" / "confirm" / f"{request_id}.json", {"structure_valid": not (family_index == 0 and state == "A" and condition == "V1")})
    runner.write_csv(review_root / "ratings.csv", ratings)
    runner.write_csv(review_root / "blind_mapping.DO_NOT_SHOW_RATER.csv", mapping)
    protocol_path = tmp_path / "protocol.json"
    runner.write_json(protocol_path, runner.read_json(code_root() / "configs" / "protocol.template.json"))
    result = runner.score(review_root, run_root, tmp_path / "metrics", protocol_path)
    assert result["status"] in {"L1V_READY_FOR_REFINEMENT_SINGLE_VIEW", "L1V_READY_FOR_REFINEMENT_MULTIVIEW"}
    summaries = {row["condition"]: row for row in result["condition_summary"]}
    assert summaries["V1"]["scene_ready"] == 17
    assert summaries["T"]["scene_ready"] == 0
