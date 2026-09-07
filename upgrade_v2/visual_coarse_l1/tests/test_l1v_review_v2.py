from __future__ import annotations

import csv
import importlib.util
import json
import zipfile
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "l1v_review_v2.py"
SPEC = importlib.util.spec_from_file_location("l1v_review_v2_test", MODULE_PATH)
assert SPEC and SPEC.loader
review_v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_v2)


def graph() -> dict:
    return {
        "schema_version": "l1v_coarse_graph_v1",
        "goal_interpretation": "move object to target",
        "objects": [
            {"id": "obj", "description": "object", "role": "manipulated_object", "observability": "observed", "status": "hypothesized"},
            {"id": "target", "description": "target", "role": "target", "observability": "observed", "status": "hypothesized"},
        ],
        "nodes": [
            {"id": "start", "description": "start", "role": "start", "observable_conditions": ["visible"], "unknown_conditions": [], "status": "hypothesized"},
            {"id": "goal", "description": "goal", "role": "goal", "observable_conditions": ["placed"], "unknown_conditions": ["stable"], "status": "hypothesized"},
        ],
        "edges": [
            {"id": "move", "src": "start", "dst": "goal", "action": "place_object", "preconditions": ["visible"], "expected_effects": ["placed"], "trigger": None, "status": "hypothesized"}
        ],
        "physical_checks": ["verify stable"],
        "needs_clarification": False,
        "clarification_question": None,
    }


def evidence(blind_id: str = "E001") -> dict:
    return {
        "blind_id": blind_id,
        "review_status": "complete",
        "reviewer": "independent test reviewer",
        "scene_facts": ["object and target are visible"],
        "field_decisions": {
            field: {"value": 1, "reason": f"Explicit evidence for {field} is recorded here."}
            for field in review_v2.RATING_FIELDS
        },
        "unsupported_visual_assertion": {"value": 0, "reason": "No unsupported visual assertion is present."},
        "recovery_branch_conditional": "not_proposed",
        "graph_citations": {"objects": ["obj", "target"], "nodes": ["start", "goal"], "edges": ["move"]},
    }


def test_explicit_evidence_requires_all_values_reasons_and_citations() -> None:
    mapping = [{"blind_id": "E001"}]
    rows = review_v2.validate_evidence_rows([evidence()], mapping, {"E001": graph()})
    assert rows[0]["scene_ready"] == 1
    incomplete = evidence()
    incomplete["field_decisions"]["scene_ready"]["reason"] = "short"
    try:
        review_v2.validate_evidence_rows([incomplete], mapping, {"E001": graph()})
    except ValueError as exc:
        assert "substantive reason" in str(exc)
    else:
        raise AssertionError("missing explicit reason unexpectedly passed")


def test_explicit_evidence_cannot_embed_unblinded_identity() -> None:
    item = evidence()
    item["condition"] = "V1"
    try:
        review_v2.validate_evidence_rows([item], [{"blind_id": "E001"}], {"E001": graph()})
    except ValueError as exc:
        assert "remain blind" in str(exc)
    else:
        raise AssertionError("unblinded evidence unexpectedly passed")


def test_layer2_contract_separates_oracle_and_online_inputs(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    review_v2.write_json(metrics / "l1v_decision.json", {"status": "L1V_READY_FOR_REFINEMENT_SINGLE_VIEW", "coarse_graph_source": "V1"})
    review_v2.write_json(metrics / "review_resolution.json", {"status": "COMPLETE_EXPLICIT_INDEPENDENT_AGENT_RERATING"})
    output = tmp_path / "layer2"
    result = review_v2.build_layer2_entry(tmp_path, metrics, output)
    assert result["status"] == "LAYER2_INTERFACE_READY_FOR_DEVELOPMENT"
    contract = json.loads((output / "observation_contract.json").read_text(encoding="utf-8"))
    assert "oracle contact truth" in contract["online_forbidden_inputs"]
    with (output / "predicate_catalog.csv").open(encoding="utf-8", newline="") as handle:
        predicates = list(csv.DictReader(handle))
    oracle = [row for row in predicates if row["oracle_only"] == "1"]
    assert [row["predicate_id"] for row in oracle] == ["sim_contact_truth"]


def test_package_preserves_review_html_image_paths(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    review = root / "review_v2"
    metrics = root / "metrics_v2"
    layer2 = root / "layer2_entry_v1"
    repo = tmp_path / "repo"
    downloads = tmp_path / "downloads"
    image = root / "prepared/images/F01/view.jpg"
    candidate = root / "run/candidates/confirm/request_1.json"
    for path, content in (
        (root / "protocol.json", "{}\n"),
        (root / "run/confirmation.files.lock.json", "{}\n"),
        (root / "evaluation_private/reference_rubric.csv", "case_id\nF01_A\n"),
        (candidate, json.dumps(graph()) + "\n"),
        (image, "test image bytes"),
        (review / "review.html", '<img src="../prepared/images/F01/view.jpg">\n'),
        (metrics / "l1v_decision.json", '{"status":"PASS","coarse_graph_source":"V1"}\n'),
        (layer2 / "README.md", "layer 2\n"),
        (repo / "upgrade_v2/visual_coarse_l1/tools/l1v_review_v2.py", "# tool\n"),
        (repo / "upgrade_v2/visual_coarse_l1/tools/record_agent_review.py", "# compatibility tool\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    review_v2.write_csv(
        review / "blind_mapping.DO_NOT_SHOW_RATER.csv",
        [{"blind_id": "E001", "request_id": "request_1", "case_id": "F01_A", "input_view_count": "1"}],
    )
    (root / "prepared/scenes.prepared.jsonl").write_text(
        json.dumps({"case_id": "F01_A", "views": [{"path": str(image.resolve())}]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(review_v2, "git_commit", lambda unused_repo: "a" * 40)

    review_v2.package_update(root, review, metrics, layer2, repo, downloads)

    package = downloads / "L1V_semantic_review_update.zip"
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        assert "review_v2/review.html" in names
        assert "prepared/images/F01/view.jpg" in names
        assert "review_inputs/F01/view.jpg" not in names
