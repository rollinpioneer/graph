"""Finite observable vocabulary and strict candidate-graph JSON Schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import write_json


PREDICATES: list[dict[str, Any]] = [
    {"name": "contact_present", "causal_observables": ["contact_sensor"], "history_window": 1, "requires_future": False, "description": "contact sensor is active"},
    {"name": "contact_absent", "causal_observables": ["contact_sensor"], "history_window": 1, "requires_future": False, "description": "contact sensor is inactive"},
    {"name": "contact_recently_lost", "causal_observables": ["contact_sensor"], "history_window": 3, "requires_future": False, "description": "contact changes from present to absent"},
    {"name": "contact_reestablished", "causal_observables": ["contact_sensor"], "history_window": 3, "requires_future": False, "description": "contact changes from absent to present"},
    {"name": "agent_approaching_object", "causal_observables": ["agent_to_object_position_x", "agent_to_object_position_y"], "history_window": 3, "requires_future": False, "description": "agent-object distance decreases over stored history"},
    {"name": "agent_receding_from_object", "causal_observables": ["agent_to_object_position_x", "agent_to_object_position_y"], "history_window": 3, "requires_future": False, "description": "agent-object distance increases over stored history"},
    {"name": "object_moving", "causal_observables": ["object_velocity_x", "object_velocity_y"], "history_window": 1, "requires_future": False, "description": "object velocity magnitude is non-negligible"},
    {"name": "object_stationary", "causal_observables": ["object_velocity_x", "object_velocity_y"], "history_window": 1, "requires_future": False, "description": "object velocity magnitude is small"},
    {"name": "object_moving_with_agent", "causal_observables": ["contact_sensor", "agent_velocity_x", "agent_velocity_y", "object_velocity_x", "object_velocity_y"], "history_window": 2, "requires_future": False, "description": "contact is present while agent and object move"},
    {"name": "object_approaching_goal", "causal_observables": ["object_to_goal_position_x", "object_to_goal_position_y"], "history_window": 3, "requires_future": False, "description": "object-goal distance decreases over stored history"},
    {"name": "object_receding_from_goal", "causal_observables": ["object_to_goal_position_x", "object_to_goal_position_y"], "history_window": 3, "requires_future": False, "description": "object-goal distance increases over stored history"},
    {"name": "collision_detected", "causal_observables": ["collision_sensor"], "history_window": 1, "requires_future": False, "description": "collision sensor is active"},
    {"name": "object_inside_goal", "causal_observables": ["object_in_goal_sensor"], "history_window": 1, "requires_future": False, "description": "object-in-goal sensor is active"},
    {"name": "stable_goal_occupancy", "causal_observables": ["object_in_goal_sensor"], "history_window": 3, "requires_future": False, "description": "object remains in goal across stored history"},
    {"name": "stagnation_detected", "causal_observables": ["object_to_goal_position_x", "object_to_goal_position_y", "object_velocity_x", "object_velocity_y"], "history_window": 5, "requires_future": False, "description": "stored history has negligible progress"},
    {"name": "progress_unknown", "causal_observables": [], "history_window": 1, "requires_future": False, "description": "available observables do not establish progress state"},
]


def predicate_vocabulary() -> dict[str, Any]:
    return {"schema": "u3_predicate_vocabulary_v1", "allowed_predicates": PREDICATES, "forbidden_predicates": ["future_success", "true_task_stage", "gold_mode", "hidden_controller_state", "scenario_id", "candidate_feasible", "exact_remaining_cost", "optimal_cost"]}


def strict_schema(vocabulary: dict[str, Any]) -> dict[str, Any]:
    names = [row["name"] for row in vocabulary["allowed_predicates"]]
    predicate_array = {"type": "array", "items": {"type": "string", "enum": names}, "uniqueItems": True}
    node = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "description", "role", "observable_predicates", "unknown_conditions", "source_cluster_ids", "evidence_segment_ids", "status"],
        "properties": {
            "id": {"type": "string", "minLength": 1}, "description": {"type": "string", "minLength": 1},
            "role": {"type": "string", "enum": ["start", "intermediate", "success_terminal", "failure_terminal", "unknown"]},
            "observable_predicates": predicate_array, "unknown_conditions": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "source_cluster_ids": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
            "evidence_segment_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}, "status": {"const": "hypothesized"},
        },
    }
    edge = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "src", "dst", "preconditions", "effects", "hypothesized_type", "source_transition_pairs", "evidence_segment_ids", "unknown_conditions", "cost_measurement_needed", "status"],
        "properties": {
            "id": {"type": "string", "minLength": 1}, "src": {"type": "string", "minLength": 1}, "dst": {"type": "string", "minLength": 1},
            "preconditions": predicate_array, "effects": predicate_array, "hypothesized_type": {"type": "string", "enum": ["forward", "failure", "recovery", "alternative", "unknown"]},
            "source_transition_pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["from_cluster_id", "to_cluster_id"], "properties": {"from_cluster_id": {"type": "integer"}, "to_cluster_id": {"type": "integer"}}}, "uniqueItems": True},
            "evidence_segment_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}, "unknown_conditions": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "cost_measurement_needed": {"type": "boolean"}, "status": {"const": "hypothesized"},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "u3_candidate_graph_v1", "type": "object", "additionalProperties": False,
        "required": ["schema_version", "graph_id", "scope", "nodes", "edges", "unresolved_questions"],
        "properties": {
            "schema_version": {"const": "u3_candidate_graph_v1"}, "graph_id": {"type": "string", "minLength": 1}, "scope": {"const": "stochastic_simulator_only"},
            "nodes": {"type": "array", "minItems": 3, "maxItems": 15, "items": node}, "edges": {"type": "array", "minItems": 2, "maxItems": 30, "items": edge},
            "unresolved_questions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["question", "affected_node_ids", "affected_edge_ids", "evidence_needed"], "properties": {"question": {"type": "string", "minLength": 1}, "affected_node_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}, "affected_edge_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}, "evidence_needed": {"type": "string", "minLength": 1}}}},
        },
    }


def write_vocabulary(output: Path) -> dict[str, Any]:
    value = predicate_vocabulary(); write_json(output, value); return value


def write_schema(vocabulary_path: Path, output: Path, example: Path) -> dict[str, Any]:
    import json
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8")); value = strict_schema(vocabulary); write_json(output, value)
    first = [row["name"] for row in vocabulary["allowed_predicates"]]
    sample = {"schema_version": "u3_candidate_graph_v1", "graph_id": "example_hypothesis", "scope": "stochastic_simulator_only", "nodes": [{"id": "start", "description": "observable initial condition", "role": "start", "observable_predicates": [first[1]], "unknown_conditions": ["route feasibility unknown"], "source_cluster_ids": [], "evidence_segment_ids": [], "status": "hypothesized"}, {"id": "transport", "description": "contacted transport hypothesis", "role": "intermediate", "observable_predicates": [first[0], first[8]], "unknown_conditions": [], "source_cluster_ids": [], "evidence_segment_ids": [], "status": "hypothesized"}, {"id": "success", "description": "observable goal occupancy", "role": "success_terminal", "observable_predicates": [first[12]], "unknown_conditions": [], "source_cluster_ids": [], "evidence_segment_ids": [], "status": "hypothesized"}], "edges": [{"id": "e1", "src": "start", "dst": "transport", "preconditions": [first[1]], "effects": [first[0]], "hypothesized_type": "forward", "source_transition_pairs": [], "evidence_segment_ids": [], "unknown_conditions": [], "cost_measurement_needed": True, "status": "hypothesized"}, {"id": "e2", "src": "transport", "dst": "success", "preconditions": [first[0]], "effects": [first[12]], "hypothesized_type": "forward", "source_transition_pairs": [], "evidence_segment_ids": [], "unknown_conditions": ["success persistence needs U4 validation"], "cost_measurement_needed": True, "status": "hypothesized"}], "unresolved_questions": [{"question": "Which observable loss configurations remain recoverable?", "affected_node_ids": ["transport"], "affected_edge_ids": [], "evidence_needed": "independent continuation validation"}]}
    write_json(example, sample); return value
