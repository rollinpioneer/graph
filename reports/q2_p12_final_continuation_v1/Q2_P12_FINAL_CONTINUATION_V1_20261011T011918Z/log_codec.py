"""Lossless normalized search-scene deltas; importing this module does no I/O.

Heap physical order and every score byte are preserved. ``node`` and ``closed``
must use the numeric state ordering of phase C ``normalized``. Dictionaries have
no semantic insertion order; JSON field ordering is covered by canonical hashes.
Every mutation can be replayed without model evaluation or search.
"""
import copy
import gzip
import hashlib
import json


SCHEMA = "CP_DISR_NORMALIZED_SCENE_DELTA_V1"
SPECIAL_FIELDS = frozenset(("heap", "closed", "node", "score_cache"))
CHECKPOINT_INTERVAL = 128


def canonical_bytes(obj):
    """Use exactly phase C obj_digest's canonical JSON encoding."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _states(values):
    return sorted(values, key=lambda state: int(state, 16))


def _validate_shape(obj):
    if not isinstance(obj, dict) or any(k not in obj for k in SPECIAL_FIELDS):
        raise ValueError("Scene is missing required normalized containers")
    closed = obj["closed"]
    if closed != _states(set(closed)):
        raise ValueError("CLOSED must be unique and sorted by numeric state")
    node = obj["node"]
    states = [row[0] for row in node]
    if states != _states(set(states)) or any(len(row) != 4 for row in node):
        raise ValueError("node must contain unique numeric-sorted four-field rows")
    if not isinstance(obj["score_cache"], dict):
        raise ValueError("score_cache must be the exact state-to-score-bit dictionary")
    if any(len(row) != 4 for row in obj["heap"]):
        raise ValueError("heap must preserve all four-field physical entries")


def _dict_delta(before, after):
    return {
        "set": {key: copy.deepcopy(value) for key, value in after.items()
                if key not in before or before[key] != value},
        "delete": sorted(set(before) - set(after)),
    }


def append(before, after, decision):
    """Return one detached, hash-chained lossless delta; perform no file writes.

    Caller writes the returned object as one compressed JSONL record. Both scenes
    remain unmodified. Explicit deletions are retained, even though this search's
    CLOSED and score cache normally grow monotonically.
    """
    _validate_shape(before)
    _validate_shape(after)
    previous_node = {row[0]: row for row in before["node"]}
    current_node = {row[0]: row for row in after["node"]}
    previous_closed = set(before["closed"])
    current_closed = set(after["closed"])
    previous_other = {key: value for key, value in before.items() if key not in SPECIAL_FIELDS}
    current_other = {key: value for key, value in after.items() if key not in SPECIAL_FIELDS}
    return {
        "schema": SCHEMA,
        "kind": "delta",
        "before_sha256": canonical_hash(before),
        "after_sha256": canonical_hash(after),
        "decision": copy.deepcopy(decision),
        "heap": copy.deepcopy(after["heap"]),
        "closed": {"add": _states(current_closed - previous_closed),
                   "delete": _states(previous_closed - current_closed)},
        "node": _dict_delta(previous_node, current_node),
        "score_cache": _dict_delta(before["score_cache"], after["score_cache"]),
        "fields": _dict_delta(previous_other, current_other),
    }


def _apply_dictionary(previous, changes):
    out = copy.deepcopy(previous)
    if set(changes) != {"set", "delete"}:
        raise ValueError("Unknown dictionary delta fields")
    if set(changes["set"]) & set(changes["delete"]):
        raise ValueError("A delta may not set and delete the same key")
    for key in changes["delete"]:
        if key not in out:
            raise ValueError("Delta deletes an absent key: " + key)
        del out[key]
    out.update(copy.deepcopy(changes["set"]))
    return out


def apply(previous, delta):
    """Rebuild one exact normalized scene and verify both chain hashes."""
    _validate_shape(previous)
    if delta.get("schema") != SCHEMA or delta.get("kind") != "delta":
        raise ValueError("Unsupported delta schema/kind")
    if canonical_hash(previous) != delta["before_sha256"]:
        raise ValueError("Delta previous-scene hash mismatch")
    previous_node = {row[0]: row for row in previous["node"]}
    node = _apply_dictionary(previous_node, delta["node"])
    closed = set(previous["closed"])
    if set(delta["closed"]["add"]) & set(delta["closed"]["delete"]):
        raise ValueError("CLOSED add/delete overlap")
    for state in delta["closed"]["delete"]:
        if state not in closed:
            raise ValueError("Delta deletes an absent CLOSED state")
        closed.remove(state)
    for state in delta["closed"]["add"]:
        if state in closed:
            raise ValueError("Delta adds an existing CLOSED state")
        closed.add(state)
    out = _apply_dictionary({key: value for key, value in previous.items()
                             if key not in SPECIAL_FIELDS}, delta["fields"])
    out["heap"] = copy.deepcopy(delta["heap"])
    out["closed"] = _states(closed)
    out["node"] = [node[state] for state in _states(node)]
    out["score_cache"] = _apply_dictionary(previous["score_cache"], delta["score_cache"])
    _validate_shape(out)
    if canonical_hash(out) != delta["after_sha256"]:
        raise ValueError("Delta reconstructed-scene hash mismatch")
    return out


def checkpoint(scene, completed_steps):
    """Create a detached full checkpoint for each 128-step boundary or stopping.

    The caller freezes when this is used: initial, each multiple of 128 complete
    new expansions, and terminal/last complete. ``completed_steps`` is descriptive
    and does not change or enter the restored search scene.
    """
    _validate_shape(scene)
    return {"schema": SCHEMA, "kind": "checkpoint", "completed_steps": completed_steps,
            "scene_sha256": canonical_hash(scene), "scene": copy.deepcopy(scene)}


def load_checkpoint(record):
    if record.get("schema") != SCHEMA or record.get("kind") != "checkpoint":
        raise ValueError("Unsupported checkpoint schema/kind")
    scene = copy.deepcopy(record["scene"])
    _validate_shape(scene)
    if canonical_hash(scene) != record["scene_sha256"]:
        raise ValueError("Checkpoint scene hash mismatch")
    return scene


def encoded_line(record):
    """Return compact UTF-8 JSONL bytes suitable for an explicit gzip writer."""
    return json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"


def replay_records(initial_scene, gzip_delta_path):
    """Stream ``(record, full_after)`` with complete hash-chain validation.

    Periodic checkpoints yield the identical current scene and do not represent
    another expansion. Consumers use ``record['kind']`` to distinguish them and
    must treat the yielded scene as read-only. Malformed/truncated gzip or JSON
    raises rather than silently discarding a supposedly committed record.
    """
    _validate_shape(initial_scene)
    current = copy.deepcopy(initial_scene)
    with gzip.open(gzip_delta_path, "rt", encoding="utf-8") as stream:
        for position, line in enumerate(stream, 1):
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "delta":
                current = apply(current, record)
            elif kind == "checkpoint":
                scene = load_checkpoint(record)
                if scene != current:
                    raise ValueError("Checkpoint is not the current scene at line " + str(position))
                current = scene
            else:
                raise ValueError("Unsupported log record kind at line " + str(position))
            yield record, current


def save_reserve_bytes(complete, partial=None, auxiliary_bytes=16 * 2**20):
    """Conservative UNCOMPRESSED reserve for final and diagnostic saves.

    Space guard should count actual files + pending encoded records + this
    reserve before committing a step. Multiplication covers pretty JSON saves,
    one periodic full checkpoint, the final scene, and a possible partial scene.
    Physical disk free space needs an independent guard. A crossing stops search;
    it must never silently discard already committed records.
    """
    complete_size = len(canonical_bytes(complete))
    partial_size = complete_size if partial is None else len(canonical_bytes(partial))
    return 4 * complete_size + 2 * partial_size + auxiliary_bytes
