from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ZERO_TOLERANCE = 1e-10


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def value_hash(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def state_payload(state):
    """Canonical task facts only; excludes source policy, file identity and episode id."""
    return {
        "task": state["task"],
        "state_index": state["state_index"],
        "available_at_ns": state["available_at_ns"],
        "physical_time_ns": state["physical_time_ns"],
        "capture_order": state["capture_order"],
        "success": state["success"],
        "terminal_failure": state["terminal_failure"],
        "gripper_closed": state["gripper_closed"],
        "eef": state["eef"],
        "objects": state["objects"],
        "events": state.get("events", []),
    }


def state_hash(state, potential_history=None):
    payload = state_payload(state)
    if potential_history is not None:
        payload["potential_history"] = potential_history
    return value_hash(payload)


def potential_history(bank):
    """Causal reward memory that is not fully represented by the one-step event list."""
    return {
        "seen_losses": sorted([list(x) for x in bank.seen]),
        "open_losses": sorted([list(x) for x in bank.open]),
        "anchors": {k: [float(v[0]), list(v[1])] for k, v in sorted(bank.anchors.items())},
    }


def active_object(state):
    active = [oid for oid, obj in state["objects"].items() if obj["held"]]
    return active[0] if len(active) == 1 else ""


def geom_signature(valid_count, matched_open_losses, geometry_component):
    return canonical([int(valid_count), int(matched_open_losses), format(float(geometry_component), ".17g")])


def guidance_category(geom, graph, tolerance=ZERO_TOLERANCE):
    gz = abs(float(geom)) <= tolerance
    rz = abs(float(graph)) <= tolerance
    if gz and rz:
        return "BOTH_ZERO"
    if gz:
        return "GRAPH_ONLY_NONZERO"
    if rz:
        return "GEOM_ONLY_NONZERO"
    return "SAME_SIGN" if (geom > 0) == (graph > 0) else "OPPOSITE_SIGN"


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for k in range(start, end):
            out[order[k]] = rank
        start = end
    return out


def spearman(values_a, values_b):
    if len(values_a) != len(values_b) or not values_a:
        return None
    a = average_ranks(values_a)
    b = average_ranks(values_b)
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return 1.0 if a == b else 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def correlation_from_sums(n, sx, sy, sxx, syy, sxy):
    if n < 2:
        return None
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if vx <= 0 or vy <= 0:
        return None
    return (n * sxy - sx * sy) / math.sqrt(vx * vy)


def quantiles(values, probs=(0, .01, .05, .25, .5, .75, .95, .99, 1)):
    values = sorted(float(x) for x in values)
    if not values:
        return {str(p): None for p in probs}
    out = {}
    for p in probs:
        pos = p * (len(values) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        frac = pos - lo
        out[str(p)] = values[lo] * (1 - frac) + values[hi] * frac
    return out


FAILURE_PRIORITY = (
    "NO_ACQUIRE_PROGRESS",
    "ONE_OBJECT_ONLY_TIMEOUT",
    "TRANSPORT_STALL",
    "INVALID_ACTION_LOOP",
    "OBJECT_SWITCH_LOOP",
    "PREMATURE_PLACE_OR_RELEASE",
    "RECOVERY_NOT_STARTED",
    "RECOVERY_STARTED_NO_REGRASP",
    "REGRASP_NO_TASK_RESUME",
    "DETERMINISTIC_ACTION_CYCLE_OTHER",
    "TIMEOUT_OTHER",
)


def repeated_cycle(actions):
    for period in range(2, 5):
        needed = period * 3
        for end in range(needed, len(actions) + 1):
            seq = actions[end - needed:end]
            if all(seq[i] == seq[i % period] for i in range(needed)):
                return True, end - needed + 1
    return False, None


def classify_features(features):
    labels = []
    if not features["ever_held_or_transport"]:
        labels.append("NO_ACQUIRE_PROGRESS")
    if features["final_valid_count"] == 1:
        labels.append("ONE_OBJECT_ONLY_TIMEOUT")
    if features["transport_stall"]:
        labels.append("TRANSPORT_STALL")
    if features["invalid_loop"] or features["last_16_invalid_fraction"] >= .5:
        labels.append("INVALID_ACTION_LOOP")
    if features["object_switch_loop"]:
        labels.append("OBJECT_SWITCH_LOOP")
    if features["premature_place_or_release"]:
        labels.append("PREMATURE_PLACE_OR_RELEASE")
    if features["loss_seen"] and not features["recovery_started"]:
        labels.append("RECOVERY_NOT_STARTED")
    if features["recovery_started"] and not features["regrasped"]:
        labels.append("RECOVERY_STARTED_NO_REGRASP")
    if features["regrasped"] and not features["resumed_after_regrasp"]:
        labels.append("REGRASP_NO_TASK_RESUME")
    cycle, _ = repeated_cycle(features["actions"])
    if cycle and not features["progress_during_cycle"]:
        labels.append("DETERMINISTIC_ACTION_CYCLE_OTHER")
    if not labels:
        labels.append("TIMEOUT_OTHER")
    labels = [name for name in FAILURE_PRIORITY if name in labels]
    return labels[0], labels[1:]


class RunningPair:
    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        x, y = float(x), float(y)
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.syy += y * y
        self.sxy += x * y

    def result(self):
        return correlation_from_sums(self.n, self.sx, self.sy, self.sxx, self.syy, self.sxy)


def counter_rows(rows, key):
    return Counter(row[key] for row in rows)
