"""Read saved Q2 evidence; never import the planner, torch, or execute pickle globals."""
import ast
import gzip
import hashlib
import io
import itertools
import json
import pickle
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
START = time.process_time()


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PrimitivePickle(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("snapshot may contain only builtin containers and scalars")


inventory = read("old_asset_inventory.json")
bad = []
for name, entry in inventory["files"].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != entry["sha256"]:
        bad.append(name)
assert not bad, bad
pre = read("old_round1/preregistration.json")
snap = PrimitivePickle(io.BytesIO((ROOT / "old_round1/q2/snapshots/ipc_p12/D0_820_k016.pkl").read_bytes())).load()
front = read("old_round1/q2/frontier/ipc_p12_k016.json")
heap = snap["heap"]
valid = [e for e in heap if e[2] not in snap["closed"]]
assert len(valid) == len(front["entries"])
entries = {(e["serial"], int(e["state"], 16)): e for e in front["entries"]}
assert all(entries[(ser, s)]["h_D0_820"] == h for h, ser, s in valid)
assert len({e[1] for e in heap}) == len(heap)
assert all(ser < snap["serial"] and s in snap["node"] for h, ser, s in heap)
assert snap["expanded"] == len(snap["closed"]) == 16
normal = dict(snap)
normal["heap"] = [{"score_float64_hex": struct.pack("<d", h).hex(), "score_float32_hex": struct.pack("<f", h).hex(), "serial": ser, "state_hex": format(s, "x"), "valid": s not in snap["closed"]} for h, ser, s in heap]
normal["closed"] = [format(s, "x") for s in sorted(snap["closed"])]
normal["node"] = [{"state_hex": format(s, "x"), "g": g, "parent_hex": None if p is None else format(p, "x"), "action_index": a} for s, (g, p, a) in sorted(snap["node"].items())]
write("old_p12_snapshot_normalized.json", normal)
pops = {}
for label in ["A", "B", "A_repeat", "B_repeat"]:
    with gzip.open(ROOT / ("old_round1/q2/fork/ipc_p12_k016_%s.pops.jsonl.gz" % label), "rt") as f:
        pops[label] = [json.loads(line) for line in f]
pairs = {}
for a, b in itertools.combinations(pops, 2):
    n = min(len(pops[a]), len(pops[b]))
    sa, sb = [r[1] for r in pops[a]][:n], [r[1] for r in pops[b]][:n]
    aa, bb = set(sa), set(sb)
    pairs[a + "|" + b] = {"prefix_length_before_dedup": n, "unique_A": len(aa), "unique_B": len(bb), "deduplicated_Jaccard_same_length": len(aa & bb) / len(aa | bb), "first_state_divergence_zero_based": next((i for i in range(n) if sa[i] != sb[i]), None), "state_sequence_identical": sa == sb, "full_record_sequence_identical": pops[a] == pops[b]}
maps = {}
for path in sorted((ROOT / "old_round2/score_maps").glob("*__M1.npy")):
    mate = path.with_name(path.name.replace("__M1", "__M2"))
    b1, b2 = np.load(path, allow_pickle=False), np.load(mate, allow_pickle=False)
    assert b1.dtype == b2.dtype == np.uint32 and b1.shape == b2.shape
    v1, v2 = b1.view(np.float32).astype(np.float64), b2.view(np.float32).astype(np.float64)
    s1, s2 = np.sign(v1[:, None] - v1[None, :]), np.sign(v2[:, None] - v2[None, :])
    maps[path.name[:-8]] = {"comparison": "same task, same model, independent M1 vs M2 passes; NOT D0 vs T1", "states": len(b1), "bitwise_equal_fraction": float((b1 == b2).mean()), "max_abs_diff": float(abs(v1 - v2).max()), "discordant_unordered_pairs": int(((s1 * s2) < 0).sum() // 2), "changed_tie_unordered_pairs": int(((s1 == 0) != (s2 == 0)).sum() // 2)}
base_files = list((ROOT / "old_round2/replay_pops").glob("*.gz"))
steps = [json.loads(s) for s in (ROOT / "old_round2/events_steps.jsonl").read_text().splitlines()]
source = (ROOT / "old_round1/scripts/q2fork.py").read_text()
ast.parse(source)
result = {
    "status": "G0_MATERIALS_AVAILABLE_WITH_DISCLOSED_HISTORICAL_LIMITATIONS",
    "protected_old_files_verified": len(inventory["files"]),
    "old_execution_cards": {d: {"path": str(ROOT / d / "experiment_card.md"), "sha256": hashlib.sha256((ROOT / d / "experiment_card.md").read_bytes()).hexdigest()} for d in ["old_round1", "old_round2"]},
    "actual_git_commit_provenance": {"frozen_core": pre["frozen_baseline"], "actual_server_head": read("identity_before.json")["head"], "old_per_card_new_commits": "NONE; both receipts explicitly say not committed/pushed; independent scripts are bound by archived hashes, not an invented result commit"},
    "p12_snapshot": {"keys": list(snap), "heap_entries": len(heap), "valid_entries": len(valid), "closed": len(snap["closed"]), "node_records": len(snap["node"]), "serial": snap["serial"], "candidate_table_matches_snapshot": True, "original_task_and_domain_binding": "task SHA in preregistration/asset_manifest; original pickle itself has no task identity", "random_state_saved": False, "delayed_scoring": "No pending successor batch at before-pop save point; all pending scores are synchronously enqueued", "score_cache_saved": False},
    "old_fork_pairs": pairs,
    "old_shared_objects_static_audit": {"search_containers": "fresh pickle load per branch; no shared heap/CLOSED/node", "evaluator": "ONE dev=evaluator(D0@820) object reused across A/B/A_repeat/B_repeat; prepare resets counters and task template", "branch_score_cache": "NeuralEval has no score cache; calls model.state_values each time", "remaining_unknown": "Shared model/template internals and numerical backend state were not serialized or isolated; log differences alone cannot assign a unique root cause"},
    "R2_score_map_recheck": maps,
    "R2_actual_base_replays": len(base_files),
    "R2_observation_steps": len(steps),
    "R2_disagreement_steps": sum(s["disagree"] for s in steps),
    "R2_Jaccard_definition": "set(full D0 popped sequence) vs set(full T1 popped sequence), no same-length truncation; both actual sequences have length 6",
    "corrections": ["R2 report claims 24 base replays; files/code/ledger give 2 tasks x 3 models x 2 maps = 12, plus 2 observations = 14", "R2 report says 首选一次都不同; all 12 saved observation rows have disagree=false", "R1 report's GPU-only causal explanation is unproven; no full final search state was saved for reunion checking"],
    "new_model_state_evaluations": 0, "new_searches": 0, "new_graph_builds": 0,
    "cpu_seconds": time.process_time() - START,
}
write("G0_audit.json", result)
print(json.dumps({"status": result["status"], "R1_pair_summary": pairs, "R2_actual_base_replays": len(base_files), "R2_disagreement_steps": result["R2_disagreement_steps"], "CPU_seconds": result["cpu_seconds"]}, ensure_ascii=False))
