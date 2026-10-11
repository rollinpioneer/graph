"""Read-only codec verification against existing phase C complete suffixes."""
import copy
import gzip
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import log_codec as codec


def verify(branches_root):
    cstart, wstart = time.process_time(), time.perf_counter()
    results = []
    for branch, last in [("A", 110), ("A_repeat", 110), ("B_repeat", 109)]:
        source = branches_root / branch / "steps.jsonl.gz"
        source_before = hashlib.sha256(source.read_bytes()).hexdigest()
        records = []
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            for position, line in enumerate(stream, 1):
                record = json.loads(line)
                if 108 <= position <= last:
                    records.append((position, record))
        if [position for position, _ in records] != list(range(108, last + 1)):
            raise AssertionError("Expected saved suffix missing: " + branch)
        restored = codec.load_checkpoint(codec.checkpoint(records[0][1]["after"], 0))
        initial = copy.deepcopy(restored)
        encoded_stream = [codec.encoded_line(codec.checkpoint(initial, 0))]
        for position, record in records[1:]:
            original_before = copy.deepcopy(restored)
            original_after = copy.deepcopy(record["after"])
            if codec.canonical_hash(restored) != record["before_sha256"]:
                raise AssertionError("Old log chain broken")
            delta = codec.append(restored, record["after"], record["decision"])
            rebuilt = codec.apply(restored, json.loads(codec.encoded_line(delta)))
            assert rebuilt == record["after"]
            assert delta["decision"] == record["decision"]
            assert restored == original_before and record["after"] == original_after
            result = {"branch": branch, "to_old_event_step": position,
                      "before_sha256": delta["before_sha256"], "after_sha256": delta["after_sha256"],
                      "full_after_bytes": len(codec.canonical_bytes(record["after"])),
                      "delta_bytes": len(codec.encoded_line(delta)),
                      "delta_gzip_bytes": len(gzip.compress(codec.encoded_line(delta), compresslevel=1, mtime=0)),
                      "node_rows_saved": len(delta["node"]["set"]),
                      "score_cache_values_saved": len(delta["score_cache"]["set"]),
                      "lossless": True}
            results.append(result)
            encoded_stream.append(codec.encoded_line(delta))
            tampered = copy.deepcopy(delta)
            tampered["after_sha256"] = "0" * 64
            try:
                codec.apply(restored, tampered)
            except ValueError:
                pass
            else:
                raise AssertionError("Corrupt chain was not rejected")
            restored = rebuilt
        encoded_stream.append(codec.encoded_line(codec.checkpoint(restored, last - 108)))
        packed = io.BytesIO(gzip.compress(b"".join(encoded_stream), compresslevel=1, mtime=0))
        replayed = list(codec.replay_records(initial, packed))
        assert len(replayed) == len(encoded_stream)
        assert replayed[-1][1] == restored
        assert [record["kind"] for record, _ in replayed] == ["checkpoint"] + ["delta"] * (last - 108) + ["checkpoint"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before

    base = {"heap": [["00000000", 3, "2", True], ["00000000", 4, "f", False]],
            "closed": ["f"], "node": [["2", 1, None, None], ["f", 2, "2", 0]],
            "score_cache": {"2": "00000000", "f": "00000000"}, "serial": 5,
            "pending_scores": [], "temporary_field": "remove me", "plan": None}
    changed = {"heap": list(reversed(base["heap"])), "closed": ["2"],
               "node": [["2", 0, None, None], ["10", 1, "2", 9]],
               "score_cache": {"2": "0000803f", "10": "00000000"}, "serial": 6,
               "pending_scores": ["10"], "plan": [9], "new_scalar": {"exact": "中文"}}
    assert codec.apply(base, codec.append(base, changed, {"state": "2"})) == changed
    return {"status": "PASS", "actual_old_log_transitions": results,
            "synthetic_deletion_update_heap_order_scalar_test": "PASS", "old_inputs_unchanged": True,
            "gzip_stream_checkpoint_and_delta_test": "PASS",
            "model_calls": 0, "search_calls": 0, "CPU_seconds": time.process_time() - cstart,
            "wall_seconds": time.perf_counter() - wstart}


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"C:\Users\jackx\Documents\ChatGPT\CP-DISR\reports\q2_phase_c_single_pop_v1\Q2_PHASE_C_SINGLE_POP_V1_20261011T005008Z\branches\ipc_p12_k002")
    print(json.dumps(verify(root), ensure_ascii=False, indent=2))
