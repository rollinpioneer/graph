"""Saved-evidence-only final p12 continuation verifier.

No model imports, evaluation, search, SSH, or changes to earlier evidence.
Usage: python analyze_continuation.py NEW_ROOT [--previous-root PREVIOUS_ROOT]
The lossless log codec must be adjacent to this file (log_codec.py or its draft).
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import struct
import time
from pathlib import Path


BRANCHES = ("A", "B", "B_repeat", "A_repeat")
EXCLUDED = ("expanded", "generated", "duplicates", "path_updates", "calls",
            "score_cache_hits", "score_cache_misses")
CONTAINERS = {"heap", "closed", "node", "score_cache"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for position, line in enumerate(stream, 1):
            yield position, json.loads(line)


def future_scene(scene):
    """Exactly the previous phase C rule; no new exclusions or serial shifts."""
    out = {k: v for k, v in scene.items() if k not in EXCLUDED}
    out["heap"] = sorted(scene["heap"], key=lambda e:
                         (struct.unpack("<f", bytes.fromhex(e[0]))[0], e[1], int(e[2], 16)))
    return out


class QuickSignature:
    """Necessary equality condition, NEVER a substitute for full content.

    Incremental XORs are order-independent summaries of every CLOSED element,
    node row, and cache entry. Equal future scenes necessarily have equal quick
    signatures; collisions can only add candidates, never suppress a match.
    Heap content is deliberately confirmed later using the frozen full rule.
    """
    def __init__(self):
        self.nodes = {}
        self.cache = {}
        self.closed = set()
        self.xors = {"node": 0, "cache": 0, "closed": 0}

    @staticmethod
    def hv(value):
        return int.from_bytes(hashlib.sha256(canonical(value)).digest(), "big")

    def update(self, scene, delta=None):
        if delta is None or delta.get("kind") != "delta":
            self.nodes = {row[0]: self.hv(row) for row in scene["node"]}
            self.cache = {k: self.hv([k, v]) for k, v in scene["score_cache"].items()}
            self.closed = set(scene["closed"])
            self.xors = {"node": 0, "cache": 0, "closed": 0}
            for value in self.nodes.values():
                self.xors["node"] ^= value
            for value in self.cache.values():
                self.xors["cache"] ^= value
            for value in self.closed:
                self.xors["closed"] ^= self.hv(value)
        else:
            for label, field, values in (("node", "node", self.nodes),
                                          ("cache", "score_cache", self.cache)):
                changes = delta[field]
                for key in changes["delete"]:
                    self.xors[label] ^= values.pop(key)
                for key, value in changes["set"].items():
                    if key in values:
                        self.xors[label] ^= values[key]
                    encoded = value if label == "node" else [key, value]
                    values[key] = self.hv(encoded)
                    self.xors[label] ^= values[key]
            for key in delta["closed"]["delete"]:
                self.closed.remove(key)
                self.xors["closed"] ^= self.hv(key)
            for key in delta["closed"]["add"]:
                self.closed.add(key)
                self.xors["closed"] ^= self.hv(key)
        necessary = {k: v for k, v in scene.items()
                     if k not in CONTAINERS and k not in EXCLUDED}
        necessary["container_sizes"] = {k: len(scene[k]) for k in CONTAINERS}
        necessary["contents_xor_sha256"] = {k: format(v, "064x") for k, v in self.xors.items()}
        return digest(necessary)


class Analysis:
    def __init__(self, root, previous=None):
        self.started_cpu = time.process_time()
        self.started_wall = time.perf_counter()
        self.root = Path(root)
        self.reg = read(self.root / "preregistration.json")
        options = [previous, self.reg.get("analysis_previous_root"),
                   self.reg.get("previous_local_root"), self.reg.get("previous_local"),
                   self.reg.get("previous_remote_root")]
        self.previous = next((Path(p) for p in options if p and Path(p).is_dir()), None)
        if self.previous is None:
            raise FileNotFoundError("No accessible frozen phase C evidence root")
        self.oldreg = read(self.previous / "preregistration.json")
        frozen = self.reg.get("coalescence", self.oldreg["coalescence"])
        assert tuple(frozen["excluded_cost_fields"]) == EXCLUDED
        assert tuple(self.oldreg["coalescence"]["excluded_cost_fields"]) == EXCLUDED
        self.codec = self.load_codec()
        self.run = read(self.root / "runner_receipt.json")
        self.rows = {b: read(self.root / "branches" / b / "receipt.json")
                     if (self.root / "branches" / b / "receipt.json").is_file()
                     else {"branch": b, "status": "NOT_RUN", "stop_reason": "MISSING_RECEIPT",
                           "actual_model_state_evaluations": 0, "completed_additional_expansions": 0}
                     for b in BRANCHES}
        self.info = {b: [] for b in BRANCHES}
        self.indices = {b: [] for b in ("A", "B")}
        self.old_suffix = {}
        self.errors = []
        self.forward_failure_records = []
        self.observed = {}
        self.current_values = {b: {} for b in BRANCHES}
        self.cpu_cap = self.reg.get("analysis_CPU_seconds_cap", 3600)
        self.auxiliary_incomplete = False

    def load_codec(self):
        choices = (self.root / "log_codec.py", self.root / "log_codec_draft.py",
                   Path(__file__).with_name("log_codec.py"),
                   Path(__file__).with_name("log_codec_draft.py"))
        path = next(p for p in choices if p.is_file())
        spec = importlib.util.spec_from_file_location("saved_scene_codec", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def guard(self):
        if time.process_time() - self.started_cpu >= self.cpu_cap:
            raise RuntimeError("ANALYSIS_CPU_LIMIT")

    def olddir(self, branch):
        return self.previous / "branches" / "ipc_p12_k002" / branch

    def observe(self, state, value, origin):
        old = self.observed.get(state)
        if old is not None and old != value:
            self.errors.append({"kind": "SAME_KEY_SCORE_BYTES_CHANGED", "state": state,
                                "old": old, "new": value, "origin": origin})
        else:
            self.observed[state] = value

    def ledger(self):
        """All calls, including failed/interrupted requests, remain charged."""
        starts, old_calls, old_failed = {}, 0, 0
        for _, rec in jsonl(self.previous / "branch_ledger.jsonl"):
            kind = rec.get("kind")
            if kind == "MODEL_STATE_START":
                starts[rec["sequence"]] = rec
                old_calls += 1
            elif kind == "MODEL_STATE_END":
                before = starts.pop(rec["sequence"])
                if rec.get("status") != "OK":
                    old_failed += 1
                elif before["event"] == "ipc_p12_k002":
                    path = self.olddir(before["branch"]) / "score_bytes.bin"
                    with path.open("rb") as stream:
                        stream.seek(before["byte_offset"])
                        raw = stream.read(4).hex()
                    assert raw == rec["value_bits_le"]
                    self.observe(before["state"], raw, "old:" + before["branch"])
        assert not starts
        oldrun = read(self.previous / "run_receipt.json")
        assert old_calls == oldrun["counts"]["actual_model_state_evaluations"]
        per_branch, current_calls = {}, 0
        for branch in BRANCHES:
            path = self.root / "branches" / branch / "model_ledger.jsonl"
            if not path.is_file():
                assert self.rows[branch].get("actual_model_state_evaluations", 0) == 0
                per_branch[branch] = {"calls": 0, "failed": 0, "ledger_missing": True}
                continue
            pending, calls, failures = {}, 0, 0
            binary = self.root / "branches" / branch / "score_bytes.bin"
            with binary.open("rb") as rawstream:
                for _, rec in jsonl(path):
                    self.guard()
                    kind = rec.get("kind")
                    if kind in ("MODEL_STATE_START", "START"):
                        seq = rec["sequence"]
                        assert seq not in pending
                        pending[seq] = rec
                        calls += 1
                        assert rec.get("branch", branch) == branch
                        assert rec.get("model", "D0@820") == "D0@820"
                        assert rec.get("protocol", self.reg.get("protocol")) == self.reg.get("protocol")
                        assert rec.get("cache_key", rec["state"]) == rec["state"]
                    elif kind in ("MODEL_STATE_END", "END"):
                        before = pending.pop(rec["sequence"])
                        if rec.get("status") == "OK":
                            rawstream.seek(before["byte_offset"])
                            raw = rawstream.read(4).hex()
                            assert raw == rec["value_bits_le"]
                            assert math.isfinite(struct.unpack("<f", bytes.fromhex(raw))[0])
                            self.observe(before["state"], raw, "new:" + branch)
                            prior = self.current_values[branch].get(before["state"])
                            if prior is not None and prior != raw:
                                self.errors.append({"kind": "WITHIN_BRANCH_SCORE_CHANGED",
                                                    "branch": branch, "state": before["state"]})
                            self.current_values[branch][before["state"]] = raw
                        else:
                            failures += 1
                            self.forward_failure_records.append({"branch": branch, **rec})
            if pending:
                # A genuinely killed process can have unpaired starts; they are failures,
                # not zero-cost or valid scores. Keep evidence and mark technical failure.
                failures += len(pending)
                self.errors.append({"kind": "UNPAIRED_MODEL_START", "branch": branch,
                                    "sequences": sorted(pending)})
            assert calls == self.rows[branch].get("actual_model_state_evaluations", calls)
            assert calls <= 40000
            assert binary.stat().st_size == calls * 4
            per_branch[branch] = {"calls": calls, "failed": failures,
                                  "score_bytes_size": binary.stat().st_size}
            current_calls += calls
        assert current_calls <= 160000
        expected = self.run.get("counts", {}).get("actual_model_state_evaluations")
        if expected is not None:
            assert current_calls == expected
        return {"current_calls": current_calls, "current_by_branch": per_branch,
                "old_phase_C_all_eight_calls": old_calls,
                "old_phase_C_failed_or_interrupted_calls": old_failed,
                "all_observed_same_key_bytes_consistent": not any(
                    e["kind"] == "SAME_KEY_SCORE_BYTES_CHANGED" for e in self.errors),
                "observational_registry_supplies_no_scores": True,
                "unique_scored_states": len(self.observed)}

    def points(self, branch):
        """Entire old 0..108 prefix followed by each losslessly rebuilt new scene."""
        initial = read(self.olddir(branch) / "initial.json")
        yield {"depth": 0, "new_depth": 0, "phase": "OLD", "position": 0,
               "decision": None, "after": initial, "delta": None}
        old108 = None
        for pos, rec in itertools.islice(jsonl(self.olddir(branch) / "steps.jsonl.gz"), 108):
            assert rec["decision"] is not None
            assert rec["after"]["expanded"] == 2 + pos
            old108 = rec["after"]
            yield {"depth": pos, "new_depth": 0, "phase": "OLD", "position": pos,
                   "decision": rec["decision"], "after": rec["after"], "delta": None}
        assert old108 is not None and old108["expanded"] == 110
        branchdir = self.root / "branches" / branch
        if not (branchdir / "initial.json").is_file():
            return
        start = read(branchdir / "initial.json")
        checkpoint = read(self.root / "checkpoints" / (branch + "_108.json"))
        if checkpoint.get("kind") == "checkpoint":
            checkpoint = self.codec.load_checkpoint(checkpoint)
        assert start == checkpoint == old108
        assert start["status"] == "RUNNING" and not start["pending_scores"]
        for state, value in start["score_cache"].items():
            self.observe(state, value, "restored:" + branch)
        newcount, delta_position, last = 0, 0, start
        delta_path = branchdir / "delta.jsonl.gz"
        if delta_path.is_file():
            for rec, scene in self.codec.replay_records(start, delta_path):
                self.guard()
                if rec["kind"] != "delta":
                    continue
                delta_position += 1
                if rec["decision"] is not None:
                    newcount += 1
                if newcount == 1 and delta_position == 1 and rec["decision"] is not None:
                    normal = min((struct.unpack("<f", bytes.fromhex(e[0]))[0], e[1],
                                  int(e[2], 16), e[0], e[2])
                                 for e in start["heap"] if e[3])
                    decision = rec["decision"]
                    assert (decision["serial"], decision["state"], decision["score_bits"]) == (
                        normal[1], normal[4], normal[3]), "FIRST_CONTINUATION_POP_NOT_ORDINARY_D0"
                assert scene["expanded"] == 110 + newcount
                for state, value in rec["score_cache"]["set"].items():
                    if state not in start["score_cache"]:
                        assert self.current_values[branch].get(state) == value, (
                            "CACHE_INSERT_HAS_NO_SUCCESSFUL_OWN_BRANCH_FORWARD", branch, state)
                    self.observe(state, value, "complete_cache:" + branch)
                last = scene
                yield {"depth": 108 + newcount, "new_depth": newcount, "phase": "NEW",
                       "position": delta_position, "decision": rec["decision"],
                       "after": scene, "delta": rec}
        finalpath = branchdir / "final_complete.json"
        if finalpath.is_file():
            assert read(finalpath) == last
        assert newcount == self.rows[branch].get("completed_additional_expansions", 0)
        assert newcount <= 4892
        assert not last["pending_scores"]

    @staticmethod
    def metrics(point):
        s = point["after"]
        return {"depth": point["depth"], "new_depth": point["new_depth"],
                "decision": point["decision"], "status": s["status"],
                "generated": s["generated"], "duplicates": s["duplicates"],
                "path_updates": s["path_updates"], "calls": s["calls"],
                "score_cache_misses": s["score_cache_misses"],
                "score_cache_hits": s["score_cache_hits"],
                "valid_OPEN": sum(bool(e[3]) for e in s["heap"]),
                "logical_OPEN_entries": len(s["heap"]), "CLOSED": len(s["closed"]),
                "scene_sha256": (point["delta"]["after_sha256"]
                                 if point["delta"] is not None else digest(s))}

    def inspect_point(self, branch, point, tracker):
        self.info[branch].append(self.metrics(point))
        if branch in self.indices:
            self.indices[branch].append({k: point[k] for k in ("depth", "phase", "position")}
                                         | {"quick": tracker.update(point["after"], point["delta"])})
        if point["phase"] == "NEW" and branch in ("A", "A_repeat", "B_repeat"):
            targets = {"A": (109, 110), "A_repeat": (109, 110), "B_repeat": (109,)}[branch]
            if point["depth"] in targets:
                old = self.old_suffix[(branch, point["depth"])]
                current = point["delta"]
                same = (current["before_sha256"] == old["before_sha256"]
                        and point["decision"] == old["decision"]
                        and point["after"] == old["after"])
                self.suffix_checks.append({"branch": branch, "event_after_step": point["depth"],
                                           "PASS_full_content": same})
                if not same:
                    self.errors.append({"kind": "OLD_SUFFIX_CHANGED", "branch": branch,
                                        "event_after_step": point["depth"]})

    def compare_pair(self, primary, repeat):
        trackers = {b: QuickSignature() for b in (primary, repeat)}
        counts = {b: 0 for b in (primary, repeat)}
        mismatch = None
        common_new, common_old, common_depth = 0, 0, 0
        for left, right in itertools.zip_longest(self.points(primary), self.points(repeat)):
            self.guard()
            for branch, point in ((primary, left), (repeat, right)):
                if point is not None:
                    self.inspect_point(branch, point, trackers[branch])
                    counts[branch] += 1
            if left is None or right is None:
                continue
            same = (left["depth"] == right["depth"]
                    and left["decision"] == right["decision"]
                    and left["after"] == right["after"])
            if not same and mismatch is None:
                mismatch = {"event_after_step": left["depth"], "left_phase": left["phase"],
                            "right_phase": right["phase"],
                            "left_sha256": digest(left["after"]), "right_sha256": digest(right["after"])}
            if same and mismatch is None:
                common_depth = left["depth"]
                if left["phase"] == "NEW" and left["decision"] is not None:
                    common_new += 1
                elif left["phase"] == "OLD" and left["decision"] is not None:
                    common_old += 1
        if mismatch is not None:
            self.errors.append({"kind": "REPEAT_FULL_CONTENT_MISMATCH", "pair": [primary, repeat], **mismatch})
        return {"pair": [primary, repeat], "old_prefix_steps_verified": common_old,
                "identical_common_new_complete_steps": common_new,
                "common_event_after_depth": common_depth, "first_mismatch": mismatch,
                "PASS_common_prefix": mismatch is None and common_old == 108,
                "new_lengths": {b: self.rows[b].get("completed_additional_expansions", 0)
                                for b in (primary, repeat)},
                "wall_only_length_difference_is_not_technical_failure": True,
                "full_content_fields": "Every normalized scene field plus decision; no exclusions"}

    def load_point(self, branch, descriptor):
        for point in self.points(branch):
            if point["phase"] == descriptor["phase"] and point["position"] == descriptor["position"]:
                return point["after"]
        raise ValueError("Missing coalescence content candidate")

    def coalescence(self):
        lookup, candidates, matches = {}, [], []
        for point in self.indices["A"]:
            lookup.setdefault(point["quick"], []).append(point)
        for b in self.indices["B"]:
            self.guard()
            for a in lookup.get(b["quick"], ()):
                if a["depth"] == b["depth"] == 0:
                    continue
                sa, sb = self.load_point("A", a), self.load_point("B", b)
                fa, fb = future_scene(sa), future_scene(sb)
                ha, hb = digest(fa), digest(fb)
                rec = {"A": a, "B": b, "future_sha256_A": ha, "future_sha256_B": hb,
                       "hashes_match": ha == hb, "full_content_match": fa == fb}
                candidates.append(rec)
                if ha == hb and fa == fb:
                    matches.append({**rec, "E_B_minus_E_A_at_coalescence": b["depth"] - a["depth"],
                                    "exact_full_scene_also_equal": sa == sb,
                                    "exact_sha256_A": digest(sa), "exact_sha256_B": digest(sb)})
        return {"status": "CONFIRMED" if matches else "NO_OBSERVED_COALESCENCE_IN_SAVED_WINDOW",
                "rule": self.oldreg["coalescence"], "quick_filter": "Necessary equality only: retained scalar fields, container lengths, incremental XOR SHA256 over CLOSED/node/cache; no semantic exclusions added",
                "scope": "All saved old 0..108 and every reconstructed new complete scene; cross-depth comparisons; preintervention (0,0) excluded",
                "A_saved_points": len(self.indices["A"]), "B_saved_points": len(self.indices["B"]),
                "hash_candidates": candidates, "content_confirmed_matches": matches,
                "equal_remaining_wall_or_budget_claim": False}

    def load_suffix(self):
        self.suffix_checks = []
        for branch, targets in (("A", {109, 110}), ("A_repeat", {109, 110}), ("B_repeat", {109})):
            for position, rec in jsonl(self.olddir(branch) / "steps.jsonl.gz"):
                if position in targets:
                    self.old_suffix[(branch, position)] = rec
            assert all((branch, depth) in self.old_suffix for depth in targets)

    def plans(self):
        result = {}
        for branch in BRANCHES:
            row = self.rows[branch]
            final_path = self.root / "branches" / branch / "final_complete.json"
            if not final_path.is_file() or row.get("status") != "SOLVED":
                result[branch] = {"status": "NO_COMPLETE_PLAN"}
                continue
            scene = read(final_path)
            paths = (self.root / "plans" / (branch + ".json"),
                     self.root / "plans" / ("ipc_p12_k002_" + branch + ".json"),
                     self.root / "branches" / branch / "plan.json")
            path = next((p for p in paths if p.is_file()), None)
            if path is None:
                result[branch] = {"status": "MISSING_ACTION_CONTRACT_ARTIFACT"}
                self.errors.append({"kind": "PLAN_ARTIFACT_MISSING", "branch": branch})
                continue
            plan = read(path)
            indices = plan.get("indices", plan.get("action_indices"))
            legal = plan.get("legal_and_goal", plan.get("plan_valid", False))
            valid = (row.get("plan_valid") is True and legal is True and indices == scene["plan"]
                     and len(indices) == row.get("total_plan_length"))
            result[branch] = {"status": "PASS_SAVED_NATIVE_ACTION_CONTRACT" if valid else "INVALID",
                              "plan_length": len(indices) if indices is not None else None,
                              "indices_sha256": digest(indices), "artifact_sha256": sha(path),
                              "file": str(path.relative_to(self.root)), "optimality": "UNKNOWN",
                              "validation": "Original task contract replay executed by runner; saved result and exact indices cross-checked here"}
            if not valid:
                self.errors.append({"kind": "PLAN_CONTRACT_FAILURE", "branch": branch})
        for a, b in (("A", "A_repeat"), ("B", "B_repeat")):
            if result[a].get("status", "").startswith("PASS") and result[b].get("status", "").startswith("PASS"):
                if result[a]["indices_sha256"] != result[b]["indices_sha256"]:
                    self.errors.append({"kind": "TERMINAL_PLAN_REPEAT_MISMATCH", "pair": [a, b]})
        return result

    def finish(self):
        ledger = self.ledger()
        self.load_suffix()
        repeats = [self.compare_pair("A", "A_repeat"), self.compare_pair("B", "B_repeat")]
        co = self.coalescence()
        plans = self.plans()
        expected_suffixes = {(b, d) for b, d in self.old_suffix}
        observed_suffixes = {(r["branch"], r["event_after_step"]) for r in self.suffix_checks}
        suffix_unreached = sorted(expected_suffixes - observed_suffixes)
        for branch in BRANCHES:
            if self.rows[branch].get("stop_reason") == "TECHNICALLY_INVALID":
                self.errors.append({"kind": "RUNNER_TECHNICALLY_INVALID", "branch": branch})
            if self.rows[branch].get("status") in ("NOT_RUN", "FAILED_BEFORE_SEARCH", "INFRASTRUCTURE_FAILURE"):
                self.errors.append({"kind": "FORMAL_BRANCH_NOT_EXECUTED", "branch": branch,
                                    "status": self.rows[branch].get("status")})
        startup_evidence = {}
        for branch in BRANCHES:
            startup = self.root / "branches" / branch / "startup_suffix_verify.json"
            startup_evidence[branch] = read(startup) if startup.is_file() else None
            if startup.is_file() and any(not row.get("pass", row.get("PASS_full_content", False))
                                        for row in startup_evidence[branch]):
                self.errors.append({"kind": "RUNNER_STARTUP_SUFFIX_FAILURE", "branch": branch})
        complete_new = min(self.rows[b].get("completed_additional_expansions", 0) for b in BRANCHES)
        common_depth = 108 + complete_new
        metrics = {}
        for branch in BRANCHES:
            matches = [r for r in self.info[branch] if r["depth"] == common_depth]
            metrics[branch] = matches[-1] if matches else None
        valid = not self.errors and all(r["PASS_common_prefix"] for r in repeats)
        all_solved = all(self.rows[b].get("status") == "SOLVED" and self.rows[b].get("plan_valid") is True
                         and plans[b]["status"].startswith("PASS") for b in BRANCHES)
        primaries_solved = all(self.rows[b].get("status") == "SOLVED" and self.rows[b].get("plan_valid") is True
                               and plans[b]["status"].startswith("PASS") for b in ("A", "B"))
        de = self.rows["B"].get("completed_additional_expansions", 0) - self.rows["A"].get("completed_additional_expansions", 0)
        dl = (self.rows["B"]["total_plan_length"] - self.rows["A"]["total_plan_length"]) if primaries_solved else None
        if not valid:
            status = "TECHNICALLY_INVALID"
        elif all_solved:
            status = "VALID_COMPLETE_EFFECT" if de or dl else "VALID_COMPLETE_NO_OBSERVED_EFFECT"
        elif primaries_solved:
            status = "TERMINAL_REPEAT_INCOMPLETE"
        elif co["content_confirmed_matches"]:
            status = "VALID_COALESCENCE"
        elif any(self.rows[b].get("status") == "SOLVED" for b in ("A", "B")):
            status = "VALID_BOUNDED_COMPLETION_DIFFERENCE"
        else:
            status = "CENSORED_UNRESOLVED"
        decision_div = scene_div = new_decision_div = None
        aa = {r["depth"]: r for r in self.info["A"]}
        bb = {r["depth"]: r for r in self.info["B"]}
        for depth in range(2, common_depth + 1):
            if depth in aa and depth in bb:
                if decision_div is None and aa[depth]["decision"] != bb[depth]["decision"]:
                    decision_div = depth
                if scene_div is None and aa[depth]["scene_sha256"] != bb[depth]["scene_sha256"]:
                    scene_div = depth
                if depth > 108 and new_decision_div is None and aa[depth]["decision"] != bb[depth]["decision"]:
                    new_decision_div = depth
        event = {"event": "ipc_p12_k002", "status": status, "technical_valid": valid,
                 "all_four_solved_legal_and_repeated": all_solved and valid,
                 "delta_E_B_minus_A": de if all_solved and valid else None,
                 "delta_L_B_minus_A": dl if all_solved and valid else None,
                 "descriptive_primary_terminal_delta_E": de if primaries_solved and valid else None,
                 "descriptive_primary_terminal_delta_L": dl if primaries_solved and valid else None,
                 "terminal_repeat_incomplete": primaries_solved and not all_solved,
                 "common_new_complete_window_all_four": complete_new,
                 "common_event_after_complete_window_all_four": common_depth,
                 "fixed_window_metrics": metrics, "first_subsequent_decision_divergence": decision_div,
                 "first_subsequent_scene_divergence": scene_div, "coalescence": co["status"],
                 "first_new_continuation_decision_divergence_event_after_step": new_decision_div,
                 "longer_suffix_repeat_coverage": {
                     b: {"uncovered_new_complete_steps": max(0, self.rows[b].get("completed_additional_expansions", 0)
                               - repeats[0 if b.startswith("A") else 1]["identical_common_new_complete_steps"])}
                     for b in BRANCHES},
                 "full_efficiency": "OBSERVED_FOR_THIS_FIXED_EVENT" if all_solved and valid else "UNKNOWN",
                 "H3": "UNRESOLVED", "optimality": "UNKNOWN",
                 "Struct004_frozen_result": {"delta_E": 0, "delta_L": 0, "plan_length": 14, "rerun": False},
                 "post_phase_C_terminal_supplement": True, "new_independent_task": False,
                 "CPU_results_generalize_to_old_GPU_deployment": False,
                 "final_p12_computation_closeout": True, "further_automatic_runs": 0}
        for branch in BRANCHES:
            row = self.rows[branch]
            row["branch"] = branch
            row["old_logical_prefix_complete_expansions"] = 108
            row["event_after_logical_complete_expansions"] = 108 + row.get("completed_additional_expansions", 0)
            row["task_cumulative_complete_expansions"] = 2 + row["event_after_logical_complete_expansions"]
            row["terminal_repeat_confirmed"] = all_solved and valid
            row["true_total_expansions_if_unsolved"] = "UNKNOWN" if row.get("status") != "SOLVED" else row["event_after_logical_complete_expansions"]
        oldreceipt = read(self.previous / "final_receipt.json")
        oldrows = read(self.previous / "branch_results.json")
        current_cpu = self.run.get("process_cpu_seconds", self.run.get("worker_cpu_seconds", self.run.get("CPU_worker_sum_seconds")))
        resource = {"search_path_accounting": {b: {"old_prefix_expansions": 108,
                         "new_complete_expansions": self.rows[b]["completed_additional_expansions"],
                         "event_after_E": self.rows[b]["event_after_logical_complete_expansions"],
                         "task_cumulative_expansions": self.rows[b]["task_cumulative_complete_expansions"]} for b in BRANCHES},
                    "research_resource_accounting": {"old_phase_C_all_eight": oldreceipt,
                         "old_actual_branches_including_rolled_back_tail": oldrows,
                         "current_runner": self.run, "current_actual_calls": ledger["current_calls"],
                         "old_plus_current_actual_calls": ledger["old_phase_C_all_eight_calls"] + ledger["current_calls"],
                         "current_runner_cpu_seconds_as_reported": current_cpu,
                         "analysis_process_cpu_seconds_so_far": time.process_time() - self.started_cpu,
                         "rollback_does_not_erase_old_compute": True},
                    "timing_claim": "Segmented restored searches; prefix/load/restore/continuation separately recorded, not uninterrupted end-to-end performance",
                    "GPU": 0, "training": 0, "fitting": 0, "certification": 0, "full_graphs": 0}
        repeat_report = {"pairs": repeats, "old_formal_suffix_checks": self.suffix_checks,
                         "old_suffix_not_reached": suffix_unreached,
                         "runner_startup_suffix_evidence": startup_evidence,
                         "technical_errors": self.errors,
                         "terminal_repeat_status": "CONFIRMED" if all_solved and valid else "TERMINAL_REPEAT_INCOMPLETE" if primaries_solved else "NO_COMPLETE_TERMINAL_COMPARISON"}
        save(self.root, "repeatability_checks.json", repeat_report)
        save(self.root, "coalescence_checks.json", co)
        save(self.root, "event_results.json", event)
        save(self.root, "score_ledger_verification.json", ledger)
        save(self.root, "plan_validation_checks.json", plans)
        save(self.root, "resource_accounting.json", resource)
        fields = sorted({k for row in self.rows.values() for k in row})
        with (self.root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for branch in BRANCHES:
                writer.writerow(self.rows[branch])
        self.write_reports(event, repeat_report, co, ledger)
        receipt = {"role": "SAVED_EVIDENCE_ONLY", "model_forwards": 0, "search_runs": 0,
                   "cpu_seconds": time.process_time() - self.started_cpu,
                   "wall_seconds": time.perf_counter() - self.started_wall,
                   "technical_errors": self.errors, "full_log_chains_verified": True,
                   "event_status": status, "original_files_written": 0,
                   "common_window_only_for_censored_AB_comparison": True}
        save(self.root, "analysis_receipt.json", receipt)
        return event

    def write_reports(self, event, repeat, co, ledger):
        text = ["# 重复性及恢复核查", "", "先比较同处理完整现场共同前缀，再比较 A/B。时间长度差单列；完整现场比较没有忽略字段。", "", "```json", json.dumps(repeat, ensure_ascii=False, indent=2), "```", ""]
        (self.root / "repeatability_report.md").write_text("\n".join(text), encoding="utf-8")
        text = ["# 完整汇合核查", "", "沿用阶段 C 冻结规则：只排除7项非决策统计，保留父链接、g/action、缓存、有效和失效条目、插入序号与 serial；仅规范化堆物理排列。摘要候选之后核对实际完整内容。", "", "旧0—108步和本次所有可重建完整现场均纳入跨深度比较。未观察到仅指保存窗口，不代表永不汇合，不推出剩余墙钟相同。", "", "```json", json.dumps(co, ensure_ascii=False, indent=2), "```", ""]
        (self.root / "coalescence_report.md").write_text("\n".join(text), encoding="utf-8")
        text = ["# p12 最后一次受控续跑与收口", "", "本卡是看到阶段 C 结果后批准的终局补充，四支均从各自第108次完整展开后恢复，首步为普通 D0 搜索，没有再次干预。", "", "裁决：`" + event["status"] + "`。", "", "| 分支 | 本次完整展开 | 事件后累计展开 | 终止类型 | 合法计划长度 |", "|---|---:|---:|---|---:|"]
        for branch in BRANCHES:
            row = self.rows[branch]
            text.append("| {} | {} | {} | {} | {} |".format(branch, row.get("completed_additional_expansions"), row.get("event_after_logical_complete_expansions"), row.get("stop_reason"), row.get("total_plan_length")))
        text += ["", "四支共同窗口：事件后累计{}次完整展开，其中本次新增{}次。".format(event["common_event_after_complete_window_all_four"], event["common_new_complete_window_all_four"]), "", "确认的完整 ΔE（B−A）：{}；ΔL：{}。完整效率：{}。".format(event["delta_E_B_minus_A"] if event["delta_E_B_minus_A"] is not None else "UNKNOWN", event["delta_L_B_minus_A"] if event["delta_L_B_minus_A"] is not None else "UNKNOWN", event["full_efficiency"]), "", "汇合：`" + co["status"] + "`。", "", "Struct004 原结果保留：ΔE=0、ΔL=0、合法计划长度14，轨迹及动作序列不同；本卡没有重跑。", "", "A/B 的描述性终局差异只有在四支合法完成且同处理重复通过后才能升级为确认效应。截断值不替代真实总成本；正常 OPEN 耗尽单列，不能据此声称任务无解。计划合法性来自原任务动作合同回放，最优性 UNKNOWN。", "", "资源分别记搜索路径与研究实际消耗：旧阶段 C 的全部8支、回退尾部、重复、失败和中断均保留。分段恢复时间不能写成连续端到端性能。", "", "本次真实求值{}次；旧阶段 C 实际求值{}次，两轮合计{}次，旧消耗未因回退删除。".format(ledger["current_calls"], ledger["old_phase_C_all_eight_calls"], ledger["current_calls"] + ledger["old_phase_C_all_eight_calls"]), "", "H3仍未决。不推断T1整体优劣、监督因果、新算法必要性或CPU结果对旧GPU部署的适用性。本卡不是独立泛化任务。", "", "p12个案计算到此收口：不追加第五支、训练、认证、扩大预算、R4或自动续跑。", ""]
        (self.root / "research_decision.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--previous-root")
    args = parser.parse_args()
    analyzer = Analysis(args.root, args.previous_root)
    try:
        result = analyzer.finish()
    except Exception as error:
        save(analyzer.root, "analysis_failure.json", {"error": repr(error),
              "cpu_seconds": time.process_time() - analyzer.started_cpu,
              "no_model_or_search_calls": True, "conclusion": "TECHNICAL_OR_ANALYSIS_INCOMPLETE_DO_NOT_INFER_EFFECT"})
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
