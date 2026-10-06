"""CP-DISR-C1-BW-FROZEN-EVAL-A03: one-shot evaluation of the three FROZEN A02 final imitation checkpoints on A0 / A1 / A2 / B plus the exact-planner reference (evaluation only: no training, no optimizer).

The ID gate of A02 is not touched (IMITATION_ID_GATE_FAIL stays). A03 adds NO replacement performance gate: A0 is a diagnostic slice that never blocks A1 / A2 / B. Every (checkpoint, case) is
evaluated once; progress lives in an append-only ledger, a case that was started but not recorded is TECHNICAL_INCOMPLETE and is never silently re-run. Models, environment, planner and metrics are the
frozen implementations (``train.evaluate_case``); this module only loads, orders, records and reports.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
from pathlib import Path

CARD = "CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1"
AMENDMENT = "CP-DISR-C1-BW-FROZEN-EVAL-A03"
BASE_COMMIT = "b6804664e0f0ddb2f9b7068d9c49fdf5e6254a6a"
A02_PREP_COMMIT = "8ea6a71a54a1f25fd402b1e14e835728f93b5667"
FREEZE_COMMIT = "490a3b11d49f6155baf2a6b6c430edac6609d2af"
BRANCH = "codex/cp-disr-c1-blocksworld-main-v1"
BASE_REL = "runs/final_master/c1_route_b/blocksworld_main_v1"
OLD_ROOTS = (BASE_REL + "/20261006T015313Z_490a3b11", BASE_REL + "/imitation_a02/20261006T053652Z_8ea6a71a")
A02_ROOT = OLD_ROOTS[1]
OUT_REL = BASE_REL + "/evaluation_a03"
DOC_REL = "docs/c1_blocksworld/CP_DISR_C1_Blocksworld_Frozen_Policy_Evaluation_Amendment_A03.md"
MODELS = {
    "B2": {"run_id": "R-C1-BW-IL-B2-0", "method": "B2-CACHED", "path": A02_ROOT + "/runs/R-C1-BW-IL-B2-0/checkpoints/final.pt", "bytes": 19860355,
           "sha256": "29c7eab495b4dea7a3f30f40fe7321619bdbe941153b3a69c2a937a528d61e11"},
    "QMARK": {"run_id": "R-C1-BW-IL-QMARK-0", "method": "B1-K+QMARK-BW", "path": A02_ROOT + "/runs/R-C1-BW-IL-QMARK-0/checkpoints/final.pt", "bytes": 19862899,
              "sha256": "50fa46d9c1ea39d91839f3980ae0c985549e339cf91317adb1a3505008f3ed87"},
    "ASNET": {"run_id": "R-C1-BW-IL-ASNET-0", "method": "ASNET-READOUT", "path": A02_ROOT + "/runs/R-C1-BW-IL-ASNET-0/checkpoints/final.pt", "bytes": 19860355,
              "sha256": "9b02eb9c4f80204802273fa92df02ea21093fd54e2fbcbf672846fa31df847da"},
}
SPLITS = {
    "a0": {"path": "configs/splits/c1_bw_a0_iso_v1.json", "sha256": "c15346690583b700766e1adb73c46580d83439506972ddd09b7773637dd168f6", "n": 24},
    "a1": {"path": "configs/splits/c1_bw_a1_color_reverse_v1.json", "sha256": "4e31998725be0c81d6e7f9306beaf4c2b7a7629e7400482f4f18541b1a9c1d96", "n": 32},
    "a2": {"path": "configs/splits/c1_bw_a2_noniso_v1.json", "sha256": "1207d455afaa7b8c5b8e4d2f62d5b49dbf2c29e07a944ef85a543396b4a7a349", "n": 32},
    "b": {"path": "configs/splits/c1_bw_b_scale_v1.json", "sha256": "9dcf596cc72d9102182f71963edb90d4fd576bd3a9063be40309be0a0ea89220", "n": 48},
}
TRAIN_DEV = {"path": "configs/splits/c1_bw_train_dev_v1.json", "sha256": "64225ed50d69aa8c42d9ce36470bdb4705b792f464e97a88b23b2ba8cdc0b913"}
ORDER = ("a0", "a1", "a2", "b")
FROZEN_SOURCES = ("src/cp_disr/blocksworld/state.py", "src/cp_disr/blocksworld/contracts.py", "src/cp_disr/blocksworld/planner.py", "src/cp_disr/blocksworld/canonical.py",
                  "src/cp_disr/blocksworld/generator.py", "src/cp_disr/blocksworld/metrics.py", "src/cp_disr/blocksworld/environment.py", "src/cp_disr/blocksworld/splits.py",
                  "src/cp_disr/blocksworld/train.py", "src/cp_disr/blocksworld/classification.py", "src/cp_disr/blocksworld/launch.py", "src/cp_disr/blocksworld/imitation.py",
                  "src/cp_disr/blocksworld/imitation_launch.py", "src/cp_disr/c1_blocksworld_policies.py", "src/cp_disr/c1_qmark_policy.py", "src/cp_disr/neural.py",
                  "src/cp_disr/torch_rl.py", "src/cp_disr/rl.py", "scripts/c1_bw_eval.py", "scripts/c1_bw_il_eval.py")
NEW_SOURCES = ("src/cp_disr/blocksworld/eval_a03.py", "scripts/c1_bw_eval_a03.py", "tests/test_c1_bw_eval_a03.py", DOC_REL)
STATE_LABELS = {"original_ppo_state": "PPO_ID_GATE_FAIL", "original_a02_state": "IMITATION_ID_GATE_FAIL", "original_rep_label": "NOT_ISSUED"}


class A03Error(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ assets and identity
def asset_check(root):
    root = Path(root)
    out = {"models": {}, "splits": {}, "ok": True}
    for key, m in MODELS.items():
        p = root / m["path"]
        side = p.with_suffix(".json")
        row = {"exists": p.is_file(), "bytes_expected": m["bytes"], "sha256_expected": m["sha256"]}
        if row["exists"]:
            row.update({"bytes": p.stat().st_size, "sha256": sha256_file(p)})
            row["sidecar_sha256"] = json.loads(side.read_text())["sha256"] if side.is_file() else None
        row["ok"] = bool(row["exists"] and row["bytes"] == m["bytes"] and row["sha256"] == m["sha256"] and row["sidecar_sha256"] == m["sha256"])
        out["models"][key] = row
        out["ok"] &= row["ok"]
    for name, s in list(SPLITS.items()) + [("train_dev", dict(TRAIN_DEV, n=None))]:
        p = root / s["path"]
        row = {"sha256_expected": s["sha256"], "sha256": sha256_file(p) if p.is_file() else None}
        if name != "train_dev" and p.is_file():
            row["cases"] = len(json.loads(p.read_text())["cases"])
        row["ok"] = row["sha256"] == s["sha256"] and (name == "train_dev" or row["cases"] == s["n"])
        out["splits"][name] = row
        out["ok"] &= row["ok"]
    return out


def source_identity(root):
    root = Path(root)
    frozen = {}
    for rel in FROZEN_SOURCES:
        now = sha256_file(root / rel)
        base = hashlib.sha256(subprocess.check_output(["git", "-C", str(root), "show", "%s:%s" % (BASE_COMMIT, rel)])).hexdigest()
        frozen[rel] = {"sha256": now, "sha256_at_base_commit": base, "unchanged_since_base": now == base}
    new = {rel: {"sha256": sha256_file(root / rel)} for rel in NEW_SOURCES if (root / rel).is_file()}
    trees = {rel: git(root, "rev-parse", "%s:%s" % (BASE_COMMIT, rel)) for rel in OLD_ROOTS}
    return {"base_commit": BASE_COMMIT, "frozen_sources": frozen, "new_sources_registered_separately": new, "old_root_tree_ids_at_base": trees,
            "all_frozen_unchanged": all(v["unchanged_since_base"] for v in frozen.values())}


def check_identity(root, run_root):
    """Evaluation may start only on a clean tracked tree whose frozen sources, new sources and (read-only) old result trees equal the registered identity."""
    root, run_root = Path(root), Path(run_root)
    own = str(run_root.resolve().relative_to(Path(root).resolve()))                  # the A03 run root's own (registered) files change while the evaluation runs
    if git(root, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)%s" % own):
        raise A03Error("tracked tree is dirty")
    try:
        git(root, "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD")
    except subprocess.CalledProcessError:
        raise A03Error("the A03 base commit is not an ancestor of HEAD")
    reg = json.loads((run_root / "registration" / "source_identity.json").read_text())
    now = source_identity(root)
    for rel, v in reg["frozen_sources"].items():
        if now["frozen_sources"][rel]["sha256"] != v["sha256"]:
            raise A03Error("frozen source changed since registration: %s" % rel)
    for rel, v in reg["new_sources_registered_separately"].items():
        if sha256_file(root / rel) != v["sha256"]:
            raise A03Error("registered A03 source changed: %s" % rel)
    for rel in OLD_ROOTS:
        if git(root, "rev-parse", "HEAD:%s" % rel) != reg["old_root_tree_ids_at_base"][rel]:
            raise A03Error("an old result directory changed: %s" % rel)
    return git(root, "rev-parse", "HEAD")


def register(root, authorization_text, stamp=None):
    root = Path(root).resolve()
    if git(root, "rev-parse", "HEAD") != BASE_COMMIT or git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain", "--untracked-files=no"):
        raise A03Error("registration needs branch %s at the base commit with a clean tracked tree" % BRANCH)
    chk = asset_check(root)
    if not chk["ok"]:
        raise A03Error("asset check failed: %s" % json.dumps(chk)[:500])
    doc_sha = sha256_file(root / DOC_REL)
    run_root = root / OUT_REL / ("%s_%s" % (stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), doc_sha[:8]))
    if run_root.exists():
        raise A03Error("already registered: %s" % run_root)
    for sub in ("registration", "receipts", "eval", "results"):
        (run_root / sub).mkdir(parents=True)
    manifest = {}
    for name in ORDER:
        manifest[name] = [c["case_id"] for c in json.loads((root / SPLITS[name]["path"]).read_text())["cases"]]
    spec = {"card": CARD, "amendment": AMENDMENT, "base_commit": BASE_COMMIT, "branch": BRANCH, "a03_doc_sha256": doc_sha, "authorization_text": authorization_text,
            "authorization_text_sha256": hashlib.sha256(authorization_text.encode()).hexdigest(), **STATE_LABELS, "a02_prep_commit": A02_PREP_COMMIT, "freeze_commit": FREEZE_COMMIT,
            "models": MODELS, "splits": SPLITS, "train_dev": TRAIN_DEV, "new_training_runs": 0, "optimizer_steps": 0, "checkpoint_selection": "A02_FINAL_ONLY",
            "id_gate_relaxed_or_rewritten": False, "evaluation_release_rule_changed": True, "a0_performance_stop_removed_for_all": True, "learned_episodes_planned": 408,
            "planner_cases_planned": 136, "case_manifest": manifest, "test_exposure": {"label": "PREMATERIALIZED_SUITE_WITH_DISCLOSED_SMOKE_EXPOSURE", "registered_A02_checkpoint_test_exposure_before_A03": False,
                                                                                    "protocol_selection_used_ID_results": True,
                                                                                    "note": "an unregistered three-update smoke checkpoint was dry-run on A0/A1/A2/B files before registration (PPO stop-point summary); not a fully blind suite"},
            "receptive_field_status": "NOT_TESTABLE_STRUCTURALLY_EMPTY", "registered": utc_now()}
    write_json(run_root / "registration" / "a03_spec.json", spec)
    write_json(run_root / "registration" / "source_identity.json", source_identity(root))
    write_json(run_root / "receipts" / "asset_check.json", chk)
    (run_root / "receipts" / "case_ledger.jsonl").write_text("")
    return run_root


# ------------------------------------------------------------------------------------------------ ledger
def ledger_append(run_root, event, actor, split, case_id, **extra):
    path = Path(run_root) / "receipts" / "case_ledger.jsonl"
    line = json.dumps({"ts": utc_now(), "event": event, "actor": actor, "split": split, "case_id": case_id, **extra}, sort_keys=True, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def ledger_state(run_root):
    last = {}
    for line in (Path(run_root) / "receipts" / "case_ledger.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            last[(e["actor"], e["split"], e["case_id"])] = e["event"]
    return last


def _finite(ep):
    for d in ep.get("decisions", []):
        for p in d["probs"].values():
            if not math.isfinite(p):
                return False
    return True


# ------------------------------------------------------------------------------------------------ evaluation
def load_model(root, key, device):
    from .imitation import make_imitation_policy
    from ..torch_rl import load_checkpoint
    m = MODELS[key]
    path = Path(root) / m["path"]
    if sha256_file(path) != m["sha256"]:
        raise A03Error("checkpoint hash mismatch for %s" % key)
    policy = make_imitation_policy(m["method"], device, 0)
    load_checkpoint(path, policy)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def evaluate_method(run_root, key, policy, split_cases, solver, hops, evaluate_case, log=print):
    """Evaluate every registered (split, case) once, in the frozen order; per-case progress is durable. Returns counts."""
    run_root = Path(run_root)
    out = run_root / "eval" / key
    out.mkdir(parents=True, exist_ok=True)
    done = ledger_state(run_root)
    counts = {"completed": 0, "skipped_completed": 0, "technical_incomplete": 0}
    for split in ORDER:
        for c in split_cases[split]:
            k = (key, split, c.case_id)
            st = done.get(k)
            if st == "COMPLETED":
                counts["skipped_completed"] += 1
                continue
            if st in ("STARTED", "TECHNICAL_INCOMPLETE"):
                if st == "STARTED":
                    ledger_append(run_root, "TECHNICAL_INCOMPLETE", key, split, c.case_id, reason="started earlier without a recorded result; not re-run")
                counts["technical_incomplete"] += 1
                continue
            ledger_append(run_root, "STARTED", key, split, c.case_id, checkpoint_sha256=MODELS[key]["sha256"])
            try:
                t0 = time.perf_counter()
                ep = evaluate_case(policy, c, solver, hops, True)
                ep["wall_seconds"] = time.perf_counter() - t0
                if not _finite(ep):
                    raise A03Error("non-finite probability")
            except Exception as exc:                                                     # recorded, never retried
                ledger_append(run_root, "TECHNICAL_INCOMPLETE", key, split, c.case_id, reason="%s: %s" % (type(exc).__name__, exc))
                counts["technical_incomplete"] += 1
                continue
            with open(out / "_episodes.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"split": split, **ep}, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            ledger_append(run_root, "COMPLETED", key, split, c.case_id)
            counts["completed"] += 1
        log("[a03] %s %s done %s" % (key, split, counts))
    return counts


def summarize_episodes(eps, planned):
    from .train import summarize
    s = summarize(eps) if eps else {"n": 0, "success_n": 0, "decision_perfect_n": 0, "success_rate": None, "decision_perfect_rate": None}
    s["planned"] = planned
    s["technical_incomplete"] = planned - len(eps)
    return s


def assemble_method(run_root, key):
    """eval/<key>/{a0,a1,a2,b}_final.json (episodes without decisions) and decision_log.jsonl from the durable per-case records."""
    run_root = Path(run_root)
    out = run_root / "eval" / key
    spec = json.loads((run_root / "registration" / "a03_spec.json").read_text())
    rows = [json.loads(x) for x in (out / "_episodes.jsonl").read_text().splitlines() if x.strip()]
    with open(out / "decision_log.jsonl", "w", encoding="utf-8") as f:
        for e in rows:
            for d in e.get("decisions", []):
                f.write(json.dumps({"method": key, "split": e["split"], "case_id": e["case_id"], **d}, default=str) + "\n")
    for split in ORDER:
        eps = [{k: v for k, v in e.items() if k != "decisions"} for e in rows if e["split"] == split]
        write_json(out / ("%s_final.json" % split), {"method": key, "run_id": MODELS[key]["run_id"], "split": split, "checkpoint_sha256": MODELS[key]["sha256"],
                                                      "summary": summarize_episodes(eps, len(spec["case_manifest"][split])), "episodes": eps})


def run_planner(run_root, split_cases, log=print):
    """The reference plan of every initial problem: a COLD solver per case (search nodes and wall time are from-scratch values)."""
    from . import planner as P
    from . import state as S
    run_root = Path(run_root)
    done = ledger_state(run_root)
    out = {"splits": {}, "cold_cache_per_case": True}
    for split in ORDER:
        rows = []
        for c in split_cases[split]:
            k = ("planner", split, c.case_id)
            if done.get(k) in ("COMPLETED", "STARTED", "TECHNICAL_INCOMPLETE"):
                raise A03Error("planner case already started: %s" % (k,))
            ledger_append(run_root, "STARTED", "planner", split, c.case_id)
            try:
                s = P.Solver()
                t0 = time.perf_counter()
                plan = s.one_optimal_plan(c.problem.init, c.problem.goal)
                cpu = time.perf_counter() - t0
                st = c.problem.init
                for act in plan:
                    st = S.apply(st, act)
                rows.append({"case_id": c.case_id, "n_blocks": c.problem.n, "optimal_length": c.optimal_length, "plan_length": len(plan), "success": S.goal_satisfied(c.problem, st),
                             "timeout": False, "expanded_nodes": s.stats.expanded, "generated_nodes": s.stats.generated, "wall_seconds": cpu})
            except Exception as exc:
                ledger_append(run_root, "TECHNICAL_INCOMPLETE", "planner", split, c.case_id, reason="%s: %s" % (type(exc).__name__, exc))
                continue
            ledger_append(run_root, "COMPLETED", "planner", split, c.case_id)
        out["splits"][split] = rows
        log("[a03] planner %s %d cases" % (split, len(rows)))
    write_json(run_root / "eval" / "planner_baseline.json", out)
    return out


# ------------------------------------------------------------------------------------------------ A0 pairing (diagnostic only)
def _relabelings(dev, a0):
    """All block maps m (dev index -> A0 index) that carry the dev problem onto the A0 problem (colours, initial state, goal)."""
    n = len(dev.problem.names)
    if n != len(a0.problem.names):
        return []
    out = []
    for perm in itertools.permutations(range(n)):
        def mp(v):
            return perm[v] if v >= 0 else v
        if all(a0.problem.colors[perm[i]] == dev.problem.colors[i] for i in range(n)) and all(a0.problem.init[perm[i]] == mp(dev.problem.init[i]) for i in range(n)) \
                and all(a0.problem.goal[perm[i]] == mp(dev.problem.goal[i]) for i in range(n)):
            out.append(perm)
    return out


def _map_action(aid, dev_names, a0_names, perm):
    parts = aid.split(":")
    idx = {n: i for i, n in enumerate(dev_names)}
    return ":".join(parts[:2] + [a0_names[perm[idx[x]]] if x in idx else x for x in parts[2:]])


def a0_pair(dev_case, a0_case, dev_ep, a0_ep):
    if dev_ep is None or a0_ep is None or "decisions" not in dev_ep or "decisions" not in a0_ep:
        return {"status": "PAIR_DETAIL_UNAVAILABLE"}
    maps = _relabelings(dev_case, a0_case)
    if not maps:
        return {"status": "PAIR_DETAIL_UNAVAILABLE", "reason": "no block relabelling carries the dev problem onto the A0 problem"}
    dev_names, a0_names = dev_case.problem.names, a0_case.problem.names
    seq_a0 = [d["selected"] for d in a0_ep["decisions"]]
    same_any = any([_map_action(d["selected"], dev_names, a0_names, p) for d in dev_ep["decisions"]] == seq_a0 for p in maps)
    return {"status": "PAIRED", "n_valid_relabelings": len(maps), "same_selected_sequence": same_any, "same_success": dev_ep["success"] == a0_ep["success"],
            "same_decision_perfect": dev_ep["decision_perfect"] == a0_ep["decision_perfect"], "steps_dev": dev_ep["steps"], "steps_a0": a0_ep["steps"]}


# ------------------------------------------------------------------------------------------------ report tables
def cell(eps, field):
    return "%d/%d" % (sum(1 for e in eps if e[field]), len(eps))


def paired_counts(a, b, field):
    ka = {e["case_id"]: e[field] for e in a}
    kb = {e["case_id"]: e[field] for e in b}
    ids = sorted(set(ka) & set(kb))
    return {"n": len(ids), "both": sum(ka[i] and kb[i] for i in ids), "only_first": sum(ka[i] and not kb[i] for i in ids), "only_second": sum(kb[i] and not ka[i] for i in ids),
            "neither": sum(not ka[i] and not kb[i] for i in ids)}
