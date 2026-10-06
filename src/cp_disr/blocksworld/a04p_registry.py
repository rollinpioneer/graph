"""Shared registration / identity / one-shot ledger helpers for the goal-progress runbook (phase 1 A04P and phase 2). Evaluation never trains and never opens an optimizer."""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path

from . import eval_a03 as E

BASE_COMMIT = "5140f16144d5361408cd5ddc35d18310a5ce9d71"
BRANCH = "codex/cp-disr-c1-bw-goal-progress-v1"
RUNBOOK_REL = "docs/c1_blocksworld/CP_DISR_C1_Audit_Based_Goal_Progress_Runbook_v1.md"
GP_REL = E.BASE_REL + "/goal_progress_v1"
A03_ROOT = E.BASE_REL + "/evaluation_a03/20261006T070810Z_a8f9593a"
B2_CKPT = E.MODELS["B2"]
OLD_ROOTS = E.OLD_ROOTS + (A03_ROOT,)
PILOT_REL = "configs/splits/c1_bw_a04p_pilot80_v1.json"
FROZEN_SOURCES = E.FROZEN_SOURCES + ("src/cp_disr/blocksworld/eval_a03.py", "scripts/c1_bw_eval_a03.py")
PILOT_NAMESPACE = "C1-BW-A04P-PILOT-v1"
# (slice, n, count, goal heights cycle, top colour: 0 = RED (training pattern), 1 = BLUE (colour reversal))
PILOT_SLICES = (("P-A1-4", 4, 8, ((3, 1), (2, 1, 1)), 1), ("P-A1-5", 5, 8, ((3, 1, 1), (2, 1, 1, 1)), 1),
                ("P-A2-4", 4, 6, ((2, 2),), 0), ("P-A2-5a", 5, 5, ((3, 2),), 0), ("P-A2-5b", 5, 5, ((2, 2, 1),), 0),
                ("P-B6", 6, 16, ((3, 3), (2, 2, 2), (4, 2), (3, 2, 1)), 0), ("P-B7", 7, 16, ((3, 2, 2), (4, 3), (3, 3, 1), (4, 2, 1)), 0), ("P-B8", 8, 16, ((3, 3, 2), (4, 2, 2), (4, 4), (3, 3, 1, 1)), 0))


def sha256_file(path):
    return E.sha256_file(path)


def source_identity(root, new_sources):
    root = Path(root)
    frozen = {}
    for rel in FROZEN_SOURCES:
        now = sha256_file(root / rel)
        base = hashlib.sha256(subprocess.check_output(["git", "-C", str(root), "show", "%s:%s" % (BASE_COMMIT, rel)])).hexdigest()
        frozen[rel] = {"sha256": now, "sha256_at_base_commit": base, "unchanged_since_base": now == base}
    return {"base_commit": BASE_COMMIT, "frozen_sources": frozen, "new_sources": {rel: {"sha256": sha256_file(root / rel)} for rel in new_sources if (root / rel).is_file()},
            "old_root_tree_ids_at_base": {rel: E.git(root, "rev-parse", "%s:%s" % (BASE_COMMIT, rel)) for rel in OLD_ROOTS},
            "all_frozen_unchanged": all(v["unchanged_since_base"] for v in frozen.values())}


def check_identity(root, run_root):
    """Clean tracked tree (the run root's own files excluded), frozen + registered sources unchanged, all old result trees unchanged."""
    import json
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    own = str(run_root.relative_to(root))
    if E.git(root, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)%s" % own):
        raise E.A03Error("tracked tree is dirty")
    reg = json.loads((run_root / "source_identity.json").read_text())
    for rel, v in reg["frozen_sources"].items():
        if sha256_file(root / rel) != v["sha256"]:
            raise E.A03Error("frozen source changed: %s" % rel)
    for rel, v in reg["new_sources"].items():
        if sha256_file(root / rel) != v["sha256"]:
            raise E.A03Error("registered source changed: %s" % rel)
    for rel in OLD_ROOTS:
        if E.git(root, "rev-parse", "HEAD:%s" % rel) != reg["old_root_tree_ids_at_base"][rel]:
            raise E.A03Error("an old result directory changed: %s" % rel)
    return E.git(root, "rev-parse", "HEAD")


def run_cases(run_root, actor, groups, evaluate_fn, episodes_path, log=print):
    """Evaluate every (group, case) once; progress is durable in the ledger; a started-but-unrecorded case is TECHNICAL_INCOMPLETE and never silently re-run."""
    import json
    run_root = Path(run_root)
    done = E.ledger_state(run_root)
    counts = {"completed": 0, "skipped_completed": 0, "technical_incomplete": 0}
    episodes_path = Path(episodes_path)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    for group, cases in groups:
        for c in cases:
            k = (actor, group, c.case_id)
            st = done.get(k)
            if st == "COMPLETED":
                counts["skipped_completed"] += 1
                continue
            if st in ("STARTED", "TECHNICAL_INCOMPLETE"):
                if st == "STARTED":
                    E.ledger_append(run_root, "TECHNICAL_INCOMPLETE", actor, group, c.case_id, reason="started earlier without a recorded result; not re-run")
                counts["technical_incomplete"] += 1
                continue
            E.ledger_append(run_root, "STARTED", actor, group, c.case_id)
            try:
                t0 = time.perf_counter()
                ep = evaluate_fn(c)
                ep["wall_seconds"] = time.perf_counter() - t0
                for d in ep.get("decisions", []):
                    for p in d.get("probs", {}).values():
                        if p != p or p in (float("inf"), float("-inf")):
                            raise E.A03Error("non-finite probability")
            except Exception as exc:
                E.ledger_append(run_root, "TECHNICAL_INCOMPLETE", actor, group, c.case_id, reason="%s: %s" % (type(exc).__name__, exc))
                counts["technical_incomplete"] += 1
                continue
            with open(episodes_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"group": group, **ep}, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            E.ledger_append(run_root, "COMPLETED", actor, group, c.case_id)
            counts["completed"] += 1
        log("[gp] %s %s %s" % (actor, group, counts))
    return counts
