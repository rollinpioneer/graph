#!/usr/bin/env python3
"""Offline 0D empty-patch erratum. Does not rerun episodes or overwrite historical logs."""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
OUT = ROOT / "runs" / "stage_0d"
REP = ROOT / "reports"
LOG = OUT / "random_episode_log.jsonl"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    rows = []
    if LOG.is_file():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    reconstructable = False
    reason = []
    if not rows:
        reason.append("random_episode_log.jsonl missing or empty")
    else:
        sample = rows[0]
        if "decisions" in sample and sample["decisions"]:
            reconstructable = True
        else:
            reason.append("episode log stripped per-decision nonempty/index fields; only slim episode rows remain")
        if all(int(r.get("n_empty_patch") or 0) == int(r.get("n_decisions") or 0) for r in rows):
            reason.append("n_empty_patch equals n_decisions on every episode, matching cid-not-in-int-list tautology")
    families = {}
    for task_id in ("T_B", "T_C"):
        eps = [r for r in rows if r.get("task_id") == task_id]
        families[task_id] = {
            "n_episodes": len(eps),
            "n_success": sum(1 for r in eps if r.get("success")),
            "n_decisions": sum(int(r.get("n_decisions") or 0) for r in eps),
            "logged_empty_patch_sum": sum(int(r.get("n_empty_patch") or 0) for r in eps),
            "prior_empty_episodes": sum(1 for r in eps if r.get("prior_empty")),
            "prior_nonempty_episodes": sum(1 for r in eps if not r.get("prior_empty")),
            "prior_edge_count_histogram": dict(Counter(int(r.get("prior_edge_count") or 0) for r in eps)),
            "selected_action_empty_patch_rate": "UNRELIABLE/NOT_RECONSTRUCTABLE",
            "legal_candidate_empty_patch_rate": "UNRELIABLE/NOT_RECONSTRUCTABLE",
            "historical_empty_patch_rate_was": 1.0,
            "historical_empty_patch_rate_status": "UNRELIABLE/NOT_RECONSTRUCTABLE",
        }
    doc = {
        "stage": "0D",
        "kind": "empty_patch_identity_erratum",
        "created_at": utc(),
        "overwrites_historical_logs": False,
        "reran_episodes": False,
        "original_success_records_retained": True,
        "bug": "run_episode compared string cid against integer nonempty index list (cid not in nonempty), so selected empty-patch count incremented on every decision",
        "reconstructable": reconstructable,
        "reconstructable_reason": reason,
        "denominators": {
            "selected_action_empty_patch_rate": "n_selected_empty / n_decisions",
            "legal_candidate_empty_patch_rate": "n_legal_candidates_with_empty_patch / n_legal_candidates",
            "prior_empty": "episode-start prior_edge_count == 0 (not missing cache)",
            "missing_cache": "registered cache file absent; never inferred from empty prior",
            "dp_strict_zero": "neural DP diagnostic, not empty patch",
            "no_measurement": "perception/fact missing, not empty patch",
        },
        "families": families,
        "note": "Uniform action sampling and success events are unaffected. Do not back-out rates from success counts.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "empty_patch_erratum.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# Stage 0D empty-patch statistical erratum",
        "",
        "Status: historical `empty_patch_rate=1.0` is **UNRELIABLE/NOT_RECONSTRUCTABLE**.",
        "",
        "Cause: `sample_action()` returned integer candidate indices in `nonempty`, but `run_episode()` used `cid not in nonempty` with string canonical IDs. That comparison is a tautology and counted every selected action as an empty patch.",
        "",
        "This file does **not** overwrite `random_episode_log.jsonl`, `decision_structure.csv`, `stage_0d_result.json`, or original success records. Episodes were **not** rerun.",
        "",
        "Correct denominators going forward:",
        "- selected-action empty patch rate: selected actions whose nominal contract patch is empty / decisions",
        "- legal-candidate empty patch rate: legal candidates whose nominal contract patch is empty / legal candidates",
        "- empty prior, empty nominal patch, DP==0, and missing measurement are distinct",
        "- missing source cache is not a legal empty prior",
        "",
        "Offline reconstruction: **not possible**. The slim episode log dropped per-decision `n_nonempty_patch` / index identity. `n_empty_patch` equals `n_decisions` on all 200 rows.",
        "",
        "Original uniform successes remain the exposure record: T_B 13/100, T_C 35/100.",
        "",
    ]
    (REP / "stage_0d_empty_patch_erratum.md").write_text("\n".join(md), encoding="utf-8")
    summary = REP / "stage_0d_summary.md"
    if summary.is_file():
        text = summary.read_text(encoding="utf-8")
        marker = "empty-patch statistical erratum"
        if marker not in text:
            text = text.rstrip() + "\n\n## Empty-patch accounting erratum\n\nSee `reports/stage_0d_empty_patch_erratum.md`. Historical `empty_patch_rate=1.0` is UNRELIABLE/NOT_RECONSTRUCTABLE; success counts are unchanged.\n"
            summary.write_text(text, encoding="utf-8")
    print(json.dumps({"reconstructable": reconstructable, "path": str(OUT / "empty_patch_erratum.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
