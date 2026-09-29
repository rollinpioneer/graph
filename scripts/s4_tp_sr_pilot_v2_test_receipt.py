#!/usr/bin/env python3
"""Build inventory/amendment_test_receipt.json from saved junit logs (targeted V2 tests, full suite on head, full suite on the unmodified base)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse(xml):
    cases, fails = 0, {}
    for tc in ET.parse(xml).getroot().iter("testcase"):
        cases += 1
        f = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        if f is not None:
            fails[f"{tc.get('classname')}::{tc.get('name')}"] = (f.get("message") or "") + "\n" + (f.text or "")
    skipped = sum(1 for _ in ET.parse(xml).getroot().iter("skipped"))
    return {"tests": cases, "skipped": skipped, "failures": fails}


def norm(t):
    t = re.sub(r"/home/xushijie2/graph_cp_disr_[a-z0-9_]+", "<WT>", t)
    t = re.sub(r"/tmp/pytest-of-[^/]+/pytest-\d+", "<TMP>", t)
    t = re.sub(r"0x[0-9a-f]+", "<ADDR>", t)
    return t.strip()


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def main(d, out):
    d, out = Path(d), Path(out)
    logs = out / "inventory/test_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.suffix in (".log", ".xml", ".rc", ".txt"):
            shutil.copy(f, logs / f.name)
    t, h, b = parse(d / "targeted_head.xml"), parse(d / "full_head.xml"), parse(d / "full_base.xml")
    new_fail = sorted(set(h["failures"]) - set(b["failures"]))
    gone = sorted(set(b["failures"]) - set(h["failures"]))
    common = sorted(set(h["failures"]) & set(b["failures"]))
    same = {k: sha(norm(h["failures"][k])) == sha(norm(b["failures"][k])) for k in common}
    base_clean = (d / "base_status.txt").read_text().strip() == "" and (d / "base_status_after.txt").read_text().strip() == ""
    if not h["failures"]:
        cls = "NONE"
    elif not new_fail and common and all(same.values()) and base_clean:
        cls = "KNOWN_PREEXISTING_FAILURE"
    else:
        cls = "NEW_OR_UNMATCHED_FAILURE"
    doc = {"targeted_tests": t["tests"], "targeted_failures": sorted(t["failures"]), "targeted_all_passed": t["tests"] > 0 and not t["failures"],
           "compileall_head_rc": int((d / "compile_head.rc").read_text().strip()),
           "full_suite_head": {"tests": h["tests"], "skipped": h["skipped"], "failed": sorted(h["failures"])},
           "full_suite_base": {"commit": (d / "base_commit.txt").read_text().strip(), "worktree_clean_before_and_after": base_clean, "tests": b["tests"], "skipped": b["skipped"], "failed": sorted(b["failures"])},
           "new_full_suite_failures": len(new_fail), "new_failures": new_fail, "failures_fixed_on_head": gone,
           "common_failures_identical_normalized_traceback": same,
           "full_suite_failure_classification": cls,
           "known_preexisting_failure_conditions": {"fails_on_unmodified_base": bool(common), "same_traceback_and_assertion": bool(common) and all(same.values()),
                                                   "no_other_new_failures": not new_fail, "base_and_head_logs_saved": str(logs)},
           "failure_text_sha256": {k: {"head": sha(norm(h["failures"][k])), "base": sha(norm(b["failures"][k]))} for k in common}}
    (out / "inventory/amendment_test_receipt.json").write_text(json.dumps(doc, indent=1))
    (out / "inventory/test_base_head_comparison.md").write_text(
        "# Full-suite base vs head\n\n" + f"- base commit {doc['full_suite_base']['commit']}: {b['tests']} tests, failures {sorted(b['failures'])}\n"
        + f"- head: {h['tests']} tests, failures {sorted(h['failures'])}\n- new failures on head: {new_fail}\n- identical normalized failure text for shared failures: {same}\n"
        + f"- classification: {cls}\n- targeted V2 tests: {t['tests']} run, failures {sorted(t['failures'])}\n")
    print(json.dumps({k: doc[k] for k in ("targeted_all_passed", "new_full_suite_failures", "full_suite_failure_classification")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
