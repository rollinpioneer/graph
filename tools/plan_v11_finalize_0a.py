#!/usr/bin/env python3
"""Write Plan v1.1 Stage 0A reports from measured reference executions. No PPO."""
from __future__ import annotations
import json, hashlib, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
OUT = ROOT / "runs" / "stage_0a"
REP = ROOT / "reports"
ST = ROOT / "status"
MIG = ROOT / "migration"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()

def main():
    payload = json.loads((OUT / "reference_execution_manifest.json").read_text(encoding="utf-8"))
    attempts = []
    ap = OUT / "reference_attempts.jsonl"
    if ap.exists():
        for line in ap.read_text(encoding="utf-8").splitlines():
            if line.strip():
                attempts.append(json.loads(line))
    fam = payload["families"]
    H = payload["H"]
    ok = bool(payload.get("ok")) and all(fam[t]["successes"] >= 5 and fam[t]["T_ref_median"] for t in ("D0", "T_B", "T_C"))
    tcap = {
        "D0": 16384 * float(fam["D0"]["d_ref_median"]),
        "T_B": 65536 * float(fam["T_B"]["d_ref_median"]),
        "T_C": 65536 * float(fam["T_C"]["d_ref_median"]),
    }
    status = "PASS" if ok else "BLOCKED"
    doc = {
        "stage": "0A",
        "plan_version": "1.1",
        "method_version": "2.1.1",
        "document_version": "3.1",
        "status": status,
        "execution_reason": None,
        "git_hash": git_hash(),
        "completed_at": utc(),
        "new_rl_runs": 0,
        "H_seconds": H,
        "H_rule": "suite H = max family median legal-scripted success episode duration (seconds)",
        "d_ref_rule": "per-family median per-skill duration of the legal success episodes used for T_ref",
        "T_ref_rule": "per-family median episode duration of 5 legal scripted successes",
        "Tcap_rule": "Tcap = Ncap * d_ref (Method 2.1.1; not 2*Ncap*d_ref)",
        "families": fam,
        "Tcap_seconds": tcap,
        "reference_attempts": len(attempts),
        "runtime_src": payload.get("runtime_src"),
        "old_stage_status_path_unmodified": "experiments/stage_status/stage_0a.json",
        "notes": [
            "Old experiments/stage_status/stage_0a.json remains the v2.1 BLOCKED historical ledger and was not overwritten.",
            "T_B is a distinct family (Inside(target,container) AND AtBuffer(second_object,buffer)); not a T_A relabel.",
            "Reference executions are not RL/BC data and are not counted as new-profile training runs.",
        ],
        "must_bind_remaining": [
            "qwen3.8-max-0902 model access still MUST_VERIFY on first live API call (Stage 0C T_B materialization)."
        ],
    }
    write_json(OUT / "calibration.json", doc)
    write_json(ST / "stage_0a.json", doc)
    write_json(MIG / "reference_execution_manifest.json", payload)
    md = []
    md.append("# Stage 0A — Environment, Interface and Time Binding (Plan v1.1 / Method 2.1.1)")
    md.append("")
    md.append("Status: `" + status + "`.")
    md.append("")
    md.append("This report is the new-profile Stage 0A ledger. Historical v2.1 experiments/stage_status/stage_0a.json (BLOCKED) was snapshotted and not overwritten.")
    md.append("")
    md.append("## Calibration")
    md.append("")
    md.append("| family | attempts | legal successes | T_ref median (s) | d_ref median (s) | success cases |")
    md.append("|---|---:|---:|---:|---:|---|")
    for t in ("D0", "T_B", "T_C"):
        f = fam[t]
        md.append("| %s | %s | %s | %s | %s | %s |" % (t, f["attempts"], f["successes"], f["T_ref_median"], f["d_ref_median"], ",".join(f["success_case_ids"])))
    md.append("")
    md.append("- Suite H = max T_ref = **%s seconds**." % H)
    md.append("- Discount: Gamma = 2^(-d_seconds/H) with H from this measurement.")
    md.append("- Tcap(D0 smoke) = 16384 * d_ref(D0) = **%s s**." % tcap["D0"])
    md.append("- Tcap(T_B core) = 65536 * d_ref(T_B) = **%s s**." % tcap["T_B"])
    md.append("- Tcap(T_C core) = 65536 * d_ref(T_C) = **%s s**." % tcap["T_C"])
    md.append("")
    md.append("## Evidence files")
    md.append("")
    md.append("- runs/stage_0a/reference_attempts.jsonl sha256=%s" % sha(OUT / "reference_attempts.jsonl"))
    md.append("- runs/stage_0a/reference_execution_manifest.json sha256=%s" % sha(OUT / "reference_execution_manifest.json"))
    md.append("- runs/stage_0a/reference_run.log sha256=%s" % sha(OUT / "reference_run.log"))
    md.append("")
    md.append("## Binding")
    md.append("")
    md.append("Reused verified LIBERO/robosuite/MuJoCo stack, DashScope Beijing endpoint binding, and D0/T_C splits. New T_B split: configs/splits/T_B_stage_0a.json (train seeds 4100+, dev seeds 8100+).")
    md.append("")
    md.append("New-profile RL training runs this stage: **0**.")
    md.append("")
    (REP / "stage_0a_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "H": H, "ok": ok}, ensure_ascii=False))
    if not ok:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
