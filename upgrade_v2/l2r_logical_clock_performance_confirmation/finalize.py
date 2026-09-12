from __future__ import annotations

import zipfile
from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_json, sha256, write_json


def finalize(artifact_root:Path,external_root:Path,package:Path|None=None)->dict:
    generator=read_json(artifact_root/"physical_v1/generator_gate.json")
    performance_path=artifact_root/"evaluation_v1/method_metrics.json"
    performance=read_json(performance_path) if performance_path.is_file() else {"candidate_performance_pass":False}
    conformance=read_json(artifact_root/"conformance_v1/conformance.json")
    passed=bool(generator.get("performance_evaluation_allowed") and performance.get("candidate_performance_pass") and conformance.get("status")=="PASS")
    decision={"schema":"l2rar2_r21_final_decision_v1","physical_generator_pass":bool(generator.get("performance_evaluation_allowed")),
              "candidate_performance_pass":bool(performance.get("candidate_performance_pass")),"logical_clock_conformance_pass":conformance.get("status")=="PASS",
              "selected_candidate_id":"O_C3_CLP2_LOGICAL_CLOCK" if passed else None,"r21_status":"PASS" if passed else "FAIL",
              "r20_decision_changed":False,"candidate_changed":False,"thresholds_changed":False,"time_scoring_changed":False,"physical_reference_changed":False,"l3_entry_allowed":False}
    final=artifact_root/"final_v1"; write_json(final/"decision.json",decision)
    (final/"report.md").write_text("# R21 logical-clock performance confirmation\n\n"
        f"- Physical Generator Gate: {'PASS' if decision['physical_generator_pass'] else 'FAIL'}\n"
        f"- Candidate physical performance: {'PASS' if decision['candidate_performance_pass'] else 'FAIL'}\n"
        f"- Logical-clock conformance: {'PASS' if decision['logical_clock_conformance_pass'] else 'FAIL'}\n"
        f"- Selected candidate: {decision['selected_candidate_id'] or 'none'}\n- L3 started: no\n",encoding="utf-8")
    write_json(final/"external_data.json",{"path":str(external_root.resolve()),"size_bytes":sum(p.stat().st_size for p in external_root.rglob("*") if p.is_file()),"raw_data_committed":False})
    files=[{"path":str(p.relative_to(artifact_root)),"sha256":sha256(p),"size_bytes":p.stat().st_size} for p in sorted(artifact_root.rglob("*")) if p.is_file() and p!=final/"package_manifest.json"]
    write_json(final/"package_manifest.json",{"schema":"l2rar2_r21_package_manifest_v1","files":files,"file_count":len(files)})
    if package:
        package.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as archive:
            for item in files+[ {"path":"final_v1/package_manifest.json"} ]: archive.write(artifact_root/item["path"],item["path"])
        write_json(package.with_suffix(package.suffix+".sha256.json"),{"path":str(package),"sha256":sha256(package),"size_bytes":package.stat().st_size})
    return decision
