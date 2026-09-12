from __future__ import annotations
import zipfile
from pathlib import Path
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_json,sha256,write_json


def finalize(artifact:Path,external:Path,package:Path|None=None)->dict:
    historical=read_json(artifact/"zero_physics_v1/gate.json"); conformance=read_json(artifact/"conformance_v1/conformance.json"); generator=read_json(artifact/"physical_v1/generator_gate.json"); performance=read_json(artifact/"evaluation_v1/summary.json")
    passed=historical["status"]==conformance["status"]==performance["status"]=="PASS" and generator["evaluation_allowed"]
    decision={"schema":"l2rar2_r22_final_decision_v1","status":"PASS" if passed else "FAIL","historical_replay_pass":historical["status"]=="PASS","conformance_pass":conformance["status"]=="PASS","physical_generator_pass":generator["evaluation_allowed"],"physical_performance_pass":performance["status"]=="PASS","selected_candidate_id":"O_C3_CLP3_CANONICAL_TIME" if passed else None,"l3_entry_ready":passed,"l3_entry_allowed":False,"l3_started":False}
    final=artifact/"final_v1"; write_json(final/"decision.json",decision); write_json(final/"external_data.json",{"path":str(external.resolve()),"size_bytes":sum(p.stat().st_size for p in external.rglob("*") if p.is_file())})
    files=[{"path":str(p.relative_to(artifact)),"sha256":sha256(p),"size_bytes":p.stat().st_size} for p in sorted(artifact.rglob("*")) if p.is_file() and p!=final/"package_manifest.json"]
    write_json(final/"package_manifest.json",{"schema":"l2rar2_r22_package_manifest_v1","files":files,"file_count":len(files)})
    if package:
        package.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as z:
            for item in files+[{"path":"final_v1/package_manifest.json"}]: z.write(artifact/item["path"],item["path"])
        write_json(package.with_suffix(package.suffix+".sha256.json"),{"path":str(package),"sha256":sha256(package),"size_bytes":package.stat().st_size})
    return decision
