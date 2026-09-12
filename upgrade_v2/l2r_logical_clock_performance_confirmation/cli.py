from __future__ import annotations

import argparse, json
from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json
from .conformance import run as run_conformance
from .protocol import protocol_lock
from .registry import registry


def main() -> int:
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    q=s.add_parser("write-static"); q.add_argument("--output-root",type=Path,required=True)
    q=s.add_parser("write-locks"); q.add_argument("--repo",type=Path,required=True); q.add_argument("--output",type=Path,required=True); q.add_argument("--runner-commit",required=True)
    q=s.add_parser("conformance"); q.add_argument("--repo",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
    q=s.add_parser("collect"); q.add_argument("--output-root",type=Path,required=True); q.add_argument("--workers",type=int,default=2)
    for name in ("detect","build-inputs","run-candidates"):
        q=s.add_parser(name); q.add_argument("--root",type=Path,required=True)
    q=s.add_parser("build-reference"); q.add_argument("--root",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
    q=s.add_parser("generator"); q.add_argument("--root",type=Path,required=True); q.add_argument("--reference-root",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
    q=s.add_parser("evaluate"); q.add_argument("--repo",type=Path,required=True); q.add_argument("--root",type=Path,required=True); q.add_argument("--reference-root",type=Path,required=True); q.add_argument("--gate",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
    q=s.add_parser("finalize"); q.add_argument("--artifact-root",type=Path,required=True); q.add_argument("--external-root",type=Path,required=True); q.add_argument("--package",type=Path)
    a=p.parse_args()
    if a.command=="write-static": a.output_root.mkdir(parents=True,exist_ok=True); write_json(a.output_root/"protocol_lock.json",protocol_lock()); write_json(a.output_root/"confirmation_registry.json",registry()); result={"status":"PASS"}
    elif a.command=="write-locks":
        from .static_lock import write_locks
        result=write_locks(a.repo,a.output,a.runner_commit)
    elif a.command=="conformance": result=run_conformance(a.repo,a.output)
    elif a.command=="collect":
        from .collector import collect
        result=collect(a.output_root,a.workers)
    elif a.command=="detect":
        from .pipeline import detect
        result=detect(a.root)
    elif a.command=="build-inputs":
        from .pipeline import build_inputs
        result=build_inputs(a.root)
    elif a.command=="run-candidates":
        from .pipeline import run_candidates
        result=run_candidates(a.root)
    elif a.command=="build-reference":
        from .pipeline import build_reference
        result=build_reference(a.root,a.output)
    elif a.command=="generator":
        from .pipeline import generator
        result=generator(a.root,a.reference_root,a.output)
    elif a.command=="evaluate":
        from .evaluation import evaluate
        result=evaluate(a.repo,a.root,a.reference_root,a.gate,a.output)
    elif a.command=="finalize":
        from .finalize import finalize
        result=finalize(a.artifact_root,a.external_root,a.package)
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result.get("status")!="FAIL" else 2


if __name__=="__main__": raise SystemExit(main())
