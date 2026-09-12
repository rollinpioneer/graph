from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_adapter import run_online_candidates
from .collector import collect
from .detector_adapter import detect_confirmation
from .evaluation import evaluate
from .generator_gate import validate_generator
from .io_utils import write_json
from .logical_fault_injection import build_candidate_inputs
from .package_results import summarize
from .protocol import protocol_lock
from .reference_builder import build_reference
from .static_lock import write_locks


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p=sub.add_parser("write-protocol"); p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("write-locks"); p.add_argument("--repo",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--runner-commit",required=True)
    p=sub.add_parser("collect"); p.add_argument("--repo",type=Path); p.add_argument("--registry",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--workers",type=int,default=2)
    p=sub.add_parser("detect-rgb"); p.add_argument("--confirmation-root",type=Path,required=True); p.add_argument("--workers",type=int,default=4)
    p=sub.add_parser("build-candidate-input"); p.add_argument("--confirmation-root",type=Path,required=True)
    p=sub.add_parser("run-candidates"); p.add_argument("--confirmation-root",type=Path,required=True)
    p=sub.add_parser("build-reference"); p.add_argument("--confirmation-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True)
    p=sub.add_parser("validate-generator"); p.add_argument("--confirmation-root",type=Path,required=True); p.add_argument("--reference-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("evaluate"); p.add_argument("--repo",type=Path,required=True); p.add_argument("--confirmation-root",type=Path,required=True); p.add_argument("--reference-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True)
    p=sub.add_parser("summarize"); p.add_argument("--artifact-root",type=Path,required=True); p.add_argument("--external-data-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--package-path",type=Path)
    args=parser.parse_args()
    if args.command=="write-protocol": result=protocol_lock(); write_json(args.output,result)
    elif args.command=="write-locks": result=write_locks(args.repo,args.output_root,args.runner_commit)
    elif args.command=="collect": result=collect(args.registry,args.output_root,args.workers)
    elif args.command=="detect-rgb": result=detect_confirmation(args.confirmation_root,args.workers)
    elif args.command=="build-candidate-input": result=build_candidate_inputs(args.confirmation_root)
    elif args.command=="run-candidates": result=run_online_candidates(args.confirmation_root)
    elif args.command=="build-reference": result=build_reference(args.confirmation_root,args.output_root)
    elif args.command=="validate-generator": result=validate_generator(args.confirmation_root,args.reference_root,args.output); write_json(args.output,result)
    elif args.command=="evaluate": result=evaluate(args.repo,args.confirmation_root,args.reference_root,args.output_root)
    elif args.command=="summarize": result=summarize(args.artifact_root,args.output_root,args.external_data_root,args.package_path)
    else: raise AssertionError(args.command)
    print(json.dumps(result,indent=2,sort_keys=True,default=str)); return 0 if result.get("status") != "FAIL" else 2


if __name__=="__main__": raise SystemExit(main())
