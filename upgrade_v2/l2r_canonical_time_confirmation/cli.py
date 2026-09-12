from __future__ import annotations
import argparse,json
from pathlib import Path


def main()->int:
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    q=s.add_parser("collect"); q.add_argument("--output",type=Path,required=True); q.add_argument("--workers",type=int,default=2)
    for name in ("detect","build-inputs","run-candidates"): q=s.add_parser(name); q.add_argument("--root",type=Path,required=True)
    q=s.add_parser("reference"); q.add_argument("--root",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
    q=s.add_parser("generator"); q.add_argument("--root",type=Path,required=True); q.add_argument("--reference",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
    q=s.add_parser("finalize"); q.add_argument("--artifact",type=Path,required=True); q.add_argument("--external",type=Path,required=True); q.add_argument("--package",type=Path)
    a=p.parse_args()
    if a.command=="collect":
        from .collector import collect; result=collect(a.output,a.workers)
    elif a.command=="detect":
        from .pipeline import detect; result=detect(a.root)
    elif a.command=="build-inputs":
        from .pipeline import build_inputs; result=build_inputs(a.root)
    elif a.command=="run-candidates":
        from .pipeline import run_candidates; result=run_candidates(a.root)
    elif a.command=="reference":
        from .pipeline import reference; result=reference(a.root,a.output)
    elif a.command=="generator":
        from .pipeline import generator; result=generator(a.root,a.reference,a.output)
    elif a.command=="finalize":
        from .finalize import finalize; result=finalize(a.artifact,a.external,a.package)
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result.get("status")!="FAIL" else 2
if __name__=="__main__": raise SystemExit(main())
