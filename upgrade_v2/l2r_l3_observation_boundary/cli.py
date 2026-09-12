import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);q=s.add_parser("validate");q.add_argument("--repo",type=Path,required=True);q=s.add_parser("static");q.add_argument("--repo",type=Path,required=True);q.add_argument("--output",type=Path,required=True);q.add_argument("--commit",required=True);q=s.add_parser("collect");q.add_argument("--output",type=Path,required=True);q.add_argument("--workers",type=int,default=2);q=s.add_parser("evaluate");q.add_argument("--root",type=Path,required=True);q.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.cmd=="validate":from .static_lock import validate;r=validate(a.repo)
 elif a.cmd=="static":from .static_lock import write_static;r=write_static(a.repo,a.output,a.commit)
 elif a.cmd=="collect":from .runner import collect;r=collect(a.output,a.workers)
 else:from .evaluation import evaluate;r=evaluate(a.root,a.output)
 print(json.dumps(r,indent=2,sort_keys=True));return 0 if r.get("status")=="PASS" or r.get("passed") else 2
if __name__=="__main__":raise SystemExit(main())

