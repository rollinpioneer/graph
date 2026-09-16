from __future__ import annotations
import argparse,json
from .experiment import prepare,evaluate
from .stats import analyze

def main():
    ap=argparse.ArgumentParser(description='P2A reference-MDP utility experiment; not physics')
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--out',required=True); p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('weights'); p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--v6-tools',required=True); p.add_argument('--extra-methods',required=True)
    p=sub.add_parser('train'); p.add_argument('--weights',required=True); p.add_argument('--out',required=True)
    p.add_argument('--method',required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=5000); p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--device',default='cpu')
    p=sub.add_parser('evaluate'); p.add_argument('--checkpoint',required=True); p.add_argument('--data',required=True)
    p.add_argument('--split',choices=['validation','test'],required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('analyze'); p.add_argument('--results',required=True); p.add_argument('--out',required=True); p.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.cmd=='prepare': result=prepare(a.out,a.smoke)
    elif a.cmd=='weights':
        from .score import score_dataset
        result=score_dataset(a.data,a.out,a.v6_tools,a.extra_methods)
    elif a.cmd=='train':
        from .train import train_one
        result=train_one(a.weights,a.out,a.method,a.seed,a.steps,a.batch_size,a.device)
    elif a.cmd=='evaluate': result=evaluate(a.checkpoint,a.data,a.split,a.out)
    else: result=analyze(a.results,a.out,a.smoke)
    print(json.dumps(dict(command=a.cmd,output=str(result)),indent=2))
if __name__=='__main__': main()
