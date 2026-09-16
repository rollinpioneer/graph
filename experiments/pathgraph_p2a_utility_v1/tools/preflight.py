#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2a.score import load_frozen
from p2a.io import dump

BASE='8e86b246e900e1e12dbb21c88b95046a06c8e4c5'
FROZEN=['upgrade_v2/p1_reward_repair_v6','upgrade_v2/p1_mainline_grasp_v6gm1',
'upgrade_v2/p1_independent_state_holdout_v1','tools/stage5/lib']
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--v6-tools',required=True);p.add_argument('--extra-methods',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    subprocess.run(['git','-C',str(a.repo),'cat-file','-e',BASE+'^{commit}'],check=True)
    diff=subprocess.check_output(['git','-C',str(a.repo),'diff','--name-only',BASE,'--',*FROZEN],text=True)
    if diff.strip(): raise RuntimeError('frozen file changes: '+diff)
    _,_,_,lock=load_frozen(a.v6_tools,a.extra_methods)
    import numpy,torch
    dump(a.out,dict(status='PASS_SOURCE_ONLY',base=BASE,sources=lock,
        numpy=numpy.__version__,torch=torch.__version__,cuda_available=torch.cuda.is_available(),
        does_not_certify_training=True))
    print(a.out)
if __name__=='__main__':main()
