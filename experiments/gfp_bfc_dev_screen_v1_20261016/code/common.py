import os, sys, json, hashlib, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FORMAL = Path('/home/xushijie3/work/gfp_eta0_ablation_20261015')
PYTHON = '/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python'
sys.path.insert(0, str(ROOT/'source/gfp'))
def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def load_base(seed):
    import numpy as np, flax
    from agents.gfp import GFPAgent
    start={0:600000,3:300000}[seed]
    ds=list((FORMAL/f'runs/formal/G_task1_s{seed}/out').glob('*/date_*'));assert len(ds)==1
    cfg=json.loads((ds[0]/'config.json').read_text());ac=cfg['agent'].copy();ac['log_metrics']=True
    ag=GFPAgent.create(seed,np.zeros((1,69),np.float32),np.zeros((1,21),np.float32),ac)
    path=ds[0]/f'params_{start}.pkl'
    with open(path,'rb') as f: state=pickle.load(f)['agent']
    ag=flax.serialization.from_state_dict(ag,state)
    assert int(ag.network.step)==start+1 and int(ag.network.opt_state[0].count)==start
    return ag,cfg,path
def make_agent(seed, arm):
    from bfc import BFCAgent
    base,cfg,path=load_base(seed)
    spec=json.loads((ROOT/'protocol.json').read_text())['arms'][arm]
    if arm=='R0': return base,cfg,path
    assert base.config['target_agg']=='mean' and base.config['q_agg']=='mean'
    ag=BFCAgent(base.rng,base.network,base.config,reference=base.network.params['modules_actor'],
                tau_pi=spec['tau_pi'],epsilon=spec['epsilon'])
    return ag,cfg,path
