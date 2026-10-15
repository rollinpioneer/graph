import os,sys,json,hashlib,pickle,importlib.util,ast
from pathlib import Path
R=Path(__file__).resolve().parents[1]; E=Path('/home/xushijie3/work/grebrac_l1_v1_20261015'); U=Path('/home/xushijie3/work/flow_critic_dual_host_prep_v1_20261014/upstream_snapshots/gfp')
sys.path.insert(0,str(R/'workspaces/gfp'))
import numpy as np,jax,jax.numpy as jnp,flax
from agents.rebrac import ReBRACAgent
from utils.flax_utils import restore_agent
from training_wiring import state_digest,finite,digest,json_write
spec=importlib.util.spec_from_file_location('adopted_rebrac',E/'source/gfp/agents/rebrac.py'); m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
task='humanoidmaze-medium-navigate-singletask-task4-v0'
recipe=json.loads((R/'configs/z01_public_task_configs.json').read_text())['rebrac_'+task]['config']
s=np.load('/home/xushijie3/work/gfp_guidance_distance_z01_20261015/arrays/train_samples.npz')
obs=s['observations'][:256];act=s['actions'][:256]
batch=dict(observations=obs,actions=act,next_observations=obs[::-1].copy(),next_actions=act[::-1].copy(),rewards=-np.ones(256,np.float32),masks=np.ones(256,np.float32))
report={'optimizer_updates':0,'environment_steps':0,'runs':[],'limitations':['Existing native spawn evaluation did not inherit global RNG as the new runner does; primary evaluation is rerun with matched RNG.','Historical every-dispatch health telemetry and NumPy/Python checkpoint RNG sidecars absent; no resumable-training claim.']}
for rel in ['utils/networks.py','utils/flax_utils.py','utils/datasets.py','envs/env_utils.py','main.py']:
 a=(U/rel).read_bytes(); b=(E/'source/gfp'/rel).read_bytes()
 assert a==b,('upstream mismatch',rel)
for seed in [0,1]:
 src=E/f'runs/formal/R_task4_s{seed}'; out=next(src.glob('out/*/*/config.json')).parent
 cfg=json.loads((out/'config.json').read_text()); assert cfg['seed']==seed and cfg['env_name']==task and cfg['online_steps']==0 and cfg['offline_steps']==1000000
 assert cfg['agent']['guidance'] is False
 for k,v in recipe['agent'].items(): assert cfg['agent'][k]==v,(k,v,cfg['agent'][k])
 config=dict(cfg['agent'],log_metrics=True)
 agent=ReBRACAgent.create(seed,obs[:1],act[:1],config)
 altered=m.ReBRACAgent.create(seed,obs[:1],act[:1],config)
 assert state_digest(agent)==state_digest(altered)
 run={'seed':seed,'source':str(src),'exit':json.loads((src/'exit.json').read_text()),'checkpoints':[]}
 assert run['exit']['exit']==0
 for step in [200000,600000,800000,900000,1000000]:
  p=out/f'params_{step}.pkl'; sha=digest(p); restored=restore_agent(agent,str(out),step)
  finite((restored.network.params,restored.network.opt_state)); assert int(restored.network.step)==step+1
  assert int(restored.network.opt_state[0].count)==step
  run['checkpoints'].append(dict(step=step,path=str(p),sha256=sha,state_sha256=state_digest(restored)))
  assert digest(p)==sha
 altered=restore_agent(altered,str(out),1000000)
 key=jax.random.PRNGKey(20261015)
 def calc(a):return jax.value_and_grad(lambda p:a.total_loss(batch,p,rng=key),has_aux=True)(a.network.params)
 aa=calc(restored);bb=calc(altered); finite((aa,bb)); errs=[]
 for x,y in zip(jax.tree_util.tree_leaves(aa),jax.tree_util.tree_leaves(bb)):
  assert np.allclose(x,y,atol=1e-6,rtol=1e-6);errs.append(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
 assert state_digest(restored)==state_digest(altered)
 run['loss_gradient_max_difference']=max(errs);run['official_path_equivalent']=True
 d=R/f'runs/ReBRAC_seed{seed}';d.mkdir(exist_ok=True);json_write(d/'runtime.json',dict(output_dir=str(out),cfg=cfg,adopted=True))
 json_write(d/'status.json',dict(status='COMPLETED_ADOPTED',step=1000000,exit_code=0,source=str(src)))
 json_write(d/'checkpoint_index.json',run['checkpoints']);report['runs'].append(run)
report['status']='PASS';json_write(R/'receipts/rebrac_adoption.json',report);print(json.dumps(report),flush=True)
