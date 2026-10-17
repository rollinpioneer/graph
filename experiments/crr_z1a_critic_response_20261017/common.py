from pathlib import Path
import sys,json,pickle,hashlib,copy,os
import numpy as np
R=Path(__file__).resolve().parent
SRC=Path('/home/xushijie3/work/gfp_eta0_ablation_20261015/source/gfp')
sys.path.insert(0,str(SRC))
import gymnasium as gym,mujoco
ENV_CALLS=dict(construct=0,reset=0,step=0)
def forbidden(*a,**kw):raise RuntimeError('CRR-Z1a prohibits all environment operations')
gym.make=forbidden;gym.Env.reset=forbidden;gym.Env.step=forbidden;mujoco.mj_step=forbidden
import jax,jax.numpy as jnp,flax,optax
from flax import linen as nn
from agents.gfp import GFPAgent
from utils.flax_utils import TrainState,ModuleDict
from utils.networks import Value,ActorVectorField
nn.Module.init=forbidden;GFPAgent.create=forbidden
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False))
def digest(tree):
 h=hashlib.sha256()
 for path,v in jax.tree_util.tree_flatten_with_path(tree)[0]:
  a=np.asarray(v);h.update(str(path).encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def atomic_pickle(p,x):
 p=Path(p);assert not p.exists(),str(p);tmp=p.with_suffix('.tmp')
 with tmp.open('wb') as f:pickle.dump(x,f)
 tmp.rename(p)
C=Value((512,)*4,layer_norm=True,num_ensembles=2)
A=ActorVectorField((512,)*4,action_dim=21,layer_norm=False,need_time=False)
V=ActorVectorField((512,)*4,action_dim=21,layer_norm=False,need_time=True,encode_time_dim=64)
def cfg(seed):
 a=json.loads((R/f'sources/s{seed}_config.json').read_text())['agent'];a.update(ob_dims=(69,),action_dim=21)
 if a['eta_temperature_online'] is None:a['eta_temperature_online']=a['eta_temperature']
 assert a['q_agg']==a['target_agg']=='mean' and a['guidance_agg']=='min' and not a['target_use_next'] and a['discount']==.995 and a['batch_size']==256 and a['flow_steps']==10 and a['encoder'] is None
 return flax.core.FrozenDict(a)
def read(path,step):
 with open(path,'rb') as f:p=pickle.load(f)['agent']
 assert int(p['network']['step'])==step+1 and int(p['network']['opt_state']['0']['count'])==step
 return p
def restore(p,seed,with_optimizer=True):
 config=cfg(seed);definition=ModuleDict(dict(critic=copy.deepcopy(C),target_critic=copy.deepcopy(C),actor=copy.deepcopy(A),vabc=copy.deepcopy(V)))
 params=jax.tree_util.tree_map(jnp.asarray,p['network']['params']);tx=optax.adam(config['lr']) if with_optimizer else None
 opt=flax.serialization.from_state_dict(tx.init(params),p['network']['opt_state']) if tx else None
 net=TrainState(step=jnp.asarray(p['network']['step']),apply_fn=definition.apply,model_def=definition,params=params,tx=tx,opt_state=opt)
 g=GFPAgent(rng=jnp.asarray(p['rng']),network=net,config=config)
 if with_optimizer:assert digest(flax.serialization.to_state_dict(g))==digest(p),'full state restore mismatch'
 return g
def dataset():return {k:jnp.asarray(np.load(R/f'arrays/data_{k}.npy',mmap_mode='r')) for k in ['observations','next_observations','actions','rewards','masks']}
@jax.jit
def fixed_actions(params,obs,nobs,z):
 aa=jnp.clip(A.apply({'params':params['modules_actor']},nobs,z),-1,1);va=z
 for i in range(10):va=va+V.apply({'params':params['modules_vabc']},obs,va,jnp.full((*obs.shape[:-1],1),i/10),is_encoded=True)/10
 return aa,jnp.clip(va,-1,1)
@jax.jit
def probe(cp,tp,obs,aa,va):
 q=lambda p,a:C.apply({'params':p},obs,a).mean(0)
 g=jax.grad(lambda a:jnp.sum(q(cp,a)))(aa)
 return (q(tp,aa)+q(tp,va))/2,g
