"""Read-only baseline loading and independent CPU evaluation instrumentation."""
import copy
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('JAX_PLATFORMS')=='cpu'
ROOT=Path(__file__).resolve().parent
OLD=Path('/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012')
spec=importlib.util.spec_from_file_location('rd01common',OLD/'common.py')
rd=importlib.util.module_from_spec(spec);spec.loader.exec_module(rd)
np,jax,ogbench=rd.np,rd.jax,rd.ogbench
import gymnasium as gym
import mujoco
from utils.evaluation import evaluate as official_evaluate

DEADLINE=datetime.datetime.fromisoformat('2026-10-12T21:48:05+08:00').timestamp()
EXPECTED={'navigate':'2d79774f5efbb9d22a17da85cf87854a8000b703d84eb2e84dbf120fa768eb1d',
          'stitch':'85df2128b4eb58c56388ac975ffc78b020021a1532c16354f6ca4042e9dc0fea'}
STATE=mujoco.mjtState.mjSTATE_INTEGRATION

def check_time():
    if time.time()>DEADLINE-120:raise RuntimeError('CPU_WALL_BUDGET_STOP')
def sha(p):return rd.sha(p)
def dump(p,x):
    with open(p,'xb') as f:pickle.dump(x,f,protocol=4)
def write(p,x):
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def csv_rows(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def digest(x):return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
def integration(u):
    state=np.empty(mujoco.mj_stateSize(u.model,STATE));mujoco.mj_getState(u.model,u.data,state,STATE);return state
def wrappers(env):
    ans=[]
    while isinstance(env,gym.Wrapper):
        ans.append(dict(class_name=type(env).__name__,fields={k:copy.deepcopy(env.__dict__[k]) for k in ('_elapsed_steps','_has_reset','_terminated','_truncated') if k in env.__dict__}));env=env.env
    return ans
def snapshot(env,info,reset_seed):
    u=env.unwrapped
    return dict(integration=integration(u),qpos=u.data.qpos.copy(),qvel=u.data.qvel.copy(),
                goal=np.asarray(info['goal']).copy(),goal_xy=np.asarray(u.cur_goal_xy).copy(),task=u.cur_task_id,
                env_rng=copy.deepcopy(u.np_random.bit_generator.state),action_rng=copy.deepcopy(env.action_space.np_random.bit_generator.state),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),wrappers=wrappers(env),reset_seed=reset_seed)
def restore(env,s):
    env.reset(seed=s['reset_seed'],options={'task_id':s['task'],'render_goal':False})
    u=env.unwrapped;u.set_goal(goal_xy=s['goal_xy'].copy())
    mujoco.mj_setState(u.model,u.data,s['integration'],STATE);mujoco.mj_forward(u.model,u.data)
    mujoco.mj_setState(u.model,u.data,s['integration'],STATE)
    u.np_random.bit_generator.state=copy.deepcopy(s['env_rng'])
    env.action_space.np_random.bit_generator.state=copy.deepcopy(s['action_rng'])
    w=env
    for item in s['wrappers']:
        assert type(w).__name__==item['class_name']
        for k,v in item['fields'].items():setattr(w,k,copy.deepcopy(v))
        w=w.env
    np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
    return u.get_ob().copy(),{'goal':s['goal'].copy()}

def load(kind):
    check_time();start=time.monotonic();path=OLD/'runs'/f'formal_{kind}'/'params_1000000.pkl'
    before=sha(path);assert before==EXPECTED[kind]
    env,data,val,config=rd.load(f'pointmaze-large-{kind}-v0');agent=rd.init_agent(data,config)
    with open(path,'rb') as f:state=pickle.load(f)
    assert state['step']==1000000
    agent=rd.flax.serialization.from_state_dict(agent,state['agent']);agent=jax.device_put(agent,jax.devices('cpu')[0])
    assert all(d.platform=='cpu' for d in jax.devices())
    env.close()
    return agent,dict(path=str(path),sha256=before,bytes=path.stat().st_size,step=1000000,load_seconds=time.monotonic()-start,
                      official_commit='3535a16c901e5de1233ac5a586401b3ddfa5e30d',devices=[str(d) for d in jax.devices()])

def env_new(kind):
    env=ogbench.make_env_and_datasets(f'pointmaze-large-{kind}-v0',env_only=True)
    assert env.spec.max_episode_steps==1000 and env.unwrapped._goal_tol==1.0
    assert env.unwrapped._teleport_info is None
    return env
mode=jax.jit(lambda a,s,g:np_clip(a.network.select('actor')(s,g,temperature=0.).mode()))
def np_clip(x):return jax.numpy.clip(x,-1,1)

class Trace(gym.Wrapper):
    def __init__(self,env,scene=None):super().__init__(env);self.scene=scene;self.traces=[]
    def reset(self,**kwargs):
        check_time()
        s,info=restore(self.env,self.scene) if self.scene is not None else self.env.reset(**kwargs)
        self.traces.append(dict(initial=snapshot(self.env,info,kwargs.get('seed')),steps=[]))
        return s,info
    def step(self,a):
        check_time();s,r,term,trunc,info=self.env.step(a);u=self.unwrapped
        self.traces[-1]['steps'].append(dict(action=np.asarray(a).copy(),qpos=u.data.qpos.copy(),qvel=u.data.qvel.copy(),
                    observation=s.copy(),reward=float(r),terminated=bool(term),truncated=bool(trunc),success=float(info['success']),
                    integration=integration(u),wall_distance=None))
        return s,r,term,trunc,info

def setup_stream(env,base):
    np.random.seed(base);random.seed(base)
    env.unwrapped.np_random=gym.utils.seeding.np_random(base)[0]
    env.action_space.seed(base+2000000)

def episode(agent,env,base,task,ep,condition,keep=True):
    reset_seed=base+task*10000+ep
    if condition=='U':
        env.action_space.seed(reset_seed+2000000);s,info=env.reset(seed=reset_seed,options={'task_id':task,'render_goal':False})
    else:s,info=env.reset(options={'task_id':task,'render_goal':False})
    initial=snapshot(env,info,reset_seed if condition=='U' else None);g=info['goal'];u=env.unwrapped
    positions=[s.copy()];velocities=[u.data.qvel.copy()];actions=[];flags=[]
    term=trunc=False
    while not (term or trunc):
        check_time();a=np.asarray(mode(agent,s,g));s,r,term,trunc,info=env.step(a)
        positions.append(s.copy());velocities.append(u.data.qvel.copy());actions.append(a);flags.append((float(r),bool(term),bool(trunc),float(info['success'])))
    xy=np.asarray(positions);v=np.asarray(velocities);a=np.asarray(actions)
    row=dict(task=task,episode=ep,stream=base,condition=condition,success=float(info['success']),steps=len(a),
             initial_x=float(xy[0,0]),initial_y=float(xy[0,1]),initial_vx=float(v[0,0]),initial_vy=float(v[0,1]),
             goal_x=float(g[0]),goal_y=float(g[1]),initial_distance=float(np.linalg.norm(xy[0]-g)),
             final_distance=float(np.linalg.norm(xy[-1]-g)),path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
             first_success_step=len(a) if info['success'] else None,terminated=bool(term),truncated=bool(trunc),
             full_scene_sha256=digest((initial['qpos'],initial['qvel'],initial['goal_xy'])),
             position_goal_sha256=digest((initial['qpos'],initial['goal_xy'])),env_rng_sha256=digest(initial['env_rng']),
             action_rng_sha256=digest(initial['action_rng']),numpy_rng_sha256=digest(initial['numpy_rng']))
    if keep:return row,dict(initial=initial,states=xy,qvel=v,actions=a,flags=flags,goal=np.asarray(g).copy())
    return len(a)
