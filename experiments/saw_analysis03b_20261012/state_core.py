import copy, csv, json, pickle, random, time
from pathlib import Path
from common import *
import gymnasium as gym
import mujoco
STATE=mujoco.mjtState.mjSTATE_INTEGRATION
EV=Path('/home/xushijie3/work/graph_saw_eval02_20261012/experiments/saw_eval02_20261012')
def sha(p):return __import__("common").sha(p)
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

def env_new(kind):
    env=ogbench.make_env_and_datasets(f'pointmaze-large-{kind}-v0',env_only=True)
    assert env.spec.max_episode_steps==1000 and env.unwrapped._goal_tol==1.0
    assert env.unwrapped._teleport_info is None
    return env
mode=jax.jit(lambda a,s,g:np_clip(a.network.select('actor')(s,g,temperature=0.).mode()))
def np_clip(x):return jax.numpy.clip(x,-1,1)

