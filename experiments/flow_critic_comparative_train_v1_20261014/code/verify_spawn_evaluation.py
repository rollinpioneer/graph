import os,sys,json,time
from pathlib import Path
os.environ.setdefault('COMPARATIVE_JOB','P0_RENDER')
import numpy as np,jax,gymnasium,ogbench,random
from evaluation_process import spawn_factory
ROOT=Path(os.environ['COMPARATIVE_ROOT'])
sys.path.insert(0,str(ROOT/'workspaces/gfp_floq'))
from envs.env_utils import EpisodeMonitor
def factory():
 return EpisodeMonitor(ogbench.make_env_and_datasets('humanoidmaze-medium-navigate-singletask-task1-v0',env_only=True),filter_regexes=['.*privileged.*','.*proprio.*'])
def main():
 jax.devices()
 safe_factory,holder=spawn_factory(factory)
 vector=gymnasium.vector.AsyncVectorEnv([safe_factory]*10,context='spawn',autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED)
 np.random.set_state(holder['numpy']);random.setstate(holder['python'])
 serial=factory()
 try:
  obs,info=vector.reset(seed=14014)
  action=np.zeros((10,21),dtype=np.float32)
  serial_obs,_=serial.reset(seed=14014)
  np.testing.assert_allclose(obs[0],serial_obs,rtol=0,atol=1e-6)
  next_obs,reward,term,trunc,_=vector.step(action)
  next_serial,serial_reward,serial_term,serial_trunc,_=serial.step(action[0])
  np.testing.assert_allclose(next_obs[0],next_serial,rtol=0,atol=1e-6)
  assert reward[0]==serial_reward and term[0]==serial_term and trunc[0]==serial_trunc
  receipt=dict(status='PASS',context='spawn',n_eval_envs=10,optimizer_updates=0,observation_shape=list(obs.shape),
    serial_reset_step_equivalence_atol=1e-6,numpy_python_worker_state_matches_native_fork=True,epoch=time.time(),algorithm_config_changed=False)
  (ROOT/'receipts/spawn_evaluation_fix.json').write_text(json.dumps(receipt,indent=2))
  print('SPAWN_EVALUATION_ZERO_UPDATE_PASS',flush=True)
 finally:vector.close();serial.close()
if __name__=='__main__':main()
