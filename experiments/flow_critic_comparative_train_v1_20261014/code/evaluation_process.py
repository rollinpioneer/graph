"""Spawn workers with the same NumPy/Python state inherited by native fork workers."""
import numpy as np,random
def spawn_factory(env_fn):
 holder={'numpy':np.random.get_state(),'python':random.getstate()}
 def make():
  np.random.set_state(holder['numpy']);random.setstate(holder['python'])
  env=env_fn()
  # Gym creates its dummy environment in the parent before serializing workers.
  holder['numpy']=np.random.get_state();holder['python']=random.getstate()
  return env
 return make,holder
