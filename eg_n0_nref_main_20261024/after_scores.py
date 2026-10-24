"""Wait on completed frame receipts, then evaluate each frame once."""
import sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import pipeline as P
R=P.R
def task(t):
 _,s,f=t;start=time.time()
 while not (R/f'cpu/C3_{s:04d}_{f:04d}.json').exists() or (sys.argv[1]=='joint' and not (R/f'nv/C3_{s:04d}_{f:04d}.json').exists()):
  if time.time()-start>7200:raise TimeoutError(t)
  time.sleep(5)
 if sys.argv[1]=='dense':
  from dense_audit import run
 else:
  from joint import run
 return run(t)
if __name__=='__main__':
 assert (R/'frozen_config.json').exists();(R/sys.argv[1]).mkdir(exist_ok=True)
 with ProcessPoolExecutor(int(sys.argv[3])) as pool:list(pool.map(task,P.tasks('C3')[int(sys.argv[2])::3]))
