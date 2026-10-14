import os,sys,json,random,hashlib,time
from pathlib import Path
import numpy as np,ogbench
r=Path(os.environ['COMPARATIVE_ROOT']);task=sys.argv[1]
a=ogbench.make_env_and_datasets(task,env_only=True);b=ogbench.make_env_and_datasets(task,env_only=True)
records=[]
for seed in [20261014,20261015]:
 np.random.seed(seed);random.seed(seed);oa,_=a.reset(seed=seed)
 na,ra,ta,xa,_=a.step(np.zeros(a.action_space.shape))
 np.random.seed(seed);random.seed(seed);ob,_=b.reset(seed=seed)
 nb,rb,tb,xb,_=b.step(np.zeros(b.action_space.shape))
 np.testing.assert_array_equal(oa,ob);np.testing.assert_array_equal(na,nb)
 assert (ra,ta,xa)==(rb,tb,xb)
 records.append(dict(seed=seed,initial_observation_sha256=hashlib.sha256(oa.tobytes()).hexdigest()))
a.close();b.close()
(r/'receipts'/('matched_reset_'+task+'.json')).write_text(json.dumps(dict(status='PASS',optimizer_updates=0,initial_observations_exact=True,records=records,epoch=time.time()),indent=2))
print('MATCHED_RESET_IDENTITY_PASS',task,flush=True)
