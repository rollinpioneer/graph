from core import *
import subprocess
assert json.loads((R/'results/freeze.json').read_text())['status']=='PASS'
children=[]
for kind,args in [('replay',[])]+[(f'quality_{s:04d}',[str(s)]) for s in SCENES]:
 script='replay.py' if kind=='replay' else 'quality.py';env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0' if kind=='replay' else '')
 log=open(R/f'logs/{kind}.log','w');p=subprocess.Popen([sys.executable,str(R/'code'/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(dict(kind=kind,pid=p.pid))
write(R/'results/launch.json',dict(start=time.time(),children=children))
print(children)
