import pathlib,json,hashlib,datetime,itertools,subprocess
import numpy as np
R=pathlib.Path(__file__).resolve().parent
PREV=pathlib.Path('/home/xushijie3/work/pc_scale02_20261011/experiments/pc_scale02_20261011')
O=pathlib.Path('/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011')
def dump(n,x):(R/'results'/n).write_text(json.dumps(x,indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8388608),b''):h.update(c)
 return h.hexdigest()
ident=json.loads((PREV/'results/identity.json').read_text())
for x in ident['files']:
 p=pathlib.Path(x['path']);actual=sha(p);assert actual==x['sha256'];x['actual_sha256']=actual
ident['verified_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
ident['official_head']=subprocess.check_output(['git','-C',str(O/'EC-Diffuser'),'rev-parse','HEAD'],text=True).strip()
assert ident['official_head']=='6b5e482fe479f647d4ca3790fac70c42f4940ab2'
ident['wrapper_start_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
dump('identity.json',ident)
# Each matched object has identical 8 cm transport and radial identity. Densities retain old bands.
rng=np.random.RandomState(202611123);eef=np.array([-.00199684687,0.]);scenes=[];checks=[]
old=json.loads((PREV/'results/scenes.json').read_text())
for group in range(20):
 v=.08*np.array([[1,0],[0,1],[-1,0],[0,-1]][group%4]);rot=rng.uniform(0,2*np.pi)
 for attempt in range(100000):
  rot=rng.uniform(0,2*np.pi);angles=np.arange(6)*np.pi/3+rot
  unit=np.stack([np.cos(angles),np.sin(angles)],-1)
  if attempt%2:
   unit=np.array([[-.5,-1],[-.5,0],[-.5,1],[.5,-1],[.5,0],[.5,1]]).dot(np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]).T)
  center=rng.uniform([-.11,-.16],[.11,.16])
  pair=[]
  for density,radius in [('sparse',.085),('crowded',.059)]:
   p=center+radius*unit;g=p+v
   retreat=np.stack([np.clip(g[:,0],-.22,eef[0]),np.zeros(6)],-1)
   if np.any(np.abs(p)>[.15,.20]) or np.any(np.abs(g)>[.15,.20]) or np.linalg.norm(p-eef,axis=-1).min()<.055 or np.linalg.norm(g-retreat,axis=-1).min()<.055:break
   if any(np.allclose(p,x['initial_xy'],atol=1e-5) for x in old if x['n']==6):break
   pair.append((density,p,g))
  if len(pair)==2:break
 else:raise RuntimeError('geometry not feasible group %d'%group)
 block=group%2;colors=[0,2,4,1,3,5] if block==0 else [1,3,5,0,2,4]
 for density,p,g in pair:
  scenes.append(dict(group=group,block=block,role='P1' if group<8 else ('validation' if group<12 else 'pilot'),condition='6_'+density,n=6,initial_xy=p.tolist(),goal_xy=g.tolist(),colors=colors,transport_vector=v.tolist()))
 delta={}
 for label,idx in [('initial',1),('goal',2)]:
  p,q=pair[0][idx],pair[1][idx]
  ep=np.minimum(.25-np.abs(p[:,0]),.30-np.abs(p[:,1]))-.0175;eq=np.minimum(.25-np.abs(q[:,0]),.30-np.abs(q[:,1]))-.0175
  rp=np.linalg.norm(p-[-.45,0],axis=-1);rq=np.linalg.norm(q-[-.45,0],axis=-1)
  delta[label+'_edge_per_object']=(ep-eq).tolist();delta[label+'_reach_proxy_per_object']=(rp-rq).tolist()
  assert max(abs(ep-eq))<=.030 and max(abs(rp-rq))<=.030
 checks.append(dict(group=group,delta=delta,attempts=attempt+1))
dump('scenes_all.json',scenes);dump('geometry_gate.json',dict(passed=True,seed=202611123,per_object_tolerance_m=.030,bands={'sparse':[.075,.100],'crowded':[.052,.068]},transport_m=.08,blocks=2,checks=checks,old_scenes_sha=sha(PREV/'results/scenes.json'),excluded_old_groups=list(range(6)),reach='geometric proxy only',shape='six identity aligned points on ring or 2 by 3 grid, matched center and orientation, independent new draws'))
# Fixed diagnostic reruns: four stalls plus the only new fall, no rerun selection after outcomes.
selected=[]
for cond,groups in [('6_crowded',[0,1,2]),('6_sparse',[0,1])]:
 for x in old:
  if x['condition']==cond and x['group'] in groups:
   y=dict(x);y['role']='diagnostic';selected.append(y)
dump('diagnostic_scenes.json',selected)
print('CPU_SETUP_OK',len(scenes),len(selected),flush=True)
