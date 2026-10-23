import torch,json
from pathlib import Path
p=Path.home()/'work/grasp/eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar'
z=torch.load(p,map_location='cpu',weights_only=True)
out={'keys':list(z)}
for k,v in z.items():
 if k in ['epoch','iteration','step']:out[k]=v
 if 'optimizer' in k:
  out[k]={'keys':list(v),'state_entries':len(v.get('state',{})),'groups':[{kk:vv for kk,vv in g.items() if kk!='params'} for g in v.get('param_groups',[])]}
print(json.dumps(out,default=str))
