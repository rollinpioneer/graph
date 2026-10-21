import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
B=Path.home()/'work/grasp';P=B/'eg_sub25_b_f05_repl_v1_20261019'
R=B/'eg_f1_precision_diagnosis_v1_20261021';R.mkdir(exist_ok=True)
ck=Path(json.loads((P/'results/backbone_status.json').read_text())['checkpoint'])
assert hashlib.sha256(ck.read_bytes()).hexdigest()=='533d065386e96a2f76ad2317f0376321a3d2c350a88027d8310e9fd7cd99bc98'
d=torch.load(ck,map_location='cpu',weights_only=False)['model_state_dict'];w=d['grasp_head.conv_score.weight'];b=d['grasp_head.conv_score.bias']
out={'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(),'default_tf32_cudnn':torch.backends.cudnn.allow_tf32,'default_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'samples':[]}
for scene,frame in [(100,0),(106,128),(186,255),(57,0),(15,64)]:
 p=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
 with np.load(p) as z:x=torch.from_numpy(z['feature320'][:,256:].T.copy())[None];ref=z['pred'][:,0].astype(float);identity=str(z['backbone_sha256'])
 row={'scene':scene,'frame':frame,'backbone_sha256':identity,'variants':{}}
 for device in ['cpu','cuda']:
  for dtype in [torch.float64,torch.float32]:
   for tf32 in ([False,True] if device=='cuda' and dtype==torch.float32 else [False]):
    torch.backends.cudnn.allow_tf32=tf32;torch.backends.cuda.matmul.allow_tf32=tf32
    xx=x.to(device=device,dtype=dtype);ww=w.to(device=device,dtype=dtype);bb=b.to(device=device,dtype=dtype)
    c=torch.tensor([0,.2,.4,.6,.8,1],dtype=dtype,device=device)
    with torch.no_grad():
     yy=F.conv1d(xx,ww,bb);s=(yy[0]*c[:,None]).sum(0).cpu().numpy().astype(float)
    order=np.argsort(-s,kind='stable');delta=np.diff(ref[order])
    row['variants'][f'{device}_{dtype}_tf32_{tf32}']={'max_error':float(abs(s-ref).max()),'rmse':float(np.sqrt(np.mean((s-ref)**2))),'exact':int((s==ref).sum()),'nonnear_inversions':int((delta>=1e-5).sum()),'near_inversions':int(((delta>0)&(delta<1e-5)).sum())}
 out['samples'].append(row)
(R/'probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
