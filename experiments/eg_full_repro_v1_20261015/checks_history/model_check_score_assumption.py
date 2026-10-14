import sys,time,json
from pathlib import Path
import numpy as np
import torch
r=Path(__file__).parent
sys.argv=[sys.argv[0],'--model','economicgrasp','--camera','kinect','--dataset_root',str(r/'data/graspnet')]
from models.economicgrasp import economicgrasp,pred_decode
started=time.perf_counter();torch.manual_seed(20261015)
checkpoint=torch.load(r/'downloads/economicgrasp_kinect.tar',map_location='cpu',weights_only=True)
net=economicgrasp(seed_feat_dim=512,is_training=False).cuda();net.load_state_dict(checkpoint['model_state_dict'],strict=True);net.eval()
# Geometric synthetic cloud: engineering check only; no GraspNet AP or training.
xyz=torch.rand(1,20000,3,device='cuda')
xyz[:,:,:2]=(xyz[:,:,:2]-.5)*.32
xyz[:,:,2]=.65+(xyz[:,:,2]-.5)*.03
xyz[:,10000:15000,:2]*=.4;xyz[:,10000:15000,2]-=.05
xyz[:,15000:,:2]*=.5;xyz[:,15000:,0]+=.09;xyz[:,15000:,2]-=.08
with torch.no_grad():
 result=net({'point_clouds':xyz,'coordinates_for_voxel':xyz/.005});preds=pred_decode(result)[0]
torch.cuda.synchronize();print(json.dumps({'shape':list(preds.shape),'finite':bool(torch.isfinite(preds).all()),'score_min':float(preds[:,0].min()),'score_max':float(preds[:,0].max()),'width_min':float(preds[:,1].min()),'width_max':float(preds[:,1].max())}),flush=True);assert preds.shape==(1024,17);assert torch.isfinite(preds).all();assert torch.all((preds[:,0]>=0)&(preds[:,0]<=1));assert torch.all((preds[:,1]>=0)&(preds[:,1]<=.1))
report={'status':'PASS','official_pretrained_checkpoint':True,'checkpoint_epoch':checkpoint.get('epoch'),'synthetic_geometry_only':True,'real_data_smoke_passed':False,'training_updates':0,'predictions':list(preds.shape),'graspable_points':float(result['D: Graspable Points']),'wall_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name()}
(r/'receipts/model_environment_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
