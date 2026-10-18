from core import *
import torch,subprocess,concurrent.futures
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from dataset.graspnet_dataset import collate_fn
from models.economicgrasp import economicgrasp,pred_decode
start=time.time();ck=OLD/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar';assert sha(ck)==CFG['checkpoint_sha256'];torch.set_num_threads(1);net=economicgrasp(seed_feat_dim=512,is_training=False).cuda().eval();net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model_state_dict']);net.requires_grad_(False);cap={}
def group(m,i,o):cap['g']=o.detach().clone()
def token(m,i):cap['t']=i[0].detach().clone()
h=[net.cy_group.register_forward_hook(group),net.grasp_head.conv_score.register_forward_pre_hook(token)]
z=np.load(N/'inputs/scene_0000/0000.npz');batch={k:v.cuda() for k,v in collate_fn([{k:z[k] for k in z.files}]).items()}
with torch.no_grad():ep=net(dict(batch));pred=pred_decode(ep)[0].cpu().numpy();feat=torch.cat([cap['g'],cap['t']],1)[0].T.cpu().numpy()
for x in h:x.remove()
with torch.no_grad():plain=pred_decode(net(dict(batch)))[0].cpu().numpy()
saved=export(0,0);assert np.array_equal(plain,pred) and np.array_equal(saved[0],pred) and np.array_equal(saved[2],feat)
assert np.isfinite(saved[3]).all();assert len(set(sample_ids(pred,0,0)))==128
assert np.array_equal(utility(np.array([-1,0,.2,.4,1.2])),np.array([0,0,1,5/6,1/6],np.float32))
write(R/'results/smoke_hook.json',dict(status='PASS',hook_prediction_exact=True,cached_prediction_exact=True,group_then_token_feature_exact=True,model_frozen=True,utility_encoding_pass=True,checkpoint_sha256=sha(ck),seconds=time.time()-start,gpu_peak=torch.cuda.max_memory_allocated()))
def run(s):
 with open(R/f'logs/micro_{s}.log','w') as f:return subprocess.run([sys.executable,str(R/'code/labels.py'),'--scene',str(s),'--micro'],stdout=f,stderr=subprocess.STDOUT).returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(run,CFG['fit'][:4]))
assert not any(codes),codes
receipts=[json.loads((R/f'results/labels_{s:04d}_micro.json').read_text()) for s in CFG['fit'][:4]];assert sum(x['new_calls'] for x in receipts)==768
write(R/'results/S0.json',dict(status='PASS',label_calls=768,quality_seconds=sum(x['seconds'] for x in receipts),projected_label_worker_hours=sum(x['seconds'] for x in receipts)/768*196608/3600,hook='PASS',no_training=True))
print((R/'results/S0.json').read_text())
