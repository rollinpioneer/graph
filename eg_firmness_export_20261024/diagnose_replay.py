"""D1/D2: one cached frame, two frozen A100 forwards, no evaluation."""
import sys,json,time,random
from pathlib import Path
import numpy as np
import torch
import infer_features as I
ROOT=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 ck=I.B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar'
 assert I.sha(ck)=='a47b717ad41870e9b34e85b1487de2a3f0f134ddc1a71a98a1b55bca7fc5804a'
 net=I.economicgrasp(seed_feat_dim=512,is_training=False).cuda()
 state=torch.load(ck,map_location='cpu',weights_only=True);net.load_state_dict(state['model_state_dict'],strict=True);net.eval()
 assert not net.is_training and not net.view.is_training and not net.training
 cp=I.N0/'cache/train_0067_0000.npz';ep=I.N0/'per_pose/train_0067_0000.npz'
 a=np.load(cp);e=np.load(ep);ids=e['candidate_id'];saved={}
 def vp_hook(module,args):
  assert tuple(args[0].shape)==(1,256,1024)
  saved['vp']=I.cpu(args[0])[0].T.copy()
 def ang_hook(module,args,result):
  assert tuple(result.shape)==(1024,4,64)
  saved['ang']=I.cpu(result)[:,0,:].copy()
 handles=[net.grasp_head.register_forward_pre_hook(vp_hook),net.grasp_head.global_interaction_module.register_forward_hook(ang_hook)]
 outputs=[];times=[]
 for run in range(2):
  seed=20261024+67*1000;np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  points=a['input_points'];sample=dict(point_clouds=points,coordinates_for_voxel=points/.005,seg=a['input_seg'],cloud_colors=a['input_colors'])
  data={k:v.cuda() for k,v in I.collate_fn([sample]).items()};saved.clear()
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.no_grad():end=net(data);pred=I.cpu(I.pred_decode(end)[0])
  torch.cuda.synchronize();times.append(time.perf_counter()-start)
  outputs.append(dict(pred=pred.copy(),view_index=I.cpu(end['grasp_top_view_inds'])[0].copy(),angle_index=I.cpu(end['grasp_angle_pred'])[0,:12].argmax(0),depth_index=I.cpu(end['grasp_depth_pred'])[0,:4].argmax(0)))
  del data,end;torch.cuda.empty_cache()
 for h in handles:h.remove()
 first,second=outputs;diff=np.abs(first['pred'].astype(float)-a['pred'].astype(float));bad=np.any(diff>1e-5,axis=1);flip=first['view_index']!=a['view_index'];depth=first['depth_index']!=a['depth_index'];same=~flip
 d1=dict(pred_max_absolute_error=float(diff.max()),per_column_max_absolute_error=diff.max(0).tolist(),seeds_over_1e5=int(bad.sum()),candidate_seeds_over_1e5=int(bad[ids].sum()),view_flips_all=int(flip.sum()),view_flips_candidates=int(flip[ids].sum()),view_flip_seed_ids=np.flatnonzero(flip).tolist(),view_flip_candidate_ids=ids[flip[ids]].tolist(),depth_index_mismatches_all=int(depth.sum()),depth_index_mismatches_candidates=int(depth[ids].sum()),angle_index_mismatches_all=int((first['angle_index']!=a['angle_index']).sum()),center_max_absolute_error=float(diff[:,13:16].max()),same_view_pred_max_absolute_error=float(diff[same].max()) if same.any() else None,same_view_seeds_over_1e5=int(bad[same].sum()),seed_count=1024,candidate_count=len(ids))
 d2=dict(pred_max_absolute_error=float(np.max(np.abs(first['pred'].astype(float)-second['pred'].astype(float)))),pred_exact_equal=bool(np.array_equal(first['pred'],second['pred'])),view_index_exact_equal=bool(np.array_equal(first['view_index'],second['view_index'])))
 meta=json.loads(cp.with_suffix('.json').read_text())
 report=dict(frame='cal_0067_0000',D1=d1,D2=d2,current_gpu=torch.cuda.get_device_name(),cached_gpu=meta.get('gpu'),forward_seconds=times,tf32_matmul=True,tf32_cudnn=True,training=False,official_evaluation=False,input_sha256=I.sha(cp),per_pose_sha256=I.sha(ep),cached_receipt_sha256=I.sha(cp.with_suffix('.json')),checkpoint_sha256=I.sha(ck),infer_features_sha256=I.sha(Path(I.__file__)))
 out=ROOT/'diagnosis_cal_0067_0000.json';assert not out.exists();I.save_json(out,report)
 dest=ROOT/'diagnosis_cal_0067_0000_arrays.npz';assert not dest.exists()
 np.savez_compressed(dest,candidate_id=ids,cached_pred=a['pred'],cached_view_index=a['view_index'],first_pred=first['pred'],second_pred=second['pred'],first_view_index=first['view_index'],second_view_index=second['view_index'],first_angle_index=first['angle_index'],first_depth_index=first['depth_index'])
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
