import sys,json,hashlib,time,io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from score_replay import FrozenScoreReplay,reconstruction_metrics
torch.set_num_threads(1)
B=Path.home()/'work/grasp';P=B/'eg_sub25_b_f05_repl_v1_20261019';L=B/'eg_lg1_local_geometry_v1_20261020';P0=B/'eg_role_p0_audit_v1_20261021';R=B/'eg_f1_precision_diagnosis_v1_20261021'
sys.path.insert(0,str(B/'eg_lqr_build_confirm_v3_20261019/code'))
from scoring import fused_score
start=time.time();manifest={x['path']:x['sha256'] for x in json.loads((P0/'P0_input_manifest.json').read_text())}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ck=Path(json.loads((P/'results/backbone_status.json').read_text())['checkpoint']);backbone=sha(ck)
assert backbone=='533d065386e96a2f76ad2317f0376321a3d2c350a88027d8310e9fd7cd99bc98'
state=torch.load(ck,map_location='cpu',weights_only=False)['model_state_dict'];replay=FrozenScoreReplay(state)
w=state['grasp_head.conv_score.weight'].numpy().astype(float)[:,:,0];b=state['grasp_head.conv_score.bias'].numpy().astype(float);c=np.array([0,.2,.4,.6,.8,1]);wc=c@w;bc=c@b
split=json.loads((P/'B_head_split.json').read_text());C3=[100,106,115,124,132,139,150,157,160,169,176,186]
head=[(s,f) for k in ['fit','model_val','gate_cal'] for s in split[k] for f in split['frame_ids']]
pairs=[(s,f) for s in C3 for f in range(256)]+head
prepared_path=B/'eg_b_global_rank_pilot_v1_20261020/prepared.npz'
assert sha(prepared_path)=='8567a5babf2906c1b87ab67c695e86b0da5cc437ec919bae3d918d9e901f52f5'
prep=dict(np.load(prepared_path));index={(int(s),int(f)):i for i,(s,f) in enumerate(zip(prep['scene'],prep['frame']))}
def load(pair):
 s,f=pair;p=P/f'B/exports/subset/scene_{s:04d}/{f:04d}.npz';raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==manifest[str(p)]
 with np.load(io.BytesIO(raw)) as z:
  feat=z['feature320'];pred=z['pred'];assert str(z['backbone_sha256'])==backbone
  assert np.array_equal(z['candidate_ids'],np.arange(1024))
  assert hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()==str(z['geometry_sha256'])
  if pair in index:
   i=index[pair];assert np.array_equal(feat[prep['ids'][i]],prep['feature'][i]);assert np.array_equal(z['hypotheses'][:,0],pred)
 return s,f,feat,pred[:,0],h
rows=[];i1=[]
with ThreadPoolExecutor(12) as pool:
 for offset in range(0,len(pairs),48):
  for s,f,feat,ref,h in pool.map(load,pairs[offset:offset+48]):
   exact=replay(feat);affine=feat[:,256:].astype(float)@wc+bc
   row=dict(scene=s,frame=f,export_sha256=h,replay=reconstruction_metrics(exact,ref),collapsed_affine=reconstruction_metrics(affine,ref));rows.append(row)
   if s in C3 and f==0:
    ep=L/f'evaluation/development/{s:04d}/{f:04d}.npz';assert sha(ep)==manifest[str(ep)]
    cp=P0/f'union/{s:04d}/{f:04d}.npz'
    with np.load(ep) as z,np.load(cp) as masks:
     cd=masks['cd_mask'];ids=np.flatnonzero(~cd)
     for seed in [20261018,20261019,20261020]:
      q=z[f'Q_G_{seed}'];a,info=fused_score(ref,q);v,_=fused_score(affine,q);r,_=fused_score(exact,q)
      i1.append(dict(scene=s,seed=seed,pure_affine_F05_max_error=float(abs(v-a).max()),replay_F05_max_error=float(abs(r-a).max()),baseline_top=int(ids[np.argmax(a[ids])]),pure_affine_top=int(ids[np.argmax(v[ids])]),replay_top=int(ids[np.argmax(r[ids])]),fallback=info['fallback']))
  print(json.dumps({'frames_done':len(rows),'total':len(pairs),'seconds':time.time()-start}),flush=True)
def aggregate(key):
 return dict(max_error=max(v[key]['max_error'] for v in rows),exact_candidates=sum(v[key]['exact_candidates'] for v in rows),nonnear_inversions=sum(v[key]['nonnear_inversions'] for v in rows),near_inversions=sum(v[key]['near_inversions'] for v in rows),order_identical_frames=sum(v[key]['order_identical'] for v in rows))
report=dict(frames=len(rows),candidates=len(rows)*1024,head_frames=len(head),C3_frames=3072,backbone_sha256=backbone,prepared_sha256=sha(prepared_path),all_export_hashes_match_P0=True,head_feature_identities_match=True,replay=aggregate('replay'),collapsed_affine=aggregate('collapsed_affine'),I1=i1,I1_pure_affine_pass=all(x['pure_affine_F05_max_error']<=1e-9 and x['pure_affine_top']==x['baseline_top'] for x in i1),I1_replay_pass=all(x['replay_F05_max_error']<=1e-9 and x['replay_top']==x['baseline_top'] for x in i1),P2_status='BLOCKED_PURE_AFFINE_INITIALIZATION',training_steps=0,new_labels=0,C4_read=False,EG_backbone_forwards=0,seconds=time.time()-start,torch=torch.__version__,cudnn=torch.backends.cudnn.version(),GPU=torch.cuda.get_device_name(),readout_weight_sha256=hashlib.sha256(w.tobytes()).hexdigest())
(R/'full_frames.json').write_text(json.dumps(rows));(R/'diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
