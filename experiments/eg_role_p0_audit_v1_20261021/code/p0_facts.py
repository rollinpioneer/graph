import os,sys,json,time,hashlib,argparse,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
torch.set_num_threads(1)
B=Path.home()/'work/grasp';R=B/'eg_role_p0_audit_v1_20261021';P=B/'eg_sub25_b_f05_repl_v1_20261019';L=B/'eg_lg1_local_geometry_v1_20261020';RP=B/'eg_b_global_rank_pilot_v1_20261020'
sys.path.insert(0,str(B/'eg_lqr_build_confirm_v3_20261019/code'))
from scoring import fused_score
SEEDS=[20261018,20261019,20261020]
C3=[100,106,115,124,132,139,150,157,160,169,176,186]
ROOTS=[P/'B/evaluation/confirmation',L/'evaluation/development',B/'eg_geometry_compact_v1_20261020/evaluation/development',B/'eg_stat_shape_h_v1_20261020/evaluation/development',B/'eg_g_conditioned_v1_20261021/evaluation/development']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4*1024**2),b''):h.update(v)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
EXPECTED={};READOUT=None
def init():
 global READOUT
 for file in [L/'results/evaluation_artifact_manifest.json',P/'results/B_artifact_manifest.json',B/'eg_g_conditioned_v1_20261021/results/evaluation_artifact_manifest.json']:
  for x in json.loads(file.read_text()):EXPECTED[x['path']]=x['sha256']
 ck=Path(json.loads((P/'results/backbone_status.json').read_text())['checkpoint'])
 assert sha(ck)=='533d065386e96a2f76ad2317f0376321a3d2c350a88027d8310e9fd7cd99bc98'
 d=torch.load(ck,map_location='cpu',weights_only=False);d=d.get('model_state_dict',d)
 w=d['grasp_head.conv_score.weight'].numpy().astype(float)[:,:,0];b=d['grasp_head.conv_score.bias'].numpy().astype(float);c=np.array([0,.2,.4,.6,.8,1.]);READOUT=(c@w,c@b)
def tracked(p,manifest):
 h=sha(p);expected=EXPECTED.get(str(p));assert expected is None or h==expected,('historical_hash',str(p),h,expected)
 manifest[str(p)]=dict(path=str(p),sha256=h,bytes=p.stat().st_size,historical_hash_checked=expected is not None)
 return h
def f1(z):
 s=z['pred'][:,0].astype(float);sh=z['feature320'][:,256:].astype(float)@READOUT[0]+READOUT[1];order=np.argsort(-sh,kind='stable');diff=np.diff(s[order]);bad=diff>0
 return dict(max_error=float(np.max(np.abs(sh-s))),near_tie_inversions=int(((diff>0)&(diff<1e-5)).sum()),nonnear_inversions=int((diff>=1e-5).sum()),order_identical=bool(np.array_equal(np.argsort(-s,kind='stable'),order)))
def one(scene,frame,manifest):
 ep=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz';eh=tracked(ep,manifest)
 with np.load(ep) as z:
  pred=z['pred'];f=f1(z);gh=hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest();assert gh==str(z['geometry_sha256']);assert str(z['backbone_sha256'])=='533d065386e96a2f76ad2317f0376321a3d2c350a88027d8310e9fd7cd99bc98'
 mu=np.full(1024,np.nan);collision=np.zeros(1024,bool);empty=collision.copy();cd=None;old=None;rows=None;overlap=0
 for root in ROOTS:
  jp=root/f'{scene:04d}/{frame:04d}.json';zp=jp.with_suffix('.npz');tracked(jp,manifest);tracked(zp,manifest);rr=json.loads(jp.read_text())
  assert all(v['export_sha256']==eh and v['geometry_sha256']==gh for v in rr),('export_identity',str(jp))
  with np.load(zp) as z:
   quality=z['quality'];known=np.isfinite(quality);both=known&np.isfinite(mu)
   for name,v in [('quality',mu),('collision',collision),('empty',empty)]:assert np.array_equal(z[name][both],v[both]),('label_conflict',str(zp),name,np.flatnonzero(z[name][both]!=v[both]).tolist())
   overlap+=int(both.sum());mu[known]=quality[known];collision[known]=z['collision'][known];empty[known]=z['empty'][known]
   if 'cd_mask' in z:
    if cd is not None:assert np.array_equal(cd,z['cd_mask'])
    cd=z['cd_mask'].copy()
   if root==L/'evaluation/development':old={k:z[k].copy() for k in z.files if k.startswith(('Q_','selected_'))};rows={v['configuration']:v for v in rr}
 assert cd is not None
 checks=[]
 for name in ['EG']+[f'{g}_{s}' for g in ['O','G'] for s in SEEDS]:
  r=rows[name+'_CD'];assert float(cd.mean())==r['cd_rejection_rate']
  score,info=(pred[:,0].astype(float),{'fallback':False}) if name=='EG' else fused_score(pred[:,0],old['Q_'+name])
  ids=np.flatnonzero(~cd);top=int(ids[np.argmax(score[ids])]) if len(ids) else None;ref=r['top_candidate']
  tie=bool(top is not None and ref is not None and abs(score[top]-score[ref])<1e-12)
  checks.append(dict(configuration=name,top=top,reference=ref,match=top==ref,tie=tie,empty=not len(ids),fallback=info['fallback']))
 np.savez_compressed(R/f'union/{scene:04d}/{frame:04d}.npz',quality=mu,collision=collision,empty=empty,cd_mask=cd)
 return dict(scene=scene,frame=frame,F1=f,F2=checks,F4_overlap=overlap,known=int(np.isfinite(mu).sum()))
def scene_job(scene,limit):
 start=time.time();m={};rows=[];(R/f'union/{scene:04d}').mkdir(parents=True,exist_ok=True)
 try:
  complete=P/f'B/exports/subset/scene_{scene:04d}/COMPLETE.json';tracked(complete,m)
  for x in json.loads(complete.read_text())['source_identities']:EXPECTED[str(complete.parent/f"{x['frame']:04d}.npz")]=x['export_sha256']
  for f in range(limit):rows.append(one(scene,f,m))
  result=dict(status='PASS',scene=scene,frames=rows,manifest=list(m.values()),seconds=time.time()-start,cpu_seconds=time.process_time(),host=os.uname().nodename)
 except Exception as e:result=dict(status='IDENTITY_FAIL',scene=scene,error=str(e),traceback=traceback.format_exc(),frames_done=len(rows),manifest=list(m.values()))
 save(R/f'results/facts_{scene:04d}_{limit}.json',result);return {k:v for k,v in result.items() if k not in ['frames','manifest']}
def head_facts(limit):
 m={};d=dict(np.load(RP/'prepared.npz'));norm=L/'prepared_geometry.npz';prep=json.loads((L/'results/preparation.json').read_text())
 assert tracked(RP/'prepared.npz',m)==prep['prepared_source_sha256'];assert tracked(norm,m)==prep['geometry_cache_sha256']
 assert tracked(P/'B_frozen_split.json',m)=='a369312daafaca07a5b8dbb8bc747a5f5b8649a05590c2664ec8bb1625980bc4'
 assert tracked(P/'B_head_split.json',m)=='f272787cbf95cfd9f5ebbdaacdee2ba010712902454236a0968d218492be67ab'
 lock=json.loads((L/'method_lock.json').read_text());tracked(L/'method_lock.json',m)
 for h in lock['new_heads']:
  if h['family']=='G':assert tracked(Path(h['path']),m)==h['sha256']
 source=json.loads((RP/'results/source_manifest.json').read_text());tracked(RP/'results/source_manifest.json',m)
 for v in source['labels']:EXPECTED[v['export']]=v['export_sha256'];EXPECTED[v['label']]=v['label_sha256']
 rows=[]
 for i in range(min(limit,len(d['scene']))):
  s,f=int(d['scene'][i]),int(d['frame'][i]);ep=P/f'B/exports/subset/scene_{s:04d}/{f:04d}.npz';tracked(ep,m);lp=P/f'B/labels/{s:04d}_{f:04d}.npz';tracked(lp,m)
  with np.load(ep) as z:
   assert np.array_equal(z['candidate_ids'],np.arange(1024));assert np.array_equal(z['hypotheses'][:,0],z['pred']);assert np.array_equal(z['feature320'][d['ids'][i]],d['feature'][i]);assert np.array_equal(z['actions'][d['ids'][i]],d['action'][i]);rows.append(dict(scene=s,frame=f,F1=f1(z)))
 save(R/f'results/head_facts_{limit}.json',dict(status='PASS',frames=rows,manifest=list(m.values())))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenes',default='100');p.add_argument('--frames',type=int,default=1);p.add_argument('--workers',type=int,default=1);p.add_argument('--head',type=int,default=0);a=p.parse_args();init()
 if a.head:head_facts(a.head)
 with ProcessPoolExecutor(a.workers) as pool:
  for f in as_completed([pool.submit(scene_job,int(s),a.frames) for s in a.scenes.split(',') if s]):print(json.dumps(f.result()),flush=True)
