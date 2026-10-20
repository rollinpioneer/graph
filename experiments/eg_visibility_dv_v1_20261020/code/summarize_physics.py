from common import *
from sklearn.metrics import roc_auc_score
import pandas as pd

def ratio(mask):return float(np.mean(mask)) if len(mask) else None
def med(x):
 x=np.asarray(x);x=x[np.isfinite(x)];return float(np.median(x)) if len(x) else None
def auct(y,p):
 return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
def main():
 blocks=[];truth=[];scope=[]
 for p in sorted((R/'cache/head').glob('*/*.npz')):
  z=np.load(p);blocks.append(z['dv0']);truth.append(z['dv0_mu']);scope.extend([[int(p.parent.name),int(p.stem)]]*len(z['dv0']))
 c=np.concatenate(blocks);mu=np.concatenate(truth)
 found=c[:,0]==1;correct=np.isclose(c[:,1],mu,atol=1e-6);official=np.isclose(c[:,1],c[:,2],atol=1e-6)
 state=c[:,5:7];vis=(state==1);grazing=c[:,9:11]>=.3;reliable=vis&grazing&np.isfinite(c[:,15:17])
 strata={}
 for name,mask in [('all',np.ones(len(mu),bool)),('valid',(mu>0)&(mu<=.400001)),('weak',mu>=.59999),('no_closure',mu<0)]:
  m=mask&found;v=vis[m]
  strata[name]=dict(n=int(mask.sum()),contacts_found=int(m.sum()),both_visible=ratio(v.all(1)),one_visible=ratio(v.sum(1)==1),neither_visible=ratio(~v.any(1)),at_least_one_over_all=float((vis.any(1)&m).sum()/max(1,mask.sum())))
 normal_median=med(c[:,15:17][reliable]);normal_coverage=float(reliable.sum()/max(1,(vis&grazing).sum()))
 pass_a=ratio(correct)>=.95
 pass_b=strata['all']['at_least_one_over_all']>=.3
 pass_c=normal_median is not None and normal_median<=10
 # Missing reliable probe correspondences do not disappear from coverage.
 physics=dict(status='COMPLETE' if len(mu)==20000 else 'PARTIAL',n=len(mu),contacts_found=int(found.sum()),label_agreement=ratio(correct),agreement_found=ratio(correct[found]),formula_official_agreement=ratio(official[found]),strata=strata,
  normal_error_deg_median=normal_median,normal_pair_count=int(reliable.sum()),normal_coverage=normal_coverage,
  normal_at_GT_pixel_error_median=med(c[:,11:13][vis&grazing]),position_error_m_median=med(c[:,13:15][vis]),
  sensitivity={str(e):{str(i):float((c[:,3+2*j:5+2*j][found]==i).mean()) for i in range(4)} for j,e in enumerate([3,4,6])},
  pass_a=bool(pass_a),pass_b=bool(pass_b),pass_c=bool(pass_c),K_mode='full' if pass_a and pass_b and pass_c else 'state_position_only')
 write(R/'results/DV0.json',physics)
 if not (R/'results/dispatch_status.json').exists() or json.load(open(R/'results/dispatch_status.json'))['status']!='COMPLETE':return
 rows=[];frameauc=[];patterns=[]
 for p in sorted((R/'cache/c3').glob('*/*.npz')):
  z=np.load(p);scene=int(p.parent.name);frame=int(p.stem);ids=z['ids'];di=z['diag_ids'];ct=z['contacts'];cp=z['collision_points'];top=z['top']
  for hi,s in enumerate(SEEDS):
   chosen=top[hi];pos=np.searchsorted(ids,chosen);dp=np.searchsorted(di,chosen)
   known=ct[dp,0]==1
   # High requires actual GT contacts visibly on surface and no blocked/unknown
   # swept-body probes. Unknown contacts remain a separate category.
   v=z['V'][pos];unblocked=(v[:,[2,3,12,13,22,23]].sum(1)==0)
   high=known&(ct[dp,5:7]==1).all(1)&unblocked
   group=np.where(~known,-1,np.where(high,1,0))
   valid=(z['mu'][pos]>0)&(z['mu'][pos]<=.400001)
   for k,idx in enumerate(chosen):
    col=bool(z['collision'][pos[k]]);empty=bool(z['empty'][pos[k]])
    occult=cp[dp[k],4];contact_bad=bool(((ct[dp[k],5:7]==2)|(ct[dp[k],9:11]<.3)).any()) if known[k] else None
    rows.append(dict(scene=scene,frame=frame,head=s,rank=k+1,candidate=int(idx),valid=bool(valid[k]),collision=col,empty=empty,mu=float(z['mu'][pos[k]]),contact_found=bool(known[k]),visibility_group=int(group[k]),collision_occluded_fraction=float(occult) if np.isfinite(occult) else None,contact_occluded_or_grazing=contact_bad))
   for layer in [-1,0,1]:
    m=group==layer;frameauc.append(dict(scene=scene,frame=frame,head=s,layer=layer,n=int(m.sum()),auc=auct(valid[m],z['q'][hi,pos][m]),failure_rate=float((~valid[m]).mean()) if m.any() else np.nan))
 df=pd.DataFrame(rows);df.to_csv(R/'results/DV1_candidates.csv',index=False)
 fa=pd.DataFrame(frameauc);fa.to_csv(R/'results/DV1_frame_auc.csv',index=False)
 top=df[df['rank']==1];collision=top[top.collision&~top.empty]
 stat=[]
 for scene,g in df.groupby('scene'):
  for head,h in g.groupby('head'):
   for layer,t in h.groupby('visibility_group'):
    stat.append(dict(scene=int(scene),head=int(head),layer=int(layer),n=len(t),failure_rate=float(1-t.valid.mean()),auc_mean=med([])))
 scene_rows=[]
 for (scene,head,layer),v in fa.groupby(['scene','head','layer']):
  g=df[(df.scene==scene)&(df['head']==head)&(df.visibility_group==layer)]
  scene_rows.append(dict(scene=int(scene),head=int(head),layer=int(layer),n=len(g),failure_rate=float(1-g.valid.mean()) if len(g) else np.nan,auc=float(v.auc.mean()),frames_with_auc=int(v.auc.notna().sum())))
 pd.DataFrame(scene_rows).to_csv(R/'results/DV1_scene.csv',index=False)
 low=fa[fa.layer==0];high=fa[fa.layer==1]
 weak=df[(~df.collision)&((df.mu<0)|(df.mu>=.59999))]
 summary=dict(frames=len(list((R/'cache/c3').glob('*/*.npz'))),head_frames=len(top),top1_nonempty_collisions=len(collision),fraction_collision_occluded80=float((collision.collision_occluded_fraction>=.8).mean()),collision_gate=bool((collision.collision_occluded_fraction>=.8).mean()>=.6),
  layer_auc={str(k):dict(auc=float(v.auc.mean()) if v.auc.notna().any() else None,eligible_head_frames=int(v.auc.notna().sum()),candidate_count=int(v.n.sum())) for k,v in fa.groupby('layer')},
  layer_failure={str(k):float(1-v.valid.mean()) for k,v in df.groupby('visibility_group')},
  weak_contact_found=float(weak.contact_found.mean()),weak_occluded_or_grazing=float(weak.contact_occluded_or_grazing.dropna().mean()),note='Association only; missing GT contacts kept unknown. Layer AUROC uses each same-frame top10 subset.')
 write(R/'results/DV1.json',summary)
 print(json.dumps({'DV0':physics,'DV1':summary}),flush=True)
if __name__=='__main__':main()
