"""P1 one-pass operation audit and separately registered exploratory image readout."""
from common import *
import time,itertools
begin=time.time();es=json.loads((R/'results/episode_index.json').read_text())
(R/'receipts/freeze_vision_proxy.json').write_text(json.dumps({'utc':utc(),'role':'exploratory image readout; never retroactively qualify P2 matches','camera_source':str(OLD/'EC-Diffuser/isaac_panda_push_env.py'),'hfov_deg':35,'resolution':128,'cameras':[[.8,0,1.8],[0,-.8,1.65]],'targets':[[.12,0,1.025],[0,-.12,1.025]],'color_masks':'saturation>40, max>40; nearest normalized 6-color palette distance<.4','roi':'projected cube corners +2px; pixel counts do not equal visibility fraction','color_correspondence':'recorded getter invalid; infer from goal-image geometry by bijective assignment, require both views agree and >=5 color pixels in at least one view','dlp':'not object identity; no per-step latents saved','code_sha256':sha(__file__)},indent=2))
palette=np.array([[.6,.1,0],[0,.6,.1],[0,.1,.8],[.7,.7,0],[.5,0,.5],[0,.9,.9]])
palette/=palette.max(1)[:,None]
cameras=[np.array([.8,0,1.8]),np.array([0,-.8,1.65])];targets=[np.array([.12,0,1.025]),np.array([0,-.12,1.025])]
corners=np.array(list(itertools.product([-1,1],repeat=3)))*.0175
def project(x,v):
 f=targets[v]-cameras[v];f=f/np.linalg.norm(f);right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,f);d=x-cameras[v];dep=d@f
 return np.stack([64+64/np.tan(np.deg2rad(35)/2)*(d@right)/dep,64-64/np.tan(np.deg2rad(35)/2)*(d@up)/dep],-1)
def rgb(a):
 a=np.moveaxis(a,0,-1);return (a*255).astype('uint8') if a.max()<=1 else a.astype('uint8')
def classes(a):
 a=rgb(a).astype(float);mx=a.max(-1);good=(mx-a.min(-1)>40)&(mx>40);norm=a/np.maximum(mx[...,None],1);d=np.linalg.norm(norm[:,:,None,:]-palette,axis=-1)
 best=d.argmin(-1);best[~good|(d.min(-1)>.4)]=-1;return best
def region(x,v):
 pp=project(x+corners,v);lo=np.floor(pp.min(0)-2).astype(int).clip(0,127);hi=np.ceil(pp.max(0)+2).astype(int).clip(0,127)
 return lo,hi
def color_scores(img,xyz,v):
 cl=classes(img);ans=[]
 for x in xyz:
  lo,hi=region(x,v);p=cl[lo[1]:hi[1]+1,lo[0]:hi[0]+1];ans.append([int((p==c).sum()) for c in range(6)])
 return np.array(ans)
vision=[];operations=[];cache={};initcache={};timeline=[]
for e in es:
 if e['trajectory'] not in cache:cache[e['trajectory']]=np.load(e['trajectory'])
 if e['initial'] not in initcache:initcache[e['initial']]=np.load(e['initial'])
 z=cache[e['trajectory']];ini=initcache[e['initial']];i=e['slot'];st=z['states'][:,i];dd=z['distances'][:,i];ee=z['eef'][:,i];n=e['n']
 goal_xyz=np.c_[z['goal'][i],np.full(n,1.0425)]
 sc=[color_scores(ini['goal_image'][i,v],goal_xyz,v) for v in range(2)]
 assignments=[]
 for v in range(2):
  assignments.append(max(itertools.permutations(range(6),n),key=lambda a:sum(sc[v][j,c] for j,c in enumerate(a))))
 agree=assignments[0]==assignments[1]
 a=assignments[0];credible=agree and all(max(sc[v][j,c] for v in (0,1))>=5 for j,c in enumerate(a))
 cnts=[]
 for k,t in enumerate(z['image_steps']):
  c=[color_scores(z['images'][k,i,v],st[int(t),:,:3],v) for v in (0,1)]
  counts=[[int(c[v][j,a[j]]) for v in (0,1)] for j in range(n)]
  cnts.append({'step':int(t),'object_roi_color_pixels':counts})
  # Object-level sparse timeline, causal unknowns explicit.
  nearest=int(np.linalg.norm(st[int(t),:,:3]-ee[int(t)],axis=-1).argmin())
  timeline.append({'episode_id':e['episode_id'],'step':int(t),'pending_count':int((dd[int(t)]>=.04).sum()),'mean_goal_distance_m':float(dd[int(t)].mean()),
  'eef_nearest_object_proxy':nearest,'nearest_object_pending':bool(dd[int(t),nearest]>=.04),'per_object_error_m':dd[int(t)].tolist(),
  'visibility_proxy_pixels':counts if credible else None,'contact_or_choice':'unknown'})
 vision.append({'episode_id':e['episode_id'],'goal_assignment_per_view':[list(x) for x in assignments],
 'same_bijective_assignment':agree,'credible_color_projection_proxy':credible,'goal_roi_color_scores':[x.tolist() for x in sc],
 'counts':cnts if credible else None,'limitation':'pixel count is projection/color proxy, not validated visible surface or DLP object representation; no segmentation truth'})
 near=np.linalg.norm(st[...,:3]-ee[:,None,:],axis=-1);who=near.argmin(-1);active=near.min(-1)<.055
 switches=int(((who[1:]!=who[:-1])&active[1:]&active[:-1]).sum())
 pending=dd>=.04;steps=e['steps'];progress=(dd[:-1]-dd[1:])>.001
 actions=z['actions'][:,i];moved=np.linalg.norm(np.diff(st[...,:3],axis=0),axis=-1)>.001
 opp=int(pending[:-1].sum())
 operations.append({'episode_id':e['episode_id'],'protocol':e['protocol'],'n':n,'layout':e['layout'],'steps':steps,'all_success':e['endpoints'][str(steps)]['all_success'],
 'initial_pending':int(pending[0].sum()),'initial_transport_total_m':float(dd[0].sum()),'pending_object_control_opportunities':opp,
 'per_object_effective_progress_steps':progress.sum(0).tolist(),'progress_object_steps_per_pending_opportunity':float((progress&pending[:-1]).sum()/max(1,opp)),
 'nearest_proxy_switches':switches,'near_already_hit_proxy_steps':int((active&~pending[np.arange(len(dd)),who]).sum()),
 'active_near_proxy_control_steps':int(active[:-1].sum()),'motion_when_already_hit_object_steps':int((moved&~pending[:-1]).sum()),
 'first_no_progress':e['first_no_progress'],'first_fall':e['first_fall'],'note':'nearest/near are geometry only; not selected/contacted object'})
dump('P1_vision_proxy.json',vision);dump('P1_sparse_object_timeline.json',timeline);dump('P1_operation_by_episode.json',operations)
# Protocol-stratified quantity summaries with identical observation windows.
quantity=[]
for cohort in ('historical','PC_SCALE02'):
 layouts=sorted({e['layout'] for e in es if e['cohort']==cohort})
 for layout in layouts:
  for n in (3,6):
   selected=[e for e in es if e['cohort']==cohort and e['n']==n and (e['layout']==layout if cohort!='PC_SCALE02' else e['layout'].endswith(layout.split('_')[-1]))]
   if not selected:continue
   win=100 if cohort=='historical' else 200
   vals=[]
   for e in selected:
    z=cache[e['trajectory']];st=z['states'][:win+1,e['slot']];d=z['distances'][:win+1,e['slot']];ends=no_progress(st,d)
    vals.append({'episode_id':e['episode_id'],'window_steps':win,'fraction':float((d[-1]<.04).mean()),
    'first_fall':first_run((st[...,2]<.975).any(-1)),'first_no_progress':ends[0] if ends else None,'no_progress_window_ends':len(ends),
    'pending_object_control_opportunities':int((d[:-1]>=.04).sum()),'pending_mean_objects_per_step':float((d[:-1]>=.04).sum()/win),
    'no_progress_detection_fraction_of_controls':len(ends)/win,'no_progress_per_pending_opportunity':len(ends)/max(1,int((d[:-1]>=.04).sum()))})
   quantity.append({'cohort':cohort,'layout':layout if cohort=='historical' else layout.split('_')[-1],'n':n,'window':win,'episodes':len(vals),
    'mean_fraction':float(np.mean([v['fraction'] for v in vals])),'fall_episodes':sum(v['first_fall'] is not None for v in vals),
    'no_progress_episodes':sum(v['first_no_progress'] is not None for v in vals),'episode_rows':vals,
    'unit':'episode; each object/control opportunity normalizes exposure only, not independent replication','causal_contribution':'unidentified'})
# De-duplicate layout suffix summaries.
seen=set();quantity=[q for q in quantity if not ((q['cohort'],q['layout'],q['n']) in seen or seen.add((q['cohort'],q['layout'],q['n'])))]
dump('quantity_connection.json',quantity)
# Same-protocol successful/failing episodes with nearest initial workload; descriptive reference only.
comparisons=[]
for e in es:
 if e['endpoints'][str(e['steps'])]['all_success']:continue
 pool=[q for q in es if q['protocol']==e['protocol'] and q['layout']==e['layout'] and q['n']==e['n'] and q['endpoints'][str(q['steps'])]['all_success']]
 if pool:
  q=min(pool,key=lambda q:(abs(q['initial_transport_mean_m']-e['initial_transport_mean_m']),q['episode_id']))
  comparisons.append({'failure':e['episode_id'],'success_reference':q['episode_id'],'initial_mean_transport_abs_diff_m':abs(q['initial_transport_mean_m']-e['initial_transport_mean_m']),
  'initial_pending_counts':[e['initial_pending'],q['initial_pending']],'role':'descriptive workload-near reference; not causal matching; visibility/geometry may differ'})
dump('P1_success_failure_references.json',comparisons)
summary={'utc':utc(),'episodes':len(es),'objects':sum(e['n'] for e in es),'same_protocol_success_failure_references':len(comparisons),
'goal_color_projection_agreement':sum(v['same_bijective_assignment'] for v in vision),
'credible_goal_color_projection_proxies':sum(v['credible_color_projection_proxy'] for v in vision),'saved_getter_color_valid_episodes':0,
'contact_or_selection_truth':'unavailable','dlp_per_step_history':'not persisted; initial/goal particles do not identify object selection',
'G1_status':'pending explicit cause comparison; no causal contribution from one-pass descriptive audit','cpu_wall_seconds':time.time()-begin}
dump('P1_summary.json',summary);print(json.dumps(summary),flush=True)
