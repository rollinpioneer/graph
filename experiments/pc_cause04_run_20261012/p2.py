"""P2 frozen, outcome-independent matching; geometric proposals do not pass visibility gate."""
from common import *
import time
begin=time.time();es=json.loads((R/'results/episode_index.json').read_text());freeze=json.loads((R/'receipts/freeze_analysis.json').read_text())['P2']
bins=freeze['stage_bins'];tol=freeze['matching_tolerances']
cache={}
def data(e):
 if e['trajectory'] not in cache:cache[e['trajectory']]=np.load(e['trajectory'])
 return cache[e['trajectory']]
def features(e,start):
 z=data(e);i=e['slot'];st=z['states'][start,i];d=z['distances'][start,i];ee=z['eef'][start,i];pending=d>=.04
 return {'nearest_eef_object_m':float(np.linalg.norm(st[:,:3]-ee,axis=-1).min()),'eef_height_m':float(ee[2]),'mean_pending_goal_distance_m':float(d[pending].mean()) if pending.any() else 0.,
 'pending_count':int(pending.sum()),'min_bbox_edge_m':float(bbox_margin(st).min()),'initial_transport_mean_m':e['initial_transport_mean_m']}
def stage(start):return int(np.searchsorted(bins,start,side='right')-1)
controls=[];cases=[]
for e in es:
 if e['endpoints'][str(e['steps'])]['all_success']:
  for start in range(20,e['steps']-39,10):
   controls.append({'id':e['episode_id']+'/start%d'%start,'episode_id':e['episode_id'],'start':start,'end':start+20,'protocol':e['protocol'],'n':e['n'],'layout':e['layout'],'stage':stage(start),'features':features(e,start)})
 if e['first_no_progress'] is not None:
  t=e['first_no_progress'];cases.append({'id':e['episode_id']+'/start%d'%(t-20),'episode_id':e['episode_id'],'start':t-20,'end':t,'protocol':e['protocol'],'n':e['n'],'layout':e['layout'],'stage':stage(t-20),'features':features(e,t-20)})
dump('P2_frozen_candidate_controls.json',controls);dump('P2_frozen_cases.json',cases)
(R/'receipts/freeze_control_lists.json').write_text(json.dumps({'utc':utc(),'controls_sha256':sha(R/'results/P2_frozen_candidate_controls.json'),'cases_sha256':sha(R/'results/P2_frozen_cases.json'),'matching':'list and code frozen before comparison; no future-based rematching','code_sha256':sha(__file__)},indent=2))
byid={e['episode_id']:e for e in es};rows=[];used={}
for a in cases:
 e=byid[a['episode_id']];z=data(e);i=e['slot'];start,end=a['start'],a['end'];proposals=[]
 valid_context=start>=20 and end+20<=e['steps']
 if valid_context:
  for b in controls:
   if b['episode_id']==a['episode_id'] or any(b[k]!=a[k] for k in ('protocol','n','layout','stage')):continue
   errs={k:abs(a['features'][k]-b['features'][k]) for k in tol}
   if all(errs[k]<=tol[k]+1e-12 for k in tol):
    score=sum(errs[k]/v for k,v in tol.items() if v)
    proposals.append((score,b['episode_id'],b['start'],b,errs))
 proposals.sort(key=lambda x:x[:3])
 row={'case':a,'chain':chain(z,i,start,end),'context_valid':valid_context,'before':chain(z,i,max(0,start-20),start) if start>0 else None,
 'after':chain(z,i,end,min(end+20,e['steps'])) if end<e['steps'] else None,'images':image_refs(e,z,[max(0,start-20),start,end,min(end+20,e['steps'])]),
 'qualified_match':False,'unmatched_reason':'validated identity visibility unavailable' if valid_context else 'fixed pre/post context unavailable',
 'geometric_candidates':len(proposals),'control':None,'normal_status':'unknown','control_reuse_count':0}
 if proposals:
  _,_,_,b,errs=proposals[0];q=byid[b['episode_id']];zz=data(q);j=q['slot'];cs,ce=b['start'],b['end'];dd=zz['distances'][ce:ce+21,j];st=zz['states'][ce:ce+21,j]
  future_progress=float((dd[0]-dd.min(0)).max());normal=bool(future_progress>=.01 and not (st[...,2]<.975).any())
  used[b['id']]=used.get(b['id'],0)+1
  row.update({'control':b,'matching_abs_errors':errs,'matching_scope':'geometric proposal only, visibility gate not passed','control_chain':chain(zz,j,cs,ce),
  'control_before':chain(zz,j,cs-20,cs),'control_after':chain(zz,j,ce,ce+20),'normal_confirmation_future_max_object_improvement_m':future_progress,
  'normal_status':'offline normal-confirmed' if normal else 'not normal-confirmed; no rematching','control_reuse_count':used[b['id']],
  'control_images':image_refs(q,zz,[cs-20,cs,ce,ce+20])})
 rows.append(row)
# Every duration summary is an episode unit; overlapping windows are not independent.
duration=[]
for e in es:
 ends=e['no_progress_ends'];runs=[]
 for t in ends:
  if not runs or t>runs[-1][-1]+1:runs.append([t])
  else:runs[-1].append(t)
 duration.append({'episode_id':e['episode_id'],'n':e['n'],'protocol':e['protocol'],'layout':e['layout'],'window_count_overlapping':len(ends),
 'episodes_as_independent_units':1,'no_progress_runs':len(runs),'max_consecutive_window_ends':max((len(x) for x in runs),default=0),
 'first_online_detection':e['first_no_progress'],'interpretation':'duration of detector satisfaction; not duration of proven abnormal policy stall'})
dump('P2_matches_and_chains.json',rows);table('P2_episode_durations.csv',duration)
summary={'utc':utc(),'first_window_cases':len(cases),'candidate_windows':len(controls),'candidate_success_episodes':len({b['episode_id'] for b in controls}),
'geometric_proposals':sum(r['control'] is not None for r in rows),'normal_confirmed_proposals':sum(r['normal_status']=='offline normal-confirmed' for r in rows),
'qualified_matches':0,'gate':'FAILED: visibility/identity at window start not validated; geometric matches are insufficient',
'normal_future_rule':freeze['normal_confirmation_offline'],'reuse_total':sum(max(0,v-1) for v in used.values()),
'contact':'unknown even where geometry is near','conclusion':'no-progress observed; abnormal-policy stall is unestablished; do not launch stall intervention',
'cpu_wall_seconds':time.time()-begin}
dump('P2_summary.json',summary);print(json.dumps(summary),flush=True)
