"""Stationary object-center examples; occlusion/gripper/rotation remain variable."""
from common import *
cases=json.loads((R/'results/representative_cases.json').read_text());idx=json.loads((R/'runs/offline_formal/data/decomposition_index.json').read_text());roi=json.loads((R/'results/P1_object_ROI_evidence.json').read_text());rr={(x['case_id'],x['step'],x['view'],x['object_id']):x for x in roi};ans=[]
for c in cases:
 e=c['episode'];z=np.load(e['trajectory']);st=z['states'][:,e['slot']];z.close();recs=sorted([x for x in idx if x['case_id']==c['case_id'] and x['step'] is not None],key=lambda x:x['step'])
 for a,b in zip(recs[:-1],recs[1:]):
  ta,tb=a['step'],b['step'];zz=np.load(a['raw_path']);za={k:zz[k] for k in ['image','rec']};zz.close();zz=np.load(b['raw_path']);zb={k:zz[k] for k in ['image','rec']};zz.close()
  for j in range(e['n']):
   move=float(np.linalg.norm(st[tb,j,:3]-st[ta,j,:3]));quat=float(np.abs(st[tb,j,3:7]-st[ta,j,3:7]).max())
   if move>.001:continue
   for v in [0,1]:
    x=rr[c['case_id'],ta,v,j];y=rr[c['case_id'],tb,v,j]
    if not(x['visible_proxy_qualified'] and y['visible_proxy_qualified']):continue
    lo=np.minimum(x['roi_xyxy'][:2],y['roi_xyxy'][:2]);hi=np.maximum(x['roi_xyxy'][2:],y['roi_xyxy'][2:]);sl=(slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1));ar=za['rec'][v,:,sl[0],sl[1]];br=zb['rec'][v,:,sl[0],sl[1]];ai=za['image'][v,:,sl[0],sl[1]]/255.;bi=zb['image'][v,:,sl[0],sl[1]]/255.
    ans.append({'case_id':c['case_id'],'n':e['n'],'object_id':j,'view':v,'steps':[ta,tb],'object_center_displacement_m':move,'quaternion_maxabs_change':quat,'visible_color_pixels':[x['visible_color_pixels'],y['visible_color_pixels']],'decoded_color_pixels':[x['reconstruction_color_pixels'],y['reconstruction_color_pixels']],'local_RGB_RMS':float(np.sqrt(np.square(ai-bi).mean())),'local_decoded_RMS':float(np.sqrt(np.square(ar-br).mean())),'limitation':'center stationary does not freeze occlusion, lighting, gripper or rotation; particle IDs never treated as objects'})
dump('P1_stationary_object_frame_pairs.json',ans);print('STATIONARY_ROI_PAIRS',len(ans))
