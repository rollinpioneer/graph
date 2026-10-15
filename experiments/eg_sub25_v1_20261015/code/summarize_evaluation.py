from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';OUT=R/'runs/EG_SUB25_V1/evaluation';cfg=json.loads((D/'evaluation_run_config.json').read_text())
scene_rows=[];frame_rows=[];summaries={};inputs={};prediction_count=0
for part,scenes in cfg['partitions'].items():
 summaries[part]={}
 for role in cfg['inference_models'][part]:
  root=OUT/'predictions'/role/part;receipt=json.loads((root/'INFERENCE_COMPLETE.json').read_text());assert receipt['status']=='PASS' and receipt['frames']==len(scenes)*256
  lines=[json.loads(l) for l in (root/'inference_metrics.jsonl').read_text().splitlines()];assert len(lines)==len(scenes)*256;metric={(x['scene'],x['frame']):x for x in lines};assert len(metric)==len(lines)
  inputs[(part,role)]=receipt['input_manifest_sha256'];aps=[]
  for scene in scenes:
   result=OUT/'results'/role/part/f'scene_{scene:04d}';proof=json.loads((result/'COMPLETE.json').read_text());ap=np.load(result/'ap.npy');assert proof['status']=='PASS' and ap.shape==(256,50,6) and hashlib.sha256((result/'ap.npy').read_bytes()).hexdigest()==proof['ap_sha256'];aps.append(ap)
   scene_rows.append({k:proof[k] for k in ['model','partition','scene','frames','ap_percent','ap04_percent','ap08_percent','wall_seconds']})
   for f in range(256):
    m=metric[(scene,f)];p=root/f'scene_{scene:04d}/kinect/{f:04d}.npy';assert hashlib.sha256(p.read_bytes()).hexdigest()==m['prediction_sha256'];g=np.load(p);assert g.shape==(1024,17) and np.isfinite(g).all();prediction_count+=1
    with np.load(result/f'evaluated/{f:04d}.npz') as z:q=z['quality'];col=z['collision']
    frame_rows.append({'model':role,'partition':part,'scene':scene,'frame':f,'ap_percent':float(ap[f].mean()*100),'ap04_percent':float(ap[f,:,1].mean()*100),'ap08_percent':float(ap[f,:,3].mean()*100),'decoded_grasps':len(g),'official_evaluated_grasps':len(q),'top1_quality':float(q[0]) if len(q) else None,'top1_collision':bool(col[0]) if len(q) else None,'top1_valid04':bool(0<q[0]<=.4 and not col[0]) if len(q) else False,'top1_valid08':bool(0<q[0]<=.8 and not col[0]) if len(q) else False,'top10_collision_fraction':float(col[:10].mean()) if len(q) else None})
  stack=np.asarray(aps);summaries[part][role]={'scenes':scenes,'frames':len(lines),'ap_percent':float(stack.mean()*100),'ap04_percent':float(stack[:,:,:,1].mean()*100),'ap08_percent':float(stack[:,:,:,3].mean()*100),'inference_wall_seconds':receipt['wall_seconds'],'forward_decode_ms_per_frame':receipt['forward_decode_seconds']/len(lines)*1000,'input_manifest_sha256':receipt['input_manifest_sha256']}
for part in ['diagnostic_seen','diagnostic_novel']:assert inputs[(part,'subset')]==inputs[(part,'official')]
assert prediction_count==4608 and len(scene_rows)==18 and len(frame_rows)==4608
for name,rows in [('per_scene_results.csv',scene_rows),('per_frame_results.csv',frame_rows)]:
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
paired=[]
for part in ['diagnostic_seen','diagnostic_novel']:
 for scene in cfg['partitions'][part]:
  a=next(x for x in scene_rows if x['scene']==scene and x['model']=='subset');b=next(x for x in scene_rows if x['scene']==scene and x['model']=='official');paired.append({'partition':part,'scene':scene,'subset_ap_percent':a['ap_percent'],'official_ap_percent':b['ap_percent'],'subset_minus_official_pp':a['ap_percent']-b['ap_percent']})
with (OUT/'paired_scene_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
selected=[]
for part in ['diagnostic_seen','diagnostic_novel']:
 for scene in cfg['partitions'][part]:
  rows=sorted([x for x in frame_rows if x['model']=='subset' and x['scene']==scene],key=lambda x:(x['ap_percent'],x['frame']))
  for tag,row in [('lowest_1',rows[0]),('lowest_2',rows[1]),('median',rows[len(rows)//2])]:selected.append({'partition':part,'scene':scene,'frame':row['frame'],'selection':tag,'subset_ap_percent':row['ap_percent'],'exploratory_posthoc':True})
assert len(selected)==12;(D/'BAD_CASE_SELECTION.json').write_text(json.dumps({'selection_rule':cfg['bad_case_selection'],'cases':selected},indent=2))
result={'status':'PASS','model_frame_evaluations':4608,'scene_evaluations':18,'summaries':summaries,'paired_scene_results':paired,'paired_input_manifests_equal':True,'AP_definition':'Mean official precision tensor over complete256frames,top1..50,frictions0.2..1.2;percent units','dev_interpretation':'Held-out scenes from official training partition;development only','official_scene_interpretation':'Frozen exploratory test scenes,not full Seen/Novel split AP','checkpoints':cfg['models'],'native_eval_scene_sha256':cfg['graspnet_api_eval_scene_sha256'],'bad_case_status':'SELECTED_PENDING_ANALYSIS'}
(OUT/'evaluation_results.json').write_text(json.dumps(result,indent=2));(D/'EVALUATION_COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(summaries,indent=2))
