import json,csv,hashlib,math,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];ROOT=R/'runs/EG_FULL_V1';results={};scenes=[];identities=[]
assert json.loads((R/'data_recovery/TRAINING_COMPLETE.json').read_text())['status']=='PASS'
for split,start in [('seen',100),('similar',130),('novel',160)]:
 d=ROOT/'test'/split;a=np.load(d/f'ap_kinect_{split}.npy',allow_pickle=False)
 assert a.shape==(30,256,50,6) and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1,(split,a.shape)
 rows=[json.loads(x) for x in (d/'inference_metrics.jsonl').read_text().splitlines()]
 assert len(rows)==7680
 batch_times={};counts=[]
 for offset,row in enumerate(rows):
  scene=start+offset//256;frame=offset%256;assert row['scene']==f'scene_{scene:04d}' and row['frame']==frame
  p=d/f'scene_{scene:04d}'/'kinect'/f'{frame:04d}.npy';g=np.load(p,allow_pickle=False)
  assert g.ndim==2 and g.shape[1]==17 and np.isfinite(g).all() and len(g)==row['saved_count']==row['finite_count']
  assert len(g)>0
  counts.append(len(g));batch_times[row['batch_idx']]=row['batch_forward_decode_seconds']
  for q in [p,d/'input_samples'/f'scene_{scene:04d}'/'kinect'/f'{frame:04d}.npz']:
   h=hashlib.sha256(q.read_bytes()).hexdigest();identities.append(f'{q.relative_to(ROOT)}\t{q.stat().st_size}\t{h}\n')
 for i in range(30):
  scenes.append({'split':split,'scene':start+i,'frames':256,'ap_percent':100*float(a[i].mean()),'ap_04_percent':100*float(a[i,:,:,1].mean()),'ap_08_percent':100*float(a[i,:,:,3].mean()),'mean_saved_grasps':float(np.mean(counts[i*256:(i+1)*256]))})
 results[split]={'status':'PASS','scenes':30,'frames':7680,'ap_percent':100*float(a.mean()),'ap_04_percent':100*float(a[:,:,:,1].mean()),'ap_08_percent':100*float(a[:,:,:,3].mean()),'forward_decode_seconds':sum(batch_times.values()),'forward_decode_mean_ms_per_frame':1000*sum(batch_times.values())/7680,'saved_grasps_min':min(counts),'saved_grasps_max':max(counts),'saved_grasps_mean':float(np.mean(counts)),'raw_ap_array':str(d/f'ap_kinect_{split}.npy')}
manifest=ROOT/'prediction_input_manifest.tsv';manifest.write_text(''.join(identities));result={'status':'PASS','experiment_id':'EG-FULL-REPRO-V1','checkpoint':str(ROOT/'train/economicgrasp_epoch10.tar'),'camera':'kinect','seed':20261015,'splits':results,'official_reference_ap_percent':{'seen':62.59,'similar':51.73,'novel':19.54},'ap_is_not_robot_success_rate':True,'collision_thresh':0,'official_top_k':50,'prediction_input_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
(ROOT/'evaluation_results.json').write_text(json.dumps(result,indent=2));(R/'data_recovery/EVALUATION_COMPLETE.json').write_text(json.dumps(result,indent=2))
with (ROOT/'per_scene_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(scenes[0]));w.writeheader();w.writerows(scenes)
print(json.dumps(results),flush=True)
