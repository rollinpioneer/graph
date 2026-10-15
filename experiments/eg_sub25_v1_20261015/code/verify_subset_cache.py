from pathlib import Path
import json,hashlib,time,shutil
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';ROOT=R/'data/graspnet';OLD=Path('/home/xushijie3/work/eg_full_repro_v1_20261015')
assert json.loads((OLD/'data_recovery/SUPERVISION_VALIDATION.json').read_text())['status']=='PASS'
split=json.loads((D/'frozen_split.json').read_text());started=time.time();records=[];label_bytes=0;observation_bytes=0
for scene in split['train_scene_ids']:
 name=f'scene_{scene:04d}'
 for frame in split['train_frame_ids']:
  p=ROOT/'graspness'/name/'kinect'/f'{frame:04d}.npy';digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==json.loads(Path(str(p)+'.sha256.json').read_text())['sha256'];label_bytes+=p.stat().st_size;records.append(f'{p.relative_to(ROOT)}\t{digest}\n')
  for folder,ext in [('rgb','png'),('depth','png'),('label','png'),('meta','mat')]:observation_bytes+=(ROOT/'scenes'/name/'kinect'/folder/f'{frame:04d}.{ext}').stat().st_size
 p=ROOT/'economic_grasp_label_300views'/f'{name}_labels.npz';digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==json.loads(Path(str(p)+'.sha256.json').read_text())['sha256'];label_bytes+=p.stat().st_size;records.append(f'{p.relative_to(ROOT)}\t{digest}\n')
text=''.join(records);(D/'selected_label_manifest.tsv').write_text(text)
result={'status':'PASS','train_frames':6400,'economic_scenes':50,'label_identity_count':len(records),'selected_labels_bytes':label_bytes,'selected_train_observation_bytes':observation_bytes,'manifest_sha256':hashlib.sha256(text.encode()).hexdigest(),'frozen_split_sha256':hashlib.sha256((D/'frozen_split.json').read_bytes()).hexdigest(),'reused_full_shape_range_semantic_validation':str(OLD/'data_recovery/SUPERVISION_VALIDATION.json'),'full_validation_sha256':hashlib.sha256((OLD/'data_recovery/SUPERVISION_VALIDATION.json').read_bytes()).hexdigest(),'all_selected_label_sha256_rechecked':True,'supervision_generated_again':False,'full_existing_data_deleted':False,'free_bytes':shutil.disk_usage(R).free,'wall_seconds':time.time()-started}
(D/'SUPERVISION_VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
