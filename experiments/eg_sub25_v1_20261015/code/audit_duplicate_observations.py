from pathlib import Path
import json,hashlib,zlib,os
import numpy as np
os.nice(10)
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';OLD=Path('/home/xushijie3/work/eg_full_repro_v1_20261015/data_recovery');E=R/'runs/EG_SUB25_V1/evaluation';data=R/'data/graspnet'
audit=json.loads((D/'DIAGNOSTIC_OBSERVATION_AUDIT.json').read_text());expected=json.loads((OLD/'official_zip_crc_summary.json').read_text())['groups'];records={188:[],189:[]}
with (OLD/'scenes.tar.files.jsonl').open() as f:
 for line in f:
  x=json.loads(line)
  for scene in records:
   if x['path'].startswith(f'scenes/scene_{scene:04d}/'):records[scene].append(x)
for entry in audit['scenes']:
 scene=entry['scene'];poses=np.load(data/f'scenes/scene_{scene:04d}/kinect/camera_poses.npy');entry['unique_camera_pose_matrices']=len(np.unique(poses.reshape(256,-1),axis=0))
 if scene not in records:continue
 current=[]
 for x in records[scene]:
  p=data/x['path'];b=p.read_bytes();crc=f'{zlib.crc32(b)&0xffffffff:08x}';assert crc==x['crc32'] and len(b)==x['bytes'];current.append({'path':x['path'],'bytes':len(b),'crc32':crc})
 canonical=''.join(f"{x['path']}\t{x['bytes']}\t{x['crc32']}\n" for x in sorted(current,key=lambda x:x['path']));v={'files':len(current),'bytes':sum(x['bytes'] for x in current),'sha256_of_path_size_crc32_list':hashlib.sha256(canonical.encode()).hexdigest()};assert v==expected[f'scene_{scene:04d}'];entry['fresh_official_crc_recheck']={'status':'PASS',**v,'reference_sha256':hashlib.sha256((OLD/'official_zip_crc_summary.json').read_bytes()).hexdigest()}
 groups=entry['channels']['rgb']['duplicate_groups'];entry['repeated_observation_AP']={}
 for g in groups:
  ids=g['frames'];ph=[]
  for frame in ids:
   with np.load(E/'inputs/diagnostic_novel'/f'scene_{scene:04d}/kinect/{frame:04d}.npz') as z:ph.append(hashlib.sha256(z['point_clouds'].tobytes()).hexdigest())
  v={'frames':ids,'unique_pointcloud_arrays':len(set(ph)),'unique_camera_poses':len(np.unique(poses[ids].reshape(len(ids),-1),axis=0)),'models':{}}
  for role in ['subset','official']:
   ap=np.load(E/'results'/role/'diagnostic_novel'/f'scene_{scene:04d}/ap.npy')[ids].mean(axis=(1,2))*100;v['models'][role]={'AP_min':float(ap.min()),'AP_max':float(ap.max()),'AP_mean':float(ap.mean())}
  entry['repeated_observation_AP'][g['sha256']]=v
 print('AUDIT',scene,'OFFICIAL_CRC_PASS','POSES',entry['unique_camera_pose_matrices'],'REPEATED_GROUPS',[(len(v['frames']),v['unique_pointcloud_arrays'],v['unique_camera_poses'],v['models']) for v in entry['repeated_observation_AP'].values()],flush=True)
audit['interpretation']='Native256frame AP unchanged;repeated RGB/depth/meta/seg observations are not independent viewpoints. Same-observation AP variation may include random point sampling/numerics;no controlled variance decomposition was done.'
(D/'DIAGNOSTIC_OBSERVATION_AUDIT.json').write_text(json.dumps(audit,indent=2))
