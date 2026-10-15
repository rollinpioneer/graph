import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
import cpu_knn_probe as cpu
R=cpu.R;S=cpu.S;source=S/'dataset/generate_graspness.py';text=source.read_text();root=R/'data/graspnet/graspness'
available=sorted(p for p in root.glob('scene_*') if len(list((p/'kinect').glob('*.npy')))==256)
chosen=[available[i] for i in np.unique(np.linspace(0,len(available)-1,min(12,len(available)),dtype=int))];started=time.time();rows=[]
for scene in chosen:
 sid=int(scene.name.split('_')[1]);output=R/'runs/EG_FULL_V1/cpu_knn_cross_scene_check';frames=[0,127,255]
 sys.argv=[str(source),'--dataset_root',str(R/'data/graspnet'),'--camera_type','kinect','--scene_start',str(sid),'--scene_end',str(sid+1),'--output_root',str(output),'--cache_scene_graspness']
 marker='for ann_id in range(cfgs.frame_end):';assert text.count(marker)==1
 code=text.replace(marker,'for ann_id in [0, 127, 255]:');exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
 for frame in frames:
  expected=scene/'kinect'/f'{frame:04d}.npy';actual=output/scene.name/'kinect'/f'{frame:04d}.npy';a=np.load(expected);b=np.load(actual);proof=json.loads(Path(str(expected)+'.sha256.json').read_text());assert hashlib.sha256(expected.read_bytes()).hexdigest()==proof['sha256']
  row={'scene':sid,'frame':frame,'exact_equal':bool(np.array_equal(a,b)),'max_abs_diff':float(np.max(np.abs(a-b))),'gpu_sha256':proof['sha256'],'cpu_sha256':hashlib.sha256(actual.read_bytes()).hexdigest()};rows.append(row);print(json.dumps(row),flush=True)
  if not row['exact_equal']:raise RuntimeError('CPU equivalence failed; preserve existing official GPU labels')
result={'status':'PASS','probe_only':True,'formal_label_generation_switched':False,'scenes':len(chosen),'frames':rows,'stats':cpu.stats,'wall_seconds':time.time()-started,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cpu_code_sha256':hashlib.sha256(Path(cpu.__file__).read_bytes()).hexdigest()};(R/'data_recovery/CPU_KNN_CROSS_SCENE_CHECK.json').write_text(json.dumps(result,indent=2));print('CROSS_SCENE_CPU_EQUIVALENCE_PASS',flush=True)
