"""Freeze N0 panel and create a minimal independent frozen-EG inference runner."""
import json, hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
excluded=[2,9,31,33,64,68,76,82]
scenes=np.random.default_rng(20261024).choice([s for s in range(100) if s not in excluded],12,replace=False).tolist()
plan=dict(seed=20261024,training_scenes=scenes,training_frames=[0,64,128,192],
 C3_scenes=[100,106,115,124,132,139,150,157,160,169,176,186],C3_frames=[0,64,128,192],P2_frames=[32,96,160,224],
 excluded_training_scenes=excluded,voxel_size=.0025,truncation=.010,epsilon=.0025,kappa_grid=[0,.1,.25,.5],
 initial_check=[[scenes[0],0],[scenes[1],0]],agreement_min=.99,
 agreement_definition='Report both pointwise continuous projection and identical voxel-center projection; do not silently replace one gate by the other. If continuous fails, report it explicitly before any C3 comparison.',
 calibration_false_filter_denominator='GT clean and nonempty (no FC needed); feasible separately on C3',
 calibration_ties='choose strict r>threshold with maximum scene-equal recall subject to scene-equal false-filter budget; ties: lower false filter, then lower kappa',
 OP1='EG original calibration false filter',OP2='EG original calibration false filter +0.05',
 C3_formal_runs=0,training=False,parameter_updates=0,C4_C5=False)
(R/'protocol.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
src=(R.parents[1]/'work/eg_pose/infer.py').read_text()
src=src.replace("ROOT=B/'eg_pose_quality_diagnosis_v1_20261022'","ROOT=B/'eg_n0_nvblox_20261024'")
src=src.replace("ROOT/'study_plan.json'","ROOT/'protocol.json'")
start=src.index("pairs=[('C3'")
end=src.index("(ROOT/'cache').mkdir",start)
src=src[:start]+"pairs=[('train',s,f) for s in plan['training_scenes'] for f in plan['training_frames']]\nif MODE=='pilot':pairs=[('train',s,f) for s,f in plan['initial_check']]\nif MODE=='P2':pairs=[('P2',s,f) for s in plan['C3_scenes'] for f in plan['P2_frames']]\n"+src[end:]
start=src.index(" if MODE=='reprobe':")
end=src.index(" if dest.exists():",start)
src=src[:start]+src[end:]
start=src.index(" if split=='train':\n  out.update(label_probe")
end=src.index(" cloud,_=",start)
src=src[:start]+src[end:]
src=src.replace("int(out['batch_valid_mask'].sum()) if split=='train' else None","None")
start=src.index('def label_probe(')
end=src.index('plan=json.loads',start)
src=src[:start]+src[end:]
(R/'infer_frozen.py').write_text(src,encoding='utf-8')
print(json.dumps(plan))
