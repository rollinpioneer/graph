import numpy as np,json
from pathlib import Path
B=Path.home()/'work/grasp';a=np.load(B/'eg_pose_quality_diagnosis_v1_20261022/cache/train_0002_0000.npz');b=np.load(B/'eg_module_controlled_diagnosis_20261023/label_mismatch.npz')
for k in ['batch_grasp_point','batch_valid_mask','batch_grasp_score','batch_grasp_rotations','batch_grasp_depth','batch_grasp_width']:
 x,y=a[k],b[k];ix=np.flatnonzero(np.any(x!=y,axis=-1)) if x.ndim>1 else np.flatnonzero(x!=y);print(k,len(ix),'maxabs',np.max(abs(x.astype(float)-y.astype(float))));print('examples',[(int(i),x[i].tolist(),y[i].tolist()) for i in ix[:10]])
ix=np.flatnonzero(a['batch_valid_mask']!=b['batch_valid_mask']);print('mask dist',[(int(i),float(a['dist'][i]),float(np.linalg.norm(a['pred'][i,13:16]-b['batch_grasp_point'][i])),bool(a['valid_point_mask'][i]),bool(a['valid_view_mask'][i])) for i in ix])
