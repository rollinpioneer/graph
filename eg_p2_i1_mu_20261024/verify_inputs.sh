source ~/work/grasp/eg_sub25_b_f05_repl_v1_20261019/activate.sh
cd ~/work/grasp/eg_p2_i1_mu_20261024
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
import json,hashlib
from pathlib import Path
import numpy as np
r=Path.cwd();n0=r.parent/'eg_n0_nref_main_20261024'
assert not (r/'STOP_ON_MISMATCH').exists()
paths=sorted((r/'evaluation').glob('*.npz'));assert len(paths)==48
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in paths:
 receipt=json.loads(p.with_suffix('.json').read_text());cp=n0/'cpu'/('P2_'+p.name);ep=n0/'per_pose'/('P2_'+p.name)
 assert sha(cp)==receipt['input_sha256'] and sha(ep)==receipt['per_pose_sha256'] and sha(p)==receipt['output_sha256']
 a=np.load(p);c=np.load(cp);e=np.load(ep)
 assert np.array_equal(a['pred'],c['pred'])
 for k in ['candidate_id','original_angle','object_id','collision','empty']:assert np.array_equal(a[k],e[k]),(p.name,k)
print(json.dumps(dict(frames=48,stored_pred_equal_to_input=True,input_hashes_unchanged=True,output_hashes_verified=True,stored_identity_and_labels_equal=True)))
PY
