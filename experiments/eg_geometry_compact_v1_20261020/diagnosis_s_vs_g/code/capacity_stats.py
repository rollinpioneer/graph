import os,json,numpy as np
from pathlib import Path
r=Path(os.environ['R']);d=r/'diagnosis_s_vs_g'
with np.load(d/'results/retention_by_pose.npz') as z:counts=z['true_counts']
with np.load(r.parent/'eg_b_global_rank_pilot_v1_20261020/prepared.npz') as z:roles=z['role']
out=[]
for role in ['fit','model_val','gate_cal']:
 c=counts[roles==role,:,0];n=c.sum(-1);cut=(c>np.array([24,24,16,16,48])).any(-1)
 out.append(dict(role=role,original_poses=int(n.size),fraction_true_union_le128=float((n<=128).mean()),fraction_can_hold_all_but_quota_truncates=float(((n<=128)&cut).mean()),true_union_p90=float(np.quantile(n,.9)),true_union_p99=float(np.quantile(n,.99))))
(d/'results/capacity_utilization.json').write_text(json.dumps(out,indent=2)+'\n')
