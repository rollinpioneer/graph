"""Dataset-only schema, temporal boundary, and deterministic transition check."""
import numpy as np,json,pathlib,hashlib
r=pathlib.Path(__file__).resolve().parents[1];p=r/'sources/Pipeline-val.npz';d=dict(np.load(p));o=d['obs'].astype('float64');n=d['next_obs'].astype('float64');a=d['action'];ends=np.where(np.logical_or(d['done'],d['truncated']))[0];mask=np.ones(len(o)-1,bool);mask[ends[:-1]]=False
lag=[]
for j in range(25):
 pred=np.clip(o[:,0]+o[:,27+j],0,200);err=np.abs(pred-n[:,0]);lag.append(dict(action_history_index=j,mean_absolute_error=float(err.mean()),max_absolute_error=float(err.max())))
best=min(lag,key=lambda x:x['mean_absolute_error']);report=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in d.items()},transitions=len(o),trajectories=len(ends),lengths=np.diff(np.r_[-1,ends]).tolist(),index_field_present='index' in d,terminal_count=int(d['done'].sum()),truncation_count=int(d['truncated'].sum()),within_trajectory_continuity_max=float(np.max(np.abs(n[:-1][mask]-o[1:][mask]))),action_history_shift_max=float(np.max(np.abs(n[:,28:]-o[:,27:-1]))),new_action_entry_max=float(np.max(np.abs(n[:,27:28]-a))),observation_history_shift_max=float(np.max(np.abs(n[:,3:27]-o[:,2:26]))),best_lag=best,all_lags=lag,environment_steps=0,optimizer_updates=0)
(r/'reports/pipeline_data_check.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['fields','all_lags','lengths']},indent=2));print('lag18,23',lag[18],lag[23])
