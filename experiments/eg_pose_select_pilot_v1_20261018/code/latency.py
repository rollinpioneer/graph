from core import *
import torch
os.nice(10);torch.set_num_threads(1);heads=json.loads((R/'results/selected_heads.json').read_text());gate=json.loads((R/'results/gate_calibration.json').read_text());tau=float('inf') if gate['tau'] is None else gate['tau'];pred,hyp,feat,act,actions=export(CFG['development'][0],0);records=[]
for h in heads:
 ck=torch.load(R/f'heads/{h["seed"]}/step{h["selected_step"]}.pt',map_location='cpu',weights_only=False);m=make_model().eval();m.load_state_dict(ck['model']);times=[]
 for rep in range(23):
  t=time.perf_counter();x=np.concatenate([np.broadcast_to(((feat-ck['mean'])/ck['std'])[:,None,:],(1024,6,320)),act],-1).copy()
  with torch.no_grad():q=m(torch.from_numpy(x)).squeeze(-1).numpy()
  ch=select(q,actions,'S4',tau);new=hyp[np.arange(1024),ch].copy()
  if rep>=3:times.append((time.perf_counter()-t)*1000)
 assert np.array_equal(new[:,[0,1,2,13,14,15,16]],pred[:,[0,1,2,13,14,15,16]])
 records.append(dict(seed=h['seed'],median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),repetitions=20))
write(R/'results/latency.json',dict(status='PASS',device='CPU single thread',candidates=1024,actions_scored=6,selected_policy='S4',measure='normalization, six-action head, threshold selection and geometry writeback; excludes EG, IO, NMS and CD; not end-to-end robot latency',rows=records,matched_EG_path_ratio='not measured here; S2 lightweight decision requires same-input matched benchmark'))
