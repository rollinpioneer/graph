"""CPU reconstruction of frozen action normalization bounds and source semantics."""
import os,pathlib,pickle,json,numpy as np,time,hashlib
R=pathlib.Path(os.environ['R']);p=pathlib.Path(os.environ['OLD'])/'resources/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl';begin=time.monotonic()
data=pickle.load(p.open('rb'));a=data['actions'].astype(np.float32);lengths=data.get('path_lengths',np.full(a.shape[0],a.shape[1]));flat=np.concatenate([a[i,:int(n)] for i,n in enumerate(lengths)]);lo=flat.min(0);hi=flat.max(0)
x={'min':lo.tolist(),'max':hi.tolist(),'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'normalizer':'LimitsNormalizer','formula':'normalized=2*(action-min)/(max-min)-1; unnormalize clips generated normalized actions to[-1,1] if extrema exceed 1+eps','saved_action':'after unnormalization; preclip normalized proposal was not saved in prior recorder; inverse estimates only the clipped normalized action','path_lengths_used':True,'CPU_seconds':time.monotonic()-begin}
(R/'receipts/action_normalizer_audit.json').write_text(json.dumps(x,indent=2));print(x,flush=True)
