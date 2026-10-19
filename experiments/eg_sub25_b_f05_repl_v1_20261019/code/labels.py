import argparse,resource
from repl_common import *
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);a=p.parse_args();start=time.time();os.nice(10);Native,_=native_class();ctx=Native(a.scene);split=json.loads((R/'B_head_split.json').read_text());assert a.scene in split['fit']+split['model_val']+split['gate_cal'];calls=0;checks=[]
for frame in split['frame_ids']:
 source=R/f'B/exports/subset/scene_{a.scene:04d}/{frame:04d}.npz';target=R/f'B/labels/{a.scene:04d}_{frame:04d}.npz';target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
 with np.load(source) as z:pred=z['pred'];ids=sample_ids(pred,a.scene,frame);hyp=z['hypotheses'][ids];feat=z['feature320'][ids];act=z['actions'][ids];actids=z['action_ids'][ids];assert np.array_equal(hyp[:,0],pred[ids])
 ctx.frame(frame);q,c,e,_=ctx.score(hyp.reshape(-1,17));q=q.reshape(128,6);c=c.reshape(128,6);e=e.reshape(128,6);assert np.isfinite(q).all();assert feat.shape==(128,320) and act.shape==(128,6,4)
 if frame==split['frame_ids'][0]:
  oq,oc,oe,_=ctx.score(pred[ids[:16]]);assert np.array_equal(oq,q[:16,0]) and np.array_equal(oc,c[:16,0]) and np.array_equal(oe,e[:16,0]);checks.append(dict(frame=frame,original_action_quality_exact=True))
 np.savez_compressed(target,ids=ids,q=q,collision=c,empty=e,feature=feat,action=act,actions=actids,export_sha256=np.array(sha256(source)));calls+=768;print('LABEL',a.scene,frame,calls,flush=True)
write(R/f'results/labels_{a.scene:04d}.json',dict(status='PASS',scene=a.scene,actions=calls,seconds=time.time()-start,cpu_seconds=time.process_time(),rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checks=checks))
