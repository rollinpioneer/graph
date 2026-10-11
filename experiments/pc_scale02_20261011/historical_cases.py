"""Index actual stored frames; preserve timing limits of historical camera sampling."""
import json,pathlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/historical_cases';OUT.mkdir(exist_ok=True)
OLD=pathlib.Path('/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011')
rows=json.loads((ROOT/'results/historical_events.json').read_text());index=[]
for r in rows:
    n,seed,i=r['n'],r['seed'],r['slot']
    path=OLD/('runs/core%d/data/trajectory_%d.npz'%(n,seed));z=np.load(path)
    targets=[('fall',r['full_available']['first_fall']),('no_progress',r['full_available']['first_no_progress'])]
    departures=[t for t in r['full_available']['departure_steps'] if t is not None]
    targets.append(('departure',min(departures) if departures else None))
    frames=None
    for kind,t in targets:
        if t is None:continue
        ts=z['image_steps'];before=np.flatnonzero(ts<=t);after=np.flatnonzero(ts>=t)
        a=int(before[-1]) if len(before) else 0;b=int(after[0]) if len(after) else len(ts)-1
        if frames is None:frames=z['images']
        fig,axes=plt.subplots(2,2,figsize=(5,5))
        for k,frame in enumerate([a,b]):
            for view in range(2):
                image=frames[frame,i,view].transpose(1,2,0)
                assert np.std(image)>0
                axes[k,view].imshow(image);axes[k,view].axis('off');axes[k,view].set_title('stored step %d, view %d'%(ts[frame],view))
        fig.suptitle('%dC seed %d slot %d: %s detected at %d'%(n,seed,i,kind,t))
        filename='%dC_%d_%d_%s.png'%(n,seed,i,kind);fig.tight_layout();fig.savefig(OUT/filename,dpi=100);plt.close(fig)
        lo=max(0,t-10);hi=min(len(z['states']),t+11)
        statefile=filename.replace('.png','.json')
        (OUT/statefile).write_text(json.dumps({'record_steps':list(range(lo,hi)),'states':z['states'][lo:hi,i].tolist(),
            'distances':z['distances'][lo:hi,i].tolist(),'eef':z['eef'][lo:hi,i].tolist()},indent=2))
        index.append({'n':n,'seed':seed,'slot':i,'event':kind,'detection_step':t,'stored_image_steps':[int(ts[a]),int(ts[b])],
                      'image':filename,'state_segment':statefile,'original_trajectory':str(path),
                      'caveat':'Historical images are sampled every 10 steps; frames bracket detection and may not exactly show event onset.'})
(OUT/'index.json').write_text(json.dumps(index,indent=2));print('historical_case_count',len(index),flush=True)
