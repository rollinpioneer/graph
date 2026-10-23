"""Frozen-query R1.2: exact SmoothL1 output gradients and executable-width saturation.
No model imports, network inference, parameter updates, or changes to source data.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

SCENES=[2,9,31,33,68,64,76,82]
FRAMES=[0,128]

def writecsv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

def run(cache,out):
    rows=[];sources=[]
    for s in SCENES:
        for f in FRAMES:
            tag=f'train_{s:04d}_{f:04d}';p=cache/f'{tag}.npz';a=np.load(p)
            valid=a['batch_valid_mask'];n=int(valid.sum());w=a['batch_grasp_width'];raw=a['raw_width'].reshape(-1)
            target=w*10;err=raw-target;grad=np.clip(err,-1,1)
            assert np.allclose(a['pred'][:,1],np.clip(1.2*raw/10,0,.1),atol=1e-7)
            for i in np.flatnonzero(valid):
                high=bool(1.2*raw[i]/10>=.1);low=bool(raw[i]<=0)
                rows.append(dict(frame_id=tag,scene=s,frame=f,candidate_id=int(i),object_id=int(a['matched_object_id'][i]),
                    q_ref=float(a['batch_grasp_score'][i]),zero_quality=bool(a['batch_grasp_score'][i]==0),
                    width_ref_m=float(w[i]),width_target_raw=float(target[i]),predicted_raw_width=float(raw[i]),
                    scaled_width_before_clip_m=float(1.2*raw[i]/10),final_width_m=float(a['pred'][i,1]),
                    reference_zero_width=bool(w[i]==0),reference_over100mm=bool(w[i]>np.float32(.1)),reference_scaling_would_saturate=bool(1.2*w[i]>=.1),
                    high_saturation=high,low_saturation=low,width_gradient_raw=float(grad[i]),
                    width_gradient_weighted_normalized=float(10*grad[i]/n),valid_denominator=n,
                    smoothl1_linear_regime=bool(abs(err[i])>=1),
                    high_saturation_push_further=bool(high and grad[i]<0),low_saturation_push_further=bool(low and grad[i]>0),
                    width_subgroup='zero_width' if w[i]==0 else ('over100mm' if w[i]>np.float32(.1) else 'in_range_nonzero')))
            sources.append(dict(path=str(p),sha256=sha(p)))
    writecsv(out/'width_diagnostics.csv',rows)
    summary=[]
    for scene in [None]+SCENES:
        for pop in ['zero','positive']:
            for subgroup in ['all','zero_width','over100mm','in_range_nonzero']:
                rr=[r for r in rows if (scene is None or r['scene']==scene) and r['zero_quality']==(pop=='zero') and (subgroup=='all' or r['width_subgroup']==subgroup)]
                if not rr:continue
                grad=np.array([r['width_gradient_raw'] for r in rr]);wg=np.array([r['width_gradient_weighted_normalized'] for r in rr])
                summary.append(dict(scene='all' if scene is None else scene,population=pop,width_subgroup=subgroup,n=len(rr),
                    scenes=len(set(r['scene'] for r in rr)),high_saturated=sum(r['high_saturation'] for r in rr),low_saturated=sum(r['low_saturation'] for r in rr),
                    saturated_push_further=sum(r['high_saturation_push_further'] or r['low_saturation_push_further'] for r in rr),
                    target_scaling_would_saturate=sum(r['reference_scaling_would_saturate'] for r in rr),smoothl1_linear_regime=sum(r['smoothl1_linear_regime'] for r in rr),
                    gradient_abs_mean=float(abs(grad).mean()),gradient_signed_mean=float(grad.mean()),gradient_l2=float(np.linalg.norm(grad)),
                    gradient_weighted_normalized_l2=float(np.linalg.norm(wg))))
    writecsv(out/'width_scene_summary.csv',summary)
    zz=[r for r in rows if r['zero_quality']];pp=[r for r in rows if not r['zero_quality']]
    result=dict(n_valid=len(rows),n_zero=len(zz),n_positive=len(pp),scenes=SCENES,frames=FRAMES,
        formula='d SmoothL1(raw_width,10*w_ref)/d raw_width=clip(raw_width-10*w_ref,-1,1); weighted denominator=10/N_valid per frame',
        scope='Fixed inference queries and frozen labels. Output gradient is not shared parameter gradient or an Adam update.',
        output_gradient_zero_share_squared_norm=float(sum(r['width_gradient_weighted_normalized']**2 for r in zz)/sum(r['width_gradient_weighted_normalized']**2 for r in rows)),
        source_files=sources,all_scene_summary=[r for r in summary if r['scene']=='all'],updates=0,inference=0)
    (out/'width_diagnostics_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);run(args.cache,args.out)
