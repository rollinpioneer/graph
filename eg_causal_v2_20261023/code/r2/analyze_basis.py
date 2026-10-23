"""Analyze R2 exports with scene-paired inference; no new physical labels."""
import csv,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(sys.argv[1]); OUT=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT
OUT.mkdir(parents=True,exist_ok=True)
K=[0,1,3,6]
def sm(x):
    y=np.exp(x.astype(float)-x.max(0,keepdims=True));return y/y.sum(0,keepdims=True)
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def paired_scene_ci(rows,key):
    scenes=sorted(set(r['scene'] for r in rows))
    means=np.array([np.mean([r[key] for r in rows if r['scene']==s]) for s in scenes])
    rng=np.random.default_rng(20261023)
    bs=means[rng.integers(0,len(means),(20000,len(means)))].mean(1)
    return dict(mean=float(means.mean()),ci95=list(map(float,np.quantile(bs,[.025,.975]))),
        scenes=len(scenes),positive_scenes=int((means>0).sum()),negative_scenes=int((means<0).sum()),
        zero_scenes=int((means==0).sum()),per_scene={str(s):float(v) for s,v in zip(scenes,means)})
rows=[];basis=[];pipeline=[];receipts=[]
for path in sorted(ROOT.glob('train_*.npz')):
    z=np.load(path);tag=path.stem;scene=int(z['scene']);frame=int(z['frame']);ids=z['candidate_id'];n=len(ids)
    receipts.append(json.loads(path.with_suffix('.json').read_text()))
    grid=((z['grid_mu']>0)&(z['grid_mu']<=.4)&~z['grid_collision']&~z['grid_empty'])
    assert np.isfinite(z['grid_mu']).all()
    a0=z['baseline_a'];d0=z['baseline_d'];orig=grid[np.arange(n),a0,d0]
    p0=sm(z['k0_angle_logits'][:12]);dprob0=sm(z['k0_depth_logits'][:4]);qprob0=sm(z['k0_quality_logits'])
    for k in K:
        ac=z[f'k{k}_a_physical'];dc=z[f'k{k}_d'];feas=grid[np.arange(n),ac,dc]
        pa=np.roll(sm(z[f'k{k}_angle_logits'][:12]),k,axis=0)
        pd=sm(z[f'k{k}_depth_logits'][:4]);pq=sm(z[f'k{k}_quality_logits'])
        extra_a=sm(z[f'k{k}_angle_logits'])[-1];extra_d=sm(z[f'k{k}_depth_logits'])[-1]
        mp=(p0+pa)/2
        js=.5*np.sum(p0*np.log(np.maximum(p0,1e-300)/np.maximum(mp,1e-300))+pa*np.log(np.maximum(pa,1e-300)/np.maximum(mp,1e-300)),axis=0)
        for j,cid in enumerate(ids):
            rows.append(dict(frame=tag,scene=scene,frame_id=frame,candidate_id=int(cid),
                selection='raw_top1' if j==0 else ('rank_stratified' if j<8 else 'random'),k=k,phi_deg=k*15,
                valid_mask=bool(z['valid_mask'][j]),q_ref=float(z['q_ref'][j]),matched_object_id=int(z['matched_object_id'][j]),
                baseline_angle=int(a0[j]),baseline_depth=int(d0[j]),new_physical_angle=int(ac[j]),new_depth=int(dc[j]),
                angle_changed=int(ac[j]!=a0[j]),depth_changed=int(dc[j]!=d0[j]),pose_changed=int(ac[j]!=a0[j] or dc[j]!=d0[j]),
                baseline_feasible=int(orig[j]),new_feasible=int(feas[j]),rescue=int(~orig[j]&feas[j]),harm=int(orig[j]&~feas[j]),
                feasible_delta=int(feas[j])-int(orig[j]),baseline_mu=float(z['grid_mu'][j,a0[j],d0[j]]),new_mu=float(z['grid_mu'][j,ac[j],dc[j]]),
                new_collision=bool(z['grid_collision'][j,ac[j],dc[j]]),new_empty=bool(z['grid_empty'][j,ac[j],dc[j]]),
                feasible_grid_cells=int(grid[j].sum()),angle_prob_L1=float(abs(pa[:,j]-p0[:,j]).sum()),angle_prob_JS=float(js[j]),
                depth_prob_L1=float(abs(pd[:,j]-dprob0[:,j]).sum()),quality_prob_L1=float(abs(pq[:,j]-qprob0[:,j]).sum()),
                angle_extra_class_probability=float(extra_a[j]),depth_extra_class_probability=float(extra_d[j]),
                baseline_score=float(z['k0_score'][j]),new_score=float(z[f'k{k}_score'][j]),score_delta=float(z[f'k{k}_score'][j]-z['k0_score'][j]),
                raw_width_delta=float(z[f'k{k}_raw_width'][j]-z['k0_raw_width'][j]),frozen_width_m=float(z['baseline_pred'][j,1]),
                predicted_width_secondary_unevaluated_m=float(z[f'k{k}_decoded_width_secondary_only'][j]),
                secondary_width_delta_m=float(z[f'k{k}_decoded_width_secondary_only'][j]-z['baseline_pred'][j,1])))
        ix0=int(np.argmax(z['k0_score']));ix1=int(np.argmax(z[f'k{k}_score']))
        assert ix0==0, 'chosen set includes original raw Top1 as its first query'
        for geometry,score,index in [('G0','S0',ix0),('G1','S0',ix0),('G0','S1',ix1),('G1','S1',ix1)]:
            outcome=orig if geometry=='G0' else feas
            pipeline.append(dict(scene=scene,frame=tag,k=k,composition=geometry+score,
                candidate_id=int(ids[index]),feasible=int(outcome[index]),
                baseline_feasible=int(orig[ix0]),feasible_delta=int(outcome[index])-int(orig[ix0]),
                pool='same_16_preselected_queries',postprocessing='none_not_full_pipeline'))
    valid=z['basis_all_valid_mask'];count=len(valid)
    for j in range(count):
        basis.append(dict(scene=scene,frame=tag,candidate_id=j,valid_mask=bool(valid[j]),
            approach_difference_deg=float(z['basis_train_infer_approach_deg_all'][j]),
            projected_roll_difference_deg=float(z['basis_train_infer_roll_deg_all'][j]),
            distance_to_nearest_15deg_grid_deg=float(z['basis_train_infer_offgrid_deg_all'][j])))

assert len(rows)==256*4,('Expected 256 fixed queries x4',len(rows))
assert len(receipts)==16
write('basis_reencoding_results.csv',rows)
write('basis_fixed_pool_compositions.csv',pipeline)
write('basis_training_inference_audit.csv',basis)
summary=dict(scope={'frames':16,'scenes':8,'queries':256,'shifts':K,'new_analytic_evaluations':0,
                   'parameter_updates':0,'primary_width':'frozen_original','full_pipeline_evaluated':False,
                   'C3_accessed':False},conditions={},basis_audit={},invariants={})
for k in K:
    r=[v for v in rows if v['k']==k];summary['conditions'][str(k)]={}
    for selection in ['all','raw_top1','rank_stratified','random']:
        x=r if selection=='all' else [v for v in r if v['selection']==selection]
        summary['conditions'][str(k)][selection]=dict(n=len(x),original_feasible=sum(v['baseline_feasible'] for v in x),
            new_feasible=sum(v['new_feasible'] for v in x),rescue=sum(v['rescue'] for v in x),harm=sum(v['harm'] for v in x),
            paired_scene_delta=paired_scene_ci(x,'feasible_delta'),
            pose_changed=sum(v['pose_changed'] for v in x),angle_changed=sum(v['angle_changed'] for v in x),depth_changed=sum(v['depth_changed'] for v in x),
            mean_angle_prob_L1=float(np.mean([v['angle_prob_L1'] for v in x])),mean_angle_prob_JS=float(np.mean([v['angle_prob_JS'] for v in x])),
            mean_depth_prob_L1=float(np.mean([v['depth_prob_L1'] for v in x])),
            mean_angle_extra_class_probability=float(np.mean([v['angle_extra_class_probability'] for v in x])),
            mean_depth_extra_class_probability=float(np.mean([v['depth_extra_class_probability'] for v in x])),
            mean_abs_secondary_width_change_mm=float(np.mean([abs(v['secondary_width_delta_m'])*1000 for v in x])),
            mean_quality_prob_L1=float(np.mean([v['quality_prob_L1'] for v in x])),
            mean_abs_score_delta=float(np.mean([abs(v['score_delta']) for v in x])))
    summary['conditions'][str(k)]['fixed16_compositions']={}
    for composition in ['G0S0','G1S0','G0S1','G1S1']:
        x=[v for v in pipeline if v['k']==k and v['composition']==composition]
        summary['conditions'][str(k)]['fixed16_compositions'][composition]=dict(n=len(x),feasible=sum(v['feasible'] for v in x),
            paired_scene_delta=paired_scene_ci(x,'feasible_delta'))
for pop in ['all','valid']:
    x=basis if pop=='all' else [v for v in basis if v['valid_mask']]
    aa=np.array([v['approach_difference_deg'] for v in x]);rr=np.array([abs(v['projected_roll_difference_deg']) for v in x]);off=np.array([v['distance_to_nearest_15deg_grid_deg'] for v in x])
    summary['basis_audit'][pop]=dict(n=len(x),approach_deg_quantiles=dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(aa,[0,.25,.5,.75,1])))),
        absolute_roll_deg_quantiles=dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(rr,[0,.25,.5,.75,1])))),
        offgrid_deg_quantiles=dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(off,[0,.25,.5,.75,1])))),
        within_1_degree_of_15deg_grid=int((off<=1).sum()),absolute_roll_over_15deg=int((rr>15).sum()))
summary['invariants']=dict(historical_all_exact=all(r['historical_replay_exact'] for r in receipts),
    k0_all_exact=all(r['k0_exact'] for r in receipts),unchanged_parameters_and_buffers=all(r['model_and_buffers_unchanged'] for r in receipts),
    max_physical_action_set_error=max(c['physical_48_action_set_max_abs'] for r in receipts for c in r['conditions']),
    max_cached_control_pose_error=max(c['cached_output_relabel_identity_pose_max_abs'] for r in receipts for c in r['conditions']),
    cached_control_prob_all_exact=all(c['cached_output_relabel_prob_exact'] for r in receipts for c in r['conditions']),
    neighbors_all_exact=all(c['neighbors_exact'] for r in receipts for c in r['conditions']))
summary['interpretation_limits']=[
    'Integer-bin basis changes preserve physical action set at original widths; outputs are not basis-equivariant.',
    'A sensitivity or average benefit under an arbitrary representation does not prove original basis is a defect or causal historical training failure.',
    'Training/inference basis audit uses frozen inference queries with train-label bases, not replayed historical training batches; approaches also differ.',
    'Secondary new widths are recorded but unevaluated; old grid labels cannot label them.',
    'Eight scenes; scenario bootstrap intervals are descriptive and selections are prespecified strata, not a representative deployment pool.',
    'Raw Top1 is 16/256 queries; fixed16 score compositions are neither full original pool nor collision/NMS/object-Top10/Top50 pipeline.',
    'No phi selected by outcome; all preregistered conditions reported.']
(OUT/'basis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
