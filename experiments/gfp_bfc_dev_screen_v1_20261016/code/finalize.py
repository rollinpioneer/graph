"""Read-only final analysis: no model imports, updates, or environment steps."""
import json, hashlib, math, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])

summary=read(OUT/'summary.json'); queue=read(ROOT/'receipts/queue.json'); repair=read(ROOT/'receipts/repair_queue.json')
assert not queue['active'] and not queue['pending'] and not queue['evaluation_pending']
assert not repair['active'] and not repair['pending'] and not repair['failed'] and len(repair['done'])==9
assert len(queue['done'])==27 and len(queue['failed'])==9
assert all(x['exit_code']==0 for x in queue['done']+repair['done'])
assert summary['total_recorded_updates']==1200000 and summary['matched_initial_hashes']
records={}; probes={}; checkpoints=[]; diagnostics=[]; costs=[]; references=None
samples=[json.loads(x) for x in (ROOT/'logs/resource_samples.jsonl').read_text().splitlines()]
for row in summary['rows']:
    seed,arm=row['seed'],row['arm']; d=ROOT/'runs'/f's{seed}_{arm}'
    assert row['status']=='completed' and row['cost']['source_unchanged']
    p=read(d/'probes.json');probes[seed,arm]=p
    assert len(p)==21 and [x['updates'] for x in p]==list(range(0,100001,5000))
    assert all(math.isfinite(v) for x in p for v in x.values() if isinstance(v,(float,int)))
    for it in (4,50000,100000):
        meta=read(d/f'checkpoint_{it}.json'); path=d/f'checkpoint_{it}.pkl'
        assert meta['full_restore']=='PASS' and meta['updates']==it and sha(path)==meta['sha256']
        checkpoints.append(dict(seed=seed,arm=arm,path=str(path),bytes=path.stat().st_size,**meta))
    for it in (50000,100000):
        ep=d/f'eval_{it}/results.json'
        if not ep.exists():ep=d/f'eval_{it}_repair1/results.json'
        e=read(ep);e['episodes'].sort(key=lambda x:x['episode']);assert len(e['episodes'])==100
        assert e['optimizer_updates']==0 and e['checkpoint_unchanged']
        assert e['checkpoint_sha256']==read(d/f'checkpoint_{it}.json')['sha256']
        h=[x['initial_obs_sha256'] for x in e['episodes']]
        if references is None:references=h
        assert h==references
        assert e['physics_substeps']==e['control_calls']*5
        records[seed,arm,it]=e
    b=p[0];flags=[(x['V_online_start']-x['V_online_all'])-(b['V_online_start']-b['V_online_all'])>=15 and x['dqda_norm_start']>=2*b['dqda_norm_start'] for x in p[1:]]
    first=next((x['updates'] for x,f in zip(p[1:],flags) if f),None)
    before=[x.get('critic/clipped_fraction') for x in p[1:] if first is not None and x['updates']<first and 'critic/clipped_fraction' in x]
    diagnostics.append(dict(seed=seed,arm=arm,anomaly_points=row['anomaly_points'],longest_anomaly_run=row['longest_anomaly_run'],first_anomaly=first,
        final_gap=p[-1]['V_online_start']-p[-1]['V_online_all'],gap_increment=(p[-1]['V_online_start']-p[-1]['V_online_all'])-(b['V_online_start']-b['V_online_all']),
        final_gradient_ratio=p[-1]['dqda_norm_start']/b['dqda_norm_start'],final_gradient_cosine=p[-1]['dqda_cos_start'],final_action_shift=p[-1]['action_shift_start'],
        sampled_clip_fraction_before_anomaly_mean=float(np.mean(before)) if before else None,final_clip_fraction=p[-1].get('critic/clipped_fraction'),
        max_sampled_clip_fraction=max([x.get('critic/clipped_fraction',0) for x in p]),final_delta_abs=p[-1].get('critic/delta_abs'),final_accepted_abs=p[-1].get('critic/accepted_abs')))
    job=next(x for x in queue['done'] if x['id']==f'train_s{seed}_{arm}')
    mem=[x['gpu_memory_mib'] for sample in samples for x in sample['active'] if x['id']==job['id'] and x.get('gpu_memory_mib') is not None]
    costs.append(dict(seed=seed,arm=arm,gpu=job['gpu'],process_residence_hours_upper=(job['ended']-job['started'])/3600,
        measured_training_seconds=row['cost']['seconds'],cpu_seconds=row['cost']['cpu_seconds'],peak_rss_gib=row['cost']['max_rss_kib']/1048576,
        sampled_gpu_memory_peak_mib=max(mem) if mem else None))
assets=read(ROOT/'receipts/assets.json')
for source in assets['sources'].values():assert sha(source['path'])==source['sha256']
failed=[]
for job in queue['failed']:
    it=int(job['id'].rsplit('_',1)[1]);d=ROOT/'runs'/f"s{job['seed']}_{job['arm']}";p=read(d/f'eval_{it}/progress.json');e=records[job['seed'],job['arm'],it]
    assert p['completed']==100
    failed.append(dict(id=job['id'],external_steps=p['external_steps'],control_calls=p['control_calls'],physics_substeps_derived=p['control_calls']*5,
        repaired_external_steps=e['external_steps'],trajectory_length_reproduced=p['external_steps']==e['external_steps']))
paired=[];rng=np.random.default_rng(20261016)
for seed in (0,3):
    for it in (50000,100000):
        for arm in ('R1','R2','R3','R4','R5'):
            for comparator in (['R0'] if arm in ('R1','R2','R3') else ['R0','R2','R3']):
                a=np.array([x['success'] for x in records[seed,arm,it]['episodes']]);b=np.array([x['success'] for x in records[seed,comparator,it]['episodes']]);delta=a-b
                boot=delta[rng.integers(0,100,size=(10000,100))].mean(1)*100
                paired.append(dict(seed=seed,updates=it,arm=arm,comparator=comparator,difference_pp=float(delta.mean()*100),wins=int((delta>0).sum()),losses=int((delta<0).sum()),
                    conditional_paired_bootstrap_95_percentile_pp=np.percentile(boot,[2.5,97.5]).tolist()))
jobs=queue['done']+queue['failed']+repair['done']
valid_calls=sum(x['control_calls'] for x in records.values());bad_calls=sum(x['control_calls'] for x in failed)
total=dict(optimizer_updates=1200000,online_updates=0,primary_conditions=24,primary_episodes=2400,failed_reporting_episodes=900,
    successful_eval_external_steps=sum(x['external_steps'] for x in records.values()),failed_eval_external_steps=sum(x['external_steps'] for x in failed),
    successful_eval_control_calls=valid_calls,failed_eval_control_calls=bad_calls,preparation_control_calls=80,total_control_calls=valid_calls+bad_calls+80,
    total_physics_substeps=5*(valid_calls+bad_calls+80),control_call_cap=6750000,
    train_process_residence_hours_upper=sum(x['process_residence_hours_upper'] for x in costs),
    eval_process_residence_hours_upper=sum((x['ended']-x['started'])/3600 for x in jobs if x.get('kind','eval')=='eval'),
    failed_eval_process_residence_hours_upper=sum((x['ended']-x['started'])/3600 for x in queue['failed']),
    successful_eval_rollout_hours=sum(x['seconds'] for x in records.values())/3600,
    training_cpu_hours=sum(x['cpu_seconds'] for x in costs)/3600,execution_wall_hours=(max(x['ended'] for x in jobs)-min(x['started'] for x in jobs))/3600,
    gpu_busy_hours='NOT_MEASURED; shared-device process residence is not exclusive GPU compute time',eval_cpu_hours='NOT_MEASURED')
assert total['total_control_calls']<=6750000
scores={a:dict(final_mean=float(np.mean([records[s,a,100000]['success_percent'] for s in (0,3)])),four_point_mean=float(np.mean([records[s,a,it]['success_percent'] for s in (0,3) for it in (50000,100000)]))) for a in [f'R{i}' for i in range(6)]}
write('checkpoint_index.json',checkpoints);write('paired_comparisons.json',paired);write('mechanism_diagnostics.json',diagnostics);write('costs.json',dict(total=total,training=costs,reporting_failures=failed))
write('final_validation.json',dict(time=time.time(),training_exit0=12,evaluation_exit0=24,original_reporting_exit1=9,checkpoint_hashes_and_restore_pass=36,
    source_checkpoint_hashes_unchanged=2,all_initial_obs_hashes_match=True,all_sampled_diagnostics_finite=True,all_experiment_processes_finished=True,
    algorithm_decision='NO_GO_BFC_V1; no uniformly stable EMA identified',scores=scores,analysis_optimizer_updates=0,analysis_environment_steps=0))
lines=['# GFP-BFC first-round final report','',
'Decision: **NO-GO for advancing BFC V1 on this evidence.** Both BFC settings failed the frozen consecutive-anomaly criterion in both windows. Neither outperformed the best tested EMA on average. The slowest EMA preserved seed 0 but collapsed in seed 3; it is a useful comparator, not a generally validated remedy. No further training is launched.','',
'## Completion and identity','',
'12 branches completed exactly 100k updates each, total 1.2M; 24 primary evaluations completed, 100 episodes each. The nine original reporting failures remain recorded; the explicitly authorized repairs completed successfully. All 36 saved checkpoint identities and full-restore receipts pass, both source checkpoint hashes are unchanged, and all 24 initial-observation hash vectors match. No non-finite sampled losses/probes were found.','',
'Source: upstream GFP 1abbd5f0e287caf9bda70e7bc655d0a499ac8401; formal G_task1_s0 600k and G_task1_s3 300k forks; OGBench 1.1.0. Full actual config/source identities are in receipts/assets.json and runs/*/identity.json. The reference Actor only changes the TD Actor contribution; online Actor deployment and all other specified routes remain unchanged.','',
'## Success (%) — all checkpoints, no best-checkpoint selection','',
table(['Arm','tau','epsilon','s0 +50k','s0 +100k','s3 +50k','s3 +100k','final two-window mean'],[[a,read(ROOT/'protocol.json')['arms'][a]['tau_pi'],read(ROOT/'protocol.json')['arms'][a]['epsilon'],*[records[s,a,it]['success_percent'] for s in (0,3) for it in (50000,100000)],scores[a]['final_mean']] for a in scores]),'',
'R3 is the best tested EMA by final two-window mean (37%) and four-point descriptive mean (40.5%). BFC R4/R5 final means are 14%/12%, differences -23/-25 pp versus R3. Their four-point means are 23.5%/21%. Thus the exploratory +10 pp average benefit criterion fails. R4 improves over same-tau R2 at +100k by +1/+13 pp (s0/s3), but at +50k by -13/0 pp; this is not stable dominance over even the matched-tau baseline.','',
'## Return and final start-region membership','',
table(['seed','arm','offset','return','final in start %'],[[s,a,it,records[s,a,it]['return_mean'],records[s,a,it]['start_stay_percent']] for s in (0,3) for a in scores for it in (50000,100000)]),'',
'Final start-region membership is an endpoint measure; it is not a claim that the robot remained there for the entire episode. s3 R1/R2/R4/R5 can finish mostly outside the start yet have low success. Leaving the start is insufficient for task completion.','',
'## Mechanism and temporal behavior','',
table(['seed','arm','anomaly points/20','longest run','first offset','final gap rise','final gradient/fork','final action shift'],[[x['seed'],x['arm'],x['anomaly_points'],x['longest_anomaly_run'],x['first_anomaly'],round(x['gap_increment'],3),round(x['final_gradient_ratio'],3),round(x['final_action_shift'],3)] for x in diagnostics]),'',
'An anomaly requires BOTH start-minus-overall value-gap rise >=15 and start action-gradient norm >=2 times fork. Two consecutive 5k probes mean not avoided. All arms except s0 R3 meet that failure criterion. This diagnostic is not NaN, a universal instability theorem, or an automatic training stop. Some seed-3 arms recover partial success after an earlier anomaly; counts and endpoint performance must be interpreted together.','',
table(['seed','BFC arm','mean clip fraction before first anomaly','max sampled clip fraction','final clip fraction','final mean abs d','final accepted abs d'],[[x['seed'],x['arm'],*[None if x[k] is None else round(x[k],5) for k in ['sampled_clip_fraction_before_anomaly_mean','max_sampled_clip_fraction','final_clip_fraction','final_delta_abs','final_accepted_abs']]] for x in diagnostics if x['arm'] in ('R4','R5')]),'',
'Clipping entries are sampled training-minibatch diagnostics (every 5k), not a full-run or start-state-specific trigger rate. Their presence before anomalies shows the limiter can engage without preventing the measured failure; it does not establish causal protection. Complete Q/gradient cosine/action shift/loss traces are retained in runs/*/probes.json and mechanism_diagnostics.json.','',
'## Paired comparisons and replay limitation','',
'paired_comparisons.json reports per-window, per-checkpoint paired wins/losses and 10,000 episode-bootstrap percentile intervals versus R0, and for BFC versus R2 and R3. These intervals condition on a trained model and the reset distribution; they do not estimate training-seed uncertainty, correct multiple comparisons, or remove selection bias from choosing the best EMA on these same windows.','',
'Five of the nine repaired evaluations have different total rollout lengths from their failed predecessors despite the same declared checkpoint/protocol/RNG schedule. The failed runs did not save complete episode outputs, so strict replay equality cannot be checked and the cause is unresolved. New primary initial states match, but that does not prove full deterministic trajectories. Do not present repairs as bitwise replays or small score differences as robust advantages. No extra replay was run. Exact old/new lengths are in costs.json.','',
'## Actual cost','',
table(['Measure','Value'],[[k,round(v,6) if isinstance(v,float) else v] for k,v in total.items()]),'',
table(['seed','arm','GPU','process residence h (upper proxy)','CPU seconds','peak host RSS GiB','sampled own GPU MiB'],[[x['seed'],x['arm'],x['gpu'],round(x['process_residence_hours_upper'],4),round(x['cpu_seconds'],2),round(x['peak_rss_gib'],3),x['sampled_gpu_memory_peak_mib']] for x in costs]),'',
'Process residence includes initialization, compilation, probing and polling lag; GPU allocation need not last the whole interval. Shared-GPU residence sums are not exclusive GPU hours. GPU memory values are observed per-process samples, not guaranteed continuous peaks. Evaluation CPU usage was not recorded. Failed evaluation physics counts are derived as control calls times fixed frame_skip=5, verified against all successful evaluations. Preparation includes two constructor resets (80 control calls), not zero physical work.','',
'## Decision and bounds','',
table(['Question','Decision'],[['Is BFC V1 ready for broader validation?','NO-GO: fails anomaly prevention and advantage over best EMA.'],['Is a pure EMA already a stable fix?','DIAGNOSE: R3 works in s0, fails in s3.'],['Should this run add seeds, tune epsilon or train from scratch?','No; outside authorization and unsupported by this first-round screen.']]),'',
'These are two selected warm-start development windows, not independent broad validation. Original source checkpoints lack historical NumPy/Python sidecars: branches restore model/Adam/Agent RNG and pair their new sampling, but do not claim exact replay of the historical training. B1a is historical context under a different evaluation schedule and is not used as a matched numerical comparator. No normal-window, long-horizon, from-scratch or task4 effectiveness claim is supported.','',
'Full weights, data and per-step trajectories remain on the server. Final lightweight artifacts: summary.json, checkpoint_index.json, paired_comparisons.json, mechanism_diagnostics.json, costs.json, final_validation.json and diagnostics.png.','']
(OUT/'FINAL_REPORT.md').write_text('\n'.join(lines))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,4,figsize=(17,8))
colors=plt.cm.tab10.colors
for ri,seed in enumerate((0,3)):
    for ai,arm in enumerate(scores):
        p=probes[seed,arm];x=np.array([v['updates'] for v in p])/1000;b=p[0];c=colors[ai]
        axes[ri,0].plot([50,100],[records[seed,arm,it]['success_percent'] for it in (50000,100000)],'o-',label=arm,color=c)
        axes[ri,1].plot(x,[(v['V_online_start']-v['V_online_all'])-(b['V_online_start']-b['V_online_all']) for v in p],color=c)
        axes[ri,2].plot(x,[v['dqda_norm_start']/b['dqda_norm_start'] for v in p],color=c)
        if arm in ('R4','R5'):axes[ri,3].plot(x[1:],[v.get('critic/clipped_fraction',0) for v in p[1:]],label=arm,color=c)
    axes[ri,1].axhline(15,color='gray',ls='--');axes[ri,2].axhline(2,color='gray',ls='--')
    for j,title in enumerate(['Success %','Value-gap rise','Action-gradient norm / fork','Sampled clip fraction']):
        axes[ri,j].set_title(f'seed {seed}: {title}');axes[ri,j].set_xlabel('Additional updates (k)');axes[ri,j].grid(alpha=.2)
    axes[ri,0].set_ylim(-3,103);axes[ri,0].legend(ncol=3,fontsize=8);axes[ri,3].legend()
fig.suptitle('GFP BFC development screen: selected warm-start windows, no broad generalization')
fig.tight_layout();fig.savefig(OUT/'diagnostics.png',dpi=150);plt.close(fig)
print(json.dumps(dict(scores=scores,cost=total),indent=2))
